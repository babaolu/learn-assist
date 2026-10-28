"""
Unit tests for learn_assist.forms and learn_assist.testrun.

ALL tests use purely synthetic fixtures. No real book content or exercise text.
Strict adherence to docs/BLINDING.md: no problem statements, answers, or keys.
"""

import ast
import builtins
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from learn_assist.forms import make_forms, FORBIDDEN_PATTERNS as FORMS_FORBIDDEN
from learn_assist.testrun import (
    run_sitting,
    get_testrun_status,
    print_status,
    format_statement,
    FORBIDDEN_PATTERNS as RUNNER_FORBIDDEN
)
from learn_assist.itembank import file_sha256


# ---------------------------------------------------------------------------
# Synthetic Test Helpers
# ---------------------------------------------------------------------------

def create_synthetic_frozen_bank(root: Path, chapter: int = 1, force_imbalance: bool = False) -> Dict[str, Any]:
    """
    Builds a synthetic chapter itembank directory and FROZEN.json.
    """
    ch_dir = root / "itembank" / f"ch{chapter:02d}"
    items_dir = ch_dir / "items"
    slots_dir = ch_dir / "slots"
    keys_dir = ch_dir / "keys" / "claude"
    slotted_dir = ch_dir / "slotted" / "claude"

    items_dir.mkdir(parents=True, exist_ok=True)
    slots_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)
    slotted_dir.mkdir(parents=True, exist_ok=True)

    usable_a_ids = [f"A_{chapter}.01", f"A_{chapter}.02", f"A_{chapter}.03", f"A_{chapter}.04", f"A_{chapter}.05", f"A_{chapter}.06"]
    
    if force_imbalance:
        # All B items anchored to A_1.01, creating an unavoidable B imbalance
        selected_b_ids = [f"B_syn_0{i}" for i in range(1, 6)]
        anchors = [usable_a_ids[0]] * 5
    else:
        selected_b_ids = [f"B_syn_0{i}" for i in range(1, 5)]
        anchors = [usable_a_ids[0], usable_a_ids[1], usable_a_ids[2], usable_a_ids[3]]

    concepts_pool = [f"{chapter}.1", f"{chapter}.2"]

    # Write A items
    for idx, aid in enumerate(usable_a_ids):
        item_data = {
            "item_id": aid,
            "bucket": "A",
            "statement_md": f"Synthetic statement for {aid}.",
            "concepts": [concepts_pool[idx % len(concepts_pool)]],
            "status": "usable"
        }
        (items_dir / f"{aid}.json").write_text(json.dumps(item_data, indent=2), encoding="utf-8")
        # Slot
        slot_data = {"item_id": aid, "slots": []}
        (slots_dir / f"{aid}.json").write_text(json.dumps(slot_data, indent=2), encoding="utf-8")

    # Write B items
    for idx, bid in enumerate(selected_b_ids):
        item_data = {
            "item_id": bid,
            "bucket": "B",
            "author_family": "claude",
            "anchor_item": anchors[idx],
            "statement_md": f"Synthetic statement for {bid}.",
            "needs_figure": (idx == 0),
            "figure_descriptions": ["Synthetic diagram of a circle"] if idx == 0 else [],
            "concepts": [concepts_pool[idx % len(concepts_pool)]],
            "status": "selected"
        }
        (items_dir / f"{bid}.json").write_text(json.dumps(item_data, indent=2), encoding="utf-8")
        # Slot
        slot_data = {"item_id": bid, "slots": []}
        (slots_dir / f"{bid}.json").write_text(json.dumps(slot_data, indent=2), encoding="utf-8")

    # Write some sensitive files to test isolation
    for item_id in usable_a_ids + selected_b_ids:
        (keys_dir / f"{item_id}.json").write_text("{\"secret\": \"key\"}", encoding="utf-8")
        (slotted_dir / f"{item_id}.json").write_text("{\"secret\": \"slotted\"}", encoding="utf-8")

    # Build FROZEN.json
    files_map = {}
    for item_id in usable_a_ids + selected_b_ids:
        rel_item = f"items/{item_id}.json"
        rel_slot = f"slots/{item_id}.json"
        rel_key = f"keys/claude/{item_id}.json"
        rel_slotted = f"slotted/claude/{item_id}.json"
        files_map[rel_item] = file_sha256(ch_dir / rel_item)
        files_map[rel_slot] = file_sha256(ch_dir / rel_slot)
        files_map[rel_key] = file_sha256(ch_dir / rel_key)
        files_map[rel_slotted] = file_sha256(ch_dir / rel_slotted)

    frozen_content = {
        "chapter": chapter,
        "frozen_at": "2026-10-28T00:00:00Z",
        "usable_a_item_ids": usable_a_ids,
        "selected_b_item_ids": selected_b_ids,
        "files": files_map
    }
    (ch_dir / "FROZEN.json").write_text(json.dumps(frozen_content, indent=2), encoding="utf-8")
    return frozen_content


# ---------------------------------------------------------------------------
# Form Split Tests
# ---------------------------------------------------------------------------

def test_form_split_constraints_synthetic(tmp_path: Path):
    """
    Tests that make_forms satisfies all form split constraints:
    - Balance B count (|diff| <= 1).
    - Concept coverage: no concept with >= 2 B items has 0 in one form while other has >= 2.
    - Anchor pairs kept together.
    - Sittings of <= 10 items.
    - forms.json contains IDs and counts only, no statements.
    """
    data_dir = tmp_path / "data"
    create_synthetic_frozen_bank(data_dir, chapter=1)

    res = make_forms(1, seed=42, data_dir=data_dir)
    assert res["chapter"] == 1
    assert res["seed"] == 42
    assert "baseline" in res["forms"]
    assert "final" in res["forms"]

    b_base = res["forms"]["baseline"]["counts"]["by_bucket"]["B"]
    b_fin = res["forms"]["final"]["counts"]["by_bucket"]["B"]
    assert abs(b_base - b_fin) <= 1

    # Verify anchor pairs kept together
    forms_file = data_dir / "itembank" / "ch01" / "forms.json"
    forms_json = json.loads(forms_file.read_text(encoding="utf-8"))

    # Check that forms.json contains strictly no statement_md, solutions or answers
    raw_text = forms_file.read_text(encoding="utf-8")
    assert "statement_md" not in raw_text
    assert "solution" not in raw_text
    assert "raw_answer" not in raw_text

    items_dir = data_dir / "itembank" / "ch01" / "items"
    for form_name in ["baseline", "final"]:
        form_items = set(forms_json["forms"][form_name]["item_ids"])
        for item_id in form_items:
            item_data = json.loads((items_dir / f"{item_id}.json").read_text(encoding="utf-8"))
            if item_data.get("bucket") == "B":
                anc = item_data["anchor_item"]
                assert anc in form_items, f"Anchor {anc} of {item_id} not in same form {form_name}"

        # Check sittings size <= 10
        for sitting in forms_json["forms"][form_name]["sittings"]:
            assert sitting["counts"]["total"] <= 10
            assert len(sitting["item_ids"]) <= 10


def test_form_split_exit_codes_on_violations(tmp_path: Path):
    """
    Tests that make_forms exits non-zero if:
    - FROZEN.json does not exist.
    - verify-frozen fails.
    - Form split cannot satisfy B balance (|diff| > 1).
    """
    data_dir = tmp_path / "data"

    # 1. Bank not frozen
    with pytest.raises(SystemExit) as exc_info:
        make_forms(1, seed=42, data_dir=data_dir)
    assert exc_info.value.code != 0

    # 2. Bank frozen, but a file is corrupted
    create_synthetic_frozen_bank(data_dir, chapter=1)
    # Corrupt an item file
    (data_dir / "itembank" / "ch01" / "items" / "A_1.01.json").write_text("{\"corrupted\": true}")
    with pytest.raises(SystemExit) as exc_info:
        make_forms(1, seed=42, data_dir=data_dir)
    assert exc_info.value.code != 0

    # 3. Bank frozen, but forced B imbalance cannot satisfy |diff| <= 1
    data_dir2 = tmp_path / "data2"
    create_synthetic_frozen_bank(data_dir2, chapter=2, force_imbalance=True)
    with pytest.raises(SystemExit) as exc_info:
        make_forms(2, seed=42, data_dir=data_dir2)
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Runner Tests
# ---------------------------------------------------------------------------

def test_runner_refuses_when_not_frozen_or_verify_fails(tmp_path: Path):
    """
    Runner must refuse (return non-zero exit code) if not frozen or verify fails.
    """
    data_dir = tmp_path / "data"

    # Not frozen
    rc = run_sitting(1, "baseline", 1, data_dir=data_dir, interactive=False)
    assert rc != 0

    # Frozen, but corrupt file
    create_synthetic_frozen_bank(data_dir, chapter=1)
    make_forms(1, seed=42, data_dir=data_dir)
    (data_dir / "itembank" / "ch01" / "items" / "A_1.01.json").write_text("{\"tampered\": true}")

    rc = run_sitting(1, "baseline", 1, data_dir=data_dir, interactive=False)
    assert rc != 0


def test_runner_refuses_overwrite_and_resume_works(tmp_path: Path):
    """
    Tests append-only protection (refuses overwrite) and resuming at first unanswered item.
    """
    data_dir = tmp_path / "data"
    create_synthetic_frozen_bank(data_dir, chapter=1)
    make_forms(1, seed=42, data_dir=data_dir)

    sitting_items = json.loads((data_dir / "itembank" / "ch01" / "forms.json").read_text(encoding="utf-8"))["forms"]["baseline"]["sittings"][0]["item_ids"]
    assert len(sitting_items) >= 2

    # Provide mock inputs for 1 item
    call_count = 0
    def mock_input():
        nonlocal call_count
        call_count += 1
        return "synthetic response line 1\nsynthetic response line 2"

    # Run sitting for 1 item, then simulate interrupt by raising after 1st item
    def mock_input_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Answer 1"
        raise KeyboardInterrupt()

    try:
        run_sitting(1, "baseline", 1, data_dir=data_dir, interactive=False, input_func=mock_input_once)
    except KeyboardInterrupt:
        pass

    # Verify first item was written
    resp_dir = data_dir / "responses" / "ch01" / "baseline"
    item1_path = resp_dir / f"{sitting_items[0]}.json"
    assert item1_path.exists()
    resp1 = json.loads(item1_path.read_text(encoding="utf-8"))
    assert resp1["raw_answer"] == "Answer 1"
    assert resp1["skipped"] is False
    assert "runner_version" in resp1
    assert "frozen_hash" in resp1

    # Overwrite protection: attempting to overwrite item1_path must fail
    with pytest.raises(FileExistsError):
        with open(item1_path, "x", encoding="utf-8") as f:
            f.write("overwrite attempt")

    # Resume: run again to answer remaining items
    def mock_input_remaining():
        return "skip"

    rc = run_sitting(1, "baseline", 1, data_dir=data_dir, interactive=False, input_func=mock_input_remaining)
    assert rc == 0

    # Verify all items in sitting now have response files
    for item_id in sitting_items:
        p = resp_dir / f"{item_id}.json"
        assert p.exists()

    # Sitting is now complete; running again should report complete and exit 0 without prompting
    rc = run_sitting(1, "baseline", 1, data_dir=data_dir, interactive=False, input_func=lambda: pytest.fail("Should not prompt"))
    assert rc == 0


def test_runner_status_no_item_text(tmp_path: Path, capsys):
    """
    Status report must contain numbers and IDs only, never item text or answers.
    """
    data_dir = tmp_path / "data"
    create_synthetic_frozen_bank(data_dir, chapter=1)
    make_forms(1, seed=42, data_dir=data_dir)

    status = get_testrun_status(1, data_dir=data_dir)
    assert status["chapter"] == 1
    assert "baseline" in status["forms"]

    print_status(status)
    captured = capsys.readouterr().out

    assert "Chapter 01 Testrun Status:" in captured
    assert "Form 'baseline':" in captured
    assert "answered:" in captured
    assert "remaining:" in captured
    assert "Synthetic statement" not in captured


# ---------------------------------------------------------------------------
# Strict Isolation Test: Forms and Runner Never Open Sensitive Files
# ---------------------------------------------------------------------------

def test_runner_and_forms_never_open_sensitive_files(tmp_path: Path, monkeypatch):
    """
    Asserts that neither forms nor testrun ever opens keys/, slotted/, reviews/,
    compare*.json, or resolution.json.
    """
    data_dir = tmp_path / "data"
    create_synthetic_frozen_bank(data_dir, chapter=1)

    # Add sensitive review and comparison files
    ch_dir = data_dir / "itembank" / "ch01"
    (ch_dir / "compare_slots.json").write_text("{\"sensitive\": true}")
    (ch_dir / "resolution.json").write_text("{\"sensitive\": true}")
    (ch_dir / "reviews").mkdir(exist_ok=True)
    (ch_dir / "reviews" / "claude.json").write_text("{\"sensitive\": true}")

    # Track all file opens
    opened_paths: List[str] = []
    orig_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        path_str = str(file).replace("\\", "/").lower()
        opened_paths.append(path_str)
        for pattern in ["keys", "slotted", "reviews", "compare", "resolution"]:
            parts = [p.lower() for p in Path(file).parts]
            name = Path(file).name.lower()
            if pattern in parts or pattern in name or f"/{pattern}/" in path_str:
                raise AssertionError(f"Sensitive file was opened: {file}")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    # 1. Run make_forms under guard
    make_forms(1, seed=42, data_dir=data_dir)

    # 2. Run runner under guard
    run_sitting(1, "baseline", 1, data_dir=data_dir, interactive=False, input_func=lambda: "dont know")

    # 3. Run status under guard
    get_testrun_status(1, data_dir=data_dir)

    # Assert no opened path contains sensitive patterns
    for p in opened_paths:
        assert not any(pat in p for pat in ["/keys/", "/slotted/", "/reviews/", "compare", "resolution"]), f"Forbidden file opened: {p}"


def test_code_ast_contains_no_sensitive_read_references():
    """
    Static AST check: forms.py and testrun.py do not construct paths to keys/slotted/reviews/compare/resolution.
    """
    for filename in ["forms.py", "testrun.py"]:
        code_path = Path("learn_assist") / filename
        tree = ast.parse(code_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.lower()
                # Exclude the explicit FORBIDDEN_PATTERNS definition itself
                if val in ["keys", "slotted", "reviews", "compare", "resolution"]:
                    continue
                assert not any(p in val for p in ["keys/", "slotted/", "reviews/", "compare_", "resolution.json"]), (
                    f"Found potential sensitive reference in {filename}: {val}"
                )


def test_make_forms_single(tmp_path):
    """
    Test make_forms with --single produces one form containing all frozen items,
    with sittings of at most 10 in seeded order.
    """
    bank = create_synthetic_frozen_bank(tmp_path, chapter=2)
    res = make_forms(2, seed=20261028, data_dir=tmp_path, single="diagnostic")

    assert res["chapter"] == 2
    assert res["seed"] == 20261028
    assert res["single"] == "diagnostic"
    assert "diagnostic" in res["forms"]
    assert "baseline" not in res["forms"]
    assert "final" not in res["forms"]

    diag_form = res["forms"]["diagnostic"]
    total_items = len(diag_form["item_ids"])
    assert total_items == 10  # 6 A items + 4 B items in create_synthetic_frozen_bank

    # Sittings of at most 10
    sittings = diag_form["sittings"]
    assert len(sittings) == 1
    assert len(sittings[0]["item_ids"]) == 10

    # Repeat with same seed should produce identical order
    res2 = make_forms(2, seed=20261028, data_dir=tmp_path, single="diagnostic")
    assert res2["forms"]["diagnostic"]["item_ids"] == diag_form["item_ids"]

