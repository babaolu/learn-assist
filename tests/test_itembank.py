"""
Unit tests for learn_assist.itembank.

ALL tests use purely synthetic fixtures. No real book content or exercise text.
Covers:
- Extraction of fake exercise section with sub-parts, figure references, and errata
- Schema validation pass and failure modes
- Numeric compare across units (e.g. 1 T vs 10000 G, Hz vs rad/s)
- Expression compare equal but written differently (simplification & numeric spot-check)
- Disputed numeric values outside rel_tol
- Text part resolution to needs_grading
- Report output containing strictly no problem statement or key text
"""

import json
import pytest
from pathlib import Path
from learn_assist.itembank import (
    extract_chapter_a,
    validate_key,
    validate_key_file,
    compare_numeric,
    compare_expression,
    compare_expression_status,
    compare_choice,
    compare_chapter_keys,
    generate_report,
    normalize_unit,
    parse_expr_convention,
    validate_slots,
    validate_slot_file,
    validate_slotted_key,
    validate_slotted_key_file,
    compare_chapter_slots
)


# ---------------------------------------------------------------------------
# Synthetic Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_a_synthetic(tmp_path: Path):
    """
    Tests extraction of a synthetic chapter with:
    - Multiple problems
    - Sub-parts (a), (b), (c)
    - References to equations and figures
    - Figure presence in cross_model
    - Synthetic errata matching
    """
    data_dir = tmp_path / "data"
    printed_dir = data_dir / "chapters_printed"
    cross_dir = data_dir / "cross_model"
    printed_dir.mkdir(parents=True)
    cross_dir.mkdir(parents=True)

    # Synthetic cross_model page with a figure description
    fake_fig_desc = "[Figure: synthetic test schematic diagram of coil]"
    (cross_dir / "page_0010_cross.md").write_text(f"Some text\n{fake_fig_desc}\n", encoding="utf-8")

    # Synthetic erratum
    errata_data = [
        {
            "id": "E_TEST_01",
            "pdf_page": 10,
            "printed": "SYNTHETIC_PRINTED_TYPO",
            "corrected": "SYNTHETIC_CORRECTED",
            "status": "approved"
        }
    ]
    (data_dir / "errata.json").write_text(json.dumps(errata_data), encoding="utf-8")

    # Synthetic printed chapter
    fake_chapter_md = """<!-- PDF Page 9 | Book Page 1 -->
# Chapter 1 Synthetic Mechanics

Some intro text.

<!-- PDF Page 10 | Book Page 2 -->
## Exercises

1.1 First synthetic item question without subparts.
Refers to Eq. (1.5) and Fig. 1.2.
Also mentions SYNTHETIC_PRINTED_TYPO in the statement.
Notice shown in the following figure for testing.

1.2 Second synthetic problem with multiple subparts.
(a) First subpart synthetic question text.
(b) Second subpart synthetic question text.
(c) Third subpart synthetic question text.
Refers to Eq. (1.12).
"""
    (printed_dir / "chapter_01.md").write_text(fake_chapter_md, encoding="utf-8")

    out_items_dir = data_dir / "itembank" / "ch01" / "items"
    count, ids, warnings = extract_chapter_a(1, data_dir=data_dir, output_dir=out_items_dir)

    assert count == 2
    assert ids == ["A_1.01", "A_1.02"]
    assert len(warnings) == 0

    # Inspect item 1.01
    item1_path = out_items_dir / "A_1.01.json"
    assert item1_path.exists()
    item1 = json.loads(item1_path.read_text(encoding="utf-8"))
    assert item1["item_id"] == "A_1.01"
    assert item1["bucket"] == "A"
    assert item1["source_label"] == "1.1"
    assert item1["pdf_page"] == 10
    assert item1["parts"] == []
    assert "Eq. (1.5)" in item1["references"]
    assert "Fig. 1.2" in item1["references"]
    assert item1["needs_figure"] is True
    assert fake_fig_desc in item1["figure_descriptions"]
    assert item1["errata_ids"] == ["E_TEST_01"]
    assert item1["status"] == "unkeyed"

    # Inspect item 1.02
    item2_path = out_items_dir / "A_1.02.json"
    assert item2_path.exists()
    item2 = json.loads(item2_path.read_text(encoding="utf-8"))
    assert item2["item_id"] == "A_1.02"
    assert item2["source_label"] == "1.2"
    assert item2["parts"] == ["a", "b", "c"]
    assert "Eq. (1.12)" in item2["references"]
    assert item2["needs_figure"] is False
    assert item2["errata_ids"] == []


# ---------------------------------------------------------------------------
# Key Schema Validation Tests
# ---------------------------------------------------------------------------

def test_key_schema_validation_pass(tmp_path: Path):
    """Tests that a fully valid key passes validation."""
    valid_key = {
        "item_id": "A_1.01",
        "model_family": "gemini",
        "model_id": "gemini-pro-test",
        "solved_at": "2026-10-18T10:00:00Z",
        "solution_md": "Step 1: calculate. Step 2: result.",
        "final_answers": [
            {
                "part": "",
                "kind": "numeric",
                "value": "42.0",
                "unit": "T",
                "rel_tol": 0.02
            }
        ],
        "assumptions": ["synthetic assumption"],
        "confidence": "high"
    }
    valid, errors = validate_key(valid_key)
    assert valid is True
    assert len(errors) == 0

    # Also test valid key from file
    key_dir = tmp_path / "keys" / "gemini"
    key_dir.mkdir(parents=True)
    key_file = key_dir / "A_1.01.json"
    key_file.write_text(json.dumps(valid_key), encoding="utf-8")

    f_valid, f_errors = validate_key_file(key_file)
    assert f_valid is True
    assert len(f_errors) == 0


def test_key_schema_validation_failures():
    """Tests various invalid key structures fail with informative error messages."""
    # 1. Missing required field
    bad1 = {"item_id": "A_1.01"}
    valid, errors = validate_key(bad1)
    assert valid is False
    assert any("Missing required field" in e for e in errors)

    # 2. Invalid model family
    bad2 = {
        "item_id": "A_1.01", "model_family": "unsupported_llm", "model_id": "m1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "...", "final_answers": [],
        "assumptions": [], "confidence": "high"
    }
    valid, errors = validate_key(bad2)
    assert valid is False
    assert any("Invalid model_family" in e for e in errors)

    # 3. Invalid confidence
    bad3 = {
        "item_id": "A_1.01", "model_family": "claude", "model_id": "m1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "...", "final_answers": [],
        "assumptions": [], "confidence": "extreme"
    }
    valid, errors = validate_key(bad3)
    assert valid is False
    assert any("Invalid confidence" in e for e in errors)

    # 4. Expression without sympy parseable string
    bad4 = {
        "item_id": "A_1.01", "model_family": "claude", "model_id": "m1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "...",
        "final_answers": [{"part": "a", "kind": "expression", "value": "x^2", "sympy": "x *** 2 + /"}],
        "assumptions": [], "confidence": "medium"
    }
    valid, errors = validate_key(bad4)
    assert valid is False
    assert any("sympy parse error" in e for e in errors)

    # 5. Numeric value not evaluatable
    bad5 = {
        "item_id": "A_1.01", "model_family": "claude", "model_id": "m1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "...",
        "final_answers": [{"part": "a", "kind": "numeric", "value": "not_a_number"}],
        "assumptions": [], "confidence": "low"
    }
    valid, errors = validate_key(bad5)
    assert valid is False
    assert any("could not be evaluated" in e for e in errors)


# ---------------------------------------------------------------------------
# Unit Comparison & Tolerance Tests
# ---------------------------------------------------------------------------

def test_numeric_compare_across_units():
    """Tests unit-normalised numeric comparison (e.g. 1 T vs 10000 G, Hz vs rad/s)."""
    # 1 Tesla vs 10,000 Gauss
    ans_t = {"kind": "numeric", "value": "1.0", "unit": "T", "rel_tol": 0.02}
    ans_g = {"kind": "numeric", "value": "10000.0", "unit": "G", "rel_tol": 0.02}
    assert compare_numeric(ans_t, ans_g) is True

    # 0.5 T vs 5000 G
    assert compare_numeric(
        {"kind": "numeric", "value": "0.5", "unit": "T"},
        {"kind": "numeric", "value": "5000", "unit": "G"}
    ) is True

    # Frequency: 1 Hz vs 2*pi rad/s
    ans_hz = {"kind": "numeric", "value": "1.0", "unit": "Hz"}
    ans_rad = {"kind": "numeric", "value": "6.2831853", "unit": "rad/s"}
    assert compare_numeric(ans_hz, ans_rad) is True

    # SI prefixes: 1 MHz vs 1000 kHz
    ans_mhz = {"kind": "numeric", "value": "1.0", "unit": "MHz"}
    ans_khz = {"kind": "numeric", "value": "1000.0", "unit": "kHz"}
    assert compare_numeric(ans_mhz, ans_khz) is True

    # Incompatible units
    assert compare_numeric(
        {"kind": "numeric", "value": "1.0", "unit": "T"},
        {"kind": "numeric", "value": "1.0", "unit": "s"}
    ) is False


def test_numeric_compare_disputed():
    """Tests that numeric values exceeding rel_tol are flagged as disputed."""
    ans1 = {"kind": "numeric", "value": "100.0", "unit": "T", "rel_tol": 0.02}
    ans2 = {"kind": "numeric", "value": "105.0", "unit": "T", "rel_tol": 0.02}
    assert compare_numeric(ans1, ans2) is False


# ---------------------------------------------------------------------------
# Expression & Choice Comparison Tests
# ---------------------------------------------------------------------------

def test_expression_compare_equal_different_form():
    """Tests expressions that are symbolically equal but formatted differently."""
    # Trigonometric identity
    expr1 = "sin(2*x)"
    expr2 = "2*sin(x)*cos(x)"
    assert compare_expression(expr1, expr2) is True

    # Polynomial expansion
    expr3 = "(x + 1)**2"
    expr4 = "x**2 + 2*x + 1"
    assert compare_expression(expr3, expr4) is True

    # Unequal expressions
    assert compare_expression("x + 1", "x + 2") is False


def test_choice_compare():
    """Tests choice answers comparison."""
    assert compare_choice("A", "a") is True
    assert compare_choice(" True ", "true") is True
    assert compare_choice("A", "B") is False


def test_expression_convention_parsing_and_equality():
    """
    Tests expression parsing and comparison with convention symbols:
    - 'gamma*B0' parses as a product of two Symbol instances.
    - '2*pi*f0' equals 'f0*2*pi'.
    """
    import sympy
    from sympy import Mul, Symbol

    # 'gamma*B0' parses as a product of two Symbol instances
    expr1 = parse_expr_convention("gamma*B0")
    assert isinstance(expr1, Mul)
    assert len(expr1.args) == 2
    assert all(isinstance(a, Symbol) for a in expr1.args)
    assert {a.name for a in expr1.args} == {"gamma", "B0"}

    # '2*pi*f0' equals 'f0*2*pi'
    assert compare_expression("2*pi*f0", "f0*2*pi") is True


def test_expression_symbol_mismatch_item_needs_grading(tmp_path: Path):
    """
    Tests that comparison of 'gyro*B0' vs 'gamma*B0' gives
    part status 'symbol_mismatch' and item status 'needs_grading' (not disputed).
    """
    # Direct expression comparison status check
    st = compare_expression_status("gyro*B0", "gamma*B0")
    assert st == "symbol_mismatch"

    # Full item-bank comparison check
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    items_dir = ch_dir / "items"
    keys_claude = ch_dir / "keys" / "claude"
    keys_gemini = ch_dir / "keys" / "gemini"
    items_dir.mkdir(parents=True)
    keys_claude.mkdir(parents=True)
    keys_gemini.mkdir(parents=True)

    item_id = "A_1.01"
    item_obj = {
        "item_id": item_id,
        "bucket": "A",
        "source_label": "1.1",
        "pdf_page": 1,
        "statement_md": "Synthetic problem statement",
        "parts": ["a"],
        "references": [],
        "needs_figure": False,
        "figure_descriptions": [],
        "errata_ids": [],
        "status": "unkeyed"
    }
    (items_dir / f"{item_id}.json").write_text(json.dumps(item_obj), encoding="utf-8")

    key_c = {
        "item_id": item_id, "model_family": "claude", "model_id": "c1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "...",
        "final_answers": [{"part": "a", "kind": "expression", "value": "gyro*B0", "sympy": "gyro*B0"}],
        "assumptions": [], "confidence": "high"
    }
    key_g = {
        "item_id": item_id, "model_family": "gemini", "model_id": "g1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "...",
        "final_answers": [{"part": "a", "kind": "expression", "value": "gamma*B0", "sympy": "gamma*B0"}],
        "assumptions": [], "confidence": "high"
    }
    (keys_claude / f"{item_id}.json").write_text(json.dumps(key_c), encoding="utf-8")
    (keys_gemini / f"{item_id}.json").write_text(json.dumps(key_g), encoding="utf-8")

    rep = compare_chapter_keys(1, data_dir=data_dir)
    res = rep["results"][item_id]

    assert res["status"] == "needs_grading"
    assert len(res["parts"]) == 1
    assert res["parts"][0]["status"] == "symbol_mismatch"
    assert rep["summary"]["needs_grading"] == 1
    assert rep["summary"]["key_disputed"] == 0

    item_after = json.loads((items_dir / f"{item_id}.json").read_text(encoding="utf-8"))
    assert item_after["status"] == "needs_grading"


# ---------------------------------------------------------------------------
# Full Dual-Model Compare and Blinding Verification
# ---------------------------------------------------------------------------

def test_compare_chapter_keys_and_blinding(tmp_path: Path):
    """
    Tests dual-family comparison:
    - Item 1: verified (numeric agreement)
    - Item 2: disputed (numeric disagreement)
    - Item 3: text part giving needs_grading
    - Report generation: counts only, no statement or solution text
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    items_dir = ch_dir / "items"
    keys_claude = ch_dir / "keys" / "claude"
    keys_gemini = ch_dir / "keys" / "gemini"
    items_dir.mkdir(parents=True)
    keys_claude.mkdir(parents=True)
    keys_gemini.mkdir(parents=True)

    # 3 Items with synthetic statement text
    for i, item_id in enumerate(["A_1.01", "A_1.02", "A_1.03"]):
        item_obj = {
            "item_id": item_id,
            "bucket": "A",
            "source_label": f"1.{i+1}",
            "pdf_page": 10,
            "statement_md": f"SECRET_FIXTURE_PROBLEM_STATEMENT_{item_id}",
            "parts": ["a"],
            "references": [],
            "needs_figure": (item_id == "A_1.01"),
            "figure_descriptions": [],
            "errata_ids": [],
            "status": "unkeyed"
        }
        (items_dir / f"{item_id}.json").write_text(json.dumps(item_obj), encoding="utf-8")

    # Key 1: Both agree on 1 T vs 10000 G -> key_verified
    key_c1 = {
        "item_id": "A_1.01", "model_family": "claude", "model_id": "c1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "SECRET_SOLUTION_C1",
        "final_answers": [{"part": "a", "kind": "numeric", "value": "1.0", "unit": "T", "rel_tol": 0.02}],
        "assumptions": [], "confidence": "high"
    }
    key_g1 = {
        "item_id": "A_1.01", "model_family": "gemini", "model_id": "g1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "SECRET_SOLUTION_G1",
        "final_answers": [{"part": "a", "kind": "numeric", "value": "10000", "unit": "G", "rel_tol": 0.02}],
        "assumptions": [], "confidence": "high"
    }
    (keys_claude / "A_1.01.json").write_text(json.dumps(key_c1), encoding="utf-8")
    (keys_gemini / "A_1.01.json").write_text(json.dumps(key_g1), encoding="utf-8")

    # Key 2: Disputed numeric (10 vs 20) -> key_disputed
    key_c2 = {
        "item_id": "A_1.02", "model_family": "claude", "model_id": "c1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "SECRET_SOLUTION_C2",
        "final_answers": [{"part": "a", "kind": "numeric", "value": "10.0", "unit": "T", "rel_tol": 0.02}],
        "assumptions": [], "confidence": "high"
    }
    key_g2 = {
        "item_id": "A_1.02", "model_family": "gemini", "model_id": "g1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "SECRET_SOLUTION_G2",
        "final_answers": [{"part": "a", "kind": "numeric", "value": "20.0", "unit": "T", "rel_tol": 0.02}],
        "assumptions": [], "confidence": "medium"
    }
    (keys_claude / "A_1.02.json").write_text(json.dumps(key_c2), encoding="utf-8")
    (keys_gemini / "A_1.02.json").write_text(json.dumps(key_g2), encoding="utf-8")

    # Key 3: Text part -> needs_grading
    key_c3 = {
        "item_id": "A_1.03", "model_family": "claude", "model_id": "c1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "SECRET_SOLUTION_C3",
        "final_answers": [{"part": "a", "kind": "text", "value": "Explanation A"}],
        "assumptions": [], "confidence": "high"
    }
    key_g3 = {
        "item_id": "A_1.03", "model_family": "gemini", "model_id": "g1",
        "solved_at": "2026-10-18T10:00:00Z", "solution_md": "SECRET_SOLUTION_G3",
        "final_answers": [{"part": "a", "kind": "text", "value": "Explanation B"}],
        "assumptions": [], "confidence": "high"
    }
    (keys_claude / "A_1.03.json").write_text(json.dumps(key_c3), encoding="utf-8")
    (keys_gemini / "A_1.03.json").write_text(json.dumps(key_g3), encoding="utf-8")

    # Run compare
    report_dict = compare_chapter_keys(1, data_dir=data_dir)
    assert report_dict["items_compared"] == 3
    assert report_dict["summary"]["key_verified"] == 1
    assert report_dict["summary"]["key_disputed"] == 1
    assert report_dict["summary"]["needs_grading"] == 1

    # Check updated item statuses
    assert json.loads((items_dir / "A_1.01.json").read_text(encoding="utf-8"))["status"] == "key_verified"
    assert json.loads((items_dir / "A_1.02.json").read_text(encoding="utf-8"))["status"] == "key_disputed"
    assert json.loads((items_dir / "A_1.03.json").read_text(encoding="utf-8"))["status"] == "needs_grading"

    # Run report
    rep_output = generate_report(1, data_dir=data_dir)

    # Blinding assertions: verify counts present and NO secret fixture text is leaked
    assert "Total items: 3" in rep_output
    assert "Bucket A: 3" in rep_output
    assert "key_verified: 1" in rep_output
    assert "key_disputed: 1" in rep_output
    assert "needs_grading: 1" in rep_output
    assert "Items requiring figure: 1" in rep_output
    assert "claude: 3" in rep_output
    assert "gemini: 3" in rep_output

    for item_id in ["A_1.01", "A_1.02", "A_1.03"]:
        assert f"SECRET_FIXTURE_PROBLEM_STATEMENT_{item_id}" not in rep_output
        assert f"SECRET_SOLUTION_C" not in rep_output
        assert f"SECRET_SOLUTION_G" not in rep_output
        assert "Explanation A" not in rep_output
        assert "Explanation B" not in rep_output


# ---------------------------------------------------------------------------
# Slot Schema and Slotted Key Tests
# ---------------------------------------------------------------------------

def test_slot_validation_pass_and_fail(tmp_path: Path):
    """Tests slot schema validation pass and failure modes."""
    valid_slots = {
        "item_id": "A_1.01",
        "slots": [
            {
                "slot_id": "s1",
                "part": "a",
                "asks": "Magnitude of field",
                "kind": "numeric",
                "unit": "T",
                "rel_tol": 0.02
            },
            {
                "slot_id": "s2",
                "part": "b",
                "asks": "Larmor frequency",
                "kind": "expression",
                "symbols_allowed": ["gamma", "B0"]
            },
            {
                "slot_id": "s3",
                "part": "c",
                "asks": "Axis",
                "kind": "choice",
                "choices": ["x", "y", "z"]
            },
            {
                "slot_id": "s4",
                "part": "",
                "asks": "Mechanism description",
                "kind": "text"
            }
        ]
    }
    ok, errs = validate_slots(valid_slots)
    assert ok is True
    assert len(errs) == 0

    # Also test file validation
    slot_file = tmp_path / "A_1.01.json"
    slot_file.write_text(json.dumps(valid_slots), encoding="utf-8")
    ok_f, errs_f = validate_slot_file(slot_file)
    assert ok_f is True
    assert len(errs_f) == 0

    # 1. Duplicate slot_id
    bad_dup = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Ask 1", "kind": "text"},
            {"slot_id": "s1", "asks": "Ask 2", "kind": "text"}
        ]
    }
    ok, errs = validate_slots(bad_dup)
    assert ok is False
    assert any("Duplicate slot_id" in e for e in errs)

    # 2. Numeric slot missing unit
    bad_num = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Field", "kind": "numeric"}
        ]
    }
    ok, errs = validate_slots(bad_num)
    assert ok is False
    assert any("requires non-empty string 'unit'" in e for e in errs)

    # 3. Choice slot missing choices
    bad_choice = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Axis", "kind": "choice", "choices": []}
        ]
    }
    ok, errs = validate_slots(bad_choice)
    assert ok is False
    assert any("requires non-empty list 'choices'" in e for e in errs)

    # 4. Expression slot with non-convention symbol in symbols_allowed
    bad_expr = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Frequency", "kind": "expression", "symbols_allowed": ["gamma", "NON_CONV_SYM"]}
        ]
    }
    ok, errs = validate_slots(bad_expr)
    assert ok is False
    assert any("contains non-convention symbol" in e for e in errs)


def test_slotted_key_validation_cases():
    """Tests slotted key validation: wrong kind, missing slot, bad unit, and pass modes."""
    slots_spec = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Field", "kind": "numeric", "unit": "T"},
            {"slot_id": "s2", "asks": "Frequency", "kind": "expression", "symbols_allowed": ["gamma", "B0"]},
            {"slot_id": "s3", "asks": "Axis", "kind": "choice", "choices": ["x", "y", "z"]},
            {"slot_id": "s4", "asks": "Text", "kind": "text"}
        ]
    }

    # Pass case
    valid_key = {
        "item_id": "A_1.01",
        "model_family": "claude",
        "model_id": "c1",
        "filled_at": "2026-10-22T00:00:00Z",
        "answers": {
            "s1": {"value": "1.5", "unit": "T"},
            "s2": {"sympy": "gamma * B0"},
            "s3": {"value": "x"},
            "s4": {"value": "A valid explanation."}
        },
        "unanswerable": []
    }
    ok, errs = validate_slotted_key(valid_key, slots_spec)
    assert ok is True
    assert len(errs) == 0

    # Pass case with unanswerable
    valid_key_unans = {
        "item_id": "A_1.01",
        "model_family": "gemini",
        "model_id": "g1",
        "filled_at": "2026-10-22T00:00:00Z",
        "answers": {
            "s1": {"value": "15000", "unit": "G"},  # Gauss converts to Tesla
            "s2": {"sympy": "B0 * gamma"},
            "s3": {"value": "Y"}
        },
        "unanswerable": ["s4"]
    }
    ok, errs = validate_slotted_key(valid_key_unans, slots_spec)
    assert ok is True
    assert len(errs) == 0

    # 1. Wrong kind - non-numeric value in numeric slot
    bad_kind_num = dict(valid_key, answers=dict(valid_key["answers"], s1={"value": "not_a_number", "unit": "T"}))
    ok, errs = validate_slotted_key(bad_kind_num, slots_spec)
    assert ok is False
    assert any("could not be evaluated" in e for e in errs)

    # 2. Wrong kind - disallowed symbol in expression slot
    bad_kind_expr = dict(valid_key, answers=dict(valid_key["answers"], s2={"sympy": "gamma * B0 * omega0"}))
    ok, errs = validate_slotted_key(bad_kind_expr, slots_spec)
    assert ok is False
    assert any("disallowed symbol" in e for e in errs)

    # 3. Wrong kind - choice not in allowed list
    bad_kind_choice = dict(valid_key, answers=dict(valid_key["answers"], s3={"value": "w"}))
    ok, errs = validate_slotted_key(bad_kind_choice, slots_spec)
    assert ok is False
    assert any("not in allowed choices" in e for e in errs)

    # 4. Missing slot (neither in answers nor in unanswerable)
    bad_missing = dict(valid_key, answers={k: v for k, v in valid_key["answers"].items() if k != "s4"})
    ok, errs = validate_slotted_key(bad_missing, slots_spec)
    assert ok is False
    assert any("Missing slots from key" in e for e in errs)

    # 5. Bad unit (numeric slot with 'T' given incompatible unit 's')
    bad_unit = dict(valid_key, answers=dict(valid_key["answers"], s1={"value": "1.0", "unit": "s"}))
    ok, errs = validate_slotted_key(bad_unit, slots_spec)
    assert ok is False
    assert any("does not convert to slot unit" in e for e in errs)


def test_compare_slots_scenarios(tmp_path: Path):
    """
    Tests compare-slots with:
    - agree case -> key_verified
    - disagree case -> key_disputed
    - text case -> key_verified_partial
    - unanswered case -> incomplete
    And verifies report output uses slot results.
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    slots_dir = ch_dir / "slots"
    items_dir = ch_dir / "items"
    keys_c_dir = ch_dir / "slotted" / "claude"
    keys_g_dir = ch_dir / "slotted" / "gemini"

    for d in [slots_dir, items_dir, keys_c_dir, keys_g_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup 4 synthetic items
    # Item 1: Agree case (numeric 1 T vs 10000 G, and choice "x" vs "x") -> key_verified
    s1 = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Field", "kind": "numeric", "unit": "T"},
            {"slot_id": "s2", "asks": "Axis", "kind": "choice", "choices": ["x", "y"]}
        ]
    }
    kc1 = {
        "item_id": "A_1.01", "model_family": "claude", "model_id": "c1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "1.0", "unit": "T"}, "s2": {"value": "x"}}, "unanswerable": []
    }
    kg1 = {
        "item_id": "A_1.01", "model_family": "gemini", "model_id": "g1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "10000", "unit": "G"}, "s2": {"value": "X"}}, "unanswerable": []
    }

    # Item 2: Disagree case (numeric 1.0 T vs 2.0 T) -> key_disputed
    s2 = {
        "item_id": "A_1.02",
        "slots": [
            {"slot_id": "s1", "asks": "Field", "kind": "numeric", "unit": "T"}
        ]
    }
    kc2 = {
        "item_id": "A_1.02", "model_family": "claude", "model_id": "c1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "1.0", "unit": "T"}}, "unanswerable": []
    }
    kg2 = {
        "item_id": "A_1.02", "model_family": "gemini", "model_id": "g1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "2.0", "unit": "T"}}, "unanswerable": []
    }

    # Item 3: Text case (non-text agrees + text slot remains) -> key_verified_partial
    s3 = {
        "item_id": "A_1.03",
        "slots": [
            {"slot_id": "s1", "asks": "Axis", "kind": "choice", "choices": ["x", "y"]},
            {"slot_id": "s2", "asks": "Explanation", "kind": "text"}
        ]
    }
    kc3 = {
        "item_id": "A_1.03", "model_family": "claude", "model_id": "c1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "x"}, "s2": {"value": "Explanation from Claude"}}, "unanswerable": []
    }
    kg3 = {
        "item_id": "A_1.03", "model_family": "gemini", "model_id": "g1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "x"}, "s2": {"value": "Explanation from Gemini"}}, "unanswerable": []
    }

    # Item 4: Unanswered case (s1 answered, s2 unanswerable in one key) -> incomplete
    s4 = {
        "item_id": "A_1.04",
        "slots": [
            {"slot_id": "s1", "asks": "Field", "kind": "numeric", "unit": "T"},
            {"slot_id": "s2", "asks": "Frequency", "kind": "expression", "symbols_allowed": ["gamma", "B0"]}
        ]
    }
    kc4 = {
        "item_id": "A_1.04", "model_family": "claude", "model_id": "c1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "1.0", "unit": "T"}, "s2": {"sympy": "gamma * B0"}}, "unanswerable": []
    }
    kg4 = {
        "item_id": "A_1.04", "model_family": "gemini", "model_id": "g1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {"s1": {"value": "1.0", "unit": "T"}}, "unanswerable": ["s2"]
    }

    for idx, (s_data, c_data, g_data) in enumerate([(s1, kc1, kg1), (s2, kc2, kg2), (s3, kc3, kg3), (s4, kc4, kg4)], 1):
        i_id = f"A_1.{idx:02d}"
        (slots_dir / f"{i_id}.json").write_text(json.dumps(s_data), encoding="utf-8")
        (keys_c_dir / f"{i_id}.json").write_text(json.dumps(c_data), encoding="utf-8")
        (keys_g_dir / f"{i_id}.json").write_text(json.dumps(g_data), encoding="utf-8")
        item_obj = {"item_id": i_id, "bucket": "A", "source_label": f"1.{idx}", "status": "unkeyed"}
        (items_dir / f"{i_id}.json").write_text(json.dumps(item_obj), encoding="utf-8")

    # Run compare-slots
    report_dict = compare_chapter_slots(1, data_dir=data_dir)
    assert report_dict["items_compared"] == 4

    summary = report_dict["summary"]
    assert summary["key_verified"] == 1
    assert summary["key_disputed"] == 1
    assert summary["key_verified_partial"] == 1
    assert summary["incomplete"] == 1

    res = report_dict["results"]
    assert res["A_1.01"]["status"] == "key_verified"
    assert res["A_1.02"]["status"] == "key_disputed"
    assert res["A_1.03"]["status"] == "key_verified_partial"
    assert res["A_1.04"]["status"] == "incomplete"

    # Verify compare_slots.json exists on disk
    cmp_file = ch_dir / "compare_slots.json"
    assert cmp_file.exists()

    # Verify updated item statuses on disk
    assert json.loads((items_dir / "A_1.01.json").read_text(encoding="utf-8"))["status"] == "key_verified"
    assert json.loads((items_dir / "A_1.02.json").read_text(encoding="utf-8"))["status"] == "key_disputed"
    assert json.loads((items_dir / "A_1.03.json").read_text(encoding="utf-8"))["status"] == "key_verified_partial"
    assert json.loads((items_dir / "A_1.04.json").read_text(encoding="utf-8"))["status"] == "incomplete"

    # Run report command: must show slot-based statuses and slot counts by kind
    rep_text = generate_report(1, data_dir=data_dir)
    assert "Items by status (slot-based):" in rep_text
    assert "key_verified: 1" in rep_text
    assert "key_disputed: 1" in rep_text
    assert "key_verified_partial: 1" in rep_text
    assert "incomplete: 1" in rep_text
    assert "Slots by kind:" in rep_text
    assert "numeric: 3" in rep_text
    assert "choice: 2" in rep_text
    assert "text: 1" in rep_text
    assert "expression: 1" in rep_text
