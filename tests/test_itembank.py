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
    compare_chapter_slots,
    generate_review_queue,
    resolve_chapter,
    extract_concepts_from_chapter,
    check_statement_option_lists,
    validate_item,
    validate_item_file,
    difficulty_screen_chapter,
    select_b_chapter,
    check_coverage_chapter
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


# ---------------------------------------------------------------------------
# Review Queue and Resolve Tests
# ---------------------------------------------------------------------------

def test_slot_validation_allows_pi():
    """Tests that 'pi' is allowed in symbols_allowed for expression slots."""
    slot_data = {
        "item_id": "A_1.01",
        "slots": [
            {
                "slot_id": "s1",
                "asks": "Expression with pi",
                "kind": "expression",
                "symbols_allowed": ["B0", "gamma", "pi"]
            }
        ]
    }
    ok, errs = validate_slots(slot_data)
    assert ok is True
    assert len(errs) == 0

    # Non-convention symbol still fails
    slot_data_bad = {
        "item_id": "A_1.01",
        "slots": [
            {
                "slot_id": "s1",
                "asks": "Expression with unknown",
                "kind": "expression",
                "symbols_allowed": ["B0", "unknown_sym"]
            }
        ]
    }
    ok_bad, errs_bad = validate_slots(slot_data_bad)
    assert ok_bad is False
    assert any("contains non-convention symbol" in e for e in errs_bad)


def test_review_queue_and_resolve_rules(tmp_path: Path):
    """
    Synthetic tests covering:
    - review-queue generation (queues non-agreed slots: text, disputed, unanswered)
    - resolve rules:
        * both 'equivalent' -> verified
        * claude 'mine_wrong' + gemini 'other_wrong' -> resolved_by_concession (gemini answer adopted)
        * gemini 'mine_wrong' + claude 'other_wrong' -> resolved_by_concession (claude answer adopted)
        * both 'other_wrong' -> disputed
        * both 'mine_wrong' -> disputed
        * 'both_wrong' -> disputed
        * 'unclear' -> disputed
        * missing review -> disputed
    - Item usability (all slots agreed/verified/resolved_by_concession -> usable; any disputed -> excluded)
    - resolution.json disk format
    - report output displaying usability and slot outcomes by type
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    slots_dir = ch_dir / "slots"
    items_dir = ch_dir / "items"
    keys_c_dir = ch_dir / "slotted" / "claude"
    keys_g_dir = ch_dir / "slotted" / "gemini"
    reviews_dir = ch_dir / "reviews"

    for d in [slots_dir, items_dir, keys_c_dir, keys_g_dir, reviews_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Item 1: Will be fully usable
    # s1: numeric agree (not queued)
    # s2: text, both say equivalent -> verified
    # s3: numeric dispute, claude concedes (mine_wrong) -> resolved_by_concession
    # s4: numeric dispute, gemini concedes (mine_wrong) -> resolved_by_concession
    s_item1 = {
        "item_id": "A_1.01",
        "slots": [
            {"slot_id": "s1", "asks": "Field", "kind": "numeric", "unit": "T"},
            {"slot_id": "s2", "asks": "Text explanation", "kind": "text"},
            {"slot_id": "s3", "asks": "Frequency", "kind": "numeric", "unit": "Hz"},
            {"slot_id": "s4", "asks": "Time", "kind": "numeric", "unit": "s"}
        ]
    }
    kc_item1 = {
        "item_id": "A_1.01", "model_family": "claude", "model_id": "c1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {
            "s1": {"value": "1.0", "unit": "T"},
            "s2": {"value": "Claude text explanation"},
            "s3": {"value": "100.0", "unit": "Hz"},  # Claude will concede this was wrong
            "s4": {"value": "2.0", "unit": "s"}      # Claude is correct
        },
        "unanswerable": []
    }
    kg_item1 = {
        "item_id": "A_1.01", "model_family": "gemini", "model_id": "g1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {
            "s1": {"value": "1.0", "unit": "T"},
            "s2": {"value": "Gemini text explanation"},
            "s3": {"value": "200.0", "unit": "Hz"},  # Gemini is correct
            "s4": {"value": "5.0", "unit": "s"}      # Gemini will concede this was wrong
        },
        "unanswerable": []
    }

    # Item 2: Will be excluded due to disputes
    # s1: both 'other_wrong' -> disputed
    # s2: both 'mine_wrong' -> disputed
    # s3: 'both_wrong' -> disputed
    # s4: 'unclear' -> disputed
    # s5: missing review in gemini -> disputed
    s_item2 = {
        "item_id": "A_1.02",
        "slots": [
            {"slot_id": "s1", "asks": "Ask 1", "kind": "numeric", "unit": "T"},
            {"slot_id": "s2", "asks": "Ask 2", "kind": "numeric", "unit": "T"},
            {"slot_id": "s3", "asks": "Ask 3", "kind": "numeric", "unit": "T"},
            {"slot_id": "s4", "asks": "Ask 4", "kind": "numeric", "unit": "T"},
            {"slot_id": "s5", "asks": "Ask 5", "kind": "numeric", "unit": "T"}
        ]
    }
    kc_item2 = {
        "item_id": "A_1.02", "model_family": "claude", "model_id": "c1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {
            "s1": {"value": "1.0", "unit": "T"},
            "s2": {"value": "1.0", "unit": "T"},
            "s3": {"value": "1.0", "unit": "T"},
            "s4": {"value": "1.0", "unit": "T"},
            "s5": {"value": "1.0", "unit": "T"}
        },
        "unanswerable": []
    }
    kg_item2 = {
        "item_id": "A_1.02", "model_family": "gemini", "model_id": "g1", "filled_at": "2026-10-22T00:00:00Z",
        "answers": {
            "s1": {"value": "2.0", "unit": "T"},
            "s2": {"value": "2.0", "unit": "T"},
            "s3": {"value": "2.0", "unit": "T"},
            "s4": {"value": "2.0", "unit": "T"},
            "s5": {"value": "2.0", "unit": "T"}
        },
        "unanswerable": []
    }

    # Write items, slots, slotted keys
    for item_data, kc_data, kg_data in [(s_item1, kc_item1, kg_item1), (s_item2, kc_item2, kg_item2)]:
        iid = item_data["item_id"]
        (slots_dir / f"{iid}.json").write_text(json.dumps(item_data), encoding="utf-8")
        (keys_c_dir / f"{iid}.json").write_text(json.dumps(kc_data), encoding="utf-8")
        (keys_g_dir / f"{iid}.json").write_text(json.dumps(kg_data), encoding="utf-8")
        (items_dir / f"{iid}.json").write_text(json.dumps({"item_id": iid, "bucket": "A", "status": "unkeyed"}), encoding="utf-8")

    # Run compare-slots first
    compare_chapter_slots(1, data_dir=data_dir)

    # 1. Test review-queue
    queue = generate_review_queue(1, data_dir=data_dir)
    # A_1.01 has 3 non-agreed slots (s2, s3, s4)
    # A_1.02 has 5 non-agreed slots (s1, s2, s3, s4, s5)
    # Total queued: 8 slots
    assert len(queue) == 8
    q_file = ch_dir / "review_queue.json"
    assert q_file.exists()
    assert {"item_id": "A_1.01", "slot_id": "s2"} in queue
    assert {"item_id": "A_1.01", "slot_id": "s1"} not in queue  # agreed slot not queued

    # 2. Write reviews covering all rules
    revs_c = [
        # A_1.01
        {"item_id": "A_1.01", "slot_id": "s2", "verdict": "equivalent", "reason": "Both capture the same physics"},
        {"item_id": "A_1.01", "slot_id": "s3", "verdict": "mine_wrong", "reason": "Claude missed a factor of 2"},
        {"item_id": "A_1.01", "slot_id": "s4", "verdict": "other_wrong", "reason": "Gemini used wrong time constant"},
        # A_1.02
        {"item_id": "A_1.02", "slot_id": "s1", "verdict": "other_wrong", "reason": "I think I am right"},
        {"item_id": "A_1.02", "slot_id": "s2", "verdict": "mine_wrong", "reason": "I made an error"},
        {"item_id": "A_1.02", "slot_id": "s3", "verdict": "both_wrong", "reason": "Neither is correct"},
        {"item_id": "A_1.02", "slot_id": "s4", "verdict": "unclear", "reason": "Ambiguous"},
        {"item_id": "A_1.02", "slot_id": "s5", "verdict": "equivalent", "reason": "Only claude reviewed"}
    ]
    revs_g = [
        # A_1.01
        {"item_id": "A_1.01", "slot_id": "s2", "verdict": "equivalent", "reason": "Both describe relaxation correctly"},
        {"item_id": "A_1.01", "slot_id": "s3", "verdict": "other_wrong", "reason": "Claude missed factor of 2"},
        {"item_id": "A_1.01", "slot_id": "s4", "verdict": "mine_wrong", "reason": "I used wrong time constant"},
        # A_1.02
        {"item_id": "A_1.02", "slot_id": "s1", "verdict": "other_wrong", "reason": "I think I am right"},
        {"item_id": "A_1.02", "slot_id": "s2", "verdict": "mine_wrong", "reason": "I also made an error"},
        {"item_id": "A_1.02", "slot_id": "s3", "verdict": "other_wrong", "reason": "Dispute"},
        {"item_id": "A_1.02", "slot_id": "s4", "verdict": "equivalent", "reason": "Dispute"},
        # s5 is omitted from gemini reviews (testing missing review)
    ]
    (reviews_dir / "claude.json").write_text(json.dumps(revs_c), encoding="utf-8")
    (reviews_dir / "gemini.json").write_text(json.dumps(revs_g), encoding="utf-8")

    # 3. Run resolve
    res_rep = resolve_chapter(1, data_dir=data_dir)
    assert (ch_dir / "resolution.json").exists()

    summary = res_rep["summary"]
    assert summary["usable_items"] == 1
    assert summary["excluded_items"] == 1
    assert res_rep["usable_item_ids"] == ["A_1.01"]
    assert res_rep["excluded_item_ids"] == ["A_1.02"]

    # Slot outcomes check
    outcomes = res_rep["slot_outcomes"]
    assert outcomes["agreed"] == 1
    assert outcomes["verified"] == 1
    assert outcomes["resolved_by_concession"] == 2
    assert outcomes["disputed"] == 5
    assert outcomes["total"] == 9

    # Check adopted answers for resolved_by_concession
    item1_slots = {s["slot_id"]: s for s in res_rep["items"]["A_1.01"]["slots"]}
    assert item1_slots["s3"]["outcome"] == "resolved_by_concession"
    assert item1_slots["s3"]["agreed_correct"] == "gemini"
    assert item1_slots["s3"]["adopted_answer"]["value"] == "200.0"

    assert item1_slots["s4"]["outcome"] == "resolved_by_concession"
    assert item1_slots["s4"]["agreed_correct"] == "claude"
    assert item1_slots["s4"]["adopted_answer"]["value"] == "2.0"

    # 4. Verify generate_report with resolution results
    rep_text = generate_report(1, data_dir=data_dir)
    assert "Items by usability:" in rep_text
    assert "usable: 1" in rep_text
    assert "excluded: 1" in rep_text
    assert "Slot outcomes by type:" in rep_text
    assert "agreed: 1" in rep_text
    assert "verified: 1" in rep_text
    assert "resolved_by_concession: 2" in rep_text
    assert "disputed: 5" in rep_text


# ---------------------------------------------------------------------------
# Bucket B and Concept Tooling Tests
# ---------------------------------------------------------------------------

def test_extract_concepts_synthetic(tmp_path: Path):
    """
    Tests concept extraction from lowest-level numbered sections:
    - 1.x.y lowest level
    - 1.x if no subsections
    - ignores sub-subsections like 1.x.y.z
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "chapters"
    ch_dir.mkdir(parents=True)

    fake_text = """# Chapter 1 Synthetic Physics
## Section 1.1 Spin Systems
### 1.1.1 Magnetic Moments
Text for 1.1.1.
### 1.1.2 Magnetization Vectors
Text for 1.1.2.
## 1.2 Isolated Phenomena
Text for 1.2 without subsections.
## 1.3 Detection Methods
### 1.3.1 Induction Coils
Text for 1.3.1.
### 1.3.1.1 Coil Geometry
Sub-sub section that should not be a concept.
"""
    (ch_dir / "chapter_01.md").write_text(fake_text, encoding="utf-8")

    concepts = extract_concepts_from_chapter(1, data_dir=data_dir)
    assert len(concepts) == 4
    assert [c["concept_id"] for c in concepts] == ["1.1.1", "1.1.2", "1.2", "1.3.1"]
    assert concepts[0]["title"] == "Magnetic Moments"
    assert concepts[1]["title"] == "Magnetization Vectors"
    assert concepts[2]["title"] == "Isolated Phenomena"
    assert concepts[3]["title"] == "Induction Coils"

    concepts_json = data_dir / "itembank" / "ch01" / "concepts.json"
    assert concepts_json.exists()
    saved = json.loads(concepts_json.read_text(encoding="utf-8"))
    assert saved == concepts


def test_b_item_criteria_validation():
    """
    Tests Bucket B criteria validation:
    - >= 2 distinct criteria from 1..6 pass
    - 1 criterion fails
    - duplicate criteria (<2 distinct) fail
    - out of range criterion fails
    - non-integer criterion fails
    - missing or empty justification fails
    """
    valid_item = {
        "item_id": "B_claude_01",
        "bucket": "B",
        "author_family": "claude",
        "author_model_id": "claude-3-7-sonnet",
        "statement_md": "Calculate the precession frequency of a spin system.",
        "anchor_item": "A_1.01",
        "criteria": [1, 3],
        "criteria_justification": {
            "1": "Tests transfer to a new field geometry.",
            "3": "Requires calculation of net precession."
        },
        "concepts": ["1.1.1"],
        "status": "drafted"
    }

    # Pass
    ok, errors = validate_item(valid_item, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert ok, f"Expected pass, got errors: {errors}"

    # Only 1 criterion
    bad_one = dict(valid_item, criteria=[1], criteria_justification={"1": "Only one"})
    ok, errors = validate_item(bad_one, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("at least 2 distinct criteria" in e for e in errors)

    # Duplicate criteria
    bad_dup = dict(valid_item, criteria=[2, 2], criteria_justification={"2": "Dup"})
    ok, errors = validate_item(bad_dup, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("at least 2 distinct criteria" in e for e in errors)

    # Out of range (7)
    bad_range = dict(valid_item, criteria=[1, 7], criteria_justification={"1": "A", "7": "B"})
    ok, errors = validate_item(bad_range, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("Invalid criterion 7" in e for e in errors)

    # Non-integer criterion
    bad_type = dict(valid_item, criteria=["1", "2"])
    ok, errors = validate_item(bad_type, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("must be integer in 1..6" in e for e in errors)

    # Missing justification
    bad_just = dict(valid_item, criteria=[1, 2], criteria_justification={"1": "Only 1"})
    ok, errors = validate_item(bad_just, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("Missing or empty criteria_justification for criterion 2" in e for e in errors)


def test_unknown_concept_fails():
    """
    Tests that any unknown concept ID in concepts tag fails validation.
    """
    valid_concepts = ["1.1.1", "1.1.2"]

    # Bucket B item with unknown concept
    item_b = {
        "item_id": "B_gemini_01",
        "bucket": "B",
        "author_family": "gemini",
        "author_model_id": "gemini-2.5-pro",
        "statement_md": "Calculate relaxation time.",
        "anchor_item": "A_1.01",
        "criteria": [1, 2],
        "criteria_justification": {"1": "reason 1", "2": "reason 2"},
        "concepts": ["1.1.1", "9.9.9"],
        "status": "drafted"
    }
    ok, errors = validate_item(item_b, usable_anchors={"A_1.01"}, concepts=valid_concepts)
    assert not ok
    assert any("Unknown concept ID: 9.9.9" in e for e in errors)

    # Bucket A item with unknown concept
    item_a = {
        "item_id": "A_1.01",
        "bucket": "A",
        "concepts": ["9.9.9"]
    }
    ok, errors = validate_item(item_a, concepts=valid_concepts)
    assert not ok
    assert any("Unknown concept ID: 9.9.9" in e for e in errors)

    # Bucket B empty concepts list fails
    item_b_empty = dict(item_b, concepts=[])
    ok, errors = validate_item(item_b_empty, usable_anchors={"A_1.01"}, concepts=valid_concepts)
    assert not ok
    assert any("non-empty list" in e for e in errors)


def test_non_usable_anchor_fails():
    """
    Tests that a Bucket B item referencing a non-usable anchor item fails validation.
    """
    item_b = {
        "item_id": "B_claude_01",
        "bucket": "B",
        "author_family": "claude",
        "author_model_id": "claude-3-7-sonnet",
        "statement_md": "Determine magnetic field.",
        "anchor_item": "A_1.99",
        "criteria": [1, 2],
        "criteria_justification": {"1": "reason 1", "2": "reason 2"},
        "concepts": ["1.1.1"],
        "status": "drafted"
    }

    # Anchor not in usable_anchors
    ok, errors = validate_item(item_b, usable_anchors={"A_1.01", "A_1.02"}, concepts=["1.1.1"])
    assert not ok
    assert any("not a usable Bucket A item in resolution.json" in e for e in errors)

    # Anchor not starting with A_
    bad_prefix = dict(item_b, anchor_item="B_gemini_01")
    ok, errors = validate_item(bad_prefix, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("must be a Bucket A item" in e for e in errors)


def test_option_list_detection():
    """
    Tests detection of option lists in statement_md:
    - which of the following
    - (A)...(B)...
    - (a)...(b)...
    """
    assert check_statement_option_lists("Which of the following describes Larmor precession?")
    assert check_statement_option_lists("Calculate B1: (A) 10 uT (B) 20 uT (C) 30 uT")
    assert check_statement_option_lists("Calculate B1:\n(a) 10 uT\n(b) 20 uT")
    assert not check_statement_option_lists("A spin-1/2 particle is in a 1.5 T field. Find the precession frequency.")

    # Validation integration
    item = {
        "item_id": "B_claude_01",
        "bucket": "B",
        "author_family": "claude",
        "author_model_id": "claude-3-7-sonnet",
        "statement_md": "Which of the following is correct regarding T1 relaxation?",
        "anchor_item": "A_1.01",
        "criteria": [1, 2],
        "criteria_justification": {"1": "r1", "2": "r2"},
        "concepts": ["1.1.1"],
        "status": "drafted"
    }
    ok, errors = validate_item(item, usable_anchors={"A_1.01"}, concepts=["1.1.1"])
    assert not ok
    assert any("option list or 'which of the following'" in e for e in errors)


def test_difficulty_screen_flags(tmp_path: Path):
    """
    Tests difficulty screen rules:
    - shorter solution_md in chars -> flagged
    - higher confidence (low < medium < high) -> flagged
    - longer solution_md and equal/lower confidence -> passes
    - flagged if EITHER family flags it
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    items_dir = ch_dir / "items"
    keys_c = ch_dir / "keys" / "claude"
    keys_g = ch_dir / "keys" / "gemini"
    items_dir.mkdir(parents=True)
    keys_c.mkdir(parents=True)
    keys_g.mkdir(parents=True)

    # Base anchor A key: length 200, confidence "medium"
    a_key_c = {"solution_md": "X" * 200, "confidence": "medium"}
    a_key_g = {"solution_md": "Y" * 200, "confidence": "medium"}
    (keys_c / "A_1.01.json").write_text(json.dumps(a_key_c), encoding="utf-8")
    (keys_g / "A_1.01.json").write_text(json.dumps(a_key_g), encoding="utf-8")

    # Item 1: non-author gemini key is shorter (100 chars < 200) -> flagged
    (items_dir / "B_claude_01.json").write_text(json.dumps({
        "item_id": "B_claude_01", "bucket": "B", "anchor_item": "A_1.01", "author_family": "claude"
    }), encoding="utf-8")
    (keys_c / "B_claude_01.json").write_text(json.dumps({
        "solution_md": "S" * 250, "confidence": "medium"
    }), encoding="utf-8")
    (keys_g / "B_claude_01.json").write_text(json.dumps({
        "solution_md": "S" * 100, "confidence": "medium"
    }), encoding="utf-8")

    # Item 2: non-author gemini key has higher confidence ("high" > "medium") -> flagged
    (items_dir / "B_claude_02.json").write_text(json.dumps({
        "item_id": "B_claude_02", "bucket": "B", "anchor_item": "A_1.01", "author_family": "claude"
    }), encoding="utf-8")
    (keys_c / "B_claude_02.json").write_text(json.dumps({
        "solution_md": "S" * 250, "confidence": "medium"
    }), encoding="utf-8")
    (keys_g / "B_claude_02.json").write_text(json.dumps({
        "solution_md": "S" * 250, "confidence": "high"
    }), encoding="utf-8")

    # Item 3: non-author gemini key is longer and lower confidence ("low" < "medium") -> PASSES
    (items_dir / "B_claude_03.json").write_text(json.dumps({
        "item_id": "B_claude_03", "bucket": "B", "anchor_item": "A_1.01", "author_family": "claude"
    }), encoding="utf-8")
    (keys_c / "B_claude_03.json").write_text(json.dumps({
        "solution_md": "S" * 100, "confidence": "high"  # author key is ignored
    }), encoding="utf-8")
    (keys_g / "B_claude_03.json").write_text(json.dumps({
        "solution_md": "S" * 250, "confidence": "low"
    }), encoding="utf-8")

    # Item 4: non-author claude key is longer and equal confidence ("medium" == "medium") -> PASSES
    (items_dir / "B_gemini_01.json").write_text(json.dumps({
        "item_id": "B_gemini_01", "bucket": "B", "anchor_item": "A_1.01", "author_family": "gemini"
    }), encoding="utf-8")
    (keys_c / "B_gemini_01.json").write_text(json.dumps({
        "solution_md": "S" * 250, "confidence": "medium"
    }), encoding="utf-8")
    (keys_g / "B_gemini_01.json").write_text(json.dumps({
        "solution_md": "S" * 100, "confidence": "high"  # author key is ignored
    }), encoding="utf-8")

    # Item 5: item with no non-author key -> UNSCREENED
    (items_dir / "B_gemini_02.json").write_text(json.dumps({
        "item_id": "B_gemini_02", "bucket": "B", "anchor_item": "A_1.01", "author_family": "gemini"
    }), encoding="utf-8")
    (keys_g / "B_gemini_02.json").write_text(json.dumps({
        "solution_md": "S" * 300, "confidence": "low"
    }), encoding="utf-8")

    rep = difficulty_screen_chapter(1, data_dir=data_dir)
    assert (ch_dir / "difficulty.json").exists()
    assert set(rep["flagged_item_ids"]) == {"B_claude_01", "B_claude_02"}
    assert set(rep["passed_item_ids"]) == {"B_claude_03", "B_gemini_01"}
    assert set(rep["unscreened_item_ids"]) == {"B_gemini_02"}
    assert rep["summary"]["flagged_items"] == 2
    assert rep["summary"]["passed_items"] == 2
    assert rep["summary"]["unscreened_items"] == 1


def test_select_b_coverage_and_threshold(tmp_path: Path):
    """
    Tests select-b:
    - Prioritizes covering every concept with at least one B item
    - Balances author families
    - Exits non-zero if <5 items selected
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    items_dir = ch_dir / "items"
    items_dir.mkdir(parents=True)

    concepts_data = [
        {"concept_id": "1.1.1", "title": "Concept 1"},
        {"concept_id": "1.1.2", "title": "Concept 2"},
        {"concept_id": "1.1.3", "title": "Concept 3"}
    ]
    (ch_dir / "concepts.json").write_text(json.dumps(concepts_data), encoding="utf-8")

    # Empty difficulty.json so no items flagged
    (ch_dir / "difficulty.json").write_text(json.dumps({"flagged_item_ids": []}), encoding="utf-8")

    # Subcase 1: Only 4 candidate items -> fails threshold (<5)
    for i in range(1, 5):
        (items_dir / f"B_claude_0{i}.json").write_text(json.dumps({
            "item_id": f"B_claude_0{i}",
            "bucket": "B",
            "author_family": "claude",
            "concepts": ["1.1.1"],
            "status": "usable"
        }), encoding="utf-8")

    res = select_b_chapter(1, target=5, data_dir=data_dir)
    assert res["selected_count"] == 4
    assert res["success"] is False
    with pytest.raises(SystemExit):
        select_b_chapter(1, target=5, data_dir=data_dir, exit_on_failure=True)

    # Subcase 2: Add more candidate items covering all concepts across families
    # Add Gemini and Claude items
    items_to_create = [
        ("B_claude_01", "claude", ["1.1.1"]),
        ("B_gemini_01", "gemini", ["1.1.2"]),
        ("B_claude_02", "claude", ["1.1.3"]),
        ("B_gemini_02", "gemini", ["1.1.1"]),
        ("B_claude_03", "claude", ["1.1.2"]),
        ("B_gemini_03", "gemini", ["1.1.3"])
    ]
    for iid, fam, tags in items_to_create:
        (items_dir / f"{iid}.json").write_text(json.dumps({
            "item_id": iid,
            "bucket": "B",
            "author_family": fam,
            "concepts": tags,
            "status": "usable"
        }), encoding="utf-8")

    res2 = select_b_chapter(1, target=5, data_dir=data_dir)
    assert res2["selected_count"] == 5
    assert res2["success"] is True
    assert set(res2["covered_concepts"]) == {"1.1.1", "1.1.2", "1.1.3"}
    assert res2["uncovered_with_candidate"] == []
    # Check balancing: 3 and 2
    f_counts = res2["family_counts"]
    assert sorted([f_counts["claude"], f_counts["gemini"]]) == [2, 3]

    # Test check_coverage_chapter
    cov = check_coverage_chapter(1, data_dir=data_dir)
    assert cov["success"] is True
    assert cov["uncovered_with_candidate"] == []


def test_weak_review_warning(tmp_path: Path):
    """
    Tests that generate_report emits WEAK-REVIEW warning when
    family review agreement rate > 0.95, and does not when <= 0.95.
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    rev_dir = ch_dir / "reviews"
    rev_dir.mkdir(parents=True)

    # Case A: 38 equivalent out of 39 (97.4% > 0.95)
    revs_c = [{"verdict": "equivalent"}] * 20
    revs_g = [{"verdict": "equivalent"}] * 18 + [{"verdict": "disputed"}]
    (rev_dir / "claude.json").write_text(json.dumps(revs_c), encoding="utf-8")
    (rev_dir / "gemini.json").write_text(json.dumps(revs_g), encoding="utf-8")

    rep_a = generate_report(1, data_dir=data_dir)
    assert "Family review agreement:" in rep_a
    assert "WARNING: WEAK-REVIEW" in rep_a
    assert "97.4%" in rep_a

    # Case B: 8 equivalent out of 10 (80.0% <= 0.95)
    revs_c2 = [{"verdict": "equivalent"}] * 4 + [{"verdict": "other_wrong"}]
    revs_g2 = [{"verdict": "equivalent"}] * 4 + [{"verdict": "mine_wrong"}]
    (rev_dir / "claude.json").write_text(json.dumps(revs_c2), encoding="utf-8")
    (rev_dir / "gemini.json").write_text(json.dumps(revs_g2), encoding="utf-8")

    rep_b = generate_report(1, data_dir=data_dir)
    assert "Family review agreement:" in rep_b
    assert "80.0%" in rep_b
    assert "WARNING: WEAK-REVIEW" not in rep_b


def test_validate_b_item_chapter_inference_and_cli(tmp_path: Path):
    """
    Tests that validate_item_file correctly infers chapter from folder path
    and accepts explicit chapter argument for B_<family>_NN items.
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    items_dir = ch_dir / "items"
    items_dir.mkdir(parents=True)

    # Concepts file
    (ch_dir / "concepts.json").write_text(json.dumps([{"concept_id": "1.1", "title": "Concept 1"}]), encoding="utf-8")
    # Resolution file with usable anchors
    (ch_dir / "resolution.json").write_text(json.dumps({"usable_item_ids": ["A_1.01"]}), encoding="utf-8")

    item_data = {
        "item_id": "B_gemini_01",
        "bucket": "B",
        "author_family": "gemini",
        "author_model_id": "gemini-3.8-flash",
        "statement_md": "Calculate the precession frequency in a magnetic field.",
        "anchor_item": "A_1.01",
        "criteria": [1, 2],
        "criteria_justification": {"1": "criterion 1", "2": "criterion 2"},
        "concepts": ["1.1"],
        "status": "drafted"
    }
    item_file = items_dir / "B_gemini_01.json"
    item_file.write_text(json.dumps(item_data), encoding="utf-8")

    # 1. Validate with explicit chapter
    val, errs = validate_item_file(item_file, chapter=1, data_dir=data_dir)
    assert val is True, f"Errors: {errs}"

    # 2. Validate without explicit chapter (infer from folder 'ch01')
    val_inf, errs_inf = validate_item_file(item_file, data_dir=data_dir)
    assert val_inf is True, f"Inferred errors: {errs_inf}"


def test_key_solver_block_schema(tmp_path: Path):
    """
    Tests solver block requirements:
    - Required for any key on an item the family did not author
    - Not required for key on item authored by the same family
    - Validates well_posed (bool), needs_external_info (bool), issue (str)
    """
    # 1. Authored item (gemini key for B_gemini_01): no solver block required
    own_key = {
        "item_id": "B_gemini_01", "model_family": "gemini", "model_id": "gemini-3.8-flash",
        "solved_at": "2026-10-22T10:00:00Z", "solution_md": "...", "final_answers": [],
        "assumptions": [], "confidence": "high"
    }
    v_own, e_own = validate_key(own_key)
    assert v_own is True, f"Errors: {e_own}"

    # 2. Non-authored item (gemini key for B_claude_01): solver block required
    non_auth_key = {
        "item_id": "B_claude_01", "model_family": "gemini", "model_id": "gemini-3.8-flash",
        "solved_at": "2026-10-22T10:00:00Z", "solution_md": "...", "final_answers": [],
        "assumptions": [], "confidence": "high"
    }
    v_na, e_na = validate_key(non_auth_key)
    assert v_na is False
    assert any("Missing required 'solver' block" in e for e in e_na)

    # 3. Non-authored item with valid solver block: passes
    valid_solver_key = dict(non_auth_key)
    valid_solver_key["solver"] = {
        "well_posed": True,
        "needs_external_info": False,
        "issue": ""
    }
    v_ok, e_ok = validate_key(valid_solver_key)
    assert v_ok is True, f"Errors: {e_ok}"

    # 4. Invalid solver block fields
    bad_solver_key = dict(non_auth_key)
    bad_solver_key["solver"] = {
        "well_posed": "not_a_bool",
        "needs_external_info": False,
        "issue": ""
    }
    v_bad, e_bad = validate_key(bad_solver_key)
    assert v_bad is False
    assert any("'solver.well_posed' must be a boolean" in e for e in e_bad)


def test_difficulty_screen_unscreened(tmp_path: Path):
    """
    Tests that items with no non-author key are marked 'unscreened' and never 'pass'.
    """
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "itembank" / "ch01"
    items_dir = ch_dir / "items"
    keys_g = ch_dir / "keys" / "gemini"
    items_dir.mkdir(parents=True)
    keys_g.mkdir(parents=True)

    # Base anchor A key for gemini
    (keys_g / "A_1.01.json").write_text(json.dumps({"solution_md": "Y" * 200, "confidence": "medium"}), encoding="utf-8")

    # B item authored by claude, but NO gemini key exists
    (items_dir / "B_claude_01.json").write_text(json.dumps({
        "item_id": "B_claude_01", "bucket": "B", "anchor_item": "A_1.01", "author_family": "claude"
    }), encoding="utf-8")

    rep = difficulty_screen_chapter(1, data_dir=data_dir)
    assert rep["summary"]["unscreened_items"] == 1
    assert "B_claude_01" in rep["unscreened_item_ids"]
    assert "B_claude_01" not in rep["passed_item_ids"]
    assert "B_claude_01" not in rep["flagged_item_ids"]
    assert rep["items"]["B_claude_01"]["status"] == "unscreened"

