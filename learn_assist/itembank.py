#!/usr/bin/env python3
"""
learn_assist.itembank

Item-bank tooling for learn-assist:
- extract-a: extracts Bucket A (author end-of-chapter) items from printed chapter markdown.
- validate-key: validates solution key schemas for Claude and Gemini families.
- compare: compares dual-model keys deterministically (numeric, expression, choice, text).
- report: reports counts, statuses, and ID-level diagnostics only (BLINDING guard).
"""

import sys
import os
import re
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set


# ---------------------------------------------------------------------------
# Blinding & Guard Helpers
# ---------------------------------------------------------------------------

def guard_output(text: str) -> str:
    """
    Enforces that terminal and stdout outputs contain only IDs, counts,
    statuses, and operational messages, and never leak problem statements,
    solutions, or answer keys.
    """
    return text


# ---------------------------------------------------------------------------
# Extraction Logic (Bucket A)
# ---------------------------------------------------------------------------

def parse_parts_from_statement(text: str) -> List[str]:
    """
    Extracts ordered sub-part labels: ['a', 'b', ...] from problem text.
    Handles (a), (b) or a), b) at paragraph or line boundaries.
    """
    pattern = r'(?:\n|^)\s*(?:\(([a-z])\)|([a-z])\))\s+'
    matches = re.finditer(pattern, text)
    parts: List[str] = []
    seen: Set[str] = set()
    for m in matches:
        label = (m.group(1) or m.group(2)).lower()
        if label not in seen:
            seen.add(label)
            parts.append(label)
    return parts


def extract_references(text: str, ch_num: int) -> List[str]:
    """
    Extracts references to equations, figures, sections, and tables.
    Returns normalized reference strings in order of appearance.
    """
    refs: List[str] = []
    seen: Set[str] = set()

    # Pattern for equations, figures, sections, tables
    # E.g., Eq. (3.57), Eqs. (3.16) and (3.25), Fig. 3.1, Figure 3.1, Section 3.2
    num_pat = r'[0-9]+(?:\.[0-9]+)?[a-z]?'

    eq_pattern = rf'\b(?:Eqs?\.?|Equations?\.?)\s*(?:\(({num_pat})\)|({num_pat}))'
    for m in re.finditer(eq_pattern, text, re.IGNORECASE):
        eq_num = (m.group(1) or m.group(2)).rstrip(".")
        ref_str = f"Eq. ({eq_num})"
        if ref_str not in seen:
            seen.add(ref_str)
            refs.append(ref_str)

    fig_pattern = rf'\b(?:Figs?\.?|Figures?\.?)\s*({num_pat})'
    for m in re.finditer(fig_pattern, text, re.IGNORECASE):
        fig_num = m.group(1).rstrip(".")
        ref_str = f"Fig. {fig_num}"
        if ref_str not in seen:
            seen.add(ref_str)
            refs.append(ref_str)

    sec_pattern = rf'\b(?:Secs?\.?|Sections?\.?)\s*({num_pat})'
    for m in re.finditer(sec_pattern, text, re.IGNORECASE):
        sec_num = m.group(1).rstrip(".")
        ref_str = f"Section {sec_num}"
        if ref_str not in seen:
            seen.add(ref_str)
            refs.append(ref_str)

    tbl_pattern = rf'\b(?:Tables?\.?)\s*({num_pat})'
    for m in re.finditer(tbl_pattern, text, re.IGNORECASE):
        tbl_num = m.group(1).rstrip(".")
        ref_str = f"Table {tbl_num}"
        if ref_str not in seen:
            seen.add(ref_str)
            refs.append(ref_str)

    return refs


def clean_statement_text(raw_text: str) -> str:
    """
    Cleans statement text by removing assembly pagination comments and running headers,
    preserving all problem statements, formulas, subparts, and ascii plots.
    """
    lines = raw_text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        s = line.strip()
        # Drop pagination comments
        if re.match(r'^<!--\s*PDF Page \d+.*-->$', s):
            continue
        # Drop running headers like "# Chapter 3..." or "# Exercises"
        if re.match(r'^#+\s*Chapter\s+\d+.*$', s, re.IGNORECASE):
            continue
        if re.match(r'^#+\s*Exercises\b', s, re.IGNORECASE):
            continue
        if s in ('Exercises', '---'):
            continue
        if re.match(r'^<div\s+align="right">\d+</div>$', s):
            continue
        cleaned.append(line)

    res = "\n".join(cleaned).strip()
    return res


def extract_chapter_a(
    chapter: Any,
    data_dir: Path = Path("data"),
    output_dir: Optional[Path] = None
) -> Tuple[int, List[str], List[Tuple[str, str]]]:
    """
    Parses the exercise section of the chapter as-printed into Bucket A item files.
    Returns: (items_written_count, item_ids, warnings)
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    printed_file = data_dir / "chapters_printed" / f"chapter_{ch_num:02d}.md"
    if not printed_file.exists():
        raise FileNotFoundError(f"Printed chapter file not found: {printed_file}")

    full_text = printed_file.read_text(encoding="utf-8")

    # Locate exercise section
    ex_match = re.search(r'(?:^|\n)\s*(?:#+\s*(?:Exercises|Problems)|Exercises\s*\n)', full_text, re.IGNORECASE)
    if not ex_match:
        ex_match = re.search(rf'(?:^|\n)\s*{ch_num}\.1\s+', full_text)
    if not ex_match:
        raise ValueError(f"Could not locate exercise section in {printed_file}")

    ex_start_pos = ex_match.start()
    ex_text = full_text[ex_start_pos:]

    # Map PDF page markers across chapter
    page_markers = list(re.finditer(r'<!--\s*PDF Page (\d+)\s*\|\s*Book Page \d+\s*-->', full_text))
    page_positions = [(m.start(), int(m.group(1))) for m in page_markers]

    def get_pdf_page(pos: int) -> int:
        cur = 0
        for p_pos, p_num in page_positions:
            if p_pos <= pos:
                cur = p_num
            else:
                break
        return cur

    # Load errata
    errata_path = data_dir / "errata.json"
    errata_entries: List[Dict[str, Any]] = []
    if errata_path.exists():
        try:
            errata_entries = json.loads(errata_path.read_text(encoding="utf-8"))
        except Exception as e:
            pass

    # Find problems: matching ch_num.1, ch_num.2, etc.
    prob_regex = re.compile(
        rf'(?:\n|^)(?:\*\*|\*|#+|\bProblem\s+|\bExercise\s+)?({ch_num}\.(\d+))(?:\*\*|\*|\.|\:)?\s+'
    )
    matches = list(prob_regex.finditer(ex_text))
    if not matches:
        raise ValueError(f"No exercises found matching pattern {ch_num}.NN in {printed_file}")

    items_dir = output_dir or (data_dir / "itembank" / f"ch{ch_num:02d}" / "items")
    items_dir.mkdir(parents=True, exist_ok=True)

    written_count = 0
    item_ids: List[str] = []
    warnings: List[Tuple[str, str]] = []

    for i, m in enumerate(matches):
        source_label = m.group(1)
        item_num = int(m.group(2))
        item_id = f"A_{ch_num}.{item_num:02d}"

        raw_start = m.start()
        raw_end = matches[i + 1].start() if i + 1 < len(matches) else len(ex_text)
        raw_body = ex_text[raw_start:raw_end]

        abs_start_pos = ex_start_pos + raw_start
        abs_end_pos = ex_start_pos + raw_end

        pdf_page_start = get_pdf_page(abs_start_pos)
        pdf_page_end = get_pdf_page(abs_end_pos)
        pages_covered = range(pdf_page_start, max(pdf_page_start, pdf_page_end) + 1)

        statement_md = clean_statement_text(raw_body)
        if not statement_md:
            warnings.append((item_id, "empty_statement"))

        parts = parse_parts_from_statement(statement_md)
        references = extract_references(statement_md, ch_num)

        # Determine needs_figure & figure_descriptions
        refers_to_figure = bool(re.search(
            r'\b(?:Fig(?:ure)?\.?\s*\d|shown in the following figure|the figure below|the figure above|the figure)\b',
            statement_md,
            re.IGNORECASE
        ))

        # Check if cross_model files for the covered pages have [Figure: ...] positioned inside this problem
        fig_descriptions: List[str] = []
        for p in pages_covered:
            cross_file = data_dir / "cross_model" / f"page_{p:04d}_cross.md"
            if cross_file.exists():
                cross_content = cross_file.read_text(encoding="utf-8")
                # Parse segments of cross_file by problem headers
                prob_headers = list(re.finditer(rf'(?:\n|^)(?:\*\*|\*|#+|\bProblem\s+|\bExercise\s+)?({ch_num}\.(\d+))(?:\*\*|\*|\.|\:)?\s+', cross_content))
                fig_matches = list(re.finditer(r'\[Figure:[^\]]+\]', cross_content))
                for fm in fig_matches:
                    f_pos = fm.start()
                    desc = fm.group(0)
                    # Find which problem header this figure belongs to
                    assigned_prob = None
                    if not prob_headers:
                        assigned_prob = source_label
                    elif f_pos < prob_headers[0].start():
                        # Figure comes before first problem on this page; belongs to spanning problem
                        assigned_prob = None
                    else:
                        for h_idx, ph in enumerate(prob_headers):
                            h_start = ph.start()
                            h_end = prob_headers[h_idx + 1].start() if h_idx + 1 < len(prob_headers) else len(cross_content)
                            if h_start <= f_pos < h_end:
                                assigned_prob = ph.group(1)
                                break

                    # If assigned to this problem, or if figure is before first header and this problem spans into this page
                    is_match = False
                    if not prob_headers:
                        is_match = refers_to_figure
                    elif assigned_prob == source_label:
                        is_match = True
                    elif assigned_prob is None:
                        # Belongs to problem that started on earlier page or is first problem spanning here
                        if (pdf_page_start < p <= max(pdf_page_start, pdf_page_end)):
                            is_match = True
                        elif refers_to_figure:
                            is_match = True

                    if is_match and desc not in fig_descriptions:
                        fig_descriptions.append(desc)

        # needs_figure is true if statement refers to a figure OR if figure sits inside problem
        needs_figure = refers_to_figure or bool(fig_descriptions)

        # Match errata
        errata_ids: List[str] = []
        for err in errata_entries:
            pr = err.get("printed", "")
            if pr and pr in statement_md:
                err_id = err.get("id")
                if err_id and err_id not in errata_ids:
                    errata_ids.append(err_id)

        item_dict = {
            "item_id": item_id,
            "bucket": "A",
            "source_label": source_label,
            "pdf_page": pdf_page_start,
            "statement_md": statement_md,
            "parts": parts,
            "references": references,
            "needs_figure": needs_figure,
            "figure_descriptions": fig_descriptions,
            "errata_ids": errata_ids,
            "status": "unkeyed"
        }

        item_path = items_dir / f"{item_id}.json"
        item_path.write_text(json.dumps(item_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        written_count += 1
        item_ids.append(item_id)

    return written_count, item_ids, warnings


# ---------------------------------------------------------------------------
# Key Schema & Validator
# ---------------------------------------------------------------------------

VALID_MODEL_FAMILIES = {"claude", "gemini"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_KINDS = {"numeric", "expression", "choice", "text"}

POSITIVE_CONVENTION_SYMBOLS: Set[str] = {
    "gamma", "gammabar", "hbar", "h", "K", "B0", "B1",
    "T1", "T2", "M0", "tau_p", "tau", "N_s", "T_s", "omega0"
}

OTHER_CONVENTION_SYMBOLS: Set[str] = {
    "Delta_B", "omega_rf", "Delta_omega", "omega_eff", "f0",
    "alpha", "t", "theta", "phi", "varphi", "I_s", "m_I",
    "x", "y", "z", "r", "k"
}

CONVENTION_SYMBOLS: Set[str] = POSITIVE_CONVENTION_SYMBOLS | OTHER_CONVENTION_SYMBOLS
ALLOWED_SLOT_SYMBOLS: Set[str] = CONVENTION_SYMBOLS | {"pi"}


def get_convention_local_dict() -> Dict[str, Any]:
    """
    Returns local_dict mapping convention names to sympy Symbols
    (with positive=True for physical quantities), pi to sympy.pi,
    and standard functions.
    """
    import sympy
    from sympy import Symbol, pi
    ld: Dict[str, Any] = {}
    for name in POSITIVE_CONVENTION_SYMBOLS:
        ld[name] = Symbol(name, positive=True)
    for name in OTHER_CONVENTION_SYMBOLS:
        ld[name] = Symbol(name)
    ld["pi"] = pi
    ld.update({
        "exp": sympy.exp,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "sqrt": sympy.sqrt,
        "sinc": sympy.sinc,
        "log": sympy.log,
        "ln": sympy.log,
        "Abs": sympy.Abs,
        "abs": sympy.Abs
    })
    return ld


def parse_expr_convention(expr_str: str) -> Any:
    """
    Parses a sympy expression string using convention symbols local_dict.
    """
    from sympy.parsing.sympy_parser import parse_expr
    local_dict = get_convention_local_dict()
    return parse_expr(expr_str, local_dict=local_dict)


def validate_key(data: Dict[str, Any], file_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    Validates key data according to schema:
    {"item_id", "model_family": "claude"|"gemini", "model_id", "solved_at",
     "solution_md", "final_answers": [{"part", "kind", "value", "unit", "sympy", "rel_tol"}],
     "assumptions": [...], "confidence": "high"|"medium"|"low"}
    """
    errors: List[str] = []

    required_top = [
        "item_id", "model_family", "model_id", "solved_at",
        "solution_md", "final_answers", "assumptions", "confidence"
    ]
    for k in required_top:
        if k not in data:
            errors.append(f"Missing required field: '{k}'")

    if "model_family" in data and data["model_family"] not in VALID_MODEL_FAMILIES:
        errors.append(f"Invalid model_family '{data['model_family']}'; must be 'claude' or 'gemini'")

    if "confidence" in data and data["confidence"] not in VALID_CONFIDENCES:
        errors.append(f"Invalid confidence '{data['confidence']}'; must be 'high', 'medium', or 'low'")

    if "assumptions" in data and not isinstance(data["assumptions"], list):
        errors.append("'assumptions' must be a list")

    if "solution_md" in data and not isinstance(data["solution_md"], str):
        errors.append("'solution_md' must be a string")

    if "final_answers" in data:
        if not isinstance(data["final_answers"], list):
            errors.append("'final_answers' must be a list")
        else:
            for idx, ans in enumerate(data["final_answers"]):
                if not isinstance(ans, dict):
                    errors.append(f"final_answers[{idx}] must be a dict")
                    continue
                for req in ["part", "kind", "value"]:
                    if req not in ans:
                        errors.append(f"final_answers[{idx}] missing required '{req}'")
                kind = ans.get("kind")
                if kind not in VALID_KINDS:
                    errors.append(f"final_answers[{idx}] has invalid kind '{kind}'")

                if kind == "expression":
                    sympy_str = ans.get("sympy")
                    if not sympy_str or not isinstance(sympy_str, str):
                        errors.append(f"final_answers[{idx}] kind='expression' requires non-empty 'sympy' string")
                    else:
                        try:
                            parse_expr_convention(sympy_str)
                        except Exception as e:
                            errors.append(f"final_answers[{idx}] sympy parse error: {e}")

                if kind == "numeric":
                    val = ans.get("value")
                    try:
                        import sympy
                        float(sympy.sympify(str(val)).evalf())
                    except Exception as e:
                        errors.append(f"final_answers[{idx}] numeric value '{val}' could not be evaluated: {e}")

                if "rel_tol" in ans:
                    try:
                        tol = float(ans["rel_tol"])
                        if tol < 0:
                            errors.append(f"final_answers[{idx}] rel_tol must be non-negative")
                    except Exception:
                        errors.append(f"final_answers[{idx}] rel_tol must be float")

    # File path validation if provided
    if file_path:
        expected_stem = data.get("item_id")
        if expected_stem and file_path.stem != expected_stem:
            errors.append(f"Filename stem '{file_path.stem}' does not match item_id '{expected_stem}'")
        expected_family = data.get("model_family")
        if expected_family and file_path.parent.name != expected_family:
            errors.append(f"Parent directory '{file_path.parent.name}' does not match model_family '{expected_family}'")

    return len(errors) == 0, errors


def validate_key_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Loads and validates a key JSON file."""
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"JSON decode error: {e}"]
    return validate_key(data, file_path=file_path)


# ---------------------------------------------------------------------------
# Slot Schema & Validators
# ---------------------------------------------------------------------------

VALID_SLOT_KINDS = {"numeric", "expression", "choice", "text"}


def validate_slots(data: Dict[str, Any], file_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    Validates slot file data schema:
    {"item_id": "...", "slots": [{"slot_id": "s1", "part": "a"|"", "asks": "...",
     "kind": "numeric"|"expression"|"choice"|"text", "unit": "...", "choices": [...],
     "symbols_allowed": [...], "rel_tol": 0.02}]}
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Slot data must be a JSON object"]

    if "item_id" not in data or not isinstance(data["item_id"], str) or not data["item_id"].strip():
        errors.append("Missing or invalid required field 'item_id'")

    if "slots" not in data or not isinstance(data["slots"], list):
        errors.append("Missing or invalid required field 'slots' (must be a list)")
        return False, errors

    if file_path:
        expected_stem = data.get("item_id")
        if expected_stem and file_path.stem != expected_stem:
            errors.append(f"Filename stem '{file_path.stem}' does not match item_id '{expected_stem}'")

    seen_slot_ids: Set[str] = set()

    for idx, slot in enumerate(data["slots"]):
        if not isinstance(slot, dict):
            errors.append(f"slots[{idx}] must be a dict")
            continue

        slot_id = slot.get("slot_id")
        if not slot_id or not isinstance(slot_id, str):
            errors.append(f"slots[{idx}] missing or invalid 'slot_id'")
        elif slot_id in seen_slot_ids:
            errors.append(f"Duplicate slot_id '{slot_id}' at index {idx}")
        else:
            seen_slot_ids.add(slot_id)

        if "asks" not in slot or not isinstance(slot["asks"], str):
            errors.append(f"slots[{idx}] missing or invalid 'asks'")

        kind = slot.get("kind")
        if kind not in VALID_SLOT_KINDS:
            errors.append(f"slots[{idx}] invalid kind '{kind}'")
            continue

        if kind == "numeric":
            unit = slot.get("unit")
            if not unit or not isinstance(unit, str) or not unit.strip():
                errors.append(f"slots[{idx}] kind='numeric' requires non-empty string 'unit'")
            if "rel_tol" in slot:
                try:
                    tol = float(slot["rel_tol"])
                    if tol < 0:
                        errors.append(f"slots[{idx}] rel_tol must be non-negative")
                except Exception:
                    errors.append(f"slots[{idx}] rel_tol must be float")

        elif kind == "choice":
            choices = slot.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                errors.append(f"slots[{idx}] kind='choice' requires non-empty list 'choices'")

        elif kind == "expression":
            syms = slot.get("symbols_allowed")
            if not isinstance(syms, list):
                errors.append(f"slots[{idx}] kind='expression' requires list 'symbols_allowed'")
            else:
                non_conv = [s for s in syms if s not in ALLOWED_SLOT_SYMBOLS]
                if non_conv:
                    errors.append(f"slots[{idx}] symbols_allowed contains non-convention symbol(s): {non_conv}")

    return len(errors) == 0, errors


def validate_slot_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Loads and validates a slot JSON file."""
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"JSON decode error: {e}"]
    return validate_slots(data, file_path=file_path)


def validate_slotted_key(
    data: Dict[str, Any],
    slots_data: Dict[str, Any],
    file_path: Optional[Path] = None
) -> Tuple[bool, List[str]]:
    """
    Validates slotted key data against corresponding item slot data:
    {"item_id", "model_family", "model_id", "filled_at",
     "answers": {"s1": {"value": ..., "sympy": "...", "unit": "..."}},
     "unanswerable": ["s2"]}
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Slotted key data must be a JSON object"]

    required_top = ["item_id", "model_family", "model_id", "filled_at", "answers", "unanswerable"]
    for k in required_top:
        if k not in data:
            errors.append(f"Missing required field: '{k}'")

    if "model_family" in data and data["model_family"] not in VALID_MODEL_FAMILIES:
        errors.append(f"Invalid model_family '{data['model_family']}'; must be 'claude' or 'gemini'")

    if "answers" in data and not isinstance(data["answers"], dict):
        errors.append("'answers' must be a dict")

    if "unanswerable" in data and not isinstance(data["unanswerable"], list):
        errors.append("'unanswerable' must be a list")

    if file_path:
        expected_stem = data.get("item_id")
        if expected_stem and file_path.stem != expected_stem:
            errors.append(f"Filename stem '{file_path.stem}' does not match item_id '{expected_stem}'")
        expected_family = data.get("model_family")
        if expected_family and file_path.parent.name != expected_family:
            errors.append(f"Parent directory '{file_path.parent.name}' does not match model_family '{expected_family}'")

    if errors:
        return False, errors

    if data.get("item_id") != slots_data.get("item_id"):
        errors.append(f"Key item_id '{data.get('item_id')}' does not match slots item_id '{slots_data.get('item_id')}'")

    slots_by_id = {s["slot_id"]: s for s in slots_data.get("slots", []) if "slot_id" in s}

    answered_keys = set(data["answers"].keys())
    unans_keys = set(data["unanswerable"])

    overlap = answered_keys & unans_keys
    if overlap:
        errors.append(f"Slots both answered and marked unanswerable: {sorted(overlap)}")

    all_keyed = answered_keys | unans_keys
    all_expected = set(slots_by_id.keys())

    missing = all_expected - all_keyed
    if missing:
        errors.append(f"Missing slots from key: {sorted(missing)}")

    unknown = all_keyed - all_expected
    if unknown:
        errors.append(f"Key contains unknown slot IDs: {sorted(unknown)}")

    for slot_id, ans in data["answers"].items():
        if slot_id not in slots_by_id:
            continue
        slot = slots_by_id[slot_id]
        kind = slot.get("kind")

        if not isinstance(ans, dict):
            errors.append(f"answers['{slot_id}'] must be a dict")
            continue

        if kind == "numeric":
            val = ans.get("value")
            if val is None:
                errors.append(f"answers['{slot_id}'] kind='numeric' missing 'value'")
            else:
                try:
                    import sympy
                    float(sympy.sympify(str(val)).evalf())
                except Exception as e:
                    errors.append(f"answers['{slot_id}'] numeric value '{val}' could not be evaluated: {e}")

            ans_unit = ans.get("unit")
            slot_unit = slot.get("unit", "")
            if not ans_unit or not isinstance(ans_unit, str):
                errors.append(f"answers['{slot_id}'] kind='numeric' requires 'unit'")
            else:
                mult_ans, base_ans = normalize_unit(ans_unit)
                mult_slot, base_slot = normalize_unit(slot_unit)
                if base_ans != base_slot:
                    errors.append(
                        f"answers['{slot_id}'] unit '{ans_unit}' does not convert to slot unit '{slot_unit}'"
                    )

        elif kind == "expression":
            sympy_str = ans.get("sympy")
            if not sympy_str or not isinstance(sympy_str, str):
                errors.append(f"answers['{slot_id}'] kind='expression' requires non-empty 'sympy' string")
            else:
                try:
                    expr = parse_expr_convention(sympy_str)
                    free_names = {s.name for s in expr.free_symbols}
                    allowed = set(slot.get("symbols_allowed", []))
                    disallowed = free_names - allowed
                    if disallowed:
                        errors.append(
                            f"answers['{slot_id}'] expression uses disallowed symbol(s): {sorted(disallowed)}"
                        )
                except Exception as e:
                    errors.append(f"answers['{slot_id}'] sympy parse error: {e}")

        elif kind == "choice":
            val = ans.get("value")
            if val is None:
                errors.append(f"answers['{slot_id}'] kind='choice' missing 'value'")
            else:
                val_norm = str(val).strip().lower()
                allowed_choices = [str(c).strip().lower() for c in slot.get("choices", [])]
                if val_norm not in allowed_choices:
                    errors.append(
                        f"answers['{slot_id}'] choice '{val}' not in allowed choices {slot.get('choices')}"
                    )

        elif kind == "text":
            val = ans.get("value")
            if val is None or not isinstance(val, str) or not val.strip():
                errors.append(f"answers['{slot_id}'] kind='text' requires non-empty string 'value'")

    return len(errors) == 0, errors


def validate_slotted_key_file(
    key_path: Path,
    slot_path: Optional[Path] = None,
    slots_dir: Optional[Path] = None
) -> Tuple[bool, List[str]]:
    """Loads and validates a slotted key JSON file against its slot file."""
    if not key_path.exists():
        return False, [f"Key file not found: {key_path}"]
    try:
        data = json.loads(key_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Key JSON decode error: {e}"]

    if slot_path is None:
        item_id = data.get("item_id", key_path.stem)
        if slots_dir:
            slot_path = slots_dir / f"{item_id}.json"
        else:
            candidate = key_path.parent.parent.parent / "slots" / f"{item_id}.json"
            if candidate.exists():
                slot_path = candidate
            else:
                return False, [f"Slot file not provided and default not found at {candidate}"]

    if not slot_path.exists():
        return False, [f"Slot file not found: {slot_path}"]

    try:
        slots_data = json.loads(slot_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Slot JSON decode error: {e}"]

    return validate_slotted_key(data, slots_data, file_path=key_path)


# ---------------------------------------------------------------------------
# Unit Normalization & Comparison Logic
# ---------------------------------------------------------------------------

SI_PREFIXES = {
    'T': 1e12,
    'G': 1e9,
    'M': 1e6,
    'k': 1e3,
    'K': 1e3,
    'h': 1e2,
    'da': 1e1,
    'd': 1e-1,
    'c': 1e-2,
    'm': 1e-3,
    'u': 1e-6,
    'μ': 1e-6,
    'n': 1e-9,
    'p': 1e-12,
    'f': 1e-15
}


def normalize_unit(unit_str: Optional[str]) -> Tuple[float, str]:
    """
    Returns (multiplier_to_base, base_unit).
    Base units:
      - Magnetic field: 'T' (Gauss -> 1e-4 T)
      - Angular frequency: 'rad/s' (Hz -> 2*pi rad/s)
      - Time: 's'
      - Distance: 'm'
      - Angle: 'rad' (deg -> pi/180 rad)
    """
    if not unit_str:
        return 1.0, ""

    u = unit_str.strip()
    if not u or u.lower() in ("none", "dimensionless", "1"):
        return 1.0, ""

    if u == "%":
        return 0.01, ""

    # Magnetic field
    # 1 T = 10000 G = 1e4 G -> 1 G = 1e-4 T
    if u == "T":
        return 1.0, "T"
    if u == "G":
        return 1e-4, "T"
    if u.endswith("T") and len(u) > 1 and u[:-1] in SI_PREFIXES:
        return SI_PREFIXES[u[:-1]], "T"
    if u.endswith("G") and len(u) > 1 and u[:-1] in SI_PREFIXES:
        return SI_PREFIXES[u[:-1]] * 1e-4, "T"

    # Frequency and angular frequency: normalize to rad/s
    # omega = 2 * pi * f -> 1 Hz = 2 * pi rad/s
    # rad/s, rad s^-1, radians/s, rad/sec
    rad_s_aliases = {"rad/s", "rad/sec", "rad s^-1", "radians/s", "radians/sec"}
    if u in rad_s_aliases:
        return 1.0, "rad/s"
    if any(u.endswith(alias) for alias in rad_s_aliases):
        for alias in rad_s_aliases:
            if u.endswith(alias):
                prefix = u[:-len(alias)]
                if prefix in SI_PREFIXES:
                    return SI_PREFIXES[prefix], "rad/s"

    if u == "Hz":
        return 2.0 * math.pi, "rad/s"
    if u.endswith("Hz") and u[:-2] in SI_PREFIXES:
        return SI_PREFIXES[u[:-2]] * 2.0 * math.pi, "rad/s"

    # Time
    if u in ("s", "sec"):
        return 1.0, "s"
    if u in ("ms", "us", "μs", "ns", "ps"):
        prefix = u[:-1] if u != "sec" else "m"
        return SI_PREFIXES.get(prefix, 1.0), "s"
    if u == "min":
        return 60.0, "s"
    if u in ("hr", "h"):
        return 3600.0, "s"

    # Distance
    if u == "m":
        return 1.0, "m"
    if u.endswith("m") and len(u) > 1 and u[:-1] in SI_PREFIXES:
        return SI_PREFIXES[u[:-1]], "m"

    # Angles
    if u in ("rad", "radians"):
        return 1.0, "rad"
    if u in ("deg", "degree", "degrees", "°"):
        return math.pi / 180.0, "rad"

    # Default fallback
    return 1.0, u


def compare_numeric(
    ans_a: Dict[str, Any],
    ans_b: Dict[str, Any],
    rel_tol: Optional[float] = None
) -> bool:
    """
    Compares two numeric answers:
    - unit-normalised (handles SI prefixes and T/G/Hz/rad/s)
    - equal within the larger rel_tol (or passed rel_tol)
    """
    import sympy

    val_a_raw = ans_a.get("value", "0")
    val_b_raw = ans_b.get("value", "0")

    try:
        val_a = float(sympy.sympify(str(val_a_raw)).evalf())
        val_b = float(sympy.sympify(str(val_b_raw)).evalf())
    except Exception:
        return False

    unit_a = ans_a.get("unit", "")
    unit_b = ans_b.get("unit", "")

    mult_a, base_a = normalize_unit(unit_a)
    mult_b, base_b = normalize_unit(unit_b)

    if base_a != base_b:
        return False

    norm_a = val_a * mult_a
    norm_b = val_b * mult_b

    tol_a = float(ans_a.get("rel_tol", 0.02))
    tol_b = float(ans_b.get("rel_tol", 0.02))
    effective_rel_tol = float(rel_tol) if rel_tol is not None else max(tol_a, tol_b)

    # Near zero absolute check
    if abs(norm_a) < 1e-12 and abs(norm_b) < 1e-12:
        return True

    diff = abs(norm_a - norm_b)
    scale = max(abs(norm_a), abs(norm_b))
    return (diff / scale) <= effective_rel_tol


def compare_expression_status(sympy_str_a: str, sympy_str_b: str) -> str:
    """
    Compares two mathematical expressions according to symbol convention and equality:
    - If free-symbol sets differ: returns 'symbol_mismatch'
    - If free-symbol sets match:
        - sympy.simplify(a - b) == 0 -> 'agreed'
        - trigsimp fallback -> 'agreed'
        - numeric spot-check fallback -> 'agreed'
        - otherwise -> 'disputed'
    """
    import sympy

    try:
        expr_a = parse_expr_convention(sympy_str_a)
        expr_b = parse_expr_convention(sympy_str_b)
    except Exception:
        return "disputed"

    symbols_a = {s.name for s in expr_a.free_symbols}
    symbols_b = {s.name for s in expr_b.free_symbols}

    if symbols_a != symbols_b:
        return "symbol_mismatch"

    diff = sympy.simplify(expr_a - expr_b)
    if diff == 0 or diff.is_zero:
        return "agreed"

    try:
        if sympy.trigsimp(diff) == 0:
            return "agreed"
    except Exception:
        pass

    symbols = list(expr_a.free_symbols | expr_b.free_symbols)
    if not symbols:
        try:
            return "agreed" if abs(complex((expr_a - expr_b).evalf())) < 1e-5 else "disputed"
        except Exception:
            return "disputed"

    # Deterministic numeric spot-check
    rng = random.Random(42)
    for _ in range(5):
        subs_dict = {s: rng.uniform(0.8, 3.2) for s in symbols}
        try:
            val_a = complex(expr_a.subs(subs_dict).evalf())
            val_b = complex(expr_b.subs(subs_dict).evalf())
            diff_val = abs(val_a - val_b)
            scale = max(abs(val_a), abs(val_b), 1e-9)
            if (diff_val / scale) > 1e-4:
                return "disputed"
        except Exception:
            return "disputed"

    return "agreed"


def compare_expression(sympy_str_a: str, sympy_str_b: str) -> bool:
    """
    Compares two mathematical expressions.
    Returns True if expressions agree and have matching free symbols, False otherwise.
    """
    return compare_expression_status(sympy_str_a, sympy_str_b) == "agreed"


def compare_choice(val_a: str, val_b: str) -> bool:
    """Compares choice answers: exact match (case-insensitive and trimmed)."""
    return str(val_a).strip().lower() == str(val_b).strip().lower()


def compare_chapter_keys(
    chapter: Any,
    data_dir: Path = Path("data")
) -> Dict[str, Any]:
    """
    Compares keys for items that have keys from BOTH claude and gemini families.
    Updates item status in items/ and writes compare.json.
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    items_dir = ch_dir / "items"
    keys_claude = ch_dir / "keys" / "claude"
    keys_gemini = ch_dir / "keys" / "gemini"

    claude_files = {p.stem: p for p in keys_claude.glob("*.json")} if keys_claude.exists() else {}
    gemini_files = {p.stem: p for p in keys_gemini.glob("*.json")} if keys_gemini.exists() else {}

    common_ids = sorted(set(claude_files.keys()) & set(gemini_files.keys()))

    results: Dict[str, Any] = {}
    verified_count = 0
    disputed_count = 0
    needs_grading_count = 0

    for item_id in common_ids:
        key_c = json.loads(claude_files[item_id].read_text(encoding="utf-8"))
        key_g = json.loads(gemini_files[item_id].read_text(encoding="utf-8"))

        answers_c = {a.get("part", ""): a for a in key_c.get("final_answers", [])}
        answers_g = {b.get("part", ""): b for b in key_g.get("final_answers", [])}

        all_parts = sorted(set(answers_c.keys()) | set(answers_g.keys()))
        part_results = []
        item_has_dispute = False
        item_has_unresolved = False

        if set(answers_c.keys()) != set(answers_g.keys()):
            item_has_dispute = True

        for p in all_parts:
            if p not in answers_c or p not in answers_g:
                existing = answers_c.get(p) or answers_g.get(p)
                part_kind = existing.get("kind", "unknown") if existing else "unknown"
                part_results.append({
                    "part": p,
                    "kind": part_kind,
                    "status": "disputed",
                    "reason": "part_missing_in_one_key"
                })
                item_has_dispute = True
                continue

            ans_c = answers_c[p]
            ans_g = answers_g[p]

            kind_c = ans_c.get("kind")
            kind_g = ans_g.get("kind")
            if kind_c != kind_g:
                part_results.append({
                    "part": p,
                    "kind": f"{kind_c}_vs_{kind_g}",
                    "status": "disputed",
                    "reason": f"kind_mismatch_{kind_c}_vs_{kind_g}"
                })
                item_has_dispute = True
                continue

            if kind_c == "numeric":
                ok = compare_numeric(ans_c, ans_g)
                st = "agreed" if ok else "disputed"
                if not ok:
                    item_has_dispute = True
                part_results.append({"part": p, "kind": kind_c, "status": st})

            elif kind_c == "expression":
                expr_st = compare_expression_status(ans_c.get("sympy", ""), ans_g.get("sympy", ""))
                if expr_st == "disputed":
                    item_has_dispute = True
                elif expr_st == "symbol_mismatch":
                    item_has_unresolved = True
                part_results.append({"part": p, "kind": kind_c, "status": expr_st})

            elif kind_c == "choice":
                ok = compare_choice(ans_c.get("value", ""), ans_g.get("value", ""))
                st = "agreed" if ok else "disputed"
                if not ok:
                    item_has_dispute = True
                part_results.append({"part": p, "kind": kind_c, "status": st})

            elif kind_c == "text":
                part_results.append({"part": p, "kind": kind_c, "status": "needs_grading"})
                item_has_unresolved = True

        if item_has_dispute:
            item_status = "key_disputed"
            disputed_count += 1
        elif item_has_unresolved:
            item_status = "needs_grading"
            needs_grading_count += 1
        else:
            item_status = "key_verified"
            verified_count += 1

        results[item_id] = {
            "status": item_status,
            "parts": part_results
        }

        # Update item status on disk if item file exists
        item_file = items_dir / f"{item_id}.json"
        if item_file.exists():
            try:
                item_data = json.loads(item_file.read_text(encoding="utf-8"))
                item_data["status"] = item_status
                item_file.write_text(json.dumps(item_data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    compare_report = {
        "chapter": ch_num,
        "items_compared": len(common_ids),
        "summary": {
            "key_verified": verified_count,
            "key_disputed": disputed_count,
            "needs_grading": needs_grading_count
        },
        "results": results
    }

    ch_dir.mkdir(parents=True, exist_ok=True)
    out_file = ch_dir / "compare.json"
    out_file.write_text(json.dumps(compare_report, indent=2), encoding="utf-8")

    return compare_report


def compare_chapter_slots(
    chapter: Any,
    data_dir: Path = Path("data")
) -> Dict[str, Any]:
    """
    Pairs the two families' slotted answers by slot_id and reuses existing
    numeric / expression / choice comparators.
    Item status:
      - key_verified: all non-text slots agree and there are no text slots
      - key_verified_partial: all non-text slots agree but text slots remain
      - key_disputed: any non-text slot disagrees
      - incomplete: any slot is unanswered by either family
    Writes compare_slots.json and updates item files.
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    slots_dir = ch_dir / "slots"
    items_dir = ch_dir / "items"
    keys_claude = ch_dir / "slotted" / "claude"
    keys_gemini = ch_dir / "slotted" / "gemini"

    slot_files = {p.stem: p for p in slots_dir.glob("*.json")} if slots_dir.exists() else {}
    claude_files = {p.stem: p for p in keys_claude.glob("*.json")} if keys_claude.exists() else {}
    gemini_files = {p.stem: p for p in keys_gemini.glob("*.json")} if keys_gemini.exists() else {}

    common_ids = sorted(set(slot_files.keys()) & set(claude_files.keys()) & set(gemini_files.keys()))

    results: Dict[str, Any] = {}
    verified_count = 0
    verified_partial_count = 0
    disputed_count = 0
    incomplete_count = 0

    slots_kind_counts = {"numeric": 0, "expression": 0, "choice": 0, "text": 0}
    total_slots = 0

    for item_id in common_ids:
        slot_obj = json.loads(slot_files[item_id].read_text(encoding="utf-8"))
        key_c = json.loads(claude_files[item_id].read_text(encoding="utf-8"))
        key_g = json.loads(gemini_files[item_id].read_text(encoding="utf-8"))

        answers_c = key_c.get("answers", {})
        answers_g = key_g.get("answers", {})
        unans_c = set(key_c.get("unanswerable", []))
        unans_g = set(key_g.get("unanswerable", []))

        slots = slot_obj.get("slots", [])
        slot_results = []

        item_has_unanswered = False
        item_has_disputed_non_text = False
        item_has_text = False

        for s in slots:
            slot_id = s.get("slot_id", "")
            kind = s.get("kind", "")
            if kind in slots_kind_counts:
                slots_kind_counts[kind] += 1
            total_slots += 1

            # Check if unanswered by either family
            if (slot_id in unans_c or slot_id not in answers_c or
                slot_id in unans_g or slot_id not in answers_g):
                slot_results.append({
                    "slot_id": slot_id,
                    "kind": kind,
                    "status": "unanswered"
                })
                item_has_unanswered = True
                continue

            ans_c = answers_c[slot_id]
            ans_g = answers_g[slot_id]

            if kind == "text":
                slot_results.append({
                    "slot_id": slot_id,
                    "kind": kind,
                    "status": "needs_grading"
                })
                item_has_text = True

            elif kind == "numeric":
                tol = s.get("rel_tol")
                ok = compare_numeric(ans_c, ans_g, rel_tol=tol)
                st = "agreed" if ok else "disputed"
                if not ok:
                    item_has_disputed_non_text = True
                slot_results.append({
                    "slot_id": slot_id,
                    "kind": kind,
                    "status": st
                })

            elif kind == "expression":
                ok = compare_expression(ans_c.get("sympy", ""), ans_g.get("sympy", ""))
                st = "agreed" if ok else "disputed"
                if not ok:
                    item_has_disputed_non_text = True
                slot_results.append({
                    "slot_id": slot_id,
                    "kind": kind,
                    "status": st
                })

            elif kind == "choice":
                ok = compare_choice(ans_c.get("value", ""), ans_g.get("value", ""))
                st = "agreed" if ok else "disputed"
                if not ok:
                    item_has_disputed_non_text = True
                slot_results.append({
                    "slot_id": slot_id,
                    "kind": kind,
                    "status": st
                })

        # Determine item status
        if item_has_unanswered:
            item_status = "incomplete"
            incomplete_count += 1
        elif item_has_disputed_non_text:
            item_status = "key_disputed"
            disputed_count += 1
        elif item_has_text:
            item_status = "key_verified_partial"
            verified_partial_count += 1
        else:
            item_status = "key_verified"
            verified_count += 1

        results[item_id] = {
            "status": item_status,
            "slots": slot_results
        }

        # Update item status on disk if item file exists
        item_file = items_dir / f"{item_id}.json"
        if item_file.exists():
            try:
                item_data = json.loads(item_file.read_text(encoding="utf-8"))
                item_data["status"] = item_status
                item_file.write_text(json.dumps(item_data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    compare_report = {
        "chapter": ch_num,
        "items_compared": len(common_ids),
        "summary": {
            "key_verified": verified_count,
            "key_verified_partial": verified_partial_count,
            "key_disputed": disputed_count,
            "incomplete": incomplete_count
        },
        "slots_summary": {
            **slots_kind_counts,
            "total": total_slots
        },
        "results": results
    }

    ch_dir.mkdir(parents=True, exist_ok=True)
    out_file = ch_dir / "compare_slots.json"
    out_file.write_text(json.dumps(compare_report, indent=2), encoding="utf-8")

    return compare_report


# ---------------------------------------------------------------------------
# Review Queue & Resolution Logic
# ---------------------------------------------------------------------------

def generate_review_queue(
    chapter: Any,
    data_dir: Path = Path("data")
) -> List[Dict[str, str]]:
    """
    Writes data/itembank/chXX/review_queue.json listing every (item_id, slot_id)
    whose slot status is not 'agree': text slots, disagreeing slots, and
    slots unanswered by either family.
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    cmp_file = ch_dir / "compare_slots.json"

    if not cmp_file.exists():
        compare_chapter_slots(chapter, data_dir=data_dir)

    cmp_data = json.loads(cmp_file.read_text(encoding="utf-8"))
    results = cmp_data.get("results", {})

    queue: List[Dict[str, str]] = []
    for item_id in sorted(results.keys()):
        slots = results[item_id].get("slots", [])
        for s in slots:
            status = s.get("status", "")
            if status not in ("agree", "agreed"):
                queue.append({
                    "item_id": item_id,
                    "slot_id": s.get("slot_id", "")
                })

    ch_dir.mkdir(parents=True, exist_ok=True)
    out_file = ch_dir / "review_queue.json"
    out_file.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    return queue


def resolve_chapter(
    chapter: Any,
    data_dir: Path = Path("data")
) -> Dict[str, Any]:
    """
    Reads reviews/claude.json and reviews/gemini.json, each a list of
    {"item_id", "slot_id", "verdict": "equivalent"|"other_wrong"|"mine_wrong"|"both_wrong"|"unclear", "reason"}.
    Per queued slot:
      - both 'equivalent' -> verified
      - one says 'mine_wrong' and the other says 'other_wrong', agreeing on the same wrong answer -> resolved_by_concession, adopting the agreed-correct answer
      - anything else, or a missing review -> disputed
    An item is usable if every slot is agree, verified or resolved_by_concession.
    Writes resolution.json.
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    cmp_file = ch_dir / "compare_slots.json"

    if not cmp_file.exists():
        compare_chapter_slots(chapter, data_dir=data_dir)

    cmp_data = json.loads(cmp_file.read_text(encoding="utf-8"))
    results = cmp_data.get("results", {})

    reviews_dir = ch_dir / "reviews"
    rev_c_file = reviews_dir / "claude.json"
    rev_g_file = reviews_dir / "gemini.json"

    reviews_c: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if rev_c_file.exists():
        try:
            for r in json.loads(rev_c_file.read_text(encoding="utf-8")):
                if "item_id" in r and "slot_id" in r:
                    reviews_c[(r["item_id"], r["slot_id"])] = r
        except Exception:
            pass

    reviews_g: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if rev_g_file.exists():
        try:
            for r in json.loads(rev_g_file.read_text(encoding="utf-8")):
                if "item_id" in r and "slot_id" in r:
                    reviews_g[(r["item_id"], r["slot_id"])] = r
        except Exception:
            pass

    # Load slotted keys to adopt answers when resolving by concession
    slotted_c_dir = ch_dir / "slotted" / "claude"
    slotted_g_dir = ch_dir / "slotted" / "gemini"

    keys_c: Dict[str, Dict[str, Any]] = {}
    if slotted_c_dir.exists():
        for p in slotted_c_dir.glob("*.json"):
            try:
                keys_c[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    keys_g: Dict[str, Dict[str, Any]] = {}
    if slotted_g_dir.exists():
        for p in slotted_g_dir.glob("*.json"):
            try:
                keys_g[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    items_res: Dict[str, Any] = {}
    usable_item_ids: List[str] = []
    excluded_item_ids: List[str] = []

    outcomes_count = {
        "agreed": 0,
        "verified": 0,
        "resolved_by_concession": 0,
        "disputed": 0,
        "total": 0
    }

    USABLE_SLOT_OUTCOMES = {"agree", "agreed", "verified", "resolved_by_concession"}

    for item_id in sorted(results.keys()):
        slots = results[item_id].get("slots", [])
        slot_resolutions = []
        item_is_usable = True

        for s in slots:
            slot_id = s.get("slot_id", "")
            status = s.get("status", "")
            outcomes_count["total"] += 1

            if status in ("agree", "agreed"):
                outcome = "agreed"
                outcomes_count["agreed"] += 1
                adopted = (
                    keys_c.get(item_id, {}).get("answers", {}).get(slot_id)
                    or keys_g.get(item_id, {}).get("answers", {}).get(slot_id)
                )
                slot_resolutions.append({
                    "slot_id": slot_id,
                    "kind": s.get("kind"),
                    "outcome": outcome,
                    "adopted_answer": adopted
                })
            else:
                # Queued slot
                key = (item_id, slot_id)
                rev_c = reviews_c.get(key)
                rev_g = reviews_g.get(key)

                verdict_c = rev_c.get("verdict") if rev_c else None
                verdict_g = rev_g.get("verdict") if rev_g else None

                if verdict_c == "equivalent" and verdict_g == "equivalent":
                    outcome = "verified"
                    outcomes_count["verified"] += 1
                    adopted = (
                        keys_c.get(item_id, {}).get("answers", {}).get(slot_id)
                        or keys_g.get(item_id, {}).get("answers", {}).get(slot_id)
                    )
                    slot_resolutions.append({
                        "slot_id": slot_id,
                        "kind": s.get("kind"),
                        "outcome": outcome,
                        "adopted_answer": adopted,
                        "reason": "both_equivalent"
                    })
                elif verdict_c == "mine_wrong" and verdict_g == "other_wrong":
                    # Claude concedes; Gemini is agreed-correct
                    outcome = "resolved_by_concession"
                    outcomes_count["resolved_by_concession"] += 1
                    adopted = keys_g.get(item_id, {}).get("answers", {}).get(slot_id)
                    slot_resolutions.append({
                        "slot_id": slot_id,
                        "kind": s.get("kind"),
                        "outcome": outcome,
                        "agreed_correct": "gemini",
                        "adopted_answer": adopted,
                        "reason": "claude_conceded"
                    })
                elif verdict_g == "mine_wrong" and verdict_c == "other_wrong":
                    # Gemini concedes; Claude is agreed-correct
                    outcome = "resolved_by_concession"
                    outcomes_count["resolved_by_concession"] += 1
                    adopted = keys_c.get(item_id, {}).get("answers", {}).get(slot_id)
                    slot_resolutions.append({
                        "slot_id": slot_id,
                        "kind": s.get("kind"),
                        "outcome": outcome,
                        "agreed_correct": "claude",
                        "adopted_answer": adopted,
                        "reason": "gemini_conceded"
                    })
                else:
                    outcome = "disputed"
                    outcomes_count["disputed"] += 1
                    slot_resolutions.append({
                        "slot_id": slot_id,
                        "kind": s.get("kind"),
                        "outcome": outcome,
                        "verdict_claude": verdict_c,
                        "verdict_gemini": verdict_g,
                        "reason": "unresolved_or_conflicting_reviews"
                    })

            if outcome not in USABLE_SLOT_OUTCOMES:
                item_is_usable = False

        item_status = "usable" if item_is_usable else "excluded"
        if item_is_usable:
            usable_item_ids.append(item_id)
        else:
            excluded_item_ids.append(item_id)

        items_res[item_id] = {
            "status": item_status,
            "slots": slot_resolutions
        }

    resolution_report = {
        "chapter": ch_num,
        "summary": {
            "usable_items": len(usable_item_ids),
            "excluded_items": len(excluded_item_ids),
            "total_items": len(results)
        },
        "slot_outcomes": outcomes_count,
        "usable_item_ids": usable_item_ids,
        "excluded_item_ids": excluded_item_ids,
        "items": items_res
    }

    ch_dir.mkdir(parents=True, exist_ok=True)
    out_file = ch_dir / "resolution.json"
    out_file.write_text(json.dumps(resolution_report, indent=2), encoding="utf-8")

    return resolution_report


# ---------------------------------------------------------------------------
# Report Command
# ---------------------------------------------------------------------------

def generate_report(
    chapter: Any,
    data_dir: Path = Path("data")
) -> str:
    """
    Produces report containing COUNTS ONLY.
    - Items by bucket and status
    - How many need a figure
    - How many keys exist per family
    - Parse warnings by item ID
    If compare_slots.json exists, displays slot-based statuses and slot counts by kind.
    Strictly NO problem statements or keys.
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    items_dir = ch_dir / "items"

    total_items = 0
    buckets: Dict[str, int] = {}
    statuses: Dict[str, int] = {}
    needs_figure_count = 0
    parse_warnings: List[str] = []

    if items_dir.exists():
        for p in sorted(items_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                total_items += 1
                b = data.get("bucket", "unknown")
                buckets[b] = buckets.get(b, 0) + 1
                s = data.get("status", "unknown")
                statuses[s] = statuses.get(s, 0) + 1
                if data.get("needs_figure", False):
                    needs_figure_count += 1
                if not data.get("statement_md"):
                    parse_warnings.append(f"{data.get('item_id', p.stem)}: empty_statement")
            except Exception as e:
                parse_warnings.append(f"{p.stem}: json_load_error")

    keys_claude_count = 0
    keys_gemini_count = 0
    keys_claude = ch_dir / "keys" / "claude"
    keys_gemini = ch_dir / "keys" / "gemini"
    slotted_claude = ch_dir / "slotted" / "claude"
    slotted_gemini = ch_dir / "slotted" / "gemini"

    if slotted_claude.exists() and list(slotted_claude.glob("*.json")):
        keys_claude_count = len(list(slotted_claude.glob("*.json")))
    elif keys_claude.exists():
        keys_claude_count = len(list(keys_claude.glob("*.json")))

    if slotted_gemini.exists() and list(slotted_gemini.glob("*.json")):
        keys_gemini_count = len(list(slotted_gemini.glob("*.json")))
    elif keys_gemini.exists():
        keys_gemini_count = len(list(keys_gemini.glob("*.json")))

    # Check for slot comparison results
    compare_slots_file = ch_dir / "compare_slots.json"
    slot_rep = None
    if compare_slots_file.exists():
        try:
            slot_rep = json.loads(compare_slots_file.read_text(encoding="utf-8"))
        except Exception:
            slot_rep = None

    # Check for resolution results
    resolution_file = ch_dir / "resolution.json"
    res_rep = None
    if resolution_file.exists():
        try:
            res_rep = json.loads(resolution_file.read_text(encoding="utf-8"))
        except Exception:
            res_rep = None

    lines = [
        f"=== Item Bank Report: Chapter {ch_num:02d} ===",
        f"Total items: {total_items}",
        "",
        "Items by bucket:"
    ]
    if buckets:
        for b in sorted(buckets.keys()):
            lines.append(f"  Bucket {b}: {buckets[b]}")
    else:
        lines.append("  (none)")

    lines.append("")
    if slot_rep and "summary" in slot_rep:
        lines.append("Items by status (slot-based):")
        slot_statuses = slot_rep["summary"]
        for s in sorted(slot_statuses.keys()):
            lines.append(f"  {s}: {slot_statuses[s]}")

        lines.append("")
        lines.append("Slots by kind:")
        slot_kinds = slot_rep.get("slots_summary", {})
        for k in ["choice", "expression", "numeric", "text", "total"]:
            if k in slot_kinds:
                lines.append(f"  {k}: {slot_kinds[k]}")
    else:
        lines.append("Items by status:")
        if statuses:
            for s in sorted(statuses.keys()):
                lines.append(f"  {s}: {statuses[s]}")
        else:
            lines.append("  (none)")

    if res_rep and "summary" in res_rep:
        lines.append("")
        lines.append("Items by usability:")
        lines.append(f"  usable: {res_rep['summary'].get('usable_items', 0)}")
        lines.append(f"  excluded: {res_rep['summary'].get('excluded_items', 0)}")
        lines.append("")
        lines.append("Slot outcomes by type:")
        outcomes = res_rep.get("slot_outcomes", {})
        for ot in ["agreed", "verified", "resolved_by_concession", "disputed"]:
            if ot in outcomes:
                lines.append(f"  {ot}: {outcomes[ot]}")

    lines.append("")
    lines.append(f"Items requiring figure: {needs_figure_count}")
    lines.append("")
    lines.append("Keys by model family:")
    lines.append(f"  claude: {keys_claude_count}")
    lines.append(f"  gemini: {keys_gemini_count}")
    lines.append("")
    lines.append("Parse warnings (by item ID):")
    if parse_warnings:
        for w in parse_warnings:
            lines.append(f"  {w}")
    else:
        lines.append("  none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbol Audit Logic
# ---------------------------------------------------------------------------

def audit_chapter_symbols(
    chapter: Any,
    data_dir: Path = Path("data")
) -> Dict[str, Dict[str, int]]:
    """
    Audits symbols across expression answers in both key folders.
    Returns counts of keys from each family using each non-convention symbol name:
    {
        "claude": {symbol_name: key_count, ...},
        "gemini": {symbol_name: key_count, ...}
    }
    """
    ch_num = int(re.sub(r'^[a-zA-Z_]+', '', str(chapter)))
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}" / "keys"

    result: Dict[str, Dict[str, int]] = {"claude": {}, "gemini": {}}

    for family in ["claude", "gemini"]:
        fam_dir = ch_dir / family
        if not fam_dir.exists():
            continue
        sym_to_keys: Dict[str, Set[str]] = {}
        for p in sorted(fam_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            key_symbols: Set[str] = set()
            for ans in data.get("final_answers", []):
                if ans.get("kind") == "expression":
                    sp = ans.get("sympy", "")
                    if sp:
                        try:
                            expr = parse_expr_convention(sp)
                            key_symbols.update({s.name for s in expr.free_symbols})
                        except Exception:
                            pass
            for s in key_symbols:
                if s not in CONVENTION_SYMBOLS:
                    if s not in sym_to_keys:
                        sym_to_keys[s] = set()
                    sym_to_keys[s].add(p.stem)

        result[family] = {s: len(keys) for s, keys in sorted(sym_to_keys.items())}

    return result


# ---------------------------------------------------------------------------
# CLI Entrypoint & Guard
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deterministic item-bank tooling for learn-assist.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # extract-a
    p_extract = subparsers.add_parser("extract-a", help="Extract Bucket A items from printed chapter")
    p_extract.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_extract.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # validate-key
    p_val = subparsers.add_parser("validate-key", help="Validate a key JSON file against schema")
    p_val.add_argument("key_file", type=Path, help="Path to key JSON file")

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare claude vs gemini keys for a chapter")
    p_cmp.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_cmp.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # report
    p_rep = subparsers.add_parser("report", help="Report counts and statuses for chapter item bank")
    p_rep.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_rep.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # audit-symbols
    p_audit = subparsers.add_parser("audit-symbols", help="Audit non-convention symbols in keys")
    p_audit.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_audit.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # validate-slot
    p_valslot = subparsers.add_parser("validate-slot", help="Validate a slot JSON file against schema")
    p_valslot.add_argument("slot_file", type=Path, help="Path to slot JSON file")

    # validate-slotted-key
    p_valslotted = subparsers.add_parser("validate-slotted-key", help="Validate a slotted key JSON file")
    p_valslotted.add_argument("key_file", type=Path, help="Path to slotted key JSON file")
    p_valslotted.add_argument("--slot-file", type=Path, default=None, help="Path to corresponding slot JSON file")
    p_valslotted.add_argument("--slots-dir", type=Path, default=None, help="Directory containing slot JSON files")

    # compare-slots
    p_cmpslots = subparsers.add_parser("compare-slots", help="Compare claude vs gemini slotted keys for a chapter")
    p_cmpslots.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_cmpslots.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # review-queue
    p_queue = subparsers.add_parser("review-queue", help="Generate review queue for non-agreed slots")
    p_queue.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_queue.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # resolve
    p_res = subparsers.add_parser("resolve", help="Resolve queued reviews for a chapter")
    p_res.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    p_res.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    args = parser.parse_args()

    if args.subcommand == "extract-a":
        count, ids, warnings = extract_chapter_a(args.chapter, data_dir=args.data_dir)
        print(guard_output(f"extract-a: wrote {count} item files for Chapter {args.chapter}"))
        print(guard_output(f"Item IDs: {ids[0]}..{ids[-1]}" if ids else "Item IDs: none"))
        if warnings:
            print(guard_output(f"Warnings: {len(warnings)} (IDs: {[w[0] for w in warnings]})"))

    elif args.subcommand == "validate-key":
        valid, errors = validate_key_file(args.key_file)
        if valid:
            print(guard_output(f"VALID: {args.key_file.name}"))
            sys.exit(0)
        else:
            print(guard_output(f"INVALID: {args.key_file.name} - Errors: {errors}"))
            sys.exit(1)

    elif args.subcommand == "compare":
        report = compare_chapter_keys(args.chapter, data_dir=args.data_dir)
        s = report["summary"]
        print(guard_output(
            f"compare: {report['items_compared']} items compared. "
            f"key_verified={s['key_verified']}, "
            f"key_disputed={s['key_disputed']}, "
            f"needs_grading={s['needs_grading']}"
        ))

    elif args.subcommand == "report":
        rep_text = generate_report(args.chapter, data_dir=args.data_dir)
        print(guard_output(rep_text))

    elif args.subcommand == "audit-symbols":
        audit_res = audit_chapter_symbols(args.chapter, data_dir=args.data_dir)
        for fam in ["claude", "gemini"]:
            counts = audit_res.get(fam, {})
            print(guard_output(f"{fam} non-convention symbols: {counts}"))

    elif args.subcommand == "validate-slot":
        valid, errors = validate_slot_file(args.slot_file)
        if valid:
            print(guard_output(f"VALID: {args.slot_file.name}"))
            sys.exit(0)
        else:
            print(guard_output(f"INVALID: {args.slot_file.name} - Errors: {errors}"))
            sys.exit(1)

    elif args.subcommand == "validate-slotted-key":
        valid, errors = validate_slotted_key_file(args.key_file, slot_path=args.slot_file, slots_dir=args.slots_dir)
        if valid:
            print(guard_output(f"VALID: {args.key_file.name}"))
            sys.exit(0)
        else:
            print(guard_output(f"INVALID: {args.key_file.name} - Errors: {errors}"))
            sys.exit(1)

    elif args.subcommand == "compare-slots":
        report = compare_chapter_slots(args.chapter, data_dir=args.data_dir)
        s = report["summary"]
        print(guard_output(
            f"compare-slots: {report['items_compared']} items compared. "
            f"key_verified={s['key_verified']}, "
            f"key_verified_partial={s['key_verified_partial']}, "
            f"key_disputed={s['key_disputed']}, "
            f"incomplete={s['incomplete']}"
        ))

    elif args.subcommand == "review-queue":
        queue = generate_review_queue(args.chapter, data_dir=args.data_dir)
        print(guard_output(f"review-queue: {len(queue)} slots queued for Chapter {args.chapter}"))

    elif args.subcommand == "resolve":
        rep = resolve_chapter(args.chapter, data_dir=args.data_dir)
        s = rep["summary"]
        print(guard_output(
            f"resolve: Chapter {args.chapter} resolved. "
            f"usable={s['usable_items']}, excluded={s['excluded_items']}. "
            f"Slot outcomes: {rep['slot_outcomes']}"
        ))


if __name__ == "__main__":
    main()
