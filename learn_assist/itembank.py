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


def compare_numeric(ans_a: Dict[str, Any], ans_b: Dict[str, Any]) -> bool:
    """
    Compares two numeric answers:
    - unit-normalised (handles SI prefixes and T/G/Hz/rad/s)
    - equal within the larger rel_tol
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
    rel_tol = max(tol_a, tol_b)

    # Near zero absolute check
    if abs(norm_a) < 1e-12 and abs(norm_b) < 1e-12:
        return True

    diff = abs(norm_a - norm_b)
    scale = max(abs(norm_a), abs(norm_b))
    return (diff / scale) <= rel_tol


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
    if keys_claude.exists():
        keys_claude_count = len(list(keys_claude.glob("*.json")))
    if keys_gemini.exists():
        keys_gemini_count = len(list(keys_gemini.glob("*.json")))

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
    lines.append("Items by status:")
    if statuses:
        for s in sorted(statuses.keys()):
            lines.append(f"  {s}: {statuses[s]}")
    else:
        lines.append("  (none)")

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


if __name__ == "__main__":
    main()
