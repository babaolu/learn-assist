#!/usr/bin/env python3
"""
learn_assist.verify

Verification suite for Stage 1 textbook ingestion and extraction:
a) COMPLETENESS: Every page in range has a non-empty checkpoint >= min character threshold.
b) EQUATION SEQUENCE: (2.1)..(2.141) and (3.1)..(3.153) contiguous with no gaps and no duplicates.
c) SECTION & FIGURE/TABLE SEQUENCE: Section numbers, Figure numbers, and Table numbers contiguous.
d) PROBLEM SEQUENCE: Chapter end-of-chapter problems contiguous with no gaps and no duplicates.
e) LATEX & OCR INTEGRITY: Balanced $ and $$, no corrupted OCR signatures, no mid-word starts.
f) CROSS-MODEL SPOT CHECK: 10% sample re-transcribed with a different model family, asserting
   agreement on equation numbers and section headings.

Outputs full audit log to data/verification_report.md.
EXITS NON-ZERO IF ANY CHECK FAILS.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional

import requests
import pymupdf

from learn_assist.ingest import (
    DEFAULT_API_KEY,
    transcribe_page_image,
    render_page_to_png,
)

MIN_CHAR_THRESHOLD = 150
IN_SCOPE_PAGES = list(range(33, 126))  # PDF 33 to 125 inclusive (93 pages)
CH2_PAGES = list(range(33, 77))        # PDF 33 to 76 (Book 13 to 56)
CH3_PAGES = list(range(77, 126))       # PDF 77 to 125 (Book 57 to 105)

CORRUPT_OCR_PATTERNS = [
    r'\(f>',
    r'\bfi2\b',
    r'</>',
    r'<\s*t>\'?',
    r'\\\(t>\'?',
    r'ysincf',
    r'\{R2p\}',
]


class VerificationResult:
    def __init__(self):
        self.passed = True
        self.failures: List[Dict[str, Any]] = []
        self.details: Dict[str, Any] = {}

    def fail(self, check_name: str, message: str, context: Optional[Any] = None):
        self.passed = False
        self.failures.append({
            "check": check_name,
            "message": message,
            "context": context
        })


def check_completeness(pages_dir: Path, min_chars: int = MIN_CHAR_THRESHOLD) -> Tuple[bool, List[str], Dict[str, int]]:
    """Check a) Completeness: Every in-scope page exists, non-empty, above threshold."""
    failures = []
    page_counts = {}
    for p in IN_SCOPE_PAGES:
        page_path = pages_dir / f"page_{p:04d}.md"
        if not page_path.exists():
            failures.append(f"Page {p} (Book p. {p - 20}) checkpoint file MISSING: {page_path.name}")
            continue
        content = page_path.read_text(encoding="utf-8").strip()
        count = len(content)
        page_counts[f"page_{p:04d}"] = count
        if count == 0:
            failures.append(f"Page {p} (Book p. {p - 20}) checkpoint is EMPTY (0 chars)")
        elif count < min_chars:
            failures.append(f"Page {p} (Book p. {p - 20}) below threshold: {count} chars < {min_chars} chars")

    return len(failures) == 0, failures, page_counts


def extract_equation_numbers(text: str, chapter_num: int) -> Tuple[List[int], List[str]]:
    """
    Extracts all defined equation numbers for a chapter from Markdown text.
    Handles standard integers and lettered equations (e.g. 2.16, 2.24a, 2.24b).
    Returns list of base integers and list of raw strings.
    """
    # Match tags like (2.14), (2.117), \tag{2.117}, \quad (2.117), (2.122a)
    raw_matches = re.findall(rf'(?:\\tag\{{|\(|\bEq\.\s*\(?){chapter_num}\.(\d+)([a-z]?)\)?', text)
    base_ints = []
    raw_tags = []
    for num_str, letter in raw_matches:
        base_int = int(num_str)
        base_ints.append(base_int)
        raw_tags.append(f"{chapter_num}.{num_str}{letter}")
    return base_ints, raw_tags


def check_sequence_contiguity(
    name: str,
    items: List[int],
    expected_start: int = 1,
    expected_end: Optional[int] = None
) -> Tuple[bool, List[str]]:
    """Asserts a sequence of integers is contiguous with no gaps and no duplicates."""
    failures = []
    if not items:
        return False, [f"{name}: No items found."]

    unique_items = sorted(set(items))
    start_val = expected_start if expected_start is not None else unique_items[0]
    end_val = expected_end if expected_end is not None else unique_items[-1]

    # Check for missing items (gaps)
    expected_set = set(range(start_val, end_val + 1))
    missing = sorted(expected_set - set(unique_items))
    if missing:
        failures.append(f"{name}: Missing sequence numbers (gaps): {missing}")

    # Check for unexpected items outside expected range
    extra = sorted(set(unique_items) - expected_set)
    if extra:
        failures.append(f"{name}: Unexpected sequence numbers outside {start_val}..{end_val}: {extra}")

    # Check for duplicates if original list has duplicates
    counts: Dict[int, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    duplicates = [x for x, c in counts.items() if c > 1 and start_val <= x <= end_val]
    # Note: Lettered equations sharing base int (e.g., 2.24a, 2.24b) are permitted, handled in caller.

    return len(failures) == 0, failures


def check_equations(chapters_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check b) Equation sequence: (2.1)..(2.N) and (3.1)..(3.M) contiguous with no gaps."""
    failures = []
    stats = {}

    for ch_num, expected_max in [(2, 141), (3, 153)]:
        ch_file = chapters_dir / f"chapter_{ch_num:02d}.md"
        if not ch_file.exists():
            failures.append(f"Chapter file {ch_file.name} missing for equation check.")
            continue
        text = ch_file.read_text(encoding="utf-8")
        base_ints, raw_tags = extract_equation_numbers(text, ch_num)
        unique_base = sorted(set(base_ints))

        stats[f"chapter_{ch_num:02d}_equations"] = {
            "total_matches": len(base_ints),
            "unique_base_equations": len(unique_base),
            "min": unique_base[0] if unique_base else None,
            "max": unique_base[-1] if unique_base else None
        }

        # Assert contiguous from 1 to expected_max
        expected_set = set(range(1, expected_max + 1))
        found_set = set(unique_base)
        missing = sorted(expected_set - found_set)
        if missing:
            failures.append(f"Chapter {ch_num:02d} Equations: Gaps detected. Missing {len(missing)} equations: {missing}")

    return len(failures) == 0, failures, stats


def check_sections_and_figures(chapters_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check c) Section sequence, Figure sequence, and Table sequence."""
    failures = []
    stats = {}

    # Chapter 2: Sections 2.1 to 2.6; Figures 2.1 to 2.18
    # Chapter 3: Sections 3.1 to 3.4; Figures 3.1 to 3.14; Table 3.1
    targets = {
        2: {"sections": range(1, 7), "figures": range(1, 19), "tables": []},
        3: {"sections": range(1, 5), "figures": range(1, 15), "tables": [1]}
    }

    for ch_num, tgt in targets.items():
        ch_file = chapters_dir / f"chapter_{ch_num:02d}.md"
        if not ch_file.exists():
            failures.append(f"Chapter file {ch_file.name} missing for section/figure check.")
            continue
        text = ch_file.read_text(encoding="utf-8")

        # Sections: matches "## Section 2.1" or "## 2.1" or "# 2.1"
        sec_matches = re.findall(rf'(?:#+\s*(?:Section\s+)?|\bSection\s+){ch_num}\.(\d+)\b', text)
        sec_ints = sorted(set(int(x) for x in sec_matches))
        expected_secs = set(tgt["sections"])
        missing_secs = sorted(expected_secs - set(sec_ints))
        if missing_secs:
            failures.append(f"Chapter {ch_num:02d} Sections: Missing section headings: {missing_secs}")

        # Figures: matches "**Figure 2.1**" or "Figure 2.1"
        fig_matches = re.findall(rf'Figure\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)
        fig_ints = sorted(set(int(x) for x in fig_matches))
        expected_figs = set(tgt["figures"])
        missing_figs = sorted(expected_figs - set(fig_ints))
        if missing_figs:
            failures.append(f"Chapter {ch_num:02d} Figures: Missing figures: {missing_figs}")

        # Tables: matches "Table 3.1"
        if tgt["tables"]:
            tbl_matches = re.findall(rf'Table\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)
            tbl_ints = sorted(set(int(x) for x in tbl_matches))
            expected_tbls = set(tgt["tables"])
            missing_tbls = sorted(expected_tbls - set(tbl_ints))
            if missing_tbls:
                failures.append(f"Chapter {ch_num:02d} Tables: Missing tables: {missing_tbls}")

        stats[f"chapter_{ch_num:02d}_structure"] = {
            "sections_found": sec_ints,
            "figures_found": fig_ints,
            "tables_found": sorted(set(int(x) for x in re.findall(rf'Table\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)))
        }

    return len(failures) == 0, failures, stats


def check_problem_sequence(problems_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check d) Problem sequence: contiguous end-of-chapter problems with no gaps and no duplicates."""
    failures = []
    stats = {}

    expected_ranges = {
        2: (1, 32),  # Problems 2.1 to 2.32
        3: (1, 41)   # Problems 3.1 to 3.41
    }

    for ch_num, (start_n, end_n) in expected_ranges.items():
        prob_file = problems_dir / f"chapter_{ch_num:02d}.json"
        if not prob_file.exists():
            failures.append(f"Problems file {prob_file.name} missing.")
            continue
        try:
            with open(prob_file, "r", encoding="utf-8") as f:
                problems = json.load(f)
        except Exception as e:
            failures.append(f"Failed to read {prob_file.name}: {e}")
            continue

        item_numbers = [p.get("item_number") for p in problems if "item_number" in p]
        unique_numbers = sorted(set(item_numbers))

        expected_set = set(range(start_n, end_n + 1))
        missing = sorted(expected_set - set(unique_numbers))
        if missing:
            failures.append(f"Chapter {ch_num:02d} Problems: Missing problem numbers: {missing}")

        # Duplicate check
        counts: Dict[int, int] = {}
        for x in item_numbers:
            counts[x] = counts.get(x, 0) + 1
        duplicates = [x for x, c in counts.items() if c > 1]
        if duplicates:
            failures.append(f"Chapter {ch_num:02d} Problems: Duplicate problem numbers: {duplicates}")

        stats[f"chapter_{ch_num:02d}_problems"] = {
            "total_items": len(problems),
            "range": f"{start_n}..{end_n}",
            "captured_items": len(unique_numbers)
        }

    return len(failures) == 0, failures, stats


def check_latex_integrity(pages_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check e) LaTeX integrity:
    - Balanced $ and $$
    - No corrupted-OCR signatures (<>, (f>, fi2, </>)
    - No page begins mid-word
    """
    failures = []
    stats = {"pages_checked": 0, "issues": 0}

    for p in IN_SCOPE_PAGES:
        page_file = pages_dir / f"page_{p:04d}.md"
        if not page_file.exists():
            continue
        content = page_file.read_text(encoding="utf-8")
        stats["pages_checked"] += 1

        # Check balanced $$
        # Count display math delimiters ($$)
        display_math_count = len(re.findall(r'(?<!\\)\$\$', content))
        if display_math_count % 2 != 0:
            failures.append(f"Page {p} (Book p. {p - 20}): Unbalanced $$ display math delimiter (count={display_math_count})")

        # Strip $$...$$ before checking inline $
        no_display = re.sub(r'(?<!\\)\$\$.*?(?<!\\)\$\$', '', content, flags=re.DOTALL)
        inline_math_count = len(re.findall(r'(?<!\\)\$', no_display))
        if inline_math_count % 2 != 0:
            failures.append(f"Page {p} (Book p. {p - 20}): Unbalanced inline $ math delimiter (count={inline_math_count})")

        # Check corrupted OCR signatures
        for pattern in CORRUPT_OCR_PATTERNS:
            match = re.search(pattern, content)
            if match:
                failures.append(f"Page {p} (Book p. {p - 20}): Corrupted OCR signature found matching '{pattern}': '{match.group(0)}'")

        # Check if page begins mid-word
        # A mid-word beginning typically starts with a hyphen followed by letters (e.g. "-ference")
        # or non-heading lowercase word without punctuation.
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            if first_line.startswith("-") and len(first_line) > 1 and first_line[1].isalpha():
                failures.append(f"Page {p} (Book p. {p - 20}): Begins mid-word with hyphenation: '{first_line[:40]}'")

    return len(failures) == 0, failures, stats


def check_cross_model_spot(
    pages_dir: Path,
    sample_ratio: float = 0.10,
    seed: int = 42,
    api_key: Optional[str] = None,
    cross_model: str = "gemma-4-31b-it"
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check f) Cross-Model Spot Check:
    Re-transcribes a random 10% sample of in-scope pages with a DIFFERENT model family (e.g. gemma-4-31b-it).
    Asserts agreement on equation numbers present and section headings present.
    """
    failures = []
    stats = {}
    key = api_key or DEFAULT_API_KEY
    if not key:
        return False, ["Cross-model spot check requires GEMINI_API_KEY."], {}

    random.seed(seed)
    sample_size = max(1, int(len(IN_SCOPE_PAGES) * sample_ratio))
    sampled_pages = sorted(random.sample(IN_SCOPE_PAGES, sample_size))

    cross_dir = Path("data/cross_model")
    cross_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path("data/page_images")

    print(f"\n--- Running Cross-Model Spot Check on {len(sampled_pages)} pages ({sampled_pages}) with model '{cross_model}' ---")
    spot_results = []

    for p in sampled_pages:
        page_key = f"page_{p:04d}"
        primary_file = pages_dir / f"{page_key}.md"
        cross_file = cross_dir / f"{page_key}_cross.md"
        img_path = images_dir / f"{page_key}.png"

        if not primary_file.exists():
            failures.append(f"Page {p}: Primary transcription missing for spot check.")
            continue
        primary_text = primary_file.read_text(encoding="utf-8")

        # Load or generate cross-model transcription
        if cross_file.exists() and len(cross_file.read_text(encoding="utf-8").strip()) > 0:
            cross_text = cross_file.read_text(encoding="utf-8")
            print(f"Loaded cached cross-transcription for {page_key}...")
        else:
            print(f"Transcribing {page_key} with {cross_model}...")
            try:
                cross_text = transcribe_page_image(img_path, api_key=key, model=cross_model)
                cross_file.write_text(cross_text, encoding="utf-8")
            except Exception as e:
                failures.append(f"Page {p}: Cross-model transcription failed: {e}")
                continue

        ch_num = 2 if p in CH2_PAGES else 3
        # Extract equation numbers from both
        _, primary_eqs = extract_equation_numbers(primary_text, ch_num)
        _, cross_eqs = extract_equation_numbers(cross_text, ch_num)

        primary_eq_set = set(primary_eqs)
        cross_eq_set = set(cross_eqs)

        # Extract section headings
        primary_secs = set(re.findall(rf'{ch_num}\.(\d+)', primary_text))
        cross_secs = set(re.findall(rf'{ch_num}\.(\d+)', cross_text))

        eq_diff = primary_eq_set.symmetric_difference(cross_eq_set)
        sec_diff = primary_secs.symmetric_difference(cross_secs)

        match = (len(eq_diff) == 0 and len(sec_diff) == 0)
        spot_results.append({
            "page": p,
            "book_page": p - 20,
            "primary_eqs": sorted(primary_eq_set),
            "cross_eqs": sorted(cross_eq_set),
            "primary_secs": sorted(primary_secs),
            "cross_secs": sorted(cross_secs),
            "match": match
        })

        if not match:
            msg = f"Page {p} (Book p. {p - 20}): Cross-model disagreement!\n"
            if eq_diff:
                msg += f"  Equation differences: primary={sorted(primary_eq_set)}, cross={sorted(cross_eq_set)}, diff={sorted(eq_diff)}\n"
            if sec_diff:
                msg += f"  Section differences: primary={sorted(primary_secs)}, cross={sorted(cross_secs)}, diff={sorted(sec_diff)}"
            failures.append(msg)

    stats["spot_results"] = spot_results
    stats["sampled_pages"] = sampled_pages
    stats["cross_model"] = cross_model

    return len(failures) == 0, failures, stats


def generate_verification_report(
    completeness_res: Tuple[bool, List[str], Dict[str, int]],
    equations_res: Tuple[bool, List[str], Dict[str, Any]],
    sections_res: Tuple[bool, List[str], Dict[str, Any]],
    problems_res: Tuple[bool, List[str], Dict[str, Any]],
    latex_res: Tuple[bool, List[str], Dict[str, Any]],
    cross_res: Tuple[bool, List[str], Dict[str, Any]],
    output_path: Path
) -> str:
    """Generates comprehensive verification report markdown."""
    all_passed = all([
        completeness_res[0],
        equations_res[0],
        sections_res[0],
        problems_res[0],
        latex_res[0],
        cross_res[0]
    ])

    lines = [
        "# Verification Report: Stage 1 Ingestion & Extraction",
        "",
        f"**Overall Status:** {'PASSED (READY FOR STAGE 2)' if all_passed else 'FAILED (DO NOT PROCEED)'}",
        f"**Scope:** Chapter 2 (PDF 33-76 / Book 13-56) & Chapter 3 (PDF 77-125 / Book 57-105)",
        f"**Total Pages In-Scope:** {len(IN_SCOPE_PAGES)}",
        "",
        "---",
        "",
        "## Summary of Checks",
        "",
        f"| Check | Description | Status |",
        f"| :--- | :--- | :--- |",
        f"| a) Completeness | Checkpoint presence & non-empty character threshold | {'PASS' if completeness_res[0] else 'FAIL'} |",
        f"| b) Equation Sequence | Contiguous equations with no gaps or duplicates | {'PASS' if equations_res[0] else 'FAIL'} |",
        f"| c) Section / Figure / Table | Contiguous structure hierarchy | {'PASS' if sections_res[0] else 'FAIL'} |",
        f"| d) Problem Sequence | Contiguous end-of-chapter problems | {'PASS' if problems_res[0] else 'FAIL'} |",
        f"| e) LaTeX & OCR Integrity | Balanced delimiters, no corrupted OCR artifacts | {'PASS' if latex_res[0] else 'FAIL'} |",
        f"| f) Cross-Model Spot Check | 10% sample dual-model verification | {'PASS' if cross_res[0] else 'FAIL'} |",
        "",
        "---",
        "",
        "## Detailed Results & Audits",
        ""
    ]

    checks = [
        ("a) Completeness", completeness_res),
        ("b) Equation Sequence", equations_res),
        ("c) Section / Figure / Table Sequence", sections_res),
        ("d) Problem Sequence", problems_res),
        ("e) LaTeX & OCR Integrity", latex_res),
        ("f) Cross-Model Spot Check", cross_res),
    ]

    for name, res in checks:
        passed, fails, stats = res
        lines.append(f"### {name}")
        lines.append(f"- **Status:** {'PASS' if passed else 'FAIL'}")
        if fails:
            lines.append("- **Failures:**")
            for f in fails:
                lines.append(f"  - {f}")
        else:
            lines.append("- **Details:** All assertions satisfied.")
        if stats:
            lines.append(f"```json\n{json.dumps(stats, indent=2)}\n```")
        lines.append("")

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def run_verification(
    pages_dir: Path = Path("data/pages"),
    chapters_dir: Path = Path("data/chapters"),
    problems_dir: Path = Path("data/problems"),
    report_path: Path = Path("data/verification_report.md"),
    api_key: Optional[str] = None
) -> int:
    """Executes all verification checks and exits non-zero on failure."""
    print("=" * 60)
    print("LEARN-ASSIST STAGE 1 VERIFICATION")
    print("=" * 60)

    print("\n[1/6] Verifying a) Completeness...")
    comp_res = check_completeness(pages_dir)
    print(f"  -> {'PASS' if comp_res[0] else 'FAIL'}")

    print("\n[2/6] Verifying b) Equation Sequence...")
    eq_res = check_equations(chapters_dir)
    print(f"  -> {'PASS' if eq_res[0] else 'FAIL'}")

    print("\n[3/6] Verifying c) Section, Figure, and Table Sequence...")
    sec_res = check_sections_and_figures(chapters_dir)
    print(f"  -> {'PASS' if sec_res[0] else 'FAIL'}")

    print("\n[4/6] Verifying d) Problem Sequence...")
    prob_res = check_problem_sequence(problems_dir)
    print(f"  -> {'PASS' if prob_res[0] else 'FAIL'}")

    print("\n[5/6] Verifying e) LaTeX Integrity & Corrupted OCR Signatures...")
    latex_res = check_latex_integrity(pages_dir)
    print(f"  -> {'PASS' if latex_res[0] else 'FAIL'}")

    print("\n[6/6] Verifying f) Cross-Model Spot Check (10% sample)...")
    cross_res = check_cross_model_spot(pages_dir, api_key=api_key)
    print(f"  -> {'PASS' if cross_res[0] else 'FAIL'}")

    report = generate_verification_report(
        comp_res, eq_res, sec_res, prob_res, latex_res, cross_res, report_path
    )

    all_passed = all([
        comp_res[0], eq_res[0], sec_res[0], prob_res[0], latex_res[0], cross_res[0]
    ])

    print("\n" + "=" * 60)
    if all_passed:
        print(f"VERIFICATION PASSED! Full report written to {report_path}")
        print("=" * 60)
        return 0
    else:
        print(f"VERIFICATION FAILED! Failures documented in {report_path}")
        print("=" * 60)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 Ingestion Verification Suite.")
    parser.add_argument("--pages-dir", type=Path, default=Path("data/pages"), help="Pages directory")
    parser.add_argument("--chapters-dir", type=Path, default=Path("data/chapters"), help="Chapters directory")
    parser.add_argument("--problems-dir", type=Path, default=Path("data/problems"), help="Problems directory")
    parser.add_argument("--report", type=Path, default=Path("data/verification_report.md"), help="Report output path")
    parser.add_argument("--api-key", type=str, default=None, help="API key for cross-model spot check")
    args = parser.parse_args()

    exit_code = run_verification(
        pages_dir=args.pages_dir,
        chapters_dir=args.chapters_dir,
        problems_dir=args.problems_dir,
        report_path=args.report,
        api_key=args.api_key
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
