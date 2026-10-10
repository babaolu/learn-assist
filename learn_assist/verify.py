#!/usr/bin/env python3
"""
learn_assist.verify

Verification suite for Stage 1 textbook ingestion and extraction:
a) COMPLETENESS: Every page in range has a non-empty checkpoint >= min character threshold.
b) EQUATION SEQUENCE: (2.1)..(2.141) and (3.1)..(3.154) contiguous with no gaps and no duplicates,
   with letter suffixes starting from 'a' with no gaps, and no base appearing both bare and lettered.
c) SECTION & FIGURE/TABLE SEQUENCE: Section numbers, subsections, Figure numbers, and Table numbers contiguous.
d) PROBLEM SEQUENCE: Chapter end-of-chapter problems contiguous with no gaps and no duplicates.
e) LATEX & OCR INTEGRITY: Balanced $ and $$, no corrupted OCR signatures, no mid-word starts.
f) PDF TEXT LAYER 100% COVERAGE: Independent comparison of equation labels from the PDF OCR text layer
   versus transcription across all 93 in-scope pages.
g) CROSS-MODEL SPOT CHECK: Dual-model verification on densest and random pages (ranked by PDF text layer),
   classifying outcomes as AGREE, DISAGREE, or UNVERIFIED, with 180s+ timeout and exponential backoff.
h) SUB-EQUATION SWEEP: Full scan of all pages for equation jumps > 1 or bare labels where lettered exist.

Outputs full audit log to data/verification_report.md.
EXITS NON-ZERO IF ANY CHECK FAILS.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import json
import time
import base64
import random
import argparse
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict

import requests
import pymupdf

from learn_assist.ingest import (
    DEFAULT_API_KEY,
    clean_markdown_fences,
    find_source_pdf,
    TRANSCRIPTION_PROMPT,
)

MIN_CHAR_THRESHOLD = 150
IN_SCOPE_PAGES = list(range(33, 126))  # PDF 33 to 125 inclusive (93 pages)
CH2_PAGES = list(range(33, 77))        # PDF 33 to 76 (Book 13 to 56)
CH3_PAGES = list(range(77, 126))       # PDF 77 to 125 (Book 57 to 105)

# Standard sample of 9 pages (dense and random)
DEFAULT_SPOT_SAMPLE = [36, 46, 47, 50, 61, 64, 68, 114, 119]

CORRUPT_OCR_PATTERNS = [
    r'\(f>',
    r'\bfi2\b',
    r'</>',
    r'<\s*t>\'?',
    r'\\\(t>\'?',
    r'ysincf',
    r'\{R2p\}',
]


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


def extract_equation_labels(text: str, ch_num: int) -> List[Tuple[int, str, str, int, str]]:
    """
    Extracts all defined equation labels for a chapter from Markdown text.
    Returns list of tuples: (base_num, letter_suffix, full_tag, line_no, raw_line).
    Handles:
      (2.79a), (2.80b), (2.79), (2.79\\text{a}), (2.79\\mathrm{a}), \\tag{2.79a}, etc.
    Filters out in-text citations such as Eq. (2.9), Eqs. (2.44), Equation (2.101).
    """
    labels = []
    lines = text.splitlines()

    # Matches equation labels like:
    # \tag{2.79a}, \tag{2.79\text{a}}, (2.79a), (2.79\text{a}), (2.79)
    raw_pattern = re.compile(
        rf'(?:\\tag\*?\{{|\(\s*){ch_num}\.(\d+)(?:\\(?:text|mathrm|mathbf)\{{([a-zA-Z])\}}|\s*([a-zA-Z]))?\s*(?:\}}|\))'
    )

    in_math_block = False

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            line_in_math = True
        elif "$$" in stripped:
            count = stripped.count("$$")
            if count % 2 == 1:
                in_math_block = not in_math_block
                line_in_math = True
            else:
                line_in_math = in_math_block or True
        else:
            line_in_math = in_math_block

        for m in raw_pattern.finditer(line):
            prefix = line[:m.start()].rstrip()
            suffix = line[m.end():].lstrip()

            # Filter citations
            if re.search(r'\b(?:Eqs?\.?|Equations?\.?|see|from|in|and|or|ref|with|to|by|into|of)\s*\(?$', prefix, re.IGNORECASE):
                continue
            if re.search(r'\b(?:between|similarly\s+to|as\s+in|according\s+to)\b.*$', prefix, re.IGNORECASE):
                continue

            is_tag = r"\tag" in m.group(0) or prefix.endswith(r"\tag")
            has_math_spacing = bool(re.search(r'(?:\\(?:quad|qquad|hfill|hspace\{[^}]+\})|&)\s*$', prefix))
            is_display_math = line_in_math or has_math_spacing or is_tag

            if not is_display_math:
                continue

            num = int(m.group(1))
            letter = (m.group(2) or m.group(3) or "").lower()
            tag = f"{ch_num}.{num}{letter}"
            labels.append((num, letter, tag, line_no, line.strip()))

    return labels


def extract_equation_numbers(text: str, chapter_num: int) -> Tuple[List[int], List[str]]:
    """
    Extracts base integers and raw tags for backwards compatibility.
    """
    labels = extract_equation_labels(text, chapter_num)
    base_ints = [item[0] for item in labels]
    raw_tags = [item[2] for item in labels]
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

    # Check for duplicates
    counts: Dict[int, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    duplicates = [x for x, c in counts.items() if c > 1 and start_val <= x <= end_val]
    if duplicates:
        failures.append(f"{name}: Duplicate sequence numbers: {duplicates}")

    return len(failures) == 0, failures


def check_equation_sequence(
    labels: List[Tuple[int, str, str, int, str]],
    ch_num: int,
    expected_max: int
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Asserts:
    i.  base numbers run 1..expected_max with no gaps and no duplicate bare definitions
    ii. for any base number with letter suffixes, letters run from 'a' with no gaps,
        and that base must NOT also appear bare.
    """
    failures = []
    if not labels:
        return False, [f"Chapter {ch_num:02d} Equations: No equations found."], {}

    by_num = defaultdict(list)
    for num, letter, tag, line_no, raw_line in labels:
        by_num[num].append((letter, tag, line_no))

    unique_bases = sorted(by_num.keys())
    expected_bases = set(range(1, expected_max + 1))
    found_bases = set(unique_bases)

    # Check gaps in base numbers
    missing = sorted(expected_bases - found_bases)
    if missing:
        failures.append(f"Chapter {ch_num:02d} Equations: Missing base equation numbers (gaps): {missing}")

    extra = sorted(found_bases - expected_bases)
    if extra:
        failures.append(f"Chapter {ch_num:02d} Equations: Unexpected base numbers outside 1..{expected_max}: {extra}")

    # Check duplicate bare and letter contiguity
    for num, occurrences in sorted(by_num.items()):
        bare_occurrences = [tag for letter, tag, _ in occurrences if letter == ""]
        letter_occurrences = [letter for letter, tag, _ in occurrences if letter != ""]

        # If appears both bare and lettered -> FAILURE
        if bare_occurrences and letter_occurrences:
            failures.append(
                f"Chapter {ch_num:02d} Equation {ch_num}.{num}: Appears both bare and lettered (bare {bare_occurrences} and suffixes {sorted(set(letter_occurrences))})"
            )

        # Duplicate bare definitions
        if len(bare_occurrences) > 1:
            failures.append(
                f"Chapter {ch_num:02d} Equation {ch_num}.{num}: Duplicate bare definition ({len(bare_occurrences)} times)"
            )

        # Letter suffixes contiguity
        if letter_occurrences:
            unique_letters = sorted(set(letter_occurrences))
            expected_letters = [chr(ord('a') + i) for i in range(len(unique_letters))]
            if unique_letters != expected_letters:
                failures.append(
                    f"Chapter {ch_num:02d} Equation {ch_num}.{num}: Letter suffix gap or does not start at 'a' (found {unique_letters}, expected {expected_letters})"
                )
            if len(letter_occurrences) != len(unique_letters):
                failures.append(
                    f"Chapter {ch_num:02d} Equation {ch_num}.{num}: Duplicate letter suffix in {letter_occurrences}"
                )

    stats = {
        "total_labels": len(labels),
        "unique_base_equations": len(unique_bases),
        "min": unique_bases[0] if unique_bases else None,
        "max": unique_bases[-1] if unique_bases else None,
        "lettered_base_count": len([n for n, occ in by_num.items() if any(l != "" for l, _, _ in occ)])
    }
    return len(failures) == 0, failures, stats


def check_equations(chapters_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check b) Equation sequence: (2.1)..(2.141) and (3.1)..(3.154) contiguous with no gaps."""
    failures = []
    stats = {}

    for ch_num, expected_max in [(2, 141), (3, 154)]:
        ch_file = chapters_dir / f"chapter_{ch_num:02d}.md"
        if not ch_file.exists():
            failures.append(f"Chapter file {ch_file.name} missing for equation check.")
            continue
        text = ch_file.read_text(encoding="utf-8")
        labels = extract_equation_labels(text, ch_num)
        ch_passed, ch_fails, ch_stats = check_equation_sequence(labels, ch_num, expected_max)
        if not ch_passed:
            failures.extend(ch_fails)
        stats[f"chapter_{ch_num:02d}_equations"] = ch_stats

    return len(failures) == 0, failures, stats


def extract_section_headings(text: str, ch_num: int) -> List[str]:
    """
    Extracts section headings for a chapter.
    Matches headings strictly in heading positions (^#+ ... or list/quote heading indicators).
    Does NOT match numbers inside parentheses (e.g. (2.75)) or numbers in running text.
    Stops before exercises/problems.
    """
    sections = []
    seen = set()
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s*(?:Exercises|Problems)\b", stripped, re.IGNORECASE):
            break
        # Match markdown headings or backticked/itemized headings
        m = re.search(rf"(?:^|[`\s])#+\s*(?:Section\s+)?({ch_num}\.\d+(?:\.\d+)*)\b", stripped)
        if not m:
            m = re.match(rf"^[-*]\s*(?:Section\s+heading:\s*`?|Section\s+header:\s*`?|Section\s+)?#*\s*({ch_num}\.\d+(?:\.\d+)*)\b", stripped)
        if m:
            sec_str = m.group(1)
            if sec_str not in seen:
                seen.add(sec_str)
                sections.append(sec_str)
    return sections


def check_section_hierarchy(
    sections: List[str],
    ch_num: int,
    expected_level1_range: range
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Asserts:
    Level 1: ch.M runs contiguous 1..max without gaps.
    Level 2+: subsections under each section run contiguous from 1.
    """
    failures = []
    if not sections:
        return False, [f"Chapter {ch_num:02d} Sections: No sections found."], {}

    level1_ints = []
    subsections = defaultdict(set)

    for sec in sections:
        parts = [int(p) for p in sec.split(".")]
        if len(parts) >= 2:
            level1_ints.append(parts[1])
        if len(parts) >= 3:
            parent = f"{parts[0]}.{parts[1]}"
            subsections[parent].add(parts[2])

    # Check level 1 contiguity
    unique_l1 = sorted(set(level1_ints))
    expected_l1 = set(expected_level1_range)
    missing_l1 = sorted(expected_l1 - set(unique_l1))
    if missing_l1:
        failures.append(f"Chapter {ch_num:02d} Sections: Missing Level 1 sections: {missing_l1}")

    # Check subsection contiguity (Level 2)
    for parent, subs in sorted(subsections.items()):
        sorted_subs = sorted(subs)
        expected_subs = list(range(1, max(sorted_subs) + 1))
        if sorted_subs != expected_subs:
            missing_subs = sorted(set(expected_subs) - subs)
            failures.append(f"Chapter {ch_num:02d} Subsections under {parent}: Gaps detected (found {sorted_subs}, missing {missing_subs})")

    stats = {
        "sections_found": unique_l1,
        "all_headings": sections,
        "subsections_found": {k: sorted(v) for k, v in subsections.items()}
    }
    return len(failures) == 0, failures, stats


def check_sections_and_figures(chapters_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check c) Section sequence, Figure sequence, and Table sequence."""
    failures = []
    stats = {}

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

        # Sections check
        headings = extract_section_headings(text, ch_num)
        sec_passed, sec_fails, sec_stats = check_section_hierarchy(headings, ch_num, tgt["sections"])
        if not sec_passed:
            failures.extend(sec_fails)

        # Figures check
        fig_matches = re.findall(rf'Figure\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)
        fig_ints = sorted(set(int(x) for x in fig_matches))
        expected_figs = set(tgt["figures"])
        missing_figs = sorted(expected_figs - set(fig_ints))
        if missing_figs:
            failures.append(f"Chapter {ch_num:02d} Figures: Missing figures: {missing_figs}")

        # Tables check
        tbl_ints = []
        if tgt["tables"]:
            tbl_matches = re.findall(rf'Table\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)
            tbl_ints = sorted(set(int(x) for x in tbl_matches))
            expected_tbls = set(tgt["tables"])
            missing_tbls = sorted(expected_tbls - set(tbl_ints))
            if missing_tbls:
                failures.append(f"Chapter {ch_num:02d} Tables: Missing tables: {missing_tbls}")

        stats[f"chapter_{ch_num:02d}_structure"] = {
            "sections_found": sec_stats.get("sections_found", []),
            "all_headings": headings,
            "figures_found": fig_ints,
            "tables_found": tbl_ints
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
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            if first_line.startswith("-") and len(first_line) > 1 and first_line[1].isalpha():
                failures.append(f"Page {p} (Book p. {p - 20}): Begins mid-word with hyphenation: '{first_line[:40]}'")

    return len(failures) == 0, failures, stats


def extract_pdf_text_labels(page_text: str, ch_num: int) -> List[str]:
    """
    Extracts equation labels of the form (ch_num.M) or (ch_num.Ma) from the PDF text layer.
    Filters out in-text citations like Eq. (2.9), Eqs. (2.44), etc.
    """
    labels = []
    lines = page_text.splitlines()
    for line in lines:
        line_s = line.strip()
        for m in re.finditer(rf'\({ch_num}\.(\d+[a-z]?)\)', line_s):
            prefix = line_s[:m.start()].strip()
            suffix = line_s[m.end():].strip()
            # Ignore citations in prose: preceded by Eq., Eqs., Equation, in, see, from, and, or, of
            if re.search(r'\b(?:Eqs?\.?|ofEq\.?|Equations?\.?|see|from|in|and|or|between|with|to)\b', prefix, re.IGNORECASE):
                continue
            if any(w in prefix.lower() for w in ['defined', 'given', 'using', 'as shown', 'recall']):
                continue
            # If line has words after the equation tag, it's running text unless punctuation
            if re.search(r'^[a-zA-Z]{2,}', suffix):
                continue
            labels.append(f"{ch_num}.{m.group(1).lower()}")
    return sorted(set(labels))


def check_pdf_text_layer_coverage(
    pdf_path: Optional[Path] = None,
    pages_dir: Path = Path("data/pages"),
    in_scope_pages: List[int] = IN_SCOPE_PAGES
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check f) INDEPENDENT LABEL COUNT FROM THE PDF TEXT LAYER — 100% coverage:
    For every in-scope page:
    - extracts equation labels from the PDF text layer for that page
    - extracts equation labels from the transcription for that page
    - compares the two sets and flags any page where they differ
    Treats label present in text layer but absent from transcription as FAILURE.
    Treats label present in transcription but absent from text layer as WARNING.
    """
    source_pdf = pdf_path or find_source_pdf(Path("book"))
    if not source_pdf or not source_pdf.exists():
        return False, [f"PDF source file not found at {source_pdf}"], {}

    doc = pymupdf.open(source_pdf)
    failures = []
    warnings = []
    per_page_results = []
    failed_pages = []
    warning_pages = []
    clean_pages = []

    for p in in_scope_pages:
        ch = 2 if p in CH2_PAGES else 3
        pdf_page = doc[p - 1]
        pdf_text = pdf_page.get_text()
        pdf_lbls = extract_pdf_text_labels(pdf_text, ch)

        trans_file = pages_dir / f"page_{p:04d}.md"
        trans_text = trans_file.read_text(encoding="utf-8") if trans_file.exists() else ""
        trans_lbls = sorted(set(item[2] for item in extract_equation_labels(trans_text, ch)))

        missing_in_trans = sorted(set(pdf_lbls) - set(trans_lbls))
        missing_in_pdf = sorted(set(trans_lbls) - set(pdf_lbls))

        if missing_in_trans:
            status = "FAIL"
            failed_pages.append(p)
            failures.append(
                f"Page {p} (Book p. {p - 20}): Label(s) present in PDF text layer but MISSING from transcription: {missing_in_trans}"
            )
        elif missing_in_pdf:
            status = "WARN"
            warning_pages.append(p)
            warnings.append(
                f"Page {p} (Book p. {p - 20}): Label(s) present in transcription but absent from lossy PDF text layer: {missing_in_pdf}"
            )
        else:
            status = "PASS"
            clean_pages.append(p)

        per_page_results.append({
            "page": p,
            "book_page": p - 20,
            "pdf_labels": pdf_lbls,
            "trans_labels": trans_lbls,
            "missing_in_trans": missing_in_trans,
            "missing_in_pdf": missing_in_pdf,
            "status": status
        })

    stats = {
        "total_pages": len(in_scope_pages),
        "clean_pages_count": len(clean_pages),
        "failed_pages_count": len(failed_pages),
        "warning_pages_count": len(warning_pages),
        "failed_pages": failed_pages,
        "warning_pages": warning_pages,
        "warnings": warnings,
        "per_page_results": per_page_results
    }

    return len(failures) == 0, failures, stats


def check_sub_equation_sweep(
    pages_dir: Path,
    in_scope_pages: List[int] = IN_SCOPE_PAGES
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check h) SUB-EQUATION SWEEP:
    Independent of sampling, scans every in-scope page's transcription for:
    1. Pages where the equation labels jump by more than one (within page or across adjacent pages).
    2. Any label appearing bare that appears lettered elsewhere.
    Flags all offending pages.
    """
    failures = []
    flagged_pages = set()
    page_labels: Dict[int, List[Tuple[int, str, str, int, str]]] = {}

    for p in in_scope_pages:
        page_file = pages_dir / f"page_{p:04d}.md"
        if not page_file.exists():
            continue
        ch = 2 if p in CH2_PAGES else 3
        text = page_file.read_text(encoding="utf-8")
        page_labels[p] = extract_equation_labels(text, ch)

    # 1. Collect all base numbers that appear with letter suffixes anywhere
    lettered_bases: Dict[int, Set[int]] = {2: set(), 3: set()}
    for p, lbls in page_labels.items():
        ch = 2 if p in CH2_PAGES else 3
        for num, letter, tag, line_no, raw in lbls:
            if letter:
                lettered_bases[ch].add(num)

    # 2. Check for bare appearances of bases that appear lettered elsewhere
    for p, lbls in page_labels.items():
        ch = 2 if p in CH2_PAGES else 3
        for num, letter, tag, line_no, raw in lbls:
            if not letter and num in lettered_bases[ch]:
                flagged_pages.add(p)
                failures.append(
                    f"Page {p} (Book p. {p - 20}): Equation {tag} appears bare, but base {ch}.{num} appears with letter suffixes elsewhere."
                )

    # 3. Check for internal page jumps > 1
    for p, lbls in page_labels.items():
        ch = 2 if p in CH2_PAGES else 3
        if len(lbls) < 2:
            continue
        base_seq = [item[0] for item in lbls]
        for i in range(len(base_seq) - 1):
            prev = base_seq[i]
            curr = base_seq[i + 1]
            if curr - prev > 1:
                flagged_pages.add(p)
                failures.append(
                    f"Page {p} (Book p. {p - 20}): Equation sequence jumps internally on page by {curr - prev} from {ch}.{prev} to {ch}.{curr}."
                )
            elif curr < prev:
                flagged_pages.add(p)
                failures.append(
                    f"Page {p} (Book p. {p - 20}): Equation sequence goes backwards internally on page from {ch}.{prev} to {ch}.{curr}."
                )

    # 4. Check for cross-page jumps > 1 between adjacent pages in same chapter
    last_p = None
    last_base = None
    last_ch = None
    for p in sorted(page_labels.keys()):
        lbls = page_labels[p]
        if not lbls:
            continue
        ch = 2 if p in CH2_PAGES else 3
        first_base = lbls[0][0]
        if last_ch == ch and last_base is not None:
            if first_base - last_base > 1:
                flagged_pages.add(last_p)
                flagged_pages.add(p)
                failures.append(
                    f"Page {p} (Book p. {p - 20}): Cross-page equation jump by {first_base - last_base} from {ch}.{last_base} (Page {last_p}) to {ch}.{first_base}."
                )
        last_p = p
        last_base = lbls[-1][0]
        last_ch = ch

    stats = {
        "pages_scanned": len(page_labels),
        "flagged_pages_count": len(flagged_pages),
        "flagged_pages": sorted(flagged_pages),
        "failures_count": len(failures)
    }
    return len(failures) == 0, failures, stats


def select_stratified_sample(
    pages_dir: Path,
    pdf_path: Optional[Path] = None,
    in_scope_pages: List[int] = IN_SCOPE_PAGES,
    sample_ratio: float = 0.10,
    dense_ratio: float = 0.50,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Selects sample of in-scope pages stratified by equation count:
    - Ranks in-scope pages by TEXT-LAYER label count (to prevent circular bias)
    - Samples the densest pages preferentially
    - Samples random pages for an unbiased signal
    Returns (dense_pages, random_pages).
    """
    source_pdf = pdf_path or find_source_pdf(Path("book"))
    page_eq_counts = []

    if source_pdf and source_pdf.exists():
        doc = pymupdf.open(source_pdf)
        for p in in_scope_pages:
            ch = 2 if p in CH2_PAGES else 3
            pdf_text = doc[p - 1].get_text()
            count = len(extract_pdf_text_labels(pdf_text, ch))
            page_eq_counts.append((p, count))
    else:
        # Fallback to transcription count if PDF unavailable
        for p in in_scope_pages:
            page_file = pages_dir / f"page_{p:04d}.md"
            ch = 2 if p in CH2_PAGES else 3
            count = 0
            if page_file.exists():
                text = page_file.read_text(encoding="utf-8")
                count = len(extract_equation_labels(text, ch))
            page_eq_counts.append((p, count))

    # Rank by count desc, ties broken by page asc
    page_eq_counts.sort(key=lambda x: (x[1], -x[0]), reverse=True)

    total_sample = max(1, int(round(len(in_scope_pages) * sample_ratio)))
    dense_count = max(1, int(round(total_sample * dense_ratio)))
    random_count = total_sample - dense_count

    dense_pages = sorted([p for p, c in page_eq_counts[:dense_count]])
    remaining_pages = [p for p, c in page_eq_counts[dense_count:]]

    rng = random.Random(seed)
    random_pages = sorted(rng.sample(remaining_pages, random_count))

    return dense_pages, random_pages


def transcribe_image_with_retry(
    img_path: Path,
    api_key: str,
    model: str = "gemma-4-31b-it",
    timeout: int = 180,
    max_retries: int = 5,
    initial_delay: float = 3.0,
    backoff_factor: float = 2.0
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Transcribes page image with exponential backoff and timeout >= 180s.
    Returns (transcription_text, status, failure_reason)
    Status is 'VERIFIED' (success) or 'UNVERIFIED' (timeout/error).
    """
    if not img_path.exists():
        return None, "UNVERIFIED", f"Image {img_path.name} does not exist"

    with open(img_path, "rb") as f:
        img_bytes = f.read()
    b64_data = base64.b64encode(img_bytes).decode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": TRANSCRIPTION_PROMPT},
                {"inline_data": {"mime_type": "image/png", "data": b64_data}}
            ]
        }],
        "generationConfig": {"temperature": 0.0}
    }

    delay = initial_delay
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return clean_markdown_fences(parts[0].get("text", "")), "VERIFIED", None
                last_err = f"Empty response structure: {data}"
            elif resp.status_code in (429, 500, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                time.sleep(delay)
                delay *= backoff_factor
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                break
        except requests.exceptions.Timeout:
            last_err = f"Read timeout (>{timeout}s)"
            time.sleep(delay)
            delay *= backoff_factor
        except Exception as exc:
            last_err = f"Exception: {exc}"
            time.sleep(delay)
            delay *= backoff_factor

    return None, "UNVERIFIED", f"Failed after {max_retries} attempts: {last_err}"


def check_cross_model_spot(
    pages_dir: Path,
    sample_ratio: float = 0.10,
    seed: int = 42,
    api_key: Optional[str] = None,
    cross_model: str = "gemma-4-31b-it",
    retranscribe: bool = False,
    sampled_pages: Optional[List[int]] = None,
    cross_dir: Path = Path("data/cross_model"),
    pdf_path: Optional[Path] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check g) Cross-Model Spot Check:
    - Classifies three outcomes separately: AGREE, DISAGREE, UNVERIFIED.
    - A timeout is recorded as UNVERIFIED (never DISAGREE or AGREE).
    - Any page left UNVERIFIED is a coverage failure.
    - Reports achieved coverage percentage vs target.
    - Evaluates dense and random groups separately, ranked by PDF text layer.
    """
    failures = []
    cross_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path("data/page_images")

    source_pdf = pdf_path or find_source_pdf(Path("book"))
    pdf_doc = pymupdf.open(source_pdf) if (source_pdf and source_pdf.exists()) else None

    # Determine sample
    if sampled_pages is not None:
        all_sampled = sorted(sampled_pages)
        # Classify by equation density using PDF text layer count
        page_counts = []
        for p in all_sampled:
            ch = 2 if p in CH2_PAGES else 3
            if pdf_doc:
                cnt = len(extract_pdf_text_labels(pdf_doc[p - 1].get_text(), ch))
            else:
                pf = pages_dir / f"page_{p:04d}.md"
                cnt = len(extract_equation_labels(pf.read_text(), ch)) if pf.exists() else 0
            page_counts.append((p, cnt))
        page_counts.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        dense_pages = sorted([p for p, c in page_counts[:len(all_sampled)//2]])
        random_pages = sorted([p for p in all_sampled if p not in dense_pages])
    elif not retranscribe:
        all_sampled = sorted(DEFAULT_SPOT_SAMPLE)
        # Classify default sample by text-layer density:
        # Pages 46 (8), 119 (8), 50 (7), 36 (6), 61 (4) -> dense
        # Pages 64 (2), 47 (0), 68 (0), 114 (0) -> random
        dense_pages = [36, 46, 50, 61, 119]
        random_pages = [47, 64, 68, 114]
    else:
        dense_pages, random_pages = select_stratified_sample(
            pages_dir, pdf_path=source_pdf, in_scope_pages=IN_SCOPE_PAGES, sample_ratio=sample_ratio, seed=seed
        )
        all_sampled = sorted(dense_pages + random_pages)

    target_coverage_pct = (len(all_sampled) / len(IN_SCOPE_PAGES)) * 100.0

    print(f"\n--- Cross-Model Spot Check: {len(all_sampled)} pages total (Dense: {dense_pages}, Random: {random_pages}) ---")

    spot_results = []
    agreed_count = 0
    disagreed_count = 0
    unverified_count = 0

    for p in all_sampled:
        group = "dense" if p in dense_pages else "random"
        page_key = f"page_{p:04d}"
        primary_file = pages_dir / f"{page_key}.md"
        cross_file = cross_dir / f"{page_key}_cross.md"
        img_path = images_dir / f"{page_key}.png"

        if not primary_file.exists():
            failures.append(f"Page {p}: Primary transcription missing for spot check.")
            spot_results.append({
                "page": p,
                "book_page": p - 20,
                "group": group,
                "status": "UNVERIFIED",
                "reason": "Primary transcription missing"
            })
            unverified_count += 1
            continue

        primary_text = primary_file.read_text(encoding="utf-8")
        ch_num = 2 if p in CH2_PAGES else 3

        # Check cached cross-transcription or retranscribe
        cross_text = None
        unverified_reason = None
        if cross_file.exists() and len(cross_file.read_text(encoding="utf-8").strip()) > 0:
            cross_text = cross_file.read_text(encoding="utf-8")
        elif retranscribe:
            key = api_key or DEFAULT_API_KEY
            if not key:
                unverified_reason = "No API key available for re-transcription"
            else:
                cross_text, status, err = transcribe_image_with_retry(img_path, key, model=cross_model)
                if cross_text:
                    cross_file.write_text(cross_text, encoding="utf-8")
                else:
                    unverified_reason = err
        else:
            unverified_reason = "Cross-transcription not cached / timed out in previous run"

        if cross_text is None:
            spot_results.append({
                "page": p,
                "book_page": p - 20,
                "group": group,
                "status": "UNVERIFIED",
                "reason": unverified_reason
            })
            unverified_count += 1
            continue

        # Both transcriptions present: compare equations and section headings
        primary_eq_labels = extract_equation_labels(primary_text, ch_num)
        cross_eq_labels = extract_equation_labels(cross_text, ch_num)
        primary_eqs = sorted(set(item[2] for item in primary_eq_labels))
        cross_eqs = sorted(set(item[2] for item in cross_eq_labels))

        primary_secs = sorted(set(extract_section_headings(primary_text, ch_num)))
        cross_secs = sorted(set(extract_section_headings(cross_text, ch_num)))

        eq_diff = sorted(set(primary_eqs).symmetric_difference(set(cross_eqs)))
        sec_diff = sorted(set(primary_secs).symmetric_difference(set(cross_secs)))

        if not eq_diff and not sec_diff:
            status = "AGREE"
            agreed_count += 1
        else:
            status = "DISAGREE"
            disagreed_count += 1
            msg = f"Page {p} (Book p. {p - 20}): Cross-model disagreement!\n"
            if eq_diff:
                msg += f"  Equation differences: primary={primary_eqs}, cross={cross_eqs}, diff={eq_diff}\n"
            if sec_diff:
                msg += f"  Section differences: primary={primary_secs}, cross={cross_secs}, diff={sec_diff}"
            failures.append(msg)

        spot_results.append({
            "page": p,
            "book_page": p - 20,
            "group": group,
            "status": status,
            "primary_eqs": primary_eqs,
            "cross_eqs": cross_eqs,
            "primary_secs": primary_secs,
            "cross_secs": cross_secs,
            "eq_diff": eq_diff,
            "sec_diff": sec_diff
        })

    # Coverage calculations
    verified_count = agreed_count + disagreed_count
    achieved_coverage_pct = (verified_count / len(IN_SCOPE_PAGES)) * 100.0

    if unverified_count > 0:
        unverified_pages = [r["page"] for r in spot_results if r["status"] == "UNVERIFIED"]
        failures.append(
            f"Cross-model check: Coverage failure. Achieved {achieved_coverage_pct:.1f}% ({verified_count}/{len(IN_SCOPE_PAGES)} pages) vs target {target_coverage_pct:.1f}%. "
            f"{unverified_count} page(s) left UNVERIFIED: {unverified_pages}"
        )

    stats = {
        "sampled_pages": all_sampled,
        "dense_group": dense_pages,
        "random_group": random_pages,
        "target_coverage_pct": round(target_coverage_pct, 1),
        "achieved_coverage_pct": round(achieved_coverage_pct, 1),
        "agreed_count": agreed_count,
        "disagreed_count": disagreed_count,
        "unverified_count": unverified_count,
        "results": spot_results
    }

    return (len(failures) == 0 and unverified_count == 0), failures, stats


def generate_verification_report(
    completeness_res: Tuple[bool, List[str], Dict[str, int]],
    equations_res: Tuple[bool, List[str], Dict[str, Any]],
    sections_res: Tuple[bool, List[str], Dict[str, Any]],
    problems_res: Tuple[bool, List[str], Dict[str, Any]],
    latex_res: Tuple[bool, List[str], Dict[str, Any]],
    pdf_cov_res: Tuple[bool, List[str], Dict[str, Any]],
    cross_res: Tuple[bool, List[str], Dict[str, Any]],
    sweep_res: Tuple[bool, List[str], Dict[str, Any]],
    output_path: Path
) -> str:
    """Generates comprehensive verification report markdown."""
    all_passed = all([
        completeness_res[0],
        equations_res[0],
        sections_res[0],
        problems_res[0],
        latex_res[0],
        pdf_cov_res[0],
        cross_res[0],
        sweep_res[0]
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
        f"| b) Equation Sequence | Contiguous equations with no gaps, duplicates, or bare/letter conflicts | {'PASS' if equations_res[0] else 'FAIL'} |",
        f"| c) Section / Figure / Table | Contiguous structure hierarchy (sections & subsections) | {'PASS' if sections_res[0] else 'FAIL'} |",
        f"| d) Problem Sequence | Contiguous end-of-chapter problems | {'PASS' if problems_res[0] else 'FAIL'} |",
        f"| e) LaTeX & OCR Integrity | Balanced delimiters, no corrupted OCR artifacts | {'PASS' if latex_res[0] else 'FAIL'} |",
        f"| f) PDF Text Layer 100% Coverage | Full independent comparison of PDF text layer vs transcription | {'PASS' if pdf_cov_res[0] else 'FAIL'} |",
        f"| g) Cross-Model Spot Check | Dual-model verification (AGREE / DISAGREE / UNVERIFIED, coverage %) | {'PASS' if cross_res[0] else 'FAIL'} |",
        f"| h) Sub-Equation Sweep | Full scan for equation jumps > 1 and bare/lettered conflicts | {'PASS' if sweep_res[0] else 'FAIL'} |",
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
        ("f) PDF Text Layer 100% Coverage", pdf_cov_res),
        ("g) Cross-Model Spot Check", cross_res),
        ("h) Sub-Equation Sweep", sweep_res),
    ]

    for name, res in checks:
        passed, fails, stats = res
        lines.append(f"### {name}")
        lines.append(f"- **Status:** {'PASS' if passed else 'FAIL'}")
        if fails:
            lines.append("- **Failures:**")
            for f in fails:
                for sub_line in f.splitlines():
                    lines.append(f"  - {sub_line}" if not sub_line.startswith("  ") else sub_line)
        else:
            lines.append("- **Details:** All assertions satisfied.")

        # Special rendering for Check f (PDF Text Layer Coverage) table
        if name == "f) PDF Text Layer 100% Coverage" and "per_page_results" in stats:
            lines.append("")
            lines.append(f"- **Total Pages Audited:** {stats['total_pages']}")
            lines.append(f"- **Clean Matches (Exact Match):** {stats['clean_pages_count']}")
            lines.append(f"- **Pages with Failures (In PDF, Missing in Transcription):** {stats['failed_pages_count']} ({stats['failed_pages']})")
            lines.append(f"- **Pages with Warnings (In Transcription, Absent from Lossy Text Layer):** {stats['warning_pages_count']} ({stats['warning_pages']})")
            lines.append("")
            lines.append("#### Full Per-Page Comparison Table (93 Pages)")
            lines.append("")
            lines.append("| Page | Book Page | PDF Text Layer Labels | Transcription Labels | Missing in Trans (FAIL) | Missing in PDF (WARN) | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in stats["per_page_results"]:
                pdf_str = ", ".join(r["pdf_labels"]) if r["pdf_labels"] else "—"
                trans_str = ", ".join(r["trans_labels"]) if r["trans_labels"] else "—"
                fail_str = ", ".join(r["missing_in_trans"]) if r["missing_in_trans"] else "—"
                warn_str = ", ".join(r["missing_in_pdf"]) if r["missing_in_pdf"] else "—"
                lines.append(f"| {r['page']} | {r['book_page']} | {pdf_str} | {trans_str} | {fail_str} | {warn_str} | {r['status']} |")
            lines.append("")
        elif stats:
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
    pdf_path: Optional[Path] = None,
    report_path: Path = Path("data/verification_report.md"),
    api_key: Optional[str] = None,
    retranscribe: bool = False
) -> int:
    """Executes all verification checks and exits non-zero on failure."""
    print("=" * 60)
    print("LEARN-ASSIST STAGE 1 VERIFICATION")
    print("=" * 60)

    print("\n[1/8] Verifying a) Completeness...")
    comp_res = check_completeness(pages_dir)
    print(f"  -> {'PASS' if comp_res[0] else 'FAIL'}")

    print("\n[2/8] Verifying b) Equation Sequence...")
    eq_res = check_equations(chapters_dir)
    print(f"  -> {'PASS' if eq_res[0] else 'FAIL'}")

    print("\n[3/8] Verifying c) Section, Figure, and Table Sequence...")
    sec_res = check_sections_and_figures(chapters_dir)
    print(f"  -> {'PASS' if sec_res[0] else 'FAIL'}")

    print("\n[4/8] Verifying d) Problem Sequence...")
    prob_res = check_problem_sequence(problems_dir)
    print(f"  -> {'PASS' if prob_res[0] else 'FAIL'}")

    print("\n[5/8] Verifying e) LaTeX Integrity & Corrupted OCR Signatures...")
    latex_res = check_latex_integrity(pages_dir)
    print(f"  -> {'PASS' if latex_res[0] else 'FAIL'}")

    print("\n[6/8] Verifying f) PDF Text Layer 100% Coverage...")
    pdf_cov_res = check_pdf_text_layer_coverage(pdf_path=pdf_path, pages_dir=pages_dir)
    print(f"  -> {'PASS' if pdf_cov_res[0] else 'FAIL'}")

    print("\n[7/8] Verifying g) Cross-Model Spot Check...")
    cross_res = check_cross_model_spot(pages_dir, api_key=api_key, retranscribe=retranscribe, pdf_path=pdf_path)
    print(f"  -> {'PASS' if cross_res[0] else 'FAIL'}")

    print("\n[8/8] Verifying h) Sub-Equation Sweep (93 pages)...")
    sweep_res = check_sub_equation_sweep(pages_dir)
    print(f"  -> {'PASS' if sweep_res[0] else 'FAIL'}")

    report = generate_verification_report(
        comp_res, eq_res, sec_res, prob_res, latex_res, pdf_cov_res, cross_res, sweep_res, report_path
    )

    all_passed = all([
        comp_res[0], eq_res[0], sec_res[0], prob_res[0], latex_res[0], pdf_cov_res[0], cross_res[0], sweep_res[0]
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
    parser.add_argument("--pdf-path", type=Path, default=None, help="Source PDF file path")
    parser.add_argument("--report", type=Path, default=Path("data/verification_report.md"), help="Report output path")
    parser.add_argument("--api-key", type=str, default=None, help="API key for cross-model spot check")
    parser.add_argument("--retranscribe", action="store_true", default=False, help="Enable API re-transcription of uncached spot check pages")
    args = parser.parse_args()

    exit_code = run_verification(
        pages_dir=args.pages_dir,
        chapters_dir=args.chapters_dir,
        problems_dir=args.problems_dir,
        pdf_path=args.pdf_path,
        report_path=args.report,
        api_key=args.api_key,
        retranscribe=args.retranscribe
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
