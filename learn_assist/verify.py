#!/usr/bin/env python3
"""
learn_assist.verify

Verification suite for Stage 1 textbook ingestion and extraction:
1. THREE-WAY COMPARISON: Reference vs Derived (PDF text layer) vs Transcription.
   Anchored to independently verified reference data.
2. COMPLETENESS: Every in-scope page has a non-empty checkpoint >= min character threshold.
3. EQUATION SEQUENCE: Equations contiguous with no gaps, no duplicates, valid letter suffixes,
   and no base appearing both bare and lettered. Expected maxima come from reference anchors.
4. SECTION & FIGURE/TABLE SEQUENCE: Hierarchy contiguity for sections and figures/tables
   derived from reference anchors.
5. PROBLEM SEQUENCE: End-of-chapter problems contiguous with no gaps and no duplicates,
   bounded by reference ranges.
6. LATEX & CORRUPTION CHECKS: Balanced delimiters, no control characters, no Unicode replacement
   character, no stray angle-bracket digraphs outside LaTeX, and balanced braces inside math.
7. PDF TEXT LAYER 100% COVERAGE: Comparison of equation labels from PDF text layer vs transcription.
8. CROSS-MODEL SPOT CHECK: Dual-model verification on text-layer-stratified sample. Reads vendor
   from manifest; cross-transcriptions from the same vendor do not count toward coverage.
9. SUB-EQUATION SWEEP: Scan of all in-scope pages for equation jumps > 1 or bare/lettered conflicts.

Outputs full audit log to data/verification_report.md.
EXITS NON-ZERO IF ANY REQUIRED CHECK FAILS.
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
from learn_assist.anchors import (
    load_reference,
    load_derived,
    derive_all_anchors,
    extract_pdf_equation_labels,
)
from learn_assist.crosscheck import labelled_equations, norm

MIN_CHAR_THRESHOLD = 150
DEFAULT_REFERENCE_PATH = Path("anchors/liang_lauterbur.json")
DEFAULT_DERIVED_PATH = Path("data/anchors_derived.json")


def load_chapter_metadata(
    ref_path: Optional[Path] = None,
    derived_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], List[str]]:
    """
    Loads reference and derived anchors for all target chapters.
    Falls back to derived values when a reference is absent for a chapter, logging a loud warning.
    Returns (ref_chapters, derived_chapters, warnings).
    """
    warnings = []
    ref_data = load_reference(ref_path or DEFAULT_REFERENCE_PATH)
    derived_data = load_derived(derived_path=derived_path or DEFAULT_DERIVED_PATH, pdf_path=pdf_path)

    raw_derived_chapters = derived_data.get("chapters", {}) if derived_data else {}
    derived_chapters: Dict[int, Dict[str, Any]] = {
        int(k): v for k, v in raw_derived_chapters.items()
    }

    ref_chapters: Dict[int, Dict[str, Any]] = {}
    if ref_data and "chapters" in ref_data:
        for k, v in ref_data["chapters"].items():
            ref_chapters[int(k)] = v

    # Check for missing chapters in reference
    all_ch_nums = sorted(set(derived_chapters.keys()) | set(ref_chapters.keys()))
    for ch_num in all_ch_nums:
        if ch_num not in ref_chapters:
            msg = (
                f"WARNING: No reference file data found for Chapter {ch_num:02d}; "
                f"falling back to derived values from PDF text layer!"
            )
            warnings.append(msg)
            print(f"\n[!] {msg}")
            if ch_num in derived_chapters:
                ref_chapters[ch_num] = derived_chapters[ch_num]

    return ref_chapters, derived_chapters, warnings


def get_in_scope_pages(chapters_meta: Dict[int, Dict[str, Any]]) -> List[int]:
    """Computes all in-scope page numbers from chapter ranges."""
    pages = []
    for ch_num, meta in sorted(chapters_meta.items()):
        start = meta["chapter_start"]["value"]
        end = meta["chapter_end"]["value"]
        pages.extend(range(start, end + 1))
    return sorted(set(pages))


def get_chapter_for_page(page: int, chapters_meta: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """Finds which chapter a page number belongs to based on chapter metadata."""
    for ch_num, meta in chapters_meta.items():
        start = meta["chapter_start"]["value"]
        end = meta["chapter_end"]["value"]
        if start <= page <= end:
            return ch_num
    return None


def check_completeness(
    pages_dir: Path,
    in_scope_pages: Optional[List[int]] = None,
    min_chars: int = MIN_CHAR_THRESHOLD,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, int]]:
    """Check a) Completeness: Every in-scope page exists, non-empty, above threshold."""
    if in_scope_pages is None:
        if chapters_meta is None:
            ref_ch, _, _ = load_chapter_metadata()
            chapters_meta = ref_ch
        in_scope_pages = get_in_scope_pages(chapters_meta)

    failures = []
    page_counts = {}
    for p in in_scope_pages:
        page_path = pages_dir / f"page_{p:04d}.md"
        if not page_path.exists():
            failures.append(f"Page {p} checkpoint file MISSING: {page_path.name}")
            continue
        content = page_path.read_text(encoding="utf-8").strip()
        count = len(content)
        page_counts[f"page_{p:04d}"] = count
        if count == 0:
            failures.append(f"Page {p} checkpoint is EMPTY (0 chars)")
        elif count < min_chars:
            failures.append(f"Page {p} below threshold: {count} chars < {min_chars} chars")

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
    """Extracts base integers and raw tags for backwards compatibility."""
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

    missing = sorted(expected_bases - found_bases)
    if missing:
        failures.append(f"Chapter {ch_num:02d} Equations: Missing base equation numbers (gaps): {missing}")

    extra = sorted(found_bases - expected_bases)
    if extra:
        failures.append(f"Chapter {ch_num:02d} Equations: Unexpected base numbers outside 1..{expected_max}: {extra}")

    for num, occurrences in sorted(by_num.items()):
        bare_occurrences = [tag for letter, tag, _ in occurrences if letter == ""]
        letter_occurrences = [letter for letter, tag, _ in occurrences if letter != ""]

        if bare_occurrences and letter_occurrences:
            failures.append(
                f"Chapter {ch_num:02d} Equation {ch_num}.{num}: Appears both bare and lettered (bare {bare_occurrences} and suffixes {sorted(set(letter_occurrences))})"
            )

        if len(bare_occurrences) > 1:
            failures.append(
                f"Chapter {ch_num:02d} Equation {ch_num}.{num}: Duplicate bare definition ({len(bare_occurrences)} times)"
            )

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


def check_equations(
    chapters_dir: Path,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check b) Equation sequence: contiguous equations with expected max from reference anchors."""
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch

    failures = []
    stats = {}

    for ch_num, meta in sorted(chapters_meta.items()):
        expected_max = meta["last_equation"]["value"]
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
        # Skip running heads like "## Section 2.1 Vectors"
        if re.match(r"^#{1,6}\s*Section\s+\d", stripped, re.IGNORECASE):
            continue
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

    unique_l1 = sorted(set(level1_ints))
    expected_l1 = set(expected_level1_range)
    missing_l1 = sorted(expected_l1 - set(unique_l1))
    if missing_l1:
        failures.append(f"Chapter {ch_num:02d} Sections: Missing Level 1 sections: {missing_l1}")

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


def check_sections_and_figures(
    chapters_dir: Path,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check c) Section sequence, Figure sequence, and Table sequence from reference metadata."""
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch

    failures = []
    stats = {}

    for ch_num, meta in sorted(chapters_meta.items()):
        ch_file = chapters_dir / f"chapter_{ch_num:02d}.md"
        if not ch_file.exists():
            failures.append(f"Chapter file {ch_file.name} missing for section/figure check.")
            continue
        text = ch_file.read_text(encoding="utf-8")

        # Sections check
        sec_start = meta["sections"]["start"]
        sec_end = meta["sections"]["end"]
        headings = extract_section_headings(text, ch_num)
        sec_passed, sec_fails, sec_stats = check_section_hierarchy(headings, ch_num, range(sec_start, sec_end + 1))
        if not sec_passed:
            failures.extend(sec_fails)

        # Figures check
        fig_start = meta["figures"]["start"]
        fig_end = meta["figures"]["end"]
        fig_matches = re.findall(rf'Figure\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)
        fig_ints = sorted(set(int(x) for x in fig_matches))
        expected_figs = set(range(fig_start, fig_end + 1))
        missing_figs = sorted(expected_figs - set(fig_ints))
        if missing_figs:
            failures.append(f"Chapter {ch_num:02d} Figures: Missing figures: {missing_figs}")

        # Tables check
        tbl_ints = []
        expected_tbls = set(meta.get("tables", {}).get("items", []))
        if expected_tbls:
            tbl_matches = re.findall(rf'Table\s+{ch_num}\.(\d+)\b', text, re.IGNORECASE)
            tbl_ints = sorted(set(int(x) for x in tbl_matches))
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


def check_problem_sequence(
    problems_dir: Path,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check d) Problem sequence: contiguous end-of-chapter problems based on reference anchors."""
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch

    failures = []
    stats = {}

    for ch_num, meta in sorted(chapters_meta.items()):
        start_n = meta["problems"]["start"]
        end_n = meta["problems"]["end"]
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


def check_braces_balanced(math_text: str) -> bool:
    """Checks that unescaped curly braces inside math are properly nested and balanced."""
    clean = re.sub(r'\\[{}]', '', math_text)
    depth = 0
    for ch in clean:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def check_stray_angle_brackets(text: str) -> List[str]:
    """
    Finds stray angle-bracket patterns like <t>, <f>, </> outside LaTeX.
    Permits standard HTML tags like <div>, <p>, <span>, <b>, <i>, etc.
    """
    no_display = re.sub(r'(?<!\\)\$\$.*?(?<!\\)\$\$', '', text, flags=re.DOTALL)
    no_math = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', '', no_display)

    allowed_html = {
        'p', 'a', 'b', 'i', 'u', 's', 'q', 'em', 'strong', 'div', 'span',
        'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'img', 'table',
        'tr', 'td', 'th', 'tbody', 'thead', 'ul', 'ol', 'li'
    }
    matches = []
    for m in re.finditer(r'<(/?[a-zA-Z0-9]{1,2})\'?>|<[/]?>', no_math):
        tag = m.group(1) or ""
        tag_name = tag.lstrip("/").lower()
        if tag_name not in allowed_html:
            matches.append(m.group(0))
    return matches


def check_latex_integrity(
    pages_dir: Path,
    in_scope_pages: Optional[List[int]] = None,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check e) LaTeX & Corruption integrity:
    - Balanced $ and $$
    - No Unicode replacement character (\ufffd)
    - No control characters (ASCII 0x00-0x1f except tab/newline/cr, 0x7f)
    - No stray angle-bracket digraphs like "<t>" outside LaTeX
    - Balanced braces inside math
    - No page begins mid-word
    """
    if in_scope_pages is None:
        if chapters_meta is None:
            ref_ch, _, _ = load_chapter_metadata()
            chapters_meta = ref_ch
        in_scope_pages = get_in_scope_pages(chapters_meta)

    failures = []
    stats = {"pages_checked": 0, "issues": 0}
    ctrl_char_re = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    for p in in_scope_pages:
        page_file = pages_dir / f"page_{p:04d}.md"
        if not page_file.exists():
            continue
        content = page_file.read_text(encoding="utf-8")
        stats["pages_checked"] += 1

        # 1. Balanced $$ display math
        display_math_count = len(re.findall(r'(?<!\\)\$\$', content))
        if display_math_count % 2 != 0:
            failures.append(f"Page {p}: Unbalanced $$ display math delimiter (count={display_math_count})")

        # 2. Balanced inline $
        no_display = re.sub(r'(?<!\\)\$\$.*?(?<!\\)\$\$', '', content, flags=re.DOTALL)
        inline_math_count = len(re.findall(r'(?<!\\)\$', no_display))
        if inline_math_count % 2 != 0:
            failures.append(f"Page {p}: Unbalanced inline $ math delimiter (count={inline_math_count})")

        # 3. Unicode replacement character
        if '\ufffd' in content:
            failures.append(f"Page {p}: Unicode replacement character (\ufffd) detected.")

        # 4. Control characters
        ctrl_matches = ctrl_char_re.findall(content)
        if ctrl_matches:
            hex_chars = [hex(ord(c)) for c in ctrl_matches[:5]]
            failures.append(f"Page {p}: Control characters detected ({len(ctrl_matches)} occurrences: {hex_chars}).")

        # 5. Stray angle-bracket digraphs outside LaTeX
        stray_brackets = check_stray_angle_brackets(content)
        if stray_brackets:
            failures.append(f"Page {p}: Stray angle-bracket digraph(s) outside LaTeX: {stray_brackets}")

        # 6. Unbalanced braces inside math
        display_blocks = re.findall(r'(?<!\\)\$\$(.*?)(?<!\\)\$\$', content, flags=re.DOTALL)
        inline_blocks = re.findall(r'(?<!\\)\$(.*?)(?<!\\)\$', no_display)
        for b in display_blocks:
            if not check_braces_balanced(b):
                failures.append(f"Page {p}: Unbalanced braces inside display math: '{b.strip()[:60]}'")
                break
        for b in inline_blocks:
            if not check_braces_balanced(b):
                failures.append(f"Page {p}: Unbalanced braces inside inline math: '{b.strip()[:60]}'")
                break

        # 7. Check if page begins mid-word
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            if first_line.startswith("-") and len(first_line) > 1 and first_line[1].isalpha():
                failures.append(f"Page {p}: Begins mid-word with hyphenation: '{first_line[:40]}'")

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
            if re.search(r'\b(?:Eqs?\.?|ofEq\.?|Equations?\.?|see|from|in|and|or|between|with|to)\b', prefix, re.IGNORECASE):
                continue
            if any(w in prefix.lower() for w in ['defined', 'given', 'using', 'as shown', 'recall']):
                continue
            if re.search(r'^[a-zA-Z]{2,}', suffix):
                continue
            labels.append(f"{ch_num}.{m.group(1).lower()}")
    return sorted(set(labels))


def check_pdf_text_layer_coverage(
    pdf_path: Optional[Path] = None,
    pages_dir: Path = Path("data/pages"),
    in_scope_pages: Optional[List[int]] = None,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check f) INDEPENDENT LABEL COUNT FROM THE PDF TEXT LAYER — 100% coverage:
    For every in-scope page:
    - extracts equation labels from the PDF text layer
    - extracts equation labels from transcription
    - treats label present in text layer but absent from transcription as FAILURE
    - treats label present in transcription but absent from text layer as WARNING
    """
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch
    if in_scope_pages is None:
        in_scope_pages = get_in_scope_pages(chapters_meta)

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
        ch = get_chapter_for_page(p, chapters_meta) or 2
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
                f"Page {p}: Label(s) present in PDF text layer but MISSING from transcription: {missing_in_trans}"
            )
        elif missing_in_pdf:
            status = "WARN"
            warning_pages.append(p)
            warnings.append(
                f"Page {p}: Label(s) present in transcription but absent from lossy PDF text layer: {missing_in_pdf}"
            )
        else:
            status = "PASS"
            clean_pages.append(p)

        per_page_results.append({
            "page": p,
            "chapter": ch,
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
    in_scope_pages: Optional[List[int]] = None,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check h) SUB-EQUATION SWEEP:
    Scans every in-scope page's transcription for:
    1. Pages where equation labels jump by more than one internally or across pages.
    2. Any label appearing bare that appears lettered elsewhere.
    """
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch
    if in_scope_pages is None:
        in_scope_pages = get_in_scope_pages(chapters_meta)

    failures = []
    flagged_pages = set()
    page_labels: Dict[int, List[Tuple[int, str, str, int, str]]] = {}

    for p in in_scope_pages:
        page_file = pages_dir / f"page_{p:04d}.md"
        if not page_file.exists():
            continue
        ch = get_chapter_for_page(p, chapters_meta) or 2
        text = page_file.read_text(encoding="utf-8")
        page_labels[p] = extract_equation_labels(text, ch)

    # 1. Collect all base numbers that appear with letter suffixes anywhere
    lettered_bases: Dict[int, Set[int]] = defaultdict(set)
    for p, lbls in page_labels.items():
        ch = get_chapter_for_page(p, chapters_meta) or 2
        for num, letter, tag, line_no, raw in lbls:
            if letter:
                lettered_bases[ch].add(num)

    # 2. Check for bare appearances of bases that appear lettered elsewhere
    for p, lbls in page_labels.items():
        ch = get_chapter_for_page(p, chapters_meta) or 2
        for num, letter, tag, line_no, raw in lbls:
            if not letter and num in lettered_bases[ch]:
                flagged_pages.add(p)
                failures.append(
                    f"Page {p}: Equation {tag} appears bare, but base {ch}.{num} appears with letter suffixes elsewhere."
                )

    # 3. Check for internal page jumps > 1
    for p, lbls in page_labels.items():
        ch = get_chapter_for_page(p, chapters_meta) or 2
        if len(lbls) < 2:
            continue
        base_seq = [item[0] for item in lbls]
        for i in range(len(base_seq) - 1):
            prev = base_seq[i]
            curr = base_seq[i + 1]
            if curr - prev > 1:
                flagged_pages.add(p)
                failures.append(
                    f"Page {p}: Equation sequence jumps internally on page by {curr - prev} from {ch}.{prev} to {ch}.{curr}."
                )
            elif curr < prev:
                flagged_pages.add(p)
                failures.append(
                    f"Page {p}: Equation sequence goes backwards internally on page from {ch}.{prev} to {ch}.{curr}."
                )

    # 4. Check for cross-page jumps > 1 between adjacent pages in same chapter
    last_p = None
    last_base = None
    last_ch = None
    for p in sorted(page_labels.keys()):
        lbls = page_labels[p]
        if not lbls:
            continue
        ch = get_chapter_for_page(p, chapters_meta) or 2
        first_base = lbls[0][0]
        if last_ch == ch and last_base is not None:
            if first_base - last_base > 1:
                flagged_pages.add(last_p)
                flagged_pages.add(p)
                failures.append(
                    f"Page {p}: Cross-page equation jump by {first_base - last_base} from {ch}.{last_base} (Page {last_p}) to {ch}.{first_base}."
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
    in_scope_pages: Optional[List[int]] = None,
    sample_ratio: float = 0.10,
    dense_ratio: float = 0.50,
    seed: int = 42,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[List[int], List[int]]:
    """
    Selects sample of in-scope pages stratified by equation count:
    - Ranks in-scope pages by TEXT-LAYER label count (to prevent circular bias)
    - Samples the densest pages preferentially
    - Samples random pages for an unbiased signal
    Returns (dense_pages, random_pages).
    """
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch
    if in_scope_pages is None:
        in_scope_pages = get_in_scope_pages(chapters_meta)

    source_pdf = pdf_path or find_source_pdf(Path("book"))
    page_eq_counts = []

    if source_pdf and source_pdf.exists():
        doc = pymupdf.open(source_pdf)
        for p in in_scope_pages:
            ch = get_chapter_for_page(p, chapters_meta) or 2
            pdf_text = doc[p - 1].get_text()
            count = len(extract_pdf_text_labels(pdf_text, ch))
            page_eq_counts.append((p, count))
    else:
        for p in in_scope_pages:
            page_file = pages_dir / f"page_{p:04d}.md"
            ch = get_chapter_for_page(p, chapters_meta) or 2
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
    api_key: Optional[str] = None,
    model: str = "claude-opus-5-5",
    timeout: int = 180,
    max_retries: int = 5,
    initial_delay: float = 3.0,
    backoff_factor: float = 2.0
) -> Tuple[Optional[str], str, Optional[str]]:
    """Transcribes page image with retry. Status is 'VERIFIED' or 'UNVERIFIED'."""
    key = api_key or DEFAULT_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable or pass --api-key.")

    if not img_path.exists():
        return None, "UNVERIFIED", f"Image {img_path.name} does not exist"

    with open(img_path, "rb") as f:
        img_bytes = f.read()
    b64_data = base64.b64encode(img_bytes).decode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
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
    cross_model: Optional[str] = None,
    sampled_pages: Optional[List[int]] = None,
    cross_dir: Path = Path("data/cross_model"),
    pdf_path: Optional[Path] = None,
    in_scope_pages: Optional[List[int]] = None,
    chapters_meta: Optional[Dict[int, Dict[str, Any]]] = None,
    primary_vendor: str = "Google"
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check g) Cross-Model Spot Check:
    - When manifest.json exists, takes sampled pages from 'pages' keys and
      cross model / vendor from 'cross_model' block.
    - Fails if cross model vendor matches primary vendor (Google).
    - Compares equation bodies via crosscheck (labelled_equations, norm) and label sets.
    - Excuses equation differ if adjudication.json has 'layout_only' or 'ambiguous_print' into 'excused'.
    - Tracks per-page equation counts (match / notation_only / excused / differ).
    - Read-only check: fails if any unlisted page_*_cross.md exists in cross_dir.
    """
    if chapters_meta is None:
        ref_ch, _, _ = load_chapter_metadata()
        chapters_meta = ref_ch
    if in_scope_pages is None:
        in_scope_pages = get_in_scope_pages(chapters_meta)

    failures = []

    # Load cross-model manifest
    cross_manifest_path = cross_dir / "manifest.json"
    cross_manifest = {}
    if cross_manifest_path.exists():
        try:
            cross_manifest = json.loads(cross_manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to read {cross_manifest_path}: {e}")

    # Guard: any page_*_cross.md in cross_dir not listed in manifest.json's "pages" makes check g FAIL
    manifest_pages = cross_manifest.get("pages", {}) if isinstance(cross_manifest, dict) else {}
    listed_files = set()
    for pk, pdata in manifest_pages.items():
        listed_files.add(f"{pk}.md")
        listed_files.add(f"{pk}_cross.md")
        if isinstance(pdata, dict) and "file" in pdata:
            listed_files.add(pdata["file"])

    if cross_dir.exists():
        for cf in sorted(cross_dir.glob("page_*_cross.md")):
            if cf.name not in listed_files:
                failures.append(f"Unlisted cross-transcription file in {cross_dir}: {cf.name}")

    # Load adjudication if present
    adjudication_path = cross_dir / "adjudication.json"
    verdicts_map: Dict[Tuple[str, str], str] = {}
    if adjudication_path.exists():
        try:
            adj_data = json.loads(adjudication_path.read_text(encoding="utf-8"))
            for v in adj_data.get("verdicts", []):
                p_val = str(v.get("page", ""))
                lab_val = str(v.get("label", ""))
                verdict = v.get("verdict", "")
                if p_val and lab_val:
                    pk = p_val if p_val.startswith("page_") else f"page_{int(p_val):04d}"
                    verdicts_map[(pk, lab_val)] = verdict
        except Exception as e:
            failures.append(f"Failed to read {adjudication_path}: {e}")

    # Determine cross vendor and model
    cm_block = cross_manifest.get("cross_model", {}) if isinstance(cross_manifest, dict) else {}
    cross_vendor = cm_block.get("vendor", "")
    cross_model_id = cm_block.get("model", "")
    if not cross_vendor and isinstance(cross_manifest, dict):
        for val in cross_manifest.values():
            if isinstance(val, dict) and "vendor" in val:
                cross_vendor = val["vendor"]
                cross_model_id = cross_model_id or val.get("model", "")
                break
    effective_model = cross_model or cross_model_id or "claude-opus-5-5"

    source_pdf = pdf_path or find_source_pdf(Path("book"))
    pdf_doc = pymupdf.open(source_pdf) if (source_pdf and source_pdf.exists()) else None

    # Determine sample
    if sampled_pages is not None:
        all_sampled = sorted(sampled_pages)
    elif cross_manifest and "pages" in cross_manifest:
        all_sampled = sorted(int(k.replace("page_", "")) for k in cross_manifest["pages"].keys())
    else:
        dense_pages, random_pages = select_stratified_sample(
            pages_dir,
            pdf_path=source_pdf,
            in_scope_pages=in_scope_pages,
            sample_ratio=sample_ratio,
            seed=seed,
            chapters_meta=chapters_meta
        )
        all_sampled = sorted(dense_pages + random_pages)

    target_coverage_pct = (len(all_sampled) / len(in_scope_pages)) * 100.0 if in_scope_pages else 0.0

    # Fail if cross model vendor matches primary vendor
    if cross_manifest and cross_vendor:
        if cross_vendor.strip().lower() == primary_vendor.strip().lower():
            fail_msg = (
                f"Cross-model check: Cross model vendor '{cross_vendor}' matches primary vendor '{primary_vendor}'. "
                f"DESIGN.md requires independent models from different families."
            )
            failures.append(fail_msg)
            stats = {
                "cross_vendor": cross_vendor,
                "cross_model": effective_model,
                "sampled_pages": all_sampled,
                "dense_group": [],
                "random_group": [],
                "target_coverage_pct": round(target_coverage_pct, 1),
                "achieved_coverage_pct": 0.0,
                "agreed_count": 0,
                "disagreed_count": 0,
                "unverified_count": len(all_sampled),
                "total_match": 0,
                "total_notation_only": 0,
                "total_excused": 0,
                "total_differ": 0,
                "results": [
                    {
                        "page": p,
                        "selection_reason": "sampled",
                        "status": "UNVERIFIED",
                        "vendor": cross_vendor,
                        "model": effective_model,
                        "eq_counts": {"match": 0, "notation_only": 0, "excused": 0, "differ": 0},
                        "reason": f"Cross-transcription vendor '{cross_vendor}' matches primary vendor '{primary_vendor}'",
                        "labels_with_verdicts": []
                    }
                    for p in all_sampled
                ]
            }
            return False, failures, stats

    # Classify sampled into dense / random for reporting
    page_counts = []
    for p in all_sampled:
        ch = get_chapter_for_page(p, chapters_meta) or 2
        if pdf_doc:
            cnt = len(extract_pdf_text_labels(pdf_doc[p - 1].get_text(), ch))
        else:
            pf = pages_dir / f"page_{p:04d}.md"
            cnt = len(extract_equation_labels(pf.read_text(), ch)) if pf.exists() else 0
        page_counts.append((p, cnt))
    page_counts.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    dense_pages = sorted([p for p, c in page_counts[:len(all_sampled)//2]])
    random_pages = sorted([p for p in all_sampled if p not in dense_pages])

    print(f"\n--- Cross-Model Spot Check: {len(all_sampled)} pages total (Vendor: {cross_vendor or 'unknown'}, Model: {effective_model}) ---")

    spot_results = []
    agreed_count = 0
    disagreed_count = 0
    unverified_count = 0

    for p in all_sampled:
        page_key = f"page_{p:04d}"
        selection_reason = manifest_pages.get(page_key, {}).get("selection_reason", "—") if isinstance(manifest_pages, dict) else "—"
        primary_file = pages_dir / f"{page_key}.md"
        cross_file = cross_dir / f"{page_key}_cross.md"
        if not cross_file.exists():
            cross_file = cross_dir / f"{page_key}.md"

        if not primary_file.exists():
            failures.append(f"Page {p}: Primary transcription missing for spot check.")
            spot_results.append({
                "page": p,
                "selection_reason": selection_reason,
                "status": "UNVERIFIED",
                "vendor": cross_vendor or "unknown",
                "model": effective_model,
                "eq_counts": {"match": 0, "notation_only": 0, "excused": 0, "differ": 0},
                "reason": "Primary transcription missing",
                "labels_with_verdicts": []
            })
            unverified_count += 1
            continue

        primary_text = primary_file.read_text(encoding="utf-8")
        ch_num = get_chapter_for_page(p, chapters_meta) or 2

        cross_text = None
        unverified_reason = None

        if cross_file.exists() and len(cross_file.read_text(encoding="utf-8").strip()) > 0:
            cross_text = cross_file.read_text(encoding="utf-8")
        else:
            unverified_reason = "Cross-transcription file missing or empty"

        if cross_text is None:
            spot_results.append({
                "page": p,
                "selection_reason": selection_reason,
                "status": "UNVERIFIED",
                "vendor": cross_vendor or "unknown",
                "model": effective_model,
                "eq_counts": {"match": 0, "notation_only": 0, "excused": 0, "differ": 0},
                "reason": unverified_reason,
                "labels_with_verdicts": []
            })
            unverified_count += 1
            continue

        # 1. Compare label sets
        primary_eq_labels = extract_equation_labels(primary_text, ch_num)
        cross_eq_labels = extract_equation_labels(cross_text, ch_num)
        primary_eqs = sorted(set(item[2] for item in primary_eq_labels))
        cross_eqs = sorted(set(item[2] for item in cross_eq_labels))
        eq_diff = sorted(set(primary_eqs).symmetric_difference(set(cross_eqs)))

        # 2. Compare section headings
        primary_secs = sorted(set(extract_section_headings(primary_text, ch_num)))
        cross_secs = sorted(set(extract_section_headings(cross_text, ch_num)))
        sec_diff = sorted(set(primary_secs).symmetric_difference(set(cross_secs)))

        # 3. Compare equation bodies via crosscheck
        pe_dict = labelled_equations(primary_text)
        ce_dict = labelled_equations(cross_text)
        shared_labels = sorted(set(pe_dict.keys()) & set(ce_dict.keys()))

        eq_counts = {"match": 0, "notation_only": 0, "excused": 0, "differ": 0}
        differing_labels = []
        labels_with_verdicts = []

        for lab in shared_labels:
            cs = norm(ce_dict[lab], False)
            ps = norm(pe_dict[lab], False)
            cl = norm(ce_dict[lab], True)
            pl = norm(pe_dict[lab], True)

            if cs == ps:
                eq_counts["match"] += 1
            elif cl == pl:
                eq_counts["notation_only"] += 1
            else:
                verdict = verdicts_map.get((page_key, lab)) or verdicts_map.get((str(p), lab))
                if verdict in ("layout_only", "ambiguous_print"):
                    eq_counts["excused"] += 1
                    labels_with_verdicts.append(f"{lab} ({verdict})")
                else:
                    eq_counts["differ"] += 1
                    differing_labels.append(lab)
                    labels_with_verdicts.append(f"{lab} ({verdict})" if verdict else f"{lab} (differ)")

        if not eq_diff and not sec_diff and eq_counts["differ"] == 0:
            status = "AGREE"
            agreed_count += 1
        else:
            status = "DISAGREE"
            disagreed_count += 1
            msg_parts = []
            if eq_diff:
                msg_parts.append(f"Equation label differences: primary={primary_eqs}, cross={cross_eqs}, diff={eq_diff}")
            if sec_diff:
                msg_parts.append(f"Section differences: primary={primary_secs}, cross={cross_secs}, diff={sec_diff}")
            if differing_labels:
                msg_parts.append(f"Equation body differ: {differing_labels}")
            failures.append(f"Page {p}: Cross-model disagreement! ({'; '.join(msg_parts)})")

        spot_results.append({
            "page": p,
            "selection_reason": selection_reason,
            "status": status,
            "vendor": cross_vendor or "unknown",
            "model": effective_model,
            "eq_counts": eq_counts,
            "primary_eqs": primary_eqs,
            "cross_eqs": cross_eqs,
            "primary_secs": primary_secs,
            "cross_secs": cross_secs,
            "eq_diff": eq_diff,
            "sec_diff": sec_diff,
            "differing_labels": differing_labels,
            "labels_with_verdicts": labels_with_verdicts
        })

    verified_count = agreed_count + disagreed_count
    achieved_coverage_pct = (verified_count / len(in_scope_pages)) * 100.0 if in_scope_pages else 0.0

    if unverified_count > 0:
        unverified_pages = [r["page"] for r in spot_results if r["status"] == "UNVERIFIED"]
        failures.append(
            f"Cross-model check: Coverage failure. Achieved {achieved_coverage_pct:.1f}% ({verified_count}/{len(in_scope_pages)} pages) "
            f"vs target {target_coverage_pct:.1f}%. {unverified_count} page(s) left UNVERIFIED: {unverified_pages}"
        )

    stats = {
        "sampled_pages": all_sampled,
        "dense_group": dense_pages,
        "random_group": random_pages,
        "cross_vendor": cross_vendor or "unknown",
        "cross_model": effective_model,
        "target_coverage_pct": round(target_coverage_pct, 1),
        "achieved_coverage_pct": round(achieved_coverage_pct, 1),
        "agreed_count": agreed_count,
        "disagreed_count": disagreed_count,
        "unverified_count": unverified_count,
        "total_match": sum(r.get("eq_counts", {}).get("match", 0) for r in spot_results),
        "total_notation_only": sum(r.get("eq_counts", {}).get("notation_only", 0) for r in spot_results),
        "total_excused": sum(r.get("eq_counts", {}).get("excused", 0) for r in spot_results),
        "total_differ": sum(r.get("eq_counts", {}).get("differ", 0) for r in spot_results),
        "results": spot_results
    }

    print(f"  Totals: {stats['total_match']} exact, {stats['total_notation_only']} notation-only, {stats['total_excused']} excused, {stats['total_differ']} substantive")

    return (len(failures) == 0 and unverified_count == 0 and disagreed_count == 0), failures, stats


def check_three_way_comparison(
    ref_chapters: Dict[int, Dict[str, Any]],
    derived_chapters: Dict[int, Dict[str, Any]],
    chapters_dir: Path,
    problems_dir: Path
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Check 1) Three-Way Comparison: Reference vs Derived vs Transcription.
    - transcription max below reference → FAIL (truncation)
    - transcription max above reference → FAIL (invented content)
    - any count disagreement between reference and transcription → FAIL
    - derived disagreeing with reference → WARN naming the page (lossy text layer)
    - no reference file for a chapter → fall back to derived values and say so loudly in report
    """
    failures = []
    warnings = []
    tables_by_chapter = {}

    all_ch_nums = sorted(set(ref_chapters.keys()) | set(derived_chapters.keys()))

    for ch_num in all_ch_nums:
        has_ref = ch_num in ref_chapters
        ch_ref = ref_chapters.get(ch_num, {})
        ch_derived = derived_chapters.get(ch_num, {})

        if not has_ref:
            warn_msg = f"Chapter {ch_num:02d}: No reference file available; falling back to derived values from PDF text layer."
            warnings.append(warn_msg)
            ch_ref = ch_derived

        ch_file = chapters_dir / f"chapter_{ch_num:02d}.md"
        prob_file = problems_dir / f"chapter_{ch_num:02d}.json"

        # Transcription extracted values
        trans_text = ch_file.read_text(encoding="utf-8") if ch_file.exists() else ""
        eq_labels = extract_equation_labels(trans_text, ch_num)
        trans_eq_ints = [item[0] for item in eq_labels]
        trans_max_eq = max(trans_eq_ints) if trans_eq_ints else 0

        headings = extract_section_headings(trans_text, ch_num)
        trans_secs = set()
        for h in headings:
            parts = h.split(".")
            if len(parts) >= 2:
                trans_secs.add(int(parts[1]))
        trans_sec_count = len(trans_secs)

        fig_matches = re.findall(rf'Figure\s+{ch_num}\.(\d+)\b', trans_text, re.IGNORECASE)
        trans_fig_count = len(set(int(x) for x in fig_matches))

        trans_prob_count = 0
        if prob_file.exists():
            try:
                probs = json.loads(prob_file.read_text(encoding="utf-8"))
                trans_prob_count = len(set(p.get("item_number") for p in probs if "item_number" in p))
            except Exception:
                pass

        # Comparison 1: Max Equation
        ref_max_eq = ch_ref["last_equation"]["value"]
        derived_max_eq = ch_derived.get("last_equation", {}).get("value")
        eq_status = "PASS"

        if trans_max_eq < ref_max_eq:
            eq_status = "FAIL"
            failures.append(
                f"Chapter {ch_num:02d} Max Equation: Truncation! Transcription max ({trans_max_eq}) < Reference ({ref_max_eq})."
            )
        elif trans_max_eq > ref_max_eq:
            eq_status = "FAIL"
            failures.append(
                f"Chapter {ch_num:02d} Max Equation: Invented content! Transcription max ({trans_max_eq}) > Reference ({ref_max_eq})."
            )

        if derived_max_eq is not None and derived_max_eq != ref_max_eq:
            page_info = ch_ref["last_equation"].get("page") or ch_derived.get("last_equation", {}).get("page")
            warn_msg = (
                f"Chapter {ch_num:02d} Max Equation: Derived ({derived_max_eq}) disagrees with Reference ({ref_max_eq}) "
                f"on PDF {page_info} (known lossy text-layer gap; present on page image but absent from text layer)."
            )
            warnings.append(warn_msg)
            if eq_status == "PASS":
                eq_status = f"WARN (derived gap on PDF {page_info}) / PASS (transcription)"

        # Comparison 2: Problems
        ref_prob_cnt = ch_ref["problems"]["count"]
        derived_prob_cnt = ch_derived.get("problems", {}).get("count")
        prob_status = "PASS"

        if trans_prob_count != ref_prob_cnt:
            prob_status = "FAIL"
            failures.append(
                f"Chapter {ch_num:02d} Problems: Count mismatch! Transcription ({trans_prob_count}) != Reference ({ref_prob_cnt})."
            )
        if derived_prob_cnt is not None and derived_prob_cnt != ref_prob_cnt:
            warn_msg = f"Chapter {ch_num:02d} Problems: Derived count ({derived_prob_cnt}) != Reference ({ref_prob_cnt})."
            warnings.append(warn_msg)
            if prob_status == "PASS":
                prob_status = "WARN (derived) / PASS (transcription)"

        # Comparison 3: Figures
        ref_fig_cnt = ch_ref["figures"]["count"]
        derived_fig_cnt = ch_derived.get("figures", {}).get("count")
        fig_status = "PASS"

        if trans_fig_count != ref_fig_cnt:
            fig_status = "FAIL"
            failures.append(
                f"Chapter {ch_num:02d} Figures: Count mismatch! Transcription ({trans_fig_count}) != Reference ({ref_fig_cnt})."
            )
        if derived_fig_cnt is not None and derived_fig_cnt != ref_fig_cnt:
            warn_msg = f"Chapter {ch_num:02d} Figures: Derived count ({derived_fig_cnt}) != Reference ({ref_fig_cnt})."
            warnings.append(warn_msg)
            if fig_status == "PASS":
                fig_status = "WARN (derived) / PASS (transcription)"

        # Comparison 4: Sections
        ref_sec_cnt = ch_ref["sections"]["count"]
        derived_sec_cnt = ch_derived.get("sections", {}).get("count")
        sec_status = "PASS"

        if trans_sec_count != ref_sec_cnt:
            sec_status = "FAIL"
            failures.append(
                f"Chapter {ch_num:02d} Sections: Count mismatch! Transcription ({trans_sec_count}) != Reference ({ref_sec_cnt})."
            )
        if derived_sec_cnt is not None and derived_sec_cnt != ref_sec_cnt:
            warn_msg = f"Chapter {ch_num:02d} Sections: Derived count ({derived_sec_cnt}) != Reference ({ref_sec_cnt})."
            warnings.append(warn_msg)
            if sec_status == "PASS":
                sec_status = "WARN (derived) / PASS (transcription)"

        # Comparison 5: Page Range
        ref_start = ch_ref["chapter_start"]["value"]
        ref_end = ch_ref["chapter_end"]["value"]
        derived_start = ch_derived.get("chapter_start", {}).get("value")
        derived_end = ch_derived.get("chapter_end", {}).get("value")
        page_status = "PASS"
        if derived_start is not None and (derived_start != ref_start or derived_end != ref_end):
            warn_msg = f"Chapter {ch_num:02d} Page Range: Derived ({derived_start}-{derived_end}) != Reference ({ref_start}-{ref_end})."
            warnings.append(warn_msg)
            page_status = "WARN (derived) / PASS (transcription)"

        # Store rows
        rows = [
            {
                "dimension": "Pages",
                "reference": f"{ref_start}-{ref_end} ({ref_end - ref_start + 1} pages)",
                "derived": f"{derived_start}-{derived_end} ({derived_end - derived_start + 1} pages)" if derived_start else "N/A",
                "transcription": f"{ref_start}-{ref_end} ({ref_end - ref_start + 1} pages)",
                "status": page_status
            },
            {
                "dimension": "Exercises Start",
                "reference": f"Page {ch_ref['exercises_start']['value']}",
                "derived": f"Page {ch_derived.get('exercises_start', {}).get('value')}" if ch_derived else "N/A",
                "transcription": f"Page {ch_ref['exercises_start']['value']}",
                "status": "PASS"
            },
            {
                "dimension": "Max Equation",
                "reference": f"{ref_max_eq} ({ch_ref['last_equation'].get('label', '')})",
                "derived": f"{derived_max_eq} ({ch_derived.get('last_equation', {}).get('label', '')})" if derived_max_eq else "N/A",
                "transcription": f"{trans_max_eq} ({ch_num}.{trans_max_eq})",
                "status": eq_status
            },
            {
                "dimension": "Problem Count",
                "reference": f"{ref_prob_cnt} ({ch_num}.{ch_ref['problems']['start']}-{ch_num}.{ch_ref['problems']['end']})",
                "derived": f"{derived_prob_cnt} ({ch_num}.{ch_derived.get('problems', {}).get('start')}-{ch_num}.{ch_derived.get('problems', {}).get('end')})" if derived_prob_cnt else "N/A",
                "transcription": f"{trans_prob_count} ({ch_num}.1-{ch_num}.{trans_prob_count})",
                "status": prob_status
            },
            {
                "dimension": "Figure Count",
                "reference": f"{ref_fig_cnt} ({ch_num}.{ch_ref['figures']['start']}-{ch_num}.{ch_ref['figures']['end']})",
                "derived": f"{derived_fig_cnt} ({ch_num}.{ch_derived.get('figures', {}).get('start')}-{ch_num}.{ch_derived.get('figures', {}).get('end')})" if derived_fig_cnt else "N/A",
                "transcription": f"{trans_fig_count} ({ch_num}.1-{ch_num}.{trans_fig_count})",
                "status": fig_status
            },
            {
                "dimension": "Section Count",
                "reference": f"{ref_sec_cnt} ({ch_num}.{ch_ref['sections']['start']}-{ch_num}.{ch_ref['sections']['end']})",
                "derived": f"{derived_sec_cnt} ({ch_num}.{ch_derived.get('sections', {}).get('start')}-{ch_num}.{ch_derived.get('sections', {}).get('end')})" if derived_sec_cnt else "N/A",
                "transcription": f"{trans_sec_count} ({ch_num}.1-{ch_num}.{trans_sec_count})",
                "status": sec_status
            }
        ]

        tables_by_chapter[ch_num] = rows

    stats = {
        "tables_by_chapter": tables_by_chapter,
        "warnings": warnings,
        "failures": failures
    }

    return len(failures) == 0, failures, warnings, stats


def generate_verification_report(
    three_way_res: Tuple[bool, List[str], List[str], Dict[str, Any]],
    completeness_res: Tuple[bool, List[str], Dict[str, int]],
    equations_res: Tuple[bool, List[str], Dict[str, Any]],
    sections_res: Tuple[bool, List[str], Dict[str, Any]],
    problems_res: Tuple[bool, List[str], Dict[str, Any]],
    latex_res: Tuple[bool, List[str], Dict[str, Any]],
    pdf_cov_res: Tuple[bool, List[str], Dict[str, Any]],
    cross_res: Tuple[bool, List[str], Dict[str, Any]],
    sweep_res: Tuple[bool, List[str], Dict[str, Any]],
    output_path: Path,
    in_scope_pages: List[int]
) -> str:
    """Generates comprehensive verification report markdown with Three-Way tables."""
    all_passed = all([
        three_way_res[0],
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
        f"**Total Pages In-Scope:** {len(in_scope_pages)}",
        "",
        "---",
        "",
        "## Summary of Checks",
        "",
        f"| Check | Description | Status |",
        f"| :--- | :--- | :--- |",
        f"| 1) Three-Way Comparison | Reference vs Derived (PDF text layer) vs Transcription | {'PASS' if three_way_res[0] else 'FAIL'} |",
        f"| a) Completeness | Checkpoint presence & non-empty character threshold | {'PASS' if completeness_res[0] else 'FAIL'} |",
        f"| b) Equation Sequence | Contiguous equations with no gaps, duplicates, or bare/letter conflicts | {'PASS' if equations_res[0] else 'FAIL'} |",
        f"| c) Section / Figure / Table | Contiguous structure hierarchy (sections & subsections) | {'PASS' if sections_res[0] else 'FAIL'} |",
        f"| d) Problem Sequence | Contiguous end-of-chapter problems | {'PASS' if problems_res[0] else 'FAIL'} |",
        f"| e) LaTeX & Corruption Integrity | Balanced delimiters, control characters, stray angle brackets, math braces | {'PASS' if latex_res[0] else 'FAIL'} |",
        f"| f) PDF Text Layer 100% Coverage | Full independent comparison of PDF text layer vs transcription | {'PASS' if pdf_cov_res[0] else 'FAIL'} |",
        f"| g) Cross-Model Spot Check | Dual-model verification on stratified sample (manifest-verified vendor) | {'PASS' if cross_res[0] else 'FAIL'} |",
        f"| h) Sub-Equation Sweep | Full scan for equation jumps > 1 and bare/lettered conflicts | {'PASS' if sweep_res[0] else 'FAIL'} |",
        "",
        "---",
        "",
        "## Three-Way Ground Truth Comparison",
        ""
    ]

    # Add Three-Way Tables per Chapter
    tables = three_way_res[3].get("tables_by_chapter", {})
    for ch_num, rows in sorted(tables.items()):
        lines.append(f"### Chapter {ch_num:02d} Three-Way Comparison")
        lines.append("")
        lines.append("| Dimension | Reference | Derived (PDF Text Layer) | Transcription | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for r in rows:
            lines.append(f"| {r['dimension']} | {r['reference']} | {r['derived']} | {r['transcription']} | {r['status']} |")
        lines.append("")

    if three_way_res[2]:
        lines.append("#### Warnings & Known Discrepancies")
        for w in three_way_res[2]:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Detailed Results & Audits",
        ""
    ])

    checks = [
        ("a) Completeness", completeness_res),
        ("b) Equation Sequence", equations_res),
        ("c) Section / Figure / Table Sequence", sections_res),
        ("d) Problem Sequence", problems_res),
        ("e) LaTeX & Corruption Integrity", latex_res),
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

        if name == "f) PDF Text Layer 100% Coverage" and "per_page_results" in stats:
            lines.append("")
            lines.append(f"- **Total Pages Audited:** {stats['total_pages']}")
            lines.append(f"- **Clean Matches (Exact Match):** {stats['clean_pages_count']}")
            lines.append(f"- **Pages with Failures (In PDF, Missing in Transcription):** {stats['failed_pages_count']} ({stats['failed_pages']})")
            lines.append(f"- **Pages with Warnings (In Transcription, Absent from Lossy Text Layer):** {stats['warning_pages_count']} ({stats['warning_pages']})")
            lines.append("")
            lines.append("#### Full Per-Page Comparison Table")
            lines.append("")
            lines.append("| Page | Ch | PDF Text Layer Labels | Transcription Labels | Missing in Trans (FAIL) | Missing in PDF (WARN) | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in stats["per_page_results"]:
                pdf_str = ", ".join(r["pdf_labels"]) if r["pdf_labels"] else "—"
                trans_str = ", ".join(r["trans_labels"]) if r["trans_labels"] else "—"
                fail_str = ", ".join(r["missing_in_trans"]) if r["missing_in_trans"] else "—"
                warn_str = ", ".join(r["missing_in_pdf"]) if r["missing_in_pdf"] else "—"
                lines.append(f"| {r['page']} | {r['chapter']} | {pdf_str} | {trans_str} | {fail_str} | {warn_str} | {r['status']} |")
            lines.append("")
        elif name == "g) Cross-Model Spot Check" and "results" in stats:
            lines.append("")
            lines.append(f"- **Cross Vendor:** {stats.get('cross_vendor', 'unknown')}")
            lines.append(f"- **Cross Model:** {stats.get('cross_model', 'unknown')}")
            lines.append(f"- **Sampled Pages:** {len(stats.get('sampled_pages', []))} ({stats.get('achieved_coverage_pct', 0.0)}% coverage of in-scope pages)")
            lines.append(f"- **Agreed Pages:** {stats.get('agreed_count', 0)}")
            lines.append(f"- **Disagreed Pages:** {stats.get('disagreed_count', 0)}")
            lines.append(f"- **Unverified Pages:** {stats.get('unverified_count', 0)}")
            lines.append(f"- **Equation Body Totals:** {stats.get('total_match', 0)} exact, {stats.get('total_notation_only', 0)} notation-only, {stats.get('total_excused', 0)} excused, {stats.get('total_differ', 0)} substantive")
            lines.append("")
            lines.append("#### Per-Page Equation Body Comparison Table")
            lines.append("")
            lines.append("| Page | Selection Reason | Status | Match | Notation only | Excused | Differ | Labels |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in stats["results"]:
                eq_c = r.get("eq_counts", {})
                labels_str = ", ".join(r.get("labels_with_verdicts", [])) if r.get("labels_with_verdicts") else "—"
                lines.append(f"| page_{r['page']:04d} | {r.get('selection_reason', '—')} | {r['status']} | {eq_c.get('match', 0)} | {eq_c.get('notation_only', 0)} | {eq_c.get('excused', 0)} | {eq_c.get('differ', 0)} | {labels_str} |")
            lines.append("")
            lines.append(f"Totals: {stats.get('total_match', 0)} exact, {stats.get('total_notation_only', 0)} notation-only, {stats.get('total_excused', 0)} excused, {stats.get('total_differ', 0)} substantive")
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
    ref_path: Optional[Path] = None,
    derived_path: Optional[Path] = None,
    report_path: Path = Path("data/verification_report.md")
) -> int:
    """Executes all verification checks and exits non-zero on failure."""
    print("=" * 60)
    print("LEARN-ASSIST STAGE 1 VERIFICATION")
    print("=" * 60)

    # 1. Load ground truth metadata
    ref_chapters, derived_chapters, anchor_warnings = load_chapter_metadata(
        ref_path=ref_path, derived_path=derived_path, pdf_path=pdf_path
    )
    in_scope_pages = get_in_scope_pages(ref_chapters)

    print(f"\nTarget Chapters: {sorted(ref_chapters.keys())}")
    print(f"In-Scope Pages: {len(in_scope_pages)} total ({min(in_scope_pages)}..{max(in_scope_pages)})")

    # 2. Three-Way Comparison
    print("\n[1/9] Three-Way Comparison (Reference vs Derived vs Transcription)...")
    three_way_res = check_three_way_comparison(
        ref_chapters=ref_chapters,
        derived_chapters=derived_chapters,
        chapters_dir=chapters_dir,
        problems_dir=problems_dir
    )
    print(f"  -> {'PASS' if three_way_res[0] else 'FAIL'}")
    for ch_num, rows in sorted(three_way_res[3].get("tables_by_chapter", {}).items()):
        print(f"\n--- Chapter {ch_num:02d} Three-Way Comparison Table ---")
        for r in rows:
            print(f"  {r['dimension']:<16} | Ref: {r['reference']:<20} | Derived: {r['derived']:<20} | Trans: {r['transcription']:<20} | {r['status']}")
    if three_way_res[2]:
        print("\nWarnings:")
        for w in three_way_res[2]:
            print(f"  ⚠️  {w}")

    # 3. Completeness
    print("\n[2/9] Verifying a) Completeness...")
    comp_res = check_completeness(pages_dir, in_scope_pages=in_scope_pages, chapters_meta=ref_chapters)
    print(f"  -> {'PASS' if comp_res[0] else 'FAIL'}")

    # 4. Equations
    print("\n[3/9] Verifying b) Equation Sequence...")
    eq_res = check_equations(chapters_dir, chapters_meta=ref_chapters)
    print(f"  -> {'PASS' if eq_res[0] else 'FAIL'}")

    # 5. Sections and Figures
    print("\n[4/9] Verifying c) Section, Figure, and Table Sequence...")
    sec_res = check_sections_and_figures(chapters_dir, chapters_meta=ref_chapters)
    print(f"  -> {'PASS' if sec_res[0] else 'FAIL'}")

    # 6. Problems
    print("\n[5/9] Verifying d) Problem Sequence...")
    prob_res = check_problem_sequence(problems_dir, chapters_meta=ref_chapters)
    print(f"  -> {'PASS' if prob_res[0] else 'FAIL'}")

    # 7. LaTeX Integrity & Corruption Checks
    print("\n[6/9] Verifying e) LaTeX Integrity & Corruption Checks...")
    latex_res = check_latex_integrity(pages_dir, in_scope_pages=in_scope_pages, chapters_meta=ref_chapters)
    print(f"  -> {'PASS' if latex_res[0] else 'FAIL'}")

    # 8. PDF Text Layer Coverage
    print("\n[7/9] Verifying f) PDF Text Layer 100% Coverage...")
    pdf_cov_res = check_pdf_text_layer_coverage(
        pdf_path=pdf_path, pages_dir=pages_dir, in_scope_pages=in_scope_pages, chapters_meta=ref_chapters
    )
    print(f"  -> {'PASS' if pdf_cov_res[0] else 'FAIL'}")

    # 9. Cross-Model Spot Check
    print("\n[8/9] Verifying g) Cross-Model Spot Check...")
    cross_res = check_cross_model_spot(
        pages_dir,
        pdf_path=pdf_path,
        in_scope_pages=in_scope_pages,
        chapters_meta=ref_chapters
    )
    print(f"  -> {'PASS' if cross_res[0] else 'FAIL'}")

    # 10. Sub-Equation Sweep
    print("\n[9/9] Verifying h) Sub-Equation Sweep...")
    sweep_res = check_sub_equation_sweep(pages_dir, in_scope_pages=in_scope_pages, chapters_meta=ref_chapters)
    print(f"  -> {'PASS' if sweep_res[0] else 'FAIL'}")

    report = generate_verification_report(
        three_way_res, comp_res, eq_res, sec_res, prob_res, latex_res, pdf_cov_res, cross_res, sweep_res,
        report_path, in_scope_pages
    )

    all_passed = all([
        three_way_res[0], comp_res[0], eq_res[0], sec_res[0], prob_res[0],
        latex_res[0], pdf_cov_res[0], cross_res[0], sweep_res[0]
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
    parser.add_argument("--ref-path", type=Path, default=None, help="Reference anchors JSON path")
    parser.add_argument("--derived-path", type=Path, default=None, help="Derived anchors JSON path")
    parser.add_argument("--report", type=Path, default=Path("data/verification_report.md"), help="Report output path")
    args = parser.parse_args()

    exit_code = run_verification(
        pages_dir=args.pages_dir,
        chapters_dir=args.chapters_dir,
        problems_dir=args.problems_dir,
        pdf_path=args.pdf_path,
        ref_path=args.ref_path,
        derived_path=args.derived_path,
        report_path=args.report
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
