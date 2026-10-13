"""
learn_assist.anchors

Derives ground-truth structural anchors per chapter directly from the source PDF
text layer alone (never from transcription):
- chapter start page: page whose text layer begins with "Chapter N"
- chapter end page: page before next chapter start, skipping blank pages
- exercises start page: first page with "Exercises" or "Problems" heading
- maximum equation label across pages before exercises
- problem numbers from exercises pages
- figure numbers from captions

Writes derived anchors to data/anchors_derived.json.
Loads reference anchors from anchors/<book>.json.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pymupdf

from learn_assist.ingest import find_source_pdf


def is_chapter_start(page_text: str, ch_num: int) -> bool:
    """
    Checks if a page's text layer begins with 'Chapter N'.
    Running headers on interior pages read '<page number> Chapter N ...'
    or similar, so the page-number prefix distinguishes them.
    """
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not lines:
        return False
    first_line = lines[0]
    # Check if first line begins with Chapter <N> (not prefixed by digits or other text)
    pattern = rf"^Chapter\s+{ch_num}(?:\s*|\s+.*)$"
    return bool(re.match(pattern, first_line, re.IGNORECASE)) and not bool(re.match(r"^\d+", first_line))


def find_chapter_start(doc: pymupdf.Document, ch_num: int) -> Optional[int]:
    """
    Scans PDF pages (1-indexed) to find the page where Chapter N begins.
    """
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text()
        if is_chapter_start(text, ch_num):
            return page_idx + 1
    return None


def find_chapter_end(
    doc: pymupdf.Document,
    ch_num: int,
    next_ch_start: Optional[int] = None
) -> Optional[int]:
    """
    Finds the page before next chapter's start, skipping blank pages.
    """
    target_next_start = next_ch_start
    if target_next_start is None:
        target_next_start = find_chapter_start(doc, ch_num + 1)

    if target_next_start is not None:
        p = target_next_start - 1
        while p >= 1:
            page_text = doc[p - 1].get_text().strip()
            if page_text:
                return p
            p -= 1
        return None

    # If no next chapter, find last non-blank page in document
    p = len(doc)
    while p >= 1:
        if doc[p - 1].get_text().strip():
            return p
        p -= 1
    return None


def find_exercises_start(doc: pymupdf.Document, ch_start: int, ch_end: int) -> Optional[int]:
    """
    Finds the first page in [ch_start, ch_end] containing an Exercises or Problems heading.
    """
    for p in range(ch_start, ch_end + 1):
        text = doc[p - 1].get_text()
        for line in text.splitlines():
            if re.match(r"^\s*(?:Exercises|Problems)\b", line, re.IGNORECASE):
                return p
    return None


def extract_pdf_equation_labels(page_text: str, ch_num: int) -> List[Tuple[int, str, str]]:
    """
    Extracts equation labels of the form (ch_num.M) or (ch_num.Ma) from PDF text layer.
    Returns list of (base_num, letter_suffix, full_label).
    """
    labels = []
    lines = page_text.splitlines()
    for line in lines:
        line_s = line.strip()
        for m in re.finditer(rf'\({ch_num}\.(\d+)([a-z]?)\)', line_s, re.IGNORECASE):
            prefix = line_s[:m.start()].strip()
            suffix = line_s[m.end():].strip()
            if re.search(r'\b(?:Eqs?\.?|ofEq\.?|Equations?\.?|see|from|in|and|or|between|with|to)\b', prefix, re.IGNORECASE):
                continue
            if any(w in prefix.lower() for w in ['defined', 'given', 'using', 'as shown', 'recall']):
                continue
            if re.search(r'^[a-zA-Z]{2,}', suffix):
                continue
            base_num = int(m.group(1))
            letter = m.group(2).lower()
            tag = f"{ch_num}.{base_num}{letter}"
            labels.append((base_num, letter, tag))
    return labels


def extract_pdf_max_equation(
    doc: pymupdf.Document,
    ch_start: int,
    exercises_start: int,
    ch_num: int
) -> Dict[str, Any]:
    """
    Extracts the maximum equation label across the chapter's pages before exercises.
    """
    max_num = 0
    max_label = ""
    max_page = 0
    found_labels = []

    end_page = exercises_start - 1 if exercises_start else ch_start
    for p in range(ch_start, end_page + 1):
        text = doc[p - 1].get_text()
        lbls = extract_pdf_equation_labels(text, ch_num)
        for base_num, letter, tag in lbls:
            found_labels.append((base_num, tag, p))
            if base_num > max_num:
                max_num = base_num
                max_label = tag
                max_page = p

    return {
        "value": max_num,
        "label": max_label or (f"{ch_num}.{max_num}" if max_num else ""),
        "page": max_page,
        "source": "text layer"
    }


def extract_pdf_problems(
    doc: pymupdf.Document,
    exercises_start: int,
    ch_end: int,
    ch_num: int
) -> Dict[str, Any]:
    """
    Extracts problem numbers from the exercises pages.
    """
    probs = set()
    for p in range(exercises_start, ch_end + 1):
        text = doc[p - 1].get_text()
        for line in text.splitlines():
            m = re.match(rf"^\s*{ch_num}\.(\d+)\b", line)
            if m:
                probs.add(int(m.group(1)))

    sorted_probs = sorted(probs)
    return {
        "start": sorted_probs[0] if sorted_probs else 0,
        "end": sorted_probs[-1] if sorted_probs else 0,
        "count": len(sorted_probs),
        "items": sorted_probs,
        "source": "text layer"
    }


def extract_pdf_figures(
    doc: pymupdf.Document,
    ch_start: int,
    ch_end: int,
    ch_num: int
) -> Dict[str, Any]:
    """
    Extracts figure numbers from captions in the chapter.
    """
    figs = set()
    for p in range(ch_start, ch_end + 1):
        text = doc[p - 1].get_text()
        for line in text.splitlines():
            for m in re.finditer(rf"\bFigure\s+{ch_num}\.(\d+)\b", line, re.IGNORECASE):
                figs.add(int(m.group(1)))

    sorted_figs = sorted(figs)
    return {
        "start": sorted_figs[0] if sorted_figs else 0,
        "end": sorted_figs[-1] if sorted_figs else 0,
        "count": len(sorted_figs),
        "items": sorted_figs,
        "source": "text layer"
    }


def extract_pdf_sections(
    doc: pymupdf.Document,
    ch_start: int,
    exercises_start: int,
    ch_num: int
) -> Dict[str, Any]:
    """
    Extracts top-level section numbers in the chapter before exercises.
    """
    secs = set()
    end_p = exercises_start - 1 if exercises_start else ch_start
    for p in range(ch_start, end_p + 1):
        text = doc[p - 1].get_text()
        for line in text.splitlines():
            m = re.search(rf"\bSection\s+{ch_num}\.(\d+)\b", line, re.IGNORECASE)
            if not m:
                m = re.match(rf"^\s*{ch_num}\.(\d+)\b", line)
            if m:
                secs.add(int(m.group(1)))

    sorted_secs = sorted(secs)
    return {
        "start": sorted_secs[0] if sorted_secs else 0,
        "end": sorted_secs[-1] if sorted_secs else 0,
        "count": len(sorted_secs),
        "items": sorted_secs,
        "source": "text layer"
    }


def extract_pdf_tables(
    doc: pymupdf.Document,
    ch_start: int,
    ch_end: int,
    ch_num: int
) -> Dict[str, Any]:
    """
    Extracts table numbers in the chapter.
    """
    tbls = set()
    for p in range(ch_start, ch_end + 1):
        text = doc[p - 1].get_text()
        for line in text.splitlines():
            for m in re.finditer(rf"\bTable\s+{ch_num}\.(\d+)\b", line, re.IGNORECASE):
                tbls.add(int(m.group(1)))

    sorted_tbls = sorted(tbls)
    return {
        "items": sorted_tbls,
        "count": len(sorted_tbls),
        "source": "text layer"
    }


def derive_chapter_anchors(
    doc: pymupdf.Document,
    ch_num: int,
    next_ch_start: Optional[int] = None
) -> Dict[str, Any]:
    """
    Derives all structural anchors for a single chapter directly from the PDF text layer.
    """
    start_p = find_chapter_start(doc, ch_num)
    if start_p is None:
        raise ValueError(f"Could not find start page for Chapter {ch_num}")

    end_p = find_chapter_end(doc, ch_num, next_ch_start=next_ch_start)
    if end_p is None:
        raise ValueError(f"Could not find end page for Chapter {ch_num}")

    ex_start = find_exercises_start(doc, start_p, end_p)
    if ex_start is None:
        raise ValueError(f"Could not find exercises start page for Chapter {ch_num}")

    max_eq = extract_pdf_max_equation(doc, start_p, ex_start, ch_num)
    problems = extract_pdf_problems(doc, ex_start, end_p, ch_num)
    figures = extract_pdf_figures(doc, start_p, end_p, ch_num)
    sections = extract_pdf_sections(doc, start_p, ex_start, ch_num)
    tables = extract_pdf_tables(doc, start_p, end_p, ch_num)

    return {
        "chapter_start": {
            "value": start_p,
            "source": "text layer"
        },
        "chapter_end": {
            "value": end_p,
            "source": "text layer"
        },
        "exercises_start": {
            "value": ex_start,
            "source": "text layer"
        },
        "last_equation": max_eq,
        "problems": problems,
        "figures": figures,
        "sections": sections,
        "tables": tables
    }


def derive_all_anchors(
    pdf_path: Optional[Path] = None,
    chapters: Optional[List[int]] = None,
    output_path: Optional[Path] = Path("data/anchors_derived.json")
) -> Dict[str, Any]:
    """
    Derives anchors for all specified chapters and writes to output_path.
    """
    source_pdf = pdf_path or find_source_pdf(Path("book"))
    if not source_pdf or not source_pdf.exists():
        raise FileNotFoundError(f"Source PDF not found: {source_pdf}")

    target_chapters = chapters or [2, 3]
    doc = pymupdf.open(source_pdf)

    derived = {
        "source_pdf": str(source_pdf.name),
        "chapters": {}
    }

    for idx, ch in enumerate(target_chapters):
        next_ch = target_chapters[idx + 1] if idx + 1 < len(target_chapters) else ch + 1
        next_start = find_chapter_start(doc, next_ch)
        derived["chapters"][str(ch)] = derive_chapter_anchors(doc, ch, next_ch_start=next_start)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(derived, indent=2), encoding="utf-8")

    return derived


def load_reference(ref_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Loads committed reference anchors file (e.g. anchors/liang_lauterbur.json).
    Returns dict or None if file does not exist.
    """
    target = ref_path or Path("anchors/liang_lauterbur.json")
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Failed to load reference anchors from {target}: {e}")
        return None


def load_derived(
    derived_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
    chapters: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Loads derived anchors from data/anchors_derived.json, or computes them from PDF.
    """
    target = derived_path or Path("data/anchors_derived.json")
    if target.exists():
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            pass
    return derive_all_anchors(pdf_path=pdf_path, chapters=chapters, output_path=target)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Derive chapter structural anchors from PDF text layer.")
    parser.add_argument("--pdf-path", type=Path, default=None, help="Path to source PDF")
    parser.add_argument("--output", type=Path, default=Path("data/anchors_derived.json"), help="Output JSON path")
    parser.add_argument("--chapters", type=int, nargs="+", default=[2, 3], help="Chapters to derive")
    args = parser.parse_args()

    result = derive_all_anchors(pdf_path=args.pdf_path, chapters=args.chapters, output_path=args.output)
    print(f"Derived anchors for chapters {args.chapters} written to {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
