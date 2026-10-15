#!/usr/bin/env python3
"""
learn_assist.assemble

Assembles per-page markdown checkpoints into full chapter markdown files:
- data/chapters/chapter_02.md (PDF pages 33..76, Book pages 13..56)
- data/chapters/chapter_03.md (PDF pages 77..125, Book pages 57..105)

Extracts structured end-of-chapter problems into:
- data/problems/chapter_02.json
- data/problems/chapter_03.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


CHAPTER_CONFIG = {
    2: {
        "title": "Mathematical Fundamentals",
        "start_page": 33,
        "end_page": 76,
        "output_md": "data/chapters/chapter_02.md",
        "output_problems": "data/problems/chapter_02.json"
    },
    3: {
        "title": "Signal Generation and Detection",
        "start_page": 77,
        "end_page": 125,
        "output_md": "data/chapters/chapter_03.md",
        "output_problems": "data/problems/chapter_03.json"
    }
}


def assemble_chapter_markdown(start_page: int, end_page: int, pages_dir: Path) -> str:
    """Concatenates page markdown files for a given page range."""
    chunks = []
    for p in range(start_page, end_page + 1):
        page_path = pages_dir / f"page_{p:04d}.md"
        if not page_path.exists():
            raise FileNotFoundError(f"Missing checkpoint page: {page_path}")
        content = page_path.read_text(encoding="utf-8").strip()
        book_page = p - 20
        header = f"<!-- PDF Page {p} | Book Page {book_page} -->\n"
        chunks.append(f"{header}\n{content}")
    return "\n\n".join(chunks) + "\n"


def parse_subparts(text: str) -> List[Dict[str, str]]:
    """Detects subparts like (a) ..., (b) ... or a) ..., b) ... in problem text."""
    pattern = r'(?:\n|^)\s*(?:\(([a-d])\)|([a-d])\))\s+([^\n]+(?:\n(?!\s*(?:\([a-d]\)|[a-d]\))).*)*)'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if not matches:
        return []

    subparts = []
    for m in matches:
        label = m.group(1) or m.group(2)
        part_text = m.group(3).strip()
        subparts.append({
            "label": label,
            "text": part_text
        })
    return subparts


def extract_problems(chapter_num: int, chapter_text: str) -> List[Dict[str, Any]]:
    """
    Extracts structured problems (Exercises) for the chapter.
    Looks for the Exercises / Problems section, then splits into individual problem items.
    """
    # Find start of Exercises section
    exercises_match = re.search(r'(?:#+\s*(?:Exercises|Problems)|(?:\n|^)\s*Exercises\s*\n)', chapter_text, re.IGNORECASE)
    if exercises_match:
        exercises_text = chapter_text[exercises_match.start():]
    else:
        # Fallback to search from the first occurrence of Chapter.1 e.g. 2.1 or 3.1 in problem context
        first_prob = re.search(rf'(?:\n|^)\s*{chapter_num}\.1\s+', chapter_text)
        if first_prob:
            exercises_text = chapter_text[first_prob.start():]
        else:
            exercises_text = chapter_text

    # Match problems numbered like "2.1 ...", "2.2 ...", "Problem 2.1 ...", "**2.1** ..."
    # Problem header pattern: start of line, optional prefix, e.g., 2.1 or **2.1** or Problem 2.1
    prob_regex = re.compile(
        rf'(?:\n|^)(?:\*\*|\*|#+|\bProblem\s+|\bExercise\s+)?({chapter_num}\.(\d+))(?:\*\*|\*|\.|\:)?\s+',
        re.MULTILINE
    )

    matches = list(prob_regex.finditer(exercises_text))
    if not matches:
        return []

    problems = []
    for i, m in enumerate(matches):
        full_num = m.group(1)
        item_num = int(m.group(2))
        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(exercises_text)

        raw_body = exercises_text[start_idx:end_idx].strip()
        # Clean up any trailing chapter navigation or notes
        raw_body = re.sub(r'<!-- PDF Page \d+ \| Book Page \d+ -->', '', raw_body).strip()

        subparts = parse_subparts(raw_body)

        problems.append({
            "number": full_num,
            "chapter": chapter_num,
            "item_number": item_num,
            "full_text": f"{full_num} {raw_body}".strip(),
            "sub_parts": subparts
        })

    # Deduplicate by item_number keeping the longest body if duplicates occur
    dedup: Dict[int, Dict[str, Any]] = {}
    for p in problems:
        n = p["item_number"]
        if n not in dedup or len(p["full_text"]) > len(dedup[n]["full_text"]):
            dedup[n] = p

    return sorted(dedup.values(), key=lambda x: x["item_number"])


def clean_printed_string(printed: str) -> str:
    """
    Cleans a printed string for quotation inside an erratum note:
    - Removes equation labels matching \\quad\\s*\\(\\d+\\.\\d+[a-z]?\\) or \\tag{...}
    - Removes $$ and $ delimiters
    - Strips empty lines that were only delimiters
    """
    cleaned = re.sub(r"\\quad\s*\(\d+\.\d+[a-z]?\)", "", printed)
    cleaned = re.sub(r"\\tag\{[^}]*\}", "", cleaned)
    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"\$\$|\$", "", line).rstrip()
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def apply_errata(
    chapter_text: str,
    errata: List[Dict[str, Any]],
    chapter_num: Optional[int] = None
) -> str:
    """
    Applies approved errata to chapter text.
    - Entries with status != 'approved' are ignored.
    - If status == 'approved':
      - The 'printed' string MUST be found exactly once.
      - If missing or appears more than once, raises ValueError.
      - Applied entries replace 'printed' with 'corrected' in place.
      - Judgement entries leave text unchanged.
      - Notes are placed on their own lines immediately AFTER the enclosing unit:
        - If inside a $$ display block (or spans several), after the closing $$ of the last block.
        - Otherwise, after the end of the paragraph containing the correction.
      - Applied note format:
        > **Erratum {id}** (corrected above). The printed book has:
        > ```latex
        > <cleaned printed string>
        > ```
      - Judgement note format:
        > **Possible erratum {id}** (text left as printed): {basis}
    """
    result = chapter_text
    ch_info = f" in Chapter {chapter_num}" if chapter_num else ""
    approved = [e for e in errata if e.get("status") == "approved"]

    # Sort approved errata by position of appearance in the chapter text
    positions = []
    for entry in approved:
        printed = entry.get("printed", "")
        e_id = entry.get("id", "unknown")
        count = result.count(printed)
        if count == 0:
            raise ValueError(
                f"Errata {e_id}{ch_info}: 'printed' string not found: {printed!r}"
            )
        elif count > 1:
            raise ValueError(
                f"Errata {e_id}{ch_info}: 'printed' string found {count} times (expected exactly 1): {printed!r}"
            )
        idx = result.find(printed)
        positions.append((idx, entry))

    positions.sort(key=lambda x: x[0])

    for _, entry in positions:
        e_id = entry.get("id", "unknown")
        printed = entry.get("printed", "")
        corrected = entry.get("corrected")
        kind = entry.get("kind", "certain")
        basis = entry.get("basis", "")

        idx = result.find(printed)
        if idx == -1:
            raise ValueError(
                f"Errata {e_id}{ch_info}: 'printed' string not found during replacement: {printed!r}"
            )

        replacement_text = printed if kind == "judgement" else corrected
        before = result[:idx]
        inside_math = (before.count("$$") % 2 == 1) or ("$$" in printed)

        result = result[:idx] + replacement_text + result[idx + len(printed):]
        rep_end = idx + len(replacement_text)

        if inside_math:
            closing_dd = result.find("$$", rep_end)
            if closing_dd == -1:
                raise ValueError(
                    f"Errata {e_id}{ch_info}: unclosed $$ after display math replacement"
                )
            insert_pos = closing_dd + 2
        else:
            m = re.search(r"(\n\s*\n|\n(?=\$\$|\#|<!--)|$)", result[rep_end:])
            insert_pos = rep_end + m.start()

        if kind == "judgement":
            note = f"> **Possible erratum {e_id}** (text left as printed): {basis}"
        else:
            clean_printed = clean_printed_string(printed)
            quoted_lines = "\n".join(f"> {line}" for line in clean_printed.splitlines())
            note = (
                f"> **Erratum {e_id}** (corrected above). The printed book has:\n"
                f"> ```latex\n"
                f"{quoted_lines}\n"
                f"> ```"
            )

        before_text = result[:insert_pos].rstrip("\n")
        after_text = result[insert_pos:].lstrip("\n")
        if after_text:
            result = f"{before_text}\n\n{note}\n\n{after_text}"
        else:
            result = f"{before_text}\n\n{note}\n" if chapter_text.endswith("\n") else f"{before_text}\n\n{note}"

    return result


def run_assembly(
    pages_dir: Path = Path("data/pages"),
    chapters_dir: Path = Path("data/chapters"),
    printed_dir: Path = Path("data/chapters_printed"),
    problems_dir: Path = Path("data/problems"),
    errata_path: Optional[Path] = Path("data/errata.json")
):
    """
    Assembles all configured chapters into:
    - data/chapters_printed/ : assembled exactly from data/pages (as printed)
    - data/chapters/ : with approved errata applied and visible notes
    - data/problems/ : structured problem extractions
    """
    chapters_dir.mkdir(parents=True, exist_ok=True)
    printed_dir.mkdir(parents=True, exist_ok=True)
    problems_dir.mkdir(parents=True, exist_ok=True)

    all_errata = []
    if errata_path and errata_path.exists():
        try:
            all_errata = json.loads(errata_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to load {errata_path}: {e}")

    for ch_num, cfg in CHAPTER_CONFIG.items():
        print(f"\n--- Assembling Chapter {ch_num:02d}: {cfg['title']} ---")
        ch_printed = assemble_chapter_markdown(cfg["start_page"], cfg["end_page"], pages_dir)

        # 1. Output to chapters_printed/
        out_printed_path = printed_dir / f"chapter_{ch_num:02d}.md"
        out_printed_path.write_text(ch_printed, encoding="utf-8")
        print(f"Saved printed chapter to {out_printed_path} ({len(ch_printed)} characters)")

        # 2. Apply errata for data/chapters/
        ch_errata = [
            e for e in all_errata
            if cfg["start_page"] <= e.get("pdf_page", 0) <= cfg["end_page"]
        ]
        ch_corrected = apply_errata(ch_printed, ch_errata, chapter_num=ch_num)
        out_md_path = chapters_dir / f"chapter_{ch_num:02d}.md"
        out_md_path.write_text(ch_corrected, encoding="utf-8")
        print(f"Saved corrected chapter to {out_md_path} ({len(ch_corrected)} characters)")

        # 3. Extract problems
        problems = extract_problems(ch_num, ch_corrected)
        out_prob_path = problems_dir / f"chapter_{ch_num:02d}.json"
        with open(out_prob_path, "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        print(f"Extracted {len(problems)} problems into {out_prob_path}")
        if problems:
            print(f"  Range: {problems[0]['number']} to {problems[-1]['number']}")


def main():
    parser = argparse.ArgumentParser(description="Assemble transcribed pages into chapters and extract problems.")
    parser.add_argument("--pages-dir", type=Path, default=Path("data/pages"), help="Directory with page markdown checkpoints")
    parser.add_argument("--chapters-dir", type=Path, default=Path("data/chapters"), help="Output directory for corrected chapters")
    parser.add_argument("--printed-dir", type=Path, default=Path("data/chapters_printed"), help="Output directory for printed chapters")
    parser.add_argument("--problems-dir", type=Path, default=Path("data/problems"), help="Output directory for problems")
    parser.add_argument("--errata-path", type=Path, default=Path("data/errata.json"), help="Path to errata.json")
    args = parser.parse_args()
    run_assembly(
        pages_dir=args.pages_dir,
        chapters_dir=args.chapters_dir,
        printed_dir=args.printed_dir,
        problems_dir=args.problems_dir,
        errata_path=args.errata_path
    )


if __name__ == "__main__":
    main()

