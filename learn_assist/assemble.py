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


def run_assembly(pages_dir: Path = Path("data/pages")):
    """Assembles all configured chapters and extracts problems."""
    chapters_dir = Path("data/chapters")
    problems_dir = Path("data/problems")
    chapters_dir.mkdir(parents=True, exist_ok=True)
    problems_dir.mkdir(parents=True, exist_ok=True)

    for ch_num, cfg in CHAPTER_CONFIG.items():
        print(f"\n--- Assembling Chapter {ch_num:02d}: {cfg['title']} ---")
        ch_md = assemble_chapter_markdown(cfg["start_page"], cfg["end_page"], pages_dir)
        out_md_path = Path(cfg["output_md"])
        out_md_path.write_text(ch_md, encoding="utf-8")
        print(f"Saved {out_md_path} ({len(ch_md)} characters)")

        problems = extract_problems(ch_num, ch_md)
        out_prob_path = Path(cfg["output_problems"])
        with open(out_prob_path, "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        print(f"Extracted {len(problems)} problems into {out_prob_path}")
        if problems:
            print(f"  Range: {problems[0]['number']} to {problems[-1]['number']}")


def main():
    parser = argparse.ArgumentParser(description="Assemble transcribed pages into chapters and extract problems.")
    parser.add_argument("--pages-dir", type=Path, default=Path("data/pages"), help="Directory with page markdown checkpoints")
    args = parser.parse_args()
    run_assembly(pages_dir=args.pages_dir)


if __name__ == "__main__":
    main()
