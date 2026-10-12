#!/usr/bin/env python3
"""
learn_assist.ingest

Resumable ingestion pipeline for textbook pages:
1. Renders in-scope PDF pages to PNG at 300 DPI into data/page_images/
2. Transcribes page images via vision model into GitHub-flavoured Markdown with LaTeX math
3. Checkpoints each page immediately to data/pages/page_NNNN.md
4. Writes data/manifest.json tracking hashes, ranges, model, character counts, and timestamps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import sys
import time
import json
import base64
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import pymupdf


DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_DPI = 300
DEFAULT_START_PAGE = 33
DEFAULT_END_PAGE = 125

TRANSCRIPTION_PROMPT = """You are transcribing a scanned textbook page from Liang & Lauterbur, "Principles of Magnetic Resonance Imaging", into GitHub-flavoured Markdown.

CRITICAL TRANSCRIPTION RULES:
1. Mathematics: Transcribe ALL equations and mathematical symbols into precise LaTeX.
   - Use inline $...$ for inline formulas.
   - Use display $$...$$ for standalone display equations.
2. Equation Numbers: Preserve equation numbers exactly as printed (e.g. "(2.1)", "(2.117)", "(3.14)").
   - For display equations, include the equation number at the end, e.g.
     $$
     E = mc^2 \\quad (3.14)
     $$
     or inside the line as printed.
3. Headings: Preserve all chapter and section headings (e.g. "# Chapter 2", "## Section 2.6 The Radon Transform", "### 2.6.1 Definition"), subheadings, and titles.
4. Figures & Tables: Preserve figure captions (e.g. "**Figure 3.1** ...") and table structures using Markdown table syntax.
5. Exercises / Problems: Transcribe end-of-chapter problems and exercises completely with their numbers (e.g. "2.1", "3.28") and all sub-parts ((a), (b), (c)).
6. Verbatim Fidelity: Do NOT summarise, paraphrase, omit, or correct anything. Transcribe the full text verbatim.
7. Clean Output: Output ONLY the transcribed Markdown. Do not include introductory text, conversational meta-commentary, or markdown code fence wrappers (do not wrap the entire output in ```markdown ... ```).
"""


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_source_pdf(book_dir: Path) -> Path:
    """Locates the source PDF inside the book/ directory."""
    pdfs = list(book_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No source PDF found in {book_dir.resolve()}.")
    return pdfs[0]


def render_page_to_png(doc: pymupdf.Document, page_num: int, output_path: Path, dpi: int = DEFAULT_DPI) -> Path:
    """Renders a 1-indexed PDF page to a PNG image file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        pix.save(str(output_path))
    return output_path


def clean_markdown_fences(text: str) -> str:
    """Removes outer markdown code fence wrappers if present."""
    text = text.strip()
    if text.startswith("```markdown") and text.endswith("```"):
        text = text[len("```markdown"): -3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text


def transcribe_page_image(
    image_path: Path,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> str:
    """Calls the vision model to transcribe the page image with retries on rate limits."""
    key = api_key or DEFAULT_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable or provide --api-key.")

    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "image/png", "data": img_b64}},
                {"text": TRANSCRIPTION_PROMPT}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0
        }
    }

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Empty candidates in response: {data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"No content parts in candidate: {candidates[0]}")
                raw_text = parts[0].get("text", "")
                return clean_markdown_fences(raw_text)
            elif resp.status_code in (429, 500, 503, 504):
                print(f"[Retry {attempt}/{max_retries}] HTTP {resp.status_code} for {image_path.name}. Backing off {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                resp.raise_for_status()
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to transcribe {image_path.name} after {max_retries} attempts: {exc}")
            print(f"[Retry {attempt}/{max_retries}] Error ({exc}). Backing off {delay:.1f}s...")
            time.sleep(delay)
            delay *= backoff_factor

    raise RuntimeError(f"Exhausted retries for {image_path.name}")


def run_ingestion(
    pdf_path: Optional[Path] = None,
    start_page: int = DEFAULT_START_PAGE,
    end_page: int = DEFAULT_END_PAGE,
    dpi: int = DEFAULT_DPI,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    force: bool = False,
    delay_between_calls: float = 0.5,
) -> Dict[str, Any]:
    """Runs the resumable ingestion pipeline over the specified page range."""
    key = api_key or DEFAULT_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable or pass --api-key.")

    book_dir = Path("book")
    source_pdf = pdf_path or find_source_pdf(book_dir)
    if not source_pdf.exists():
        raise FileNotFoundError(f"PDF does not exist: {source_pdf}")

    data_dir = Path("data")
    images_dir = data_dir / "page_images"
    pages_dir = data_dir / "pages"
    manifest_path = data_dir / "manifest.json"

    images_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening PDF: {source_pdf}")
    doc = pymupdf.open(str(source_pdf))
    total_doc_pages = len(doc)
    print(f"Total PDF pages: {total_doc_pages}. In-scope range: PDF {start_page} to {end_page} (inclusive).")

    if start_page < 1 or end_page > total_doc_pages or start_page > end_page:
        raise ValueError(f"Invalid page range: {start_page} to {end_page} (document has {total_doc_pages} pages)")

    # Load existing manifest if present
    manifest = {}
    if manifest_path.exists() and not force:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    pdf_sha256 = manifest.get("source_pdf_sha256") or compute_sha256(source_pdf)
    pages_manifest = manifest.get("pages", {})

    total_chars = 0
    pages_processed = 0

    print(f"Ingesting pages using model '{model}' at {dpi} DPI...")
    for page_num in range(start_page, end_page + 1):
        page_key = f"page_{page_num:04d}"
        md_path = pages_dir / f"{page_key}.md"
        img_path = images_dir / f"{page_key}.png"

        # Checkpoint check: skip completed pages
        if md_path.exists() and not force:
            content = md_path.read_text(encoding="utf-8")
            if len(content.strip()) > 0:
                char_count = len(content)
                total_chars += char_count
                pages_manifest[page_key] = {
                    "page_number": page_num,
                    "book_page": page_num - 20,
                    "char_count": char_count,
                    "timestamp": pages_manifest.get(page_key, {}).get("timestamp", time.time()),
                    "status": "cached"
                }
                print(f"[{page_num}/{end_page}] Cached ({char_count} chars): {md_path.name}")
                continue

        # Render page
        render_page_to_png(doc, page_num, img_path, dpi=dpi)

        # Transcribe
        print(f"[{page_num}/{end_page}] Transcribing PDF page {page_num} (Book p. {page_num - 20})...")
        transcription = transcribe_page_image(img_path, api_key=key, model=model)

        # Save checkpoint immediately
        md_path.write_text(transcription, encoding="utf-8")
        char_count = len(transcription)
        total_chars += char_count
        pages_processed += 1

        pages_manifest[page_key] = {
            "page_number": page_num,
            "book_page": page_num - 20,
            "char_count": char_count,
            "timestamp": time.time(),
            "status": "transcribed"
        }
        print(f"[{page_num}/{end_page}] Saved {md_path.name} ({char_count} chars)")

        # Manifest incremental checkpoint
        manifest_data = {
            "source_pdf": source_pdf.name,
            "source_pdf_sha256": pdf_sha256,
            "page_range": {
                "start": start_page,
                "end": end_page,
                "chapter_02": [33, 76],
                "chapter_03": [77, 125]
            },
            "dpi": dpi,
            "model": model,
            "total_pages_in_scope": end_page - start_page + 1,
            "total_characters": total_chars,
            "pages": pages_manifest
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        if delay_between_calls > 0:
            time.sleep(delay_between_calls)

    print(f"\nIngestion complete! Ingested/verified {end_page - start_page + 1} pages. Total chars: {total_chars}.")
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Ingest textbook pages via vision model.")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to textbook PDF")
    parser.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE, help="Starting PDF page (default: 33)")
    parser.add_argument("--end-page", type=int, default=DEFAULT_END_PAGE, help="Ending PDF page (default: 125)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="DPI for rendering (default: 300)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID (default: gemini-2.5-flash)")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--force", action="store_true", help="Force re-transcription of cached pages")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds")
    args = parser.parse_args()

    run_ingestion(
        pdf_path=args.pdf,
        start_page=args.start_page,
        end_page=args.end_page,
        dpi=args.dpi,
        model=args.model,
        api_key=args.api_key,
        force=args.force,
        delay_between_calls=args.delay,
    )


if __name__ == "__main__":
    main()
