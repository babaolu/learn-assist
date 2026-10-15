"""Deterministic running-text cross-check between primary and cross-model page transcriptions.

Extracts prose, normalises text, performs word-level diff, aligns and compares
inline math, locates candidate differences in source PDF scans, generates 200 DPI crops,
and writes structured JSON and review Markdown.
"""

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from learn_assist.crosscheck import norm

INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)")

HIGH_PRIO_WORDS = {
    "not", "no", "never", "cannot", "none", "neither", "nor",
    "on", "off", "increase", "decrease", "increasing", "decreasing",
    "increases", "decreases", "larger", "smaller", "large", "small",
    "more", "less", "before", "after", "positive", "negative"
}

ANTONYM_PREFIXES = ("un", "in", "non")


def extract_prose(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract prose and inline math tokens from a page markdown.
    Returns: (list_of_word_tokens, list_of_math_strings)
    """
    # 1. Remove $$...$$ display blocks
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)

    # 2. Remove [Figure: ...] descriptions (including multiline and inline variants)
    text = re.sub(r"(?m)^\s*\[Figure:.*\]\s*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Figure:[\s\S]*?\](?=\s*(?:\n\n|\Z|\n\s*[*#\w]))", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Figure:.*?\]", " ", text, flags=re.IGNORECASE)

    # 3. Running-head lines
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            clean = re.sub(r"^[#\s\-—–]+", "", line).strip()
            clean_no_cs = re.sub(r"\b(?:Chapter\s+\d+|Section\s+\d+(?:\.\d+)?)\b", "", clean, flags=re.IGNORECASE)
            has_page_num = bool(re.search(r"\b\d+\b", clean_no_cs))
            has_cs = bool(re.search(r"\b(?:Chapter\s+\d+|Section\s+\d+(?:\.\d+)?|Exercises)\b", clean, re.IGNORECASE))
            is_lone_num = bool(re.match(r"^\d+$", clean))

            if has_page_num and has_cs:
                # Combined running head on single line (e.g. "14 — Chapter 2 Mathematical Fundamentals")
                lines[i] = ""
            elif is_lone_num:
                # Line 1 is page number, check if line 2 is the title
                lines[i] = ""
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        c2 = re.sub(r"^[#\s\-—–]+", "", lines[j]).strip()
                        if re.match(r"^(?:Chapter\s+\d+|Section\s+\d+(?:\.\d+)?|Exercises)\b", c2, re.IGNORECASE):
                            lines[j] = ""
                        break
            elif has_cs:
                # Line 1 is title, check if line 2 is page number
                lines[i] = ""
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        c2 = re.sub(r"^[#\s\-—–]+", "", lines[j]).strip()
                        if re.match(r"^\d+$", c2):
                            lines[j] = ""
                        break
            break

    # Strip drop folios (trailing standalone page number at bottom of page)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            clean_last = re.sub(r"^[#\s\-—–]+", "", lines[i]).strip()
            if re.match(r"^\d+$", clean_last):
                lines[i] = ""
            break

    text = "\n".join(lines)

    # 4. Footnote markers like $^7$, [^7], or unicode superscripts
    text = re.sub(r"\[\^\d+\]:?", " ", text)
    text = re.sub(r"\$\^\{?\d+\}?\$", " ", text)
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", " ", text)

    # 5. Extract inline math into placeholder tokens MATH_k
    math_list: List[str] = []

    def repl_math(m: re.Match) -> str:
        k = len(math_list)
        math_list.append(m.group(1).strip())
        return f" MATH_{k} "

    text = INLINE_RE.sub(repl_math, text)

    # 6. Markdown markers (#, >, *, **, ■, ---)
    text = re.sub(r"[#>■]", " ", text)
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"-{3,}", " ", text)

    # Normalise prose: lowercase, unify quotes and dashes, rejoin words hyphenated across lines, collapse whitespace
    text = text.lower()
    text = re.sub(r"[“”„]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[—–]|--", "-", text)

    # Rejoin words hyphenated across lines
    text = re.sub(r"(\b[a-z]+)-\s*\n\s*([a-z]+\b)", r"\1\2", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    return tokens, math_list


def to_key(token: str) -> str:
    """Normalize a token for sequence matching (collapse math placeholders, strip outer punct)."""
    if re.match(r"^math_\d+[^\w]*$", token):
        return "__math__"
    k = re.sub(r"^[^\w]+|[^\w]+$", "", token)
    return k if k else token


def is_high_priority(p_text: str, c_text: str) -> bool:
    """Determine if difference involves negation, number, direction, or antonym prefix."""
    for s in (p_text, c_text):
        if re.search(r"\d", s):
            return True
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", s.lower())
        for t in tokens:
            if t in HIGH_PRIO_WORDS:
                return True
            if t.endswith("n't"):
                return True

    p_words = set(re.findall(r"\b\w+\b", p_text.lower()))
    c_words = set(re.findall(r"\b\w+\b", c_text.lower()))
    for pw in p_words:
        for cw in c_words:
            if pw != cw:
                for prefix in ANTONYM_PREFIXES:
                    if pw.startswith(prefix) and pw[len(prefix):] == cw:
                        return True
                    if cw.startswith(prefix) and cw[len(prefix):] == pw:
                        return True
    return False


def display_tokens(tokens: List[str], m_list: List[str]) -> str:
    """Reconstruct human-readable text by expanding MATH_k tokens back to LaTeX."""
    res = []
    for t in tokens:
        m = re.match(r"^math_(\d+)([^\w]*)$", t)
        if m:
            idx = int(m.group(1))
            punct = m.group(2)
            res.append(f"${m_list[idx]}${punct}")
        else:
            res.append(t)
    return " ".join(res)


def make_snippet(tokens: List[str], m_list: List[str], start: int, end: int) -> str:
    """Format ~8 words of context either side of diff range [start:end]."""
    ctx_start = max(0, start - 8)
    ctx_end = min(len(tokens), end + 8)
    parts = []
    if ctx_start < start:
        parts.append(display_tokens(tokens[ctx_start:start], m_list))
    if start < end:
        diff_str = display_tokens(tokens[start:end], m_list)
        parts.append(f"**{diff_str}**")
    if end < ctx_end:
        parts.append(display_tokens(tokens[end:ctx_end], m_list))
    return " ".join(parts)


def diff_page(
    page_name: str,
    pw: List[str],
    pm: List[str],
    cw: List[str],
    cm: List[str]
) -> List[Dict[str, Any]]:
    """Perform word-level diff and math alignment on one page."""
    p_keys = [to_key(t) for t in pw]
    c_keys = [to_key(t) for t in cw]

    sm = difflib.SequenceMatcher(None, p_keys, c_keys)
    candidates: List[Dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Check aligned inline math placeholders
            for pi, cj in zip(range(i1, i2), range(j1, j2)):
                if p_keys[pi] == "__math__" and c_keys[cj] == "__math__":
                    p_idx = int(re.match(r"^math_(\d+)", pw[pi]).group(1))
                    c_idx = int(re.match(r"^math_(\d+)", cw[cj]).group(1))
                    p_math = pm[p_idx]
                    c_math = cm[c_idx]
                    if norm(p_math, True) != norm(c_math, True):
                        p_snip = make_snippet(pw, pm, pi, pi + 1)
                        c_snip = make_snippet(cw, cm, cj, cj + 1)
                        p_diff = f"${p_math}$"
                        c_diff = f"${c_math}$"
                        candidates.append({
                            "page": page_name,
                            "kind": "inline_math",
                            "primary_diff": p_diff,
                            "cross_diff": c_diff,
                            "primary": p_snip,
                            "cross": c_snip,
                            "priority": "high" if is_high_priority(p_math, c_math) else "normal",
                            "p_range": (pi, pi + 1),
                            "c_range": (cj, cj + 1),
                        })
        else:
            p_sub = pw[i1:i2]
            c_sub = cw[j1:j2]

            # 1. Drop punctuation-only opcodes
            p_al = "".join(c for c in "".join(p_sub) if c.isalnum())
            c_al = "".join(c for c in "".join(c_sub) if c.isalnum())
            if p_al == c_al:
                continue

            # 2. Drop hyphenation split/join opcodes
            p_nh_al = "".join(c for c in "".join(p_sub).replace("-", "") if c.isalnum())
            c_nh_al = "".join(c for c in "".join(c_sub).replace("-", "") if c.isalnum())
            if p_nh_al == c_nh_al:
                continue

            # 3. Check if purely math tokens that normalize identically
            p_math_indices = [int(m.group(1)) for t in p_sub for m in [re.match(r"^math_(\d+)", t)] if m]
            c_math_indices = [int(m.group(1)) for t in c_sub for m in [re.match(r"^math_(\d+)", t)] if m]
            p_words_only = [t for t in p_sub if not re.match(r"^math_\d+", t)]
            c_words_only = [t for t in c_sub if not re.match(r"^math_\d+", t)]
            p_words_al = "".join(c for c in "".join(p_words_only) if c.isalnum())
            c_words_al = "".join(c for c in "".join(c_words_only) if c.isalnum())

            if p_math_indices and c_math_indices and p_words_al == c_words_al:
                p_norm = "".join(norm(pm[idx], True) for idx in p_math_indices)
                c_norm = "".join(norm(cm[idx], True) for idx in c_math_indices)
                if p_norm == c_norm:
                    continue

            p_diff = display_tokens(p_sub, pm)
            c_diff = display_tokens(c_sub, cm)
            p_snip = make_snippet(pw, pm, i1, i2)
            c_snip = make_snippet(cw, cm, j1, j2)
            priority = "high" if is_high_priority(p_diff, c_diff) else "normal"

            candidates.append({
                "page": page_name,
                "kind": tag,
                "primary_diff": p_diff,
                "cross_diff": c_diff,
                "primary": p_snip,
                "cross": c_snip,
                "priority": priority,
                "p_range": (i1, i2),
                "c_range": (j1, j2),
            })

    return candidates


def load_exclusions(repo_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load known errata and adjudication verdicts indexed by page."""
    errata_path = repo_root / "data/errata.json"
    adj_path = repo_root / "data/cross_model/adjudication.json"

    by_page: Dict[str, List[Dict[str, Any]]] = {}

    if errata_path.exists():
        errata = json.loads(errata_path.read_text(encoding="utf-8"))
        for e in errata:
            p_id = f"page_{e['pdf_page']:04d}"
            by_page.setdefault(p_id, []).append({
                "source": f"erratum {e['id']}",
                "label": e.get("label"),
                "texts": [e.get("printed", ""), e.get("corrected", ""), e.get("basis", "")]
            })

    if adj_path.exists():
        adj = json.loads(adj_path.read_text(encoding="utf-8"))
        for v in adj.get("verdicts", []):
            p_id = v.get("page")
            if p_id:
                by_page.setdefault(p_id, []).append({
                    "source": f"adjudication {v.get('label')}",
                    "label": v.get("label"),
                    "texts": [v.get("label", ""), v.get("detail", "")]
                })

    return by_page


def check_exclusion(cand: Dict[str, Any], exclusions_by_page: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """Check if candidate matches an existing erratum or adjudication verdict."""
    page = cand["page"]
    page_excls = exclusions_by_page.get(page, [])
    if not page_excls:
        return None

    cand_text = f"{cand.get('primary_diff', '')} {cand.get('cross_diff', '')} {cand.get('primary', '')} {cand.get('cross', '')}".lower()

    for ex in page_excls:
        # 1. Match by label
        if ex.get("label"):
            clean_lab = str(ex["label"]).replace("(", "").replace(")", "").strip()
            if clean_lab and clean_lab in cand_text:
                return ex["source"]

        # 2. Match by overlapping distinctive words
        for t in ex.get("texts", []):
            if not t:
                continue
            words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", t.lower())
                     if w not in {"frac", "begin", "left", "right", "text", "quad", "mathrm", "with", "from", "that", "this", "have"}]
            if len(words) >= 2:
                matches = [w for w in words if w in cand_text]
                if len(matches) >= 2:
                    return ex["source"]
            elif len(words) == 1 and words[0] in cand_text:
                return ex["source"]

    return None


def get_snippet_search_phrases(snippet: str) -> List[str]:
    """Generate 4-6 word clean search phrases from a snippet."""
    clean = snippet.replace("**", " ")
    words: List[str] = []
    for w in clean.split():
        if not re.search(r"\\|\$|math_\d+", w):
            cw = re.sub(r"^[^\w]+|[^\w]+$", "", w)
            if cw and len(cw) > 1:
                words.append(cw)

    phrases = []
    for n in (6, 5, 4, 3):
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i:i + n]))
    return phrases


def locate_and_render_crop(
    pdf_doc: pymupdf.Document,
    pdf_page_num: int,
    cand: Dict[str, Any],
    crops_dir: Path,
    crop_filename: str
) -> Optional[str]:
    """Search for 4-6 word shared context phrases in PDF and render 200 DPI crop."""
    try:
        page = pdf_doc[pdf_page_num - 1]
    except Exception:
        return None

    phrases = get_snippet_search_phrases(cand.get("primary", "")) + get_snippet_search_phrases(cand.get("cross", ""))

    rects = []
    for ph in phrases:
        rects = page.search_for(ph)
        if rects:
            break

    if not rects:
        return None

    r = rects[0]
    line_h = max(r.height, 12.0)

    y0 = max(0.0, r.y0 - line_h * 1.5)
    y1 = min(page.rect.height, r.y1 + line_h * 1.5)
    x0 = 35.0
    x1 = min(page.rect.width - 35.0, 520.0)

    crop_rect = pymupdf.Rect(x0, y0, x1, y1)
    mat = pymupdf.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat, clip=crop_rect)

    crops_dir.mkdir(parents=True, exist_ok=True)
    out_path = crops_dir / crop_filename
    pix.save(str(out_path))

    return str(out_path.relative_to(crops_dir.parent.parent))


def run_textcheck(repo_root: Optional[Path] = None) -> Tuple[Dict[str, Any], str]:
    """Run full textcheck across all pages in manifest."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    manifest_path = repo_root / "data/cross_model/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pdf_files = list((repo_root / "book").glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("Source PDF not found in book/")
    pdf_doc = pymupdf.open(pdf_files[0])

    exclusions_by_page = load_exclusions(repo_root)
    crops_dir = repo_root / "data/cross_model/textcheck_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    active_candidates: List[Dict[str, Any]] = []
    excluded_candidates: List[Dict[str, Any]] = []

    page_items = sorted(manifest["pages"].items())

    for page_name, page_info in page_items:
        pf = repo_root / "data/pages" / f"{page_name}.md"
        cf = repo_root / "data/cross_model" / f"{page_name}_cross.md"

        if not pf.exists() or not cf.exists():
            continue

        pw, pm = extract_prose(pf.read_text(encoding="utf-8"))
        cw, cm = extract_prose(cf.read_text(encoding="utf-8"))

        cands = diff_page(page_name, pw, pm, cw, cm)
        pdf_page_num = page_info["pdf_page"]

        page_cand_idx = 1
        for c in cands:
            excl_source = check_exclusion(c, exclusions_by_page)
            if excl_source:
                c["exclusion_source"] = excl_source
                excluded_candidates.append(c)
            else:
                crop_fname = f"{page_name}_c{page_cand_idx}.png"
                crop_rel = locate_and_render_crop(
                    pdf_doc, pdf_page_num, c, crops_dir, crop_fname
                )
                c["crop"] = crop_rel
                c["id"] = f"{page_name}_c{page_cand_idx}"
                page_cand_idx += 1
                active_candidates.append(c)

    # Build summary
    by_kind = {"insert": 0, "delete": 0, "replace": 0, "inline_math": 0}
    by_prio = {"high": 0, "normal": 0}
    crops_found = 0
    crops_missing = 0

    for c in active_candidates:
        by_kind[c["kind"]] += 1
        by_prio[c["priority"]] += 1
        if c.get("crop"):
            crops_found += 1
        else:
            crops_missing += 1

    summary = {
        "pages": len(page_items),
        "total_candidates": len(active_candidates),
        "high_priority": by_prio["high"],
        "normal_priority": by_prio["normal"],
        "by_kind": by_kind,
        "crops_found": crops_found,
        "crops_missing": crops_missing,
        "excluded_count": len(excluded_candidates)
    }

    def clean_cand(c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": c.get("id"),
            "page": c["page"],
            "kind": c["kind"],
            "priority": c["priority"],
            "primary": c["primary"],
            "cross": c["cross"],
            "primary_diff": c.get("primary_diff"),
            "cross_diff": c.get("cross_diff"),
            "crop": c.get("crop")
        }

    def clean_excl(c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "page": c["page"],
            "kind": c["kind"],
            "source": c.get("exclusion_source"),
            "primary": c["primary"],
            "cross": c["cross"],
            "primary_diff": c.get("primary_diff"),
            "cross_diff": c.get("cross_diff")
        }

    out_json_data = {
        "summary": summary,
        "candidates": [clean_cand(c) for c in active_candidates],
        "exclusions": [clean_excl(c) for c in excluded_candidates]
    }

    json_path = repo_root / "data/cross_model/textcheck.json"
    json_path.write_text(json.dumps(out_json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Running-Text Cross-Check Review",
        "",
        "## Summary",
        "",
        "| Pages | Total Candidates | High Priority | Normal Priority | Inline Math | Crops Found | Crops Missing | Excluded (Known Errata/Verdicts) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| {summary['pages']} | {summary['total_candidates']} | {summary['high_priority']} | {summary['normal_priority']} | {summary['by_kind']['inline_math']} | {summary['crops_found']} | {summary['crops_missing']} | {summary['excluded_count']} |",
        "",
        "### Breakdown by Kind",
        f"- **Insert:** {summary['by_kind']['insert']}",
        f"- **Delete:** {summary['by_kind']['delete']}",
        f"- **Replace:** {summary['by_kind']['replace']}",
        f"- **Inline Math:** {summary['by_kind']['inline_math']}",
        "",
        "---",
        "",
        "## Candidates",
        ""
    ]

    for c in active_candidates:
        cid = c.get("id", f"{c['page']}_c")
        md_lines.append(f"### {cid} ({c['kind']}, {c['priority']})")
        if c.get("crop"):
            md_lines.append(f"Crop: `{c['crop']}`")
        else:
            md_lines.append("Crop: null")
        md_lines.append("")
        md_lines.append(f"PRIMARY: {c['primary']}")
        md_lines.append(f"CROSS: {c['cross']}")
        md_lines.append("")
        md_lines.append("VERDICT: ")
        md_lines.append("")

    if excluded_candidates:
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## Exclusions (Covered by Errata or Adjudication)")
        md_lines.append("")
        for e in excluded_candidates:
            md_lines.append(f"- **{e['page']}** ({e['kind']}) — Covered by {e.get('exclusion_source')}")
            md_lines.append(f"  - PRIMARY: {e['primary']}")
            md_lines.append(f"  - CROSS: {e['cross']}")
            md_lines.append("")

    md_content = "\n".join(md_lines)
    md_path = repo_root / "data/cross_model/TEXTCHECK.md"
    md_path.write_text(md_content, encoding="utf-8")

    return out_json_data, md_content


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data, md = run_textcheck(repo_root)
    s = data["summary"]
    print("Textcheck complete.")
    print(f"Pages: {s['pages']}")
    print(f"Total candidates: {s['total_candidates']} (High: {s['high_priority']}, Normal: {s['normal_priority']})")
    print(f"By kind: {s['by_kind']}")
    print(f"Crops: {s['crops_found']} found, {s['crops_missing']} missing")
    print(f"Exclusions: {s['excluded_count']}")


if __name__ == "__main__":
    main()
