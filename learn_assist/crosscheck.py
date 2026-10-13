"""Equation-body comparison between a primary and a cross-model page transcription.

labelled_equations(md) -> {label: latex_body}; norm(body, loose) -> canonical string.
Status per shared label: "match" (identical after strict normalisation),
"notation_only" (identical after dropping font/vector wrappers and braces), "differ".
CLI: python -m learn_assist.crosscheck <cross_dir> <primary_pages_dir> <out_json>
(cross files may be named page_NNNN.md or page_NNNN_cross.md)
"""
import difflib
import json
import re
import sys
from pathlib import Path

LABEL_RE = re.compile(
    r"(?:\\tag\{\s*(\d+)\.(\d+)\s*([a-z]?)\s*\})"
    r"|(?:\(\s*(\d+)\.(\d+)\s*(?:\\text\{\s*([a-z])\s*\}|\\mathrm\{\s*([a-z])\s*\}|([a-z]))?\s*\))"
)
DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.S)


INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)")


def labelled_equations(md: str):
    """Return {label: body} for every labelled equation (display, or inline with a label)."""
    md = re.sub(r"(?m)^>\s?", "", md)
    out = {}
    blocks = DISPLAY_RE.findall(md)
    rest = DISPLAY_RE.sub(" ", md)
    blocks += [b for b in INLINE_RE.findall(rest) if LABEL_RE.search(b)]
    for block in blocks:
        prev = 0
        for m in LABEL_RE.finditer(block):
            g = m.groups()
            if g[0]:
                lab = f"{g[0]}.{g[1]}{g[2] or ''}"
            else:
                lab = f"{g[3]}.{g[4]}{g[5] or g[6] or g[7] or ''}"
            body = block[prev:m.start()]
            prev = m.end()
            out[lab] = body
    return out


def all_labels_anywhere(md: str):
    """Labels that appear as display-equation tags (for coverage)."""
    return set(labelled_equations(md))


SUBS = [
    (r"\\\\(?:\[[^\]]*\])?", ";"),
    (r"\\(?:overset|stackrel|xrightarrow)\{([^{}]*)\}(?:\{([^{}]*)\})?", r"-[\1]\2"),
    (r"\^\*_(\{[^{}]*\}|\w)", r"_\1^*"),
    (r"0_\{?([+-])\}?", r"0\1"),
    (r"\\(?:q?quad|,|;|!|:|\s)", " "),
    (r"\\(?:left|right|big|Big|bigg|Bigg)(?![a-zA-Z])", ""),
    (r"\\(?:displaystyle|textstyle)", ""),
    (r"\\[td]frac", r"\\frac"),
    (r"\\(?:triangleq|stackrel\{\\Delta\}\{=\}|overset\{\\triangle\}\{=\}|stackrel\{\\triangle\}\{=\})", "≜"),
    (r"\\(?:cdots|ldots|dots)", "..."),
    (r"\\(?:longleftrightarrow|leftrightarrow|Longleftrightarrow)", "<->"),
    (r"\\(?:longrightarrow|rightarrow|to)(?![a-zA-Z])", "->"),
    (r"\\geq?(?![a-zA-Z])", ">="),
    (r"\\leq?(?![a-zA-Z])", "<="),
    (r"\\neq?(?![a-zA-Z])", "!="),
    (r"\\begin\{(?:cases|bmatrix|pmatrix|aligned|array)\}(?:\{[^}]*\})?", "{"),
    (r"\\end\{(?:cases|bmatrix|pmatrix|aligned|array)\}", "}"),
    (r"\\\\(?:\[[^\]]*\])?", ";"),
    (r"&", " "),
    (r"\\text\{\s*,?\s*(?:if|and|for|or)?\s*", r"\\text{"),
]

STRIP_WRAPPERS = r"\\(?:mathrm|text|operatorname|mathbf|boldsymbol|bm|mathit|vec|mathbb|mathcal|textbf|textit|hat|tilde)"


def norm(s: str, loose: bool) -> str:
    s = s.strip()
    s = re.sub(r"^\\text\{[^{}]*:\s*\}", "", s)
    for pat, rep in SUBS:
        s = re.sub(pat, rep, s)
    if loose:
        s = re.sub(r"\\xrightarrow\{((?:[^{}]|\{[^{}]*\})*)\}", r"\\overset{\1}{->}", s)
        s = re.sub(r"\\stackrel", r"\\overset", s)
        # drop font / decoration wrappers but keep content
        for _ in range(3):
            s = re.sub(STRIP_WRAPPERS + r"\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(STRIP_WRAPPERS + r"(?![a-zA-Z])", "", s)
        s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    if loose:
        s = re.sub(r"[;,.]|\\(?![a-zA-Z])", "", s)
    s = s.rstrip(",.;")
    return s


def prose(md: str) -> str:
    t = DISPLAY_RE.sub(" ", md)
    t = re.sub(r"\$[^$]*\$", " M ", t)
    t = re.sub(r"[#*>_`\[\]■—-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def main():
    cdir, pdir, out = map(Path, sys.argv[1:4])
    report = {"pages": {}}
    tot = {"labels_claude": 0, "labels_primary": 0, "matched_strict": 0,
           "matched_loose_only": 0, "differ": 0, "missing_in_primary": 0,
           "missing_in_claude": 0}
    for cf in sorted(cdir.glob("page_*.md")):
        pf = pdir / cf.name.replace("_cross", "")
        c = cf.read_text(encoding="utf-8")
        p = pf.read_text(encoding="utf-8") if pf.exists() else ""
        ce, pe = labelled_equations(c), labelled_equations(p)
        page = {"labels_claude": sorted(ce), "labels_primary": sorted(pe),
                "missing_in_primary": sorted(set(ce) - set(pe)),
                "missing_in_claude": sorted(set(pe) - set(ce)),
                "equations": {}}
        for lab in sorted(set(ce) & set(pe)):
            cs, ps = norm(ce[lab], False), norm(pe[lab], False)
            cl, pl = norm(ce[lab], True), norm(pe[lab], True)
            if cs == ps:
                status = "match"
                tot["matched_strict"] += 1
            elif cl == pl:
                status = "notation_only"
                tot["matched_loose_only"] += 1
            else:
                status = "differ"
                tot["differ"] += 1
            entry = {"status": status,
                     "loose_ratio": round(difflib.SequenceMatcher(None, cl, pl).ratio(), 3)}
            if status != "match":
                entry["claude"] = ce[lab].strip()
                entry["primary"] = pe[lab].strip()
                entry["claude_loose"] = cl
                entry["primary_loose"] = pl
            page["equations"][lab] = entry
        page["prose_ratio"] = round(difflib.SequenceMatcher(None, prose(c), prose(p)).ratio(), 3)
        tot["labels_claude"] += len(ce)
        tot["labels_primary"] += len(pe)
        tot["missing_in_primary"] += len(page["missing_in_primary"])
        tot["missing_in_claude"] += len(page["missing_in_claude"])
        report["pages"][cf.stem.replace("_cross", "")] = page
    report["totals"] = tot
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot, indent=1))
    for k, pg in report["pages"].items():
        d = [l for l, e in pg["equations"].items() if e["status"] == "differ"]
        n = [l for l, e in pg["equations"].items() if e["status"] == "notation_only"]
        print(f"{k}: prose={pg['prose_ratio']} differ={d} notation={n} "
              f"miss_primary={pg['missing_in_primary']} miss_claude={pg['missing_in_claude']}")


if __name__ == "__main__":
    main()
