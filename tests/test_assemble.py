"""
Unit tests for errata application and chapter assembly in learn_assist.assemble.
Tests:
- An approved entry is applied with a blockquote note after the enclosing unit
- A pending entry is not applied (neither text changed nor note added)
- A rejected entry is not applied
- A judgement entry adds a note > **Possible erratum ... but leaves text unchanged
- A printed string that is missing raises an error
- A printed string that appears twice raises an error
- Assembly produces both chapters_printed/ and chapters/
- No note text ever appears between a $$ pair
- Every assembled chapter has an even number of $$ delimiters, and every display block is balanced
- Each equation label appears exactly once per chapter in data/chapters/ (same count as data/chapters_printed/)
- A multi-block erratum (like E10) produces exactly one note after its last block
- A mid-equation erratum (like E17) leaves the equation in one piece
"""

import re
import json
import pytest
from pathlib import Path
from learn_assist.assemble import apply_errata, run_assembly


def test_approved_erratum_applied_with_note():
    text = r"We note that P(p, \theta) is the Radon transform."
    errata = [
        {
            "id": "E03",
            "pdf_page": 56,
            "label": "2.99",
            "printed": r"P(p, \theta)",
            "corrected": r"P(p, \phi)",
            "basis": "Angle is defined as phi, not theta.",
            "kind": "certain",
            "status": "approved"
        }
    ]
    result = apply_errata(text, errata, chapter_num=2)
    expected = (
        "We note that P(p, \\phi) is the Radon transform.\n\n"
        "> **Erratum E03** (corrected above). The printed book has:\n"
        "> ```latex\n"
        "> P(p, \\theta)\n"
        "> ```"
    )
    assert result == expected


def test_pending_erratum_not_applied():
    text = r"We note that P(p, \theta) is the Radon transform."
    errata = [
        {
            "id": "E03",
            "pdf_page": 56,
            "label": "2.99",
            "printed": r"P(p, \theta)",
            "corrected": r"P(p, \phi)",
            "basis": "Angle is defined as phi, not theta.",
            "kind": "certain",
            "status": "pending_review"
        }
    ]
    result = apply_errata(text, errata, chapter_num=2)
    assert result == text


def test_rejected_erratum_not_applied():
    text = "Equation (3.25) can be expressed as"
    errata = [
        {
            "id": "E12",
            "pdf_page": 83,
            "label": None,
            "printed": "Equation (3.25) can be expressed as",
            "corrected": "Equation (3.16) can be expressed as",
            "basis": "Not an erratum. Eq. (3.21) in matrix form is exactly Eq. (3.25), so the forward reference is correct.",
            "kind": "certain",
            "status": "rejected"
        }
    ]
    result = apply_errata(text, errata, chapter_num=3)
    assert result == text


def test_judgement_erratum_adds_note_leaves_text_unchanged():
    text = "This frequency shift disappears when the $B_1$ field is turned on."
    errata = [
        {
            "id": "E15",
            "pdf_page": 91,
            "label": None,
            "printed": "This frequency shift disappears when the $B_1$ field is turned on.",
            "corrected": "This frequency shift disappears when the $B_1$ field is turned off.",
            "basis": "The Bloch-Siegert shift is induced by the B_1 field and disappears when B_1 is turned off.",
            "kind": "judgement",
            "status": "approved"
        }
    ]
    result = apply_errata(text, errata, chapter_num=3)
    expected = (
        "This frequency shift disappears when the $B_1$ field is turned on.\n\n"
        "> **Possible erratum E15** (text left as printed): The Bloch-Siegert shift is induced by the B_1 field and disappears when B_1 is turned off."
    )
    assert result == expected


def test_printed_string_missing_raises_error():
    text = "Text without the target string."
    errata = [
        {
            "id": "E01",
            "pdf_page": 52,
            "label": "2.89",
            "printed": "nonexistent string",
            "corrected": "replacement",
            "basis": "Some basis",
            "kind": "certain",
            "status": "approved"
        }
    ]
    with pytest.raises(ValueError, match="printed.*not found"):
        apply_errata(text, errata, chapter_num=2)


def test_printed_string_appears_twice_raises_error():
    text = "Duplicate target. Duplicate target."
    errata = [
        {
            "id": "E01",
            "pdf_page": 52,
            "label": "2.89",
            "printed": "Duplicate target.",
            "corrected": "replacement",
            "basis": "Some basis",
            "kind": "certain",
            "status": "approved"
        }
    ]
    with pytest.raises(ValueError, match="expected exactly 1"):
        apply_errata(text, errata, chapter_num=2)


def test_run_assembly_writes_both_chapters_and_chapters_printed(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    chapters_dir = tmp_path / "chapters"
    printed_dir = tmp_path / "chapters_printed"
    problems_dir = tmp_path / "problems"
    pages_dir.mkdir()

    for p in range(33, 126):
        (pages_dir / f"page_{p:04d}.md").write_text(
            f"Content of page {p}.\n$$ x = 1 \\quad (2.1) $$\n## Exercises\n2.1 First problem."
        )

    errata = [
        {
            "id": "E01",
            "pdf_page": 33,
            "label": None,
            "printed": "Content of page 33.",
            "corrected": "Corrected content of page 33.",
            "basis": "Typo on page 33",
            "kind": "certain",
            "status": "approved"
        }
    ]
    errata_file = tmp_path / "errata.json"
    errata_file.write_text(json.dumps(errata))

    run_assembly(
        pages_dir=pages_dir,
        chapters_dir=chapters_dir,
        printed_dir=printed_dir,
        problems_dir=problems_dir,
        errata_path=errata_file
    )

    ch2_printed = (printed_dir / "chapter_02.md").read_text()
    assert "Content of page 33." in ch2_printed
    assert "Corrected content of page 33." not in ch2_printed
    assert "**Erratum E01**" not in ch2_printed

    ch2_corrected = (chapters_dir / "chapter_02.md").read_text()
    assert "Corrected content of page 33." in ch2_corrected
    assert "> **Erratum E01** (corrected above). The printed book has:" in ch2_corrected


def test_no_note_between_dollar_pairs():
    for ch in [2, 3]:
        ch_path = Path(f"data/chapters/chapter_{ch:02d}.md")
        if not ch_path.exists():
            continue
        text = ch_path.read_text(encoding="utf-8")
        parts = text.split("$$")
        math_chunks = parts[1::2]
        for i, chunk in enumerate(math_chunks):
            assert "**Erratum" not in chunk, f"Found erratum note in display math chunk {i} in Chapter {ch}"
            assert "**Possible erratum" not in chunk, f"Found possible erratum in display math chunk {i} in Chapter {ch}"


def test_even_balanced_display_math_delimiters():
    for ch in [2, 3]:
        ch_path = Path(f"data/chapters/chapter_{ch:02d}.md")
        if not ch_path.exists():
            continue
        text = ch_path.read_text(encoding="utf-8")
        count = text.count("$$")
        assert count % 2 == 0, f"Chapter {ch} has odd number of $$ delimiters ({count})"


def test_equation_labels_appear_exactly_once():
    pat = re.compile(r"\\quad\s*\((\d+\.\d+[a-z]?)\)")
    for ch in [2, 3]:
        corr_path = Path(f"data/chapters/chapter_{ch:02d}.md")
        print_path = Path(f"data/chapters_printed/chapter_{ch:02d}.md")
        if not corr_path.exists() or not print_path.exists():
            continue
        corr_text = corr_path.read_text(encoding="utf-8")
        print_text = print_path.read_text(encoding="utf-8")
        corr_labels = pat.findall(corr_text)
        print_labels = pat.findall(print_text)
        assert corr_labels == print_labels, f"Labels in Chapter {ch} do not match printed labels exactly"
        assert len(corr_labels) == len(set(corr_labels)), f"Duplicate equation label in Chapter {ch}"


def test_multiblock_erratum_produces_single_note_after_last_block():
    text = (
        "Here is a derivation:\n"
        "$$\n"
        "= \\text{step 1}\n"
        "$$\n"
        "$$\n"
        "= \\text{step 2}\n"
        "$$\n"
        "$$\n"
        "= \\text{step 3}\n"
        "$$\n"
        "Following text."
    )
    errata = [
        {
            "id": "E10",
            "pdf_page": 71,
            "label": None,
            "printed": "= \\text{step 1}\n$$\n$$\n= \\text{step 2}\n$$\n$$\n= \\text{step 3}",
            "corrected": "= \\text{corrected 1}\n$$\n$$\n= \\text{corrected 2}\n$$\n$$\n= \\text{corrected 3}",
            "basis": "Limits should be corrected.",
            "kind": "certain",
            "status": "approved"
        }
    ]
    result = apply_errata(text, errata, chapter_num=2)
    assert result.count("$$") == text.count("$$")
    assert result.count("**Erratum E10**") == 1
    parts = result.split("$$")
    assert len(parts) == 7
    assert "= \\text{corrected 3}" in parts[5]
    assert "**Erratum E10**" in parts[6]


def test_mid_equation_erratum_leaves_equation_in_one_piece():
    text = (
        "Phase shift equation:\n"
        "$$\n"
        "\\tan \\varphi_0 = \\frac{M_x(0+)}{M_y(0+)} = \\frac{\\sin \\theta \\cos \\theta (1 - \\cos \\alpha)}{\\sin \\theta \\sin \\alpha} \\\\\n"
        "= \\tan \\frac{\\alpha}{2} \\frac{\\Delta\\omega_0}{\\omega_{\\text{eff}}} \\quad (3.111)\n"
        "$$\n"
        "It is evident from Eq. (3.111)..."
    )
    errata = [
        {
            "id": "E17",
            "pdf_page": 108,
            "label": "3.111",
            "printed": "\\tan \\varphi_0 = \\frac{M_x(0+)}{M_y(0+)}",
            "corrected": "\\tan \\varphi_0 = \\frac{M_{x'}(0+)}{M_{y'}(0+)}",
            "basis": "Ratio should be primed.",
            "kind": "probable",
            "status": "approved"
        }
    ]
    result = apply_errata(text, errata, chapter_num=3)
    parts = result.split("$$")
    assert len(parts) == 3
    eq_body = parts[1]
    assert "\\tan \\varphi_0 = \\frac{M_{x'}(0+)}{M_{y'}(0+)} = \\frac{\\sin \\theta" in eq_body
    assert "**Erratum E17**" not in eq_body
    assert "**Erratum E17**" in parts[2]
