"""
Unit tests for errata application and chapter assembly in learn_assist.assemble.
Tests:
- An approved entry is applied together with its note [Erratum <id>: the printed book has <printed>.]
- A pending entry is not applied (neither text changed nor note added)
- A judgement entry adds a note [Possible erratum <id>: <basis>] but leaves text unchanged
- A printed string that is missing raises an error
- A printed string that appears twice raises an error
- Assembly produces both chapters_printed/ and chapters/
"""

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
    expected = r"We note that P(p, \phi) [Erratum E03: the printed book has P(p, \theta).] is the Radon transform."
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
        "This frequency shift disappears when the $B_1$ field is turned on. "
        "[Possible erratum E15: The Bloch-Siegert shift is induced by the B_1 field and disappears when B_1 is turned off.]"
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

    # Create dummy pages for chapter 2 (33..76) and chapter 3 (77..125)
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

    # Verify chapters_printed has uncorrected content
    ch2_printed = (printed_dir / "chapter_02.md").read_text()
    assert "Content of page 33." in ch2_printed
    assert "Corrected content of page 33." not in ch2_printed
    assert "[Erratum E01:" not in ch2_printed

    # Verify chapters has corrected content + note
    ch2_corrected = (chapters_dir / "chapter_02.md").read_text()
    assert "Corrected content of page 33. [Erratum E01: the printed book has Content of page 33..]" in ch2_corrected
