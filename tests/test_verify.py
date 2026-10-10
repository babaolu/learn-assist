"""
Unit tests for the verification suite in learn_assist.verify.
Tests check logic using synthetic fixtures with deliberate faults:
- Missing and empty checkpoint pages
- Equation gaps and unexpected numbers
- Section, figure, and table sequence gaps
- Problem sequence gaps and duplicates
- Unbalanced LaTeX math delimiters ($ and $$)
- Corrupted OCR artifact signatures
"""

import pytest
from pathlib import Path
from learn_assist.verify import (
    check_completeness,
    check_sequence_contiguity,
    check_problem_sequence,
    check_latex_integrity,
    extract_equation_numbers,
    CORRUPT_OCR_PATTERNS,
)


def test_completeness_passes_when_all_present(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    # Create valid synthetic pages for 33..125
    for p in range(33, 126):
        (pages_dir / f"page_{p:04d}.md").write_text("Valid page content with sufficient characters. " * 10)

    passed, failures, counts = check_completeness(pages_dir, min_chars=100)
    assert passed is True
    assert len(failures) == 0
    assert len(counts) == 93


def test_completeness_catches_missing_and_empty_pages(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for p in range(33, 126):
        if p == 45:
            continue  # Deliberately missing page 45
        elif p == 50:
            (pages_dir / f"page_{p:04d}.md").write_text("")  # Empty page
        elif p == 60:
            (pages_dir / f"page_{p:04d}.md").write_text("Short")  # Below threshold
        else:
            (pages_dir / f"page_{p:04d}.md").write_text("Valid content. " * 20)

    passed, failures, counts = check_completeness(pages_dir, min_chars=50)
    assert passed is False
    assert any("45" in f and "MISSING" in f for f in failures)
    assert any("50" in f and "EMPTY" in f for f in failures)
    assert any("60" in f and "below threshold" in f for f in failures)


def test_sequence_contiguity_catches_gaps_and_extras():
    # Contiguous sequence: 1..10
    passed, fails = check_sequence_contiguity("TestSeq", list(range(1, 11)), expected_start=1, expected_end=10)
    assert passed is True
    assert len(fails) == 0

    # Sequence with deliberate gap (missing 5)
    passed_gap, fails_gap = check_sequence_contiguity("TestSeq", [1, 2, 3, 4, 6, 7, 8, 9, 10], expected_start=1, expected_end=10)
    assert passed_gap is False
    assert any("gaps" in f and "5" in f for f in fails_gap)

    # Sequence with extra number (11 when 1..10 expected)
    passed_extra, fails_extra = check_sequence_contiguity("TestSeq", list(range(1, 12)), expected_start=1, expected_end=10)
    assert passed_extra is False
    assert any("outside" in f and "11" in f for f in fails_extra)


def test_equation_number_extraction_and_lettering():
    sample_text = """
    We have the linear system:
    $$
    Ax = b \\quad (2.1)
    $$
    And decomposed components:
    $$
    A_1 x = b_1 \\quad (2.2a)
    $$
    $$
    A_2 x = b_2 \\quad (2.2b)
    $$
    Referencing Eq. (2.3) later in text.
    """
    base_ints, raw_tags = extract_equation_numbers(sample_text, 2)
    assert 1 in base_ints
    assert 2 in base_ints
    assert 3 in base_ints
    assert "2.2a" in raw_tags
    assert "2.2b" in raw_tags


def test_problem_sequence_catches_gaps_and_duplicates(tmp_path: Path):
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()

    # Good chapter 2: 1..32
    good_ch2 = [{"item_number": i, "number": f"2.{i}", "full_text": f"Problem 2.{i}"} for i in range(1, 33)]
    (problems_dir / "chapter_02.json").write_text(__import__("json").dumps(good_ch2))

    # Bad chapter 3: missing 5, duplicate 10
    bad_ch3 = [{"item_number": i, "number": f"3.{i}", "full_text": f"Problem 3.{i}"} for i in range(1, 42) if i != 5]
    bad_ch3.append({"item_number": 10, "number": "3.10", "full_text": "Duplicate problem 3.10"})
    (problems_dir / "chapter_03.json").write_text(__import__("json").dumps(bad_ch3))

    passed, failures, stats = check_problem_sequence(problems_dir)
    assert passed is False
    assert any("Chapter 03" in f and "Missing" in f and "5" in f for f in failures)
    assert any("Chapter 03" in f and "Duplicate" in f and "10" in f for f in failures)


def test_latex_integrity_catches_unbalanced_delimiters(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    # Create dummy in-scope pages
    for p in range(33, 126):
        (pages_dir / f"page_{p:04d}.md").write_text("Normal content with $x = 1$ and $$y = 2$$.")

    # Corrupt page 40 with unbalanced inline $
    (pages_dir / "page_0040.md").write_text("Unbalanced inline $x = 1 without closing.")

    # Corrupt page 41 with unbalanced display $$
    (pages_dir / "page_0041.md").write_text("Unbalanced display $$E = mc^2 without closing.")

    passed, failures, stats = check_latex_integrity(pages_dir)
    assert passed is False
    assert any("Page 40" in f and "Unbalanced inline $" in f for f in failures)
    assert any("Page 41" in f and "Unbalanced $$" in f for f in failures)


def test_latex_integrity_catches_corrupted_ocr_signatures(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    for p in range(33, 126):
        (pages_dir / f"page_{p:04d}.md").write_text("Clean mathematical text.")

    # Inject corrupted OCR tokens into page 61
    (pages_dir / "page_0061.md").write_text("Here we have {R2p}(q, (f>,z) and fi2 corrupted.")

    passed, failures, stats = check_latex_integrity(pages_dir)
    assert passed is False
    assert any("Page 61" in f and "(f>" in f for f in failures)
    assert any("Page 61" in f and "fi2" in f for f in failures)
    assert any("Page 61" in f and "{R2p}" in f for f in failures)


def test_latex_integrity_catches_mid_word_start(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    for p in range(33, 126):
        (pages_dir / f"page_{p:04d}.md").write_text("Properly capitalized start.")

    # Inject hyphenated mid-word start
    (pages_dir / "page_0070.md").write_text("-cession of the nuclei around the magnetic field.")

    passed, failures, stats = check_latex_integrity(pages_dir)
    assert passed is False
    assert any("Page 70" in f and "mid-word" in f for f in failures)
