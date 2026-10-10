"""
Unit tests for the verification suite in learn_assist.verify.
Tests check logic using synthetic fixtures with deliberate faults:
- Missing and empty checkpoint pages
- Equation gaps and unexpected numbers
- Letter suffix gaps, bare & lettered conflicts, duplicate equations
- Equation numbers in parentheses that must not be read as sections
- Real structure of Page 50: reports section 2.5.2 and NOT sections 75-80
- Cross-model outcomes: AGREE, DISAGREE, and UNVERIFIED (timeouts must NOT be agreement or disagreement)
- Sub-equation sweep: catches equation jumps > 1 and bare labels where lettered exist
- Section and subsection hierarchy contiguity
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
    check_equations,
    check_equation_sequence,
    check_sections_and_figures,
    check_section_hierarchy,
    check_cross_model_spot,
    check_sub_equation_sweep,
    extract_equation_labels,
    extract_equation_numbers,
    extract_section_headings,
    CORRUPT_OCR_PATTERNS,
)


def test_completeness_passes_when_all_present(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
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
            continue
        elif p == 50:
            (pages_dir / f"page_{p:04d}.md").write_text("")
        elif p == 60:
            (pages_dir / f"page_{p:04d}.md").write_text("Short")
        else:
            (pages_dir / f"page_{p:04d}.md").write_text("Valid content. " * 20)

    passed, failures, counts = check_completeness(pages_dir, min_chars=50)
    assert passed is False
    assert any("45" in f and "MISSING" in f for f in failures)
    assert any("50" in f and "EMPTY" in f for f in failures)
    assert any("60" in f and "below threshold" in f for f in failures)


def test_sequence_contiguity_catches_gaps_and_extras():
    passed, fails = check_sequence_contiguity("TestSeq", list(range(1, 11)), expected_start=1, expected_end=10)
    assert passed is True
    assert len(fails) == 0

    passed_gap, fails_gap = check_sequence_contiguity("TestSeq", [1, 2, 3, 4, 6, 7, 8, 9, 10], expected_start=1, expected_end=10)
    assert passed_gap is False
    assert any("gaps" in f and "5" in f for f in fails_gap)

    passed_extra, fails_extra = check_sequence_contiguity("TestSeq", list(range(1, 12)), expected_start=1, expected_end=10)
    assert passed_extra is False
    assert any("outside" in f and "11" in f for f in fails_extra)


def test_equation_number_extraction_and_lettering():
    sample_text = r"""
    We have the linear system:
    $$
    Ax = b \quad (2.1)
    $$
    And decomposed components:
    $$
    A_1 x = b_1 \quad (2.2a)
    $$
    $$
    A_2 x = b_2 \quad (2.2\text{b})
    $$
    Referencing Eq. (2.3) later in text.
    """
    labels = extract_equation_labels(sample_text, 2)
    tags = [item[2] for item in labels]
    assert "2.1" in tags
    assert "2.2a" in tags
    assert "2.2b" in tags
    assert "2.3" not in tags  # Citation Eq. (2.3) must NOT be captured as equation definition


def test_equation_sequence_passes_valid_lettered():
    labels = [
        (1, "", "2.1", 1, ""),
        (2, "a", "2.2a", 2, ""),
        (2, "b", "2.2b", 3, ""),
        (3, "", "2.3", 4, "")
    ]
    passed, fails, stats = check_equation_sequence(labels, 2, 3)
    assert passed is True
    assert len(fails) == 0
    assert stats["unique_base_equations"] == 3


def test_equation_sequence_catches_bare_and_lettered_conflict():
    # Base 3 appears both bare and with letter suffix 'a'
    faulty_labels = [
        (1, "", "2.1", 1, ""),
        (2, "", "2.2", 2, ""),
        (3, "", "2.3", 3, ""),
        (3, "a", "2.3a", 4, "")
    ]
    passed, fails, _ = check_equation_sequence(faulty_labels, 2, 3)
    assert passed is False
    assert any("both bare and lettered" in f for f in fails)


def test_equation_sequence_catches_letter_suffix_gap():
    # Base 2 has 'a' and 'c' (missing 'b')
    gap_labels = [
        (1, "", "2.1", 1, ""),
        (2, "a", "2.2a", 2, ""),
        (2, "c", "2.2c", 3, ""),
        (3, "", "2.3", 4, "")
    ]
    passed, fails, _ = check_equation_sequence(gap_labels, 2, 3)
    assert passed is False
    assert any("Letter suffix gap" in f for f in fails)

    # Base 2 starts at 'b' (missing 'a')
    no_a_labels = [
        (1, "", "2.1", 1, ""),
        (2, "b", "2.2b", 2, ""),
        (3, "", "2.3", 3, "")
    ]
    passed_no_a, fails_no_a, _ = check_equation_sequence(no_a_labels, 2, 3)
    assert passed_no_a is False
    assert any("does not start at 'a'" in f for f in fails_no_a)


def test_equation_sequence_catches_duplicates():
    # Duplicate bare definition for 2.2
    dup_bare = [
        (1, "", "2.1", 1, ""),
        (2, "", "2.2", 2, ""),
        (2, "", "2.2", 3, ""),
        (3, "", "2.3", 4, "")
    ]
    passed, fails, _ = check_equation_sequence(dup_bare, 2, 3)
    assert passed is False
    assert any("Duplicate bare definition" in f for f in fails)

    # Duplicate letter suffix 'a' for 2.2
    dup_letter = [
        (1, "", "2.1", 1, ""),
        (2, "a", "2.2a", 2, ""),
        (2, "a", "2.2a", 3, ""),
        (3, "", "2.3", 4, "")
    ]
    passed_l, fails_l, _ = check_equation_sequence(dup_letter, 2, 3)
    assert passed_l is False
    assert any("Duplicate letter suffix" in f for f in fails_l)


def test_equation_parentheses_not_read_as_section():
    sample_text = r"""
    ## 2.1 Vectors
    Consider an equation:
    $$
    \mathbf{A} = (A_x, A_y, A_z) \quad (2.2)
    $$
    And in running text (2.75) appears in parentheses.
    """
    headings = extract_section_headings(sample_text, 2)
    assert headings == ["2.1"]
    assert "2.2" not in headings
    assert "2.75" not in headings


def test_page_50_fixture_structure():
    """
    Synthetic fixture matching page 50's real content:
    - Must report section 2.5.2
    - Must NOT report sections 75-80
    - Must extract equations (2.75), (2.76), (2.77), (2.78), (2.79a), (2.79b), (2.80a), (2.80b)
    """
    page_50_fixture = r"""
    # Chapter 2 Mathematical Fundamentals

    An important property of higher-dimensional Fourier transforms is that they can be expressed as sequential one-dimensional transforms along each dimension. For example, the two-dimensional forward Fourier transform can be written as
    $$
    \{\mathcal{F}\rho\}(k_x, k_y) = \int_{-\infty}^{\infty} \left\{ \int_{-\infty}^{\infty} \rho(x, y)e^{-i2\pi k_x x} dx \right\} e^{-i2\pi k_y y} dy \quad (2.75)
    $$
    This property makes it possible to treat higher-dimensional Fourier transform imaging problems in the one-dimensional setting. In general, the following recursive relationship exists for expressing the $n$-dimensional Fourier transform in terms of lower-dimensional transforms:
    $$
    \mathcal{F}_n = \mathcal{F}_m \mathcal{F}_{n-m} \quad (2.76)
    $$
    where $\mathcal{F}_n$ is the $n$-dimensional Fourier transform operator. As an example, we have $\mathcal{F}_4 = \mathcal{F}_3\mathcal{F}_1 = \mathcal{F}_2\mathcal{F}_2 = \mathcal{F}_1\mathcal{F}_1\mathcal{F}_1\mathcal{F}_1$, which means that the four-dimensional Fourier transform can be decomposed into one three-dimensional transform cascaded with a one-dimensional transform, or into two cascaded two-dimensional transforms, or into four cascaded one-dimensional transforms.

    ## 2.5.2 Properties

    Some properties of the Fourier transform relevant to MRI are summarized below for easy reference. For notational simplicity, we assume $\rho(x) \longleftrightarrow_{\mathcal{F}} S(k)$ whenever appropriate. All the formulas listed are based on the one-dimensional Fourier transform; extension to higher dimensions is in most cases straightforward and left to the reader.

    (a) Uniqueness:
    $$
    \rho_1(x) = \rho_2(x) \longrightarrow S_1(k) = S_2(k) \quad (2.77)
    $$
    (b) Linearity:
    $$
    a\rho_1(x) + b\rho_2(x) \longleftrightarrow aS_1(k) + bS_2(k) \quad (2.78)
    $$
    (c) Shifting theorem:
    $$
    \rho(x - x_0) \longleftrightarrow S(k)e^{-i2\pi k x_0} \quad (2.79\text{a})
    $$
    $$
    e^{i2\pi k_0 x}\rho(x) \longleftrightarrow S(k - k_0) \quad (2.79\text{b})
    $$
    (d) Modulation:
    $$
    \rho(x)\cos(2\pi k_0 x) \longleftrightarrow \frac{1}{2}[S(k + k_0) + S(k - k_0)] \quad (2.80\text{a})
    $$
    $$
    \rho(x)\sin(2\pi k_0 x) \longleftrightarrow \frac{1}{2}i[S(k + k_0) - S(k - k_0)] \quad (2.80\text{b})
    $$
    """
    headings = extract_section_headings(page_50_fixture, 2)
    assert headings == ["2.5.2"]
    for eq_num in range(75, 81):
        assert f"2.{eq_num}" not in headings
        assert str(eq_num) not in headings

    labels = extract_equation_labels(page_50_fixture, 2)
    tags = [item[2] for item in labels]
    assert tags == ["2.75", "2.76", "2.77", "2.78", "2.79a", "2.79b", "2.80a", "2.80b"]


def test_section_hierarchy_contiguity():
    # Good hierarchy
    good_headings = ["2.1", "2.2", "2.3", "2.3.1", "2.3.2", "2.4"]
    passed, fails, stats = check_section_hierarchy(good_headings, 2, range(1, 5))
    assert passed is True
    assert len(fails) == 0

    # Gap in Level 1 (missing 2.3)
    gap_l1 = ["2.1", "2.2", "2.4"]
    passed_l1, fails_l1, _ = check_section_hierarchy(gap_l1, 2, range(1, 5))
    assert passed_l1 is False
    assert any("Missing Level 1" in f and "3" in f for f in fails_l1)

    # Gap in Level 2 (2.3 has 2.3.1 and 2.3.3, missing 2.3.2)
    gap_l2 = ["2.1", "2.2", "2.3", "2.3.1", "2.3.3", "2.4"]
    passed_l2, fails_l2, _ = check_section_hierarchy(gap_l2, 2, range(1, 5))
    assert passed_l2 is False
    assert any("Subsections under 2.3" in f and "missing [2]" in f for f in fails_l2)


def test_sub_equation_sweep_detects_jumps_and_conflicts(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    # Create 3 pages:
    # Page 33: Equation 2.1, 2.2
    # Page 34: Equation 2.3, 2.6 (internal jump from 3 to 6)
    # Page 35: Equation 2.7
    (pages_dir / "page_0033.md").write_text(r"$$ x = 1 \quad (2.1) $$ $$ x = 2 \quad (2.2) $$")
    (pages_dir / "page_0034.md").write_text(r"$$ x = 3 \quad (2.3) $$ $$ x = 6 \quad (2.6) $$")
    (pages_dir / "page_0035.md").write_text(r"$$ x = 7 \quad (2.7) $$")

    passed, failures, stats = check_sub_equation_sweep(pages_dir, in_scope_pages=[33, 34, 35])
    assert passed is False
    assert 34 in stats["flagged_pages"]
    assert any("Page 34" in f and "jumps internally" in f and "3 to 2.6" in f for f in failures)

    # Test bare where lettered exists elsewhere:
    # Page 36 has bare (2.8), but Page 37 has (2.8a) and (2.8b)
    (pages_dir / "page_0036.md").write_text(r"$$ x = 8 \quad (2.8) $$")
    (pages_dir / "page_0037.md").write_text(r"$$ x = 8 \quad (2.8a) $$ $$ x = 9 \quad (2.8b) $$")
    passed_b, fails_b, stats_b = check_sub_equation_sweep(pages_dir, in_scope_pages=[36, 37])
    assert passed_b is False
    assert 36 in stats_b["flagged_pages"]
    assert any("Page 36" in f and "appears bare, but base 2.8 appears with letter suffixes" in f for f in fails_b)


def test_cross_model_classifies_agree_disagree_unverified(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()

    # Mock primary page 40 and cross page 40 (AGREE)
    (pages_dir / "page_0040.md").write_text(r"## 2.1 Intro $$ y = 1 \quad (2.1) $$")
    (cross_dir / "page_0040_cross.md").write_text(r"## 2.1 Intro $$ y = 1 \quad (2.1) $$")

    # Mock primary page 41 and cross page 41 (DISAGREE on equations)
    (pages_dir / "page_0041.md").write_text(r"## 2.2 Methods $$ y = 2 \quad (2.2a) $$")
    (cross_dir / "page_0041_cross.md").write_text(r"## 2.2 Methods $$ y = 2 \quad (2.2) $$")

    # Mock primary page 42 with NO cross file (UNVERIFIED)
    (pages_dir / "page_0042.md").write_text(r"## 2.3 Results $$ y = 3 \quad (2.3) $$")

    import learn_assist.verify as vmod
    orig_scope = vmod.IN_SCOPE_PAGES
    try:
        vmod.IN_SCOPE_PAGES = [40, 41, 42]
        passed, failures, stats = check_cross_model_spot(
            pages_dir=pages_dir,
            sampled_pages=[40, 41, 42],
            retranscribe=False,
            cross_dir=cross_dir
        )
        assert passed is False  # Coverage failure because 42 is UNVERIFIED and 41 is DISAGREE
        status_by_page = {r["page"]: r["status"] for r in stats["results"]}
        assert status_by_page[40] == "AGREE"
        assert status_by_page[41] == "DISAGREE"
        assert status_by_page[42] == "UNVERIFIED"
        assert stats["agreed_count"] == 1
        assert stats["disagreed_count"] == 1
        assert stats["unverified_count"] == 1
    finally:
        vmod.IN_SCOPE_PAGES = orig_scope


def test_cross_model_timeout_is_unverified_not_agreement_or_disagreement(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()
    (pages_dir / "page_0050.md").write_text(r"$$ x = 1 \quad (2.1) $$")

    import learn_assist.verify as vmod
    orig_scope = vmod.IN_SCOPE_PAGES
    try:
        vmod.IN_SCOPE_PAGES = [50]
        # When retranscribe=False and cross file does not exist, status is UNVERIFIED (never AGREE or DISAGREE)
        passed, failures, stats = check_cross_model_spot(
            pages_dir=pages_dir,
            sampled_pages=[50],
            retranscribe=False,
            cross_dir=cross_dir
        )
        assert passed is False  # Coverage failure because page is UNVERIFIED
        results = stats["results"]
        assert len(results) == 1
        res = results[0]
        assert res["status"] == "UNVERIFIED"
        assert res["status"] != "AGREE"
        assert res["status"] != "DISAGREE"
        assert stats["unverified_count"] == 1
        assert stats["agreed_count"] == 0
        assert stats["disagreed_count"] == 0
    finally:
        vmod.IN_SCOPE_PAGES = orig_scope


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

    for p in range(33, 126):
        (pages_dir / f"page_{p:04d}.md").write_text("Normal content with $x = 1$ and $$y = 2$$.")

    (pages_dir / "page_0040.md").write_text("Unbalanced inline $x = 1 without closing.")
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

    (pages_dir / "page_0070.md").write_text("-cession of the nuclei around the magnetic field.")

    passed, failures, stats = check_latex_integrity(pages_dir)
    assert passed is False
    assert any("Page 70" in f and "mid-word" in f for f in failures)
