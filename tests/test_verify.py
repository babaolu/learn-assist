"""
Unit tests for the verification suite in learn_assist.verify and learn_assist.anchors.
Tests check logic using synthetic fixtures with deliberate faults:
- Missing and empty checkpoint pages
- Equation gaps and unexpected numbers
- Letter suffix gaps, bare & lettered conflicts, duplicate equations
- Equation numbers in parentheses that must not be read as sections
- Real structure of Page 50: reports section 2.5.2 and NOT sections 75-80
- Cross-model outcomes: AGREE, DISAGREE, and UNVERIFIED (timeouts must NOT be agreement or disagreement)
- Cross-model vendor verification: same-vendor cross-transcriptions stop counting toward coverage
- Sub-equation sweep: catches equation jumps > 1 and bare labels where lettered exist
- Section and subsection hierarchy contiguity
- Problem sequence gaps and duplicates
- Unbalanced LaTeX math delimiters ($ and $$)
- Corruption checks: Unicode replacement character, control characters, stray angle brackets outside LaTeX, unbalanced math braces
- Three-way comparison checks: truncation, invented content, derived lossy gap with WARN, missing reference fallback
- Chapter start detection: distinguishes "Chapter 3" from "58 Chapter 3"
"""

import json
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
    check_pdf_text_layer_coverage,
    check_three_way_comparison,
    extract_equation_labels,
    extract_equation_numbers,
    extract_section_headings,
    extract_pdf_text_labels,
    select_stratified_sample,
    check_braces_balanced,
    check_stray_angle_brackets,
)
from learn_assist.anchors import is_chapter_start


def test_completeness_passes_when_all_present(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    pages = list(range(33, 126))
    for p in pages:
        (pages_dir / f"page_{p:04d}.md").write_text("Valid page content with sufficient characters. " * 10)

    passed, failures, counts = check_completeness(pages_dir, in_scope_pages=pages, min_chars=100)
    assert passed is True
    assert len(failures) == 0
    assert len(counts) == 93


def test_completeness_catches_missing_and_empty_pages(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    pages = list(range(33, 126))
    for p in pages:
        if p == 45:
            continue
        elif p == 50:
            (pages_dir / f"page_{p:04d}.md").write_text("")
        elif p == 60:
            (pages_dir / f"page_{p:04d}.md").write_text("Short")
        else:
            (pages_dir / f"page_{p:04d}.md").write_text("Valid content. " * 20)

    passed, failures, counts = check_completeness(pages_dir, in_scope_pages=pages, min_chars=50)
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
    gap_labels = [
        (1, "", "2.1", 1, ""),
        (2, "a", "2.2a", 2, ""),
        (2, "c", "2.2c", 3, ""),
        (3, "", "2.3", 4, "")
    ]
    passed, fails, _ = check_equation_sequence(gap_labels, 2, 3)
    assert passed is False
    assert any("Letter suffix gap" in f for f in fails)

    no_a_labels = [
        (1, "", "2.1", 1, ""),
        (2, "b", "2.2b", 2, ""),
        (3, "", "2.3", 3, "")
    ]
    passed_no_a, fails_no_a, _ = check_equation_sequence(no_a_labels, 2, 3)
    assert passed_no_a is False
    assert any("does not start at 'a'" in f for f in fails_no_a)


def test_equation_sequence_catches_duplicates():
    dup_bare = [
        (1, "", "2.1", 1, ""),
        (2, "", "2.2", 2, ""),
        (2, "", "2.2", 3, ""),
        (3, "", "2.3", 4, "")
    ]
    passed, fails, _ = check_equation_sequence(dup_bare, 2, 3)
    assert passed is False
    assert any("Duplicate bare definition" in f for f in fails)

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
    good_headings = ["2.1", "2.2", "2.3", "2.3.1", "2.3.2", "2.4"]
    passed, fails, stats = check_section_hierarchy(good_headings, 2, range(1, 5))
    assert passed is True
    assert len(fails) == 0

    gap_l1 = ["2.1", "2.2", "2.4"]
    passed_l1, fails_l1, _ = check_section_hierarchy(gap_l1, 2, range(1, 5))
    assert passed_l1 is False
    assert any("Missing Level 1" in f and "3" in f for f in fails_l1)

    gap_l2 = ["2.1", "2.2", "2.3", "2.3.1", "2.3.3", "2.4"]
    passed_l2, fails_l2, _ = check_section_hierarchy(gap_l2, 2, range(1, 5))
    assert passed_l2 is False
    assert any("Subsections under 2.3" in f and "missing [2]" in f for f in fails_l2)


def test_sub_equation_sweep_detects_jumps_and_conflicts(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    (pages_dir / "page_0033.md").write_text(r"$$ x = 1 \quad (2.1) $$ $$ x = 2 \quad (2.2) $$")
    (pages_dir / "page_0034.md").write_text(r"$$ x = 3 \quad (2.3) $$ $$ x = 6 \quad (2.6) $$")
    (pages_dir / "page_0035.md").write_text(r"$$ x = 7 \quad (2.7) $$")

    meta = {2: {"chapter_start": {"value": 33}, "chapter_end": {"value": 35}}}
    passed, failures, stats = check_sub_equation_sweep(pages_dir, in_scope_pages=[33, 34, 35], chapters_meta=meta)
    assert passed is False
    assert 34 in stats["flagged_pages"]
    assert any("Page 34" in f and "jumps internally" in f and "3 to 2.6" in f for f in failures)

    (pages_dir / "page_0036.md").write_text(r"$$ x = 8 \quad (2.8) $$")
    (pages_dir / "page_0037.md").write_text(r"$$ x = 8 \quad (2.8a) $$ $$ x = 9 \quad (2.8b) $$")
    meta2 = {2: {"chapter_start": {"value": 36}, "chapter_end": {"value": 37}}}
    passed_b, fails_b, stats_b = check_sub_equation_sweep(pages_dir, in_scope_pages=[36, 37], chapters_meta=meta2)
    assert passed_b is False
    assert 36 in stats_b["flagged_pages"]
    assert any("Page 36" in f and "appears bare, but base 2.8 appears with letter suffixes" in f for f in fails_b)


def test_cross_model_classifies_agree_disagree_unverified(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()

    # Primary pages
    (pages_dir / "page_0040.md").write_text(r"## 2.1 Intro $$ y = 1 \quad (2.1) $$")
    (pages_dir / "page_0041.md").write_text(r"## 2.2 Methods $$ y = 2 \quad (2.2a) $$")
    (pages_dir / "page_0042.md").write_text(r"## 2.3 Results $$ y = 3 \quad (2.3) $$")

    # Cross pages (from non-Google vendor e.g. Anthropic)
    (cross_dir / "page_0040_cross.md").write_text(r"## 2.1 Intro $$ y = 1 \quad (2.1) $$")
    (cross_dir / "page_0041_cross.md").write_text(r"## 2.2 Methods $$ y = 2 \quad (2.2) $$")

    manifest = {
        "page_0040": {"vendor": "anthropic", "model": "claude-3-5-sonnet"},
        "page_0041": {"vendor": "anthropic", "model": "claude-3-5-sonnet"}
    }
    (cross_dir / "manifest.json").write_text(json.dumps(manifest))

    meta = {2: {"chapter_start": {"value": 40}, "chapter_end": {"value": 42}}}
    passed, failures, stats = check_cross_model_spot(
        pages_dir=pages_dir,
        sampled_pages=[40, 41, 42],
        retranscribe=False,
        cross_dir=cross_dir,
        in_scope_pages=[40, 41, 42],
        chapters_meta=meta,
        primary_vendor="google"
    )
    assert passed is False  # Coverage failure because 42 is UNVERIFIED and 41 is DISAGREE
    status_by_page = {r["page"]: r["status"] for r in stats["results"]}
    assert status_by_page[40] == "AGREE"
    assert status_by_page[41] == "DISAGREE"
    assert status_by_page[42] == "UNVERIFIED"
    assert stats["agreed_count"] == 1
    assert stats["disagreed_count"] == 1
    assert stats["unverified_count"] == 1


def test_cross_model_timeout_is_unverified_not_agreement_or_disagreement(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()
    (pages_dir / "page_0050.md").write_text(r"$$ x = 1 \quad (2.1) $$")

    meta = {2: {"chapter_start": {"value": 50}, "chapter_end": {"value": 50}}}
    passed, failures, stats = check_cross_model_spot(
        pages_dir=pages_dir,
        sampled_pages=[50],
        retranscribe=False,
        cross_dir=cross_dir,
        in_scope_pages=[50],
        chapters_meta=meta
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


def test_cross_model_same_vendor_does_not_count_toward_coverage(tmp_path: Path):
    """
    Requirement 6: Cross-transcriptions from the same vendor as primary (e.g. Gemma from Google)
    stop counting toward coverage.
    """
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()

    (pages_dir / "page_0046.md").write_text(r"$$ x = 1 \quad (2.1) $$")
    (cross_dir / "page_0046_cross.md").write_text(r"$$ x = 1 \quad (2.1) $$")

    # Manifest records vendor as google (same as primary)
    manifest = {
        "page_0046": {"vendor": "google", "model": "gemma-4-31b-it"}
    }
    (cross_dir / "manifest.json").write_text(json.dumps(manifest))

    meta = {2: {"chapter_start": {"value": 46}, "chapter_end": {"value": 46}}}
    passed, failures, stats = check_cross_model_spot(
        pages_dir=pages_dir,
        sampled_pages=[46],
        retranscribe=False,
        cross_dir=cross_dir,
        in_scope_pages=[46],
        chapters_meta=meta,
        primary_vendor="google"
    )
    assert passed is False
    assert stats["results"][0]["status"] == "UNVERIFIED"
    assert "matches primary vendor" in stats["results"][0]["reason"]
    assert stats["achieved_coverage_pct"] == 0.0


def test_problem_sequence_catches_gaps_and_duplicates(tmp_path: Path):
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()

    good_ch2 = [{"item_number": i, "number": f"2.{i}", "full_text": f"Problem 2.{i}"} for i in range(1, 33)]
    (problems_dir / "chapter_02.json").write_text(json.dumps(good_ch2))

    bad_ch3 = [{"item_number": i, "number": f"3.{i}", "full_text": f"Problem 3.{i}"} for i in range(1, 42) if i != 5]
    bad_ch3.append({"item_number": 10, "number": "3.10", "full_text": "Duplicate problem 3.10"})
    (problems_dir / "chapter_03.json").write_text(json.dumps(bad_ch3))

    meta = {
        2: {"problems": {"start": 1, "end": 32}},
        3: {"problems": {"start": 1, "end": 41}}
    }
    passed, failures, stats = check_problem_sequence(problems_dir, chapters_meta=meta)
    assert passed is False
    assert any("Chapter 03" in f and "Missing" in f and "5" in f for f in failures)
    assert any("Chapter 03" in f and "Duplicate" in f and "10" in f for f in failures)


# -----------------------------------------------------------------------------
# Requirement 5b: Corruption Class Checks
# -----------------------------------------------------------------------------

def test_latex_integrity_clean_page_passes(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0033.md").write_text(
        "Clean markdown page with inline math $x = 1$ and display math:\n$$ y = \\frac{1}{2} $$\nValid."
    )
    passed, failures, stats = check_latex_integrity(pages_dir, in_scope_pages=[33])
    assert passed is True
    assert len(failures) == 0


def test_latex_integrity_catches_unicode_replacement_character(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0033.md").write_text("Text with corrupted char \ufffd here.")
    passed, failures, _ = check_latex_integrity(pages_dir, in_scope_pages=[33])
    assert passed is False
    assert any("Unicode replacement character" in f for f in failures)


def test_latex_integrity_catches_control_characters(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0033.md").write_text("Text with null byte \x00 and bell \x07.")
    passed, failures, _ = check_latex_integrity(pages_dir, in_scope_pages=[33])
    assert passed is False
    assert any("Control characters detected" in f for f in failures)


def test_latex_integrity_catches_stray_angle_bracket_digraph(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    # <t> outside LaTeX should fail
    (pages_dir / "page_0033.md").write_text("Corrupted signal s<t> in prose.")
    passed, failures, _ = check_latex_integrity(pages_dir, in_scope_pages=[33])
    assert passed is False
    assert any("Stray angle-bracket digraph" in f for f in failures)

    # But <t> or inequality inside LaTeX math is valid
    (pages_dir / "page_0034.md").write_text("Valid inside math: $$ s<t> = 1 $$ and $a < b$.")
    passed_m, failures_m, _ = check_latex_integrity(pages_dir, in_scope_pages=[34])
    assert passed_m is True
    assert len(failures_m) == 0


def test_latex_integrity_catches_unbalanced_braces_in_math(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0033.md").write_text(r"$$ \frac{1}{2 $$")
    passed, failures, _ = check_latex_integrity(pages_dir, in_scope_pages=[33])
    assert passed is False
    assert any("Unbalanced braces" in f for f in failures)

    # Escaped \{ and \} do not trigger failure
    (pages_dir / "page_0034.md").write_text(r"$$ \{ x \mid x > 0 \} $$")
    passed_esc, failures_esc, _ = check_latex_integrity(pages_dir, in_scope_pages=[34])
    assert passed_esc is True
    assert len(failures_esc) == 0


def test_latex_integrity_catches_mid_word_start(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0070.md").write_text("-cession of the nuclei around the magnetic field.")
    passed, failures, stats = check_latex_integrity(pages_dir, in_scope_pages=[70])
    assert passed is False
    assert any("Page 70" in f and "mid-word" in f for f in failures)


# -----------------------------------------------------------------------------
# Requirement 4: Synthetic Fixture Tests for Three-Way Comparison & Chapter Start
# -----------------------------------------------------------------------------

def test_three_way_transcription_truncation_fails(tmp_path: Path):
    """Transcription ends at 3.152, reference says 3.154 → FAIL (truncation)."""
    ch_dir = tmp_path / "chapters"
    ch_dir.mkdir()
    prob_dir = tmp_path / "problems"
    prob_dir.mkdir()

    # Transcription ends at 3.152
    ch3_text = "\n".join([f"$$ x = {i} \\quad (3.{i}) $$" for i in range(1, 153)])
    ch3_text += "\n## 3.1 Sec 1\nFigure 3.1\n"
    (ch_dir / "chapter_03.md").write_text(ch3_text)

    probs = [{"item_number": i} for i in range(1, 42)]
    (prob_dir / "chapter_03.json").write_text(json.dumps(probs))

    ref = {
        3: {
            "chapter_start": {"value": 77},
            "chapter_end": {"value": 125},
            "exercises_start": {"value": 121},
            "last_equation": {"value": 154, "label": "3.154", "page": 120},
            "problems": {"start": 1, "end": 41, "count": 41},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }
    derived = {
        3: {
            "chapter_start": {"value": 77},
            "chapter_end": {"value": 125},
            "exercises_start": {"value": 121},
            "last_equation": {"value": 153, "label": "3.153", "page": 120},
            "problems": {"start": 1, "end": 41, "count": 41},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }

    passed, failures, warnings, stats = check_three_way_comparison(
        ref_chapters=ref,
        derived_chapters=derived,
        chapters_dir=ch_dir,
        problems_dir=prob_dir
    )
    assert passed is False
    assert any("Truncation" in f and "152" in f and "154" in f for f in failures)


def test_three_way_transcription_invented_content_fails(tmp_path: Path):
    """Transcription has 3.155 beyond reference 3.154 → FAIL (invented content)."""
    ch_dir = tmp_path / "chapters"
    ch_dir.mkdir()
    prob_dir = tmp_path / "problems"
    prob_dir.mkdir()

    # Transcription has 3.155
    ch3_text = "\n".join([f"$$ x = {i} \\quad (3.{i}) $$" for i in range(1, 156)])
    ch3_text += "\n## 3.1 Sec 1\nFigure 3.1\n"
    (ch_dir / "chapter_03.md").write_text(ch3_text)

    probs = [{"item_number": i} for i in range(1, 42)]
    (prob_dir / "chapter_03.json").write_text(json.dumps(probs))

    ref = {
        3: {
            "chapter_start": {"value": 77},
            "chapter_end": {"value": 125},
            "exercises_start": {"value": 121},
            "last_equation": {"value": 154, "label": "3.154", "page": 120},
            "problems": {"start": 1, "end": 41, "count": 41},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }
    derived = {
        3: {
            "chapter_start": {"value": 77},
            "chapter_end": {"value": 125},
            "exercises_start": {"value": 121},
            "last_equation": {"value": 153, "label": "3.153", "page": 120},
            "problems": {"start": 1, "end": 41, "count": 41},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }

    passed, failures, warnings, stats = check_three_way_comparison(
        ref_chapters=ref,
        derived_chapters=derived,
        chapters_dir=ch_dir,
        problems_dir=prob_dir
    )
    assert passed is False
    assert any("Invented content" in f and "155" in f and "154" in f for f in failures)


def test_three_way_derived_gap_warns_not_fails(tmp_path: Path):
    """Derived 3.153, reference and transcription 3.154 → PASS with WARN recorded."""
    ch_dir = tmp_path / "chapters"
    ch_dir.mkdir()
    prob_dir = tmp_path / "problems"
    prob_dir.mkdir()

    # Transcription has 3.154 (matches reference)
    ch3_text = "\n".join([f"$$ x = {i} \\quad (3.{i}) $$" for i in range(1, 155)])
    ch3_text += "\n## 3.1 Sec 1\nFigure 3.1\n"
    (ch_dir / "chapter_03.md").write_text(ch3_text)

    probs = [{"item_number": i} for i in range(1, 42)]
    (prob_dir / "chapter_03.json").write_text(json.dumps(probs))

    ref = {
        3: {
            "chapter_start": {"value": 77},
            "chapter_end": {"value": 125},
            "exercises_start": {"value": 121},
            "last_equation": {"value": 154, "label": "3.154", "page": 120},
            "problems": {"start": 1, "end": 41, "count": 41},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }
    derived = {
        3: {
            "chapter_start": {"value": 77},
            "chapter_end": {"value": 125},
            "exercises_start": {"value": 121},
            "last_equation": {"value": 153, "label": "3.153", "page": 120},
            "problems": {"start": 1, "end": 41, "count": 41},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }

    passed, failures, warnings, stats = check_three_way_comparison(
        ref_chapters=ref,
        derived_chapters=derived,
        chapters_dir=ch_dir,
        problems_dir=prob_dir
    )
    assert passed is True
    assert len(failures) == 0
    assert len(warnings) == 1
    assert any("PDF 120" in w and "153" in w and "154" in w for w in warnings)


def test_three_way_reference_absent_falls_back_to_derived(tmp_path: Path):
    """Reference absent for a chapter → falls back to derived, flagged loudly."""
    ch_dir = tmp_path / "chapters"
    ch_dir.mkdir()
    prob_dir = tmp_path / "problems"
    prob_dir.mkdir()

    ch4_text = "$$ x = 1 \\quad (4.1) $$\n## 4.1 Sec 1\nFigure 4.1\n"
    (ch_dir / "chapter_04.md").write_text(ch4_text)
    (prob_dir / "chapter_04.json").write_text(json.dumps([{"item_number": 1}]))

    # Empty reference, derived provided for Chapter 4
    ref = {}
    derived = {
        4: {
            "chapter_start": {"value": 127},
            "chapter_end": {"value": 160},
            "exercises_start": {"value": 158},
            "last_equation": {"value": 1, "label": "4.1", "page": 157},
            "problems": {"start": 1, "end": 1, "count": 1},
            "figures": {"start": 1, "end": 1, "count": 1},
            "sections": {"start": 1, "end": 1, "count": 1}
        }
    }

    passed, failures, warnings, stats = check_three_way_comparison(
        ref_chapters=ref,
        derived_chapters=derived,
        chapters_dir=ch_dir,
        problems_dir=prob_dir
    )
    assert passed is True
    assert any("No reference file available" in w and "Chapter 04" in w for w in warnings)


def test_chapter_start_detection():
    """Chapter-start detection distinguishes 'Chapter 3' from '58 Chapter 3'."""
    start_text = "Chapter 3\nSignal Generation and Detection\nIntro text..."
    header_text = "58 Chapter 3 Signal Generation and Detection\nBody text on interior page..."
    header_num_only = "58\nChapter 3 Signal Generation and Detection\nBody text..."

    assert is_chapter_start(start_text, 3) is True
    assert is_chapter_start(header_text, 3) is False
    assert is_chapter_start(header_num_only, 3) is False
    assert is_chapter_start("Chapter 2\nMath fundamentals", 3) is False


def test_pdf_text_layer_coverage_catches_missing_in_transcription(tmp_path: Path):
    import pymupdf
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    for _ in range(32):
        doc.new_page()
    p33 = doc.new_page()
    p33.insert_text((50, 100), "Formula here \n(2.1)\n")
    doc.save(pdf_path)

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0033.md").write_text("Formula here without tag.")

    meta = {2: {"chapter_start": {"value": 33}, "chapter_end": {"value": 33}}}
    passed, failures, stats = check_pdf_text_layer_coverage(
        pdf_path=pdf_path,
        pages_dir=pages_dir,
        in_scope_pages=[33],
        chapters_meta=meta
    )
    assert passed is False
    assert len(failures) == 1
    assert any("2.1" in f and "MISSING" in f for f in failures)
    assert stats["failed_pages"] == [33]


def test_pdf_text_layer_coverage_warns_when_missing_in_pdf(tmp_path: Path):
    import pymupdf
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    for _ in range(32):
        doc.new_page()
    p33 = doc.new_page()
    p33.insert_text((50, 100), "Formula with destroyed OCR text layer")
    doc.save(pdf_path)

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0033.md").write_text(r"$$ y = x \quad (2.1) $$")

    meta = {2: {"chapter_start": {"value": 33}, "chapter_end": {"value": 33}}}
    passed, failures, stats = check_pdf_text_layer_coverage(
        pdf_path=pdf_path,
        pages_dir=pages_dir,
        in_scope_pages=[33],
        chapters_meta=meta
    )
    assert passed is True  # Warning does not cause failure
    assert len(failures) == 0
    assert stats["warning_pages"] == [33]
    assert any("2.1" in w for w in stats["warnings"])


def test_unset_api_key_raises_error(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import learn_assist.ingest as imod
    import learn_assist.verify as vmod
    monkeypatch.setattr(imod, "DEFAULT_API_KEY", None)
    monkeypatch.setattr(vmod, "DEFAULT_API_KEY", None)

    dummy_img = tmp_path / "test.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        imod.transcribe_page_image(dummy_img, api_key=None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        imod.run_ingestion(api_key=None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        vmod.transcribe_image_with_retry(dummy_img, api_key=None)

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0036.md").write_text("dummy")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        vmod.check_cross_model_spot(
            pages_dir=pages_dir,
            sampled_pages=[36],
            retranscribe=True,
            api_key=None,
            cross_dir=tmp_path / "cross"
        )


def test_cross_model_identical_labels_differing_equation_bodies_gives_disagree(tmp_path: Path):
    """
    Check g: Identical equation labels with differing equation bodies gives DISAGREE.
    Excused by adjudication with 'layout_only' or 'ambiguous_print'.
    """
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()

    # Primary page has Eq (2.1) with body: Ax = b
    (pages_dir / "page_0040.md").write_text(r"$$ Ax = b \quad (2.1) $$")
    # Cross page has same label (2.1) but differing body: Ax = c
    (cross_dir / "page_0040_cross.md").write_text(r"$$ Ax = c \quad (2.1) $$")

    manifest = {
        "cross_model": {"vendor": "Anthropic", "model": "claude-opus-5-5"},
        "pages": {"page_0040": {}}
    }
    (cross_dir / "manifest.json").write_text(json.dumps(manifest))

    meta = {2: {"chapter_start": {"value": 40}, "chapter_end": {"value": 40}}}
    passed, failures, stats = check_cross_model_spot(
        pages_dir=pages_dir,
        sampled_pages=[40],
        retranscribe=False,
        cross_dir=cross_dir,
        in_scope_pages=[40],
        chapters_meta=meta,
        primary_vendor="Google"
    )

    assert passed is False
    assert stats["disagreed_count"] == 1
    assert stats["agreed_count"] == 0
    res = stats["results"][0]
    assert res["status"] == "DISAGREE"
    assert res["eq_counts"]["differ"] == 1
    assert "2.1" in res["differing_labels"]

    # When adjudication excuses it with layout_only, status becomes AGREE
    adjudication = {
        "verdicts": [
            {"page": 40, "label": "2.1", "verdict": "layout_only"}
        ]
    }
    (cross_dir / "adjudication.json").write_text(json.dumps(adjudication))
    passed_adj, _, stats_adj = check_cross_model_spot(
        pages_dir=pages_dir,
        sampled_pages=[40],
        retranscribe=False,
        cross_dir=cross_dir,
        in_scope_pages=[40],
        chapters_meta=meta,
        primary_vendor="Google"
    )
    assert passed_adj is True
    assert stats_adj["agreed_count"] == 1
    assert stats_adj["results"][0]["status"] == "AGREE"
    assert stats_adj["results"][0]["eq_counts"]["notation_only"] == 1
    assert stats_adj["results"][0]["eq_counts"]["differ"] == 0


def test_running_head_section_not_counted_as_section():
    """
    extract_section_headings: Heading like '## Section 2.1 Vectors' is skipped as a running head.
    """
    running_head_text = r"""
    ## Section 2.1 Vectors
    This is running head commentary.
    """
    headings = extract_section_headings(running_head_text, 2)
    assert headings == []

    real_sec_text = r"""
    ## 2.1 Vectors
    Section body.
    """
    headings_real = extract_section_headings(real_sec_text, 2)
    assert headings_real == ["2.1"]

    both_text = r"""
    ## Section 2.1 Vectors
    Header line.

    ## 2.1 Vectors
    Section body.
    """
    headings_both = extract_section_headings(both_text, 2)
    assert headings_both == ["2.1"]


def test_cross_model_manifest_vendor_google_gives_fail(tmp_path: Path):
    """
    Check g: A cross manifest with vendor Google gives FAIL.
    """
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    cross_dir = tmp_path / "cross_model"
    cross_dir.mkdir()

    (pages_dir / "page_0046.md").write_text(r"$$ x = 1 \quad (2.1) $$")
    (cross_dir / "page_0046_cross.md").write_text(r"$$ x = 1 \quad (2.1) $$")

    manifest = {
        "cross_model": {"vendor": "Google", "model": "gemini-2.5-flash"},
        "pages": {"page_0046": {}}
    }
    (cross_dir / "manifest.json").write_text(json.dumps(manifest))

    meta = {2: {"chapter_start": {"value": 46}, "chapter_end": {"value": 46}}}
    passed, failures, stats = check_cross_model_spot(
        pages_dir=pages_dir,
        sampled_pages=[46],
        retranscribe=False,
        cross_dir=cross_dir,
        in_scope_pages=[46],
        chapters_meta=meta,
        primary_vendor="Google"
    )

    assert passed is False
    assert any("matches primary vendor" in f for f in failures)
    assert stats["cross_vendor"] == "Google"
    assert stats["achieved_coverage_pct"] == 0.0

