"""Unit tests for learn_assist.textcheck deterministic running-text comparison."""

import pytest
from learn_assist.textcheck import extract_prose, diff_page, is_high_priority


def test_hyphenation_split_is_not_a_candidate():
    """Hyphenation split ('two-dimensional' vs 'two dimensional') is dropped."""
    primary_md = "This is a two-dimensional Fourier transform."
    cross_md = "This is a two dimensional Fourier transform."

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_page", pw, pm, cw, cm)
    assert len(candidates) == 0


def test_punctuation_only_change_is_not_a_candidate():
    """Punctuation-only difference ('word,' vs 'word') is dropped."""
    primary_md = "In this situation, we consider the vector."
    cross_md = "In this situation we consider the vector."

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_page", pw, pm, cw, cm)
    assert len(candidates) == 0


def test_negation_change_is_high_priority_candidate():
    """'provided the force is removed' vs 'provided that no force is removed' is a high-priority candidate."""
    primary_md = "The system recovers provided the force is removed."
    cross_md = "The system recovers provided that no force is removed."

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_page", pw, pm, cw, cm)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["priority"] == "high"
    assert "no" in cand["cross_diff"].lower()


def test_changed_inline_math_is_inline_math_candidate():
    """Changed inline math $w_M$ vs $\\omega_M$ is detected as an inline_math candidate."""
    primary_md = "The resonance frequency is given by $w_M$ in this system."
    cross_md = r"The resonance frequency is given by $\omega_M$ in this system."

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_page", pw, pm, cw, cm)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["kind"] == "inline_math"
    assert "$w_M$" in cand["primary_diff"]
    assert r"$\omega_M$" in cand["cross_diff"]


def test_figure_line_present_only_in_cross_is_not_a_candidate():
    """A [Figure: ...] line present only in cross transcription is stripped and not a candidate."""
    primary_md = """
Vector addition can be performed graphically using the parallelogram rule.

**Figure 2.2** Parallelogram rule for vector addition.
"""
    cross_md = """
Vector addition can be performed graphically using the parallelogram rule.

[Figure: parallelogram with vectors $\\vec{A}$ along the base and $\\vec{B}$ up the side.]

**Figure 2.2** Parallelogram rule for vector addition.
"""

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_page", pw, pm, cw, cm)
    assert len(candidates) == 0


def test_long_page_single_word_diff_gives_exactly_one_candidate():
    """A long page whose only difference is one word -> exactly 1 candidate (autojunk=False)."""
    base_words = [
        "the", "magnetic", "field", "is", "applied", "along", "the", "z", "axis",
        "and", "the", "system", "undergoes", "precession", "with", "frequency",
        "proportional", "to", "the", "field", "strength", "as", "described", "in",
        "the", "text", "for", "nuclear", "magnetic", "resonance"
    ] * 10  # 300 words

    p_words = list(base_words)
    c_words = list(base_words)
    p_words[150] = "clockwise"
    c_words[150] = "counterclockwise"

    primary_md = " ".join(p_words)
    cross_md = " ".join(c_words)

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_long", pw, pm, cw, cm)
    assert len(candidates) == 1
    assert candidates[0]["primary_diff"] == "clockwise"
    assert candidates[0]["cross_diff"] == "counterclockwise"


def test_labeled_inline_equation_vs_display_block_is_zero_candidates():
    """Primary 'laws: $a+b=b+a \\quad (2.14a)$' vs cross with the same equation as a $$ block -> 0 candidates."""
    primary_md = """
Vector addition obeys the commutative law:
laws: $a+b=b+a \\quad (2.14a)$
This is well known.
"""
    cross_md = """
Vector addition obeys the commutative law:
$$
a+b=b+a \\quad (2.14a)
$$
This is well known.
"""

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_labeled", pw, pm, cw, cm)
    assert len(candidates) == 0


def test_constants_line_vs_display_block_is_zero_candidates():
    """Primary line '$\\gamma = 42.58$ $h = 6.6$' vs cross with an aligned $$ block -> 0 candidates."""
    primary_md = """
Substituting in the following values:
$\\gamma = 42.58$ $h = 6.6$
we obtain the result.
"""
    cross_md = """
Substituting in the following values:
$$
\\begin{aligned}
\\gamma &= 42.58 \\\\
h &= 6.6
\\end{aligned}
$$
we obtain the result.
"""

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_constants", pw, pm, cw, cm)
    assert len(candidates) == 0


def test_figure_description_in_primary_is_zero_candidates():
    """'(Figure description: a plot...)' only in primary -> 0 candidates."""
    primary_md = """
Calculate the Fourier transform of the trapezoidal pulse in the following figure using the shift theorem.
(Figure description: A plot of $\\rho(x)$ versus $x$ with marks at $-b, -a, a, b$.)
Calculate the Fourier transform of the following pulse.
"""
    cross_md = """
Calculate the Fourier transform of the trapezoidal pulse in the following figure using the shift theorem.
Calculate the Fourier transform of the following pulse.
"""

    pw, pm = extract_prose(primary_md)
    cw, cm = extract_prose(cross_md)

    candidates = diff_page("test_fig_desc", pw, pm, cw, cm)
    assert len(candidates) == 0
