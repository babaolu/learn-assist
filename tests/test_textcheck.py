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
