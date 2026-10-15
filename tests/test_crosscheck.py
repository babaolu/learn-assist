"""Unit tests for learn_assist.crosscheck derivation merging and equation parsing."""
import pytest
from learn_assist.crosscheck import labelled_equations, norm


def test_split_derivation_merges_consecutive_display_blocks_to_match():
    """
    Primary has one block 'a = b \\ = c \\quad (1.1)'.
    Cross has '$$a = b$$' followed by '$$= c \\quad (1.1)$$'.
    When merged, cross status matches primary.
    """
    primary_text = r"""
$$
a = b \\
= c \quad (1.1)
$$
"""
    cross_text = r"""
$$
a = b
$$

$$
= c \quad (1.1)
$$
"""
    pe = labelled_equations(primary_text)
    ce = labelled_equations(cross_text)

    assert "1.1" in pe
    assert "1.1" in ce

    # Strict normalisation matches
    assert norm(pe["1.1"], False) == norm(ce["1.1"], False)
    # Loose normalisation also matches
    assert norm(pe["1.1"], True) == norm(ce["1.1"], True)


def test_unlabelled_block_prose_line_labelled_block_not_merged():
    """
    Unlabelled block, then a prose line, then a labelled block -> not merged.
    Blocks separated by any prose are never merged.
    """
    md_text = r"""
$$
a = b
$$

This is an intervening prose sentence explaining the calculation.

$$
= c \quad (1.1)
$$
"""
    eqs = labelled_equations(md_text)
    assert "1.1" in eqs

    # The equation for (1.1) should only contain '= c \quad ' and NOT include 'a = b'
    body = eqs["1.1"]
    assert "a = b" not in body
    assert norm(body, False) == "=c"
