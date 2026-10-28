"""
Unit tests for learn_assist.placement.

ALL tests use purely synthetic fixtures. No real book content or exercise text.
Strict adherence to docs/BLINDING.md: no problem statements, answers, or keys.
"""

import pytest
from learn_assist.placement import (
    score_placement,
    PlacementResult,
    OVERALL_THRESHOLD,
    TOPIC_THRESHOLD,
    COUNTED_TOPICS
)
from learn_assist.itembank import validate_item, VALID_BACKGROUND_TAGS


def test_score_placement_pass_all_topics():
    """
    Synthetic test where all counted topics have slots and all are correct.
    """
    slots = []
    for topic in COUNTED_TOPICS:
        for i in range(5):
            slots.append({
                "slot_id": f"s_{topic}_{i}",
                "kind": "expression",
                "correct": True,
                "topic": topic
            })
    res = score_placement(slots)
    assert res.passed is True
    assert res.overall_rate == 1.0
    assert len(res.failing_topics) == 0
    assert all(res.topic_rates[t] == 1.0 for t in COUNTED_TOPICS)


def test_score_placement_boundary_exactly_90_overall():
    """
    Boundary case: exactly 0.90 overall checkable rate (9 of 10 correct, or 18 of 20).
    All topics have at least 0.70 rate. Must PASS.
    """
    # 20 slots total: 18 correct, 2 wrong -> 18/20 = 0.90
    # Distribute across 6 topics:
    # 5 topics get 3 slots (all correct -> 3/3 = 1.0)
    # 1 topic gets 5 slots (3 correct, 2 wrong -> 3/5 = 0.60 -> would fail topic threshold)
    # So instead: 2 topics get 4 slots (3 correct, 1 wrong -> 3/4 = 0.75 >= 0.70)
    # 4 topics get 3 slots (3 correct -> 3/3 = 1.0)
    # Total slots: 2*4 + 4*3 = 20 slots. Correct: 2*3 + 4*3 = 18 correct.
    # Overall: 18/20 = 0.90 exactly. All topics >= 0.75 >= 0.70.
    slots = []
    # 2 topics with 4 slots each (3 correct, 1 wrong)
    for topic in COUNTED_TOPICS[:2]:
        for i in range(3):
            slots.append({"kind": "numeric", "correct": True, "topic": topic})
        slots.append({"kind": "numeric", "correct": False, "topic": topic})

    # 4 topics with 3 slots each (all correct)
    for topic in COUNTED_TOPICS[2:]:
        for i in range(3):
            slots.append({"kind": "numeric", "correct": True, "topic": topic})

    res = score_placement(slots)
    assert res.overall_rate == 0.90
    assert res.passed is True
    assert len(res.failing_topics) == 0


def test_score_placement_boundary_exactly_70_topic():
    """
    Boundary case: a topic has exactly 7 of 10 slots correct (0.70).
    Overall rate is >= 0.90. Must PASS.
    """
    slots = []
    # Topic 1 has 10 slots: 7 correct, 3 wrong (0.70)
    for i in range(7):
        slots.append({"kind": "choice", "correct": True, "topic": COUNTED_TOPICS[0]})
    for i in range(3):
        slots.append({"kind": "choice", "correct": False, "topic": COUNTED_TOPICS[0]})

    # Other 5 topics have 6 slots each (all correct): 30 slots
    for topic in COUNTED_TOPICS[1:]:
        for i in range(6):
            slots.append({"kind": "choice", "correct": True, "topic": topic})

    # Total checkable: 10 + 30 = 40. Correct: 7 + 30 = 37. Rate: 37/40 = 0.925 >= 0.90.
    res = score_placement(slots)
    assert res.topic_rates[COUNTED_TOPICS[0]] == 0.70
    assert res.overall_rate == 0.925
    assert res.passed is True
    assert len(res.failing_topics) == 0


def test_score_placement_four_slot_topic_one_slip_passes():
    """
    One slip in a 4-slot topic gives 3/4 = 0.75 >= 0.70, so it PASSES.
    """
    slots = []
    # 4 slots for topic 2.1 (3 correct, 1 wrong)
    for i in range(3):
        slots.append({"kind": "numeric", "correct": True, "topic": "2.1"})
    slots.append({"kind": "numeric", "correct": False, "topic": "2.1"})

    # Make overall >= 0.90 with remaining topics
    for topic in COUNTED_TOPICS[1:]:
        for i in range(8):
            slots.append({"kind": "numeric", "correct": True, "topic": topic})

    res = score_placement(slots)
    assert res.topic_rates["2.1"] == 0.75
    assert "2.1" not in res.failing_topics
    assert res.passed is True


def test_score_placement_four_slot_topic_two_slips_fail():
    """
    Two slips in a 4-slot topic gives 2/4 = 0.50 < 0.70, so it FAILS.
    """
    slots = []
    # 4 slots for topic 2.1 (2 correct, 2 wrong)
    for i in range(2):
        slots.append({"kind": "numeric", "correct": True, "topic": "2.1"})
    for i in range(2):
        slots.append({"kind": "numeric", "correct": False, "topic": "2.1"})

    # Make overall high
    for topic in COUNTED_TOPICS[1:]:
        for i in range(10):
            slots.append({"kind": "numeric", "correct": True, "topic": topic})

    res = score_placement(slots)
    assert res.topic_rates["2.1"] == 0.50
    assert "2.1" in res.failing_topics
    assert res.passed is False


def test_score_placement_text_slots_ignored():
    """
    Text slots are graded separately and do not count toward checkable slot rates.
    """
    slots = [
        {"kind": "text", "correct": False, "topic": "2.1"},
        {"kind": "text", "correct": False, "topic": "2.2"},
        {"kind": "numeric", "correct": True, "topic": "2.1"},
        {"kind": "expression", "correct": True, "topic": "2.2"},
        {"kind": "choice", "correct": True, "topic": "2.3"},
        {"kind": "numeric", "correct": True, "topic": "2.4"},
        {"kind": "expression", "correct": True, "topic": "2.5"},
        {"kind": "choice", "correct": True, "topic": "complex_exponentials"},
    ]
    res = score_placement(slots)
    assert res.overall_rate == 1.0  # 6 checkable slots, all 6 correct
    assert res.passed is True


def test_score_placement_radon_and_bessel_count_only_toward_overall():
    """
    Radon (2.6) and Bessel functions (2.3.12) count toward overall (1),
    but are excluded from topic threshold (2).
    """
    slots = [
        # 1 slot in 2.3 that is correct
        {"kind": "numeric", "correct": True, "concepts": ["2.3.1"]},
        # 1 slot in Bessel (2.3.12) that is wrong -> should NOT lower 2.3 topic rate
        {"kind": "numeric", "correct": False, "concepts": ["2.3.12"]},
        # 1 slot in Radon (2.6.1) that is correct -> should count toward overall only
        {"kind": "numeric", "correct": True, "concepts": ["2.6.1"]},
    ]
    # Other counted topics have 1 correct slot each
    for topic in ["2.1", "2.2", "2.4", "2.5", "complex_exponentials"]:
        slots.append({"kind": "numeric", "correct": True, "topic": topic})

    res = score_placement(slots)
    # Topic 2.3 rate should be 1.0 (1/1 from 2.3.1; 2.3.12 excluded)
    assert res.topic_rates["2.3"] == 1.0
    # Overall total is 8 slots: 7 correct, 1 wrong (rate: 7/8 = 0.875)
    assert res.overall_rate == 0.875


def test_background_tag_validation():
    """
    Item validation allows valid background tags and rejects invalid ones.
    """
    item_valid = {
        "item_id": "A_2.01",
        "bucket": "A",
        "statement_md": "Synthetic statement.",
        "background": ["complex_numbers", "complex_exponentials", "linear_algebra"]
    }
    ok, errs = validate_item(item_valid)
    assert ok is True
    assert len(errs) == 0

    item_invalid = {
        "item_id": "A_2.02",
        "bucket": "A",
        "statement_md": "Synthetic statement.",
        "background": ["not_a_valid_background_tag"]
    }
    ok, errs = validate_item(item_invalid)
    assert ok is False
    assert any("Unknown background tag" in e for e in errs)
