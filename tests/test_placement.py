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
    assert res.topic_rates["vectors"] == 0.75
    assert "vectors" not in res.failing_topics
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
    assert res.topic_rates["vectors"] == 0.50
    assert "vectors" in res.failing_topics
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


def test_extract_slot_topics_concept_membership():
    """
    Test concept ID and background tag mapping to counted topics:
    vectors = 2.1; matrix_algebra = 2.2; standard_functions = 2.3.1–2.3.11;
    convolution = 2.4; fourier = 2.5.1–2.5.3; complex_exponentials = background tag "complex_exponentials".
    Overall-only: 2.3.12 and 2.6.*.
    """
    from learn_assist.placement import extract_slot_topics

    assert extract_slot_topics({"concepts": ["2.1"]}) == {"vectors"}
    assert extract_slot_topics({"concepts": ["2.2"]}) == {"matrix_algebra"}
    assert extract_slot_topics({"concepts": ["2.3.1"]}) == {"standard_functions"}
    assert extract_slot_topics({"concepts": ["2.3.11"]}) == {"standard_functions"}
    assert extract_slot_topics({"concepts": ["2.3.12"]}) == set()  # overall only
    assert extract_slot_topics({"concepts": ["2.4"]}) == {"convolution"}
    assert extract_slot_topics({"concepts": ["2.5.1"]}) == {"fourier"}
    assert extract_slot_topics({"concepts": ["2.5.3"]}) == {"fourier"}
    assert extract_slot_topics({"concepts": ["2.6.1", "2.6.6"]}) == set()  # overall only
    assert extract_slot_topics({"background": ["complex_exponentials"]}) == {"complex_exponentials"}

    # Combination
    combo = extract_slot_topics({
        "concepts": ["2.1", "2.3.5", "2.6.2"],
        "background": ["complex_exponentials"]
    })
    assert combo == {"vectors", "standard_functions", "complex_exponentials"}


def test_placement_check_chapter(tmp_path):
    """
    Test placement_check_chapter counts checkable slots per topic and enforces threshold >= 4.
    """
    import json
    from learn_assist.itembank import placement_check_chapter

    ch_dir = tmp_path / "itembank" / "ch02"
    items_dir = ch_dir / "items"
    slots_dir = ch_dir / "slots"
    items_dir.mkdir(parents=True, exist_ok=True)
    slots_dir.mkdir(parents=True, exist_ok=True)

    # 6 counted topics: vectors (2.1), matrix_algebra (2.2), standard_functions (2.3.1),
    # convolution (2.4), fourier (2.5.1), complex_exponentials (bg)
    topics_info = [
        ("A_2.01", ["2.1"], []),
        ("A_2.02", ["2.2"], []),
        ("A_2.03", ["2.3.3"], []),
        ("A_2.04", ["2.4"], []),
        ("A_2.05", ["2.5.2"], []),
        ("A_2.06", [], ["complex_exponentials"]),
    ]

    item_ids = [t[0] for t in topics_info]
    for iid, c_list, bg_list in topics_info:
        it_data = {
            "item_id": iid,
            "bucket": "A",
            "statement_md": "Synthetic statement.",
            "concepts": c_list,
            "background": bg_list
        }
        (items_dir / f"{iid}.json").write_text(json.dumps(it_data), encoding="utf-8")

        # 4 checkable slots per item
        sl_data = {
            "item_id": iid,
            "slots": [
                {"slot_id": f"{iid}_s{s}", "kind": "numeric"}
                for s in range(1, 5)
            ]
        }
        (slots_dir / f"{iid}.json").write_text(json.dumps(sl_data), encoding="utf-8")

    # Create FROZEN.json
    frozen_data = {
        "chapter": 2,
        "usable_a_item_ids": item_ids,
        "selected_b_item_ids": [],
        "files": {}
    }
    (ch_dir / "FROZEN.json").write_text(json.dumps(frozen_data), encoding="utf-8")

    # All topics have 4 slots -> PASS
    passed, counts, failing = placement_check_chapter(2, data_dir=tmp_path)
    assert passed is True
    assert len(failing) == 0
    assert all(counts[t] == 4 for t in ["vectors", "matrix_algebra", "standard_functions", "convolution", "fourier", "complex_exponentials"])

    # Make one topic have fewer than 4 slots -> FAIL
    sl_fail = {
        "item_id": "A_2.01",
        "slots": [
            {"slot_id": "A_2.01_s1", "kind": "numeric"}
        ]
    }
    (slots_dir / "A_2.01.json").write_text(json.dumps(sl_fail), encoding="utf-8")
    passed, counts, failing = placement_check_chapter(2, data_dir=tmp_path)
    assert passed is False
    assert "vectors" in failing
    assert counts["vectors"] == 1


def test_merge_tags_both_formats(tmp_path):
    """
    Test merge-tags accepting both plain list and dict with concepts & background.
    """
    import json
    from learn_assist.itembank import merge_tags_chapter

    ch_dir = tmp_path / "itembank" / "ch02"
    items_dir = ch_dir / "items"
    tags_dir = ch_dir / "tags"
    items_dir.mkdir(parents=True, exist_ok=True)
    tags_dir.mkdir(parents=True, exist_ok=True)

    items = ["A_2.01", "A_2.02", "A_2.03"]
    for iid in items:
        it_data = {
            "item_id": iid,
            "bucket": "A",
            "statement_md": "Synthetic statement.",
            "concepts": []
        }
        (items_dir / f"{iid}.json").write_text(json.dumps(it_data), encoding="utf-8")

    # claude tags: mixture of list and dict
    claude_tags = {
        "A_2.01": ["2.1", "2.2"],  # plain list
        "A_2.02": {"concepts": ["2.3.1", "2.3.2"], "background": ["complex_exponentials"]},
        "A_2.03": {"concepts": ["2.4"], "background": ["linear_algebra"]}
    }
    (tags_dir / "claude.json").write_text(json.dumps(claude_tags), encoding="utf-8")

    # gemini tags: mixture of list and dict
    gemini_tags = {
        "A_2.01": {"concepts": ["2.1", "2.4"], "background": ["trigonometry"]},
        "A_2.02": {"concepts": ["2.3.1", "2.3.5"], "background": ["complex_exponentials"]},
        "A_2.03": ["2.5.1"]  # plain list
    }
    (tags_dir / "gemini.json").write_text(json.dumps(gemini_tags), encoding="utf-8")

    res = merge_tags_chapter(2, data_dir=tmp_path)
    assert res["usable_a_count"] == 3

    # Check A_2.01: concepts inter=["2.1"], bg union=["trigonometry"]
    it1 = json.loads((items_dir / "A_2.01.json").read_text(encoding="utf-8"))
    assert it1["concepts"] == ["2.1"]
    assert it1["background"] == ["trigonometry"]

    # Check A_2.02: concepts inter=["2.3.1"], bg inter=["complex_exponentials"]
    it2 = json.loads((items_dir / "A_2.02.json").read_text(encoding="utf-8"))
    assert it2["concepts"] == ["2.3.1"]
    assert it2["background"] == ["complex_exponentials"]

    # Check A_2.03: concepts union=["2.4", "2.5.1"], bg union=["linear_algebra"]
    it3 = json.loads((items_dir / "A_2.03.json").read_text(encoding="utf-8"))
    assert it3["concepts"] == ["2.4", "2.5.1"]
    assert it3["background"] == ["linear_algebra"]

