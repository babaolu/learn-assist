"""
Placement rule for learn-assist Chapter 2 diagnostic.

Specifies thresholds and evaluation for prerequisite mastery of Chapter 2.
The learner is solid on Chapter 2 only if BOTH hold on checkable answer slots:
(1) at least 90% of all checkable slots correct;
(2) at least 70% correct within EACH counted topic:
    - vectors (2.1)
    - matrix algebra (2.2)
    - standard functions (rect, sinc, delta; 2.3)
    - convolution (2.4)
    - Fourier transform (2.5)
    - complex exponentials (background tag)

Radon (2.6) and Bessel functions count only toward (1).
Text slots are graded separately and reported but do not decide placement.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

# Data-independent configuration
OVERALL_THRESHOLD: float = 0.90
TOPIC_THRESHOLD: float = 0.70

COUNTED_TOPICS: List[str] = [
    "vectors",
    "matrix_algebra",
    "standard_functions",
    "convolution",
    "fourier",
    "complex_exponentials"
]

TOPIC_DESCRIPTIONS: Dict[str, str] = {
    "vectors": "vectors (2.1)",
    "matrix_algebra": "matrix algebra (2.2)",
    "standard_functions": "standard functions (rect, sinc, delta; 2.3)",
    "convolution": "convolution (2.4)",
    "fourier": "Fourier transform (2.5)",
    "complex_exponentials": "complex exponentials (background tag)"
}

CHECKABLE_SLOT_TYPES: Set[str] = {
    "numeric",
    "expression",
    "choice"
}

TOPIC_ALIASES: Dict[str, str] = {
    "vectors": "vectors",
    "vector": "vectors",
    "2.1": "vectors",
    "matrix_algebra": "matrix_algebra",
    "matrices": "matrix_algebra",
    "matrix": "matrix_algebra",
    "2.2": "matrix_algebra",
    "standard_functions": "standard_functions",
    "functions": "standard_functions",
    "2.3": "standard_functions",
    "convolution": "convolution",
    "2.4": "convolution",
    "fourier_transform": "fourier",
    "fourier": "fourier",
    "2.5": "fourier",
    "complex_exponentials": "complex_exponentials",
    "complex_exponential": "complex_exponentials"
}

TOPIC_TO_NUMBER: Dict[str, str] = {
    "vectors": "2.1",
    "matrix_algebra": "2.2",
    "standard_functions": "2.3",
    "convolution": "2.4",
    "fourier": "2.5",
    "complex_exponentials": "complex_exponentials"
}


class PlacementResult(NamedTuple):
    passed: bool
    overall_rate: float
    topic_rates: Dict[str, float]
    failing_topics: List[str]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
            raise KeyError(item)
        return tuple.__getitem__(self, item)


def extract_slot_topics(slot_data: Dict[str, Any]) -> Set[str]:
    """
    Determines which counted topics apply to a slot.
    vectors = 2.1; matrix_algebra = 2.2; standard_functions = 2.3.1–2.3.11;
    convolution = 2.4; fourier = 2.5.1–2.5.3; complex_exponentials = background tag "complex_exponentials".
    Overall-only: 2.3.12 and 2.6.*.
    A slot counts toward a topic if its item carries that concept or tag.
    """
    topics: Set[str] = set()

    # Direct topic or topics field
    direct_topics = []
    if "topic" in slot_data and slot_data["topic"]:
        direct_topics.append(slot_data["topic"])
    if "topics" in slot_data and isinstance(slot_data["topics"], list):
        direct_topics.extend(slot_data["topics"])

    for dt in direct_topics:
        norm = str(dt).strip().lower()
        if norm in TOPIC_ALIASES:
            topics.add(TOPIC_ALIASES[norm])
        elif norm in COUNTED_TOPICS:
            topics.add(norm)

    # Concepts tags
    concepts = slot_data.get("concepts", [])
    if isinstance(concepts, list):
        for c in concepts:
            c_str = str(c).strip()
            # Bessel function (2.3.12) exclusion from counted topics (overall-only)
            if c_str == "2.3.12" or "bessel" in c_str.lower():
                continue
            # Radon (2.6.*) exclusion from counted topics (overall-only)
            if c_str.startswith("2.6"):
                continue

            if c_str == "2.1" or c_str.startswith("2.1."):
                topics.add("vectors")
            elif c_str == "2.2" or c_str.startswith("2.2."):
                topics.add("matrix_algebra")
            elif c_str == "2.3" or c_str.startswith("2.3."):
                topics.add("standard_functions")
            elif c_str == "2.4" or c_str.startswith("2.4."):
                topics.add("convolution")
            elif c_str == "2.5" or c_str.startswith("2.5."):
                topics.add("fourier")

    # Background tags
    bg = slot_data.get("background", [])
    if isinstance(bg, list):
        for b in bg:
            b_str = str(b).strip().lower()
            if b_str == "complex_exponentials":
                topics.add("complex_exponentials")

    return topics


def score_placement(
    responses_graded: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    overall_threshold: float = OVERALL_THRESHOLD,
    topic_threshold: float = TOPIC_THRESHOLD,
    counted_topics: Optional[List[str]] = None
) -> PlacementResult:
    """
    Scores diagnostic responses for Chapter 2 prerequisite placement.

    Only code-checkable answer slots (numeric, expression, choice) decide placement.
    Text slots are graded separately and ignored here.

    Returns:
      PlacementResult(passed, overall_rate, topic_rates, failing_topics)
    """
    if counted_topics is None:
        counted_topics = list(COUNTED_TOPICS)

    # Flatten input into a list of checkable slot dicts
    checkable_slots: List[Dict[str, Any]] = []

    items_or_slots: List[Dict[str, Any]] = []
    if isinstance(responses_graded, dict):
        if "slots" in responses_graded and isinstance(responses_graded["slots"], list):
            items_or_slots = responses_graded["slots"]
        else:
            items_or_slots = list(responses_graded.values())
    else:
        items_or_slots = list(responses_graded)

    for entry in items_or_slots:
        if not isinstance(entry, dict):
            continue

        # Check if entry is an item containing multiple slots
        if "slots" in entry and isinstance(entry["slots"], list):
            item_concepts = entry.get("concepts", [])
            item_bg = entry.get("background", [])
            item_topics = entry.get("topics", [])
            item_topic = entry.get("topic")

            for s in entry["slots"]:
                if not isinstance(s, dict):
                    continue
                s_copy = dict(s)
                if "concepts" not in s_copy and item_concepts:
                    s_copy["concepts"] = item_concepts
                if "background" not in s_copy and item_bg:
                    s_copy["background"] = item_bg
                if "topics" not in s_copy and item_topics:
                    s_copy["topics"] = item_topics
                if "topic" not in s_copy and item_topic:
                    s_copy["topic"] = item_topic

                stype = str(s_copy.get("slot_type", s_copy.get("kind", s_copy.get("type", "unknown")))).lower()
                if stype in CHECKABLE_SLOT_TYPES:
                    checkable_slots.append(s_copy)
        else:
            stype = str(entry.get("slot_type", entry.get("kind", entry.get("type", "numeric")))).lower()
            if stype in CHECKABLE_SLOT_TYPES:
                checkable_slots.append(entry)

    total_checkable = len(checkable_slots)
    if total_checkable == 0:
        return PlacementResult(
            passed=False,
            overall_rate=0.0,
            topic_rates={t: 0.0 for t in counted_topics},
            failing_topics=list(counted_topics)
        )

    # (1) Overall checkable rate
    correct_count = 0
    for s in checkable_slots:
        is_corr = bool(s.get("correct", False))
        if not is_corr:
            status = str(s.get("status", "")).lower()
            if status in ("correct", "pass", "agreed", "equivalent"):
                is_corr = True
        if is_corr:
            correct_count += 1

    overall_rate = round(correct_count / total_checkable, 6)
    overall_passed = (overall_rate >= overall_threshold)

    # (2) Per-topic rate for each counted topic
    topic_rates: Dict[str, float] = {}
    failing_topics: List[str] = []

    for t in counted_topics:
        norm_t = TOPIC_ALIASES.get(t, t)
        # Find slots that exercise topic t
        slots_in_t = []
        for s in checkable_slots:
            s_topics = extract_slot_topics(s)
            if norm_t in s_topics or t in s_topics:
                slots_in_t.append(s)

        if not slots_in_t:
            # If topic was tested with 0 slots, rate is 0.0 and fails topic requirement
            t_rate = 0.0
            t_passed = False
        else:
            t_correct = 0
            for s in slots_in_t:
                is_corr = bool(s.get("correct", False))
                if not is_corr:
                    status = str(s.get("status", "")).lower()
                    if status in ("correct", "pass", "agreed", "equivalent"):
                        is_corr = True
                if is_corr:
                    t_correct += 1
            t_rate = round(t_correct / len(slots_in_t), 6)
            t_passed = (t_rate >= topic_threshold)

        topic_rates[t] = t_rate
        if norm_t != t:
            topic_rates[norm_t] = t_rate
        num_alias = TOPIC_TO_NUMBER.get(norm_t)
        if num_alias:
            topic_rates[num_alias] = t_rate

        if not t_passed:
            failing_topics.append(t)

    passed = overall_passed and (len(failing_topics) == 0)

    return PlacementResult(
        passed=passed,
        overall_rate=overall_rate,
        topic_rates=topic_rates,
        failing_topics=failing_topics
    )
