"""
Deterministic form generation for learn-assist.

Splits frozen items (usable A plus selected B) into 'baseline' and 'final' forms.
Keeps anchor pairs together, balances B count, per-concept coverage, and A count.
Chunks each form into sittings of at most 10 items in seeded random order.
Writes data/itembank/chNN/forms.json (IDs and counts only).
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from learn_assist.itembank import file_sha256, verify_frozen_chapter

# Strict enforcement: runner and forms tools must never read sensitive directories/files
FORBIDDEN_PATTERNS = ["keys", "slotted", "reviews", "compare", "resolution"]


def assert_safe_path(path: Any) -> None:
    p_str = str(path).replace("\\", "/").lower()
    for pattern in FORBIDDEN_PATTERNS:
        parts = [part.lower() for part in Path(path).parts]
        name = Path(path).name.lower()
        if pattern in parts or pattern in name or f"/{pattern}/" in p_str:
            raise PermissionError(f"Access to sensitive data is forbidden: {path}")


def make_forms(
    chapter: Any,
    seed: int,
    data_dir: Path = Path("data"),
    single: Optional[str] = None
) -> Dict[str, Any]:
    """
    Splits frozen chapter items into 'baseline' and 'final' forms, or a single form if single is specified.
    Exits non-zero if:
    - FROZEN.json is missing or verify-frozen fails.
    - When not single: the two forms differ in B count by more than 1.
    - When not single: any concept with selected B items ends up with 0 in one form while the other has >= 2.
    """
    ch_num = int(re.sub(r"^[a-zA-Z_]+", "", str(chapter)))
    data_dir = Path(data_dir)
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    frozen_file = ch_dir / "FROZEN.json"

    assert_safe_path(frozen_file)

    if not frozen_file.exists():
        print(f"Error: FROZEN.json does not exist for chapter {ch_num:02d}", file=sys.stderr)
        sys.exit(1)

    # Verify frozen integrity without reading sensitive files
    ok, rep = verify_frozen_chapter(ch_num, data_dir=data_dir, exclude_sensitive=True)
    if not ok:
        print(f"Error: verify-frozen failed for chapter {ch_num:02d}: {rep}", file=sys.stderr)
        sys.exit(1)

    frozen_hash = file_sha256(frozen_file)
    frozen_data = json.loads(frozen_file.read_text(encoding="utf-8"))

    usable_a_ids: List[str] = sorted(frozen_data.get("usable_a_item_ids", []))
    selected_b_ids: List[str] = sorted(frozen_data.get("selected_b_item_ids", []))

    if not usable_a_ids and not selected_b_ids:
        print(f"Error: No usable A or selected B items found in FROZEN.json for chapter {ch_num:02d}", file=sys.stderr)
        sys.exit(1)

    items_dir = ch_dir / "items"
    assert_safe_path(items_dir)

    # Read items (only metadata, concepts, anchor_item, bucket; statements not needed for forms.json)
    items_meta: Dict[str, Dict[str, Any]] = {}
    for aid in usable_a_ids:
        item_path = items_dir / f"{aid}.json"
        assert_safe_path(item_path)
        with open(item_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        items_meta[aid] = {
            "item_id": aid,
            "bucket": "A",
            "concepts": sorted(list(d.get("concepts", [])))
        }

    for bid in selected_b_ids:
        item_path = items_dir / f"{bid}.json"
        assert_safe_path(item_path)
        with open(item_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        items_meta[bid] = {
            "item_id": bid,
            "bucket": "B",
            "anchor_item": d.get("anchor_item"),
            "concepts": sorted(list(d.get("concepts", [])))
        }

    chunk_size = 10

    if single:
        all_items = sorted(usable_a_ids + selected_b_ids)
        form_rng = random.Random(f"{seed}_{single}")
        form_rng.shuffle(all_items)

        sittings: List[Dict[str, Any]] = []
        for i in range(0, len(all_items), chunk_size):
            chunk = all_items[i : i + chunk_size]
            sitting_idx = len(sittings) + 1

            sitting_a_count = sum(1 for item_id in chunk if items_meta[item_id]["bucket"] == "A")
            sitting_b_count = sum(1 for item_id in chunk if items_meta[item_id]["bucket"] == "B")

            sitting_concept_counts = Counter()
            sitting_concept_a = Counter()
            sitting_concept_b = Counter()
            for item_id in chunk:
                bucket = items_meta[item_id]["bucket"]
                for c in items_meta[item_id]["concepts"]:
                    sitting_concept_counts[c] += 1
                    if bucket == "A":
                        sitting_concept_a[c] += 1
                    else:
                        sitting_concept_b[c] += 1

            sittings.append({
                "sitting": sitting_idx,
                "item_ids": chunk,
                "counts": {
                    "total": len(chunk),
                    "by_bucket": {
                        "A": sitting_a_count,
                        "B": sitting_b_count
                    },
                    "by_concept": dict(sorted(sitting_concept_counts.items())),
                    "by_concept_and_bucket": {
                        "A": dict(sorted(sitting_concept_a.items())),
                        "B": dict(sorted(sitting_concept_b.items()))
                    }
                }
            })

        form_total = len(all_items)
        form_a_count = sum(1 for item_id in all_items if items_meta[item_id]["bucket"] == "A")
        form_b_count = sum(1 for item_id in all_items if items_meta[item_id]["bucket"] == "B")

        form_concept_counts = Counter()
        form_concept_a = Counter()
        form_concept_b = Counter()
        for item_id in all_items:
            bucket = items_meta[item_id]["bucket"]
            for c in items_meta[item_id]["concepts"]:
                form_concept_counts[c] += 1
                if bucket == "A":
                    form_concept_a[c] += 1
                else:
                    form_concept_b[c] += 1

        forms_output: Dict[str, Any] = {
            single: {
                "item_ids": all_items,
                "counts": {
                    "total": form_total,
                    "by_bucket": {
                        "A": form_a_count,
                        "B": form_b_count
                    },
                    "by_concept": dict(sorted(form_concept_counts.items())),
                    "by_concept_and_bucket": {
                        "A": dict(sorted(form_concept_a.items())),
                        "B": dict(sorted(form_concept_b.items()))
                    }
                },
                "sittings": sittings
            }
        }

        out_data = {
            "chapter": ch_num,
            "seed": seed,
            "single": single,
            "frozen_hash": frozen_hash,
            "forms": forms_output
        }

        out_file = ch_dir / "forms.json"
        assert_safe_path(out_file)
        out_file.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        return out_data

    # Group into anchor units: each unit has 1 A item and 0 or more attached B items
    anchor_to_b: Dict[str, List[str]] = {}
    for bid in selected_b_ids:
        anc = items_meta[bid].get("anchor_item")
        if not anc or anc not in items_meta:
            print(f"Error: B item {bid} has invalid or non-usable anchor item: {anc}", file=sys.stderr)
            sys.exit(1)
        anchor_to_b.setdefault(anc, []).append(bid)

    units: List[Dict[str, Any]] = []
    for aid in usable_a_ids:
        b_list = sorted(anchor_to_b.get(aid, []))
        b_concepts: List[str] = []
        for bid in b_list:
            b_concepts.extend(items_meta[bid]["concepts"])
        units.append({
            "anchor_a": aid,
            "b_items": b_list,
            "a_concepts": items_meta[aid]["concepts"],
            "b_concepts": b_concepts
        })

    # Separate units with B items and A-only units
    b_units = [u for u in units if u["b_items"]]
    a_units = [u for u in units if not u["b_items"]]

    # Deterministic sort before shuffle
    b_units.sort(key=lambda u: u["anchor_a"])
    a_units.sort(key=lambda u: u["anchor_a"])

    rng = random.Random(seed)
    rng.shuffle(b_units)
    rng.shuffle(a_units)

    total_b = len(selected_b_ids)
    max_b_per_form = (total_b + 1) // 2

    # Remaining B concept counts for lookahead starvation prevention
    remaining_b_concepts = Counter()
    for u in b_units:
        for c in u["b_concepts"]:
            remaining_b_concepts[c] += 1

    forms: Dict[str, Dict[str, Any]] = {
        "baseline": {
            "a": [],
            "b": [],
            "a_concepts": Counter(),
            "b_concepts": Counter()
        },
        "final": {
            "a": [],
            "b": [],
            "a_concepts": Counter(),
            "b_concepts": Counter()
        }
    }

    # Phase 1: Assign B units
    # Priority: Balance B count first, then per-concept coverage (counting A and B separately), then A count
    for u in b_units:
        for c in u["b_concepts"]:
            remaining_b_concepts[c] -= 1

        scores: Dict[str, Tuple] = {}
        for form_name, other_name in [("baseline", "final"), ("final", "baseline")]:
            cand = forms[form_name]
            other = forms[other_name]

            new_b_count = len(cand["b"]) + len(u["b_items"])
            exceeds_cap = 1 if new_b_count > max_b_per_form else 0

            # Starvation: cand has >= 1, other has 0, and no more remaining units have this concept
            starvation = 0
            for c in u["b_concepts"]:
                if cand["b_concepts"][c] >= 1 and other["b_concepts"][c] == 0 and remaining_b_concepts[c] == 0:
                    starvation += 1

            b_diff = abs(new_b_count - len(other["b"]))
            all_b_concepts = set(cand["b_concepts"]) | set(other["b_concepts"]) | set(u["b_concepts"])
            b_concept_diff = sum(
                abs(cand["b_concepts"][c] + u["b_concepts"].count(c) - other["b_concepts"][c])
                for c in all_b_concepts
            )

            all_a_concepts = set(cand["a_concepts"]) | set(other["a_concepts"]) | set(u["a_concepts"])
            a_concept_diff = sum(
                abs(cand["a_concepts"][c] + u["a_concepts"].count(c) - other["a_concepts"][c])
                for c in all_a_concepts
            )

            new_a_count = len(cand["a"]) + 1
            a_diff = abs(new_a_count - len(other["a"]))

            scores[form_name] = (exceeds_cap, starvation, b_diff, b_concept_diff, a_concept_diff, a_diff)

        if scores["baseline"] < scores["final"]:
            chosen = "baseline"
        elif scores["final"] < scores["baseline"]:
            chosen = "final"
        else:
            chosen = "baseline" if rng.random() < 0.5 else "final"

        forms[chosen]["a"].append(u["anchor_a"])
        forms[chosen]["b"].extend(u["b_items"])
        for c in u["b_concepts"]:
            forms[chosen]["b_concepts"][c] += 1
        for c in u["a_concepts"]:
            forms[chosen]["a_concepts"][c] += 1

    # Phase 2: Assign A-only units
    # Balance per-concept A coverage, then total A count
    total_a = len(usable_a_ids)
    max_a_per_form = (total_a + 1) // 2

    for u in a_units:
        scores = {}
        for form_name, other_name in [("baseline", "final"), ("final", "baseline")]:
            cand = forms[form_name]
            other = forms[other_name]

            new_a_count = len(cand["a"]) + 1
            exceeds_cap = 1 if new_a_count > max_a_per_form else 0

            all_a_concepts = set(cand["a_concepts"]) | set(other["a_concepts"]) | set(u["a_concepts"])
            a_concept_diff = sum(
                abs(cand["a_concepts"][c] + u["a_concepts"].count(c) - other["a_concepts"][c])
                for c in all_a_concepts
            )

            a_diff = abs(new_a_count - len(other["a"]))
            scores[form_name] = (exceeds_cap, a_concept_diff, a_diff)

        if scores["baseline"] < scores["final"]:
            chosen = "baseline"
        elif scores["final"] < scores["baseline"]:
            chosen = "final"
        else:
            chosen = "baseline" if rng.random() < 0.5 else "final"

        forms[chosen]["a"].append(u["anchor_a"])
        for c in u["a_concepts"]:
            forms[chosen]["a_concepts"][c] += 1

    # Validation check 1: B count difference must not exceed 1
    b_diff = abs(len(forms["baseline"]["b"]) - len(forms["final"]["b"]))
    if b_diff > 1:
        print(f"Error: Forms differ in B count by {b_diff} > 1", file=sys.stderr)
        sys.exit(1)

    # Validation check 2: No concept with selected B items ends up with 0 in one form while other has >= 2
    b_concepts_all = set(forms["baseline"]["b_concepts"].keys()) | set(forms["final"]["b_concepts"].keys())
    concept_violations: List[Tuple[str, int, int]] = []
    for c in b_concepts_all:
        cb = forms["baseline"]["b_concepts"][c]
        cf = forms["final"]["b_concepts"][c]
        if (cb == 0 and cf >= 2) or (cf == 0 and cb >= 2):
            concept_violations.append((c, cb, cf))

    if concept_violations:
        print(f"Error: Concept coverage violations in B items: {concept_violations}", file=sys.stderr)
        sys.exit(1)

    # Build sittings for each form:
    # Within each form, split into sittings of at most 10 items, mixing A and B, in a seeded random order.
    forms_output: Dict[str, Any] = {}
    chunk_size = 10

    for form_name in ["baseline", "final"]:
        form_data = forms[form_name]
        combined_items = sorted(form_data["a"] + form_data["b"])
        # Seeded random shuffle of mixed items for this form
        form_rng = random.Random(f"{seed}_{form_name}")
        form_rng.shuffle(combined_items)

        # Chunk into sittings of <= 10 items
        sittings: List[Dict[str, Any]] = []
        for i in range(0, len(combined_items), chunk_size):
            chunk = combined_items[i : i + chunk_size]
            sitting_idx = len(sittings) + 1

            sitting_a_count = sum(1 for item_id in chunk if items_meta[item_id]["bucket"] == "A")
            sitting_b_count = sum(1 for item_id in chunk if items_meta[item_id]["bucket"] == "B")

            sitting_concept_counts = Counter()
            sitting_concept_a = Counter()
            sitting_concept_b = Counter()
            for item_id in chunk:
                bucket = items_meta[item_id]["bucket"]
                for c in items_meta[item_id]["concepts"]:
                    sitting_concept_counts[c] += 1
                    if bucket == "A":
                        sitting_concept_a[c] += 1
                    else:
                        sitting_concept_b[c] += 1

            sittings.append({
                "sitting": sitting_idx,
                "item_ids": chunk,
                "counts": {
                    "total": len(chunk),
                    "by_bucket": {
                        "A": sitting_a_count,
                        "B": sitting_b_count
                    },
                    "by_concept": dict(sorted(sitting_concept_counts.items())),
                    "by_concept_and_bucket": {
                        "A": dict(sorted(sitting_concept_a.items())),
                        "B": dict(sorted(sitting_concept_b.items()))
                    }
                }
            })

        # Form-level counts
        form_total = len(combined_items)
        form_a_count = len(form_data["a"])
        form_b_count = len(form_data["b"])

        form_concept_counts = Counter()
        for item_id in combined_items:
            for c in items_meta[item_id]["concepts"]:
                form_concept_counts[c] += 1

        forms_output[form_name] = {
            "item_ids": combined_items,
            "counts": {
                "total": form_total,
                "by_bucket": {
                    "A": form_a_count,
                    "B": form_b_count
                },
                "by_concept": dict(sorted(form_concept_counts.items())),
                "by_concept_and_bucket": {
                    "A": dict(sorted(form_data["a_concepts"].items())),
                    "B": dict(sorted(form_data["b_concepts"].items()))
                }
            },
            "sittings": sittings
        }

    out_data = {
        "chapter": ch_num,
        "seed": seed,
        "frozen_hash": frozen_hash,
        "forms": forms_output
    }

    out_file = ch_dir / "forms.json"
    assert_safe_path(out_file)
    out_file.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
    return out_data


def main():
    parser = argparse.ArgumentParser(description="Deterministic form generator for learn-assist.")
    parser.add_argument("chapter", help="Chapter number (e.g. 3, 03, ch03)")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic random seed")
    parser.add_argument("--single", type=str, default=None, help="Produce a single form containing all frozen items")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    args = parser.parse_args()
    res = make_forms(args.chapter, args.seed, data_dir=args.data_dir, single=args.single)

    # BLINDING compliant output: IDs and counts only
    print(f"FORMS GENERATED: chapter={res['chapter']} seed={res['seed']}")
    for fname, fdata in res["forms"].items():
        cnt = fdata["counts"]
        print(f"Form '{fname}': total={cnt['total']} A={cnt['by_bucket']['A']} B={cnt['by_bucket']['B']} sittings={len(fdata['sittings'])}")
        for s in fdata["sittings"]:
            sc = s["counts"]
            print(f"  Sitting {s['sitting']}: total={sc['total']} A={sc['by_bucket']['A']} B={sc['by_bucket']['B']}")


if __name__ == "__main__":
    main()
