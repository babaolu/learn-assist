"""
Test runner for learn-assist.

Administers tests sitting-by-sitting from forms.json.
Enforces bank frozen integrity before running.
Shows items one at a time: plain text statement and figure descriptions only.
Captures multi-line answers, records timing and skip flags.
Writes append-only response files to data/responses/chNN/<form>/<item_id>.json.
Provides count-only status reports.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from learn_assist.itembank import file_sha256, verify_frozen_chapter

RUNNER_VERSION = "1.0.0"

# Strict isolation: runner must never read keys, slotted, reviews, compare, resolution
FORBIDDEN_PATTERNS = ["keys", "slotted", "reviews", "compare", "resolution"]


def assert_safe_path(path: Any) -> None:
    p_str = str(path).replace("\\", "/").lower()
    for pattern in FORBIDDEN_PATTERNS:
        parts = [part.lower() for part in Path(path).parts]
        name = Path(path).name.lower()
        if pattern in parts or pattern in name or f"/{pattern}/" in p_str:
            raise PermissionError(f"Access to sensitive data is forbidden: {path}")


def format_statement(
    statement_md: str,
    needs_figure: bool = False,
    figure_descriptions: Optional[List[str]] = None
) -> str:
    """
    Converts statement_md to plain text (pylatexenc if installed, else raw).
    Appends figure_descriptions if needs_figure is True.
    Never shows slots, choices, or answers.
    """
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        text = LatexNodes2Text().latex_to_text(statement_md)
    except ImportError:
        text = statement_md

    if needs_figure and figure_descriptions:
        text += "\n\n[Figure Descriptions]\n"
        for fd in figure_descriptions:
            text += f"- {fd}\n"
    return text


def read_multiline_answer() -> str:
    """
    Reads multi-line answer from stdin until a line containing only '.'.
    """
    print("Enter your answer below. End with a line containing only '.':")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_sitting(
    chapter: Any,
    form: str,
    sitting_num: int,
    data_dir: Path = Path("data"),
    interactive: bool = True,
    input_func=None
) -> int:
    """
    Administers a test sitting for a given chapter, form, and sitting index.
    Refuses unless frozen and verify-frozen passes.
    Resumable: continues at the first unanswered item.
    """
    ch_num = int(re.sub(r"^[a-zA-Z_]+", "", str(chapter)))
    data_dir = Path(data_dir)
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    frozen_file = ch_dir / "FROZEN.json"

    assert_safe_path(frozen_file)

    if not frozen_file.exists():
        print(f"Error: FROZEN.json does not exist for chapter {ch_num:02d}", file=sys.stderr)
        return 1

    ok, rep = verify_frozen_chapter(ch_num, data_dir=data_dir, exclude_sensitive=True)
    if not ok:
        print(f"Error: verify-frozen failed for chapter {ch_num:02d}: {rep}", file=sys.stderr)
        return 1

    frozen_hash = file_sha256(frozen_file)

    forms_file = ch_dir / "forms.json"
    assert_safe_path(forms_file)
    if not forms_file.exists():
        print(f"Error: forms.json does not exist for chapter {ch_num:02d}", file=sys.stderr)
        return 1

    forms_data = json.loads(forms_file.read_text(encoding="utf-8"))
    if form not in forms_data.get("forms", {}):
        print(f"Error: Form '{form}' not found in forms.json", file=sys.stderr)
        return 1

    form_obj = forms_data["forms"][form]
    sittings = form_obj.get("sittings", [])

    sitting_obj = None
    for s in sittings:
        if s.get("sitting") == sitting_num:
            sitting_obj = s
            break

    if sitting_obj is None:
        print(f"Error: Sitting {sitting_num} not found in form '{form}'", file=sys.stderr)
        return 1

    sitting_item_ids: List[str] = sitting_obj.get("item_ids", [])
    total_items = len(sitting_item_ids)

    resp_dir = data_dir / "responses" / f"ch{ch_num:02d}" / form
    assert_safe_path(resp_dir)
    resp_dir.mkdir(parents=True, exist_ok=True)

    # Resume check: find unanswered items
    unanswered_indices = []
    for idx, item_id in enumerate(sitting_item_ids):
        resp_path = resp_dir / f"{item_id}.json"
        assert_safe_path(resp_path)
        if not resp_path.exists():
            unanswered_indices.append(idx)

    if not unanswered_indices:
        print(f"Sitting {sitting_num} of form '{form}' is already complete ({total_items}/{total_items} answered).")
        return 0

    first_unanswered_idx = unanswered_indices[0]
    if first_unanswered_idx > 0:
        print(f"Resuming Sitting {sitting_num} at item {first_unanswered_idx + 1} of {total_items}...")

    # Baseline opening screen
    if form == "baseline" and first_unanswered_idx == 0:
        print("=" * 80)
        print("This is a baseline. You have not studied this chapter. Answer from what you already know, show any partial reasoning, and write 'dont know' rather than guessing wildly. Nothing you write here is graded for a score you will see now.")
        print("=" * 80)
        if interactive:
            try:
                input("Press Enter to begin...")
            except (EOFError, KeyboardInterrupt):
                pass

    items_dir = ch_dir / "items"
    assert_safe_path(items_dir)

    for item_idx in unanswered_indices:
        item_id = sitting_item_ids[item_idx]
        resp_path = resp_dir / f"{item_id}.json"
        assert_safe_path(resp_path)
        if resp_path.exists():
            continue

        item_file = items_dir / f"{item_id}.json"
        assert_safe_path(item_file)
        if not item_file.exists():
            print(f"Error: Item file {item_file} not found", file=sys.stderr)
            return 1

        with open(item_file, "r", encoding="utf-8") as f:
            item_data = json.load(f)

        statement_md = item_data.get("statement_md", "")
        needs_figure = item_data.get("needs_figure", False)
        figure_descriptions = item_data.get("figure_descriptions", [])

        formatted = format_statement(statement_md, needs_figure, figure_descriptions)

        # Administer item
        while True:
            print("\n" + "-" * 80)
            print(f"Item {item_idx + 1} of {total_items} (Sitting {sitting_num})")
            print("-" * 80)
            print(formatted)
            print("-" * 80)

            shown_at = datetime.now(timezone.utc).isoformat()
            t0 = time.time()

            if input_func is not None:
                raw_answer = input_func()
            else:
                raw_answer = read_multiline_answer()

            submitted_at = datetime.now(timezone.utc).isoformat()
            seconds = round(time.time() - t0, 2)

            ans_stripped = raw_answer.strip().lower()
            skipped = ans_stripped in ("skip", "dont know", "don't know", "dont_know")

            # Confirmation
            if interactive and input_func is None:
                try:
                    conf = input("Confirm before saving (y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    conf = "y"
                if conf not in ("y", "yes"):
                    print("Re-entering answer for this item...")
                    continue
            break

        # Save response append-only
        if resp_path.exists():
            raise FileExistsError(f"Response file already exists: {resp_path}. Overwrite refused.")

        resp_record = {
            "item_id": item_id,
            "form": form,
            "sitting": sitting_num,
            "raw_answer": raw_answer,
            "skipped": skipped,
            "shown_at": shown_at,
            "submitted_at": submitted_at,
            "seconds": seconds,
            "runner_version": RUNNER_VERSION,
            "frozen_hash": frozen_hash
        }

        # Atomic exclusive write
        with open(resp_path, "x", encoding="utf-8") as f:
            f.write(json.dumps(resp_record, indent=2) + "\n")

    print(f"\nSitting {sitting_num} of form '{form}' completed successfully.")
    return 0


def get_testrun_status(
    chapter: Any,
    data_dir: Path = Path("data")
) -> Dict[str, Any]:
    """
    Computes status counts for a chapter testrun (answered, skipped, remaining).
    Strictly numbers and IDs/names only. Never prints item or answer text.
    """
    ch_num = int(re.sub(r"^[a-zA-Z_]+", "", str(chapter)))
    data_dir = Path(data_dir)
    ch_dir = data_dir / "itembank" / f"ch{ch_num:02d}"
    forms_file = ch_dir / "forms.json"

    assert_safe_path(forms_file)

    if not forms_file.exists():
        return {"error": f"forms.json does not exist for chapter {ch_num:02d}"}

    forms_data = json.loads(forms_file.read_text(encoding="utf-8"))
    res_status: Dict[str, Any] = {
        "chapter": ch_num,
        "forms": {}
    }

    for form_name in ["baseline", "final"]:
        if form_name not in forms_data.get("forms", {}):
            continue
        form_obj = forms_data["forms"][form_name]
        sittings = form_obj.get("sittings", [])
        resp_dir = data_dir / "responses" / f"ch{ch_num:02d}" / form_name

        sittings_status = []
        tot_answered = 0
        tot_skipped = 0
        tot_remaining = 0

        for s in sittings:
            s_num = s.get("sitting", 0)
            item_ids = s.get("item_ids", [])
            s_ans = 0
            s_skp = 0
            s_rem = 0

            for iid in item_ids:
                resp_path = resp_dir / f"{iid}.json"
                assert_safe_path(resp_path)
                if resp_path.exists():
                    try:
                        resp_data = json.loads(resp_path.read_text(encoding="utf-8"))
                        if resp_data.get("skipped", False):
                            s_skp += 1
                        else:
                            s_ans += 1
                    except Exception:
                        s_ans += 1
                else:
                    s_rem += 1

            tot_answered += s_ans
            tot_skipped += s_skp
            tot_remaining += s_rem

            sittings_status.append({
                "sitting": s_num,
                "total": len(item_ids),
                "answered": s_ans,
                "skipped": s_skp,
                "remaining": s_rem
            })

        res_status["forms"][form_name] = {
            "total": tot_answered + tot_skipped + tot_remaining,
            "answered": tot_answered,
            "skipped": tot_skipped,
            "remaining": tot_remaining,
            "sittings": sittings_status
        }

    return res_status


def print_status(status: Dict[str, Any]) -> None:
    """
    Prints status report containing counts only.
    """
    if "error" in status:
        print(f"Error: {status["error"]}", file=sys.stderr)
        return

    ch_num = status["chapter"]
    print(f"Chapter {ch_num:02d} Testrun Status:")
    for form_name, fdata in status.get("forms", {}).items():
        print(f"  Form '{form_name}': total={fdata["total"]} (answered: {fdata["answered"]}, skipped: {fdata["skipped"]}, remaining: {fdata["remaining"]})")
        for s in fdata.get("sittings", []):
            print(f"    Sitting {s["sitting"]}: total={s["total"]} (answered: {s["answered"]}, skipped: {s["skipped"]}, remaining: {s["remaining"]})")


def main():
    parser = argparse.ArgumentParser(description="Test runner for learn-assist.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # start
    p_start = subparsers.add_parser("start", help="Start or resume a test sitting")
    p_start.add_argument("--chapter", required=True, help="Chapter number (e.g. 3, 03, ch03)")
    p_start.add_argument("--form", required=True, choices=["baseline", "final"], help="Form name")
    p_start.add_argument("--sitting", type=int, required=True, help="Sitting number (1-based)")
    p_start.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    # status
    p_status = subparsers.add_parser("status", help="Show testrun completion status")
    p_status.add_argument("--chapter", required=True, help="Chapter number (e.g. 3, 03, ch03)")
    p_status.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")

    args = parser.parse_args()

    if args.subcommand == "start":
        rc = run_sitting(args.chapter, args.form, args.sitting, data_dir=args.data_dir, interactive=True)
        sys.exit(rc)
    elif args.subcommand == "status":
        st = get_testrun_status(args.chapter, data_dir=args.data_dir)
        print_status(st)


if __name__ == "__main__":
    main()
