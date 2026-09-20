"""Command line entry point: rank lines or JSON records from a file or stdin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import JevJudge, OfflineJudge, build_offline_truth
from .compare import slot
from .ranker import rank


def load_items(path: str | None, key: str | None) -> list:
    raw = Path(path).read_text() if path and path != "-" else sys.stdin.read()
    text = raw.strip()
    if not text:
        return []
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            return [(k, v) for k, v in data.items()]
        if key:
            return data
        return [(str(i), entry) for i, entry in enumerate(data)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevsort",
        description="Rank items by a natural-language criterion.",
    )
    parser.add_argument("criterion", help='completes "is this ___ than that"')
    parser.add_argument("-f", "--file", help="input file; omit to read stdin")
    parser.add_argument("-k", "--key", help="field naming each record's identifier")
    parser.add_argument("-b", "--budget", type=int, help="max questions (default 6n)")
    parser.add_argument("-w", "--window", type=int, default=3)
    parser.add_argument("--no-seed", action="store_true", help="skip the scoring pass")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="simulated judge, no API key; ranks by input order, for mechanics only",
    )
    args = parser.parse_args(argv)

    items = load_items(args.file, args.key)
    if not items:
        print("no items on input", file=sys.stderr)
        return 1

    judge = (
        OfflineJudge(build_offline_truth([slot(i) for i in range(len(items))]))
        if args.offline
        else JevJudge()
    )

    result = rank(
        items,
        args.criterion,
        budget=args.budget,
        judge=judge,
        key=args.key,
        seed=not args.no_seed,
        window=args.window,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "order": result.order,
                    "questions_used": result.questions_used,
                    "rounds": result.rounds,
                    "mean_asymmetry": round(result.mean_asymmetry, 4),
                    "cycles": result.cycles,
                    "items": [
                        {
                            "key": i.key,
                            "rank": i.rank,
                            "rank_low": i.rank_low,
                            "rank_high": i.rank_high,
                            "tie_group": i.tie_group,
                            "strength": round(i.strength, 6),
                        }
                        for i in result.items
                    ],
                },
                indent=2,
            )
        )
        return 0

    width = max((len(i.key) for i in result.items), default=4)
    previous = None
    for item in result.items:
        if previous is not None and item.tie_group != previous:
            print("  " + "-" * (width + 22))
        marker = "~" if len(result.tie_groups[item.tie_group]) > 1 else " "
        print(
            f"{item.rank:>3}. {item.key:<{width}} {marker} "
            f"[{item.rank_low}-{item.rank_high}]  {item.strength:.3f}"
        )
        previous = item.tie_group

    print(
        f"\n{result.questions_used} questions in {result.rounds} rounds, "
        f"mean asymmetry {result.mean_asymmetry:.3f}",
        file=sys.stderr,
    )
    if any(len(g) > 1 for g in result.tie_groups):
        print(
            "~ marks items the comparisons could not separate; "
            "their order within a block is arbitrary",
            file=sys.stderr,
        )
    if result.cycles:
        print(f"preference cycles found: {result.cycles}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
