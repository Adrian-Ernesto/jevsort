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
                    "boundaries": [
                        {
                            "above": b.above,
                            "below": b.below,
                            "confidence": round(b.confidence, 4),
                        }
                        for b in result.boundaries
                    ],
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
    confidence_below = {b.above: b.confidence for b in result.boundaries}

    for item in result.items:
        print(f"{item.rank:>3}. {item.key:<{width}}  [{item.rank_low}-{item.rank_high}]")
        gap = confidence_below.get(item.key)
        if gap is not None:
            bar = "=" * int(round(gap * 20))
            note = "" if gap >= 0.90 else ("  coin toss" if gap < 0.60 else "  weak")
            print(f"     {'':<{width}}  {gap:5.2f} {bar}{note}")

    print(
        f"\n{result.questions_used} questions in {result.rounds} rounds, "
        f"mean asymmetry {result.mean_asymmetry:.3f}",
        file=sys.stderr,
    )
    decided = sum(1 for b in result.boundaries if b.decided)
    print(
        f"{decided} of {len(result.boundaries)} boundaries cleared 0.90; "
        "the number under each item is how much the gap below it is worth",
        file=sys.stderr,
    )
    if result.cycles:
        print(f"preference cycles found: {result.cycles}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
