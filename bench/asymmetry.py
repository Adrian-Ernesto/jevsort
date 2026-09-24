"""Measure how far the comparator is from being coherent.

For every pair this asks both directions and records forward + backward. A
comparator that respected P(a beats b) = 1 - P(b beats a) would put that sum at
1.0 every time. The documentation says the model does not guarantee it; this
measures how much it actually costs in practice.

    python bench/asymmetry.py              # real API, needs TYPESAFE_API_KEY
    python bench/asymmetry.py --offline    # simulated judge, no key needed
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data.urgency import labeled
from jevsort.client import JevJudge, OfflineJudge, build_offline_truth
from jevsort.compare import compare_pairs
from jevsort.types import Item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--criterion", default="more urgent for an engineering team to fix first"
    )
    args = parser.parse_args()

    # Offline runs write somewhere else on purpose. They are simulated,
    # and the committed results file holds measurements against the real
    # model -- an --offline run must never quietly overwrite the evidence.
    if args.out is None:
        suffix = "-offline" if args.offline else ""
        args.out = f"bench/results/asymmetry{suffix}.json"


    rows = labeled()[: args.limit]
    items = [Item(key=key, state={"ticket": text}) for key, text, _ in rows]
    slots = [f"i{i}" for i in range(len(items))]

    all_pairs = list(itertools.combinations(range(len(items)), 2))
    step = max(1, len(all_pairs) // args.pairs)
    pairs = all_pairs[::step][: args.pairs]

    judge = (
        OfflineJudge(build_offline_truth(slots), seed=7)
        if args.offline
        else JevJudge()
    )

    started = time.time()
    comparisons = compare_pairs(judge, items, pairs, args.criterion)
    elapsed = time.time() - started

    asymmetries = [c.asymmetry for c in comparisons]

    # Stratify by how far apart the two items are in the true ordering. This is
    # the measurement that matters: a comparator that is only incoherent on
    # pairs whose order was never in doubt costs nothing, while one that is
    # incoherent on neighbours is incoherent exactly where the ordering is
    # decided.
    buckets: dict[int, list[float]] = {}
    for c in comparisons:
        distance = abs(int(c.a[1:]) - int(c.b[1:]))
        buckets.setdefault(min(distance // 4, 3), []).append(c.asymmetry)
    by_distance = {
        f"{b * 4}-{b * 4 + 3}": {
            "mean_asymmetry": round(statistics.fmean(v), 4),
            "pairs": len(v),
        }
        for b, v in sorted(buckets.items())
    }
    sums = [c.forward + c.backward for c in comparisons]
    flips = sum(1 for c in comparisons if (c.forward > 0.5) == (c.backward > 0.5))

    report = {
        "mode": "offline" if args.offline else "jev",
        "items": len(items),
        "pairs": len(comparisons),
        "questions": 2 * len(comparisons),
        "requests": judge.requests_made,
        "seconds": round(elapsed, 2),
        "forward_plus_backward": {
            "mean": round(statistics.fmean(sums), 4),
            "median": round(statistics.median(sums), 4),
            "min": round(min(sums), 4),
            "max": round(max(sums), 4),
        },
        "asymmetry": {
            "mean": round(statistics.fmean(asymmetries), 4),
            "median": round(statistics.median(asymmetries), 4),
            "p90": round(sorted(asymmetries)[int(0.9 * len(asymmetries)) - 1], 4),
            "max": round(max(asymmetries), 4),
            "over_0_10": sum(1 for a in asymmetries if a > 0.10),
            "over_0_25": sum(1 for a in asymmetries if a > 0.25),
        },
        "both_directions_said_yes_or_both_said_no": flips,
        "asymmetry_by_true_rank_distance": by_distance,
        "raw": [
            {
                "a": c.a,
                "b": c.b,
                "forward": round(c.forward, 6),
                "backward": round(c.backward, 6),
                "sum": round(c.forward + c.backward, 6),
                "asymmetry": round(c.asymmetry, 6),
            }
            for c in comparisons
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    a = report["asymmetry"]
    s = report["forward_plus_backward"]
    print(f"mode                       {report['mode']}")
    print(f"pairs                      {report['pairs']} ({report['questions']} questions, {report['requests']} requests, {report['seconds']}s)")
    print(f"forward + backward  mean   {s['mean']}   (1.0 = coherent)")
    print(f"                    range  {s['min']} to {s['max']}")
    print(f"asymmetry           mean   {a['mean']}")
    print(f"                    median {a['median']}")
    print(f"                    p90    {a['p90']}")
    print(f"                    max    {a['max']}")
    print(f"pairs over 0.10            {a['over_0_10']} of {report['pairs']}")
    print(f"pairs over 0.25            {a['over_0_25']} of {report['pairs']}")
    print(f"both directions agreed     {flips} of {report['pairs']}  (self-contradiction)")
    print("\nasymmetry by true rank distance between the two items:")
    for span, stats in by_distance.items():
        print(f"  distance {span:<6} mean {stats['mean_asymmetry']:.4f}   ({stats['pairs']} pairs)")
    print(f"\nraw responses written to   {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
