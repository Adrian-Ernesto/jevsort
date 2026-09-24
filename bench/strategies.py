"""Compare three ways of ordering the same items against a known ground truth.

    score_sort   one score per item, sorted by probability          n questions
    quicksort    the language's sort, comparator asks the model     ~n log n questions
    jevsort      seed, then symmetrized comparisons where it counts  budgeted

Quicksort is run several times on the same input. A correct comparator would
give the same answer every time; the spread across runs is reported as
instability, because a sort built on a probabilistic comparator has no defined
result.

    python bench/strategies.py --offline
    python bench/strategies.py                 # needs TYPESAFE_API_KEY
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data.urgency import labeled
from jevsort import rank
from jevsort.client import JevJudge, OfflineJudge, build_offline_truth, question_id
from jevsort.compare import slot
from jevsort.seed import seed_scores
from jevsort.types import Item


def kendall_tau(order: list[str], truth: dict[str, int]) -> float:
    """Rank correlation with the known ordering. 1.0 is perfect, -1.0 reversed."""
    keys = [k for k in order if k in truth]
    concordant = discordant = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if truth[keys[i]] < truth[keys[j]]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def inversions(order: list[str], truth: dict[str, int]) -> int:
    keys = [truth[k] for k in order]
    return sum(
        1
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
        if keys[i] > keys[j]
    )


def run_score_sort(judge, items, criterion):
    scores = seed_scores(judge, items, criterion)
    ordered = sorted(range(len(items)), key=lambda i: (-scores[slot(i)], i))
    return [items[i].key for i in ordered], len(items)


def run_quicksort(judge, items, criterion, runs):
    """Sort with a comparator that asks the model one direction per comparison."""
    orders, costs = [], []
    for _ in range(runs):
        asked = {"n": 0}

        def compare(x: int, y: int) -> int:
            asked["n"] += 1
            state = {"items": {slot(x): items[x].state, slot(y): items[y].state}}
            qid = question_id(slot(x), slot(y))
            answer = judge.ask(
                state,
                {
                    qid: (
                        f"Comparing only these two items: is `items.{slot(x)}` "
                        f"{criterion} than `items.{slot(y)}`? Answer about "
                        f"`items.{slot(x)}` relative to `items.{slot(y)}`, and "
                        f"ignore every other item."
                    )
                },
            )
            p = answer[qid]
            return -1 if p > 0.5 else (1 if p < 0.5 else 0)

        indices = sorted(range(len(items)), key=functools.cmp_to_key(compare))
        orders.append([items[i].key for i in indices])
        costs.append(asked["n"])
    return orders, costs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--budget", type=int, default=0, help="0 means 6n")
    parser.add_argument("--quicksort-runs", type=int, default=3)
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
        args.out = f"bench/results/strategies{suffix}.json"


    rows = labeled()[: args.limit]
    items = [Item(key=key, state={"ticket": text}) for key, text, _ in rows]
    truth = {key: true_rank for key, _, true_rank in rows}
    slots = [slot(i) for i in range(len(items))]
    budget = args.budget or 6 * len(items)

    def make_judge(seed: int):
        return (
            OfflineJudge(build_offline_truth(slots), noise=0.18, seed=seed)
            if args.offline
            else JevJudge()
        )

    report: dict = {
        "mode": "offline" if args.offline else "jev",
        "items": len(items),
        "criterion": args.criterion,
        "strategies": {},
    }

    judge = make_judge(11)
    started = time.time()
    order, cost = run_score_sort(judge, items, args.criterion)
    report["strategies"]["score_sort"] = {
        "tau": round(kendall_tau(order, truth), 4),
        "inversions": inversions(order, truth),
        "questions": cost,
        "requests": judge.requests_made,
        "seconds": round(time.time() - started, 2),
        "order": order,
    }

    judge = make_judge(12)
    started = time.time()
    orders, costs = run_quicksort(judge, items, args.criterion, args.quicksort_runs)
    taus = [kendall_tau(o, truth) for o in orders]
    distinct = len({tuple(o) for o in orders})
    report["strategies"]["quicksort"] = {
        "tau_mean": round(statistics.fmean(taus), 4),
        "tau_min": round(min(taus), 4),
        "tau_max": round(max(taus), 4),
        "inversions_mean": round(statistics.fmean(inversions(o, truth) for o in orders), 2),
        "questions_mean": round(statistics.fmean(costs), 1),
        "requests": judge.requests_made,
        "seconds": round(time.time() - started, 2),
        "runs": args.quicksort_runs,
        "distinct_orders_across_runs": distinct,
        "unstable": distinct > 1,
        "orders": orders,
    }

    judge = make_judge(13)
    started = time.time()
    result = rank(items, args.criterion, budget=budget, judge=judge)
    report["strategies"]["jevsort"] = {
        "tau": round(kendall_tau(result.order, truth), 4),
        "inversions": inversions(result.order, truth),
        "questions": result.questions_used,
        "requests": judge.requests_made,
        "seconds": round(time.time() - started, 2),
        "budget": budget,
        "rounds": result.rounds,
        "mean_asymmetry": round(result.mean_asymmetry, 4),
        "tie_groups": result.tie_groups,
        "cycles": result.cycles,
        "order": result.order,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    s = report["strategies"]
    print(f"mode: {report['mode']}   items: {report['items']}   budget: {budget}\n")
    print(f"{'strategy':<12} {'tau':>8} {'inv':>6} {'questions':>10} {'requests':>9} {'stable':>7}")
    print("-" * 56)
    print(f"{'score_sort':<12} {s['score_sort']['tau']:>8} {s['score_sort']['inversions']:>6} "
          f"{s['score_sort']['questions']:>10} {s['score_sort']['requests']:>9} {'yes':>7}")
    q = s["quicksort"]
    print(f"{'quicksort':<12} {q['tau_mean']:>8} {q['inversions_mean']:>6} "
          f"{q['questions_mean']:>10} {q['requests']:>9} {('no' if q['unstable'] else 'yes'):>7}")
    j = s["jevsort"]
    print(f"{'jevsort':<12} {j['tau']:>8} {j['inversions']:>6} "
          f"{j['questions']:>10} {j['requests']:>9} {'yes':>7}")
    print(f"\nquicksort produced {q['distinct_orders_across_runs']} different orderings "
          f"across {q['runs']} runs of identical input")
    print(f"jevsort mean comparator asymmetry: {j['mean_asymmetry']}")
    if j["tie_groups"]:
        ambiguous = [g for g in j["tie_groups"] if len(g) > 1]
        print(f"jevsort reported {len(ambiguous)} tie group(s) it could not separate")
    if j["cycles"]:
        print(f"jevsort found {len(j['cycles'])} preference cycle(s): {j['cycles']}")
    print(f"\nfull results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
