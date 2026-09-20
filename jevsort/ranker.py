"""The public entry point: rank a list of items by a natural-language criterion."""

from __future__ import annotations

from typing import Any, Sequence

from .aggregate import bradley_terry, find_cycles
from .client import Judge, JevJudge
from .compare import compare_pairs, slot
from .intervals import bootstrap_ranks, rank_intervals, swap_rate, tie_groups
from .schedule import pair_key, select_pairs
from .seed import order_by_scores, seed_scores
from .types import Boundary, Comparison, Item, RankedItem, RankResult, as_items

# Pairs asked about in one round. Each costs two questions, so this stays inside
# a single request at the default question limit.
PAIRS_PER_ROUND = 24


def rank(
    items: Sequence[Any],
    criterion: str,
    budget: int | None = None,
    judge: Judge | None = None,
    key: str | None = None,
    seed: bool = True,
    window: int = 3,
    pairs_per_round: int = PAIRS_PER_ROUND,
    resamples: int = 200,
    confidence: float = 0.90,
    cycle_margin: float = 0.05,
) -> RankResult:
    """Order ``items`` best-first by ``criterion``.

    ``criterion`` completes the sentence "is this item ___ than that one", for
    example "more urgent to handle first" or "a better answer to the question".

    ``budget`` is a ceiling on questions asked, not pairs compared. A pair costs
    two questions because both directions are asked. Left as None it defaults to
    roughly ``6 * n``, which in practice resolves most orderings well short of
    the ``n * (n - 1)`` that comparing everything would cost.

    Raising ``window`` lets the scheduler consider pairs further apart in the
    current ordering: more thorough, more expensive.
    """
    parsed = as_items(items, key=key)
    n = len(parsed)
    slots = [slot(i) for i in range(n)]
    by_slot = {slot(i): parsed[i] for i in range(n)}

    if n == 0:
        return RankResult(items=[])
    if n == 1:
        only = parsed[0]
        return RankResult(
            items=[RankedItem(only.key, 1, 1.0, 1, 1, 0)], seeded=False
        )

    if judge is None:
        judge = JevJudge()
    if budget is None:
        budget = 6 * n

    used = 0
    comparisons: list[Comparison] = []
    asked: set[tuple[str, str]] = set()

    if seed and budget >= n:
        scores = seed_scores(judge, parsed, criterion)
        used += n
        order = order_by_scores(scores, slots)
        seeded = True
    else:
        order = list(slots)
        seeded = False

    strength = {s: 1.0 for s in slots}
    rounds = 0

    while used + 2 <= budget:
        affordable = (budget - used) // 2
        limit = min(pairs_per_round, affordable)
        selected = select_pairs(order, strength, asked, limit=limit, window=window)
        if not selected:
            break

        index_of = {s: i for i, s in enumerate(slots)}
        batch = [(index_of[a], index_of[b]) for a, b in selected]
        fresh = compare_pairs(judge, parsed, batch, criterion)
        if not fresh:
            break

        comparisons.extend(fresh)
        for comparison in fresh:
            asked.add(pair_key(comparison.a, comparison.b))
        used += 2 * len(fresh)
        rounds += 1

        strength = bradley_terry(slots, comparisons)
        order = sorted(slots, key=lambda s: (-strength.get(s, 0.0), s))

    if comparisons:
        strength = bradley_terry(slots, comparisons)
        order = sorted(slots, key=lambda s: (-strength.get(s, 0.0), s))

    ranks = bootstrap_ranks(slots, comparisons, resamples=resamples)
    intervals = rank_intervals(ranks, total=n, confidence=confidence)
    groups = tie_groups(order, ranks)
    cycles = find_cycles(slots, comparisons, margin=cycle_margin)

    ranked = [
        RankedItem(
            key=by_slot[s].key,
            rank=position,
            strength=strength.get(s, 1.0),
            rank_low=intervals.get(s, (position, position))[0],
            rank_high=intervals.get(s, (position, position))[1],
            tie_group=groups.get(s, position - 1),
        )
        for position, s in enumerate(order, start=1)
    ]

    boundaries = [
        Boundary(
            above=by_slot[a].key,
            below=by_slot[b].key,
            confidence=1.0 - swap_rate(ranks, a, b),
        )
        for a, b in zip(order, order[1:])
    ]

    return RankResult(
        items=ranked,
        boundaries=boundaries,
        comparisons=comparisons,
        cycles=[[by_slot[s].key for s in cycle] for cycle in cycles],
        questions_used=used,
        seeded=seeded,
        rounds=rounds,
    )
