"""How sure the ordering is, expressed as rank intervals and tie groups."""

from __future__ import annotations

import random
from typing import Sequence

from .aggregate import bradley_terry
from .types import Comparison


def bootstrap_ranks(
    slots: Sequence[str],
    comparisons: Sequence[Comparison],
    resamples: int = 200,
    seed: int = 0,
    iterations: int = 50,
) -> dict[str, list[int]]:
    """Sample plausible orderings and record where each item lands.

    Each comparison's outcome is redrawn from Bernoulli(p) and the model refit,
    because a comparison returning 0.52 is very nearly a coin flip and an
    interval that ignored that would report a confident order built on noise.

    Each comparison is first shrunk toward 0.5 in proportion to its measured
    asymmetry, so pairs where the two directions contradicted each other carry
    correspondingly less weight. That is the second use of the extra question
    spent on symmetrizing.

    The set of pairs is deliberately not resampled. The scheduler chose those
    pairs on purpose, so they are not a random draw from a larger population,
    and resampling them would inject variance that does not exist.
    """
    if not slots:
        return {}
    if not comparisons:
        return {s: [] for s in slots}

    rng = random.Random(seed)
    observed: dict[str, list[int]] = {s: [] for s in slots}
    shrunk = [
        (c, 0.5 + (c.p - 0.5) * (1.0 - min(c.asymmetry, 1.0))) for c in comparisons
    ]

    for _ in range(resamples):
        sample = [
            Comparison(
                a=c.a,
                b=c.b,
                forward=(outcome := 1.0 if rng.random() < p else 0.0),
                backward=1.0 - outcome,
            )
            for c, p in shrunk
        ]
        strength = bradley_terry(slots, sample, iterations=iterations)
        order = sorted(slots, key=lambda s: (-strength.get(s, 0.0), s))
        for position, s in enumerate(order, start=1):
            observed[s].append(position)

    return observed


def rank_intervals(
    ranks: dict[str, list[int]], total: int, confidence: float = 0.90
) -> dict[str, tuple[int, int]]:
    """The central ``confidence`` mass of each item's sampled ranks."""
    tail = (1.0 - confidence) / 2.0
    intervals: dict[str, tuple[int, int]] = {}
    for s, samples in ranks.items():
        if not samples:
            intervals[s] = (1, total)
            continue
        ordered = sorted(samples)
        low = ordered[min(int(tail * len(ordered)), len(ordered) - 1)]
        high = ordered[min(int((1.0 - tail) * len(ordered)), len(ordered) - 1)]
        intervals[s] = (low, high)
    return intervals


def swap_rate(ranks: dict[str, list[int]], a: str, b: str) -> float:
    """How often ``b`` outranked ``a`` across the sampled orderings."""
    left, right = ranks.get(a, []), ranks.get(b, [])
    if not left or not right:
        return 0.0
    swaps = sum(1 for x, y in zip(left, right) if y < x)
    return swaps / len(left)


def tie_groups(
    order: Sequence[str], ranks: dict[str, list[int]], alpha: float = 0.10
) -> dict[str, int]:
    """Group runs of neighbouring items the comparisons could not separate.

    A boundary is drawn between two neighbours when the bootstrap put them in
    the opposite order less than ``alpha`` of the time. Everything between two
    boundaries is one group, and the order printed inside a group is arbitrary.

    Grouping is done on neighbours rather than against a group leader because
    being indistinguishable is not transitive: a may be separable from c while
    b sits too close to both to place between them. Chaining those into one
    block is the conservative reading, and it is the one that does not invent a
    boundary the evidence has not earned.
    """
    groups: dict[str, int] = {}
    current = 0

    for position, s in enumerate(order):
        if position > 0 and swap_rate(ranks, order[position - 1], s) < alpha:
            current += 1
        groups[s] = current
    return groups
