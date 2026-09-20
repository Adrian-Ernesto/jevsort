"""Choosing which pairs are worth spending the budget on."""

from __future__ import annotations

import math
from typing import Sequence


def entropy(p: float) -> float:
    """Binary entropy. Highest at p = 0.5, which is where a comparison is worth asking."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def expected_probability(strength_a: float, strength_b: float) -> float:
    """What Bradley-Terry currently believes about P(a beats b)."""
    total = strength_a + strength_b
    return 0.5 if total <= 0.0 else strength_a / total


def adjacent_pairs(order: Sequence[str]) -> list[tuple[str, str]]:
    """Neighbouring pairs in the current ordering.

    These are where the ordering is most likely to be wrong: items far apart are
    already separated by a wide strength gap, and asking about them spends
    budget confirming what is not in doubt.
    """
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def select_pairs(
    order: Sequence[str],
    strength: dict[str, float],
    asked: set[tuple[str, str]],
    limit: int,
    window: int = 3,
) -> list[tuple[str, str]]:
    """Pick up to ``limit`` unasked pairs, most informative first.

    Candidates are restricted to pairs within ``window`` positions of each other
    in the current ordering, which keeps selection linear in the number of items
    rather than quadratic. Among those, pairs are ranked by the entropy of what
    the model currently believes, so budget goes to genuinely undecided pairs.
    """
    if limit <= 0:
        return []

    scored: list[tuple[float, tuple[str, str]]] = []
    for i, a in enumerate(order):
        for offset in range(1, window + 1):
            j = i + offset
            if j >= len(order):
                break
            b = order[j]
            key = pair_key(a, b)
            if key in asked:
                continue
            p = expected_probability(strength.get(a, 1.0), strength.get(b, 1.0))
            # Closer neighbours break ties, since an inversion between adjacent
            # items costs more than one between distant ones.
            scored.append((entropy(p) - 0.001 * offset, key))

    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [key for _, key in scored[:limit]]


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)
