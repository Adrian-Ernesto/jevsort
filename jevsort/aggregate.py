"""Turning a bag of noisy pairwise comparisons into one ordering."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .types import Comparison


def bradley_terry(
    slots: Sequence[str],
    comparisons: Sequence[Comparison],
    regularization: float = 0.05,
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths from fractional wins.

    Each comparison contributes ``p`` of a win to ``a`` and ``1 - p`` to ``b``,
    so the symmetrized probability is used directly instead of being thresholded
    into a hard win. Solved with Hunter's MM algorithm, which is monotonic and
    needs no step size.

    ``regularization`` adds a small fictitious win and loss to every observed
    pair. Without it an item that won every comparison has unbounded strength
    and the fit diverges.
    """
    if not slots:
        return {}

    wins: dict[str, float] = defaultdict(float)
    counts: dict[tuple[str, str], float] = defaultdict(float)

    for comparison in comparisons:
        a, b, p = comparison.a, comparison.b, comparison.p
        wins[a] += p + regularization
        wins[b] += (1.0 - p) + regularization
        pair = _pair_key(a, b)
        counts[pair] += 1.0 + 2.0 * regularization

    opponents: dict[str, list[str]] = defaultdict(list)
    for a, b in counts:
        opponents[a].append(b)
        opponents[b].append(a)

    strength = {s: 1.0 for s in slots}

    for _ in range(iterations):
        updated: dict[str, float] = {}
        for s in slots:
            if not opponents[s] or wins[s] <= 0.0:
                updated[s] = strength[s]
                continue
            denominator = 0.0
            for other in opponents[s]:
                n = counts[_pair_key(s, other)]
                total = strength[s] + strength[other]
                if total > 0.0:
                    denominator += n / total
            updated[s] = wins[s] / denominator if denominator > 0.0 else strength[s]

        mean = sum(updated.values()) / len(updated)
        if mean > 0.0:
            updated = {s: v / mean for s, v in updated.items()}

        shift = max(abs(updated[s] - strength[s]) for s in slots)
        strength = updated
        if shift < tolerance:
            break

    return strength


def log_strength(strength: dict[str, float]) -> dict[str, float]:
    """Strengths on a log scale, where differences are comparable."""
    return {s: math.log(max(v, 1e-12)) for s, v in strength.items()}


def copeland(slots: Sequence[str], comparisons: Sequence[Comparison]) -> dict[str, float]:
    """Count majority wins per item. Used as a tie-break and a sanity check."""
    score = {s: 0.0 for s in slots}
    for comparison in comparisons:
        if comparison.p > 0.5:
            score[comparison.a] = score.get(comparison.a, 0.0) + 1.0
        elif comparison.p < 0.5:
            score[comparison.b] = score.get(comparison.b, 0.0) + 1.0
        else:
            score[comparison.a] = score.get(comparison.a, 0.0) + 0.5
            score[comparison.b] = score.get(comparison.b, 0.0) + 0.5
    return score


def find_cycles(
    slots: Sequence[str],
    comparisons: Sequence[Comparison],
    margin: float = 0.05,
) -> list[list[str]]:
    """Find groups of items whose majority preferences run in a circle.

    An edge a -> b is drawn when the model preferred a over b by more than
    ``margin``. Any strongly connected component larger than one item is a
    Condorcet cycle: a beats b beats c beats a. No ordering of those items is
    faithful to the comparisons, which is a fact about the criterion and worth
    reporting rather than smoothing away.

    Iterative Tarjan, so deep graphs cannot blow the recursion limit.
    """
    graph: dict[str, list[str]] = {s: [] for s in slots}
    for comparison in comparisons:
        if comparison.p > 0.5 + margin:
            graph[comparison.a].append(comparison.b)
        elif comparison.p < 0.5 - margin:
            graph[comparison.b].append(comparison.a)

    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    counter = 0
    components: list[list[str]] = []

    for root in slots:
        if root in index_of:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work[-1]
            if child_index == 0:
                index_of[node] = counter
                low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True

            if child_index < len(graph[node]):
                work[-1] = (node, child_index + 1)
                child = graph[node][child_index]
                if child not in index_of:
                    work.append((child, 0))
                elif on_stack.get(child):
                    low[node] = min(low[node], index_of[child])
                continue

            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

            if low[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    components.append(sorted(component))

    return components


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)
