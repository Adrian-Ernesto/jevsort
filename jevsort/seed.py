"""The cheap O(n) pass that gives the refinement stage somewhere to start."""

from __future__ import annotations

from typing import Any, Sequence

from .client import Judge
from .compare import slot
from .types import Item

# Items per seeding request. Every item in state that a question is not about is
# a distractor, so the whole candidate set is not sent at once; but one item per
# request would throw away the batching that makes this cheap. Eight is the
# compromise.
SEED_CHUNK = 8


def seed_scores(
    judge: Judge,
    items: Sequence[Item],
    criterion: str,
    chunk_size: int = SEED_CHUNK,
) -> dict[str, float]:
    """Score every item once, independently, to get a prior ordering.

    This costs one question per item and is nowhere near accurate enough to be
    the final answer -- absolute scores are not reliably comparable across
    items. It only has to be better than random, because its job is to tell the
    comparison stage which pairs are worth asking about.
    """
    scores: dict[str, float] = {}
    for start in range(0, len(items), chunk_size):
        window = list(range(start, min(start + chunk_size, len(items))))
        state: dict[str, Any] = {"items": {slot(i): items[i].state for i in window}}
        questions = {
            f"seed__{slot(i)}": (
                f"Considering `items.{slot(i)}` on its own merits: is it {criterion} "
                f"than most items of its kind?"
            )
            for i in window
        }
        answers = judge.ask(state, questions)
        for i in window:
            scores[slot(i)] = float(answers.get(f"seed__{slot(i)}", 0.5))
    return scores


def order_by_scores(scores: dict[str, float], slots: Sequence[str]) -> list[str]:
    """Best first, ties broken by slot for determinism."""
    return sorted(slots, key=lambda s: (-scores.get(s, 0.5), s))
