"""Turning pairs of items into batched, symmetrized comparison questions."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .client import Judge, question_id
from .types import Comparison, Item

# Items are addressed by slot ("i0", "i1", ...) rather than by their own keys so
# that user keys can contain anything without breaking state paths.
def slot(index: int) -> str:
    return f"i{index}"


def build_state(items: Sequence[Item], involved: Iterable[int]) -> dict[str, Any]:
    """Build request state holding only the items this batch asks about.

    Every item not being compared in this round is a distractor, and irrelevant
    state is a documented cause of lower accuracy. So state is scoped to the
    round rather than holding the whole candidate set.
    """
    return {"items": {slot(i): items[i].state for i in sorted(set(involved))}}


def build_questions(pairs: Sequence[tuple[int, int]], criterion: str) -> dict[str, str]:
    """Build both directions of every pair.

    Asking b-vs-a as well as a-vs-b costs a second question and buys two things:
    an estimate that cancels the model's directional bias, and a per-pair
    measurement of how incoherent the comparator was on that pair.
    """
    questions: dict[str, str] = {}
    for a, b in pairs:
        questions[question_id(slot(a), slot(b))] = _phrase(slot(a), slot(b), criterion)
        questions[question_id(slot(b), slot(a))] = _phrase(slot(b), slot(a), criterion)
    return questions


def _phrase(a: str, b: str, criterion: str) -> str:
    return (
        f"Comparing only these two items: is `items.{a}` {criterion} "
        f"than `items.{b}`? Answer about `items.{a}` relative to `items.{b}`, "
        f"and ignore every other item."
    )


def compare_pairs(
    judge: Judge,
    items: Sequence[Item],
    pairs: Sequence[tuple[int, int]],
    criterion: str,
) -> list[Comparison]:
    """Ask about every pair in one batch and return symmetrized comparisons."""
    if not pairs:
        return []

    involved = [index for pair in pairs for index in pair]
    state = build_state(items, involved)
    questions = build_questions(pairs, criterion)
    answers = judge.ask(state, questions)

    comparisons: list[Comparison] = []
    for a, b in pairs:
        forward = answers.get(question_id(slot(a), slot(b)))
        backward = answers.get(question_id(slot(b), slot(a)))
        if forward is None or backward is None:
            continue
        comparisons.append(
            Comparison(a=slot(a), b=slot(b), forward=forward, backward=backward)
        )
    return comparisons
