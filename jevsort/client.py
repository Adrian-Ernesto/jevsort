"""Batched access to the model, plus an offline judge for tests and demos."""

from __future__ import annotations

import math
import os
import random
from typing import Any, Protocol, Sequence

DEFAULT_MODEL = "jev-1.13"

# Questions sent in a single request. Comparisons cost two questions each, so
# this caps a round at MAX_QUESTIONS_PER_REQUEST // 2 pairs before chunking.
MAX_QUESTIONS_PER_REQUEST = 64


class Judge(Protocol):
    """Anything that can answer a batch of yes/no questions over shared state.

    Implemented by :class:`JevJudge` against the real API and by
    :class:`OfflineJudge` for tests and for running the library with no key.
    """

    def ask(self, state: dict[str, Any], questions: dict[str, str]) -> dict[str, float]:
        """Return P(yes) for each question id."""


class JevJudge:
    """Asks Jev, batching every independent question into one request.

    Batching is the whole point: the questions in a round are independent, so
    they are evaluated in parallel by the service rather than in sequence.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        max_questions: int = MAX_QUESTIONS_PER_REQUEST,
    ) -> None:
        self.model = model
        self.max_questions = max_questions
        self.requests_made = 0
        self.questions_asked = 0
        if client is not None:
            self._client = client
        else:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError(
                    "TYPESAFE_API_KEY is not set. Export it, or pass "
                    "judge=OfflineJudge() to run without the API."
                )
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient()

    def ask(self, state: dict[str, Any], questions: dict[str, str]) -> dict[str, float]:
        from typesafe_sdk import Noul

        answers: dict[str, float] = {}
        ids = list(questions)
        for start in range(0, len(ids), self.max_questions):
            chunk = ids[start : start + self.max_questions]
            payload = {qid: Noul(instructions=questions[qid]) for qid in chunk}
            response = self._client.system_one(
                state=state, questions=payload, model=self.model
            )
            self.requests_made += 1
            self.questions_asked += len(chunk)
            for qid in chunk:
                answers[qid] = float(response.nouls[qid].noul)
        return answers


class OfflineJudge:
    """A deterministic stand-in so the library runs with no API key.

    It ranks by a hidden numeric field, and deliberately reproduces the three
    properties that make ranking with a judgment model hard:

    * comparisons are noisy, and noisy in a way that varies call to call, so a
      comparison sort has no stable result;
    * the two directions of a pair do not sum to one;
    * absolute per-item scores carry an item-specific offset, so they are not
      reliably comparable across items even when pairwise judgments are sound.

    It is a simulation for tests and demos. The numbers that matter come from
    running the benchmarks against the real API.
    """

    def __init__(
        self,
        truth: dict[str, float],
        noise: float = 0.12,
        bias: float = 0.06,
        jitter: float = 0.07,
        score_offset: float = 0.22,
        seed: int = 0,
    ) -> None:
        self.truth = truth
        self.noise = noise
        self.bias = bias
        self.jitter = jitter
        self._rng = random.Random(seed)
        # Fixed per-item offsets: the same item always scores with the same
        # distortion, which is what makes absolute scores non-comparable rather
        # than merely noisy.
        offsets = random.Random(seed + 1)
        self._offset = {
            key: offsets.uniform(-score_offset, score_offset) for key in truth
        }
        self.requests_made = 0
        self.questions_asked = 0

    def ask(self, state: dict[str, Any], questions: dict[str, str]) -> dict[str, float]:
        self.requests_made += 1
        self.questions_asked += len(questions)
        answers: dict[str, float] = {}
        for qid in questions:
            if qid.startswith("seed__"):
                answers[qid] = self._score(qid[len("seed__") :])
                continue
            a, b = _parse_pair(qid)
            answers[qid] = 0.5 if a is None or b is None else self._compare(a, b)
        return answers

    def _compare(self, a: str, b: str) -> float:
        gap = self.truth.get(a, 0.0) - self.truth.get(b, 0.0)
        p = 1.0 / (1.0 + math.exp(-gap / max(self.noise, 1e-6)))
        # A fixed nudge toward "yes" regardless of direction. This is what breaks
        # forward + backward == 1, exactly as the real model does.
        p = p * (1.0 - self.bias) + self.bias
        p += self._rng.uniform(-self.jitter, self.jitter)
        return min(max(p, 0.0), 1.0)

    def _score(self, key: str) -> float:
        value = self.truth.get(key, 0.5) + self._offset.get(key, 0.0)
        value += self._rng.uniform(-self.jitter, self.jitter)
        return min(max(value, 0.0), 1.0)


def _parse_pair(qid: str) -> tuple[str | None, str | None]:
    """Recover the two item keys from a question id of the form ``cmp__a__vs__b``."""
    if not qid.startswith("cmp__") or "__vs__" not in qid:
        return None, None
    body = qid[len("cmp__") :]
    a, _, b = body.partition("__vs__")
    return a or None, b or None


def question_id(a: str, b: str) -> str:
    return f"cmp__{a}__vs__{b}"


def build_offline_truth(keys: Sequence[str]) -> dict[str, float]:
    """Assign descending hidden strengths, first key strongest."""
    n = max(len(keys) - 1, 1)
    return {key: 1.0 - (index / n) for index, key in enumerate(keys)}
