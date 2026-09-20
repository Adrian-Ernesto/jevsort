"""Core data types for jevsort."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class Item:
    """One thing to be ranked.

    ``key`` identifies the item to calling code. ``state`` is the payload sent
    to the model; it should already be filtered down to the fields the criterion
    needs, because irrelevant state measurably degrades accuracy.
    """

    key: str
    state: Any

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Item.key must be a non-empty string")


@dataclass(frozen=True)
class Comparison:
    """One symmetrized pairwise comparison between two items.

    The model is asked in both directions. ``forward`` is P(a beats b) as asked,
    ``backward`` is P(b beats a) as asked. A perfectly coherent comparator would
    satisfy ``forward + backward == 1``; Jev does not guarantee this, so we keep
    both numbers and record how far off they were.
    """

    a: str
    b: str
    forward: float
    backward: float

    @property
    def p(self) -> float:
        """Symmetrized estimate of P(a beats b), in [0, 1]."""
        return (self.forward + (1.0 - self.backward)) / 2.0

    @property
    def asymmetry(self) -> float:
        """How far the two directions were from summing to 1.

        Zero means the two askings agreed exactly. One is total disagreement.
        This is a free per-pair noise estimate: it costs nothing beyond the
        second question we already asked to symmetrize.
        """
        return abs(self.forward + self.backward - 1.0)

    @property
    def questions_used(self) -> int:
        return 2


@dataclass(frozen=True)
class Boundary:
    """The gap between two neighbouring items, and how much it is worth.

    ``confidence`` is the share of resampled orderings that kept these two in
    the printed order. A boundary at 0.55 is barely better than a coin toss, and
    reporting that is more useful than folding it into a tie group and losing
    the number.
    """

    above: str
    below: str
    confidence: float

    @property
    def decided(self) -> bool:
        return self.confidence >= 0.90


@dataclass(frozen=True)
class RankedItem:
    """An item's position in the final ordering."""

    key: str
    rank: int
    strength: float
    rank_low: int
    rank_high: int
    tie_group: int

    @property
    def span(self) -> int:
        """How many positions this item could plausibly occupy."""
        return self.rank_high - self.rank_low + 1


@dataclass
class RankResult:
    """The outcome of a ranking run."""

    items: list[RankedItem]
    boundaries: list[Boundary] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    questions_used: int = 0
    seeded: bool = True
    rounds: int = 0

    @property
    def order(self) -> list[str]:
        """Item keys, best first."""
        return [item.key for item in self.items]

    @property
    def unresolved(self) -> list[RankedItem]:
        """Items sharing a tie group with at least one other item.

        The comparisons could not separate these, so any order printed among
        them is arbitrary. These are the candidates to escalate to a reasoning
        model or a human, and the reason the library reports groups at all.
        """
        sizes: dict[int, int] = {}
        for item in self.items:
            sizes[item.tie_group] = sizes.get(item.tie_group, 0) + 1
        return [item for item in self.items if sizes[item.tie_group] > 1]

    @property
    def tie_groups(self) -> list[list[str]]:
        """Groups of items that could not be separated, best group first."""
        groups: dict[int, list[str]] = {}
        for item in self.items:
            groups.setdefault(item.tie_group, []).append(item.key)
        return [groups[g] for g in sorted(groups)]

    @property
    def mean_asymmetry(self) -> float:
        """Average comparator incoherence across every pair we asked about."""
        if not self.comparisons:
            return 0.0
        return sum(c.asymmetry for c in self.comparisons) / len(self.comparisons)

    @property
    def weakest_boundary(self) -> Boundary | None:
        """The least trustworthy place in the ordering."""
        return min(self.boundaries, key=lambda b: b.confidence, default=None)

    def confident_prefix(self, threshold: float = 0.90) -> list[str]:
        """The longest run from the top whose every boundary clears ``threshold``.

        This is usually the practical answer: not the whole ordering, but how
        far down it can be trusted before the evidence runs out.
        """
        prefix = [self.items[0].key] if self.items else []
        for boundary in self.boundaries:
            if boundary.confidence < threshold:
                break
            prefix.append(boundary.below)
        return prefix

    def top(self, n: int) -> list[str]:
        return self.order[:n]


def as_items(raw: Sequence[Any], key: str | None = None) -> list[Item]:
    """Coerce a user-supplied sequence into ``Item`` objects.

    Accepts ``Item`` instances, ``(key, state)`` pairs, plain strings, or dicts
    (with ``key`` naming the field to use as the identifier).
    """
    out: list[Item] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, Item):
            out.append(entry)
        elif isinstance(entry, tuple) and len(entry) == 2:
            out.append(Item(key=str(entry[0]), state=entry[1]))
        elif isinstance(entry, dict) and key is not None:
            if key not in entry:
                raise KeyError(f"item {index} has no field {key!r}")
            out.append(Item(key=str(entry[key]), state=entry))
        elif isinstance(entry, str):
            out.append(Item(key=entry, state=entry))
        else:
            out.append(Item(key=str(index), state=entry))

    seen = {item.key for item in out}
    if len(seen) != len(out):
        raise ValueError("item keys must be unique")
    return out
