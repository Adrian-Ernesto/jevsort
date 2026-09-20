"""Rank things with a model that cannot be trusted to be a consistent comparator."""

from .aggregate import bradley_terry, copeland, find_cycles
from .client import JevJudge, Judge, OfflineJudge, build_offline_truth
from .ranker import rank
from .types import Comparison, Item, RankedItem, RankResult

__version__ = "0.1.0"

__all__ = [
    "rank",
    "Item",
    "Comparison",
    "RankedItem",
    "RankResult",
    "Judge",
    "JevJudge",
    "OfflineJudge",
    "build_offline_truth",
    "bradley_terry",
    "copeland",
    "find_cycles",
]
