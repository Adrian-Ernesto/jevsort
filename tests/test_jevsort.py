"""Tests run entirely against OfflineJudge, so they need no API key."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevsort import OfflineJudge, build_offline_truth, rank
from jevsort.aggregate import bradley_terry, find_cycles
from jevsort.schedule import entropy, select_pairs
from jevsort.types import Comparison, as_items


def slots(n):
    return [f"i{i}" for i in range(n)]


def test_symmetrized_probability_and_asymmetry():
    c = Comparison("i0", "i1", forward=0.8, backward=0.3)
    assert abs(c.p - 0.75) < 1e-9
    assert abs(c.asymmetry - 0.1) < 1e-9
    coherent = Comparison("i0", "i1", forward=0.8, backward=0.2)
    assert coherent.asymmetry == 0.0


def test_bradley_terry_recovers_transitive_order():
    cs = [
        Comparison("i0", "i1", 0.9, 0.1),
        Comparison("i1", "i2", 0.9, 0.1),
        Comparison("i0", "i2", 0.95, 0.05),
    ]
    strength = bradley_terry(slots(3), cs)
    assert strength["i0"] > strength["i1"] > strength["i2"]


def test_bradley_terry_does_not_diverge_on_a_perfect_winner():
    cs = [Comparison("i0", "i1", 1.0, 0.0), Comparison("i0", "i2", 1.0, 0.0)]
    strength = bradley_terry(slots(3), cs)
    assert all(v == v and v != float("inf") for v in strength.values())


def test_cycle_detection_finds_rock_paper_scissors():
    cs = [
        Comparison("i0", "i1", 0.9, 0.1),
        Comparison("i1", "i2", 0.9, 0.1),
        Comparison("i2", "i0", 0.9, 0.1),
    ]
    assert find_cycles(slots(3), cs) == [["i0", "i1", "i2"]]


def test_no_cycle_when_order_is_consistent():
    cs = [
        Comparison("i0", "i1", 0.9, 0.1),
        Comparison("i1", "i2", 0.9, 0.1),
        Comparison("i0", "i2", 0.9, 0.1),
    ]
    assert find_cycles(slots(3), cs) == []


def test_entropy_peaks_at_a_coin_flip():
    assert entropy(0.5) > entropy(0.7) > entropy(0.99)


def test_scheduler_prefers_the_undecided_pair():
    order = ["i0", "i1", "i2"]
    strength = {"i0": 4.0, "i1": 3.9, "i2": 0.2}
    assert select_pairs(order, strength, set(), limit=1) == [("i0", "i1")]


def test_scheduler_skips_pairs_already_asked():
    order = ["i0", "i1", "i2"]
    strength = {"i0": 4.0, "i1": 3.9, "i2": 0.2}
    picked = select_pairs(order, strength, {("i0", "i1")}, limit=1)
    assert picked and picked[0] != ("i0", "i1")


def test_rank_recovers_a_clean_ordering():
    keys = [f"item{i}" for i in range(10)]
    judge = OfflineJudge(build_offline_truth(slots(10)), noise=0.10, seed=1)
    result = rank(keys, "more urgent", judge=judge, budget=120)
    assert result.order == keys


def test_rank_stays_within_budget():
    keys = [f"item{i}" for i in range(12)]
    judge = OfflineJudge(build_offline_truth(slots(12)), seed=2)
    result = rank(keys, "more urgent", judge=judge, budget=40)
    assert result.questions_used <= 40
    assert judge.questions_asked <= 40


def test_identical_items_are_never_separated():
    """b and c have the same hidden strength, so no boundary may fall between them."""
    judge = OfflineJudge({"i0": 1.0, "i1": 0.5, "i2": 0.5, "i3": 0.0}, seed=5)
    result = rank(["a", "b", "c", "d"], "more urgent", judge=judge, budget=60)
    group_of = {item.key: item.tie_group for item in result.items}
    assert group_of["b"] == group_of["c"]
    assert group_of["d"] != group_of["c"]
    assert {"b", "c"} <= {i.key for i in result.unresolved}


def test_well_separated_items_are_not_all_lumped_together():
    judge = OfflineJudge(build_offline_truth(slots(6)), noise=0.04, seed=8)
    result = rank([f"item{i}" for i in range(6)], "more urgent", judge=judge, budget=80)
    assert len(result.tie_groups) > 1


def test_batching_keeps_request_count_far_below_question_count():
    keys = [f"item{i}" for i in range(12)]
    judge = OfflineJudge(build_offline_truth(slots(12)), seed=3)
    result = rank(keys, "more urgent", judge=judge, budget=120)
    assert judge.requests_made < result.questions_used / 4


def test_single_item_and_empty_input():
    assert rank([], "x", judge=OfflineJudge({})).order == []
    assert rank(["only"], "x", judge=OfflineJudge({})).order == ["only"]


def test_duplicate_keys_are_rejected():
    try:
        as_items(["a", "a"])
    except ValueError:
        return
    raise AssertionError("duplicate keys should raise")
