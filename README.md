# jevsort

Ranks a list of items by a natural-language criterion, using a judgment model
that is **not** a consistent comparator, and says so when the result is not
trustworthy.

Built on [TypeSafe](https://typesafe.ai)'s System One model, Jev.

## The problem

You have n items and a fuzzy criterion: which support ticket is most urgent,
which passage best answers a query, which candidate fits the role. Jev answers
questions like that quickly and cheaply. There are two obvious ways to turn it
into an ordering, and both are broken.

**Score each item, then sort by the number.** Costs n questions. But Score
levels are absolute descriptions, and nothing guarantees that a score assigned
to one item is comparable to a score assigned to another. Someone measured
this: [`jev-orderby-bench`](https://github.com/yodablocks/jev-orderby-bench)
found that `ORDER BY` over a Jev probability failed four of six pre-registered
gates on graded product relevance.

**Ask pairwise and hand it to your language's sort.** More accurate per
comparison, but every comparison sort — `sorted`, `qsort`, `Array.sort` —
requires a comparator that is deterministic, antisymmetric and transitive. Jev
is none of these. TypeSafe's own
[jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) states
that structural invariants are not guaranteed, and specifically that
`P(noul) != 1 - P(not_noul)`. Feeding that to a comparison sort is undefined
behaviour: the result depends on the pivot order, and the same input can come
back in a different order on the next run.

This library treats the real problem — recovering an ordering from noisy,
possibly contradictory pairwise evidence — as what it is: rank aggregation, a
problem with decades of existing theory.

## How it works

1. **Seed.** One question per item gives a rough ordering. It is not accurate
   enough to be the answer; its only job is to tell the next stage which pairs
   are worth asking about.
2. **Compare, both ways.** Each pair is asked twice, a-vs-b and b-vs-a. The two
   answers are averaged, which cancels the model's directional bias. How far
   they were from summing to 1 is kept as a per-pair measurement of how
   incoherent the comparator was on that pair.
3. **Spend the budget where it matters.** After each round the model is refit
   and the ordering updated, and the next round asks about the pairs whose
   outcome is least certain, restricted to items near each other in the current
   order. Distant pairs are already separated by a wide margin; asking about
   them confirms what was not in doubt.
4. **Batch every round into one request.** The questions in a round are
   independent, so they run in parallel. A 16-item ranking is a handful of
   requests, not a few hundred sequential calls.
5. **Fit Bradley-Terry.** Every comparison contributes a fractional win rather
   than being thresholded to a hard win, so a 0.51 and a 0.99 are not treated as
   the same evidence. Solved with Hunter's MM algorithm.
6. **Report what is not known.** Outcomes are resampled to produce a rank
   interval per item, and neighbours that swap freely are grouped as tied. A
   comparison that came back incoherent is shrunk toward a coin flip first, so
   pairs the model contradicted itself on carry less weight.
7. **Surface cycles.** If the model says a beats b beats c beats a, no ordering
   is faithful to the evidence. Those groups are found with Tarjan's algorithm
   and reported rather than quietly flattened.

## Run it

```bash
pip install -e .
cp .env.example .env    # add TYPESAFE_API_KEY

printf 'database is down\ntypo in the footer\npayments failing\n' \
  | jevsort "more urgent to fix first"
```

```python
from jevsort import rank

result = rank(tickets, "more urgent to handle first", budget=200, key="id")

result.order          # keys, best first
result.tie_groups     # items that could not be separated
result.unresolved     # what to escalate to a human or a reasoning model
result.cycles         # criterion was incoherent on these items
result.mean_asymmetry # how incoherent the comparator was overall
```

No key? Everything runs against a simulated judge:

```bash
python -m pytest tests/ -q
python bench/strategies.py --offline
```

## Layout

| Path | Purpose |
|---|---|
| `jevsort/types.py` | `Item`, `Comparison`, `RankedItem`, `RankResult` |
| `jevsort/client.py` | Batched requests; `OfflineJudge` for key-free runs |
| `jevsort/compare.py` | Pair to symmetrized question pair; per-round state scoping |
| `jevsort/seed.py` | The cheap one-question-per-item prior |
| `jevsort/schedule.py` | Entropy-ranked pair selection under a budget |
| `jevsort/aggregate.py` | Bradley-Terry MM fit, Copeland, Tarjan cycle detection |
| `jevsort/intervals.py` | Outcome bootstrap, rank intervals, tie grouping |
| `jevsort/ranker.py` | `rank()`, the public entry point |
| `jevsort/cli.py` | `jevsort` command |
| `bench/asymmetry.py` | Measures how far the comparator is from coherent |
| `bench/strategies.py` | Score-sort vs comparison-sort vs jevsort against ground truth |

## Design notes

**State is scoped per round.** Irrelevant state is a documented cause of lower
accuracy, so a round's request carries only the items that round asks about, not
the whole candidate set.

**The pair set is not resampled when estimating intervals.** The scheduler picks
pairs deliberately, so they are not a random draw from a population; resampling
them would model variance that does not exist. Only the outcome of each
comparison is redrawn.

**Ties chain across neighbours.** Being indistinguishable is not transitive: a
can be separable from c while b sits too close to both. Grouping neighbours
rather than measuring against a group leader avoids inventing a boundary the
evidence has not earned.

**Budget is counted in questions, not pairs.** A pair costs two questions
because both directions are asked.

## Status

Working and tested. Benchmarks run against the real API and against a simulated
judge; the simulation reproduces directional bias, per-call noise and
non-comparable absolute scores, but it is a simulation and the numbers that
matter are the ones measured against Jev.

## Colophon

Written by Adrian Ernesto Orozco Rivera, with Claude Code as the pair. The
design calls, the benchmark methodology and the decision to measure rather than
assert are mine; a good deal of the typing is not.

## License

MIT
