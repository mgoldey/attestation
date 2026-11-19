# Replicate-aware comparison

**Date:** 2026-11-19
**Status:** implemented in `replicates.py`, called from `ledger._compare`; no new tool, no migration.
**Depends on:** the run ledger (`2026-08-12-run-ledger-design.md`), the corpus ledger
(`2026-08-20-corpus-ledger-design.md`).

## Problem

`runs.compare` ends every comparison with "each arm is a single run; no seed
replication, so this ranking cannot separate configuration from run-to-run
variance". The line is unconditional, and on a sweep that HAS seeds it is false
in the most damaging way: the seeds are on disk, and the tool ranks them
against each other.

Measured on a real sweep, not a fixture: three sklearn classifiers on the
digits dataset, five seeds each (seed = the train/test split and the model's
`random_state`), written as `results/clf_<model>_seed<N>.json` -- the commonest
way a person names a seeded result. Before this change:

```
  family clf-knn                          5 run(s)  (digits)
  family clf-logreg                       5 run(s)  (digits)
  family clf-sgd                          5 run(s)  (digits)

clf-knn — ranked by accuracy (higher_is_better)
  clf_knn_seed3                                    0.9870    540
  clf_knn_seed4                                    0.9852    540
  ...
winner: clf_knn_seed3
  caveat: each arm is a single run; no seed replication, ...
```

Two failures, one level apart:

1. **The sweep is split into one family per configuration.** `family_of`
   strips `seed<N>` as a variant token, sees a "series", and keeps everything
   before it as the family -- so `clf_knn_seed3` is family `clf-knn`, and the
   question the sweep exists for (knn vs logreg vs sgd) cannot be asked at all.
   Without the seed token the same names (`clf_knn`, `clf_logreg`) form ONE
   family, `clf`. A seed token changed what was comparable.
2. **Inside each family the replicates compete.** Five runs of one
   configuration get a named "winner", which is a statement about a random
   seed, and the caveat beneath it denies the replication that produced them.

## Decisions

### A seed token never changes a run's family (`family_of`)

The triage asked that `ledger_adapters/generic.py::family_of` be left alone
unless this spec showed it necessary. It is: with family_of unchanged the
flat layout above can only ever produce a one-arm family per configuration,
and acceptance test 1 (`fam_a_seed1..3`, `fam_b_seed1..3` -> two arms) cannot
pass. The change is the smallest rule that states the intent:

> a name's family is the family of the same name with its seed token removed.

So `clf_knn_seed3` -> `clf_knn` -> `clf`, exactly as an unseeded sweep groups.
`run_seed1` (nothing left to drop once the seed is gone) keeps `run` as its
family, which is what it was before -- replicates of a one-configuration
experiment still group. A bare `seed_3` is untouched (still family `seed`,
pinned by an existing test). Only names carrying a seed token change family,
and they change toward the family the unseeded names already had.

Nothing else in the adapter changes. In particular the subdirectory layout
(`results/kd_t4/seed1.json`) keeps its directory as the family: each
configuration directory is its own family, and its seeds now pool into one
arm within it, rather than competing. Changing `_result_family` to regroup
directories is a separate decision with its own risk (a directory name is not
a sweep variant by convention) and is out of scope.

### Grouping: the name proposes, the config can only veto

`replicates.group(arms, configs)` merges arms of one family into one replicate
arm when ALL of:

- each name carries exactly one seed token (`seed1`, `seed_1`, `seed-1`,
  `seed=1`, case-insensitive, digits only, bounded by a separator or the end),
  and removing it leaves the same non-empty base name. A nested arm
  (`run_seed1[arms.A]`) keeps its bracket suffix in the base;
- the seed values differ (two runs claiming one seed are not two samples);
- their `split`, once a seed token is removed from it, is the same (the
  generic adapter records `_split_from(stem)`, which for `kd_seed1` is
  `seed1` -- the seed again, not a split);
- where two members BOTH have a `config_json`, those configs are equal apart
  from `seed`. A missing config does not veto: most JSON results carry only
  the seed, and absence is not disagreement.

Any failed condition leaves EVERY member of that base group as a separate
arm, unchanged, and adds a caveat naming the members and the reason (for a
config mismatch, the differing keys). Partial merging was rejected: choosing
which subset is "the real configuration" is the guess this module exists to
refuse. A wrong merge silently averages two configurations -- the
family-pooling bug `compare()`'s docstring describes, one level down -- so the
asymmetric rule is deliberate: a config can stop a merge, never start one.

**A tracker's run id never takes part in pooling.** mlflow names an arm
`<run_name>/<run id[:8]>` and wandb `<program>/<short id>`; when the label
before the last `/` carries the seed token and the last segment is an
8-character id, the id is dropped before the seed is, so `clf_seed1/abc12345`
and `clf_seed2/ffe12345` pool as `clf`. Code review (2026-11-19) found the
first version left the id in, so no mlflow or wandb seed ever pooled while the
lookalike caveat advised a rename the runs already had.

**Config-only replicates are reported, not merged.** Tracker adapters
(hydra multirun `train/0`, `train/1`; wandb run ids) name runs without a seed
token. When two unmerged arms have configs with at least one non-seed key,
equal apart from a seed that differs, a caveat says they look like replicates
and how to name them so they pool. Merging on config alone was rejected:
for plain JSON results `config_json` is `{"seed": ...}` only, so "equal apart
from seed" would be vacuously true and merge an entire sweep into one arm.

### What a replicate arm reports (the shared contract)

A merged arm keeps every key a single-run arm has, plus:

| key | meaning |
|---|---|
| `n_runs` | members merged (>= 2) |
| `mean` | `math.fsum(values) / n` of each member's own best-step value |
| `value` | `= mean`, so ranking, the winner and every existing caveat read it unchanged |
| `sd` | sample standard deviation (n - 1) |
| `min`, `max` | the range, always reported |
| `ci_low`, `ci_high` | 95% percentile bootstrap interval of the mean; `None` when `n_runs < 3` |
| `runs` | member names, sorted, at most `MAX_RUN_NAMES` (5); `n_runs` is the true count |

`name` becomes the base name; `source_path` is the first member's and
`source_paths` lists every member's (uncapped, for prereg's `post_hoc` check;
the MCP layer drops it from `runs.compare`'s payload); `step`
and `split` are the members' common value or `None`; `n` (samples per
evaluation) is the members' common value, else the smallest, else `None`.
**A single-run arm gains no key.** Absent keys mean a single run; a consumer
(preregistration, `prereg.evaluate`) reads them with `.get()`.

The result dict gains `separable` and `effect_size` ONLY when some arm is a
replicate arm, so a family of single runs returns exactly the keys and the
text it returned before -- `examples/workspace`'s pinned `kdsweep` output
(including the "each arm is a single run" line) is byte-identical.

### Statistics: few, deterministic, stdlib-only, and modest in wording

- **Bootstrap.** Percentile bootstrap of the mean, `N_RESAMPLES = 2000`,
  `random.Random(BOOTSTRAP_SEED = 0)`. Indices are drawn as
  `int(rng.random() * n)` rather than through `choices`/`randrange`, so the
  output depends only on the Mersenne Twister stream for an integer seed,
  which Python keeps stable across versions; resample means use `math.fsum`.
  Two calls, two processes and Python 3.12/3.13 give the same floats, and a
  test pins them.
- **Below three runs there is no interval.** A percentile bootstrap of two
  values has three possible resample means; calling that a 95% interval would
  print confidence the data cannot carry. Mean and range are reported and a
  caveat says why there is no interval.
- **Small-n bootstrap intervals are too narrow,** and that is said rather than
  hidden: the percentile interval of n runs never extends beyond their min and
  max, and undercovers for n under about 10. The caveat attached to every
  interval says so and repeats that this is not a significance test.
- **`separable`** between the top two arms: `True` when both have intervals
  and they are disjoint, `False` when both have intervals and they overlap
  (the caveat names both intervals), `None` when either lacks one (a single
  run or `n_runs < 3`). `winner` stays the best mean regardless: refusing to
  name one would break every caller, and the caveats are where the
  qualification belongs, as it already is for "the top two differ by <5%".
- **`effect_size`** between the top two: Cohen's d with the pooled sample SD,
  `{"between": [top, second], "cohen_d": d}`, oriented so positive means the
  winner is better. Only when both have `n_runs >= 2` and the pooled SD is
  non-zero; otherwise `None`.

No p-value is computed. Arms are typically 3-5 runs, a t-test's assumptions
are not checkable at that size, and a p-value is exactly the number a reader
would quote as the result. An interval and an effect size, each labelled
with its limits, are what this data supports.

### Caveats become conditional on what was found

- the single-run line fires verbatim only when EVERY arm is a single run;
- mixed families name the single-run arms beside replicated ones;
- unequal replicate counts across replicated arms;
- `n_runs < 3`: "a CI from 2 runs is not a CI", mean and range only;
- overlapping intervals of the top two: "not separable", both intervals
  named;
- refused merges and config-only lookalikes, as above;
- one line describing how intervals were computed and their small-n limit;
- the "top two differ by <5%, too close to call" rule of thumb is skipped when
  both top arms carry intervals. It stands in for a variance the tool usually
  cannot see; when it can see one, `separable` answers instead. On the real
  sweep above it called a 1.4% gap "too close" beside disjoint intervals --
  two caveats contradicting each other.

## Placement

All logic lives in the new pure module `replicates.py` (no DB, no ledger
import -- `ledger` imports it, and so does `generic.py` for the one seed
regex, so the adapter and the grouping cannot disagree about what a seed
token is). `ledger._compare` gains one call and an assignment; `_family_rows`
selects `config_json`; `_caveats` delegates its last line. This keeps
`ledger.py` under its complexity ratchet and makes the rule testable with
literal dicts.

## Testing (acceptance)

1. `fam_a_seed1..3` + `fam_b_seed1..3` scan to two arms with `n_runs == 3`.
2. `bootstrap_ci` pinned to exact floats for a fixed input, equal across calls.
3. Seed-only name difference with different non-seed config: not merged, and
   a caveat names the differing key.
4. `examples/workspace` `kdsweep` output unchanged (golden path).
5. Overlapping intervals -> `separable is False`, caveat names both intervals;
   disjoint -> `True`; `n_runs == 2` -> no interval, `separable is None`.
6. Mutation checks (measurement-lessons section 4): removing the `group` call
   fails 1, 3 and 5; making the single-run caveat unconditional fails a test.

The real sweep above is re-run after the change and its output recorded in
the commit message, since a fixture once overstated what the ledger found
(3 real comparisons in 1045 runs).

## Out of scope

Downgrading claims on n=1 or overlapping intervals (needs a comparative-claim
grammar), `attest runs record --seed`, any migration, per-sample statistics,
p-values or multiple-comparison correction, regrouping subdirectory layouts,
a new golden-path example, SKILL.md text.
