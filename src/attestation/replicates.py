"""Seed replicates in a comparison: group them, then say what they support.

A sweep run with several seeds per configuration is the one place the ledger
can see run-to-run variance -- and before this module it ranked those seeds
against each other as rival arms, then printed a caveat denying they existed.
Here replicates of one configuration become ONE arm carrying a mean, a
deterministic bootstrap interval and a range, and the caveats describe what
was actually found.

Pure and stdlib-only: no database, no ledger import (`ledger` imports this,
and so do the adapters, for the one definition of a seed token).
See docs/superpowers/specs/2026-11-19-replicate-statistics-design.md.

Pooling is SCHEMA-DRIVEN: two runs are replicates of one configuration when
their core records share an `arm` and differ in `seeds` -- the record's own
fields, stored by the ledger as `runs.arm`/`runs.seeds_json`. A name is read
only once, by the adapter importing a legacy artifact that has no such
fields (`ledger_adapters.trackers.replicate_of`: `<arm>_seed<N>`), never
here. The rule that matters most is asymmetric: the record PROPOSES a merge
and a CONFIG can only veto one. A wrong merge silently averages two
configurations, which is worse than leaving replicates apart with a caveat.
"""

import json
import math
import random
import re

# `seed1`, `seed_1`, `seed-1`, `seed=1`: digits only, bounded by a separator
# or the end, so `seedling_3` and `seeded` are not seeds. The leading
# separator (or `/`, for a directory-qualified name) is part of the match so
# removing it leaves a clean base name.
SEED_TOKEN = re.compile(r"(?:^|[-_/])seed[-_=]?(\d+)(?=$|[-_./\[])", re.I)
# The bootstrap is FIXED: the same values give the same interval in every
# process and on every supported Python. A comparison that printed a
# different interval on each call would be unauditable.
N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95
# Below this many runs no interval is reported: a percentile bootstrap of two
# values has three possible resample means, and calling that a 95% interval
# would print a confidence the data cannot carry.
MIN_RUNS_FOR_CI = 3
# Member names listed per replicate arm. `n_runs` is the true count; a
# twenty-seed arm must not grow a payload a model reads every turn.
MAX_RUN_NAMES = 5

SINGLE_RUN_CAVEAT = (
    "each arm is a single run; no seed replication, so this ranking cannot"
    " separate configuration from run-to-run variance"
)


def strip_seed(text: str) -> tuple[str, int] | None:
    """`(text without its seed token, seed)`, or None unless exactly one
    seed token is present. The remainder may be empty (`seed_3`); separators
    left dangling at either end are trimmed.

    Two seed tokens in one name is ambiguous (which one varies?), so it is
    treated like none rather than picking one.
    """
    matches = list(SEED_TOKEN.finditer(text))
    if len(matches) != 1:
        return None
    m = matches[0]
    rest = (text[: m.start()] + text[m.end() :]).strip("-_/")
    return rest, int(m.group(1))


def identity(arm: str | None, seeds_json) -> tuple[str, dict] | None:
    """`(arm, seeds)` for a run that can pool with siblings: its record named
    an arm AND at least one seed. A run with no arm is its own configuration;
    one with no seeds (`null`, "did not say", or `{}`, "set none") cannot be
    told apart from a re-execution, so neither pools."""
    if not arm or not seeds_json:
        return None
    try:
        seeds = json.loads(seeds_json) if isinstance(seeds_json, str) else seeds_json
    except ValueError:
        return None
    return (arm, seeds) if isinstance(seeds, dict) and seeds else None


def _seed_label(seeds: dict) -> str:
    """One run's seeds as one comparable value: `3` for the common single
    `{"seed": 3}`, canonical JSON otherwise."""
    if len(seeds) == 1:
        return json.dumps(next(iter(seeds.values())))
    return json.dumps(seeds, sort_keys=True)


def mean(values: list[float]) -> float:
    """Arithmetic mean via `math.fsum`, so the result does not depend on
    the order the runs were read in."""
    return math.fsum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    """Sample standard deviation (n - 1). 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(math.fsum((v - mu) ** 2 for v in values) / (len(values) - 1))


def bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean, deterministic.

    Indices come from `int(rng.random() * n)` rather than `choices` or
    `randrange`, so the output depends only on the Mersenne Twister stream
    for an integer seed -- which Python keeps stable across versions -- and
    not on how a sampling helper happens to be implemented. With few values
    the interval is too narrow (it never leaves [min, max]); callers say so.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        math.fsum(values[int(rng.random() * n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    alpha = (1.0 - level) / 2.0
    low = means[int(math.floor(alpha * n_resamples))]
    high = means[min(n_resamples - 1, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)]
    return low, high


def cohen_d(top: dict, second: dict) -> float | None:
    """Standardized difference between two replicate arms (pooled sample SD),
    as a magnitude: the caller orders the arms, so this is how far apart the
    winner's mean sits from the runner-up's in units of run-to-run spread.

    None when either arm is a single run (no spread to standardize by) or the
    pooled SD is zero (every run identical: the ratio is undefined, not
    infinite evidence).
    """
    n1, n2 = top.get("n_runs") or 1, second.get("n_runs") or 1
    if n1 < 2 or n2 < 2:
        return None
    pooled = math.sqrt(((n1 - 1) * top["sd"] ** 2 + (n2 - 1) * second["sd"] ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return None
    return round(abs(top["mean"] - second["mean"]) / pooled, 3)


def _config(raw) -> dict | None:
    """A run's `config_json` decoded, or None when absent or unreadable."""
    if not raw:
        return None
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _unseeded_split(split: str | None) -> str | None:
    """A split with any seed token removed: the generic adapter records
    `_split_from(stem)`, which for `kd_seed1` is `seed1` -- the seed again,
    not a split -- so every replicate would otherwise differ here."""
    if not split:
        return None
    parsed = strip_seed(split)
    if parsed is None:
        return split
    return parsed[0] or None


def _common(values: list):
    """The single value every member shares, else None."""
    return values[0] if len(set(values)) == 1 else None


def _config_conflict(
    members: list[dict], configs: dict[str, dict | None], seed_keys: set[str]
) -> list[str]:
    """Config keys, other than `seed` and the keys the records name as seeds,
    on which the members that HAVE a config disagree. A missing config is not
    disagreement."""
    present = [c for c in (configs.get(m["name"]) for m in members) if c is not None]
    rests = [_canonical_without_seed(c, seed_keys) for c in present]
    keys: set[str] = set().union(*rests)
    return sorted(k for k in keys if len({r.get(k) for r in rests}) > 1)


def _canonical_without_seed(config: dict, seed_keys: set[str]) -> dict[str, str]:
    """A config's non-seed values as canonical JSON (nested keys sorted), so
    values compare by their serialized form rather than by identity."""
    skip = {"seed"} | seed_keys
    return {k: json.dumps(v, sort_keys=True) for k, v in config.items() if k not in skip}


def _veto(
    members: list[dict],
    seeds: list[dict],
    configs: dict[str, dict | None],
    taken: str | None,
) -> str | None:
    """Why these name-matched arms must NOT pool, or None if they may.
    `taken` is their base name when another arm already holds it."""
    if taken:
        # Pooling under a name another arm already holds would print two
        # rows called `base`, one of which may itself be a replicate.
        return f"a run named {taken} with no seed token also exists"
    labels = [_seed_label(s) for s in seeds]
    if len(set(labels)) != len(labels):
        return "two of them claim the same seed"
    if len({_unseeded_split(m.get("split")) for m in members}) > 1:
        return "they report different splits"
    differing = _config_conflict(members, configs, {k for s in seeds for k in s})
    if differing:
        return f"their configs differ in {', '.join(differing)}"
    return None


def _smallest_n(members: list[dict]) -> int | None:
    """Samples per evaluation for a pooled arm: the smallest member's, the
    conservative choice; None when any member's is unknown."""
    sizes = [m.get("n") for m in members]
    known = [n for n in sizes if n is not None]
    return min(known) if known and len(known) == len(sizes) else None


def _merge(base: str, members: list[dict]) -> dict:
    """One replicate arm from name-matched members (already vetted)."""
    members = sorted(members, key=lambda m: m["name"])
    values = [float(m["value"]) for m in members]
    mu = mean(values)
    ci = bootstrap_ci(values) if len(values) >= MIN_RUNS_FOR_CI else (None, None)
    return {
        "name": base,
        "status": _common([m.get("status") for m in members]) or members[0].get("status"),
        "value": mu,
        "step": _common([m.get("step") for m in members]),
        "split": _unseeded_split(members[0].get("split")),
        "source_path": members[0].get("source_path"),
        # EVERY member's artifact, not only the first: prereg's post_hoc check
        # needs the earliest of them. Not capped like `runs`; the MCP layer
        # drops it from what a model reads.
        "source_paths": [m.get("source_path") for m in members if m.get("source_path")],
        # EVERY member's name, likewise uncapped (dropped from the MCP
        # payload): wall time sums over all of them (compute.attach), and a
        # hypothesis naming a sixth seed is told it is a replicate (prereg).
        "members": [m["name"] for m in members],
        "n": _smallest_n(members),
        "n_runs": len(values),
        "mean": mu,
        "sd": sample_sd(values),
        "min": min(values),
        "max": max(values),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "runs": [m["name"] for m in members][:MAX_RUN_NAMES],
    }


def _listed(names: list[str]) -> str:
    """Names for a caveat, capped at MAX_RUN_NAMES with the remainder counted."""
    shown = ", ".join(names[:MAX_RUN_NAMES])
    extra = len(names) - MAX_RUN_NAMES
    return f"{shown} and {extra} more" if extra > 0 else shown


def _seeded_config(arm: dict, configs: dict[str, dict | None]) -> dict | None:
    """A single-run arm's config when it names a seed AND something else --
    the only kind that can say two runs differ in nothing but the seed."""
    cfg = configs.get(arm["name"])
    if arm.get("n_runs") or not cfg or "seed" not in cfg or len(cfg) < 2:
        return None
    return cfg


def _lookalikes(arms: list[dict], configs: dict[str, dict | None]) -> list[str]:
    """Caveats for single-run arms whose configs say they are replicates.

    Tracker runs (`train/0`, `train/1`) carry no seed token in their names, so
    their records have no arm and cannot pool. Configs alone never merge -- a plain JSON
    result's config is `{"seed": ...}` only, so "equal apart from seed" would
    be vacuously true across a whole sweep -- but when a config carries real
    keys and two runs differ ONLY in seed, the reader is told.
    """
    buckets: dict[tuple[str, str | None], list[tuple[str, object]]] = {}
    for arm in arms:
        cfg = _seeded_config(arm, configs)
        if cfg is not None:
            rest = json.dumps({k: v for k, v in cfg.items() if k != "seed"}, sort_keys=True)
            buckets.setdefault((rest, arm.get("split")), []).append((arm["name"], cfg["seed"]))
    out = []
    for members in buckets.values():
        if len(members) > 1 and len({str(seed) for _, seed in members}) == len(members):
            out.append(
                f"{_listed(sorted(n for n, _ in members))} have identical configs apart"
                " from seed but are listed as separate arms; if they are replicates,"
                " name them (or their tracker run names) <config>_seed<N>, or give"
                " their run records one arm and distinct seeds, so they pool into one"
            )
    return out


def _by_base(
    arms: list[dict], identities: dict[str, tuple[str, dict]]
) -> tuple[list[dict], dict[str, list[tuple[dict, dict]]]]:
    """`(arms that cannot pool, {pooled arm name: [(arm, seeds), ...]})`. An
    arm without a value never pools: it stays listed in `without_metric`."""
    loose: list[dict] = []
    buckets: dict[str, list[tuple[dict, dict]]] = {}
    for arm in arms:
        parsed = identities.get(arm["name"]) if arm.get("value") is not None else None
        if parsed is None:
            loose.append(arm)
        else:
            buckets.setdefault(parsed[0], []).append((arm, parsed[1]))
    return loose, buckets


def _clash(base: str, members: list[dict], taken: set[str]) -> str | None:
    """`base` when a run OTHER than these members already holds that name. A
    member may itself be named like its arm (a native record whose run_id is
    the arm), which is no clash."""
    return base if base in taken - {m["name"] for m in members} else None


def group(
    arms: list[dict],
    configs: dict[str, object],
    identities: dict[str, tuple[str, dict]],
) -> tuple[list[dict], list[str]]:
    """Pool seed replicates of one configuration into one arm.

    `arms` are the per-run arm dicts `ledger._compare` builds; `configs` maps
    an arm name to its run's `config_json` (a JSON string, a dict or None);
    `identities` maps an arm name to `(pooled arm name, seeds)` from its run's
    core record (`identity`), and an arm absent from it never pools.
    Returns `(arms, caveats)`: arms with each pooled group replaced by one
    replicate arm, and a caveat for every merge refused and every config-only
    lookalike. Arms without a value never pool -- they stay listed by name in
    `without_metric`, which is the finding.

    Any veto keeps EVERY member of that group apart: choosing which subset is
    "the real configuration" would be the guess this module refuses.
    """
    decoded = {name: _config(raw) for name, raw in configs.items()}
    taken = {arm["name"] for arm in arms}
    out, buckets = _by_base(arms, identities)
    notes: list[str] = []
    for base, pairs in sorted(buckets.items()):
        members = [arm for arm, _ in pairs]
        if len(members) == 1:
            # One seed of a configuration is a single run: unchanged name,
            # no new keys, exactly what it was before this module.
            out.extend(members)
            continue
        reason = _veto(members, [seed for _, seed in pairs], decoded, _clash(base, members, taken))
        if reason is None:
            out.append(_merge(base, members))
        else:
            out.extend(members)
            notes.append(
                f"{_listed(sorted(m['name'] for m in members))} differ only by seed"
                f" (one arm, distinct seeds), but {reason}; kept as separate arms"
            )
    return out, notes + _lookalikes(out, decoded)


def intervals_decide(top: dict, second: dict) -> bool:
    """True when both arms carry a bootstrap interval, so `separable` -- not
    a relative-size rule of thumb -- says whether they are too close."""
    return top.get("ci_low") is not None and second.get("ci_low") is not None


def summary(scored: list[dict]) -> dict:
    """`separable` and `effect_size` for the top two of a ranked list, or
    `{}` when no arm is a replicate arm -- so a family of single runs keeps
    exactly the keys it always had.

    `separable` is True for disjoint intervals, False for overlapping ones,
    None when either of the top two has no interval. Not a significance test.
    """
    if not any(a.get("n_runs") for a in scored):
        return {}
    if len(scored) < 2:
        return {"separable": None, "effect_size": None}
    top, second = scored[0], scored[1]
    separable = None
    if intervals_decide(top, second):
        separable = top["ci_high"] < second["ci_low"] or second["ci_high"] < top["ci_low"]
    d = cohen_d(top, second)
    effect = {"between": [top["name"], second["name"]], "cohen_d": d} if d is not None else None
    return {"separable": separable, "effect_size": effect}


def single_run_caveats(scored: list[dict]) -> list[str]:
    """The single-run caveat, now conditional on what was found.

    Every arm a single run: the line `runs.compare` has always printed,
    verbatim (golden paths pin it). Some single beside replicated: name the
    single ones. None single: nothing -- the replication it denied exists.
    """
    singles = sorted(a["name"] for a in scored if not a.get("n_runs"))
    if len(singles) == len(scored):
        return [SINGLE_RUN_CAVEAT]
    if not singles:
        return []
    verb = "is a single run" if len(singles) == 1 else "are single runs"
    return [
        f"{_listed(singles)} {verb} beside replicated arms; ranking"
        f" {'it' if len(singles) == 1 else 'them'} cannot separate configuration"
        " from run-to-run variance"
    ]


def _interval(arm: dict) -> str:
    """`name [low, high]` for a caveat."""
    return f"{arm['name']} [{arm['ci_low']:.4g}, {arm['ci_high']:.4g}]"


def _count_caveats(replicated: list[dict]) -> list[str]:
    """Caveats about how many runs each replicate arm pooled."""
    out: list[str] = []
    counts = sorted({a["n_runs"] for a in replicated})
    if len(counts) > 1:
        out.append(
            f"replicate counts differ across arms ({counts[0]}-{counts[-1]} runs);"
            " means from fewer runs are less certain"
        )
    few = sorted(a["name"] for a in replicated if a["n_runs"] < MIN_RUNS_FOR_CI)
    if few:
        out.append(
            f"{_listed(few)} {'has' if len(few) == 1 else 'have'} fewer than"
            f" {MIN_RUNS_FOR_CI} runs: mean and range reported, no interval"
            " -- a CI from 2 runs is not a CI"
        )
    return out


def stat_caveats(scored: list[dict], found: dict) -> list[str]:
    """What the replicate statistics do and do not support, for a ranked list
    and the `summary()` computed from it."""
    replicated = [a for a in scored if a.get("n_runs")]
    if not replicated:
        return []
    out = _count_caveats(replicated)
    if found.get("separable") is False:
        out.append(
            f"the top two arms' {CI_LEVEL:.0%} intervals overlap ({_interval(scored[0])}"
            f" vs {_interval(scored[1])}); not separable from these runs"
        )
    if any(a.get("ci_low") is not None for a in replicated):
        out.append(
            f"intervals are a {CI_LEVEL:.0%} percentile bootstrap of each arm's mean"
            f" ({N_RESAMPLES} resamples, fixed seed); with fewer than 10 runs they"
            " understate the spread, and this is not a significance test"
        )
    return out
