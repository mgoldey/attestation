# Scheduled refresh: bound the work, and say what it will cost

**Date:** 2026-08-23
**Status:** proposed
**Kind:** design. Small and specific.

## Problem

`install` writes `~/.hermes/scripts/attestation-refresh.sh` and registers it
with the agent (or a crontab). The script itself is careful — `flock -n` so a
slow run skips rather than races, per-step error handling so a cold Ollama
degrades instead of failing, timestamped lines because a silent no-op and a
healthy run were once indistinguishable. Each of those carries its reason in
the source. The scheduling machinery is not the problem.

What it schedules is. The script runs:

```
attest ingest      # deterministic, no model, must succeed
attest tag         # needs Ollama, best-effort
```

`attest tag` is **unbounded**. It tags every untagged item. Measured on this
machine: **2.3s/item**, so a 1000-item backlog is a ~40-minute run and a
5000-item backlog is ~3.2 hours. With an hourly schedule and `flock -n`, a
backlog larger than the interval means every subsequent wakeup logs
`SKIP: previous run still holding lock` and the ingest half — the deterministic
part that must succeed — never runs either.

That is the failure this design exists to prevent: a slow best-effort step
starving a fast mandatory one.

## Design

### 1. Bound the tagging step to the interval

`attest tag --limit N` already exists. The refresh script uses it, with N
derived from the schedule rather than guessed:

```
tag_budget = (interval_seconds * 0.5) / measured_seconds_per_item
```

Half the interval, so a run that hits its budget still leaves the next tick a
free lock. At hourly and 2.3s/item that is ~780 items. Untagged items are
picked up next pass, which is already how the script treats tag failures.

The multiplier and the per-item cost belong in the generated script as named
values with the measurement beside them, not as a bare integer.

### 2. Tell the user what a run will cost, before it runs

`attest tag` prints nothing until it finishes. A 40-minute silent command is
the classic abandonment point, and the onboarding review named it as such.

On start, when there is work to do:

```
tagging 1043 untagged items -- about 40 min at 2.3s/item
(use --limit N to do fewer; untagged items are picked up next run)
```

The estimate is measured from the trailing window of this machine's own
tagging, not a constant: a 12B model on a slower box is a different number, and
a hardcoded 2.3 would be wrong for most users. Fall back to the constant only
when there is no history.

### 3. `runs scan` belongs in the refresh, and does not today

The ledger is the capability three reviews independently identified as the
strongest, and it is the one thing in the tool that needs **no model** and
completes in under a second on a real corpus. It is absent from the scheduled
refresh while the two model-dependent steps are both present.

Add it, gated on `RESEARCH_ROOT` being set and pointing at something that
exists — silently skipped otherwise, because most users will not have one:

```
attest runs scan   # deterministic, no model, ~1s on 1045 runs; must succeed
```

Placed FIRST, ahead of ingest, so the cheapest and most reliable step cannot be
starved by anything after it.

## What this does not do

No new scheduler, no systemd units, no second cadence. The crontab/agent
registration, the lock, and the duplicate detection all stay exactly as they
are — that machinery is sound and its failure modes are already recorded.

## Open questions

- Whether the per-item estimate should live in the database (a `tag_timings`
  table) or be derived from `item_features.created_at` deltas, which already
  exist and need no new schema. The latter, probably.
- Whether a user with no `RESEARCH_ROOT` should be prompted to set one during
  `install`, given the ledger is the strongest capability and it is currently
  opt-in by an environment variable most people will never discover.
