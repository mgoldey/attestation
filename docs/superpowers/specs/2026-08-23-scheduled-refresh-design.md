# Scheduled refresh: bound the work, and say what it will cost

**Date:** 2026-08-23
**Status:** accepted
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

## Resolved questions

- **Per-item estimate source.** Derived from `item_features.tagged_at` deltas
  (not `created_at` -- that column does not exist; `tagged_at TEXT NOT NULL
  DEFAULT (datetime('now'))` is the real one, `db.py`). No new schema, no
  `tag_timings` table. `features.estimate_seconds_per_item` reads a trailing
  window of the most recent `tagged_at` rows, takes consecutive deltas, and
  excludes any delta more than 10x the window's median as a boundary between
  two separate runs rather than a slow item (a gap between runs is idle time,
  not a per-item cost, and averaging it in would swamp every genuine ~2s
  delta with an hour-long outlier). A window of fewer than two timestamps, or
  one where every delta is excluded as an outlier, falls back to the named
  constant `FALLBACK_SECONDS_PER_ITEM = 2.3`.
- **Prompting for `RESEARCH_ROOT` during install.** Declined. `install`
  already had a step (`mcp_wiring`) report BROKEN over agent wiring a
  self-hoster never asked for, fixed by making an unconfigured optional
  capability a silent skip rather than a nag (commit ddd560b). Adding an
  install-time prompt for another optional environment variable repeats the
  exact mistake that fix exists to prevent. `runs scan` stays opt-in,
  discoverable via `attest runs scan` and the CLI's own `set RESEARCH_ROOT`
  message when a user reaches for it unset, not solicited at install time.
