# Hypothesis preregistration

**Date:** 2026-11-19
**Status:** built; feature B of the 2026-11-19 triage. No MCP tool, no migration, no dependency.
**Depends on:** run records (`2026-10-17-run-record-design.md`) for the key,
the envelope and the verifier; the ledger (`2026-08-12-run-ledger-design.md`)
for `compare()` and declared metric directions; replicate statistics
(`2026-11-19-replicate-statistics-design.md`, built in parallel) for the
optional `n_runs`/`mean`/`ci_low`/`ci_high`/`runs` keys on an arm.

## What this is for

A reviewer can re-run a comparison, re-hash an artifact and re-check a claim.
The one thing they cannot reconstruct afterwards is what was predicted before
the data existed. A sweep that "confirms" a hypothesis written after its results
were read is the garden of forking paths with a signature on it.

`attest prereg add` writes the prediction down -- family, metric, a predicted
ordering of named arms and a minimum margin -- signs it with the install's
run-record key, and `attest prereg check` later evaluates it against the
ledger: `confirmed`, `refuted`, `inconclusive` or `untested`, with `post_hoc`
when the registration looks later than the evidence and `altered` when the
text no longer matches what was signed.

## The trust model (the point of the feature)

A local tool cannot prove *when* something was written. Saying "registered
before the run" on the strength of this machine's clock would be exactly the
overclaim `run_record.verify.CANNOT_CHECK` exists to prevent. So every output
of this feature -- `attest prereg add`, `attest prereg check`, `attest prereg
list` and the `hypotheses` block of `runs.compare` -- carries the same three
statements (`prereg.LIMITS`), pinned by tests in both surfaces:

**Proved** (cryptographically, offline): the key named in the record signed
exactly this hypothesis text, and the text has not changed since. The record is
a DSSE envelope over a JCS in-toto Statement (the run-record format, unchanged);
the hypothesis id is the sha256 of the hypothesis's canonical JSON, and that
same digest is the record's subject. Editing one character of the hypothesis --
in `hypotheses.toml`, in the canonical file, or in the record -- breaks a hash
or the signature and reads as `altered`. Never a silent re-hash: the id is
stored, recomputed and compared, and the verdict is computed from the *signed*
text, so an edited copy cannot change the verdict it is reported beside.

**Evidence only** (reported, never treated as proof): `registered_at` is this
machine's clock at signing. `post_hoc` compares it with the modification time of
each named arm's artifacts on this disk -- for a pooled arm the EARLIEST of
every replicate's file (`source_paths`), so re-running one seed cannot hide
that the others predate the registration. `registered_at` is stored to the
second, so mtimes are truncated to the second and an artifact from the
registration's own second counts as predating it (`post_hoc` errs toward
flagging). A seed named instead of its arm is matched by the seed rule, not
by the capped `runs` list, so a sixth seed is inconclusive too. Both
clocks belong to whoever controls the machine; an mtime is the *last* write, so
an artifact rewritten after registration hides that it existed before (a
missed `post_hoc`), and a backdated clock hides a late registration. The
workspace's git HEAD and dirtiness at registration are recorded in the signed
data; a commit date is the committer's clock too, and becomes third-party
evidence only once pushed somewhere the researcher does not control.

**Cannot check**: that the prediction was written before anyone saw the data;
that no registration was deleted (removing a record file and its entry leaves no
local trace -- commit `prereg/` to a shared repository so deletion shows up in
history); and who holds the key, unless the reader pins it (`attest record
verify --keyid`). Trusted time (RFC 3161, a transparency log) is the real fix and
needs the network, so it is out of scope, as in the run-record spec's open
decision 2.

One deliberate asymmetry: `post_hoc` set is evidence the registration was late;
`post_hoc` unset is **not** evidence it was early. The output says so.

## Storage: files in the workspace

Like `corpora.toml` and `metric_direction.toml`, registrations are files a
researcher commits beside the work, not database rows (no migration):

| File | Holds |
|---|---|
| `<root>/hypotheses.toml` | the human-diffable index: one `[[hypothesis]]` table per registration (`id`, `registered_at`, `family`, `metric`, `expect`, `by`, `allow_single_run`, `note`, `supersedes`). Appended to, never rewritten, by the tool. |
| `<root>/prereg/<id16>.hypothesis.json` | the hypothesis's canonical JSON (RFC 8785 JCS bytes). Its sha256 *is* the id. |
| `<root>/prereg/<id16>.json` | the signed run record: subject = the file above, one `prereg` check whose single finding carries the hypothesis, `registered_at` and the workspace git state. |

`<root>` is `--root`, else `RESEARCH_ROOT`, else (CLI only) the current
directory. `runs.compare` reads `RESEARCH_ROOT` only; with it unset it attaches
nothing.

The signed record is the source of truth and `hypotheses.toml` its mirror.
`check` reads both: a record with no entry is reported (`altered`: removed from
the index), so deleting a refuted prediction from the TOML does not make it
disappear while its record exists.

New files only. Registering identical content twice is refused (same id). An
edit is a new registration with `--supersedes <id>`; the old one stays, is
reported with the `superseded` flag, and is still evaluated -- replacing a
prediction after the evidence would otherwise be the cleanest way to launder it.

## The hypothesis

```
attest prereg add kdsweep --metric wer --expect "kdsweep_t4 < kdsweep_base" --by 0.005
```

- `--expect` is a chain of arm names joined by `<` or `>` (`a < b < c` is two
  pairs). It compares the metric's **raw values**: `a < b` means a's value is
  lower than b's. That makes the verdict independent of the declared direction,
  so a later edit to `metric_direction.toml` cannot flip it.
- `--by` is the minimum margin, in the metric's units, each adjacent pair must
  differ by in the predicted direction. Required; `0` is allowed and explicit.
- No direction inference: the metric must have a declared direction
  (`ledger.metric_directions()`, matched the way `compare()` matches stems) or
  `add` refuses with `ledger.unknown_direction_message`'s sentence -- the same
  one `runs.compare` and `attest runs record` print. A hypothesis about a
  metric the ledger refuses to rank could never be evaluated.
- `--allow-single-run` (default off) is the one way point values decide.

## The verdict rule

`prereg.evaluate(hypothesis, compare_result, evidence_times, registered_at=)` is
pure and DB-free: it takes the dict `ledger.compare()` returns (for the
hypothesis's family **and metric** -- a result for another metric is
`untested`, never re-read), reads replicate keys with `.get()`, and works
whether or not builder A's keys are present.

Per pair, with `lo` the arm predicted lower and `hi` the one predicted higher:

| Evidence | confirmed | refuted | otherwise |
|---|---|---|---|
| both arms carry `ci_low`/`ci_high` | `hi.ci_low - lo.ci_high >= by` | `hi.ci_high - lo.ci_low < by` | inconclusive |
| either arm has no interval, `allow_single_run` | `hi.value - lo.value >= by` | `hi.value - lo.value < by` | -- |
| either arm has no interval, default | -- | -- | inconclusive |

The interval rule is a comparison of intervals, not a significance test, and
says so. Without `allow_single_run`, point values can neither confirm nor refute
(symmetric on purpose: a single seed that happens to contradict is no more
decisive than one that happens to agree). An arm that is absent, or present
without a value for the metric, makes the pair `untested`, never `refuted`.

The claim cannot select its own evidence (the principle behind `AMBIGUOUS` in
claims.py): arms are matched by exact `name` only. A name that appears only in
another arm's `runs` list -- one seed of a replicated arm -- is `inconclusive`
with a reason, because a prediction about a hand-picked seed is the cherry-pick
this exists to prevent.

Overall: `refuted` if any pair is refuted (a decisive pair settles the chain);
else `untested` if any pair is untested; else `inconclusive` if any is; else
`confirmed`.

Flags are reported beside the verdict, never instead of it:

- `post_hoc` -- `registered_at` is later than the earliest evidence time among
  the arms the hypothesis names; the detail lists which arms predate it.
- `altered` -- the TOML text no longer hashes to its id, its `registered_at`
  differs from the signed one, the record fails verification, the record is
  missing, or a record has no TOML entry.
- `superseded` -- a later registration names this one; informational.

`attest prereg check` exits 1 on any `refuted` verdict or `altered` flag, like
`attest claims`; `post_hoc` alone does not fail it (it is evidence, not proof).

## Surfaces

- `prereg.py` (pure + small file I/O, no database, no model, no network):
  `Hypothesis`, `parse_expect`, `parse_file`, `evaluate`, `register`, `audit`,
  `evidence_times`, `attach` (the `runs.compare` block), `LIMITS`.
- `cli_prereg.py` (the `cli_bib.py` precedent: its own module, cli.py gets the
  parser hook and HELP entries only): `attest prereg add|check|list`.
- MCP: `runs.compare` gains, only when the family has registrations, a
  `hypotheses` list (id, expectation, verdict, flags, registered_at; capped at
  `MAX_HYPOTHESES_SHOWN`), `n_hypotheses`, and `prereg_limits` (the three
  statements). A family with none returns exactly today's keys. Hypotheses on a
  different metric than the comparison are listed as `untested` with the
  metric to ask for. Registration stays CLI-only: a write tool would be a 55th
  schema paid every turn, and registering should be a deliberate act.

## Out of scope

A new MCP tool; a DB migration; trusted timestamping or anything networked;
multiple-comparison correction; relative margins; per-project hypotheses (a
family is unique per project and files live in the workspace); editing
`ledger.py`, `claims.py` or any SKILL.md (a follow-up teaches
`attestation-record` to preregister and re-measures routing).

## Testing

`tests/test_prereg.py`: round trip (the record verifies with
`run_record.verify.verify_file`; one changed character of `hypotheses.toml`, of
the canonical file, or of the record reads `altered`; a removed TOML entry is
still reported); `evaluate` table-driven for all four verdicts with and without
interval keys; `post_hoc` reported beside a computed verdict; undeclared metric
refused with the ledger's sentence; a missing arm is `untested`; a seed name is
`inconclusive`; `LIMITS` present in CLI and MCP output; no network, no model, the
key from `ATTEST_RECORD_KEY` under `tmp_path`. `tests/test_ledger_mcp.py`:
`runs.compare` for a family without registrations returns the same keys as
before, and with one returns the block. Mutation checks, each run before commit:
drop the id recomputation (`altered` test fails), make point values decisive
without `allow_single_run` (inconclusive test fails), and delete one `LIMITS`
statement (both surface tests fail).
