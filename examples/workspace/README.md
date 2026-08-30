<!-- checked by tests/test_golden_paths.py -->

# Example workspace

## What you get

A self-contained research workspace for trying `attest` without a real
project: two projects, nine runs, and one paper with seven claims — three of
them deliberately wrong — so the run ledger and the claim checker have
something real to disagree with.

## Prerequisites

`none — pure local computation`

## Run it

```bash
export RESEARCH_ROOT=$PWD
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare kdsweep --metric wer
uv run attest claims speech-distill/FINDINGS.md || true
uv run attest claims speech-distill/FINDINGS.md --coverage
```

Commands are relative to this directory (`run.sh` does `cd
"$(dirname "$0")"` first). The repo-root README's "Try it in 60 seconds"
keeps its own `--root examples/workspace` form, run from the repo root; both
point at the same fixture.

## What it prints

```
winner: kdsweep_t4
  caveat: the top two arms differ by 0.0017 (2.6%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
```

Abridged — `runs scan` reports 9 run(s) across 2 project(s); `runs list`
prints all nine grouped by family; `claims` prints one verdict per claim,
ending `7 claim(s): 1 contradicted, 5 supported, 1 unsupported` plus one
malformed claim; `--coverage` lists four uncovered numbers.

## What it demonstrates

**`speech-distill/`** — a four-arm distillation sweep. `runs compare kdsweep`
ranks it and then says why not to trust the ranking: the top two arms differ by
2.6%, and every arm is a single run, so the ordering cannot separate the
configuration from run-to-run variance. `kdsweep_t4b` is `kdsweep_t4` at a
different seed, and the 0.0017 gap between them is larger than the gap between
two of the arms being ranked.

**`retrieval-ablation/`** — arms that did **not** all see the same data.
`rank_method_dense2` trained on `beir-nfcorpus` while the others used
`msmarco-dev`. `runs compare rank-method` ranks it — `ndcg` has a declared
direction — and then says the winner saw a different corpus from the rest, so
the number may not be comparable. To see the other rule, ask for a metric with
no declared direction: `runs compare rank-method --metric n_records` refuses
to rank rather than guessing which way is better. Both are the ledger's rules
holding: never rank an undeclared metric, never silently compare across
corpora.

**`planned_colbert.yaml`** — a config with no result. It is recorded as a
specification with no metrics rather than being given an invented number.

**`FINDINGS.md`** — seven claims. Five re-derive from the artifacts, one is
stale (`contradicted`), one names a run that does not exist (`unsupported`),
and one is malformed. `--coverage` lists numbers in the prose that no claim
covers at all.

### Why the file names look the way they do

The ledger reads conventions, not configuration. Family grouping comes from the
filename stem — a shared prefix plus a trailing variant token, so
`kdsweep_t2` / `kdsweep_t4` group as `kdsweep`. An earlier draft of this
example named them `distill_temp_t4`, and `temp` is in the adapter's list of
variant markers (alongside `cfg`, `seed`, `lr`, `fold`), so it read the arm as
split `tempt4`. That is the convention working as designed; the names just have
to avoid colliding with it.

## When it goes wrong

- `attest claims speech-distill/FINDINGS.md` exits 1 because one claim is
  deliberately `contradicted` — that is the point of the fixture, not a
  failure of the tool. `run.sh` is `set -e`, so that line carries `|| true`
  to let the rest of the script run; the coverage command right after it
  still exits 0.
- Running from the wrong directory (not `cd`-ed into `examples/workspace`)
  makes `--root .` scan an empty or unrelated tree; `runs scan` reports
  `0 run(s)` rather than raising.
- An `ATTEST_DB` left over from a previous run accumulates duplicate runs on
  re-scan; `run.sh` always points `ATTEST_DB` at a fresh temp file.

## Next

See the catalogue at `examples/README.md` for the other golden paths.
