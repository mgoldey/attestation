# Example workspace

A self-contained research workspace for trying `attest` without a real project.
Two projects, nine runs, one paper with claims — three of them deliberately
wrong.

```bash
export RESEARCH_ROOT=$PWD/examples/workspace
export RSS_DB=/tmp/attest-example.db

uv run attest runs scan --root examples/workspace
uv run attest runs list
uv run attest runs compare kdsweep --metric wer
uv run attest claims examples/workspace/speech-distill/FINDINGS.md
uv run attest claims examples/workspace/speech-distill/FINDINGS.md --coverage
```

## What each part demonstrates

**`speech-distill/`** — a four-arm distillation sweep. `runs compare kdsweep`
ranks it and then says why not to trust the ranking: the top two arms differ by
2.6%, and every arm is a single run, so the ordering cannot separate the
configuration from run-to-run variance. `kdsweep_t4b` is `kdsweep_t4` at a
different seed, and the 0.0017 gap between them is larger than the gap between
two of the arms being ranked.

**`retrieval-ablation/`** — arms that did **not** all see the same data.
`rank_method_dense2` trained on `beir-nfcorpus` while the others used
`msmarco-dev`. `ndcg_at_10` also has no declared direction, so
`runs compare rank-method` refuses to rank rather than guessing which way is
better. Both are the ledger's rules holding: never rank an undeclared metric,
never silently compare across corpora.

**`planned_colbert.yaml`** — a config with no result. It is recorded as a
specification with no metrics rather than being given an invented number.

**`FINDINGS.md`** — seven claims. Five re-derive from the artifacts, one is
stale (`contradicted`), one names a run that does not exist (`unsupported`),
and one is malformed. `--coverage` lists numbers in the prose that no claim
covers at all.

## Why the file names look the way they do

The ledger reads conventions, not configuration. Family grouping comes from the
filename stem — a shared prefix plus a trailing variant token, so
`kdsweep_t2` / `kdsweep_t4` group as `kdsweep`. An earlier draft of this
example named them `distill_temp_t4`, and `temp` is in the adapter's list of
variant markers (alongside `cfg`, `seed`, `lr`, `fold`), so it read the arm as
split `tempt4`. That is the convention working as designed; the names just have
to avoid colliding with it.
