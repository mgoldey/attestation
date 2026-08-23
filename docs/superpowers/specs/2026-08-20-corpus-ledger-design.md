# Corpus ledger — design

**Date:** 2026-08-20
**Status:** implemented 2026-08-21 in `56ddba9`. The corpus is detected from driver-script syntax via AST, not from a model.

## Problem

The ledger compares experiment arms and refuses to rank a metric whose
direction is undeclared. It has no such scruple about the *data*. Every
`compare` silently assumes the arms saw the same corpus, and nothing checks it.

That assumption is invisible in the artifacts. In `~/qc/scmoe`, every LM arm
records its model configuration exhaustively:

```json
{"tag": "ortho_0.1", "seed": 42, "n_params": 88815104, "n_layers": 6,
 "d_model": 512, "num_experts": 8, "epochs": 10, "best_val_loss": 5.0946}
```

and its data not at all. The corpus lives only in a call in the driver script:

```python
data = load_wikitext2(seq_len=256, batch_size=8)   # experiments/lm_mp2.py:51
```

which resolves to `Salesforce/wikitext`, config `wikitext-2-raw-v1`, tokenized
with `tiktoken` `gpt2` (`n_vocab` 50257) at `seq_len=256`. **Zero of the 34
result files record `vocab_size` or `seq_len`.** A `grep` confirms it.

So the ledger will happily print:

```
lm-mp2 — ranked by best_val_loss (lower_is_better)
  lm_mp2_baseline    5.0967
  lm_mp2_lr_trust    5.1306
```

with no way to know whether those two numbers describe the same task. Validation
loss is only comparable across runs that share a corpus *and* a tokenizer *and*
a sequence length — change the tokenizer and perplexity moves for reasons that
have nothing to do with the model. This is the same class of error the metric
direction table exists to prevent, one level down: a confident ranking of
things that were never comparable.

The failure is quiet in the worst way. Nothing errors. The table renders, the
caveats about seed replication and effect size appear and are individually
correct, and the reader concludes something false.

## Prior art

The reasonable objection to all of this is "use Sacred, or MLflow, or DVC."
Worth answering concretely, because the run-tracking half of this ledger *is*
largely a worse MLflow, and only two properties justify its existence.

**Sacred is write-side instrumentation and structurally cannot do this.**
Verified against source at `86865b0`, not documentation: the observer
interface is entirely `*_event` sinks (`started_event`, `resource_event`,
`log_metrics`) with no read path, and recording is hard-gated on
`assert self.current_run is not None`. The word `dataset` appears zero times
in `sacred/`; so do `higher_is_better`, `rank(`, and `caveat`. Pointed at
`~/qc/scmoe` -- 34 JSON files from code that never imported it -- Sacred has
nothing to say. It is maintained but janitorial (last release Nov 2024).

The same holds, with variations, across the field:

| Tool | Reads uninstrumented artifacts? | Corpus as an entity? | Guards comparisons? |
|---|---|---|---|
| Sacred | no -- push-only observers | no | no |
| MLflow / W&B | no (`wandb sync` is TFEvents-only) | data hashes, not linked to comparison | no |
| DVC | must invoke the code itself | yes, versioned properly | no |
| HF `datasets` | n/a -- a data library | yes, xxh64 fingerprint | no runs at all (metrics removed in v3.0.0) |
| Guild AI | own `guild export` archives only | no | `compare` is a `--min`/`--max` sort |
| HiPlot | archived | no | visualization only |

The gap is not that these tools are abandoned -- most are actively maintained.
It is architectural: **data identity and run comparison live in different
tools everywhere in this ecosystem**, and nothing joins them. HF divesting
metrics in v3.0.0 is that split made explicit.

So the two properties worth building for:

1. **Retroactive reading.** Zero adoption cost is the whole design constraint
   (see the module docstring). Every tool above requires the run to have been
   instrumented before it happened, which is useless for work already done.
2. **The honesty layer.** Declared metric direction, refusal to rank an
   undeclared metric, and caveats about effect size and seed replication have
   no counterpart in the ~30 tools surveyed. Direction exists only in W&B and
   is being deprecated there.

Two honest qualifications. First, "no tool does this" is a survey result, and
absence of evidence is weaker than evidence of absence -- Sumatra in
particular was not conclusively checked. Second, the retroactive approach has
a real cost the instrumented tools do not pay: **a scanner infers structure
where a tracker is told it.** Family grouping is a filename heuristic, and a
wrong grouping produces a confident comparison of unrelated runs. That is why
the grouping rule is stated in the caveat text rather than trusted silently,
and it is the same reason this document exists: the corpus guard closes one
more place where the ledger was inferring agreement it had not checked.

## What this is not

Not a data versioning system. It does not store corpora, deduplicate them,
or reconstruct them. DVC and `datasets` already do that, and a tool that asks
the user to move their data somewhere will not be adopted.

Not a tokenizer. It records what the corpus *was* as reported or measured; it
does not encode text to find out.

## Decisions

### A corpus is a first-class entity, not a run attribute

The obvious cheap design is a `corpus` column on `runs`. Rejected: a corpus has
its own attributes (source, tokenizer, split sizes, fingerprint) that would
either be duplicated across every run that used it or lost. It also has its own
lifetime — a corpus can change on disk while the runs that cite it do not — and
an attribute has nowhere to record that. Twelve runs sharing WikiText-2 should
point at one row that can be inspected once.

```
corpora(id, name, source, config, tokenizer, vocab_size, seq_len,
        fingerprint, fingerprint_kind, measured_at, source_path, notes)
corpus_splits(corpus_id, split, n_tokens, n_records, n_bytes, fingerprint)
runs.corpus_id  ->  corpora(id)   NULL when unknown
```

`corpus_splits` is long-format for the same reason `run_metrics` is: projects
name their splits differently (`val`/`valid`/`validation`/`dev`) and carry
different counts, and a wide table would need a migration per project.

Every field except `name` is nullable. **A partially-known corpus is the normal
case** and must be representable — recording "WikiText-2, tokenizer unknown" is
strictly more honest than recording nothing, provided the unknowns are visible
as unknowns rather than rendered as blanks the reader fills in themselves.

### `runs.corpus_id` is NULL by default and that is a first-class state

Most existing artifacts say nothing about data. `corpus_id IS NULL` means "the
artifact did not say", never "no corpus" and never "the default corpus". The
distinction drives the comparison guard below: unknown must not be treated as
agreement, which is exactly the mistake the current code makes.

### Three sources, in a fixed precedence

Each is independently useful and they disagree in practice, so precedence must
be declared rather than incidental. Highest wins:

1. **Manifest** (`corpora.toml`) — an explicit user declaration.
2. **Measurement** (`corpus scan --data DIR`) — what is actually on disk now.
3. **Artifacts** — corpus fields already present in result/config JSON.

Rationale: a human declaration is the only source that can state intent
(*"these arms were meant to share a corpus"*). Measurement outranks artifacts
because an artifact records what a run *believed* it read, while a measurement
records what is *there* — and when those differ, that difference is the finding.
Artifacts are the floor: free, zero adoption cost, and available today.

Provenance is recorded per corpus (`source_path`), so a reader can always see
which of the three produced a given row. A merge never silently overwrites a
declared value with a measured one; it fills gaps and reports conflicts.

### Detection reads syntax, not a model — measured, not assumed

The corpus is usually absent from the results and present in the source, so
"ask the local LLM to read the driver" is the obvious fourth source. It was
measured against `gemma4:e2b-it-q4_K_M` before being rejected.

Asked three times for the corpus in the *same* file, it returned three
identities:

| | run 0 | run 1 | run 2 | `ast` |
|---|---|---|---|---|
| source | "WikiText-2 data loading and tokenization" | "WikiText-2" | "WikiText-2" | `Salesforce/wikitext` |
| config | `wikitext-2-raw-v1` | *the module docstring* | "WikiText-2" | `wikitext-2-raw-v1` |
| tokenizer | `tiktoken.get_encoding("gpt2")` | "tiktoken (using gpt2 encoding)" | "tiktoken (gpt2 encoding)" | `gpt2` |

Only `seq_len` was stable. Worse, on a driver file that merely *calls* a loader
and states no corpus, it invented `tokenizer: "load_wikitext2"` (a function
name) and `tokenizer: "wikitext2"` (a dataset name) rather than declining.

Both failures are disqualifying *specifically here*, because a corpus name is
a **join key**: two runs share a corpus only when their identity strings match
exactly. A detector that says "WikiText-2" once and "WikiText-2 data loading
and tokenization" the next time reports two corpora where there is one, and
the guard silently fails. A detector that fabricates a tokenizer reports
agreement that was never checked — the exact bug this feature exists to close.
An unreliable guard is worse than no guard, because it is believed.

Meanwhile `ast` gets it exactly right in ~12 lines, offline, in milliseconds,
because these are *literal arguments* and require no inference at all:

```python
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")   # source, config
tiktoken.get_encoding("gpt2")                              # tokenizer
```

`corpus.detect_in_source` deliberately does not resolve variables:
`load_dataset(name)` states nothing about which corpus was used, and reporting
the variable's name would be the same fabrication in cheaper clothing.

This is the same rule the rest of the repo already follows — "No LLM in
composition tools: digest/runs_compare return structure, never prose" — and
the reason generalises. The caller is a model, and a model is good at reading
a weird loader and *telling the user* what it thinks. It is not good at
minting a stable identifier that a join depends on.

### Manifest format, mirroring `metric_direction.toml`

The repo already has a precedent for teaching the ledger a fact it cannot
derive, and it should not grow a second convention:

```toml
[corpus.wikitext2]
source = "Salesforce/wikitext"
config = "wikitext-2-raw-v1"
tokenizer = "gpt2"
vocab_size = 50257
seq_len = 256

[corpus.wikitext2.splits]
train = { n_tokens = 2_391_884 }
val   = { n_tokens = 247_289 }

# Which runs saw it. Family-level, because that is the unit of comparison.
[assign]
family.lm-mp2 = "wikitext2"
family.lm-mp2-ortho = "wikitext2"
run.stage1_hard_dense = "synthetic-hard"
```

Resolved via `LEDGER_CORPUS_FILE`, then `<workspace>/corpora.toml`, then
`~/.hermes/corpora.toml` — the same ladder as metric direction.

### Fingerprint: content-addressed, and honest about cost

`fingerprint` is `sha256` of the corpus content, with `fingerprint_kind`
naming what was hashed (`file_sha256`, `dir_sha256`, `declared`) because
hashing a directory of shards and hashing one `.txt` are not the same claim and
must not compare equal by accident.

Large corpora make full hashing untenable, so `corpus scan` records
`fingerprint_kind = "size_mtime"` by default and computes a real digest only
under `--hash`. A weak fingerprint must never be presented as a strong one:
`corpus verify` reports what kind of check it performed. A tool that says
"verified" after comparing mtimes has done real harm.

### `compare` gains a corpus guard, as a caveat not a refusal

Consistent with `_caveats()`: the tool reports what it knows and lets the
reader judge. Three cases, and the distinction between them is the whole point:

- **All arms share a corpus** — one line naming it, so the reader sees the
  comparison was checked rather than assumed.
- **Arms differ** — a caveat naming which arm saw what. For a loss/perplexity
  metric this is close to fatal and the caveat says so.
- **Any arm unknown** — a caveat saying the comparison is unverified. This is
  the current behaviour of *every* comparison; the change is that it becomes
  visible instead of implicit.

Refusal was considered and rejected: it would make the tool useless on the
corpus it was built for, where nearly every run is unknown. The honest move is
to say "unverified", not to withhold the numbers.

## Surface

```
attest corpus scan --data DIR [--hash]   measure corpora on disk
attest corpus list                       corpora with run counts
attest corpus show NAME                  one corpus in full
attest corpus verify [NAME]              recorded vs on-disk fingerprint
```

Plus MCP `corpus_list` / `corpus_detail` / `corpus_verify`, each paired with a
FastMCP-free `_impl` per the existing convention, and `runs scan` learning to
read corpus fields and apply the manifest.

## Migration

`_MIGRATIONS` entry 2: create `corpora` and `corpus_splits`, add
`runs.corpus_id`. Additive and idempotent — existing rows get NULL, which is
the correct value for "the artifact did not say".

## Risks

**Fabricated precision.** The real danger is a corpus row that looks
authoritative because it has a name, while its tokenizer and token counts were
never known. Mitigation: unknown fields render as `?` in every view, never as
blank or zero, and `corpus show` states which source supplied the row.

**Manifest drift.** A declared corpus can stop matching disk. That is what
`corpus verify` is for, and why measurement outranks artifacts.
