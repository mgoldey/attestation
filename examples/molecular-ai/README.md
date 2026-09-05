<!-- checked by tests/test_golden_paths.py -->

# Example molecular-ai

## What you get

A 48-entry reference library of canonical molecular-AI papers -- interatomic
potentials from Behler-Parrinello to MACE and MatterSim, graph networks from
MPNN to GemNet, generative models, docking, the Open Catalyst datasets, and
the protein-structure line from AlphaFold 2 to RFdiffusion -- written by
`generate.py` from real APIs, not typed to look fetched. `seeds.toml` is the
only hand-typed file: identifiers and working titles. Everything else in
`references.bib` came off the wire or out of the tagger: authors, years,
abstracts and venues from the arXiv API and CrossRef, reference lists from
Semantic Scholar (as a `cites` field), and topic tags from the real tagging
prompt against gemma4:e2b (as `keywords`). The run here is offline: it syncs
the committed file into a scratch library, searches it, walks one paper's
citation neighbourhood, and reports the concept graph the references form.

## Prerequisites

`none — pure local computation`

## Run it

```bash
export ATTEST_BIB_PATHS=$PWD/references.bib
uv run attest library sync --sources bibtex
uv run attest library search "force field" --limit 5
uv run attest library related batzner2021equivariant
uv run attest kg-report
```

Relative to this directory (`run.sh` does `cd "$(dirname "$0")"` first and
points `ATTEST_DB` at a fresh temp file).

## What it prints

```
bibtex: +48 added, 0 merged, 0 unchanged
embedded 0, 48 without a vector
```

No model server in the test run, so nothing is embedded and the search that
follows is a substring search -- and its last line says so:

```
2024  MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures  [yang2024mattersim]
2023  CHGNet: Pretrained universal neural network potential for charge-informed atomistic modeli  [deng2023chgnet]
2022  MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force  [batatia2022mace]
2022  TorchMD-NET: Equivariant Transformers for Neural Network based Molecular Potentials  [tholke2022torchmd]
2016  ANI-1: An extensible neural network potential with DFT accuracy at force field computation  [smith2016ani]
5 match(es); substring search only (no embedder); run `attest library embed` for semantic
```

Then NequIP's citation neighbourhood, both directions, in-library rows first:

```
2021  E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potenti  [batzner2021equivariant]
cites 67
  [in library] smith2016ani  ANI-1: An extensible neural network potential with DFT accuracy at force field c
  [in library] zhang2017deep  Deep Potential Molecular Dynamics: a scalable model with the accuracy of quantum
[...]
  [not in library] arxiv:1412.6980  Adam: A Method for Stochastic Optimization
[...]
cited_by 9
  [in library] deng2023chgnet  CHGNet: Pretrained universal neural network potential for charge-informed atomis
[...]
```

and the graph the 48 references form on their own -- `n_items 0`, because
nothing was ingested into this scratch database, and `n_references 48`:

```
nodes                    14
edges                    25
n_items                  0
n_references             48
```

## What it demonstrates

**The library is generated, not authored.** `generate.py` writes a seed
`.bib` from `seeds.toml`, syncs it into a scratch library with
`ATTEST_CITATION_WEB=1 ATTEST_CITATION_SCHOLAR=1` so the arXiv, CrossRef and
Semantic Scholar enrichers fill fields on rows that already exist (they never
add a paper), tags every row with `features.run_reference_tagging` -- the ONE
tagging renderer -- and writes the result with bibtexparser v2. What the
committed file lacks is what the wire did not have: 2 of 48 entries carry no
abstract, and 7 of 48 have no `cites` because Semantic Scholar's shared
unauthenticated rate limit (measured: 429 on three consecutive requests 1.2 s
apart) outlasted three resumed passes. A rate-limited row is left untouched
for the next `sync`, never recorded as a miss; the fixture reports the gap
rather than filling it by hand.

**Identity holds across sources.** 40 of the 48 seeds were arXiv ids and 8
were DOIs; Semantic Scholar attached DOIs to most of the arXiv ones and
CrossRef then enriched those, so a single row carries both ids and the
`cites` field can name a paper by either. `related` resolves an edge recorded
as `arxiv:1706.08566` to the SchNet row even though that row's identity is
now its DOI.

**Substring search fails where semantic search works, and the output says
which one ran.** Measured 2026-09-05 on this library with embeddinggemma,
top three each way:

| query | semantic | substring |
|---|---|---|
| equivariant force fields | MACE 0.550, TorchMD-NET 0.542, NequIP 0.540 | nothing |
| message passing on molecular graphs | SphereNet, DimeNet, MPNN | nothing |
| protein structure prediction | AlphaFold 2, ESM-2, ProteinMPNN | ProteinMPNN, AlphaFold 2 |
| generative models for molecules | MolGAN, GraphAF, the chemical VAE | nothing |
| catalyst adsorption energies | OC22, OC20 | nothing |
| universal interatomic potentials | M3GNet, GAP, MTP | Matbench Discovery |
| docking | DiffDock, EquiBind | DiffDock, EquiBind |
| materials stability | M3GNet, Matbench Discovery, MatterSim | nothing |
| SMILES language models | SMILES Transformer, ESM-2, JT-VAE | nothing |
| atomic environment descriptors | SOAP | nothing |

Substring returned nothing for 7 of the 10 queries; semantic search put the
paper a chemist would name first or second in 9 of 10 (the exception is
"materials stability", where the framework paper ranks second to M3GNet).
The test run above has no embedder, which is why its search line ends in
`substring search only` -- the caveat is the contract, and `cite.search`
over MCP carries the same `semantic` flag.

**References join the concept graph through `keywords`.** `kg.tag_assignments`
reads `reference_tags` beside `item_tags`, with references as negative ids so
`build_graph` needs no change. On this scratch database the graph is made of
references alone: 44 distinct tags, 14 that clear the frequency floor, one
community labelled `molecular-mechanics`. With the references removed the
same `kg-report` exits 1 with "no items yet" -- the graph is empty. Tags came
from the tagger at generation time; `attest library tag` re-tags a library
whose `.bib` carries none.

**Citation edges stay out of concept adjacency.** `cites` becomes
`reference_cites`, read only by `related` and `cite.related`; the concept
graph is co-occurrence of tags and nothing else, which the knowledge-graph
spec chose on purpose. Of NequIP's 67 references, 11 are in this library and
56 are stubs with a title and an identity, never fetched.

## When it goes wrong

- `related` for a key that is not in the file exits 1 with the count of
  references in the store; keys are the generated `<surname><year><word>`
  form, so `batzner2021equivariant`, not `nequip`.
- Regenerating needs both network flags and the model server, and Semantic
  Scholar's unauthenticated pool is shared: expect 429s. Keep the scratch
  with `--scratch DIR --steps sync` and repeat until the `failed` count in
  the `s2` bucket stops falling, then `--steps tag,write`. Without
  `--scratch` the scratch is deleted at exit and every 429 costs a full pass.
- An `ATTEST_BIB_PATHS` left unset makes `sync` glob `*.bib` in the working
  directory, which here is the same file; from any other directory it finds
  nothing and the sync reports `bibtex` absent.

## Next

Regenerate from the seeds (review the diff like any other fixture change):

```bash
ATTEST_CITATION_WEB=1 ATTEST_CITATION_SCHOLAR=1 \
  uv run --with "bibtexparser>=2.0.0b9" python generate.py --scratch /tmp/molgen --steps sync
uv run --with "bibtexparser>=2.0.0b9" python generate.py --scratch /tmp/molgen --steps tag,write
```

With the model server up, `uv run attest library embed` after the sync turns
the search semantic, and `uv run attest library tag` re-tags with your own
vocabulary. See the catalogue at `examples/README.md` for the other golden
paths, and `docs/superpowers/specs/2026-09-05-library-graph-and-molecular-ai-design.md`
for the design.
