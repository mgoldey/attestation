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
uv run attest library related batzner2022equivariant
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
2017  ANI-1: An extensible neural network potential with DFT accuracy at force field computation  [smith2017ani]
5 match(es); substring search only (no embedder); run `attest library embed` for semantic
```

Then NequIP's citation neighbourhood, both directions, in-library rows first:

```
2022  E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potenti  [batzner2022equivariant]
cites 66
  [in library] smith2017ani  ANI-1: An extensible neural network potential with DFT accuracy at force field c
  [in library] zhang2018deep  Deep Potential Molecular Dynamics: a scalable model with the accuracy of quantum
[...]
  [not in library] arxiv:1412.6980  Adam: A Method for Stochastic Optimization
[...]
cited_by 10
  [in library] deng2023chgnet  CHGNet: Pretrained universal neural network potential for charge-informed atomis
[...]
```

and the graph the 48 references form on their own -- `n_items 0`, because
nothing was ingested into this scratch database, and `n_references 48`:

```
nodes                    18
edges                    22
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
committed file lacks is what the wire did not have: 3 of 48 entries carry no
abstract (two that CrossRef has none for, and RoseTTAFold, whose CrossRef
"abstract" is the Science editor's summary signed "—VV", which the reader
now drops rather than embed), and 4 of 48 have no `cites` because their
Semantic Scholar record lists no reference with an id or a title of four or
more words. Semantic Scholar's shared unauthenticated rate limit (measured:
429 on three consecutive requests 1.2 s apart) needed five resumed passes to
answer all 48. A rate-limited row is left untouched for the next `sync`,
never recorded as a miss; a 404 is recorded, so it is not asked again; a 200
that does not parse is forgotten, not cached; the fixture reports the gap
rather than filling it by hand. Entries with no known venue are `@misc`,
and `&` is escaped, so the file compiles under `bibtex` without a warning.

**A journal and its year come from the same source.** Eleven entries of the
first generation read `Nature Communications 2021` for a paper published in
2022, because the arXiv seed's year was kept when CrossRef's venue arrived.
The store now lets the venue bring its year and records the old one as a
conflict (`attest library` shows it), so `smith2017ani` is Chemical Science
2017 and Batzner is 2022 -- the first thing a chemist checks in a `.bib`.

**Identity holds across sources.** 37 of the 48 seeds were arXiv ids and 11
were DOIs. Semantic Scholar offered a DOI to 17 arXiv rows, three of them
DataCite `10.48550/arxiv.*` forms (MACE, DiffDock, CHGNet) that name the
preprint, not a publication -- the store reads those as the arXiv id they
carry, so those rows stay arXiv-identified rather than splitting from a
later journal DOI -- and an arXiv id to three DOI seeds (GAP, SOAP, MTP).
That is how the file comes to 25 DOIs, 40 arXiv ids and 17 entries with
both. Every edge that names an in-library paper by id does so by the
identity Semantic Scholar chose, so the cross-form resolution `related`
performs -- an `arxiv:` edge landing on a row whose identity is now a DOI --
is exercised by `tests/test_library.py`, not by this data; what this data
does exercise is the title form: MACE, GemNet and Allegro cite SphereNet by
title only, and `related` lands those on the SphereNet row through its
stored normalised title.

**Semantic search ranks; substring search can only find, and the output
says which one ran.** Measured 2026-09-05 on this library with
embeddinggemma, top three each way. `expected` is the paper the author of
this example would name first for that query, written down before reading
the results so the score is falsifiable; the substring column is the
word-AND fallback `cite.search` runs with no embedder (every query word
somewhere in title, abstract, authors or key, newest first), and the first
version of this table measured a whole-phrase substring instead, which
found nothing for 7 of 10 and flattered the comparison.

The numbers are what `cite.search` emits: cosine similarity plus the
literal-match boost the order is made from (0.02 per query word found as a
whole token), so the list reads in the order it is ranked. Command:
`library.search(conn, query, embedder=Embedder(), limit=3)` on the committed
file synced and embedded into a scratch database, then the same with
`embedder=None`.

| query | expected | semantic top 3 | substring top 3 (word-AND) |
|---|---|---|---|
| equivariant force fields | NequIP or MACE | MACE 0.610, TorchMD-NET 0.602, NequIP 0.560 | MACE, TorchMD-NET |
| message passing on molecular graphs | MPNN | SphereNet, DimeNet, MPNN | PaiNN, GemNet, SphereNet (6 found) |
| protein structure prediction | AlphaFold 2 | AlphaFold 2, ESM-2, ProteinMPNN | RFdiffusion, ESM-2, EquiBind (7 found) |
| generative models for molecules | the chemical VAE or MolGAN | MolGAN, GraphAF, the chemical VAE | GraphAF, MolGAN |
| catalyst adsorption energies | OC20 | OC22, OC20 | OC22 |
| universal interatomic potentials | M3GNet or CHGNet | M3GNet, GAP, MTP | Matbench Discovery, M3GNet |
| docking | DiffDock | DiffDock, EquiBind | DiffDock, EquiBind |
| materials stability | Matbench Discovery | M3GNet, Matbench Discovery, MatterSim | Matbench Discovery |
| SMILES language models | ChemBERTa or SMILES Transformer | SMILES Transformer, ESM-2, JT-VAE | SMILES Transformer |
| atomic environment descriptors | SOAP | SOAP | SOAP |

Semantic search put the expected paper in its top three for 10 of 10 (in
the top two for 9); word-AND substring found something for 10 of 10 but had
the expected paper in its top three for 7, because it orders by year and
cannot rank -- "protein structure prediction" matches seven papers and lists
the three newest. Four semantic rows carry a category-wrong neighbour a
chemist would notice (JT-VAE is a graph model built to avoid SMILES and
ESM-2 is a protein language model; GAP and MTP are potentials fitted per
material, not universal ones; ProteinMPNN is inverse folding), which is what
a 48-paper library at a 0.9 relative floor looks like. At 48 papers the
boost (up to +0.08) exceeds the gaps between the top three, so it is doing
half the ranking; the ordering is honest about that because the boosted
number is the one shown. The test run above has no embedder, which is why
its search line ends in `substring search only` -- the caveat is the
contract, and `cite.search` over MCP carries the same `semantic` flag.

**References join the concept graph through `keywords`.** `kg.tag_assignments`
reads `reference_tags` beside `item_tags`, with references as negative ids so
`build_graph` needs no change. On this scratch database the graph is made of
references alone: 67 distinct tags, 18 that clear the frequency floor, three
communities (`machine-learning`, `deep-learning`, `materials-science`), the
largest 22% of nodes. With the references removed the same `kg-report` exits
1 with "no items yet" -- the graph is empty. Tags came from the tagger at
generation time against an EMPTY item vocabulary (this scratch database has
no items), so they are what gemma4:e2b invents for a corpus, not a
chemist's: the first generation put `molecular-mechanics` -- a term of art
for classical force fields -- on 36 of 48 papers including AlphaFold 2 and
ChemBERTa; this one still puts it on 5 (SOAP, ANI-1, PhysNet, DimeNet,
TorchMD-NET, none of which is molecular mechanics), `computer-vision` on 4,
`genomics` on ESM-2, and carries a dozen conceptual near-duplicate families
(`graph-networks` / `graph-neural-networks` / `graph-message-passing`,
`drug-design` / `drug-discovery`, ...) that `kg_aliases.toml`'s spelling
folds do not cover. That is a tagging-eval case, not a graph finding.
`attest library tag` re-tags a library whose `.bib` carries none, against
the reader's own vocabulary when the database has items.

**Citation edges stay out of concept adjacency.** `cites` becomes
`reference_cites`, read only by `related` and `cite.related`; the concept
graph is co-occurrence of tags and nothing else, which the knowledge-graph
spec chose on purpose. Of NequIP's 66 references, 11 are in this library and
55 are stubs with a title and an identity, never fetched. Across the file,
2,678 edges name 227 in-library rows by id (and a few more by title, which
`related` resolves); 311 are title-only stubs. The reader drops the id-less
entries Semantic Scholar's parsed reference lists carry that could never
name a paper (journal abbreviations, `AUTHOR CONTRIBUTIONS`, equation
fragments), but the rule -- four words of three or more letters, no brace --
is a floor, not a filter: roughly a third of the remaining title-only stubs
are still parse noise (NeurIPS checklist questions, figure captions,
changelog bullets, funding lines), and some id-bearing junk survives too
(GAP "cites" a British Journal of Ophthalmology paper, because that is what
Semantic Scholar's parse of its reference list says). What this file holds
is Semantic Scholar's reference lists as they are, not a curated
bibliography; `n_cites` counts them all.

## When it goes wrong

- `related` for a key that is not in the file exits 1 with the count of
  references in the store; keys are the generated `<surname><year><word>`
  form, so `batzner2022equivariant`, not `nequip`.
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
