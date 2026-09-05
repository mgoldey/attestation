# References in the graph, citation neighbourhoods, and a molecular-AI golden path

**Date:** 2026-09-05
**Status:** approved in brainstorm 2026-09-05 as spec 2 of 3 (approach A: a
sibling store that reuses the item pipelines; the user chose "semantic, plus
the knowledge graph" and "generated from real papers"). Written and executed
autonomously overnight on the user's instruction; deviations from spec 1 are
recorded where they occur.
**Depends on:** `2026-09-05-library-store-design.md` (the store, readers,
search, `cite.*`), `2026-08-06-knowledge-graph-design.md` (the pure graph and
why citation edges never enter concept adjacency), the golden-paths
convention enforced by `tests/test_golden_paths.py`.

## Problem

Spec 1 built the library and left three things out on purpose:

1. References carry tags (`reference_tags`) but the concept graph reads only
   `item_tags`, so what the reader cites is invisible to `kg.*`.
2. `reference_cites` holds real reference lists from Semantic Scholar and
   nothing reads them: "what does this paper cite, and which of those do I
   already have" has no tool.
3. There is no example. The feed corpus is ML-heavy and a chemistry reader
   cannot see, from anything committed, that the library finds SchNet for
   "equivariant force fields".

## Decisions

- **References join the graph as participants, not as a second graph.**
  `kg.tag_assignments` returns items' `(item_id, tag)` pairs plus references'
  `(-reference_id, tag)` pairs. Negative ids keep the function's `int` type,
  cannot collide with `items.id`, and `build_graph` only ever uses an id as a
  grouping key, so every `kg.*` tool sees references with no change. Nothing
  maps a graph id back to a row today; if something does, the sign says
  which table.
- **Citation edges stay out of concept adjacency.** The KG spec refused to
  invent `CITES` from co-occurrence; real edges from S2 are still a different
  relation from "these tags co-occur" and would misrepresent the graph's one
  meaning. They get their own read, `cite.related`, deterministic and
  model-free.
- **Tags and citation edges can arrive from a `.bib` file.** BibTeX's
  standard `keywords` field becomes `reference_tags`; a non-standard `cites`
  field (`identity|title; identity|title`) becomes `reference_cites`. This
  is what lets a golden path run with no model server and no network from a
  committed file that real software wrote, and it is also how a Zotero export
  with keywords contributes to the graph.
- **The example is generated from real papers with the real readers.**
  `generate.py` seeds ~40 canonical molecular-AI papers by arXiv id or DOI,
  runs `library.sync` with both network flags into a scratch database,
  tags them with the real tagger, and writes `references.bib` through
  bibtexparser v2 -- the same discipline as `examples/citations/`. The
  committed file is the output of that run, scrubbed of paths.
- **The golden path is offline.** `none — pure local computation`: it syncs
  the committed `.bib`, searches by substring, walks a citation
  neighbourhood, and reports graph health with references in it. The
  semantic-search measurement (which needs the model server) is recorded in
  the README as numbers taken once, not as pinned output.

## 1. Readers: keywords and cites

`ReferenceRecord` gains `tags: list[str]` (default empty). `BibtexRecords`
fills it from `keywords`, split on `,` or `;`, lowercased and folded the way
`features.ItemTags` folds (`strip().lower().replace(" ", "-")`; anything that
fails `TAG_PATTERN` is dropped). `ZoteroRecords` leaves it empty for now --
Zotero's `itemTags` join is a separate table the reader does not read yet,
noted as a follow-up rather than half-built.

`BibtexRecords` also fills `cites` from a `cites` field:
`doi:10.5555/schnet|SchNet: A continuous-filter CNN; arxiv:2101.03164|E(3)-
equivariant graph neural networks`. Each entry is `identity|title`; a bare
identity with no `|` has title `None`. Identities are stored as given (they
were produced by `library.identity`), not re-derived.

`upsert` writes `reference_tags` with `INSERT OR IGNORE` for every tag on the
record, after the row exists. Tags from a file never delete tags the tagger
wrote; they only add.

## 2. The graph

`kg.tag_assignments(conn)`:

```python
rows = conn.execute("SELECT item_id, tag FROM item_tags")
refs = conn.execute("SELECT reference_id, tag FROM reference_tags")
return [(r["item_id"], r["tag"]) for r in rows] + [(-r["reference_id"], r["tag"]) for r in refs]
```

`features.tag_vocabulary` sums `reference_tags` counts into the same
canonical totals, so the tagger is steered by what the reader cites as well
as what they read. `kg.health` gains `n_references` (distinct reference ids
in the assignments) beside its existing counts, and `attest kg-report` prints
it, so a reader can see that references are in the graph rather than assume.

The KG spec's order (aliases, frequency filter, co-occurrence) is untouched;
the DB-free ordering test still holds because `build_graph` did not change.

## 3. `cite.related`

`library.related(conn, key) -> Related | None`, pure over the tables:

```python
@dataclass
class Related:
    reference: SearchHit          # the paper itself
    cites: list[Neighbour]        # what it cites, from reference_cites
    cited_by: list[Neighbour]     # library rows whose cites include it
    n_cites: int
    n_cited_by: int

@dataclass
class Neighbour:
    identity: str
    title: str | None
    in_library: bool
    key: str | None               # bib_key or identity when in the library
    id: int | None
```

`cites` resolves each `cited_identity` against `references.identity` (and,
for `doi:`/`arxiv:` identities, the `doi`/`arxiv_id` columns, so an edge
recorded as `arxiv:...` still finds a row that later gained a DOI).
`cited_by` is the reverse: rows with a `reference_cites` entry whose
`cited_identity` matches this row's identity, DOI form, or arXiv form.
Both lists are ordered in-library first, then by title, and capped at 20
with the true counts beside them.

The tool, `cite.related(key)`, is the 48th, on the `knowledge` surface
under the existing `cite` prefix. It is a composition tool
(`test_response_size.COMPOSITION_TOOLS`: one row per edge). `kg.ask` gains
no route for it in this spec; "what does X cite" is a reasonable future
route and is left for the skill re-run in spec 3 to motivate with cases.

CLI: `attest library related KEY`.

## 4. `examples/molecular-ai/`

Files: `README.md` (the seven sections), `run.sh`, `generate.py`,
`references.bib` (generated), `seeds.toml` (the ~40 seed ids and the
hand-curated titles that make them offline rows, so the file that is typed
by hand is a list of identifiers and nothing that could be mistaken for
fetched metadata).

`run.sh`:

```bash
export ATTEST_DB="$(mktemp -d)/attest.db"
export ATTEST_BIB_PATHS=$PWD/references.bib
uv run attest library sync --sources bibtex
uv run attest library search "force field" --limit 5
uv run attest library related batzner2022equivariant
uv run attest kg-report
```

Pinned line: `bibtex: +N added, 0 merged, 0 unchanged` with the generated
count. The search is substring (no embedder in the test run) and the output
says so; the README explains why and gives the semantic numbers measured
once with the model server (§6). `kg-report` must show `n_references` equal
to the library size and concepts drawn from the `keywords`.

`generate.py`, run by hand with the network flags:

```bash
ATTEST_CITATION_WEB=1 ATTEST_CITATION_SCHOLAR=1 \
  uv run --with "bibtexparser>=2.0.0b9" python generate.py
```

1. Reads `seeds.toml`, writes a scratch `seed.bib` of `key`, `title`, and
   the id fields, syncs it into a scratch database with the enrichers armed
   (`readers_from_env` with `cache_dir` under the scratch dir so nothing
   touches `~/.hermes`), then `run_reference_tagging` with the real chat
   model.
2. Exports every row as a bibtexparser v2 entry: `title`, `author`,
   `year`, `journal` (venue), `doi`, `eprint`+`archiveprefix`, `abstract`,
   `url`, `keywords` (the tags), `cites` (the edges). Keys are
   `<firstauthor><year><firstword>` lowercased, ASCII only.
3. Scrubs: no `/home/`, no username, no `fetched_at` (the file is offline
   by definition; provenance for the example is `generate.py` itself).
4. Prints what it fetched and what it could not (an S2 miss is a paper
   with no `cites`), so the README's "When it goes wrong" can quote it.

The seeds (real, well-known; ids checked at generation, any that fail to
resolve are dropped with a note): SchNet, PhysNet, DimeNet, DimeNet++,
GemNet, NequIP, MACE, Allegro, PaiNN, TorchMD-NET / Equivariant
Transformer, SphereNet, ANI-1, ANI-2x, AIMNet2, Chemprop / D-MPNN,
MoleculeNet, Uni-Mol, GROVER, MolCLR, ChemBERTa, SMILES Transformer,
MolGAN, JT-VAE, GraphAF, DiffDock, EquiBind, TorsionDiff, Open Catalyst 2020,
OC22, Matbench Discovery, M3GNet, CHGNet, MatterSim, GNoME, AlphaFold 2,
RoseTTAFold, ESM-2, RFdiffusion, ProteinMPNN, DeePMD, DeepChem. That was
the draft list; `seeds.toml` is the record of what was seeded -- 48 papers,
all of which resolved -- and it differs: AIMNet2, Uni-Mol and DeepChem were
not carried over (Uni-Mol and AIMNet2 have ChemRxiv DOIs and could be;
DeepChem has no paper), and SE(3)-Transformers, EGNN, MPNN, CGCNN, MEGNet,
Behler-Parrinello, GAP, SOAP and MTP were added so the interatomic-potential
line starts where a chemist would start it. Still missing, in the order the
file's own `cites` edges name them: Duvenaud 2015 neural fingerprints, DTNN,
Behler 2011 symmetry functions, the Coulomb matrix, Tensor Field Networks,
ACE, sGDML, QM9, EDM for the generative line; and nothing after May 2024
(AlphaFold 3 sits inside the cutoff).

## 5. Docs and counts

Tool count 47 → 48; `cite.*(5)` → `(6)`; the knowledge surface re-measured
and CLAUDE.md's line updated with the real number; `docs/guides/agents.md`
table row; the knowledge skill names `cite.related` in its references
section; `examples/README.md` catalogue row (offline, ~5 s); CLAUDE.md docs
index gains `examples/molecular-ai`; `docs/guides/claims-and-citations.md`
"The library" section gains a sentence on `related` and on `keywords`/`cites`.

## 6. Measurements

Taken 2026-09-05 while generating the example (full tables in
`examples/molecular-ai/README.md`):

- **Semantic vs substring, ten molecular-AI queries, embeddinggemma
  (re-measured after review round 1, with an `expected` paper written down
  per query):** semantic put the expected paper in its top three for 10 of
  10 (top two for 9); the word-AND substring fallback found something for
  10 of 10 but had it in the top three for 7, since it orders by year and
  cannot rank. The first measurement compared against a whole-phrase
  substring that found nothing for 7 of 10 -- a strawman; `_substring` is
  now word-AND and the README table carries both columns and the expected
  paper. "equivariant force fields": MACE 0.610, TorchMD-NET 0.602, NequIP
  0.560 above the relative floor (cosine plus the literal boost, the number
  `cite.search` now emits).
- **Generation (second, after review round 1):** 48 of 48 seeds resolved.
  Semantic Scholar answered 48 of 48 across five resumed passes; 4 entries
  carry no `cites` (their record lists no traceable reference). arXiv
  answered 40 ids (37 seeds plus 3 DOI seeds -- GAP, SOAP, MTP -- that
  Semantic Scholar gave an arXiv id) in one batched request; CrossRef
  enriched 25. The three DataCite `10.48550/arxiv.*` DOIs Semantic Scholar
  offered (MACE, DiffDock, CHGNet) are read as the arXiv ids those rows
  already had. The first generation surfaced and fixed:
  the arXiv API 301s plain http; a rate-limited row recorded as tried; a
  paced enricher holding the write lock across its sleeps. The review of
  that generation surfaced and fixed: a 404 never marking a row tried
  (re-fetched every sync); an enricher's answer attaching by the ids it
  carried rather than to the row it was fetched for; the arXiv year kept
  beside CrossRef's journal (11 wrong citations, now the venue's year wins
  and the old year is a recorded conflict); a URL scrub eating the `},`
  that closed three abstracts (invalid BibTeX, now guarded by
  `test_every_committed_bib_is_valid_bibtex`); and id-less junk in
  Semantic Scholar's parsed reference lists (`Phys. Rev. B`, checklist
  questions, equation fragments) counted as citations -- now dropped unless
  the title has four words of three or more letters and no brace: 2,678
  edges, 227 in-library by id, 311 title-only stubs, of which roughly a
  third are still parse noise the rule cannot see (checklist questions,
  figure captions). Round 2 added: the persisted `title_key` column
  (migration 008) so a title-only .bib entry survives a second sync and a
  title stub resolves to the row that gained a DOI; a KNN prefilter so a
  filtered semantic search works above 4,096 vectors; a reader's `.bib`
  keywords on a feed paper joining the item's node instead of vanishing;
  the boosted similarity emitted as the number the order is made from; and
  `@misc` plus `\&` in the writer so the file compiles under `bibtex`.
- **Graph with and without references:** the example's scratch database has
  no items, so without references `kg-report` exits 1 ("no items yet"); with
  the 48 tagged references it reports 18 nodes, 22 edges, 67 distinct tags,
  three communities (largest 22.2%). The first generation's single
  `molecular-mechanics` community at 85.7% was the tagger failing to
  separate five obvious sub-areas on an EMPTY vocabulary, not the corpus's
  shape; this generation still shows near-duplicate tags the alias table
  does not fold. A chemist's vocabulary for this corpus is a tagging-eval
  case, not a graph finding.
- **Tagging cost:** 48 references in one pass, 0 failed, gemma4:e2b, twice.

## 7. What the example changed in the readers

Recorded because the golden path is the first real use of the enrichers:
`follow_redirects=True` and the https arXiv endpoint; transient failures
(transport errors, a 429 that outlasts one back-off, a 5xx after two) write
nothing and are counted as `failed`, so the next sync retries them; S2 paced
at 3 s per real request; `sync` commits after every enricher record.

## 8. Tests

DB-free: `_bib_tags("A, B; c d")`, `_bib_cites(...)` parsing including a
bare identity, `tag_assignments` shape over a fake connection is not
DB-free (it reads two tables) -- test it hermetic. Hermetic: sync of a
`.bib` with `keywords` and `cites` yields rows in both tables; `build_graph`
over the union contains a concept that only references carry; `related`
resolves an `arxiv:` edge to a row that has since gained a DOI; `cited_by`
finds the reverse edge; caps and counts; `cite.related` envelope and
`ToolError` for an unknown key; `tag_vocabulary` counts reference tags;
`kg.health()["n_references"]`. The golden path is tested by
`tests/test_golden_paths.py` unchanged: seven sections, the run, the pinned
line, the attribution guard over the committed `.bib`.

## What this spec does not decide

Zotero tags (`itemTags`); a `kg.ask` route to `cite.related`; showing a
cited-but-absent paper anywhere except `cite.related`; importing a cited
paper into the library (an enricher that introduces rows is exactly what
spec 1 forbade, and the right shape is a deliberate `cite.adopt` the reader
asks for, not a side effect of sync).
