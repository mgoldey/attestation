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
uv run attest library related batzner2022nequip
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
RoseTTAFold, ESM-2, RFdiffusion, ProteinMPNN, DeePMD, DeepChem. Forty-one
candidates; the committed file holds whichever resolved.

## 5. Docs and counts

Tool count 47 → 48; `cite.*(5)` → `(6)`; the knowledge surface re-measured
and CLAUDE.md's line updated with the real number; `docs/guides/agents.md`
table row; the knowledge skill names `cite.related` in its references
section; `examples/README.md` catalogue row (offline, ~5 s); CLAUDE.md docs
index gains `examples/molecular-ai`; `docs/guides/claims-and-citations.md`
"The library" section gains a sentence on `related` and on `keywords`/`cites`.

## 6. Measurements

To take when the example exists, with the model server up, recorded in the
README's *What it demonstrates* as numbers:

- For ten molecular-AI queries ("equivariant force fields", "message
  passing on molecular graphs", "protein structure prediction",
  "generative models for molecules", "catalyst adsorption energies",
  "universal interatomic potentials", "docking", "materials stability",
  "SMILES language models", "atomic environment descriptors"): the top-3
  semantic hits vs the top-3 substring hits, and how many of the ten
  substring searches return nothing at all.
- `generate.py` wall time with S2 at one request per second for N seeds.
- `kg-report` health with and without the references in the graph, on the
  example database, to show what 40 tagged references do to a graph that
  was empty.

## 7. Tests

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
