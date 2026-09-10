# Library Graph and Molecular-AI Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** References join the concept graph through their tags, `cite.related` walks real citation edges, and `examples/molecular-ai/` shows both offline from a `.bib` that real software generated from real papers.

**Architecture:** Two small extensions to spec 1's readers (`keywords`, `cites`), one union in `kg.tag_assignments` with negative ids for references, one pure read (`library.related`) exposed as the 48th tool and a CLI verb, and a golden path whose `generate.py` runs the real sync with the network flags and writes the fixture with bibtexparser v2.

**Tech Stack:** Python 3.12, sqlite3, bibtexparser 2 (generation only, via `uv run --with`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-library-graph-and-molecular-ai-design.md`

## Global Constraints

- Everything in spec 1's plan still applies: line length 100, the pre-commit gate before every commit that touches `src/`, domain modules never import `attestation.mcp`, no new `# noqa: BLE001`, hermetic tests.
- Tool count 47 → 48 and `cite.*(5)` → `(6)`: update `CLAUDE.md` (line 5, the "Agent surfaces" line with the re-measured knowledge count, the "MCP surface" line), `README.md:88`, `docs/guides/agents.md` (count + table row + "all N"), `docs/concepts.md`, `docs/architecture/research-profile.md` in the same commit as the tool.
- The golden path must pass `tests/test_golden_paths.py` unchanged: seven sections in order, prerequisite `none — pure local computation`, every fenced `Run it` command in `run.sh` verbatim, a real pinned first line under `What it prints`, no `/home/`, `github.com`, or the username anywhere under `examples/molecular-ai/`, a catalogue row in `examples/README.md` ordered by prerequisite then name, and the README opening with `<!-- checked by tests/test_golden_paths.py -->`.
- `run.sh` runs under `HOME=<tmp>` and `LLM_BASE_URL=http://127.0.0.1:9/v1`: no model server, no network, no `~/.hermes`.
- Work on branch `feat/library-graph`, created from `feat/library-store` in the same worktree; the PR targets `feat/library-store` until that merges.

---

## File map

| file | responsibility |
|---|---|
| `src/attestation/library.py` | `ReferenceRecord.tags`; `upsert` writes `reference_tags`; `related()`, `Related`, `Neighbour` |
| `src/attestation/library_readers.py` | `_bib_tags`, `_bib_cites`; `BibtexRecords` fills `tags`/`cites` |
| `src/attestation/kg.py` | `tag_assignments` union; `health()["n_references"]` |
| `src/attestation/features.py` | `tag_vocabulary` counts `reference_tags` |
| `src/attestation/mcp/citation.py` | `cite.related` |
| `src/attestation/cli.py` | `attest library related KEY`; `kg-report` prints `n_references` |
| `examples/molecular-ai/{README.md,run.sh,generate.py,seeds.toml,references.bib}` | the golden path |
| `examples/README.md` | catalogue row |
| docs listed in Global Constraints | counts and rows |
| `tests/test_library_readers.py`, `tests/test_library.py`, `tests/test_kg.py`, `tests/test_features.py`, `tests/test_library_tools.py`, `tests/test_cli.py`, `tests/test_response_size.py` | per task |

---

### Task 1: `keywords` and `cites` from a `.bib`

**Files:**
- Modify: `src/attestation/library.py` (ReferenceRecord.tags; upsert writes tags), `src/attestation/library_readers.py`
- Test: `tests/test_library_readers.py`, `tests/test_library.py`

**Interfaces:**
- Produces: `ReferenceRecord.tags: list[str]`; `library_readers._bib_tags(value: str) -> list[str]`; `library_readers._bib_cites(value: str) -> list[tuple[str, str | None]]`.

- [ ] **Step 1: Failing tests**

`tests/test_library_readers.py`:

```python
def test_bib_keywords_and_cites_fields_are_read():
    from attestation.library_readers import _bib_cites, _bib_tags

    assert _bib_tags("Force Fields, equivariant GNN; Molecular Dynamics") == [
        "force-fields", "equivariant-gnn", "molecular-dynamics",
    ]
    assert _bib_tags("!!!, ok") == ["ok"]
    assert _bib_cites("doi:10.5555/schnet|SchNet: a CNN; arxiv:2101.03164") == [
        ("doi:10.5555/schnet", "SchNet: a CNN"),
        ("arxiv:2101.03164", None),
    ]
    assert _bib_cites("") == []


def test_bibtex_records_carry_keywords_and_cites(tmp_path):
    bib = tmp_path / "k.bib"
    bib.write_text(
        "@article{k,\n  title = {K},\n  keywords = {force-fields, gnn},\n"
        "  cites = {doi:10.5555/schnet|SchNet},\n}\n"
    )
    (rec,) = library_readers.BibtexRecords([bib]).records()
    assert rec.tags == ["force-fields", "gnn"] and rec.cites == [("doi:10.5555/schnet", "SchNet")]
```

`tests/test_library.py`:

```python
def test_upsert_writes_tags_from_the_record_without_deleting_others(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    rid, _ = library.upsert(conn, _rec(bib_key="k", title="K", tags=["force-fields"]))
    conn.execute("INSERT INTO reference_tags VALUES (?, 'from-the-tagger')", (rid,))
    library.upsert(conn, _rec(source="zotero", source_key="Z", bib_key="Z", title="K", tags=["gnn"]))
    tags = {r["tag"] for r in conn.execute("SELECT tag FROM reference_tags WHERE reference_id = ?", (rid,))}
    assert tags == {"force-fields", "from-the-tagger", "gnn"}
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`library.py`: add `tags: list[str] = field(default_factory=list)` to `ReferenceRecord` (before `cites`), and in `upsert` after the source row:

```python
    if rec.tags:
        conn.executemany(
            "INSERT OR IGNORE INTO reference_tags(reference_id, tag) VALUES (?, ?)",
            [(rid, t) for t in dict.fromkeys(rec.tags)],
        )
```

`library_readers.py`:

```python
_TAG_SPLIT = re.compile(r"[,;]")


def _bib_tags(value: str) -> list[str]:
    """`keywords` -> tags, folded the way ItemTags folds; unusable ones dropped."""
    from attestation.features import TAG_PATTERN

    out = []
    for raw in _TAG_SPLIT.split(value or ""):
        tag = raw.strip().lower().replace(" ", "-")
        if tag and re.match(TAG_PATTERN, tag) and tag not in out:
            out.append(tag)
    return out


def _bib_cites(value: str) -> list[tuple[str, str | None]]:
    """`identity|title; identity` -> (identity, title-or-None) pairs."""
    out = []
    for raw in (value or "").split(";"):
        entry = raw.strip()
        if not entry:
            continue
        ident, _, title = entry.partition("|")
        out.append((ident.strip(), title.strip() or None))
    return out
```

and in `BibtexRecords.records`: `tags=_bib_tags(f.get("keywords", ""))`, `cites=_bib_cites(f.get("cites", ""))`. (`features` imports nothing from `library_readers`, so the local import is cycle-free; keep it local because `features` pulls sklearn-free but pydantic-heavy modules.)

- [ ] **Step 4: Run** `uv run pytest tests/test_library_readers.py tests/test_library.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "A .bib's keywords become reference tags and a cites field becomes citation edges"`

---

### Task 2: References in the graph and the vocabulary

**Files:**
- Modify: `src/attestation/kg.py` (`tag_assignments`, `health`), `src/attestation/features.py` (`tag_vocabulary`), `src/attestation/cli.py` (`cmd_kg_report` prints `n_references`)
- Test: `tests/test_kg.py`, `tests/test_features.py`

- [ ] **Step 1: Failing tests**

`tests/test_kg.py`:

```python
def test_references_join_the_graph_with_negative_ids(tmp_path):
    from attestation import kg
    from attestation.db import get_db
    from attestation.library import ReferenceRecord, upsert

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO items(id, title, content_hash) VALUES (1, 'i', 'h')")
    conn.executemany("INSERT INTO item_tags VALUES (1, ?)", [("gnn",), ("chemistry",)])
    for k in ("a", "b"):
        upsert(conn, ReferenceRecord(source="bibtex:/x", source_key=k, bib_key=k, title=k,
                                     tags=["force-fields", "gnn"]))
    pairs = kg.tag_assignments(conn)
    assert (1, "gnn") in pairs and (-1, "force-fields") in pairs and (-2, "gnn") in pairs
    adjacency, _ = kg.build_graph(pairs)
    # force-fields is carried ONLY by references (2 uses) and reaches the frequency floor.
    assert "force-fields" in adjacency and "gnn" in adjacency["force-fields"]
    assert kg.health(conn)["n_references"] == 2
```

`tests/test_features.py`:

```python
def test_tag_vocabulary_counts_reference_tags(tmp_path):
    from attestation.library import ReferenceRecord, upsert

    conn = seeded_db(tmp_path / "t.db")
    upsert(conn, ReferenceRecord(source="bibtex:/x", source_key="a", bib_key="a", title="A",
                                 tags=["only-in-references"]))
    assert "only-in-references" in tag_vocabulary(conn)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`kg.py`:

```python
def tag_assignments(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Every (id, tag) pair the graph is built from: items as their own ids,
    references as NEGATIVE ids. build_graph only groups by id, so the sign is
    a namespace, not a meaning -- it keeps the two tables from colliding and
    says which one a pair came from if anything ever needs to know."""
    items = conn.execute("SELECT item_id, tag FROM item_tags")
    refs = conn.execute("SELECT reference_id, tag FROM reference_tags")
    return [(r["item_id"], r["tag"]) for r in items] + [(-r["reference_id"], r["tag"]) for r in refs]
```

In `health()` (read it first: it builds `assignments = tag_assignments(conn)` at kg.py:450), add `"n_references": len({i for i, _ in assignments if i < 0})` and `"n_items": len({i for i, _ in assignments if i > 0})` to the returned dict. In `tag_vocabulary`, add a second loop over `SELECT tag, COUNT(*) n FROM reference_tags GROUP BY tag` accumulating into the same `totals`. In `cmd_kg_report`, print `n_references` on its own line right after the node/edge line (read the function; match its print style).

- [ ] **Step 4: Run** `uv run pytest tests/test_kg.py tests/test_kg_algorithms.py tests/test_kg_mcp.py tests/test_features.py tests/test_cli.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "References join the concept graph as negative ids; the vocabulary and kg-report count them"`

---

### Task 3: `library.related`, `cite.related`, `attest library related`

**Files:**
- Modify: `src/attestation/library.py`, `src/attestation/mcp/citation.py`, `src/attestation/cli.py`, `tests/test_response_size.py` (COMPOSITION_TOOLS), docs per Global Constraints, `src/attestation/skills/attestation-knowledge/SKILL.md`
- Test: `tests/test_library.py`, `tests/test_library_tools.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `library.related(conn, key) -> Related | None`; `Related.to_row() -> dict` with keys `reference`, `cites`, `cited_by`, `n_cites`, `n_cited_by`; `Neighbour` fields per spec §3.

- [ ] **Step 1: Failing tests**

`tests/test_library.py`:

```python
def test_related_resolves_edges_both_ways_and_caps(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    nequip = _store_with(conn, bib_key="nequip", title="NequIP", doi="10.1038/s41467-022-29939-5",
                         arxiv_id="2101.03164")
    schnet = _store_with(conn, bib_key="schnet", title="SchNet", arxiv_id="1706.08566")
    # An edge recorded by arXiv id must still find SchNet after it gains a DOI.
    library.upsert(conn, _rec(source="s2", source_key="x", doi="10.1038/s41467-022-29939-5",
                              fetched_at="2026-09-05",
                              cites=[("arxiv:1706.08566", "SchNet"), ("title:elsewhere:-", "Elsewhere")]))
    library.upsert(conn, _rec(bib_key="schnet2", title="SchNet", arxiv_id="1706.08566", doi="10.5555/schnet"))
    rel = library.related(conn, "nequip")
    assert rel.reference.id == nequip
    assert [(n.identity, n.in_library, n.key) for n in rel.cites] == [
        ("arxiv:1706.08566", True, "schnet"), ("title:elsewhere:-", False, None)]
    assert rel.n_cites == 2 and rel.cited_by == []
    back = library.related(conn, "schnet")
    assert [n.key for n in back.cited_by] == ["nequip"] and back.reference.id == schnet
    assert library.related(conn, "nope") is None
```

`tests/test_library_tools.py`: `citation._related("nequip")` envelope has `ok`, `reference`, `cites`, `cited_by`; unknown key → `ok: False` with the store count in the message.

`tests/test_cli.py`: `main(["library", "related", "nequip"])` prints `cites 2` and one line per neighbour with `[in library]` / `[not in library]`.

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`library.py`:

```python
@dataclass
class Neighbour:
    identity: str
    title: str | None
    in_library: bool
    key: str | None
    id: int | None

    def to_row(self) -> dict:
        return {"identity": self.identity, "title": (self.title or "")[:90] or None,
                "in_library": self.in_library, "key": self.key, "id": self.id}


@dataclass
class Related:
    reference: SearchHit
    cites: list[Neighbour]
    cited_by: list[Neighbour]
    n_cites: int
    n_cited_by: int

    def to_row(self) -> dict:
        return {"reference": self.reference.to_row(), "cites": [n.to_row() for n in self.cites],
                "cited_by": [n.to_row() for n in self.cited_by],
                "n_cites": self.n_cites, "n_cited_by": self.n_cited_by}


MAX_NEIGHBOURS = 20


def _identity_forms(row) -> list[str]:
    forms = [row["identity"]]
    if row["doi"]:
        forms.append(f"doi:{row['doi']}")
    if row["arxiv_id"]:
        forms.append(f"arxiv:{row['arxiv_id']}")
    return list(dict.fromkeys(forms))


def _row_for_identity(conn, ident: str):
    kind, _, value = ident.partition(":")
    for sql, val in (
        ('SELECT * FROM "references" WHERE identity = ?', ident),
        ('SELECT * FROM "references" WHERE doi = ?', value if kind == "doi" else None),
        ('SELECT * FROM "references" WHERE arxiv_id = ?', value if kind == "arxiv" else None),
    ):
        if val and (row := conn.execute(sql, (val,)).fetchone()):
            return row
    return None


def _neighbour(conn, ident: str, title: str | None) -> Neighbour:
    row = _row_for_identity(conn, ident)
    if row is None:
        return Neighbour(ident, title, False, None, None)
    return Neighbour(ident, row["title"], True, row["bib_key"] or row["identity"], row["id"])


def related(conn: sqlite3.Connection, key: str) -> Related | None:
    """What a paper cites and what in the library cites it, deterministic.

    Edges come from reference_cites (Semantic Scholar via sync, or a .bib
    `cites` field). A cited identity resolves to a library row by identity,
    DOI or arXiv form, so an edge recorded before a paper gained its DOI still
    lands. In-library first, then by title; capped with the true counts.
    """
    row = lookup_row(conn, key)
    if row is None:
        return None
    cites = [
        _neighbour(conn, r["cited_identity"], r["cited_title"])
        for r in conn.execute(
            "SELECT cited_identity, cited_title FROM reference_cites WHERE citing_id = ?", (row["id"],)
        )
    ]
    forms = _identity_forms(row)
    marks = ",".join("?" * len(forms))
    citing = conn.execute(
        f'SELECT DISTINCT r.* FROM "references" r JOIN reference_cites c ON c.citing_id = r.id'
        f" WHERE c.cited_identity IN ({marks})",
        forms,
    ).fetchall()
    cited_by = [Neighbour(r["identity"], r["title"], True, r["bib_key"] or r["identity"], r["id"]) for r in citing]
    order = lambda n: (not n.in_library, (n.title or "").lower())  # noqa: E731
    cites.sort(key=order)
    cited_by.sort(key=order)
    return Related(_hit(conn, row), cites[:MAX_NEIGHBOURS], cited_by[:MAX_NEIGHBOURS], len(cites), len(cited_by))
```

(Replace the lambda with a small `def _neighbour_order(n)` — ruff E731.)

`mcp/citation.py`:

```python
@tool(empty={"reference": None, "cites": [], "cited_by": [], "n_cites": 0, "n_cited_by": 0}, label="cite_related")
def _related(conn, key: str) -> dict:
    from attestation import library

    rel = library.related(conn, key)
    if rel is None:
        raise ToolError(f"no library reference matches {key!r} (store: {library.status(conn)['references']} references; cite.sync fills it)")
    return rel.to_row()
```

registered as `cite.related(key)` with the docstring: "What a paper cites and what in the library cites it. Edges come from Semantic Scholar reference lists (ATTEST_CITATION_SCHOLAR at sync time) or a .bib `cites` field; a cited paper that is not in the library is listed with `in_library: false`, never fetched. Deterministic; no model." Add `"cite.related": "one row per citation edge, in-library first"` to `COMPOSITION_TOOLS`. CLI `library related KEY` prints the reference line, then `cites N` and `cited_by N` headers with one line each: `  [in library] key  title` / `  [not in library] identity  title`.

Docs: counts 48 / `cite.*(6)`, table row in agents.md (`cite.related(key)` | what a paper cites and what cites it, from real reference lists | fast), knowledge skill sentence, `docs/guides/claims-and-citations.md` one sentence on `related` and `keywords`/`cites`. Re-measure the knowledge surface with `.venv/count_tools.py` and write the number into CLAUDE.md's Agent surfaces line.

- [ ] **Step 4: Run** `uv run pytest tests/test_library.py tests/test_library_tools.py tests/test_cli.py tests/test_response_size.py tests/test_architecture.py tests/test_skill_files.py tests/test_docs_site.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "cite.related: a paper's citation neighbourhood from real reference lists, both directions, deterministic; 48 tools"`

---

### Task 4: `examples/molecular-ai/` — generator, fixture, README, run.sh

**Files:**
- Create: `examples/molecular-ai/seeds.toml`, `generate.py`, `references.bib`, `README.md`, `run.sh`
- Modify: `examples/README.md`, `CLAUDE.md` docs index
- Test: `tests/test_golden_paths.py` (unchanged; discovers the directory)

- [ ] **Step 1: `seeds.toml`** — one `[[seeds]]` per paper with `key`, `title`, and `arxiv` or `doi` (the 41 in spec §4; look each id up as you write it and prefer arXiv ids for preprints, DOIs for journal papers).

- [ ] **Step 2: `generate.py`** (run by hand, network on):

```python
"""Generates references.bib from real papers with the real readers.

Seeds (ids and titles, seeds.toml) become offline rows in a scratch library;
the arXiv, CrossRef and Semantic Scholar enrichers fill abstracts, authors,
venues and reference lists; the real tagger tags them; bibtexparser v2 writes
the file. Nothing in references.bib was typed to look fetched.

    ATTEST_CITATION_WEB=1 ATTEST_CITATION_SCHOLAR=1 \
      uv run --with "bibtexparser>=2.0.0b9" python generate.py
"""
```

Body: load seeds; write `scratch/seed.bib` (key, title, `eprint`/`archiveprefix` or `doi`); `conn = get_db(scratch/"lib.db")`; `readers = readers_from_env(conn, bib_paths=[seed_bib], zotero_path=scratch/"none", cache_dir=scratch/"cache")`; `report = sync(conn, readers)`; print the report per source; `run_reference_tagging(conn, default_chat_fn, chat_model())` and print its stats; then for every row build a bibtexparser `Entry` with fields `title, author (" and ".join), year, journal (venue), doi, eprint+archiveprefix, abstract, url, keywords (", ".join tags), cites ("; ".join(f"{ident}|{title}" ...))`, key = `<first author surname><year><first title word>` ASCII-folded; `write_string` → `references.bib`. Assert no `/home/` and no `os.environ["USER"]` in the output before writing. Print per-seed: fetched / missed (no abstract, no cites). Time the S2 pass and print it.

- [ ] **Step 3: Run the generator** with the flags (the model server is up; S2 at 1 rps for ~41 seeds is ~1 minute). Review the `.bib` diff by eye: every entry has a title; most have abstracts and keywords; note the misses.

- [ ] **Step 4: `run.sh`** exactly as spec §4, with `set -euo pipefail` and `cd "$(dirname "$0")"`. Run it locally once with `HOME=$(mktemp -d) LLM_BASE_URL=http://127.0.0.1:9/v1` to capture real output for the README.

- [ ] **Step 5: `README.md`** with the seven sections. *What you get*: the library, where it came from, how many resolved. *Prerequisites*: `none — pure local computation`. *Run it*: the four commands. *What it prints*: the sync line first (pinned), the substring search's five lines and its `substring` caveat, the `related` output for NequIP, the kg-report lines including `n_references`. *What it demonstrates*: identity/dedup across seeds that resolved to the same DOI (if any), `keywords`→graph, `cites`→`related`, the semantic-vs-substring table measured once with the model (§6 of the spec; run `attest library search` for the ten queries with the model up and paste the top-3 each way), and that `generate.py` is the provenance. *When it goes wrong*: S2 misses, a seed that never resolved, `substring` when the embedder is down. *Next*: regenerate, then `attest library tag` with a model to see the graph re-cluster.

- [ ] **Step 6: catalogue row** in `examples/README.md` in prerequisite-then-name order: `| \`molecular-ai/\` | forty-odd canonical molecular-AI papers as a library generated from real APIs, deduplicated on sync, searched, walked by citation edge, and counted in the concept graph | \`none — pure local computation\` | ~5 s |`. Add `examples/molecular-ai:{README.md,run.sh,generate.py,seeds.toml,references.bib}` to CLAUDE.md's docs index and `molecular-ai/` to the examples list on that line.

- [ ] **Step 7: Run** `uv run pytest tests/test_golden_paths.py -q` → all pass including the offline run of this path. Then `uv run pytest tests/test_examples.py tests/test_architecture.py -q`.
- [ ] **Step 8: Commit** `git commit -m "examples/molecular-ai: a library generated from real papers, synced, searched, walked and graphed offline"`

---

### Task 5: Measurements into the spec, gate, PR

- [ ] Spec §6: paste the semantic-vs-substring results, the generator's S2 wall time, and kg-report with/without references (delete the references from a copy of the example DB and re-run `kg-report`).
- [ ] `uv run --frozen pre-commit run --all-files` green.
- [ ] `git push -u origin feat/library-graph`; `gh pr create --base feat/library-store` with the spec link, the 47→48 note, the measured numbers, and the generated-with line.

## Self-review

Spec §1 → Task 1; §2 → Task 2; §3 → Task 3; §4/§5 → Task 4; §6 → Task 5; §7 tests are in each task. No placeholders: every step names its code or the exact command. Types: `ReferenceRecord.tags` (Task 1) used by Task 2's tests; `Related.to_row()` keys match the tool's `empty=` (Task 3); `_hit` and `lookup_row` are spec 1's.
