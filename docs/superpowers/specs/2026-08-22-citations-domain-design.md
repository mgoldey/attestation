# Citations domain

**Date:** 2026-08-22
**Status:** implemented 2026-08-23 in `80c1a4f`. Deviations: CSL rendering
was already out of scope here and remains unbuilt; the `web` reader queries
CrossRef only (arXiv and Semantic Scholar would be additional readers behind
the same port, and nothing needed them yet).
**Roadmap:** spec 2 of `2026-08-21-architecture-roadmap.md`
**Depends on:** nothing. See "What changed since the roadmap".

## What changed since the roadmap

The roadmap justified this spec twice: as a domain worth having, and as "the
proof that spec 1's boundaries hold — if adding a domain is hard, spec 1 was
wrong." The second justification is void. Spec 1 (onion refactor) was
superseded on 2026-08-21 after two reviews found that a repository whose method
count tracks its call-site count is a rename, not an abstraction. There is no
repository layer to prove anything about.

What survived spec 1 is what this spec actually builds against: the four
namespaced MCP modules (`mcp/feed.py`, `mcp/knowledge.py`, `mcp/provenance.py`,
`mcp/symbolic.py`), the `@tool` decorator in `mcp/_tool.py` that owns
connection, user lookup and both envelopes, and `ports.py`'s rule for when a
Protocol is earned. This spec inherits those and adds nothing structural.

## Problem

`claims.py` can verify that a number in prose matches a run in the ledger. It
cannot verify that a *citation* in prose points at a real paper, and it has no
way to express "this claim is supported by someone else's published result"
as distinct from "this claim is supported by my run".

Meanwhile the knowledge graph is built from tags on feed items — things the
user read. What the user *cited* is a stronger signal than what they read, and
it is sitting in a `.bib` file or a Zotero library that this project cannot
see.

Both gaps are the same missing thing: a reference has no representation here.

## Scope

One `CitationPort`, four readers, and a resolver that records where each record
came from.

### The port

```python
@runtime_checkable
class CitationPort(Protocol):
    """A source of bibliographic records, keyed by citation key or identifier."""

    name: str  # "zotero" | "bibtex" | "web" — recorded on every record it returns

    def lookup(self, key: str) -> Reference | None:
        """One record by its key, or None if this source does not have it."""
        ...

    def all(self) -> Iterator[Reference]:
        """Every record this source can enumerate. Network readers raise
        NotImplementedError: you cannot enumerate CrossRef."""
        ...
```

`ports.py` states the rule: *"A protocol earns its place when a second
implementation genuinely exists."* Three do here, with genuinely different
backends (a SQLite file, a text format, an HTTP API), and the reason to name
the shape is that the resolver must treat them uniformly while recording which
one answered. That is a real abstraction, not a rename.

`all()` returning `Iterator` rather than `list` is load-bearing: a Zotero
library of 8,000 items should not be materialised to answer "is this key
present".

### Reference

```python
@dataclass(frozen=True)
class Reference:
    key: str               # citation key: "vaswani2017attention"
    title: str
    authors: list[str]
    year: int | None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    source: str = ""       # which reader produced this — never blank in practice
    fetched_at: str | None = None  # ISO date; None for offline sources
```

`source` and `fetched_at` are the provenance pair. A record from disk has
`source="bibtex"` and `fetched_at=None`; a record from CrossRef has
`source="web"` and a date. This is the mechanism that makes the offline
exception inspectable rather than merely documented.

### The four readers

| reader | backend | network | notes |
|---|---|---|---|
| `zotero` | `~/Zotero/zotero.sqlite`, read-only | no | Also the local HTTP API at `127.0.0.1:23119` when the app is running |
| `bibtex` | `.bib` files on disk | no | Read and write. Pairs with `claims.py` |
| `web` | arXiv, CrossRef, Semantic Scholar by DOI/arXiv ID | **yes** | Opt-in, default off. See below |
| `csl` | Pandoc/CSL rendering | no | **Not a reader.** See below |

**Zotero must open its SQLite read-only and must tolerate a locked file.**
Zotero holds an exclusive lock while running. Opening with
`file:...?mode=ro&immutable=1` reads a live library without corrupting it; if
that fails, fall back to the HTTP API, and if that also fails, return no
records rather than raising — a missing Zotero is an absent source, not an
error. On this machine there is no Zotero install, so this reader ships with
a synthetic fixture and is verified against the documented schema
(`items`/`itemData`/`itemDataValues`/`fields`), not against a real library.
**That limitation goes in the reader's docstring**, because a reader tested
only against a fixture its own author wrote is exactly the failure mode
`CLAUDE.md` records for this repo ("tests that pass against the bug they were
written to catch").

**CSL is not a reader and does not implement the port.** The roadmap listed it
as a fourth reader; it renders, it does not resolve. Rendering is a
presentation concern and lives at the presentation edge — a CLI subcommand, not
a `CitationPort`. Listing it as a reader would have put formatting inside the
domain.

### The offline-guarantee exception

`CLAUDE.md` states: *"Local models via Ollama; nothing leaves the machine."*
The `web` reader breaks that. The terms:

1. **Default off.** Enabled only by `ATTEST_CITATION_WEB=1`, checked at reader
   construction, not at call time — so a disabled reader cannot be coaxed into
   one request by an unusual code path.
2. **`CLAUDE.md` gains an explicit note** naming every tool that can reach the
   network. A guarantee with a documented exception is honest; a guarantee that
   quietly stopped holding is not.
3. **Per-record provenance**, via `source` and `fetched_at` above. Any answer
   can be asked where it came from.
4. **A disk cache**, so a DOI is fetched once. Cached records keep their
   original `fetched_at` — the cache does not launder a network record into
   something that looks local.

## Open questions from the roadmap, now answered

**Do citations become KG nodes or a sibling store?** Sibling store. The graph
is built by `kg.build_graph(assignments)`, which is a **pure** function over
`(item_id, tag)` pairs — `CLAUDE.md` records that purity as load-bearing, and
`kg_nodes`/`kg_edges`/`kg_meta` were deleted on 2026-08-21 precisely because
nothing read them. Putting references into the graph would mean either
inventing tags for them (polluting a vocabulary derived from read items) or
adding a second node type (ending `build_graph`'s single-type purity). A
`references` table with an optional `item_id` FK gets the join without either
cost: "papers I cited that I also read" is a query, not a graph rewrite.

**Does `claims.py` gain a cite-key verdict?** Yes, one new `VerdictKind`:
`UNCITED`. `VerdictKind` is already a closed `StrEnum` and `check_claim`
already returns a `Verdict` carrying `matched`, so the shape exists. `UNCITED`
means "this claim names a citation key that no configured source has" — a
lint, matching how `coverage()` lints numbers no claim covers. It is
deliberately NOT "this claim contradicts the cited paper": comparing a prose
assertion against a paper's contents needs a model, and every verdict in this
module is currently deterministic. Keep it that way.

**Cache policy.** Content-addressed by DOI/arXiv ID, never expiring. A
published paper's metadata does not change, and an expiring cache would turn
one network call into a recurring one — the opposite of the guarantee.

## Tool surface

Four tools under a new `cite.*` namespace, following the existing rules from
`test_architecture.py`: a tool never repeats its namespace, and every tool is
registered through `@mcp.tool(name=...)`.

| tool | returns |
|---|---|
| `cite.lookup` | one `Reference` by key or DOI, with its `source` |
| `cite.search` | references matching a title/author string, offline sources only |
| `cite.check` | claim keys with no matching reference — the `UNCITED` lint |
| `cite.sources` | which readers are configured, and which are network-backed |

`cite.sources` exists because the answer to "did this leave my machine" must be
askable from the same surface that did the leaving.

**Which surface gets them, and the one that splits.** `AGENT_SURFACES`' own
comment states the rule that decides this: *"Claims live with `runs`, not with
the graph. `runs.claims_check` verifies numbers in Markdown AGAINST recorded
runs; separating them would put a claim checker in a session that cannot see
what it checks against."*

`cite.lookup`, `cite.search` and `cite.sources` are reference resolution and
belong with `knowledge` — they pair with the reading graph, and a `knowledge`
session is the exploratory read-only one.

`cite.check` is a claim checker and by that rule belongs with `provenance`,
beside `runs.claims_check`. A session that can lint uncited claims but cannot
see the runs the *other* claims are checked against would report half a
document's problems and look complete. So `cite.check` registers under both
`knowledge` and `provenance` — the surface table already supports a tool
appearing in two surfaces (`knowledge` contains `feed.search` for exactly this
reason), and this is the same case: one tool a second surface genuinely needs.

## Success criteria

- `ATTEST_CITATION_WEB` unset ⇒ no socket is opened. Verified by a test that
  monkeypatches `httpx` to raise on any call and drives the full surface.
- Every `Reference` returned carries a non-empty `source`.
- Zotero reader returns `[]` against a locked/absent library, never raises.
- `claims.check` gains `UNCITED` without changing any existing verdict.
- `cite.lookup`/`cite.search`/`cite.sources` register under
  `ATTEST_TOOLS=knowledge` only; `cite.check` under `knowledge` and
  `provenance`, matching how `feed.search` already crosses a surface.
- Full pre-commit gate green.

## Out of scope

Writing to Zotero. Rendering bibliographies (that is the CSL CLI subcommand,
specified separately if wanted). Any model call — this domain is deterministic
end to end, and the `UNCITED` verdict is scoped narrowly to keep it so.
