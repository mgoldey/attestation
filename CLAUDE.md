# attestation

Auditable research provenance, fully local: experiment runs, verifiable claims,
a reading knowledge graph, symbolic derivations, and a personalized science feed.
Exposed as 35 MCP tools plus a small HTMX web UI and an `attest` CLI.

## Docs Index

IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any tasks in this repo. When working on a service or module, read the relevant doc files listed below before writing code.

```
[Project Docs Index]|root: .
|.:{README.md,DEMO.md,CLAUDE.md,pyproject.toml,datasette.yml,.pre-commit-config.yaml,.env.sample,LICENSE}
|src/attestation:{__init__.py,cli.py,mcp_server.py,server.py,db.py,rank.py,embed.py,llm.py,ingest.py,feeds.py,features.py,explain.py,kg.py,ledger.py,claims.py,corpus.py,symbolic.py,symbolic_ops.py,install.py,feeds.toml,feed_candidates.toml,kg_aliases.toml}
|src/attestation/ledger_adapters:{__init__.py,generic.py}
|src/attestation/skills/research-provenance:{SKILL.md,scripts/setup.sh}
|tests:{conftest.py,test_cli.py,test_db.py,test_rank.py,test_rank_relevance.py,test_embed.py,test_llm.py,test_ingest.py,test_feeds.py,test_features.py,test_explain.py,test_server.py,test_mcp_server.py,test_digest.py,test_kg.py,test_kg_algorithms.py,test_kg_mcp.py,test_ledger.py,test_ledger_adapters.py,test_ledger_mcp.py,test_claims.py,test_symbolic.py,test_symbolic_ops.py,test_symbolic_mcp.py,test_install.py,test_install_e2e.py,test_skill_files.py}
|docs:{recommendation-refinements.md,hermes-agent-plugin-research.md}
|docs/superpowers/specs:{2026-08-04-hermes-rss-design.md,2026-08-05-feature-extraction-design.md,2026-08-05-hermes-install-design.md,2026-08-05-llm-client-extraction-design.md,2026-08-06-knowledge-graph-design.md,2026-08-06-persona-and-feed-mcp-tools-design.md,2026-08-06-sympy-mcp-tools-design.md,2026-08-06-web-ui-tidy-design.md,2026-08-12-claim-checker-design.md,2026-08-12-run-ledger-design.md,2026-08-13-digest-design.md,2026-08-20-corpus-ledger-design.md}
|docs/superpowers/plans:{2026-08-04-hermes-rss.md,2026-08-05-feature-extraction.md,2026-08-05-hermes-install.md,2026-08-05-llm-client-extraction.md,2026-08-06-knowledge-graph.md,2026-08-06-persona-and-feed-mcp-tools.md,2026-08-06-sympy-mcp-tools.md}
```

Every feature has a design spec in `docs/superpowers/specs/` written before the
code. Read the spec before changing a subsystem — it records why, not just what.

### Key API Patterns (read before modifying)

```
|Entry points: cli.py:main()→`attest`|mcp_server.py:main()→`attest-mcp`|server.py:create_app()→FastAPI+HTMX @ 127.0.0.1:8899
|MCP surface: 35 @mcp.tool()s in mcp_server.py|each tool pairs with _<name>_impl() kept FastMCP-free so tests import it directly
|Ranking: rank_items(conn,embedder,user_id,since_days=14,*,exclude_clicked=True)→list[RankedItem]|blend_weight(n_clicks) mixes click_ranks vs profile_rank|classifier_probs() returns None on single-class history (guard) → order is embedding-only
|Ranking honesty: _ranking_quality() reports classifier_active + caveat|surface it rather than letting a reader assume the ranker learned something
|Candidates: _candidate_items(conn,user_id,since_days,*,exclude_clicked)|since_days=None + exclude_clicked=False = search_feed semantics (older/already-rated items are legitimate hits)
|Storage: db.py SQLite + sqlite-vec|tables: users,feeds,items,clicks,explanations,item_features,item_tags,kg_nodes,kg_edges,kg_meta,runs,run_metrics,corpora,corpus_splits
|DB path: resolve_db_path() precedence = explicit --db → RSS_DB env → ~/.hermes/skills/science-recommendations/data/hermes.db (only if it exists) → ./hermes.db
|Feeds: DB is source of truth; feeds.toml seeds first ingest only (sync_feeds uses INSERT OR IGNORE, no-op afterwards)|add_feed is register-only — fetch happens on next ingest, never inline
|Graph: kg.build_graph() derives fresh from item_tags every read|concepts = tags with uses ≥ MIN_TAG_USES(2), edges = co-occurrence ≥ MIN_EDGE_WEIGHT(2)|order is load-bearing: canonical() aliases → frequency filter → co-occurrence|stored kg_nodes/kg_edges are advisory; is_stale() never changes a read tool's answer
|Ledger: ledger.scan()→RunRecord/Metric from artifacts|compare() emits _caveats() rather than silent verdicts|adapters in ledger_adapters/ read nested result structures
|Corpus: corpus.detect_in_source() reads the corpus from driver-script syntax (AST), not from a model — result artifacts record the model exhaustively and the data not at all|runs.corpus_id links a run to its corpus; compare() guards arms that cross one rather than ranking losses from different tasks
|Claims: claims.parse_file()→find_claims()→check_claim()→Verdict|coverage() lints numbers in prose no claim covers (negatives included)
|LLM: llm.py ChatClient/EmbeddingClient→Ollama-compatible @ DEFAULT_BASE_URL|DEFAULT_CHAT_MODEL=gemma4:e2b-it-q4_K_M, DEFAULT_EMBED_MODEL=embeddinggemma|hermes3 workaround is scoped to hermes3 only
|Embeddings: embed.py DOC_PROMPT vs QUERY_PROMPT are asymmetric — index with doc, search with query|truncate_normalize() before storing
|Reliability contract: explain.py + rank.py catch broad exceptions on purpose (BLE001 per-file-ignored in pyproject)|ranking never waits on explanations; a cold Ollama degrades to a cached vector, never a 500
|No LLM in composition tools: digest/runs_compare return structure, never prose — the caller is a model
```

<!-- end docs index -->
## Working here

- **`uv run pytest` works. If it ever stops, suspect the venv, not uv.** This
  repo spent months believing `uv run` was broken under a scrubbed environment
  ("No module named 'attestation'", later "Failed to spawn: `pytest`") and
  worked around it by calling `.venv/bin/...` everywhere. The real cause was a
  stale `.venv/`: it was built when the project lived at `~/hermes-rss`, and
  26 of its console scripts still carried `#!/home/matt/hermes-rss/.venv/bin/
  python`. That interpreter no longer existed, so exec failed with ENOENT and
  uv reported it as a spawn failure. `attest`, `ruff` and `ty` were regenerated
  after the rename and kept working, which is what made the failure look
  specific to pytest. `uv sync` rebuilt the venv and fixed all 26.
  The lesson generalizes: a tool that works one way and not another usually
  means broken local state, not a tool that needs routing around.
- Gates: `uv run --frozen pre-commit run --all-files` (ruff format, ruff check,
  ty, uv.lock sync, full pytest). Hooks call `uv run --frozen ...` rather than
  `.venv/bin/...`: `.venv/` is gitignored, so those paths are missing on a
  fresh clone and on a CI runner using astral-sh/setup-uv, which is the
  environment `.github/workflows/ci.yml` runs in. `--frozen` keeps a hook from
  mutating `uv.lock` as a side effect.
- The pytest hook is ~70s and worth it: this repo's recurring failure mode has
  been tests that pass against the bug they were written to catch. CI runs the
  same five gates on Linux and macOS across Python 3.12 and 3.13, plus a wheel
  smoke test, so a bypassed hook is caught on push rather than never.
- Line length 100; ruff lint selects `E,F,W,I,BLE,RUF100`. `RUF100` reports a
  `noqa` that no longer suppresses anything — the per-file-ignores in
  `pyproject.toml` spent this repo's first release pointing at `src/hermes/`,
  a path that had not existed since the rename, silently enforcing nothing.
- Requires Python ≥3.12. Local models via Ollama; nothing leaves the machine.
