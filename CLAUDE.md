# attestation

Auditable research provenance, fully local: experiment runs, verifiable claims,
a reading knowledge graph, symbolic derivations, and a personalized science feed.
Exposed as 46 MCP tools plus a small HTMX web UI and an `attest` CLI.

## Docs Index

IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any tasks in this repo. When working on a service or module, read the relevant doc files listed below before writing code.

```
[Project Docs Index]|root: .
|.:{README.md,DEMO.md,CLAUDE.md,pyproject.toml,datasette.yml,.pre-commit-config.yaml,.env.sample,LICENSE}
|src/attestation:{__init__.py,implicit.py,personas.py,cli.py,mcp_server.py,server.py,db.py,rank.py,embed.py,llm.py,ingest.py,feeds.py,features.py,explain.py,kg.py,ledger.py,claims.py,citations.py,corpus.py,symbolic.py,symbolic_ops.py,simulate.py,ports.py,install.py,emit.py,feeds.toml,feed_candidates.toml,kg_aliases.toml}
|src/attestation/mcp:{__init__.py,_tool.py,_shared.py,ask.py,citation.py,disclosure.py,feed.py,knowledge.py,personas.py,provenance.py,routing.py,subscriptions.py,symbolic.py}
|evals:{run_tagging_eval.py,tagging_cases.json}|examples:{README.md,workspace/}
|src/attestation/ledger_adapters:{__init__.py,generic.py}
|src/attestation/skills/research-provenance:{SKILL.md,scripts/setup.sh}
|tests:{conftest.py,test_agent_surfaces.py,test_architecture.py,test_ask_routing.py,test_citations.py,test_claims.py,test_cli.py,test_db.py,test_digest.py,test_embed.py,test_emit.py,test_examples.py,test_explain.py,test_features.py,test_feeds.py,test_implicit_feedback.py,test_ingest.py,test_install.py,test_install_e2e.py,test_kg.py,test_kg_algorithms.py,test_kg_mcp.py,test_ledger.py,test_ledger_adapters.py,test_ledger_mcp.py,test_llm.py,test_mcp_server.py,test_persona_autocreate.py,test_persona_hygiene.py,test_ports.py,test_rank.py,test_rank_relevance.py,test_response_size.py,test_search.py,test_server.py,test_simulate.py,test_skill_files.py,test_symbolic.py,test_symbolic_mcp.py,test_symbolic_ops.py,test_tool_envelope.py,test_tracker_adapters.py}
|docs/architecture:{README.md,agent-flows.html}
|docs:{recommendation-refinements.md,hermes-agent-plugin-research.md}
|docs/superpowers/specs:{2026-08-04-hermes-rss-design.md,2026-08-05-feature-extraction-design.md,2026-08-05-hermes-install-design.md,2026-08-05-llm-client-extraction-design.md,2026-08-06-knowledge-graph-design.md,2026-08-06-persona-and-feed-mcp-tools-design.md,2026-08-06-sympy-mcp-tools-design.md,2026-08-06-web-ui-tidy-design.md,2026-08-12-claim-checker-design.md,2026-08-12-run-ledger-design.md,2026-08-13-digest-design.md,2026-08-20-corpus-ledger-design.md,2026-08-21-architecture-roadmap.md,2026-08-21-onion-refactor-design.md,2026-08-21-tool-surface-design.md,2026-08-22-agent-surfaces-design.md,2026-08-22-citations-domain-design.md,2026-08-22-tracker-adapters-design.md,2026-08-22-swarm-refutation.md,2026-08-22-config-emitters-design.md}
|docs/superpowers/plans:{2026-08-04-hermes-rss.md,2026-08-05-feature-extraction.md,2026-08-05-hermes-install.md,2026-08-05-llm-client-extraction.md,2026-08-06-knowledge-graph.md,2026-08-06-persona-and-feed-mcp-tools.md,2026-08-06-sympy-mcp-tools.md}
```

Every feature has a design spec in `docs/superpowers/specs/` written before the
code. Read the spec before changing a subsystem — it records why, not just what.

### Key API Patterns (read before modifying)

```
|Entry points: cli.py:main()→`attest` (incl. `attest reload` after editing MCP code)|mcp_server.py:main()→`attest-mcp`|server.py:create_app()→FastAPI+HTMX @ 127.0.0.1:8899
|Skills vs tools in Hermes' prompt: 68 skills cost 7KB (index only, ~70B each; the 25KB body loads on invoke), 67 tool schemas cost 85KB IN FULL every turn — hide TOOLS, not skills|`filament` alone is 40 tools / 39.6KB, 46% of the tool budget
|Skill DESCRIPTIONS do collide: with arxiv/blogwatcher/weights-and-biases/research-paper-writing listed beside the feed tools, routing went 6/6 -> 3/6 on gemma4:e2b|per-skill attribution at n=6 was noise (dropping arxiv scored WORSE), so only the aggregate is trustworthy
|Agent surfaces: ATTEST_TOOLS=feed|provenance|knowledge|symbolic restricts registration (feed 22 / provenance 9 / knowledge 12 / symbolic 9 with ATTEST_EXPAND=1; without it each shows 2 — the ask router plus one companion, which is progressive disclosure, not the surface size); unset serves all 50|these counts MOVE — re-measure rather than quoting: `ATTEST_TOOLS=feed ATTEST_EXPAND=1 uv run python -c "from mcp.server.fastmcp import FastMCP; from attestation.mcp import register_all; import asyncio; m=FastMCP('x'); register_all(m); print(len(asyncio.run(m.list_tools())))"`|a typo RAISES rather than serving everything|~/.hermes/config.yaml has attestation-<surface> entries, disabled by default
|Personas AUTO-CREATE on read (`@tool(autocreate_user=True)` on list/search/digest/read): refusing an unknown name and listing valid ones is what TAUGHT agents to call persona_create, and it grew a duplicate 'Matthew Goldey' days after that persona was merged away|the new reader is seeded from the corpus's top tags (never empty — an empty interests string embeds to nothing) and asked what topics to monitor|destructive tools still refuse
|SKILL.md says: never ask for a persona NAME (it is whatever was passed), ask what they read about; and answer "what can you do" by CALLING feed.list, not by listing tools
|mcp/routing.py: 4 deterministic routers (question -> tool, no model call), surfaced by mcp/ask.py returning a Pydantic Answer, so MCP emits a real outputSchema|MEASURED on gemma4:e2b over 15 turns x3: routed 13/15, flat-37 8/15, LLM swarm 7.3/15 at 2x latency — the swarm is REFUTED for routing
|NO catch-all destination: an early `doctor` tool absorbed 3 of 4 remaining misses|an ambiguous question returns options, never a default
|MCP surface: 46 tools by default (2026-08-23) NAMESPACED as feed.*(21) sym.*(8) runs.*(7) kg.*(6) cite.*(4) — the four `<surface>.tools` disclosure tools register ONLY under ATTEST_TOOLS, since with everything served they claimed to hide tools sitting beside them via @mcp.tool(name=...) in mcp/{feed,knowledge,provenance,symbolic,ask,citation,subscriptions}.py|a tool never repeats its namespace (kg.path, not kg.kg_path) — two tests in test_architecture.py enforce both rules|the flat names were removed outright, no aliases; mcp_server.py is a ~95-line entry point + one-release `_<name>_impl` aliases|mcp/_tool.py's @tool owns the ritual: connection, user lookup, and BOTH envelopes — a body returns only what it computed|expected refusals `raise ToolError(msg)` (verbatim to caller); anything else is a bug (logged, generic message)|`empty={...}` makes a failure envelope structurally match its success envelope
|Search: search_feed queries sqlite-vec with embed_query() then blends with profile rank (QUERY_WEIGHT=0.75)|RELEVANCE_FLOOR=0.90 is RELATIVE to the best hit for that query — absolute cutoffs fail because top similarity varies 0.44-0.62 by query|a literal hit is a BOOST not a floor: flooring made all 711 "llm" matches tie
|Feedback: clicks.source = ui|agent|bootstrap|simulated|implicit, and provenance decides what a row may be used for|bootstrap labels are a linear threshold on the SAME embedding the classifier trains on → tautological, excluded by evaluate_user|simulated = a chat model reacting to TEXT as the persona, independent of the vector, so trainable
|Signal scarcity is the core problem: MEASURED 2026-08-23, 11 human clicks (8 ui + 3 agent) on 3 DAYS across 19, against 5265 items — feedback that needs a gesture does not arrive|implicit.py harvests engagement as weak positives: explanation requests AND `feed.read`, via the `engagement` table (migration 005)|SKILL.md tells the agent to extract verdicts from ordinary discourse, since users never press buttons
|Only POSITIVES are inferred: no behaviour reliably means "not useful", and inferring rejection from silence poisons the class the ranker is starving for
|simulate.py: Reaction asks for `confidence` (how sure), NOT `strength` — the first version asked "how strongly you feel" and a correct rejection came back at 1 and was filtered as indifference, discarding every negative
|Ranking: rank_items(conn,embedder,user_id,since_days=14,*,exclude_clicked=True)→list[RankedItem]|blend_weight(n_clicks) mixes click_ranks vs profile_rank|classifier_probs() returns None on single-class history (guard) → order is embedding-only
|Ranking honesty: _ranking_quality() reports classifier_active + caveat|surface it rather than letting a reader assume the ranker learned something
|Candidates: _candidate_items(conn,user_id,since_days,*,exclude_clicked)|since_days=None + exclude_clicked=False = search_feed semantics (older/already-rated items are legitimate hits)
|Storage: db.py SQLite + sqlite-vec|tables: users,feeds,items,clicks,engagement,explanations,item_features,item_tags,runs,run_metrics,corpora,corpus_splits (12 APPLICATION tables — kg_* dropped 2026-08-21, and migration 004 drops them from existing DBs too)|plus item_vectors (a vec0 VIRTUAL table) and its four shadow tables, so a fresh file has 17 — Datasette refuses to open the DB without the sqlite-vec extension loaded
|DB path: resolve_db_path() precedence = explicit --db → RSS_DB env → ~/.hermes/skills/science-recommendations/data/hermes.db (only if it exists) → ./hermes.db
|Feeds: DB is source of truth; feeds.toml seeds first ingest only (sync_feeds uses INSERT OR IGNORE, no-op afterwards)|add_feed is register-only — fetch happens on next ingest, never inline
|Graph: kg.build_graph(assignments) is PURE — takes (item_id, tag) pairs, not a conn; kg.tag_assignments(conn) reads them|concepts = tags with uses ≥ MIN_TAG_USES(2), edges = co-occurrence ≥ MIN_EDGE_WEIGHT(2)|order is load-bearing: canonical() aliases → frequency filter → co-occurrence, tested DB-free|kg_nodes/kg_edges/kg_meta were DELETED 2026-08-21: nothing read them, and the 8 kg tool answers were byte-identical before and after
|Ledger: ledger.scan()→RunRecord/Metric from artifacts|compare() emits _caveats() rather than silent verdicts|adapters in ledger_adapters/ read nested result structures
|Corpus: corpus.detect_in_source() reads the corpus from driver-script syntax (AST), not from a model — result artifacts record the model exhaustively and the data not at all|runs.corpus_id links a run to its corpus; compare() guards arms that cross one rather than ranking losses from different tasks
|Claims: claims.parse_file()→find_claims()→check_claim()→Verdict|coverage() lints numbers in prose no claim covers (negatives included)|optional `cite=<key>` field adds VerdictKind.UNCITED via check_citations() — a LINT ("no source has this key"), never "the paper does not support this", which would need a model
|OFFLINE GUARANTEE + ITS ONE EXCEPTION: everything is local EXCEPT citations.WebReader, which queries CrossRef and exists only when ATTEST_CITATION_WEB is set — read at Resolver CONSTRUCTION, never at call time, so a disabled reader cannot be coaxed into a request|every Reference carries source + fetched_at (None = from disk); cite.sources reports `offline: true/false`|search() NEVER fans out to a network reader even when armed — the flag alone cannot catch that bug, since with it unset there is nothing to fan out to
|Ports: ports.py ChatPort/EmbeddingPort/EmbedderPort are structural Protocols — NO repository protocol, deliberately (see superseded onion spec)|EmbedderPort is separate from EmbeddingPort because embed.py's doc/query prompts are asymmetric|LLM: llm.py ChatClient/EmbeddingClient→any OpenAI-compatible server (zero ollama refs in library code; LLM_BASE_URL points anywhere) @ DEFAULT_BASE_URL|DEFAULT_CHAT_MODEL=gemma4:e2b-it-q4_K_M, DEFAULT_EMBED_MODEL=embeddinggemma|hermes3 workaround is scoped to hermes3 only
|explain.py: ONE model call, not two — profile synthesis runs only when a persona has NO interests text|measured on gemma4:e2b, synthesis returned meta-description ("This list of recently useful titles covers...") and produced VAGUER explanations than the stored interests string, for +2.1s|the explain prompt's refusal clause is load-bearing: without it the model claimed a termite-feed paper matched "advanced topics like AI"
|llm.py `_first_json_object()` recovers from a reply with trailing junk or a prose preamble — gemma4:e2b emitted two objects and json.loads raised straight out of chat_json|a reply with NO object still raises: recovery must not shade into invention
|MCP servers NEVER reload: one is spawned per session and holds that code until it dies|found the gateway and a Claude session both running code 5 commits stale|`attest reload` SIGTERMs every live attest-mcp; respawn is LAZY (measured: nothing for 10s, back on the next tool call)|`hermes mcp test` does NOT catch staleness — it spawns a fresh process, so it reports the code on disk
|Warmup: `attest warmup` holds models for OLLAMA_KEEP_ALIVE (default 30m), NOT forever — keep_alive=-1 pinned 5.4GB until the year 2318 and caused OOM kills on a 23GB box
|Response size: feed.list defaults to limit=4, capped at 13 (both MEASURED against a 7000-char ceiling in test_response_size.py, and asserted against these docs in test_architecture.py); `score` was REMOVED (a rank within a candidate set, not comparable across calls) and tags cap at 3 with n_tags reporting the true count|a 2B model could not render a 10-item payload and looped truncate-apologise-redump
|Embeddings: embed.py DOC_PROMPT vs QUERY_PROMPT are asymmetric — index with doc, search with query|truncate_normalize() before storing
|Reliability contract: 7 inline `# noqa: BLE001` sites (cli, install, symbolic, rank, ingest, citations, mcp/provenance), each carrying its reason — there is NO per-file-ignores section in pyproject; the count is asserted in test_architecture.py, and line numbers are deliberately NOT recorded here because citing them is how the previous version rotted|rank.py's is a specific policy, not a swallow: embedder down + warm cache serves stale, cold cache raises|ranking never waits on explanations; a cold Ollama degrades to a cached vector, never a 500
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
