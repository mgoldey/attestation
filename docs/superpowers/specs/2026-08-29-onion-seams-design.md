# Onion seams: nine cuts three lenses agree on

**Date:** 2026-08-29
**Status:** implemented 2026-08-29 — nine commits `d2526ed..9cd7e70` (five
parallel file-owner tasks, then the ports migration and the ranking golden
path), plus the final-review fix wave `ed6ac5c`; deviations and the revisited
success criteria are recorded below. The final whole-branch review
(Feathers/Bernhardt lens) found no Critical issues and three Important ones,
all fixed and re-verified: the ports test missed `from attestation import
llm`; an ingest log line pretended to know an unset URL; and three seam tests
passed against the very mutants this spec named for them — recorded in
`docs/measurement-lessons.md` §7.
**Depends on:** the tool-surface design (`2026-08-21-tool-surface-design.md`),
which superseded the full onion (`2026-08-21-onion-refactor-design.md`) and
left the standing rule this spec applies.
**Base:** `main` at `787823d`, all gates green.

## Situation, complication, answer

The full onion — ports, repositories, fakes, service facades — was designed
and superseded on the same day: a simple-versus-easy review found it
relocated the tangle rather than removing it, and that the one real braid
(`build_graph` taking a connection) cost one signature change. The rule it
left is the one `CONTRIBUTING.md` now states to newcomers: **a seam is
added when a test can name what it decomplects**, never for symmetry.

That rule has since been applied once. Fifteen modules still import
`sqlite3`, forty domain signatures take a `conn`, and every one of the 38
tests in `tests/test_rank.py` builds a database before it can rank three
items. The question is not whether to build the onion — that is settled —
but which braids remain that the rule already condemns.

Rather than one reviewer's taste, three independent reviews were run over
the same ten modules, each through a named lens, and a seam enters this
spec only where the lenses converge or where one lens names a test that
cannot be written today. The answer is nine seams, every one the same
shape — a thin reader beside a pure function, or one named function
replacing three copies — and not one new layer.

## Method

Three reviewers (read-only, in parallel, same brief and measurements,
`main` at `787823d`) each read `rank.py`, `ledger.py`, `features.py`,
`personas.py`, `feeds.py`, `mcp/feed.py`, `mcp/personas.py`, `explain.py`,
`simulate.py`, `server.py`, `db.py`, `cli.py`, and the two onion specs:

| lens | what it looks for | found |
|---|---|---|
| Hickey (*Simple Made Easy*) | braids: two roles interleaved so neither can be tested or replaced alone; refuses layers that only rename | 6 braids, all query + policy in one function |
| Hettinger (*Beyond PEP 8*, "there must be a better way") | the seam Python makes cheap: functions over plain data, one name for a repeated sequence, no private names crossing modules | 6 smells, 3 duplicated sequences |
| Karpathy (nanoGPT, software 2.0) | can one person read the loop, run a stage on a list of dicts, and measure it with a script that ships | 5 gaps, 1 prompt outside the one-renderer rule |

**Convergence rule.** A seam is accepted when two lenses name the same
braid, or when one lens names it together with a DB-free or model-free
test that cannot be written today. A finding one lens made that no test
would name is recorded as refused, with the reason. The three reports are
the evidence; their "leave alone" lists agree on the deferred imports, the
`@tool` ritual, `rank.py`'s stale-cache policy, `kg.build_graph`, the
routers, and `ledger.py`'s length.

Measurements quoted to reviewers were code lines (non-blank, non-comment);
one reviewer re-measured with `wc -l` and reported drift. There was none —
`ledger.py` is 839 code lines and 1149 total.

## The nine seams

Each seam names the braid, the smallest change, and the test that could
not be written before it. No caller-visible dict shape changes; no
behaviour changes; existing tests stay as they are and new ones are added.

### 1. `rank.rank_rows` — the blend runs on rows, not a connection (Karpathy + Hickey)

`rank_items(conn, embedder, user_id, …)` fetches candidates with one
six-condition SQL string, then blends profile cosine, the click
classifier and the preference term in ~60 lines of numpy that read the
rows only as `id/title/url/summary/source/embedding`. Add
`rank_rows(rows, profile_vec, click_rows, pref) -> list[RankedItem]`
holding everything from the `np.stack` to the return; `rank_items`
becomes fetch + `rank_rows`. `classifier_probs` takes `click_rows`
instead of pulling `_click_training_data(conn)`.

Test: `test_rank_rows_blends_three_dict_rows_without_a_database` — the
first rank test with neither `tmp_path` nor `fake_embedder`. The
tie-averaged neutral no-op (`avg_ranks`) is the mutation target.

Risk: `_preference_ready`'s downvote-only gating and `blend_weight` stay
callable exactly as now. `_candidate_items` keeps its SQL — its WHERE
clauses *are* the candidate policy, measured at 87 ms versus 4.93 s.

### 2. `rank._ranking_quality(counts, by_source)` — caveat selection beside its pure siblings (Hickey)

`ranking_quality(conn, user_id)` runs one GROUP BY and then ~80 lines of
caveat precedence — the 199/1 lopsided-history case that shipped with no
caveat before it was fixed. Extract the reducer-plus-selection as a pure
function next to `_blend_disclosure` and `_provenance_caveat`, which
already have that shape.

Test: `test_ranking_quality_flags_near_single_class_history` —
`_ranking_quality({1: 199, 0: 1}, {"ui": 200})` fires the
"almost nothing to contrast against" caveat. Caveat text and join order
are unchanged (`test_response_size.py` budgets them).

### 3. `ledger.compare` — a reader and a decision (Hickey)

`compare()` interleaves five SQL round-trips (family rows, metric
discovery, values by run, `n` by run) with metric selection, direction
lookup, arm building, `_corpus_agreement` and `_all_caveats` — all of
which are already pure. Split into `_compare_rows(conn, family, project)
-> (rows, values_by_run, n_by_run)` and a pure `_compare(rows,
values_by_run, n_by_run, metric, directions) -> dict`; `compare()` is the
two-line composition.

Test: `test_compare_picks_the_majority_metric_with_no_db` — literal dicts
in, the same `winner`/`caveats` shape `test_ledger.py` asserts out.
`scan()`'s single-transaction contract is untouched (it is not in
`compare`).

### 4. `ledger.collapse_to_last(metrics)` — one rule, moved down from presentation (Hettinger)

`mcp/provenance.py:_detail` reaches into `ledger.detail()`'s `metrics`
list and collapses a step series to its last row per metric name, inline.
Name it in `ledger.py` beside `nested_arms`; the `[:MAX_METRIC_ROWS]`
truncation and the "N of M" message stay in `provenance.py`, because that
is presentation budget.

Test: `test_collapse_to_last_keeps_the_last_row_per_metric_name` — a
hand-built list with repeats, first-appearance order preserved.

### 5. `mcp/feed._allocate_digest_budget` — the counted bug gets a DB-free home (Hickey)

`_digest_body` ranks, fetches communities and cached explanations, then
allocates `MAX_DIGEST_ITEMS` across topics and computes `shipped/dropped`
in the same function — the undercount this function's own comment records
("every live persona read '16 item(s)' while shipping 6 to 11"). Extract
`_allocate_digest_budget(grouped, unclustered, per_topic, budget)` beside
the already-pure `_cluster`.

Test: `test_digest_budget_accounts_for_per_topic_truncation` — `shipped`
equals what is actually in the returned topics plus `shown_unclustered`.
Truncation order (topics before unclustered, largest first) is unchanged.

### 6. `mcp/feed._apply_relevance_floor(sims)` — three rounds of live tuning get a regression test (Hickey)

`_semantic_hits` issues the sqlite-vec `MATCH` and applies the
`RELEVANCE_ANCHOR`/`RELEVANCE_FLOOR` policy in the same body. Split the
query (`_vector_search`) from the floor. Floating-point comparison stays
`sim >= best * RELEVANCE_FLOOR`, byte for byte.

Test: `test_relevance_floor_keeps_hits_near_the_top_anchor` —
`{1: .62, 2: .60, 3: .61, 4: .30}` keeps 1–3.

### 7. `features.top_and_bottom_keys` — no private name crosses a module (Hettinger)

`mcp/feed._profile_status` imports `features._key_stats` and
`features._score`. Promote one public function that calls them; the
Laplace smoothing is not reimplemented. This seam also adds a structural
rule: `test_no_mcp_module_imports_a_private_domain_name`, an AST walk
over `mcp/*.py` in the style of `test_no_tool_repeats_its_own_namespace`.

### 8. `personas` — one purge, one annotation (Hettinger + Hickey)

Two braids in one module. (a) The clicks/explanations/users/profile-vector
cleanup is written three times (`mcp/personas.py` twice, `personas.merge`
once) and the copies differ in whether `users` is deleted and when commit
happens. `rank.purge_feedback(conn, user_id, *, delete_user=False)` —
owned by `rank.py`, which owns `forget_profile_vector` — never commits;
each caller keeps its commit where it is, since `merge()` spans one
transaction over several personas. (b) `survey(conn)` issues per-user
queries and computes the nearest-neighbour merge suggestion in the same
loop; split `_survey_rows(conn)` from a pure `_annotate_survey(rows)`.

Tests: `test_purge_feedback_clears_clicks_explanations_and_cache` (an
in-memory sqlite with three inserts — deletes are I/O by nature) and
`test_survey_suggests_nearest_by_interest_overlap` (three literal rows,
no database; `nearest_overlap` rounded to 3 places; no `nearest` key
when there is one persona).

### 9. `feeds` — score purely, and raise instead of returning `ok` (Hickey + Hettinger)

`suggest_feeds` scores candidates by tag overlap inside the function that
ran the queries and loaded the TOML: extract `_score_candidates(liked,
subscribed, candidates, limit)`. And `add_feed`/`remove_feed`/
`preview_feed` predate `@tool` and still hand-build `{"ok": …}`, which
`mcp/subscriptions.py` unwraps with the same three lines four times:
they raise `FeedError` (a `ValueError`) instead, and the MCP layer maps
it to `ToolError` as `mcp/personas.py` already does. `cli.py` is grepped
for `["ok"]` before the change.

Tests: `test_suggest_feeds_ranks_by_tag_overlap_ties_by_title` (no TOML,
no database) and `test_feeds_functions_raise_rather_than_return_ok`.

### 10. `explain.profile_synthesis_messages(titles)` — the fourth prompt joins the one-renderer rule (Karpathy)

`synthesize_profile`'s fallback builds a `{"role": "system"}` message
inline — the only model-facing prompt with no renderer function, so it
cannot be scored the way the other three are. Give it a renderer; the
measured finding that stored interests beat synthesis is unchanged (the
fallback still fires only when interests are empty). No corpus or
optimizer is added by this spec; the renderer is what makes one possible.

Test: `test_synthesize_profile_renders_through_profile_synthesis_messages`.

(Ten numbered cuts, nine seams: 8 holds two in one module.)

## The one measurement this enables

Seam 1 makes the ranking blend the last stage of the model-facing loop
that can run on a list of dicts. `examples/ranking/` becomes a golden path
(prerequisite `none — pure local computation`): twenty hand-built rows
with fixture vectors, a profile vector, a few labelled clicks; it prints
the blended order and, beside it, the classifier-only AUC that
`evaluate_user` reports — the two numbers whose difference today can be
seen only by reading `rank.py:571` and `examples/flows/persona_eval.py`
side by side. `persona_eval.py` keeps its live end-to-end measurement;
this path is the DB-free one.

## Refused, and why

- **Any repository protocol, service class, or `*Repo`/`*Service` name.**
  Every seam above is a function over data; the superseded spec's
  postmortem stands.
- **Splitting `ledger.py` by line count.** It is one argument about when
  a comparison should not be trusted; `test_mcp_domain_modules_stay_small`
  says so in its own docstring.
- **Deleting `EmbeddingPort`** (one implementation, one call site). A
  watch item: remove it if a second embed-only backend never appears; it
  costs 13 lines today.
- **A generator for the three feature-key namespaces** in
  `features._key_stats`/`_item_keys`. No test would name what it
  decomplects; two explicit UNIONs read better than generated SQL. A
  one-line comment cross-referencing the two is allowed.
- **Touching the 81 deferred imports, the `# noqa: BLE001` sites, the
  `@tool` envelope, `_digest`'s `window_days` patch, `db.py`'s manual
  BEGIN/COMMIT, or `server.py`'s per-thread connection.** All three
  reports list these as correct on purpose, each with its reason.

## Typed dependency rules (amendment, 2026-08-29)

Asked whether providers, storage, embedding, graph and MCP code should be
separated by type, the tree gave a two-part answer. `ports.py`'s
Protocols are mentioned only by `ports.py` and `mcp/_shared.py`;
`explain.py`, `features.py`, `ingest.py` and `mcp/feed.py` import the
concrete `attestation.llm` client at module scope, so the provider
boundary exists on paper and the code walks around it. And the MCP layer
writes its own SQL — 15 literals in `mcp/feed.py`, 10 in
`mcp/personas.py`, 1 in `mcp/_tool.py` — which is the outermost layer
doing storage's job. Every other type has exactly one member (one DB,
one embedder, one graph module, one provider).

The decision: **separate by rule now, by directory when a type gets its
second member** — `ledger_adapters/` became a package at five readers,
and that trigger is the precedent. A one-file `providers/` or `storage/`
package is the shallow move the superseded onion was refused for. Two
structural tests in `tests/test_architecture.py` carry the rule instead:

- `test_domain_reaches_models_only_through_ports` — no domain module
  (`explain`, `features`, `ingest`, `simulate`, `rank`, `kg`, `claims`,
  `ledger`, `corpus`, `citations`, `implicit`, `personas`, `feeds`)
  imports `attestation.llm` at module scope. Allowed importers are the
  adapter (`embed.py`) and the composition roots (`cli.py`, `server.py`,
  `install.py`, `mcp_server.py`, `mcp/_shared.py`). Domain functions
  that need a model take a `ChatPort`/`EmbedderPort` argument, which
  makes `ports.py` load-bearing for the first time. Deferred imports
  inside function bodies count: a lazy import of the concrete client is
  still the concrete client.
- `test_mcp_layer_sql_only_ratchets_down` — the count of SQL literals
  across `mcp/*.py` is pinned at its measured value (26 at `787823d`)
  and may only fall; the failure names the file and the new count.
  Seams 5, 6 and 7 lower it; when it reaches zero, `mcp/` is a pure
  presentation type by test rather than by folder.

Moving `explain`/`features`/`ingest` onto the ports is part of this spec
(it is what the first test names); it is a signature change per module —
`chat: ChatPort` in, the client constructed at the composition root —
not a new package. The day `llm.py` gets a sibling, the first test
becomes "nothing outside `providers/` imports it" and the directory
appears; the same for a second store.

## What is tested

- Each seam lands with the test named above, in the existing test module
  for that code, and the test uses no database or model unless the seam
  says otherwise (8a).
- `test_no_mcp_module_imports_a_private_domain_name`,
  `test_domain_reaches_models_only_through_ports` and
  `test_mcp_layer_sql_only_ratchets_down` are new structural rules in
  `tests/test_architecture.py`.
- No existing test changes except to import a renamed private helper;
  the whole suite and all eight gates stay green; tool count, response
  budgets and the CLI reference are byte-identical.
- `examples/ranking/` passes `tests/test_golden_paths.py` by discovery.
- The complexity ratchet may only go down for every touched file.

## Not in scope

- A corpus or optimizer for the profile-synthesis prompt (seam 10 only
  makes it possible).
- Any change to `_candidate_items`'s SQL, `scan()`'s transaction, or
  the caveat wording.
- Re-measuring the ranking numbers in `docs/measurement-lessons.md`; the
  new golden path prints, it does not claim.

## Success criteria

- A newcomer can rank three dict rows, compare three dict arms, allocate
  a digest budget, and apply the relevance floor from a REPL with no
  database file and no model server — and the tests that prove it are the
  ones named here.
- The count of `conn: sqlite3.Connection` parameters in domain signatures
  falls (it is 40 at `787823d`; the number after is recorded here when the
  plan lands, not predicted). **Revisited below**: it rose to 44, and turned
  out to be the wrong proxy for what this criterion wanted -- see "Success
  criteria, revisited."
- No new class, protocol, or module was added; the diff is functions and
  tests.

## Implementation shape

Seven independent tasks by file owner — the typed-dependency rules and
the ports migration of `explain`/`features`/`ingest` are the seventh — — `rank.py` (+ the golden path),
`ledger.py` + `mcp/provenance.py`, `mcp/feed.py` + `features.py`,
`personas.py` + `mcp/personas.py`, `feeds.py` + `mcp/subscriptions.py`,
`explain.py` — each committed by pathspec, then one whole-branch review
by a fourth reader against this spec's "Refused" list.

## Deviations and findings

- `purge_feedback` lives in `personas.py`, not `rank.py` -- Task 8's actual
  landing put the purge beside the other persona-lifecycle functions it is
  called alongside (`mcp/personas.py`'s delete/reset paths), rather than in
  `rank.py` as seam 8's text names. `forget_profile_vector` stays in
  `rank.py`; `purge_feedback` calls it rather than duplicating the cache
  clear.
- Seam 10 (`explain.profile_synthesis_messages`) landed together with the
  typed-dependency ports migration in this task, not as a separate pass --
  both touch `explain.py`'s only model-facing surface, and splitting them
  would have meant reading `_build_graph` twice.
- `BackendUnreachable`/`backend_unreachable` moved from `llm.py` to
  `ports.py`. This is the mechanism, not incidental tidying, that makes
  `test_domain_reaches_models_only_through_ports` satisfiable: `features.py`
  and `ingest.py` both need to name and catch this condition, and before the
  move the only place to import it from was the concrete client module the
  rule forbids domain code from naming. `llm.py` re-imports both names
  (`# noqa: F401`, "re-exported: callers import them from here") so
  `from attestation.llm import BackendUnreachable` keeps working for every
  existing caller.
- `MCP_SQL_BASELINE`: the amendment's prose cites 26 at `787823d` (feed 15,
  personas 10, `_tool` 1). Measured after Wave 1 (seams 5-7 landing) it is
  21 (feed 15, personas 5, `_tool` 1 -- personas' SQL count is what fell).
  `tests/test_architecture.py` pins the ratchet at 21, the lower value, per
  "the ratchet's job is to pin the best value seen."
- `conn: sqlite3.Connection` parameter count, measured the way the spec's
  success criterion measures it -- the thirteen domain files only
  (`grep -c "conn: sqlite3.Connection" src/attestation/{rank,ledger,claims,
  kg,features,explain,simulate,corpus,citations,ingest,feeds,implicit,
  personas}.py`, summed): **44**, up from 40 at `787823d`, not down. This is
  a bad proxy for what the spec wanted, and the reason is structural rather
  than measurement noise: every seam this wave landed added a small typed
  reader (`_compare_rows`, `_compare_values`, `_survey_rows`,
  `purge_feedback`, and others) beside the pure function it extracted from a
  larger one -- exactly the shape the spec asked for -- and each new reader
  is one more function that takes `conn`. Splitting a function in two adds a
  parameter count even when it strictly reduces what needs a database to
  test. See "Success criteria, revisited" below for the measure that
  actually tracks the spec's goal.
- `run_tagging(conn, chat_fn, model, limit=None)`: `chat_fn` and `model` are
  now REQUIRED with no default, matching the brief's interface line exactly
  (fix round 1). The first landing gave both defaults and a private
  `_resolve_tagging_defaults` helper that fell back to the model name
  `"unresolved"` when neither the caller nor the environment supplied one --
  a silent behavioural narrowing from the pre-task `chat_model()` resolution,
  and nothing exercised that fallback path. Review caught it: a tag row
  recording `"unresolved"` as its model is worse than a `TypeError`.
  `_resolve_tagging_defaults` is deleted; there is nothing left for it to
  resolve. `chat_fn` raising was already the chosen behaviour for a `None`
  value, so "required, no default" subsumes it rather than adding a second
  mechanism.
  This forced call-site edits in two test files, not one: `tests/
  test_cli.py`'s three `run_tagging` monkeypatch replacements
  (`lambda conn, ...: ...`, predating this feature) needed widening to
  `lambda conn, chat_fn, model, limit=None: ...` so `cmd_tag`'s call
  (`run_tagging(conn, default_chat_fn, chat_model(), limit=args.limit)`)
  still matches their shape -- unchanged from the first round. New this
  round: `tests/test_features.py` (8 call sites: two calls in one
  idempotency test, plus six single calls) and `tests/test_tag_prompt.py`
  (3 call sites) previously called `run_tagging(conn, chat_fn=...)` with no
  `model`, relying on the deleted fallback; all eleven now pass
  `model="test-model"` (or `"test-model:1b"` where the test asserts the
  returned value). `test_run_tagging_reports_the_model_it_used` no longer
  sets `CHAT_MODEL` via `monkeypatch.setenv` -- that environment variable is
  never read by `run_tagging` any more, so keeping the `setenv` call would
  have implied a resolution path that no longer exists; the test now checks
  only that the explicitly-passed `model` round-trips into `stats["model"]`
  and every written row. These are call-site signature edits (the permitted
  class per the global constraints), not assertion changes: every test
  still checks the same behaviour it did before, against an explicit
  `model` instead of an implicit one. Neither test file is in this task's
  owned-files list; both were already touched in the first round for
  `test_cli.py`, and `test_features.py`/`test_tag_prompt.py` join it for the
  same reason -- `run_tagging`'s real signature cannot be satisfied any
  other way.
- `examples/flows/mcp_e2e.py:307` called `run_tagging(conn, chat.chat_json)`
  with the old two-argument shape; the local `chat_model: str` parameter
  already carried the resolved model name, so the fix was
  `run_tagging(conn, chat.chat_json, chat_model)`. Found and fixed after the
  controller flagged two golden-path test failures
  (`test_golden_paths.py::test_an_offline_path_runs_green_and_prints_its_pinned_line[flows]`
  and `[agents]`) that the brief's file list had missed; no other call site
  in `examples/` or `evals/` calls `run_tagging` or `explain` directly.

## Success criteria, revisited

The `conn: sqlite3.Connection` parameter count (see the Deviations item
above) rose 40 → 44 across the wave, the opposite of the direction the
original success criterion named. Read literally, that criterion is not
met. But the count was always a proxy, and this wave shows exactly how it
can move the wrong way while the actual goal is served: a seam that splits
`compare(conn, ...)` into `_compare_rows(conn, ...)` (a reader) and
`_compare(rows, ...)` (pure) adds one `conn`-taking function, not zero,
even though the whole point of the split was to make the blend/selection
logic testable without a database. The parameter count cannot distinguish
"a function that needs `conn` because it always did" from "a function that
needs `conn` because it is now the one-line reader beside a pure sibling."

The number that tracks the spec's actual goal is: how many of the tests
named in "The nine seams" now exist and run with no database and no model
server. Counted directly (no `seeded_db`, `tmp_path`, or `fake_embedder`
anywhere in the test function's own parameters or body):

| test | file | DB-free? |
|---|---|---|
| `test_rank_rows_blends_three_dict_rows_without_a_database` | `test_rank.py` | yes |
| `test_ranking_quality_flags_near_single_class_history` | `test_rank.py` | yes |
| `test_compare_picks_the_majority_metric_with_no_db` | `test_ledger.py` | yes |
| `test_collapse_to_last_keeps_the_last_row_per_metric_name` | `test_ledger.py` | yes |
| `test_digest_budget_accounts_for_per_topic_truncation` | `test_digest.py` | yes |
| `test_relevance_floor_keeps_hits_near_the_top_anchor` | `test_search.py` | yes |
| `test_survey_suggests_nearest_by_interest_overlap` | `test_persona_hygiene.py` | yes |
| `test_suggest_feeds_ranks_by_tag_overlap_ties_by_title` | `test_feeds.py` | yes |
| `test_synthesize_profile_renders_through_profile_synthesis_messages` | `test_explain.py` | yes |
| `test_purge_feedback_clears_clicks_explanations_and_cache` | `test_persona_hygiene.py` | no (takes the `seeded` fixture, itself built on `tmp_path` + `get_db`) |

**9 of the 10 named tests are DB-free.** The tenth was never claimed to be:
seam 8's own text calls the purge test out as needing "an in-memory sqlite
with three inserts -- deletes are I/O by nature." A newcomer can rank three
dict rows, compare three dict arms, allocate a digest budget, apply the
relevance floor, suggest a merge, score feed candidates, and render the
profile-synthesis prompt from a REPL with no database file and no model
server -- nine of the ten things this spec promised, with the tenth
correctly excluded by the spec's own reasoning. That is the criterion this
spec actually cared about, and it is met.
