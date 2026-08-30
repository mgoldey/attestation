# Round two: five lenses on structure, every finding dispositioned

**Date:** 2026-08-29
**Status:** approved by instruction ("address all issues from next review
round"); implementation follows in the plan of the same name.
**Depends on:** onion seams (`2026-08-29-onion-seams-design.md`), the
structure note (`docs/architecture/structure-and-integration-points.md`),
which proposed this round and its lenses.
**Base:** `main` at `d9dd19b`, all gates and CI green.

## Situation, complication, answer

The seams round settled module internals. The structure note named the
next lenses — Ousterhout on the tool surface, Wickham on data shapes,
Wilson + Broman on what a workspace must provide, and two designated
dissents, Howard (notebooks-first) and Petrov (declare the DAG) — and the
instruction for this round is to address every issue they raise.

"Address" has one meaning here: each finding is either **fixed**, with the
test that names it, or **refused in writing** with the reason a reader can
overturn. A dissent's finding that does not survive the repo's measured
constraints is addressed by the refusal; silence is not a disposition.
Five reviewers produced 34 findings; 22 are fixed below, 11 refused, and
one is a comparison the reviewer made for the record. The three seam
candidates the seams round's final review named are carried in as fixes.

## Method

Same protocol as the seams round: five read-only reviewers in parallel,
the same measurements (`46` tools — feed 21, sym 8, runs 7, kg 6, cite 4;
schema+description bytes min 138, median 608, max 2083, total 32.6 KB;
three zero-argument tools), each asked for `file:line`, a test, a "leave
alone" list, and — for the dissents — an honest "Adopt / Argued, not
adopted" split. Reports are in the plan's workspace and summarised here.

## Dispositions

### Ousterhout — the tool surface (6 findings: 5 fixed, 1 recorded)

- **O1 fixed.** `feed.personas` and `feed.persona_status` are two shallow
  readers for one question. Fold the zero-argument list into
  `feed.persona_status(user: str | None = None)`: omit `user` for every
  persona's name and interests (today's `feed.personas` payload, no
  training computation), pass it for one persona's detail. The surface
  becomes 45 tools (feed 20); the routing rules gain "who are the
  personas" → `feed.persona_status`. Test:
  `test_persona_status_with_no_user_lists_every_persona`.
- **O2 fixed.** `explain.explain()` collapses three causes into `None`
  and its docstring apologises for it. Return `ExplainResult(text: str |
  None, reason: Literal["ok", "unknown_user", "model_unreachable",
  "no_answer"])`; `_explain_item` raises a reason-specific `ToolError`
  for `unknown_user` and keeps today's wording for the other two. Test:
  `test_explain_distinguishes_unknown_user_from_model_down`.
- **O3 fixed.** `mcp/feed._resolved_tag` and `mcp/knowledge._path`'s
  inline resolve loop write the same canonicalise-check-refuse shape
  twice. `kg.resolve_or_raise(name, members, *, kind) -> str` raises a
  `ValueError` whose message names the kind and the recovery tool; both
  callers use it with their own membership set (the sets stay different —
  `_resolved_tag`'s comment records why). Test:
  `test_resolve_or_raise_reports_the_right_kind_in_its_message`. This is
  also the carried-over seam "`_resolved_tag` over a vocabulary set".
- **O4 fixed.** `runs.claims_check` and `cite.check` state their pairing
  only in prose a small model was measured to miss. Each response gains
  `checked: list[str]` — `["numeric", "citation"]` / `["numeric"]` /
  `["citation"]` — declared in `empty=` too. Test:
  `test_claims_check_response_states_what_it_checked`.
- **O5 fixed.** `sym.derivation` returns prose-shaped `steps` for
  `differentiate` with nothing in the response saying so. Add `traced:
  bool` to the payload and to `_EMPTY`. Test:
  `test_sym_derivation_flags_untraced_differentiate`.
- **O6 recorded.** `classifier_probs`'s `None` is one-dimensional and
  contained; `explain`'s was multi-cause — the comparison is why O2 is
  scoped to `explain` alone. No change.

### Wickham — data shapes (9 findings: 5 fixed, 3 refused, 1 tested)

- **W1 refused.** `ranking_quality`'s value-dependent keys are the
  measured answer to a 2000-character response budget
  (`tests/test_response_size.py`); always-present `caveat: null` would
  cost 40–60 bytes on every ranked response for a field callers already
  `.get`. The reviewer reached the same conclusion.
- **W2 fixed.** `RankedItem`, `_item_row` and the `items` row are three
  shapes for one item with silent renames (`id` → `item_id`, a join →
  `source`). `RankedItem.to_row(*, summary=False) -> dict` owns the
  projection; `mcp/feed.py` calls it at its three sites. Test:
  `test_ranked_item_to_row_matches_item_row_fields` (literal object, no
  DB). The carried-over seam "`_read_item`/`_list_items` shaper" lands
  here: the 3-tag/`n_tags` budget rule is applied inside `to_row`, so it
  is tested with no database.
- **W3 refused (already tidy).** `compare()`'s `arms` is long-format; a
  client never re-melts.
- **W4 fixed.** `Verdict.split`/`Verdict.step` are the matched run's
  values under the same names as `Claim.split`/`Claim.step`, the
  requested ones. Rename to `matched_split`/`matched_step`; call sites in
  `claims.py`, `mcp/claims_tools.py`, `cli.py`, tests. Test:
  `test_verdict_matched_split_is_distinct_from_claim_split`.
- **W5 fixed (documentation).** "persona" (prose) and "user" (code,
  schema) name the same row with no glossary line saying so. One
  sentence in `personas.py`'s module docstring and in `docs/concepts.md`'s
  persona entry. No rename.
- **W6 refused (already tidy).** `tag` (storage) vs `concept` (graph node
  after frequency filter and aliasing) is a real boundary at the module
  edge.
- **W7 fixed.** `feeds.py` still says `add_feed`/`remove_feed`/
  `preview_feed`/`suggest_feeds` while the tool surface already says
  `source_*`. Rename to `add_source`/`remove_source`/`preview_source`/
  `suggest_sources`; callers and tests follow; `CLAUDE.md`'s "add_feed is
  register-only" line follows. Test:
  `test_feeds_module_names_match_the_source_tool_vocabulary`.
- **W8 fixed.** `mcp/citation._as_dict` re-lists `Reference`'s fields and
  silently drops `arxiv_id`. `Reference.to_row()` owns the projection and
  its docstring states the omission. Test:
  `test_reference_to_row_omits_arxiv_id_on_purpose`.
- **W9 tested, fixed if red.** `_arms_for_run` writes a nested-arm key
  (`arms.Treatment_Eigen`) and a genuine eval split (`test`) into the same
  `split` column; `_caveats`' "different splits" warning could describe
  arm keys as splits. Write
  `test_caveats_do_not_confuse_nested_arm_keys_with_eval_splits` with
  literal dicts; if it fails, `_caveats` skips rows whose split came from
  `nested_arms` (the smallest change), and the outcome is recorded here.

### Wilson + Broman — the workspace contract (5 findings: 5 fixed)

- **WB1 fixed.** A results CSV needs a label column (`config_name,
  config, name, run, variant, arm, label, id`) and no document says so
  before the first scan. One sentence in `docs/guides/ledger.md`; a test
  that every name in `_label_of`'s tuple appears there.
- **WB2 fixed.** `corpora.toml` — the declare-it-yourself corpus manifest
  — exists, is tested, and is documented only in a design spec. A
  "Declaring a corpus" subsection in `docs/guides/ledger.md` and a line
  in `docs/concepts.md`; a test that the guide names the file and
  `LEDGER_CORPUS_FILE`.
- **WB3 fixed.** A run has `started` (the artifact's own time) and no
  record of when the ledger read it. Migration adds `runs.scanned_at`
  (NULL for rows recorded before it); `scan()` sets it; `detail()` and
  `runs.detail` surface it. Test: `test_detail_reports_when_it_was_scanned`.
- **WB4 verified, fixed if unstable.** Nothing says how to cite a run.
  Test whether `runs.id` survives a re-scan of an unchanged project; if it
  does, `docs/guides/ledger.md`'s Browsing section says the Datasette row
  URL is the citation and why the id is stable; if it does not, `scan()`
  upserts by `(project, name)` so it becomes stable, and the same
  sentence is written after. Outcome recorded here.
- **WB5 fixed.** `CONTRIBUTING.md`'s bug-report row asks for a hand-
  scrubbed `find` beside `attest runs scan` output, whose `diagnose_empty`
  message is already scrubbed by construction. Lead with the scan output;
  `find` becomes the fallback; note the message fires only on an empty
  scan.

### Howard — notebooks-first, the dissent (7 findings: 4 adopted, 3 refused)

- **H1 adopted.** `test_the_readme_commands_are_the_run_sh_commands`
  accepts a match inside a `#` comment; `examples/ranking/run.sh` passes
  that way. The test requires the match on an executed line;
  `examples/ranking/` is brought in line with how the other twelve paths
  satisfy it.
- **H2 adopted, narrowed.** One pinned line per path is asserted; the rest
  of "What it prints" can rot. For `none`-prerequisite paths, every line
  of every fenced block in "What it prints" must appear in real stdout,
  except elision lines (`...` / `[...]`). READMEs whose blocks do not
  match are corrected — that is the point.
- **H3 adopted.** `examples/mlflow/README.md`'s AUC table is hand-typed
  and nothing reads it. `test_the_committed_mlflow_table_matches_a_live_compare`
  parses the Markdown table and compares it to `ledger.compare(...)["arms"]`
  row for row.
- **H4 adopted.** Each `examples/<name>/README.md` opens with
  `<!-- checked by tests/test_golden_paths.py -->`; the discovery test
  requires it. Legibility only; discovery-by-directory is unchanged.
- **Refused:** `.ipynb` anywhere (the attribution guard has no JSON-output
  scrubber; `mkdocs.yml` has no notebook renderer; diff noise); per-example
  `check.py` (trades "no test edit to enrol a path" for legibility the
  pointer gives); CI re-running `--live` (the offline guarantee and a
  10–15 minute cost).

### Petrov — declare the DAG, the dissent (7 findings: 2 adopted, 5 refused)

- **P3 adopted.** `dvc.lock` records an md5 per output; `_dvc_runs` parses
  the file for params and walks past `outs:`. A hand-edited
  `metrics/0.01.json` rescanned silently as `auc=0.5` (verified in a
  scratch copy). On scan, hash each DVC metric file and compare to the
  lock's digest; on mismatch set `RunRecord.notes` to name it, and surface
  it as a caveat in `compare()`. `hashlib` only. Test:
  `test_dvc_lock_hash_mismatch_is_recorded`.
- **P5 adopted, narrowed.** `metric_direction.toml` and `corpora.toml`
  share one env-then-workspace-then-`~/.hermes` ladder written twice;
  `corpus.py`'s own comment says the repo should not grow a second
  convention, then does. One helper (`ledger._config_ladder(env_var,
  filename)` or equivalent) used by both; the two files and their names
  stay (they are documented and tested).
- **Refused:** corpus identity from a `deps:` hash (helps only projects
  that `dvc add` their inputs — not the corpus problem `corpus.py` solves);
  arms from a declared DAG (already true wherever a `dvc.yaml` exists;
  `family_of` runs only where there is nothing to declare);
  `ADAPTER_CAVEATS` as a defect (a recorded trade in the tracker-adapters
  spec); git SHA and dirty flag per run (offline and cheap per run, but
  `git log -1 -- path` is ~10 ms and the live ledger holds >1000 runs —
  a measured cost for a guarantee that cannot be promised across dormant,
  non-git projects; revisit if a per-project rather than per-run form is
  wanted); reporting `foreach` items never `dvc repro`'d (Datasette's
  `unevaluated_configs` answers it one layer up).

### Carried over from the seams round's final review (3: all fixed)

- `mcp/feed._read_item`/`_list_items` shaping → inside `RankedItem.to_row`
  (W2).
- `_resolved_tag` over a vocabulary set → `kg.resolve_or_raise` (O3).
- `ingest.run_ingest`'s per-feed policy (the `embedder_down` latch and
  per-feed failure decision) → a pure `_ingest_outcome(results) -> dict`
  over per-feed outcomes; the loop only fetches. Test:
  `test_ingest_outcome_latches_embedder_down_over_plain_outcomes` with
  literal outcomes.

## What is tested

Every fixed item names its test above; each test uses no database or
model unless the item is I/O by nature (WB3, WB4, P3 use `seeded_db` or a
scratch copy of `examples/dvc`). The tool count and per-namespace counts
in `CLAUDE.md` and `docs/guides/agents.md` follow O1 and are asserted by
`test_claude_md_tool_counts_match_the_live_surface`. Envelope changes (O4,
O5) go through `empty=`/`_EMPTY` so `tests/test_tool_envelope.py` stays
structural.

## Not in scope

- Anything under "refused". Each is written to be overturned by editing
  this section, not by re-arguing it in a review.
- Renaming the `users` table or the `feed.*` namespace (W5, W7 keep both).

## Success criteria

- Every one of the 34 findings appears above with a disposition; every
  "fixed" item has landed with its named test; every "refused" item has a
  reason a reader can check.
- The suite, the eight gates, `mkdocs build --strict` and CI are green;
  the tool surface is 45 with the counts in `CLAUDE.md` and
  `docs/guides/agents.md` matching the live surface.
- W9 and WB4 record what the test found, not what was hoped.

## Deviations and findings

Recorded during implementation of WB3, WB4, P3, P5, and W9 (ledger code
task, this wave).

- **W9: GREEN on arrival.** `test_caveats_do_not_confuse_nested_arm_keys_with_eval_splits`
  passed against `_caveats` unmodified. Traced why: `_split_rank("arms.A")`
  and `_split_rank("arms.B")` both fall through every `_EVAL_SPLITS`/
  `_TRAIN_SPLITS` prefix check and return `len(_EVAL_SPLITS)` (7) -- the same
  value `_split_rank(None)` returns for an unlabelled split. Two nested-arm
  keys therefore always land at the *same* rank as each other (and as
  "unlabelled"), so `_caveats`' `len(ranks_seen) > 1` check never sees a
  difference between them and the "arms are judged on different splits"
  caveat cannot fire on a nested-arm family. This is accidental correctness,
  not a documented invariant: the risk Wickham's finding 9 named (a real new
  eval split colliding in rank with a nested-arm key, e.g. an arm literally
  named `test.something`) is still live, since `_split_rank` would then
  return a rank *less than* 7 for that one key and trip the mismatch check
  for the wrong reason. No change made per the brief (`_caveats` stays
  byte-identical); flagged here rather than silently trusted.

- **WB4: unstable as found; made stable by upsert.** `test_run_ids_survive_a_rescan_of_an_unchanged_project`
  was RED before any change: `_replace_project` deleted every row for a
  project and re-inserted, so `runs.id` (an `INTEGER PRIMARY KEY` rowid)
  advanced on every scan even when the same `(project, name)` pairs were
  found again -- despite `runs` already declaring `UNIQUE (project, name)`,
  which prevented duplicates but did nothing for id stability under
  delete-then-insert. Fixed: migration 006 adds a named unique index
  `idx_runs_project_name` (a no-op on a fresh database, which already gets
  the same index via the inline `UNIQUE` plus an explicit named copy added
  to `SCHEMA` for consistency; real work only on a database migrated from
  before it existed), and `_replace_project` now upserts each record via
  `INSERT ... ON CONFLICT(project, name) DO UPDATE`, deleting only the rows
  for names that vanished from the artifacts. One implementation pitfall
  worth recording: `cursor.lastrowid` after an upsert that takes the UPDATE
  branch is NOT reliable in Python's sqlite3 (measured: it returned the
  *previous* INSERT's rowid -- a sibling run from earlier in the same loop
  -- rather than the row just updated, corrupting `run_metrics.run_id`
  foreign keys on the second scan of a multi-run project). Fixed by looking
  up each existing id from a `SELECT` taken before the upsert loop runs,
  and falling back to `cursor.lastrowid` only for a genuinely new row.
  `docs/guides/ledger.md`'s Browsing section is NOT edited here (owned by
  another task this wave); this paragraph is the factual basis for whoever
  writes "the Datasette row id is stable across a re-scan of the same
  project/name pair" there.

- **Config ladder (P5): the two ladders differed in two ways, not one.**
  `_metric_direction_path()` (a) took no `workspace` argument -- nothing
  had ever asked for a per-workspace override of `metric_direction.toml`
  -- and (b) always returned a `Path` regardless of whether the file
  existed, because its callers use it to name where to *create* the file in
  a refusal message (`no metric with a known direction ... Declare one
  under [metric_direction] in <path>`), and the common case for that
  message is precisely that the file does not exist yet. `manifest_path()`
  (a) took an optional `workspace` and checked it, and (b) checked
  `is_file()` itself and returned `None` when nothing was found, because
  its caller (`load_manifest`) needs to distinguish "no manifest" from "a
  manifest at an empty path". Resolution: `ledger._config_ladder(env_var,
  filename, workspace=None) -> Path` is the shared path-resolution step
  only (env, then workspace file if present, then `~/.hermes/<filename>`),
  always returning a path and never checking existence -- each caller keeps
  its own existence-checking behaviour on top. `_metric_direction_path()`
  calls it with no workspace and returns the bare path. `corpus.manifest_path()`
  calls it with the workspace and applies its own `is_file()` check,
  preserving its `Path | None` return exactly as before (verified: env-path
  missing, workspace file present, and nothing-anywhere all reproduce the
  pre-refactor behaviour). `corpus.py` imports `ledger._config_ladder`
  lazily inside the function body, not at module level, since `ledger.py`
  already imports `corpus` lazily inside `scan()`/`_link_corpora` -- a
  module-level import back would cycle.

- **P3 (dvc.lock hash mismatch): adopted as specified.** `_dvc_lock_outs`
  parses `dvc.lock`'s `outs:` block the same way `_dvc_lock_params` parses
  `params:` (line-indentation walk, no YAML library). `_dvc_runs` hashes
  each metric file with `hashlib.md5` when `dvc.lock` records a digest for
  it and compares; a mismatch sets `RunRecord.notes` to
  `"dvc.lock hash mismatch: <relpath> changed since dvc repro"`.
  `ledger._family_rows` now selects `r.notes` (previously omitted) and a
  new `_dvc_hash_caveats` helper surfaces only notes containing that exact
  substring as `compare()` caveats -- deliberately narrow, since `notes`
  also carries unrelated adapter text (a config file's prose header) that
  must not be reprinted as a trust warning.
