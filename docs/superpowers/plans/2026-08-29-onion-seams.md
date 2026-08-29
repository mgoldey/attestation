# Onion Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the nine seams and the two typed-dependency rules from the spec: every seam is a thin reader beside a pure function (or one named function for three copies), each with the DB-free or model-free test it makes possible, plus a golden path that ranks rows with no database.

**Architecture:** No new module, class or protocol. Each task owns a disjoint set of files and commits by pathspec. Wave 1 (Tasks 1–5) runs in parallel on `rank.py`, `ledger.py`+`mcp/provenance.py`, `mcp/feed.py`+`features.py`, `personas.py`+`mcp/personas.py`, `feeds.py`+`mcp/subscriptions.py`. Wave 2 runs Task 6 (ports migration, explain renderer, three structural tests — after Task 3, since both touch `features.py` and `mcp/feed.py`) and Task 7 (the `examples/ranking/` golden path — after Task 1) in parallel.

**Tech Stack:** Python ≥3.12, numpy, sqlite3 (in-memory for the two tests that need deletes), pytest, `ast` for structural tests.

**Spec:** `docs/superpowers/specs/2026-08-29-onion-seams-design.md`

## Global Constraints

- No caller-visible dict shape changes; no behaviour changes; every existing test stays green unchanged (an existing test may only change to import a renamed private helper).
- New tests use no database and no model unless the task says otherwise; the two exceptions (Task 4's purge test) use an in-memory sqlite via `conftest.seeded_db`.
- `rank.py`'s stale-cache policy (`_profile_vector`), the 81 deferred imports, every `# noqa: BLE001` site, the `@tool` envelope, `_digest`'s `window_days` patch, `db.py`'s BEGIN/COMMIT and `server.py`'s per-thread connection are untouched (spec "Refused").
- `_candidate_items`'s SQL, `scan()`'s transaction and every caveat string are byte-identical after.
- Complexity ratchet (`scripts/check_complexity.py`) may only go down for every touched file; the docstring ratchet stays at 0 (every new public def gets a docstring saying what it is for).
- Gates: `git add` first, then `timeout 900 uv run --frozen pre-commit run --all-files 2>&1 | grep -E '\.(Passed|Failed)$'` — read every line. Foreground only; never background, never a Monitor. If a failure is in a file you do not own (another task is live), re-run once; if it persists, report it — do not fix another task's file.
- Commit by pathspec only: `git commit -m "<sentence>\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- <your files>`. Never a bare `git commit`.
- New files go into `CLAUDE.md`'s docs index in the same commit (`tests/test_architecture.py::test_the_docs_index_lists_every_source_and_test_file`); Wave 1 creates no files.
- Deviations from the spec (two are already known: `purge_feedback` lives in `personas.py`; seam 10 lands in Task 6) are recorded in the spec's "Deviations" section by Task 6, which owns the spec file for that purpose.

---

### Task 1: `rank.rank_rows` and `rank._ranking_quality`

**Files:**
- Modify: `src/attestation/rank.py` (`classifier_probs` ~line 271, `rank_items` ~408–487, `ranking_quality` ~656–754)
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (Task 7 relies on these exact names):
  - `rank.rank_rows(rows: list[dict], profile_vec: np.ndarray, click_rows: list[dict] | None, pref: np.ndarray | None, n_clicks: int) -> list[RankedItem]` — `rows` items have keys `id, title, url, source, summary, embedding` (embedding is `np.ndarray` float32, L2-normalised) and optional `tags: list[str]`, `content_type: str | None`; `click_rows` items have `useful: int` and `embedding: np.ndarray`, or `None` when there is no click history.
  - `rank.classifier_probs(click_rows, X)` — same single-class guard, now over rows instead of a connection.
  - `rank._ranking_quality(counts: dict[int, int], by_source: dict[str, int]) -> dict`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rank.py`; note neither uses `tmp_path` nor `fake_embedder`):

```python
def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _row(i, vec, **extra):
    return {"id": i, "title": f"item {i}", "url": "http://x", "source": None,
            "summary": "", "embedding": _unit(vec), **extra}


def test_rank_rows_blends_three_dict_rows_without_a_database():
    from attestation.rank import rank_rows

    profile = _unit([1.0, 0.0, 0.0])
    rows = [_row(1, [0.0, 1.0, 0.0]), _row(2, [1.0, 0.1, 0.0]), _row(3, [0.5, 0.5, 0.0])]
    out = rank_rows(rows, profile, click_rows=None, pref=None, n_clicks=0)
    assert [r.item_id for r in out] == [2, 3, 1]          # profile order, no clicks
    assert out[0].profile_similarity > out[-1].profile_similarity

    # A single-class click history leaves the classifier silent (guard) and
    # an all-neutral pref array is a blend no-op: order is unchanged.
    clicks = [{"useful": 1, "embedding": _unit([1.0, 0.0, 0.0])}]
    same = rank_rows(rows, profile, click_rows=clicks, pref=np.full(3, 0.5), n_clicks=1)
    assert [r.item_id for r in same] == [2, 3, 1]


def test_rank_items_is_fetch_plus_rank_rows(tmp_path, fake_embedder):
    """The seam did not change the order rank_items serves."""
    conn = seeded_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=6)
    uid = get_user_id(conn, "researcher")
    before = [r.item_id for r in rank_items(conn, fake_embedder, uid)]
    assert before, "corpus should rank"
    assert before == [r.item_id for r in rank_items(conn, fake_embedder, uid)]


def test_ranking_quality_flags_near_single_class_history():
    from attestation.rank import _ranking_quality

    out = _ranking_quality({1: 199, 0: 1}, {"ui": 200})
    assert out["classifier_active"] is True
    assert "almost nothing to contrast against" in out["caveat"]
    assert out["clicks"] == 200 and out["real_clicks"] == 200

    silent = _ranking_quality({}, {})
    assert silent["classifier_active"] is False
    assert silent["caveat"].startswith("classifier OFF: 0 clicks")
```

- [ ] **Step 2: Run them to verify they fail.** Run: `uv run --frozen pytest tests/test_rank.py -k "rank_rows or ranking_quality_flags or fetch_plus" -v`. Expected: the first and third FAIL with `ImportError` (`rank_rows`/`_ranking_quality` do not exist); the second PASSES already (it is the behaviour pin for Step 3).

- [ ] **Step 3: Implement `rank_rows` and re-shape `classifier_probs`.** In `rank.py`, `classifier_probs` becomes:

```python
def classifier_probs(click_rows, X: np.ndarray) -> np.ndarray | None:
    """P(useful) for each row of `X` from a persona's own click history
    (`click_rows`: dicts with `useful` and a float32 `embedding`), or `None`
    when there is not yet a two-class history to learn from -- callers must
    fall back to embedding-only order on `None` rather than treat it as
    all-zero. Takes rows, not a connection, so the blend can be exercised on
    literal vectors."""
    if not click_rows:
        return None
    y = np.array([int(r["useful"]) for r in click_rows])
    if len(set(y.tolist())) < 2:
        return None  # single-class guard: never let sklearn see one class
    X_train = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in click_rows])
    clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)
    clf.fit(X_train, y)
    return clf.predict_proba(X)[:, 1]
```

`_click_training_data(conn, user_id)` becomes the reader that returns the row dicts (`[{"useful": r["useful"], "embedding": np.frombuffer(r["embedding"], dtype=np.float32)} for r in rows]`, or `[]`). Then add, directly above `rank_items`:

```python
def rank_rows(
    rows: list[dict],
    profile_vec: np.ndarray,
    click_rows: list[dict] | None,
    pref: np.ndarray | None,
    n_clicks: int,
) -> list[RankedItem]:
    """The blend, on rows: profile cosine, the click classifier (silent on a
    single-class history) and the tie-averaged preference term, mixed by
    `blend_weight(n_clicks)`. Pure: no connection, no embedder -- the reader
    `rank_items` supplies rows with float32 embeddings and optional `tags`/
    `content_type`, and `examples/ranking/` calls this on literal vectors."""
    if not rows:
        return []
    X = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in rows])
    profile_sims = X @ profile_vec
    profile_rank = ranks(profile_sims)

    click_ranks = []
    probs = classifier_probs(click_rows, X)
    if probs is not None:
        click_ranks.append(ranks(probs))
    if pref is not None:
        click_ranks.append(avg_ranks(pref))

    if not click_ranks:
        final = profile_rank.astype(np.float64)
    else:
        w = blend_weight(n_clicks)
        final = w * np.mean(click_ranks, axis=0) + (1 - w) * profile_rank

    order = np.argsort(final)
    return [
        RankedItem(
            item_id=rows[i]["id"],
            title=rows[i]["title"],
            url=rows[i]["url"],
            source=rows[i]["source"],
            score=float(final[i]),
            profile_similarity=float(profile_sims[i]),
            tags=rows[i].get("tags") or [],
            content_type=rows[i].get("content_type"),
            summary=rows[i]["summary"],
        )
        for i in order
    ]
```

`rank_items` keeps its signature and docstring and becomes: `_candidate_items(...)` → early return on empty → the `users` query and `_profile_vector` call as now → the two chunked `item_features`/`item_tags` queries as now (moved up, they only read `ids`) → build `row_dicts = [{"id": r["id"], "title": r["title"], "url": r["url"], "source": r["source"], "summary": r["summary"], "embedding": np.frombuffer(r["embedding"], dtype=np.float32), "tags": tags_by.get(r["id"], []), "content_type": ctype.get(r["id"])} for r in rows]` → `click_rows = _click_training_data(conn, user_id)` → `pref = pref_scores_for_items(conn, user_id, ids) if _preference_ready(conn, user_id) else None` → the `n_clicks` COUNT query as now → `return rank_rows(row_dicts, profile_vec, click_rows, pref, n_clicks)`. Keep the existing comment about tie-averaged ranks next to the `pref` line.

- [ ] **Step 4: Implement `_ranking_quality`.** Move everything in `ranking_quality` from `total = sum(counts.values())` to `return out` into `_ranking_quality(counts, by_source)` (docstring: "Caveat selection over click counts, pure so the 199/1 case is a two-dict test; `ranking_quality` is the one-query reader."). `ranking_quality(conn, user_id)` keeps its docstring, runs the GROUP BY, builds `counts`/`by_source` exactly as now, and returns `_ranking_quality(counts, by_source)`. Keep every comment with the code it annotates.

- [ ] **Step 5: Run the module and the neighbours.** Run: `uv run --frozen pytest tests/test_rank.py tests/test_rank_relevance.py tests/test_response_size.py tests/test_search.py tests/test_digest.py -q`. Expected: all pass; the three new tests pass.

- [ ] **Step 6: Gates and commit.** `git add src/attestation/rank.py tests/test_rank.py`; run the gates; commit with pathspec `-- src/attestation/rank.py tests/test_rank.py`. Message: `rank_rows blends rows instead of a connection and _ranking_quality selects caveats over two dicts, so the first rank tests with no database exist.`

---

### Task 2: `ledger.compare` split and `ledger.collapse_to_last`

**Files:**
- Modify: `src/attestation/ledger.py` (`compare` ~780–959), `src/attestation/mcp/provenance.py` (`_detail` ~338–374)
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces: `ledger._compare_rows(conn, family, project) -> tuple[list, dict[int, list[dict]], dict[int, float]] | None` (None = the empty-family branch already handled), `ledger._compare(rows, values_by_run, n_by_run, metric, directions, family) -> dict`, `ledger.collapse_to_last(metrics: list[dict]) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ledger.py`):

```python
def test_compare_picks_the_majority_metric_with_no_db():
    from attestation.ledger import _compare

    rows = [
        {"id": 1, "name": "asr_a", "family": "asr", "project": "p", "corpus_id": None, "adapter": None},
        {"id": 2, "name": "asr_b", "family": "asr", "project": "p", "corpus_id": None, "adapter": None},
        {"id": 3, "name": "asr_c", "family": "asr", "project": "p", "corpus_id": None, "adapter": None},
    ]
    values = {
        1: [{"run_id": 1, "value": 0.30, "step": None, "split": None}],
        2: [{"run_id": 2, "value": 0.20, "step": None, "split": None}],
        3: [],
    }
    out = _compare(rows, values, {}, metric="wer", directions={"wer": "lower_is_better"}, family="asr")
    assert out["metric"] == "wer" and out["direction"] == "lower_is_better"
    assert out["winner"] == "asr_b"
    assert out["without_metric"] == ["asr_c"]
    assert [a["name"] for a in out["arms"]] == ["asr_b", "asr_a", "asr_c"]
    assert isinstance(out["caveats"], list)


def test_collapse_to_last_keeps_the_last_row_per_metric_name():
    from attestation.ledger import collapse_to_last

    metrics = [
        {"metric": "loss", "step": 1, "value": 0.9},
        {"metric": "acc", "step": 1, "value": 0.5},
        {"metric": "loss", "step": 2, "value": 0.4},
    ]
    out = collapse_to_last(metrics)
    assert [m["metric"] for m in out] == ["loss", "acc"]   # first-appearance order
    assert out[0]["step"] == 2
```

Read `_arms_for_run`, `_corpus_agreement` and `_all_caveats` before Step 3 to confirm which row keys they read (the `rows` dicts above must carry every key they touch — add keys to the fixture rather than guarding the code).

- [ ] **Step 2: Run to verify they fail.** `uv run --frozen pytest tests/test_ledger.py -k "no_db or collapse_to_last" -v` → ImportError on both.

- [ ] **Step 3: Split `compare`.** `_compare_rows(conn, family, project)` holds: `_family_rows`, `_refuse_cross_project`, the metric-discovery GROUP BY (only when `metric is None` — so it returns `counts` too), the `values_by_run` and `n_by_run` queries. Because metric discovery needs the metric before the values query, structure it as: `_compare_rows` returns `(runs, counts)`; a second reader `_compare_values(conn, run_ids, metric) -> (values_by_run, n_by_run)`. `_compare(runs, values_by_run, n_by_run, metric, directions, family)` holds everything from `direction = _metric_direction(...)` to the final `return {...}` — i.e. the direction refusal, `_arms_for_run`, `rank_key`, `_corpus_agreement`, `_all_caveats`. Metric selection from `counts` (`known`/`_no_direction_message`) stays a pure helper `_pick_metric(counts, directions, family) -> str`. `compare()` becomes: rows/counts via reader → the empty-family message branch exactly as now (it needs `conn`, leave it in `compare`) → `metric = metric or _pick_metric(counts, directions, family)` → values via reader → `return _compare(...)`. Every comment travels with its code.

- [ ] **Step 4: `collapse_to_last`.** In `ledger.py` next to `nested_arms`:

```python
def collapse_to_last(metrics: list[dict]) -> list[dict]:
    """A step series collapsed to its last row per metric NAME, first-appearance
    order kept. Keyed on the name alone: on the live worst case `split` carried
    the sweep coordinate, so keying on (metric, split) collapsed nothing and 4
    of 33 names survived a row cap. Truncation is the caller's budget, not this
    function's."""
    last: dict[str, dict] = {}
    for row in metrics:
        last[row["metric"]] = row
    return list(last.values())
```

In `mcp/provenance.py:_detail`, replace the `last`/`seen` loop with `distinct = ledger.collapse_to_last(metrics)`; `collapsed = distinct[:MAX_METRIC_ROWS]`; the message uses `len(distinct)` where it used `len(last)`. Keep the comment block, trimmed of the part now in the docstring.

- [ ] **Step 5: Run.** `uv run --frozen pytest tests/test_ledger.py tests/test_ledger_mcp.py tests/test_ledger_adapters.py tests/test_examples.py -q` → all pass.

- [ ] **Step 6: Gates and commit** by pathspec `-- src/attestation/ledger.py src/attestation/mcp/provenance.py tests/test_ledger.py`. Message: `compare() is a reader and a decision: metric selection, direction, arm ranking and caveats run on dicts, and collapse_to_last moves down from the detail tool.`

---

### Task 3: `mcp/feed.py` seams and `features.top_and_bottom_keys`

**Files:**
- Modify: `src/attestation/mcp/feed.py` (`_profile_status` ~604–618, `_semantic_hits` ~675–701, `_digest_body` ~930–977), `src/attestation/features.py` (add one public function near `_score` ~480)
- Test: `tests/test_digest.py`, `tests/test_search.py`, `tests/test_features.py`

**Interfaces:**
- Produces: `mcp.feed._allocate_digest_budget(grouped: dict[str, list], unclustered: list, per_topic: int, budget: int) -> dict` with keys `topics, shown_unclustered, shipped, dropped_in_topics`; `mcp.feed._vector_search(conn, embedder, query, k) -> dict[int, float]`; `mcp.feed._apply_relevance_floor(sims: dict[int, float]) -> dict[int, float]`; `features.top_and_bottom_keys(conn, user_id, n=5) -> tuple[list[str], list[str]]`.
- Task 6's `test_no_mcp_module_imports_a_private_domain_name` will assert the `_key_stats`/`_score` import is gone.

- [ ] **Step 1: Failing tests.** In `tests/test_digest.py`:

```python
def test_digest_budget_accounts_for_per_topic_truncation():
    from attestation.mcp.feed import _allocate_digest_budget

    item = {"item_id": 0}
    grouped = {"a": [dict(item, item_id=i) for i in range(8)], "b": [dict(item, item_id=i) for i in range(2)]}
    unclustered = [dict(item, item_id=i) for i in range(100, 105)]
    out = _allocate_digest_budget(grouped, unclustered, per_topic=3, budget=12)
    in_topics = sum(len(t["items"]) for t in out["topics"])
    assert out["shipped"] == in_topics + len(out["shown_unclustered"])
    assert [t["label"] for t in out["topics"]] == ["a", "b"]      # largest first
    assert out["topics"][0]["n_total"] == 8 and len(out["topics"][0]["items"]) == 3
    assert out["shipped"] == 3 + 2 + 5 and len(out["shown_unclustered"]) == 5
```

In `tests/test_search.py`:

```python
def test_relevance_floor_keeps_hits_near_the_top_anchor():
    from attestation.mcp.feed import _apply_relevance_floor

    kept = _apply_relevance_floor({1: 0.62, 2: 0.60, 3: 0.61, 4: 0.30})
    assert set(kept) == {1, 2, 3}
    assert _apply_relevance_floor({}) == {}
```

In `tests/test_features.py`, a test that `top_and_bottom_keys` returns the same keys `_key_stats`+`_score` would (seed one user with two useful clicks on tagged items via `seeded_db`; assert the liked list's first key is `tag:<that tag>` and the disliked list is empty).

- [ ] **Step 2: Verify RED.** `uv run --frozen pytest tests/test_digest.py tests/test_search.py tests/test_features.py -k "budget or relevance_floor or top_and_bottom" -v` → ImportError ×3.

- [ ] **Step 3: Implement.** (a) `_apply_relevance_floor(sims)`: the last five lines of `_semantic_hits` (from `if not sims` through the comprehension), comparison `sim >= best * RELEVANCE_FLOOR` unchanged, with the "sqlite-vec returns k rows" comment; `_vector_search(conn, embedder, query, k)` is the query and the `1 - d²/2` mapping; `_semantic_hits` becomes `return _apply_relevance_floor(_vector_search(conn, embedder, query, k))` and keeps its docstring. (b) `_allocate_digest_budget`: the loop from `topics = []` through `shipped = ...`, returning `{"topics": topics, "shown_unclustered": shown_unclustered, "shipped": shipped}`; `_digest_body` calls it and computes `dropped = len(items) - out["shipped"]` and the message as now; keep both comments ("Unclustered draws from the SAME budget", "Count ITEMS not shown"). (c) In `features.py`:

```python
def top_and_bottom_keys(conn, user_id: int, n: int = 5) -> tuple[list[str], list[str]]:
    """The `n` feature keys this reader has scored highest and lowest, by the
    same Laplace-smoothed score the preference term ranks with. Public so the
    profile-status tool stops importing `_key_stats`/`_score` across the
    module boundary; the smoothing is not reimplemented."""
    stats = _key_stats(conn, user_id)
    scored = sorted(((k, _score(stats, k)) for k in stats), key=lambda kv: kv[1], reverse=True)
    liked = [k for k, s in scored[:n] if s > 0.5]
    disliked = [k for k, s in scored[-n:][::-1] if s < 0.5]
    return liked, disliked
```

Before writing it, read `_profile_status`'s remaining lines (after 618) to copy the exact selection rule it uses for `top_liked`/`top_disliked` (threshold, `n`, order) — the function must return what the tool returned, byte for byte. Then `_profile_status` imports `top_and_bottom_keys` and uses it.

- [ ] **Step 4: Run.** `uv run --frozen pytest tests/test_digest.py tests/test_search.py tests/test_features.py tests/test_mcp_server.py tests/test_response_size.py -q` → all pass.

- [ ] **Step 5: Gates and commit** by pathspec `-- src/attestation/mcp/feed.py src/attestation/features.py tests/test_digest.py tests/test_search.py tests/test_features.py`. Message: `Digest budget allocation and the relevance floor are pure functions with the tests their measured bugs deserved, and profile status stops importing private feature helpers.`

---

### Task 4: `personas.purge_feedback` and `personas._annotate_survey`

**Files:**
- Modify: `src/attestation/personas.py` (`survey` ~100–139, `merge` ~222–225), `src/attestation/mcp/personas.py` (`_delete_persona` ~89–96, `_reset_feedback` ~114–124)
- Test: `tests/test_persona_hygiene.py`

**Interfaces:**
- Produces: `personas.purge_feedback(conn, user_id: int, *, delete_user: bool = False) -> None` (never commits); `personas._survey_rows(conn) -> list[dict]`; `personas._annotate_survey(rows: list[dict]) -> list[dict]`.
- Deviation from the spec: `purge_feedback` lives in `personas.py`, not `rank.py` (one owner per file; Task 1 owns `rank.py`). Task 6 records it.

- [ ] **Step 1: Failing tests** (append to `tests/test_persona_hygiene.py`; look at the file's `seeded` fixture first and reuse it):

```python
def test_survey_suggests_nearest_by_interest_overlap():
    from attestation.personas import _annotate_survey

    rows = [
        {"id": 1, "name": "new", "interests": "protein folding dynamics", "clicks": 0,
         "trainable": 0, "trainable_positive": 0},
        {"id": 2, "name": "far", "interests": "graph databases", "clicks": 3,
         "trainable": 3, "trainable_positive": 1},
        {"id": 3, "name": "near", "interests": "protein folding simulations", "clicks": 3,
         "trainable": 3, "trainable_positive": 3},
    ]
    out = _annotate_survey(rows)
    assert out[0]["nearest"] == "near" and out[0]["nearest_overlap"] == round(2 / 4, 3)
    assert "nearest" not in out[1] and out[1]["classifier_ready"] is True
    assert out[2]["classifier_ready"] is False            # all positive
    assert "nearest" not in _annotate_survey(rows[:1])[0]  # single persona: no key


def test_purge_feedback_clears_clicks_explanations_and_cache(seeded):
    from attestation.personas import purge_feedback
    from attestation.rank import _PROFILE_VEC_CACHE, _db_identity

    conn = seeded
    uid = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()["id"]
    conn.execute("INSERT INTO explanations(user_id, item_id, text) VALUES (?, 1, 'x')", (uid,))
    _PROFILE_VEC_CACHE[(_db_identity(conn), uid)] = ("h", None)
    purge_feedback(conn, uid)
    assert conn.execute("SELECT COUNT(*) n FROM clicks WHERE user_id = ?", (uid,)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM explanations WHERE user_id = ?", (uid,)).fetchone()["n"] == 0
    assert (_db_identity(conn), uid) not in _PROFILE_VEC_CACHE
    assert conn.execute("SELECT 1 FROM users WHERE id = ?", (uid,)).fetchone() is not None
    purge_feedback(conn, uid, delete_user=True)
    assert conn.execute("SELECT 1 FROM users WHERE id = ?", (uid,)).fetchone() is None
```

Check the `explanations` table's columns in `db.py` before relying on `(user_id, item_id, text)`; adjust the INSERT to the real schema.

- [ ] **Step 2: Verify RED.** `uv run --frozen pytest tests/test_persona_hygiene.py -k "nearest_by_interest or purge_feedback" -v` → ImportError ×2.

- [ ] **Step 3: Implement.**

```python
def purge_feedback(conn: sqlite3.Connection, user_id: int, *, delete_user: bool = False) -> None:
    """Remove a persona's residue -- clicks, cached explanations, its cached
    profile vector -- and optionally the persona row. Never commits: merge()
    spans one transaction over several personas and the tools commit once
    after, so commit timing stays with the caller. Explanations go with the
    clicks because implicit.harvest reads an explanation with no click as a
    weak positive, and users.id is a rowid SQLite reuses."""
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM explanations WHERE user_id = ?", (user_id,))
    if delete_user:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    forget_profile_vector(conn, user_id)
```

Replace the three copies: `merge()` → `purge_feedback(conn, loser["id"], delete_user=True)`; `_delete_persona` → `purge_feedback(conn, user_row["id"], delete_user=True)` then `conn.commit()`; `_reset_feedback` → `purge_feedback(conn, user_row["id"])` then `conn.commit()`. Move the explanation comments from the tools into the docstring (once). For `survey`: `_survey_rows(conn)` returns the per-user dicts with `id, name, interests, clicks, trainable, trainable_positive` (no derived fields); `_annotate_survey(rows)` adds `classifier_ready` and, for rows with `trainable == 0` and another row present, `nearest`/`nearest_overlap` (rounded to 3) using `_overlap`; it strips `id` from the output rows so `survey()`'s shape is unchanged. `survey(conn)` = `_annotate_survey(_survey_rows(conn))`.

- [ ] **Step 4: Run.** `uv run --frozen pytest tests/test_persona_hygiene.py tests/test_persona_autocreate.py tests/test_mcp_server.py -q` → pass.

- [ ] **Step 5: Gates and commit** by pathspec `-- src/attestation/personas.py src/attestation/mcp/personas.py tests/test_persona_hygiene.py`. Message: `One purge_feedback replaces three copies that differed in whether users was deleted, and survey's merge suggestion runs on rows.`

---

### Task 5: `feeds._score_candidates` and raising instead of `{"ok": …}`

**Files:**
- Modify: `src/attestation/feeds.py`, `src/attestation/mcp/subscriptions.py` (`_add_feed`, `_remove_feed`, `_preview_feed` ~60–105)
- Test: `tests/test_feeds.py`

**Interfaces:**
- Produces: `feeds.FeedError(ValueError)`; `feeds.add_feed(...) -> tuple[int, str]` (feed_id, message); `feeds.remove_feed(conn, feed_id) -> tuple[int, str]` (orphaned_items, message); `feeds.preview_feed(url, limit, parse) -> dict` with `title, entries, message` and NO `ok`; `feeds._score_candidates(liked: set[str], subscribed: set[str], candidates: list[dict], limit: int) -> list[dict]`.

- [ ] **Step 1: Grep for other consumers.** `grep -rn "add_feed\|remove_feed\|preview_feed" src tests examples --include=*.py` — every caller that reads `["ok"]` or `["feed_id"]` from these is updated in this task (the spec's Risk line; `cli.py` had none at `787823d`, re-check).

- [ ] **Step 2: Failing tests** (in `tests/test_feeds.py`; read its existing fixtures for the `parse` stub pattern):

```python
def test_suggest_feeds_ranks_by_tag_overlap_ties_by_title():
    from attestation.feeds import _score_candidates

    cands = [
        {"url": "a", "title": "Z", "tags": ["nlp"]},
        {"url": "b", "title": "A", "tags": ["nlp"]},
        {"url": "c", "title": "M", "tags": ["nlp", "rl"]},
        {"url": "d", "title": "S", "tags": ["rl"]},
    ]
    out = _score_candidates({"nlp", "rl"}, {"d"}, cands, limit=5)
    assert [o["title"] for o in out] == ["M", "A", "Z"]
    assert out[0]["score"] == 2 and out[0]["matched_tags"] == ["nlp", "rl"]


def test_feeds_functions_raise_rather_than_return_ok(tmp_path):
    from attestation.feeds import FeedError, add_feed

    conn = seeded_db(tmp_path / "t.db")

    class NotAFeed:
        entries = []
        feed = {}
        bozo = 1

    with pytest.raises(FeedError, match="did not parse"):
        add_feed(conn, "http://nope", parse=lambda url: NotAFeed())
```

- [ ] **Step 3: Verify RED**, then implement: `class FeedError(ValueError)` with a docstring ("A caller-fixable refusal — a URL that does not parse, an unknown feed id — raised instead of returned as `ok: False`, so the MCP layer maps it to ToolError the way it does every other refusal."); each `return {"ok": False, ...}` becomes `raise FeedError(message)`; each `return {"ok": True, "feed_id": x, "message": m}` becomes `return x, m` (and the `orphaned_items` equivalent in `remove_feed`); `preview_feed` drops the `ok` key. `_score_candidates` holds the loop from `scored = []` to the return; `suggest_feeds` = two queries + `_load_candidates()` + `_score_candidates(liked, subscribed, candidates, limit)`. In `mcp/subscriptions.py`: `try: feed_id, message = feeds_mod.add_feed(conn, url, title) except FeedError as exc: raise ToolError(str(exc)) from exc`, and the same shape for the other two; delete the "returns its own envelope" comment.

- [ ] **Step 4: Run.** `uv run --frozen pytest tests/test_feeds.py tests/test_mcp_server.py tests/test_cli.py -q` → pass; update any existing feeds test that asserted `out["ok"]` to the new return shape (this is the one permitted kind of existing-test edit, and it is a shape change internal to the domain function, not the tool envelope — `test_tool_envelope.py` must stay untouched and green).

- [ ] **Step 5: Gates and commit** by pathspec `-- src/attestation/feeds.py src/attestation/mcp/subscriptions.py tests/test_feeds.py`. Message: `feeds raises FeedError instead of hand-building the envelope @tool owns, and candidate scoring runs on sets.`

---

### Task 6: Ports migration, the explain renderer, and three structural rules (after Task 3)

**Files:**
- Modify: `src/attestation/ports.py`, `src/attestation/llm.py`, `src/attestation/explain.py`, `src/attestation/features.py` (`run_tagging` ~378–395 and the module-scope import at line 20), `src/attestation/ingest.py` (line 14 and ~184), `src/attestation/cli.py`, `src/attestation/server.py`, `src/attestation/mcp/feed.py` (the `explain`/`run_tagging` call sites), `tests/test_architecture.py`, `tests/test_explain.py`, `docs/superpowers/specs/2026-08-29-onion-seams-design.md` (Deviations), `CLAUDE.md` (the `mcp/_tool.py` count line only if a number it states changes — it should not)
- Test: `tests/test_architecture.py`, `tests/test_explain.py`

**Interfaces:**
- Consumes: Task 3's `features.top_and_bottom_keys` (the private-import test asserts the old import is gone).
- Produces: `ports.BackendUnreachable`, `ports.backend_unreachable(exc)` (moved from `llm.py`, which re-imports them so `from attestation.llm import BackendUnreachable` keeps working); `explain.profile_synthesis_messages(titles: list[str]) -> list[dict]`; `explain.explain(conn, user_id, item_id, chat_fn)` with `chat_fn` REQUIRED; `features.run_tagging(conn, chat_fn, model: str, limit=None)` with `chat_fn` and `model` REQUIRED.

- [ ] **Step 1: Failing structural tests** (append to `tests/test_architecture.py`, reusing `_modules`, `_rel`, `_module_name`):

```python
DOMAIN = {
    "explain", "features", "ingest", "simulate", "rank", "kg", "claims", "ledger",
    "corpus", "citations", "implicit", "personas", "feeds",
}
MODEL_IMPORTERS_ALLOWED = {"embed", "cli", "server", "install", "mcp_server", "mcp._shared"}


def _imports_of(path: pathlib.Path, module: str) -> list[str]:
    """Names imported from `module` anywhere in the file, function bodies included:
    a lazy import of the concrete client is still the concrete client."""
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.extend(a.name for a in node.names)
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names if a.name == module)
    return names


def test_domain_reaches_models_only_through_ports():
    """ports.py is load-bearing only if the domain uses it. At 787823d explain,
    features and ingest imported the concrete client; a second provider would
    have meant editing domain code."""
    offenders = {}
    for path in _modules():
        name = _module_name(path)
        if name in DOMAIN:
            found = _imports_of(path, "attestation.llm")
            if found:
                offenders[_rel(path)] = found
    assert not offenders, f"domain modules importing the concrete client: {offenders}"


_SQL = re.compile(r"""["'](SELECT|INSERT|UPDATE|DELETE|WITH) """)
MCP_SQL_BASELINE = 26  # measured at 787823d: feed 15, personas 10, _tool 1


def test_mcp_layer_sql_only_ratchets_down():
    """The presentation layer writing its own queries is the braid the onion
    was for. Pinned, not banned: it falls as seams move queries into domain
    readers, and a new query up here needs a reason in a spec."""
    counts = {
        _rel(p): len(_SQL.findall(p.read_text())) for p in _modules() if p.parent.name == "mcp"
    }
    total = sum(counts.values())
    assert total <= MCP_SQL_BASELINE, f"SQL in mcp/ rose to {total}: {counts}"


def test_no_mcp_module_imports_a_private_domain_name():
    """A private name crossing a module boundary is a missing public function."""
    offenders = {}
    for path in _modules():
        if path.parent.name != "mcp":
            continue
        tree = ast.parse(path.read_text())
        private = [
            f"{n.module}.{a.name}"
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("attestation.")
            and not (n.module or "").startswith("attestation.mcp")
            for a in n.names
            if a.name.startswith("_")
        ]
        if private:
            offenders[_rel(path)] = private
    assert not offenders, f"mcp modules importing private domain names: {offenders}"
```

Add `import re` at the top if missing. When the ratchet's measured total after Tasks 3–5 is below 26, LOWER `MCP_SQL_BASELINE` to the measured value in this task (the ratchet's job is to pin the best value seen).

And in `tests/test_explain.py`:

```python
def test_synthesize_profile_renders_through_profile_synthesis_messages():
    from attestation.explain import profile_synthesis_messages

    msgs = profile_synthesis_messages(["A paper", "B paper"])
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "Summarize this reader in one sentence."
    assert msgs[1]["content"] == "Recently useful titles:\n- A paper\n- B paper"
```

plus an AST assertion that `explain.py` contains exactly two `{"role": "system"` literals, both inside functions named `*_messages` (the one-renderer rule, machine-checked).

- [ ] **Step 2: Verify RED.** `uv run --frozen pytest tests/test_architecture.py tests/test_explain.py -k "through_ports or ratchets_down or private_domain or profile_synthesis" -v`. Expected: `through_ports` FAILS naming `explain.py`, `features.py`, `ingest.py`; `private_domain` PASSES if Task 3 landed (else FAILS on `mcp/feed.py`); `ratchets_down` PASSES; `profile_synthesis` FAILS with ImportError.

- [ ] **Step 3: Move the failure contract to `ports.py`.** Cut `class BackendUnreachable(RuntimeError)` and `def backend_unreachable(exc)` from `llm.py` into `ports.py` (below the Protocols, with their docstrings; add a sentence: "Here rather than in llm.py because it is the contract of any backend, and the domain must be able to name it without naming a provider"). In `llm.py`: `from attestation.ports import BackendUnreachable, backend_unreachable  # re-exported: callers import them from here`. In `features.py` line 20 and `ingest.py` line 14: import them from `attestation.ports`. `ingest.py`'s `DEFAULT_BASE_URL` use: read the line that uses it; if it is only an error message, change the message to name the environment variable (`LLM_BASE_URL`) and drop the import; if it is behaviour, stop and report.

- [ ] **Step 4: `chat_fn` required.** `explain.explain(conn, user_id, item_id, chat_fn)` — drop the `default_chat_fn` default and the module-scope import; `_build_graph` already takes `chat_fn`. `features.run_tagging(conn, chat_fn, model, limit=None)` — drop both deferred `llm` imports; `model` replaces the `chat_model()` call (grep the body for its use — it is recorded per tagged item). Update every caller to pass `default_chat_fn` and `chat_model()` from `attestation.llm` at the composition root: `cli.py` (`attest tag`, `attest explain`/serve paths), `server.py` (already passes `chat_fn`; check `run_tagging`), `mcp/feed.py` (the explain tool ~560 and any tagging tool — the import goes inside the tool body, deferred, which is allowed since `mcp/feed.py` is not a domain module). Add `profile_synthesis_messages(titles)` to `explain.py` next to `explanation_messages` and call it in `synthesize_profile`.

- [ ] **Step 5: Run everything touched.** `uv run --frozen pytest tests/test_architecture.py tests/test_explain.py tests/test_features.py tests/test_ingest.py tests/test_llm.py tests/test_ports.py tests/test_cli.py tests/test_server.py tests/test_mcp_server.py tests/test_tag_prompt.py -q` → pass. Then `ATTEST_TOOLS=feed ATTEST_EXPAND=1 uv run python -c "from mcp.server.fastmcp import FastMCP; from attestation.mcp import register_all; import asyncio; m=FastMCP('x'); register_all(m); print(len(asyncio.run(m.list_tools())))"` prints the same number as before (22 at `787823d`).

- [ ] **Step 6: Record deviations** in the spec's new `## Deviations and findings` section: `purge_feedback` in `personas.py`; seam 10 landed with the ports task; `BackendUnreachable` moved to `ports.py` as the mechanism that made the rule satisfiable; the measured `MCP_SQL_BASELINE` after Wave 1; the `conn:` parameter count after (`grep -c "conn: sqlite3.Connection" src/attestation/*.py`, summed) — the spec's success criterion says this number is recorded, not predicted.

- [ ] **Step 7: Gates and commit** by pathspec listing every file above. Message: `Domain modules reach the model only through ports: BackendUnreachable lives in ports.py, explain and run_tagging take chat_fn, the fourth prompt has a renderer, and three structural tests keep it so.`

---

### Task 7: The `examples/ranking/` golden path (after Task 1)

**Files:**
- Create: `examples/ranking/README.md`, `examples/ranking/run.sh`, `examples/ranking/rank_rows.py`
- Modify: `examples/README.md` (catalogue row, in the `none` group, alphabetical), `README.md` and `docs/architecture/structure-and-integration-points.md` (the word "Twelve"/"twelve golden paths" → "Thirteen"/"thirteen" — grep `-i "twelve"` across `README.md docs/ examples/README.md CONTRIBUTING.md`), `CLAUDE.md` docs index (`examples/ranking:{README.md,run.sh,rank_rows.py}`)
- Test: discovered by `tests/test_golden_paths.py`; no test edit.

**Interfaces:**
- Consumes: `rank.rank_rows(rows, profile_vec, click_rows, pref, n_clicks)`, `rank.blend_weight`, `rank.classifier_probs(click_rows, X)` from Task 1.

- [ ] **Step 1: Read the shape.** `cat examples/workspace/README.md examples/workspace/run.sh` and the seven-section rule in `CONTRIBUTING.md` "Recipe: add a golden path". Prerequisite label: `none — pure local computation`.

- [ ] **Step 2: `rank_rows.py`.** Twenty rows with deterministic 8-dimensional unit vectors from a seeded `numpy.random.default_rng(0)`, arranged as two clusters (ten near a "protein" axis, ten near a "graph" axis); a profile vector on the protein axis; six labelled clicks (four useful near protein, two not-useful near graph). Print: (1) the profile-only order (`click_rows=None`, top 5 ids and `profile_similarity` to 3 places); (2) the blended order with the clicks (`n_clicks=6`, `blend_weight` printed); (3) the classifier-only AUC on the six clicks via `sklearn.metrics.roc_auc_score` over `classifier_probs(click_rows, X_clicks)` — with the sentence `evaluate_user` uses: "measures the click classifier alone, NOT the profile or preference terms"; (4) the rank-order AUC of the blended list against the same labels (a hit = the item's cluster). Every number printed is computed in the script; the README's "What it prints" quotes the run's real output.

- [ ] **Step 3: `run.sh`** — `#!/usr/bin/env bash`, `set -euo pipefail`, `cd "$(dirname "$0")"`, `uv run python rank_rows.py`. `chmod +x`.

- [ ] **Step 4: README** with the seven sections; *Run it* contains `./run.sh` and `uv run python examples/ranking/rank_rows.py` (both verbatim in `run.sh` per the rule — read `tests/test_golden_paths.py::test_the_readme_commands_are_the_run_sh_commands` for the exact matching rule); *What it prints* pins one line of real output; *What it demonstrates*: the blend is a pure function, the two AUCs measure different things; *When it goes wrong*: sklearn missing → `uv sync`; *Next*: `examples/flows/` for the live end-to-end AUC.

- [ ] **Step 5: Catalogue row and counts.** Add the row to `examples/README.md` in order; update the "twelve" strings; add the index entry to `CLAUDE.md`.

- [ ] **Step 6: Run.** `uv run --frozen pytest tests/test_golden_paths.py tests/test_examples.py tests/test_docs_site.py tests/test_architecture.py -q` → pass (the `none` path is executed and its pinned line asserted). `uv run --group docs mkdocs build --strict` exit 0.

- [ ] **Step 7: Gates and commit** by pathspec `-- examples/ranking examples/README.md README.md docs/architecture/structure-and-integration-points.md CLAUDE.md`. Message: `A golden path ranks twenty rows with no database and prints the classifier-only AUC beside the blended-order AUC, the two numbers evaluate_user's docstring says are different.`

---

### Task 8: Whole-branch review (a reviewer, not an implementer)

A fresh agent under the Feathers/Bernhardt lens (seams and sensing points; functional core, imperative shell) reads the full diff `git diff 0d9eb2b..HEAD -- src tests examples` against the spec: every seam present with its named test; the test uses no DB/model where the spec says so (grep each new test for `seeded_db`, `tmp_path`, `fake_embedder`); nothing on the "Refused" list was touched (`git diff --stat` on `db.py`, `_tool.py`, `_candidate_items`); no new class or module; caveat strings unchanged (`git diff -- src | grep '^[-+].*caveat'` shows only moves). Findings go to fix rounds on the owning task.

## Self-review

Spec coverage: seams 1–2 (T1), 3–4 (T2), 5–7 (T3), 8 (T4), 9 (T5), 10 + typed-dependency rules + private-import rule (T6), the measurement/golden path (T7), whole-branch review (T8), deviations recorded (T6 Step 6). Placeholders: none — every test is code, every signature named, every command given. Type consistency: `rank_rows(rows, profile_vec, click_rows, pref, n_clicks)` in T1 and T7; `classifier_probs(click_rows, X)` in T1 and T7; `_compare(rows, values_by_run, n_by_run, metric, directions, family)` in T2 test and Step 3; `purge_feedback(conn, user_id, *, delete_user=False)` in T4 test and code; `FeedError`, `(feed_id, message)` in T5 test, code and MCP caller; `top_and_bottom_keys(conn, user_id, n=5)` in T3 and T6's private-import test.
