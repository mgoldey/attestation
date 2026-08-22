"""Personalized science feed tools: the `feed.*` namespace.

The reader-facing half of the server: ranked feed items, the feedback that
trains the ranker, the personas that own that training, feed subscriptions, and
the topic-grouped digest that composes ranking with the reading graph.

Two properties this namespace maintains deliberately. Ranking honesty: every
tool that returns an order also returns `ranking_quality`, because a feed built
from an untrained ranker looks exactly like one built from a good one. And no
LLM runs inside `digest` -- it returns structure, never prose, because the
caller is a model that can write the prose itself.
"""

from attestation.explain import explain as explain_item_fn
from attestation.llm import default_chat_fn
from attestation.mcp._shared import MAX_LIST_LIMIT, get_embedder, ranked_items, ranking_quality
from attestation.mcp._tool import ToolError, tool
from attestation.rank import forget_profile_vector, record_click


def register(mcp) -> None:
    """Attach every feed.* tool to the server."""

    @mcp.tool(name="feed.list")
    def list_feed(user: str, limit: int = 10, since_days: int | None = 14) -> dict:
        """List this user's currently ranked, unread feed items (best first).

        Returns each item's id, title, url, source feed name, and its blended rank
        score (lower score = better/more relevant). Does NOT return HTML or full
        article text -- just enough to summarize or link to items. `limit` is capped
        at 50 items regardless of the value passed. If `user` isn't a recognized
        persona, the response names the valid users instead of raising an error.
        Each item also carries its LLM-extracted topic "tags" and "content_type"
        (paper/survey/announcement/release/blog/other) when the tagging pass has
        processed it; both are empty/null for not-yet-tagged items.

        `since_days` bounds how far back the feed reaches -- defaults to 14, so an
        empty result may mean "nothing published in the window" rather than
        "nothing relevant"; pass a larger value or `None` (unbounded) to tell them
        apart. Use `search_feed` instead for the whole archive including already-
        rated items.

        **Read `ranking_quality` before trusting the order.** With a single-class
        click history the click classifier never fires, and depending on click
        count the order may fall back partly or fully to embedding similarity --
        see the `caveat` field for which terms are actually contributing.
        """
        return _list_feed(user, limit, since_days)

    @mcp.tool(name="feed.rate")
    def record_feedback(user: str, item_id: int, useful: bool) -> dict:
        """Record whether a feed item was useful for this user, retraining their ranking.

        This writes (or overwrites) a single click record for (user, item_id) --
        calling it again for the same item just replaces the previous verdict, so it
        is safe to call repeatedly. The next list_feed call for this user will
        reflect the updated ranking once enough mixed feedback has accumulated.
        """
        return _record_feedback(user, item_id, useful)

    @mcp.tool(name="feed.explain")
    def explain_item(user: str, item_id: int) -> dict:
        """Explain in one sentence why a specific feed item was ranked for this user.

        SLOW: this calls a local LLM (via the configured OpenAI-compatible backend)
        on first request for a given (user, item_id) pair and can take several
        seconds to ~1-2 minutes depending on the chat model and hardware; results
        are cached afterward and return instantly. Only call this for items the
        user is asking about specifically, not for every item in a list.
        """
        return _explain_item(user, item_id)

    @mcp.tool(name="feed.personas")
    def list_users() -> dict:
        """List all available reader personas (users) and their interest profiles.

        Use this to discover which `user` values are valid for list_feed,
        record_feedback, and explain_item before calling them.
        """
        return _list_users()

    @mcp.tool(name="feed.source_add")
    def add_feed(url: str, title: str | None = None) -> dict:
        """Subscribe to an RSS/Atom feed.

        Validates that the URL parses as a feed, then registers it. Does NOT fetch
        its articles: items appear after the next ingest (hourly cron, or
        `attest ingest`). Use preview_feed first to check a feed's content.
        """
        return _add_feed(url, title)

    @mcp.tool(name="feed.sources")
    def list_feeds() -> dict:
        """List subscribed feeds with item counts and when each was last fetched."""
        return _list_feeds()

    @mcp.tool(name="feed.source_remove")
    def remove_feed(feed_id: int, confirm: bool = False) -> dict:
        """Unsubscribe from a feed. Requires confirm=true.

        Existing items and all feedback on them are KEPT -- only the subscription
        is removed, so no ranking history is lost.
        """
        return _remove_feed(feed_id, confirm)

    @mcp.tool(name="feed.source_preview")
    def preview_feed(url: str, limit: int = 5) -> dict:
        """Show recent entries from a feed WITHOUT subscribing to it."""
        return _preview_feed(url, limit)

    @mcp.tool(name="feed.source_suggest")
    def suggest_feeds(user: str, limit: int = 5) -> dict:
        """Suggest feeds from a curated list, scored against tags this user liked."""
        return _suggest_feeds(user, limit)

    @mcp.tool(name="feed.persona_create")
    def create_persona(name: str, interests: str) -> dict:
        """Create a reader persona from a name and an interests description.

        Ranking starts from the interests text and personalizes from the first
        record_feedback call. Use propose_interests first if you want suggestions
        drawn from what is actually in the feed.
        """
        return _create_persona(name, interests)

    @mcp.tool(name="feed.persona_update")
    def update_persona(name: str, interests: str) -> dict:
        """Replace a persona's interests text; re-steers ranking immediately."""
        return _update_persona(name, interests)

    @mcp.tool(name="feed.persona_suggest_interests")
    def propose_interests(limit: int = 12) -> dict:
        """List the most common tags in the feed, to help write an interests string."""
        return _propose_interests(limit)

    @mcp.tool(name="feed.persona_status")
    def profile_status(user: str) -> dict:
        """Show how well-trained a persona is: click count, how much ranking is
        driven by behavior vs the written interests, and top liked/disliked tags."""
        return _profile_status(user)

    @mcp.tool(name="feed.search")
    def search_feed(
        user: str,
        query: str,
        tag: str | None = None,
        content_type: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Search items by keyword (and optional tag/content_type), ranked for this user.

        Unlike list_feed this searches the whole archive and includes items already
        rated, flagging each with already_rated.

        **Read `ranking_quality` before trusting the order.** It reports whether the
        click classifier is actually active -- with a single-class click history it
        never fires, and depending on click count the order may fall back partly or
        fully to embedding similarity; see the `caveat` field for which terms are
        actually contributing.
        """
        return _search_feed(user, query, tag, content_type, limit)

    @mcp.tool(name="feed.persona_delete")
    def delete_persona(name: str, confirm: bool = False) -> dict:
        """Delete a persona AND all its feedback. Requires confirm=true. Irreversible."""
        return _delete_persona(name, confirm)

    @mcp.tool(name="feed.persona_reset")
    def reset_feedback(name: str, confirm: bool = False) -> dict:
        """Erase a persona's clicks but keep the persona. Requires confirm=true."""
        return _reset_feedback(name, confirm)

    @mcp.tool(name="feed.harvest_engagement")
    def harvest_engagement(user: str) -> dict:
        """Turn this user's past "why is this here?" questions into weak positive feedback.

        Cheap and idempotent -- no LLM call, no network. Reads what is already
        recorded and records nothing twice.

        Explicit feedback is a channel almost nobody uses, so a ranker waiting
        to be told what is useful waits forever. Asking why an item was ranked
        is engagement: weaker than a stated opinion, but real, and already
        logged. Each such item that was never rated becomes one weak positive
        under source='implicit', so a later evaluation can weight or exclude
        them.

        An item the user already rated is left alone -- an explicit "not
        useful" is never overwritten by the reader's own curiosity.

        Only positives are inferred. No behaviour reliably means "not useful",
        and inferring rejection from silence would poison the class the ranker
        is starving for. Use record_feedback for negatives.
        """
        return _harvest_engagement(user)

    @mcp.tool(name="feed.simulate_ratings")
    def simulate_feedback(user: str, limit: int = 10, confirm: bool = False) -> dict:
        """Generate simulated reader reactions for a persona, to train its ranking.

        WRITES CLICK ROWS -- needs confirm=true. SLOW: one local LLM call per
        item, several seconds each.

        The ranker's click classifier needs both useful AND not-useful feedback
        to fire at all; a persona that has only ever marked things useful ranks
        by embedding similarity alone forever. Real feedback is overwhelmingly
        positive because saying no to a recommendation is work nobody does, so
        this asks a local model to read each item and react AS the persona --
        in prose, then a verdict -- and records what it decides.

        Rows are written with source='simulated' and stay distinguishable from
        real `ui`/`agent` clicks permanently. These are a training aid, not
        evidence about the persona: each reaction carries the reasoning behind
        its verdict so a human can audit what the ranker was taught.

        Returns per-item verdicts with reasoning, plus counts. Items the model
        is unsure about are skipped rather than recorded as a coin flip.
        """
        return _simulate_feedback(user, limit, confirm)

    @mcp.tool(name="feed.digest")
    def digest(user: str, days: int = 7, per_topic: int = 3, limit: int = 30) -> dict:
        """This user's unread feed, ranked and grouped by topic — the weekly review.

        Composes the ranked feed with the reading graph's topic clusters: each item
        joins the cluster its tags overlap most, so the result reads as "here is
        what is new, by subject" rather than a flat list. Items whose tags match no
        cluster are returned in `unclustered` rather than dropped — that bucket's
        size is a real signal, since most tags are used once and never form
        concepts.

        `days` bounds how far back the ranked feed reaches (echoed as
        `window_days`); `per_topic` caps how many items each topic shows while
        `n_total` reports how many it actually had, so truncation is visible.

        Returns structure, never prose: no LLM runs inside this tool. Per-item
        `explanation` is surfaced only when `explain_item` already cached one.

        **Read `ranking_quality` before trusting the order.** It reports whether the
        click classifier is actually active -- with a single-class click history it
        never fires, and depending on click count the order may fall back partly or
        fully to embedding similarity; see the `caveat` field for which terms are
        actually contributing.
        """
        return _digest(user, days, per_topic, limit)


def _item_row(it) -> dict:
    """The item shape list_feed and digest both return."""
    return {
        "item_id": it.item_id,
        "title": it.title,
        "url": it.url,
        "source": it.source,
        "score": it.score,
        "tags": it.tags,
        "content_type": it.content_type,
    }


@tool(empty={"items": [], "ranking_quality": {}}, needs_user=True, label="list_feed")
def _list_feed(conn, user_row, limit: int = 10, since_days: int | None = 14) -> dict:
    """`since_days` defaults to rank_items' own 14-day window so list_feed's
    behavior is unchanged; digest passes its `days` through here.
    """
    limit = min(max(int(limit), 1), MAX_LIST_LIMIT)
    items = ranked_items(conn, user_row, limit, since_days)
    return {
        "message": f"{len(items)} item(s), best first",
        "items": [_item_row(it) for it in items],
        "ranking_quality": ranking_quality(conn, user_row["id"]),
    }


@tool(needs_user=True, label="record_feedback")
def _record_feedback(conn, user_row, item_id: int, useful: bool) -> dict:
    item = conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        raise ToolError(f"unknown item_id: {item_id}")
    record_click(conn, user_row["id"], item_id, useful, source="agent")
    return {
        "message": f"recorded useful={useful} for item {item_id} (user {user_row['name']}); "
        "ranking will reflect this on the next list_feed call",
    }


@tool(empty={"explanation": None}, needs_user=True, label="explain_item")
def _explain_item(conn, user_row, item_id: int) -> dict:
    item = conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        raise ToolError(f"unknown item_id: {item_id}")
    text = explain_item_fn(conn, user_row["id"], item_id, chat_fn=default_chat_fn)
    return {"explanation": text}


@tool(empty={"users": []}, label="list_users")
def _list_users(conn) -> dict:
    rows = conn.execute("SELECT name, interests FROM users ORDER BY name").fetchall()
    return {
        "message": f"{len(rows)} user(s)",
        "users": [{"name": r["name"], "interests": r["interests"]} for r in rows],
    }


@tool(empty={"feed_id": None}, label="add_feed")
def _add_feed(conn, url: str, title: str | None = None) -> dict:
    from attestation import feeds as feeds_mod

    out = feeds_mod.add_feed(conn, url, title)
    # feeds.add_feed returns its own envelope; a URL that does not parse is a
    # caller-fixable refusal, not a bug, so its message is passed through verbatim
    if not out["ok"]:
        raise ToolError(out["message"])
    return {"message": out["message"], "feed_id": out["feed_id"]}


@tool(empty={"feeds": []}, label="list_feeds")
def _list_feeds(conn) -> dict:
    from attestation import feeds as feeds_mod

    feeds = feeds_mod.list_feeds(conn)
    return {"message": f"{len(feeds)} feed(s)", "feeds": feeds}


@tool(empty={"orphaned_items": 0}, label="remove_feed")
def _remove_feed(conn, feed_id: int, confirm: bool = False) -> dict:
    from attestation import feeds as feeds_mod

    if not confirm:
        raise ToolError(
            f"refusing to remove feed {feed_id} without confirm=true. This "
            "unsubscribes the feed; its existing items and your feedback on "
            "them are kept."
        )
    out = feeds_mod.remove_feed(conn, feed_id)
    if not out["ok"]:
        raise ToolError(out["message"])
    return {"message": out["message"], "orphaned_items": out["orphaned_items"]}


@tool(empty={"title": None, "entries": []}, needs_db=False, label="preview_feed")
def _preview_feed(url: str, limit: int = 5) -> dict:
    from attestation import feeds as feeds_mod

    out = feeds_mod.preview_feed(url, limit=min(limit, MAX_LIST_LIMIT))
    if not out["ok"]:
        raise ToolError(out["message"])
    return {"message": out["message"], "title": out["title"], "entries": out["entries"]}


@tool(empty={"suggestions": []}, needs_user=True, label="suggest_feeds")
def _suggest_feeds(conn, user_row, limit: int = 5) -> dict:
    from attestation import feeds as feeds_mod

    return {
        "message": "scored against tags you marked useful",
        "suggestions": feeds_mod.suggest_feeds(conn, user_row["id"], limit=limit),
    }


@tool(empty={"user_id": None}, label="create_persona")
def _create_persona(conn, name: str, interests: str) -> dict:
    from attestation.rank import create_user

    try:
        uid = create_user(conn, name, interests)
    except ValueError as exc:
        # a duplicate or empty name is the caller's to fix, so the reason is
        # surfaced rather than flattened to "internal error"
        raise ToolError(str(exc)) from exc
    return {
        "user_id": uid,
        "message": (
            f"created persona {name!r}. Ranking starts from its interests text; "
            "record_feedback calls will personalize it from the first click."
        ),
    }


@tool(needs_user=True, label="update_persona")
def _update_persona(conn, user_row, interests: str) -> dict:
    name = user_row["name"]
    conn.execute("UPDATE users SET interests = ? WHERE id = ?", (interests, user_row["id"]))
    conn.commit()
    # Without this, the embedder-down fallback in rank._profile_vector would
    # serve a vector computed from the interests text just replaced -- it
    # returns a cached entry without comparing hashes, by design.
    forget_profile_vector(conn, user_row["id"])
    return {"message": f"updated interests for {name!r}; ranking re-embeds on next use"}


@tool(empty={"prevalent_tags": []}, label="propose_interests")
def _propose_interests(conn, limit: int = 12) -> dict:
    tags = [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM item_tags GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT ?",
            (limit,),
        )
    ]
    return {
        "prevalent_tags": tags,
        "message": (
            "most common tags in the current feed; combine the relevant ones into "
            "an interests string and pass it to create_persona"
        ),
    }


@tool(
    empty={
        "user": None,
        "interests": None,
        "clicks": 0,
        "clicks_by_source": {},
        "blend_weight": 0.0,
        "top_liked": [],
        "top_disliked": [],
    },
    needs_user=True,
    label="profile_status",
)
def _profile_status(conn, user_row) -> dict:
    from attestation.features import _key_stats, _score
    from attestation.rank import blend_weight

    user = user_row["name"]
    n_clicks = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    stats = _key_stats(conn, user_row["id"])
    scored = sorted(((k, _score(stats, k)) for k in stats), key=lambda kv: kv[1], reverse=True)
    by_source = {
        r["source"]: r["n"]
        for r in conn.execute(
            "SELECT source, COUNT(*) AS n FROM clicks WHERE user_id = ? GROUP BY source",
            (user_row["id"],),
        )
    }
    return {
        "user": user,
        "interests": user_row["interests"],
        "clicks": n_clicks,
        "clicks_by_source": by_source,
        "blend_weight": round(blend_weight(n_clicks), 3),
        "top_liked": [k for k, v in scored[:5] if v > 0.5],
        "top_disliked": [k for k, v in reversed(scored[-5:]) if v < 0.5],
        "message": (
            f"{n_clicks} click(s); ranking is {round(blend_weight(n_clicks) * 100)}% "
            "driven by observed behavior and the rest by the interests text"
        ),
    }


# How much a query's semantic match counts against the reader's profile.
# Search is a directed question, so relevance to the query dominates -- but a
# profile still breaks ties, which is why a persona's search differs from a
# stranger's. Ranking purely by profile was the old bug: the query barely
# participated and results were "highest-ranked items that contain the string".
QUERY_WEIGHT = 0.75

# Keep semantic hits within this fraction of the BEST match for this query.
#
# Relative, not absolute, because absolute thresholds do not survive contact
# with real data. Measured against embeddinggemma over 5,167 items: "large
# language models" tops out at cosine 0.619 while "cryo-EM protein structure"
# tops at 0.443, so any fixed cutoff either floods one query or starves the
# other. Similarity also decays slowly -- 0.619 to 0.500 across 200 items --
# so "everything above X" is never a clean answer.
#
# At 0.90 the same measurement gives 28 hits for a broad query, 12 for a
# middling one, and 2 for "superconductivity", which has exactly two genuinely
# superconducting papers in the corpus. The floor adapts to how well the
# archive actually covers the question, which is the property that matters.
RELEVANCE_FLOOR = 0.90


def _semantic_hits(conn, embedder, query: str, k: int) -> dict[int, float]:
    """item_id -> similarity, via the sqlite-vec index.

    Indexed with DOC_PROMPT and searched with QUERY_PROMPT: embed.py's prompts
    are asymmetric because the model was trained that way, and mixing them
    measurably degrades retrieval. `embed_query` existed for exactly this and
    had no caller until now.

    Vectors are L2-normalised by truncate_normalize, so sqlite-vec's L2
    distance d relates to cosine similarity as cos = 1 - d^2/2.
    """
    vec = embedder.embed_query(query)
    rows = conn.execute(
        "SELECT rowid, distance FROM item_vectors"
        " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (vec.tobytes(), k),
    ).fetchall()
    sims = {r["rowid"]: 1.0 - (r["distance"] ** 2) / 2.0 for r in rows}
    if not sims:
        return sims
    # sqlite-vec returns k rows whether or not they are relevant, so without a
    # floor a search for anything returns the whole archive in similarity
    # order -- the same "returns everything" failure as the old substring
    # search, only harder to notice.
    best = max(sims.values())
    return {rid: sim for rid, sim in sims.items() if sim >= best * RELEVANCE_FLOOR}


def _passes_filters(item, needle: str, similarity: dict, tag, content_type) -> bool:
    """Whether an item survives the query and the explicit filters.

    A query keeps an item if the semantic index reached it OR the words appear
    literally -- the two find different things, which is why both run.

    Tags compare CANONICAL to CANONICAL. kg_concepts hands an agent the graph's
    aliased names, and the documented workflow is to feed one straight back in
    here; comparing it against the raw stored spellings made that workflow
    under-return badly. On the live corpus `hugging-face` reached 23 of the 380
    items carrying that concept, because the other 357 are stored as
    `huggingface`, and `large-language-models` reached 226 of 1,072.
    """
    from attestation.kg import canonical

    if needle and item.item_id not in similarity and not _literal_match(item, needle):
        return False
    if tag and canonical(tag) not in {canonical(t) for t in (item.tags or [])}:
        return False
    return not (content_type and item.content_type != content_type)


def _literal_match(item, needle: str) -> bool:
    return needle in (item.title or "").lower() or needle in (item.summary or "").lower()


def _score_matches(kept: list, ranked: list, needle: str, similarity: dict) -> list:
    """Blend query relevance with the reader's profile, best first.

    The profile is a tie-breaker, not the ranking: search is a directed
    question. Ordering by profile and filtering afterwards was the old bug --
    results were the highest-ranked items that happened to contain the string,
    so the query barely participated.
    """
    profile_rank = {item.item_id: i for i, item in enumerate(ranked)}
    n = max(len(ranked), 1)
    scored = []
    for item in kept:
        literal = bool(needle) and _literal_match(item, needle)
        sim = similarity.get(item.item_id)
        relevance = sim if sim is not None else 0.0
        if literal:
            # A boost, not a floor. Flooring every literal hit at one value
            # made 711 items matching "llm" tie at the same score, so profile
            # rank silently decided the order and the query stopped
            # discriminating. A title match outweighs a body match, since a
            # body can mention a term in passing.
            relevance += 0.35 if needle in (item.title or "").lower() else 0.15
        profile_score = 1.0 - (profile_rank[item.item_id] / n)
        combined = QUERY_WEIGHT * min(relevance, 1.0) + (1.0 - QUERY_WEIGHT) * profile_score
        how = "both" if literal and sim is not None else ("literal" if literal else "semantic")
        scored.append((combined, item, how))
    scored.sort(key=lambda t: (-t[0], t[1].item_id))
    return scored


@tool(empty={"items": [], "ranking_quality": {}}, needs_user=True, label="search_feed")
def _search_feed(
    conn,
    user_row,
    query: str,
    tag: str | None = None,
    content_type: str | None = None,
    limit: int = 10,
) -> dict:
    from attestation.rank import rank_items

    limit = min(max(int(limit), 1), MAX_LIST_LIMIT)
    ranked = rank_items(
        conn, get_embedder(), user_row["id"], since_days=None, exclude_clicked=False
    )
    clicked = {
        r["item_id"]
        for r in conn.execute("SELECT item_id FROM clicks WHERE user_id = ?", (user_row["id"],))
    }

    # An empty query is a filter, not a search: keep profile order and let the
    # tag/content_type filters do the work.
    similarity: dict[int, float] = {}
    if query.strip():
        # Over-fetch so post-filtering by tag/content_type still has candidates.
        similarity = _semantic_hits(
            conn, get_embedder(), query, k=min(len(ranked), max(limit * 10, 100))
        )

    needle = query.lower().strip()
    kept = [item for item in ranked if _passes_filters(item, needle, similarity, tag, content_type)]
    scored = _score_matches(kept, ranked, needle, similarity)
    matches = [
        {
            "item_id": item.item_id,
            "title": item.title,
            "url": item.url,
            "source": item.source,
            "tags": item.tags,
            "content_type": item.content_type,
            "already_rated": item.item_id in clicked,
            # How this item was found, so a caller can tell relevance from
            # noise when a result looks surprising.
            "match": how if needle else "filter",
            "relevance": round(score, 4),
        }
        for score, item, how in scored[:limit]
    ]
    mode = "filtered" if not needle else "searched"
    return {
        "message": f"{len(matches)} item(s) {mode}, best first",
        "items": matches,
        "ranking_quality": ranking_quality(conn, user_row["id"]),
    }


@tool(needs_user=True, label="delete_persona")
def _delete_persona(conn, user_row, confirm: bool = False) -> dict:
    name = user_row["name"]
    n = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    if not confirm:
        raise ToolError(
            f"refusing to delete {name!r} without confirm=true. This would "
            f"permanently remove the persona and its {n} click(s) of training data."
        )
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_row["id"],))
    # users.id is a rowid alias SQLite reuses after the highest-id row is
    # deleted -- without this, a future persona created at the same id
    # would inherit this persona's cached explanations verbatim.
    conn.execute("DELETE FROM explanations WHERE user_id = ?", (user_row["id"],))
    conn.execute("DELETE FROM users WHERE id = ?", (user_row["id"],))
    conn.commit()
    forget_profile_vector(conn, user_row["id"])
    return {"message": f"deleted persona {name!r} and its {n} click(s)"}


@tool(needs_user=True, label="reset_feedback")
def _reset_feedback(conn, user_row, confirm: bool = False) -> dict:
    name = user_row["name"]
    n = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    if not confirm:
        raise ToolError(
            f"refusing to reset {name!r} without confirm=true. This would erase "
            f"{n} click(s); the persona and its interests text would be kept."
        )
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_row["id"],))
    conn.commit()
    return {"message": f"cleared {n} click(s) for {name!r}"}


def _digest(user: str, days: int = 7, per_topic: int = 3, limit: int = 30) -> dict:
    """`window_days` echoes the window that was ASKED for, on failure as well as
    success -- a caller that got nothing back still needs to know which window
    came up empty. `empty` cannot carry a per-call value, so it is filled in here.
    """
    out = _digest_body(user, days, per_topic, limit)
    out["window_days"] = out.get("window_days") or days
    return out


def _best_cluster(tags: set, members: list) -> tuple[str | None, int]:
    """The cluster sharing the most tags with an item.

    Ties break on label, lexically smallest first, so repeated calls agree --
    the digest is read by a model and an order that shuffles between identical
    calls reads as new information.
    """
    best_label, best_n = None, 0
    for label, concepts in members:
        overlap = len(tags & concepts)
        if overlap > best_n or (overlap == best_n and overlap and label < (best_label or "~")):
            best_label, best_n = label, overlap
    return best_label, best_n


def _cluster(items: list, members: list, cached: dict) -> tuple[dict[str, list], list]:
    """Group items by their strongest tag overlap, attaching cached explanations.

    Items matching no cluster are returned separately rather than dropped: a
    digest that silently omits them would misreport how much was read.
    """
    grouped: dict[str, list] = {}
    unclustered: list = []
    for item in items:
        label, overlap = _best_cluster(set(item.get("tags") or []), members)
        enriched = dict(item)
        if item["item_id"] in cached:
            enriched["explanation"] = cached[item["item_id"]]
        if overlap and label is not None:
            grouped.setdefault(label, []).append(enriched)
        else:
            unclustered.append(enriched)
    return grouped, unclustered


@tool(
    empty={"topics": [], "unclustered": [], "ranking_quality": {}, "window_days": 0},
    needs_user=True,
    label="digest",
)
def _digest_body(conn, user_row, days: int = 7, per_topic: int = 3, limit: int = 30) -> dict:
    from attestation import kg

    # digest alone used to answer an unknown user with `unknown user 'x'`,
    # where the other nine name the valid personas. That was an inconsistency,
    # not a choice: the caller is a model, and the list is the thing it can act
    # on. It now goes through needs_user like the rest.
    row = user_row
    # the caller's connection is reused deliberately: digest must not open a
    # second one to rank the same feed
    items_ranked = ranked_items(conn, row, min(limit, MAX_LIST_LIMIT), days)
    items = [_item_row(it) for it in items_ranked]
    if not items:
        raise ToolError("no unread items to digest")

    communities = kg.communities(conn, min_size=3)
    members = [(c["label"], set(c["members"])) for c in communities]
    cached = {
        r["item_id"]: r["text"]
        for r in conn.execute(
            "SELECT item_id, text FROM explanations WHERE user_id = ?", (row["id"],)
        )
    }

    grouped, unclustered = _cluster(items, members, cached)

    topics = [
        {
            "label": label,
            # n_total vs the shown slice: truncation must be visible
            "n_total": len(group),
            "items": group[: max(1, int(per_topic))],
        }
        for label, group in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    return {
        "message": f"{len(items)} item(s) in {len(topics)} topic(s)",
        "topics": topics,
        "unclustered": unclustered,
        "ranking_quality": ranking_quality(conn, row["id"]),
        "window_days": days,
    }


@tool(
    empty={"counts": {}, "reactions": []},
    needs_user=True,
    label="simulate_feedback",
)
def _simulate_feedback(conn, user_row, limit: int = 10, confirm: bool = False) -> dict:
    from attestation.llm import default_chat_fn
    from attestation.simulate import simulate_feedback as run_simulation
    from attestation.simulate import source_skew_caveat

    if not confirm:
        raise ToolError(
            "refusing to simulate without confirm=true. This writes click rows"
            f" for {user_row['name']!r} (marked source='simulated', so they stay"
            " distinguishable from real feedback) and calls a local LLM once per"
            " item."
        )
    limit = min(max(int(limit), 1), MAX_LIST_LIMIT)
    # Sample round-robin across FEEDS, not down the ranking.
    #
    # Ranked candidates are the wrong pool twice over. They are ordered by
    # relevance to this very persona, so the top of the list asks the model to
    # reject what the ranker just selected for relevance; and the archive is
    # lopsided -- 1,744 of any 2,000 candidates here are arXiv cs.LG, whose
    # subject matter IS matt's stated interest, so walking further down samples
    # more of the same. Measured: 1 negative in 12, and that one's own
    # reasoning called the paper "highly relevant".
    #
    # Feed is the axis that actually varies. Taking the newest few from each in
    # turn puts chemistry, neuroscience and general news in front of a reader
    # of ML papers, and the same model rejected 7 of 8 such items at full
    # confidence. A simulated reader can only reject what it is shown.
    rows = conn.execute(
        "SELECT id, title, summary FROM ("
        "  SELECT i.id, i.title, i.summary,"
        "         ROW_NUMBER() OVER (PARTITION BY i.feed_id ORDER BY i.published DESC) rn"
        "  FROM items i"
        "  WHERE i.summary IS NOT NULL AND i.summary != ''"
        "    AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)"
        ") WHERE rn <= ? ORDER BY rn, id LIMIT ?",
        (user_row["id"], max(limit // 3, 2), limit),
    ).fetchall()
    if not rows:
        raise ToolError("no unrated items to react to -- run an ingest first")
    out = run_simulation(conn, default_chat_fn, user_row["name"], rows)
    counts = out["counts"]
    caveat = source_skew_caveat(conn, user_row["id"])
    return {
        "message": (
            f"{counts['useful']} useful, {counts['not_useful']} not-useful"
            f" ({counts['skipped_unsure']} unsure, {counts['failed']} failed)"
            + (f". {caveat}" if caveat else "")
        ),
        "counts": counts,
        "reactions": out["reactions"],
        "caveat": caveat,
    }


@tool(
    empty={"candidates": 0, "recorded": 0, "skipped_already_rated": 0},
    needs_user=True,
    label="harvest_engagement",
)
def _harvest_engagement(conn, user_row) -> dict:
    from attestation.implicit import harvest

    out = harvest(conn, user_row["name"])
    return {
        "message": (
            f"recorded {out['recorded']} weak positive(s) from engagement"
            f" ({out['skipped_already_rated']} already rated)"
        ),
        **out,
    }
