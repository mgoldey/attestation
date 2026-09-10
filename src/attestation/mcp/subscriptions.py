"""Feed subscriptions: the `feed.source_*` tools.

Split out of `feed.py`, which held nineteen tools across ranking, search,
digests, personas, explanations, feedback and subscriptions -- one namespace
holding six concerns, and it hit its size cap five times in two days.

Subscriptions were the seam the code itself drew: every tool here does
`from attestation import feeds as feeds_mod`, and none of them touches
ranking, the embedder, or a persona's click history. Nothing else in the
domain imports that module.

They keep the `feed.` prefix because that is what a caller looks for -- the
namespace is the agent's map, and moving a tool between files must not move
it between namespaces.
"""

from attestation.mcp._shared import MAX_LIST_LIMIT, ItemId, Limit
from attestation.mcp._tool import ToolError, _get_user, tool


def register(mcp) -> None:
    """Attach the feed.source_* tools."""

    @mcp.tool(name="feed.source_add")
    def add_feed(url: str, title: str | None = None, user: str | None = None) -> dict:
        """Subscribe to an RSS/Atom feed, or register a standing research topic.

        An RSS URL is validated by parsing it, then registered. A topic is a
        `research:` URL -- `research:arxiv,pubmed?q=graph+neural+networks`, or
        `research:crossref?q=...&journal=Nature+Methods` -- validated without any
        network call; every hourly ingest then searches arXiv, PubMed and/or
        CrossRef for it and the hits enter the feed like any item. Does NOT fetch
        anything now: items appear after the next ingest. `user` records who
        asked (feed.sources shows it); the feed itself is shared by every reader.
        """
        return _add_feed(url, title, user)

    @mcp.tool(name="feed.sources")
    def list_feeds() -> dict:
        """List subscribed feeds and research topics (`kind`), with item counts.

        Also reports who added each and when each was last fetched.
        """
        return _list_feeds()

    @mcp.tool(name="feed.source_remove")
    def remove_feed(feed_id: ItemId, confirm: bool = False) -> dict:
        """Unsubscribe from a feed. Requires confirm=true.

        Existing items and all feedback on them are KEPT -- only the subscription
        is removed, so no ranking history is lost.
        """
        return _remove_feed(feed_id, confirm)

    @mcp.tool(name="feed.source_preview")
    def preview_feed(url: str, limit: Limit = 5) -> dict:
        """Show recent entries from a feed WITHOUT subscribing to it."""
        return _preview_feed(url, limit)

    @mcp.tool(name="feed.source_suggest")
    def suggest_feeds(user: str, limit: Limit = 5) -> dict:
        """Suggest feeds from a curated list, scored against tags this user liked."""
        return _suggest_feeds(user, limit)


@tool(empty={"feed_id": None}, label="add_feed")
def _add_feed(conn, url: str, title: str | None = None, user: str | None = None) -> dict:
    from attestation import feeds as feeds_mod

    row = _get_user(conn, user) if user else None
    added_by = row["id"] if row else None
    try:
        feed_id, message = feeds_mod.add_source(conn, url, title, added_by=added_by)
    except feeds_mod.FeedError as exc:
        raise ToolError(str(exc)) from exc
    return {"message": message, "feed_id": feed_id}


@tool(empty={"feeds": []}, label="list_feeds")
def _list_feeds(conn) -> dict:
    from attestation import feeds as feeds_mod

    feeds = feeds_mod.list_sources(conn)
    return {"message": f"{len(feeds)} feed(s)", "feeds": feeds}


@tool(empty={"orphaned_items": 0}, label="remove_feed")
def _remove_feed(conn, feed_id: ItemId, confirm: bool = False) -> dict:
    from attestation import feeds as feeds_mod

    if not confirm:
        raise ToolError(
            f"refusing to remove feed {feed_id} without confirm=true. This "
            "unsubscribes the feed; its existing items and your feedback on "
            "them are kept."
        )
    try:
        orphaned_items, message = feeds_mod.remove_source(conn, feed_id)
    except feeds_mod.FeedError as exc:
        raise ToolError(str(exc)) from exc
    return {"message": message, "orphaned_items": orphaned_items}


@tool(empty={"title": None, "entries": []}, needs_db=False, label="preview_feed")
def _preview_feed(url: str, limit: int = 5) -> dict:
    from attestation import feeds as feeds_mod

    try:
        out = feeds_mod.preview_source(url, limit=min(limit, MAX_LIST_LIMIT))
    except feeds_mod.FeedError as exc:
        raise ToolError(str(exc)) from exc
    return {"message": out["message"], "title": out["title"], "entries": out["entries"]}


@tool(empty={"suggestions": []}, needs_user=True, label="suggest_feeds")
def _suggest_feeds(conn, user_row, limit: int = 5) -> dict:
    from attestation import feeds as feeds_mod

    return {
        "message": "scored against tags you marked useful",
        "suggestions": feeds_mod.suggest_sources(conn, user_row["id"], limit=limit),
    }
