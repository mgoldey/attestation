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
from attestation.mcp._tool import ToolError, tool


def register(mcp) -> None:
    """Attach the feed.source_* tools."""

    @mcp.tool(name="feed.source_add")
    def add_feed(url: str, title: str | None = None) -> dict:
        """Subscribe to an RSS/Atom feed.

        Validates that the URL parses as a feed, then registers it. Does NOT fetch
        its articles: items appear after the next ingest (hourly cron, or
        `attest ingest`). Use `feed.source_preview` first to check a feed's content.
        """
        return _add_feed(url, title)

    @mcp.tool(name="feed.sources")
    def list_feeds() -> dict:
        """List subscribed feeds with item counts and when each was last fetched."""
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
def _remove_feed(conn, feed_id: ItemId, confirm: bool = False) -> dict:
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
