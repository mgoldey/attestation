"""MCP stdio server: exposes the recommender as native tool-calling for MCP clients
(e.g. hermes-agent) instead of the curl-based science-recommendations skill.

Each tool opens its own short-lived DB connection via resolve_db_path(None), so this
process stays stateless between calls and honors RSS_DB like the CLI/server do.
The embedder is constructed lazily and shared across calls (it's just an httpx client).
"""

import contextlib
import logging

from mcp.server.fastmcp import FastMCP

from attestation.db import get_db, resolve_db_path
from attestation.explain import explain as explain_item_fn
from attestation.llm import default_chat_fn
from attestation.rank import _PROFILE_VEC_CACHE, _db_identity, get_user, rank_items, record_click

log = logging.getLogger(__name__)

MAX_LIST_LIMIT = 50

mcp = FastMCP("attestation")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from attestation.embed import Embedder

        _embedder = Embedder()
    return _embedder


@contextlib.contextmanager
def open_db(path=None):
    """One connection per tool call: open, yield, always close.

    sqlite3.Connection is not itself a closing context manager -- its
    __enter__/__exit__ manage transactions, not the handle -- so this wraps
    get_db(resolve_db_path(path)) in a real close-on-exit contract and
    replaces the 26 repeated open/try/finally blocks that used to do this by
    hand.
    """
    conn = get_db(resolve_db_path(path))
    try:
        yield conn
    finally:
        conn.close()


def _valid_users(conn) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]


def _unknown_user_message(conn, user: str) -> str:
    valid = _valid_users(conn)
    return f"unknown user: {user!r}. Valid users: {', '.join(valid) if valid else '(none seeded)'}"


def _ranked_items(conn, user_row, limit: int, since_days: int | None) -> list:
    """Rank items for an already-resolved user row against a connection the
    caller owns. Shared by _list_feed_impl and _digest_impl so digest does not
    open a second connection to rank the same feed (see module docstring)."""
    return rank_items(conn, _get_embedder(), user_row["id"], since_days)[:limit]


def _list_feed_impl(user: str, limit: int = 10, since_days: int | None = 14) -> dict:
    """Shared implementation for the list_feed tool; kept import-testable without FastMCP.

    `since_days` defaults to rank_items' own 14-day window so list_feed's
    behavior is unchanged; digest passes its `days` through here.
    """
    limit = min(max(int(limit), 1), MAX_LIST_LIMIT)
    empty = {"items": [], "ranking_quality": {}}
    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, user), **empty}
            items = _ranked_items(conn, row, limit, since_days)
            return {
                "ok": True,
                "message": f"{len(items)} item(s), best first",
                "items": [
                    {
                        "item_id": it.item_id,
                        "title": it.title,
                        "url": it.url,
                        "source": it.source,
                        "score": it.score,
                        "tags": it.tags,
                        "content_type": it.content_type,
                    }
                    for it in items
                ],
                "ranking_quality": _ranking_quality(conn, row["id"]),
            }
        except Exception:
            log.exception("list_feed failed for user=%s", user)
            return {"ok": False, "message": "internal error ranking feed; see server logs", **empty}


def _record_feedback_impl(user: str, item_id: int, useful: bool) -> dict:
    """Shared implementation for the record_feedback tool."""
    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, user)}
            item = conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
            if item is None:
                return {"ok": False, "message": f"unknown item_id: {item_id}"}
            record_click(conn, row["id"], item_id, useful, source="agent")
            return {
                "ok": True,
                "message": f"recorded useful={useful} for item {item_id} (user {user}); "
                "ranking will reflect this on the next list_feed call",
            }
        except Exception:
            log.exception("record_feedback failed for user=%s item_id=%s", user, item_id)
            return {"ok": False, "message": "internal error recording feedback; see server logs"}


def _explain_item_impl(user: str, item_id: int) -> dict:
    """Shared implementation for the explain_item tool."""
    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {
                    "ok": False,
                    "message": _unknown_user_message(conn, user),
                    "explanation": None,
                }
            item = conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
            if item is None:
                return {"ok": False, "message": f"unknown item_id: {item_id}", "explanation": None}
            text = explain_item_fn(conn, row["id"], item_id, chat_fn=default_chat_fn)
            return {"ok": True, "message": "", "explanation": text}
        except Exception:
            log.exception("explain_item failed for user=%s item_id=%s", user, item_id)
            return {
                "ok": False,
                "message": "internal error generating explanation; see server logs",
                "explanation": None,
            }


def _list_users_impl() -> dict:
    """Shared implementation for the list_users tool."""
    with open_db() as conn:
        try:
            rows = conn.execute("SELECT name, interests FROM users ORDER BY name").fetchall()
            return {
                "ok": True,
                "message": f"{len(rows)} user(s)",
                "users": [{"name": r["name"], "interests": r["interests"]} for r in rows],
            }
        except Exception:
            log.exception("list_users failed")
            return {
                "ok": False,
                "message": "internal error listing users; see server logs",
                "users": [],
            }


def _add_feed_impl(url: str, title: str | None = None) -> dict:
    from attestation import feeds as feeds_mod

    with open_db() as conn:
        try:
            return feeds_mod.add_feed(conn, url, title)
        except Exception:
            log.exception("add_feed failed for url=%s", url)
            return {"ok": False, "feed_id": None, "message": "internal error adding feed"}


def _list_feeds_impl() -> dict:
    from attestation import feeds as feeds_mod

    with open_db() as conn:
        try:
            feeds = feeds_mod.list_feeds(conn)
            return {"ok": True, "message": f"{len(feeds)} feed(s)", "feeds": feeds}
        except Exception:
            log.exception("list_feeds failed")
            return {
                "ok": False,
                "message": "internal error listing feeds; see server logs",
                "feeds": [],
            }


def _remove_feed_impl(feed_id: int, confirm: bool = False) -> dict:
    from attestation import feeds as feeds_mod

    if not confirm:
        return {
            "ok": False,
            "message": (
                f"refusing to remove feed {feed_id} without confirm=true. This "
                "unsubscribes the feed; its existing items and your feedback on "
                "them are kept."
            ),
            "orphaned_items": 0,
        }
    with open_db() as conn:
        try:
            return feeds_mod.remove_feed(conn, feed_id)
        except Exception:
            log.exception("remove_feed failed for feed_id=%s", feed_id)
            return {
                "ok": False,
                "message": "internal error removing feed; see server logs",
                "orphaned_items": 0,
            }


def _preview_feed_impl(url: str, limit: int = 5) -> dict:
    from attestation import feeds as feeds_mod

    try:
        return feeds_mod.preview_feed(url, limit=min(limit, MAX_LIST_LIMIT))
    except Exception:
        log.exception("preview_feed failed for url=%s", url)
        return {
            "ok": False,
            "message": "internal error previewing feed",
            "title": None,
            "entries": [],
        }


def _suggest_feeds_impl(user: str, limit: int = 5) -> dict:
    from attestation import feeds as feeds_mod

    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {
                    "ok": False,
                    "message": _unknown_user_message(conn, user),
                    "suggestions": [],
                }
            return {
                "ok": True,
                "message": "scored against tags you marked useful",
                "suggestions": feeds_mod.suggest_feeds(conn, row["id"], limit=limit),
            }
        except Exception:
            log.exception("suggest_feeds failed for user=%s", user)
            return {
                "ok": False,
                "message": "internal error suggesting feeds; see server logs",
                "suggestions": [],
            }


@mcp.tool()
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
    return _list_feed_impl(user, limit, since_days)


@mcp.tool()
def record_feedback(user: str, item_id: int, useful: bool) -> dict:
    """Record whether a feed item was useful for this user, retraining their ranking.

    This writes (or overwrites) a single click record for (user, item_id) --
    calling it again for the same item just replaces the previous verdict, so it
    is safe to call repeatedly. The next list_feed call for this user will
    reflect the updated ranking once enough mixed feedback has accumulated.
    """
    return _record_feedback_impl(user, item_id, useful)


@mcp.tool()
def explain_item(user: str, item_id: int) -> dict:
    """Explain in one sentence why a specific feed item was ranked for this user.

    SLOW: this calls a local LLM (via the configured OpenAI-compatible backend)
    on first request for a given (user, item_id) pair and can take several
    seconds to ~1-2 minutes depending on the chat model and hardware; results
    are cached afterward and return instantly. Only call this for items the
    user is asking about specifically, not for every item in a list.
    """
    return _explain_item_impl(user, item_id)


@mcp.tool()
def list_users() -> dict:
    """List all available reader personas (users) and their interest profiles.

    Use this to discover which `user` values are valid for list_feed,
    record_feedback, and explain_item before calling them.
    """
    return _list_users_impl()


@mcp.tool()
def add_feed(url: str, title: str | None = None) -> dict:
    """Subscribe to an RSS/Atom feed.

    Validates that the URL parses as a feed, then registers it. Does NOT fetch
    its articles: items appear after the next ingest (hourly cron, or
    `hermes ingest`). Use preview_feed first to check a feed's content.
    """
    return _add_feed_impl(url, title)


@mcp.tool()
def list_feeds() -> dict:
    """List subscribed feeds with item counts and when each was last fetched."""
    return _list_feeds_impl()


@mcp.tool()
def remove_feed(feed_id: int, confirm: bool = False) -> dict:
    """Unsubscribe from a feed. Requires confirm=true.

    Existing items and all feedback on them are KEPT -- only the subscription
    is removed, so no ranking history is lost.
    """
    return _remove_feed_impl(feed_id, confirm)


@mcp.tool()
def preview_feed(url: str, limit: int = 5) -> dict:
    """Show recent entries from a feed WITHOUT subscribing to it."""
    return _preview_feed_impl(url, limit)


@mcp.tool()
def suggest_feeds(user: str, limit: int = 5) -> dict:
    """Suggest feeds from a curated list, scored against tags this user liked."""
    return _suggest_feeds_impl(user, limit)


def _create_persona_impl(name: str, interests: str) -> dict:
    from attestation.rank import create_user

    with open_db() as conn:
        try:
            uid = create_user(conn, name, interests)
            return {
                "ok": True,
                "user_id": uid,
                "message": (
                    f"created persona {name!r}. Ranking starts from its interests text; "
                    "record_feedback calls will personalize it from the first click."
                ),
            }
        except ValueError as exc:
            return {"ok": False, "user_id": None, "message": str(exc)}
        except Exception:
            log.exception("create_persona failed for name=%s", name)
            return {
                "ok": False,
                "user_id": None,
                "message": "internal error creating persona; see server logs",
            }


def _update_persona_impl(name: str, interests: str) -> dict:
    with open_db() as conn:
        try:
            row = get_user(conn, name)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, name)}
            conn.execute("UPDATE users SET interests = ? WHERE id = ?", (interests, row["id"]))
            conn.commit()
            return {
                "ok": True,
                "message": f"updated interests for {name!r}; ranking re-embeds on next use",
            }
        except Exception:
            log.exception("update_persona failed for name=%s", name)
            return {"ok": False, "message": "internal error updating persona; see server logs"}


def _propose_interests_impl(limit: int = 12) -> dict:
    with open_db() as conn:
        try:
            tags = [
                r["tag"]
                for r in conn.execute(
                    "SELECT tag FROM item_tags GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT ?",
                    (limit,),
                )
            ]
            return {
                "ok": True,
                "prevalent_tags": tags,
                "message": (
                    "most common tags in the current feed; combine the relevant ones into "
                    "an interests string and pass it to create_persona"
                ),
            }
        except Exception:
            log.exception("propose_interests failed for limit=%s", limit)
            return {
                "ok": False,
                "prevalent_tags": [],
                "message": "internal error proposing interests; see server logs",
            }


def _profile_status_impl(user: str) -> dict:
    from attestation.features import _key_stats, _score
    from attestation.rank import blend_weight

    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, user)}
            n_clicks = conn.execute(
                "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (row["id"],)
            ).fetchone()["c"]
            stats = _key_stats(conn, row["id"])
            scored = sorted(
                ((k, _score(stats, k)) for k in stats), key=lambda kv: kv[1], reverse=True
            )
            by_source = {
                r["source"]: r["n"]
                for r in conn.execute(
                    "SELECT source, COUNT(*) AS n FROM clicks WHERE user_id = ? GROUP BY source",
                    (row["id"],),
                )
            }
            return {
                "ok": True,
                "user": user,
                "interests": row["interests"],
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
        except Exception:
            log.exception("profile_status failed for user=%s", user)
            return {
                "ok": False,
                "user": user,
                "interests": None,
                "clicks": 0,
                "clicks_by_source": {},
                "blend_weight": 0.0,
                "top_liked": [],
                "top_disliked": [],
                "message": "internal error computing profile status; see server logs",
            }


def _search_feed_impl(
    user: str,
    query: str,
    tag: str | None = None,
    content_type: str | None = None,
    limit: int = 10,
) -> dict:
    from attestation.rank import rank_items

    limit = min(max(int(limit), 1), MAX_LIST_LIMIT)
    empty = {"items": [], "ranking_quality": {}}
    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, user), **empty}

            ranked = rank_items(
                conn, _get_embedder(), row["id"], since_days=None, exclude_clicked=False
            )
            clicked = {
                r["item_id"]
                for r in conn.execute("SELECT item_id FROM clicks WHERE user_id = ?", (row["id"],))
            }
            needle = query.lower()
            matches = []
            for item in ranked:
                if (
                    needle
                    and needle not in (item.title or "").lower()
                    and needle not in (item.summary or "").lower()
                ):
                    continue
                if tag and tag not in (item.tags or []):
                    continue
                if content_type and item.content_type != content_type:
                    continue
                matches.append(
                    {
                        "item_id": item.item_id,
                        "title": item.title,
                        "url": item.url,
                        "source": item.source,
                        "tags": item.tags,
                        "content_type": item.content_type,
                        "already_rated": item.item_id in clicked,
                    }
                )
                if len(matches) >= limit:
                    break
            return {
                "ok": True,
                "message": f"{len(matches)} match(es), best first",
                "items": matches,
                "ranking_quality": _ranking_quality(conn, row["id"]),
            }
        except Exception:
            log.exception("search_feed failed for user=%s query=%s", user, query)
            return {
                "ok": False,
                "message": "internal error searching feed; see server logs",
                **empty,
            }


def _delete_persona_impl(name: str, confirm: bool = False) -> dict:
    with open_db() as conn:
        try:
            row = get_user(conn, name)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, name)}
            n = conn.execute(
                "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (row["id"],)
            ).fetchone()["c"]
            if not confirm:
                return {
                    "ok": False,
                    "message": (
                        f"refusing to delete {name!r} without confirm=true. This would "
                        f"permanently remove the persona and its {n} click(s) of training data."
                    ),
                }
            conn.execute("DELETE FROM clicks WHERE user_id = ?", (row["id"],))
            # users.id is a rowid alias SQLite reuses after the highest-id row is
            # deleted -- without this, a future persona created at the same id
            # would inherit this persona's cached explanations verbatim.
            conn.execute("DELETE FROM explanations WHERE user_id = ?", (row["id"],))
            conn.execute("DELETE FROM users WHERE id = ?", (row["id"],))
            conn.commit()
            _PROFILE_VEC_CACHE.pop((_db_identity(conn), row["id"]), None)
            return {"ok": True, "message": f"deleted persona {name!r} and its {n} click(s)"}
        except Exception:
            log.exception("delete_persona failed for name=%s", name)
            return {"ok": False, "message": "internal error deleting persona; see server logs"}


def _reset_feedback_impl(name: str, confirm: bool = False) -> dict:
    with open_db() as conn:
        try:
            row = get_user(conn, name)
            if row is None:
                return {"ok": False, "message": _unknown_user_message(conn, name)}
            n = conn.execute(
                "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (row["id"],)
            ).fetchone()["c"]
            if not confirm:
                return {
                    "ok": False,
                    "message": (
                        f"refusing to reset {name!r} without confirm=true. This would erase "
                        f"{n} click(s); the persona and its interests text would be kept."
                    ),
                }
            conn.execute("DELETE FROM clicks WHERE user_id = ?", (row["id"],))
            conn.commit()
            return {"ok": True, "message": f"cleared {n} click(s) for {name!r}"}
        except Exception:
            log.exception("reset_feedback failed for name=%s", name)
            return {"ok": False, "message": "internal error resetting feedback; see server logs"}


@mcp.tool()
def create_persona(name: str, interests: str) -> dict:
    """Create a reader persona from a name and an interests description.

    Ranking starts from the interests text and personalizes from the first
    record_feedback call. Use propose_interests first if you want suggestions
    drawn from what is actually in the feed.
    """
    return _create_persona_impl(name, interests)


@mcp.tool()
def update_persona(name: str, interests: str) -> dict:
    """Replace a persona's interests text; re-steers ranking immediately."""
    return _update_persona_impl(name, interests)


@mcp.tool()
def propose_interests(limit: int = 12) -> dict:
    """List the most common tags in the feed, to help write an interests string."""
    return _propose_interests_impl(limit)


@mcp.tool()
def profile_status(user: str) -> dict:
    """Show how well-trained a persona is: click count, how much ranking is
    driven by behavior vs the written interests, and top liked/disliked tags."""
    return _profile_status_impl(user)


@mcp.tool()
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
    return _search_feed_impl(user, query, tag, content_type, limit)


@mcp.tool()
def delete_persona(name: str, confirm: bool = False) -> dict:
    """Delete a persona AND all its feedback. Requires confirm=true. Irreversible."""
    return _delete_persona_impl(name, confirm)


@mcp.tool()
def reset_feedback(name: str, confirm: bool = False) -> dict:
    """Erase a persona's clicks but keep the persona. Requires confirm=true."""
    return _reset_feedback_impl(name, confirm)


# ---------------------------------------------------------------------------
# symbolic math (SymPy) -- every evaluation is subprocess-isolated
# ---------------------------------------------------------------------------


def _sym_call(op_name: str, payload: dict, timeout: int) -> dict:
    """Shared bridge: run an op in isolation and flatten it into the tool contract."""
    from attestation.symbolic import run_isolated

    outcome = run_isolated(op_name, payload, timeout)
    if not outcome["ok"]:
        return {"ok": False, "message": outcome["error"], "result": None, "latex": None}
    value = outcome["value"]
    return {"ok": True, "message": "", **value}


def _sym_simplify_impl(expr: str, timeout: int = 10) -> dict:
    return _sym_call("op_simplify", {"expr": expr}, timeout)


def _sym_solve_impl(expr: str, symbol: str | None = None, timeout: int = 10) -> dict:
    return _sym_call("op_solve", {"expr": expr, "symbol": symbol}, timeout)


def _sym_differentiate_impl(
    expr: str, symbol: str | None = None, order: int = 1, timeout: int = 10
) -> dict:
    return _sym_call("op_differentiate", {"expr": expr, "symbol": symbol, "order": order}, timeout)


def _sym_integrate_impl(
    expr: str, symbol: str | None = None, bounds: list | None = None, timeout: int = 10
) -> dict:
    return _sym_call("op_integrate", {"expr": expr, "symbol": symbol, "bounds": bounds}, timeout)


def _sym_derivation_impl(
    expr: str, operation: str = "integrate", symbol: str | None = None, timeout: int = 10
) -> dict:
    return _sym_call(
        "op_derivation", {"expr": expr, "operation": operation, "symbol": symbol}, timeout
    )


def _sym_verify_impl(lhs: str, rhs: str, timeout: int = 10) -> dict:
    return _sym_call("op_verify", {"lhs": lhs, "rhs": rhs}, timeout)


def _sym_evaluate_impl(
    expr: str, subs: dict | None = None, units: str | None = None, timeout: int = 10
) -> dict:
    return _sym_call("op_evaluate", {"expr": expr, "subs": subs, "units": units}, timeout)


@mcp.tool()
def sym_simplify(expr: str, timeout: int = 10) -> dict:
    """Simplify a mathematical expression to canonical form.

    Example: "(x**2 - 1)/(x - 1)" -> "x + 1". Returns the result as text and
    LaTeX, plus how the input was parsed so a misread is visible.
    """
    return _sym_simplify_impl(expr, timeout)


@mcp.tool()
def sym_solve(expr: str, symbol: str | None = None, timeout: int = 10) -> dict:
    """Solve expr = 0 for a symbol. Example: "x**2 - 4" -> [-2, 2].

    The symbol is auto-detected when the expression has exactly one; pass
    `symbol` explicitly when there are several (otherwise the call is refused
    rather than guessing which variable you meant).
    """
    return _sym_solve_impl(expr, symbol, timeout)


@mcp.tool()
def sym_differentiate(
    expr: str, symbol: str | None = None, order: int = 1, timeout: int = 10
) -> dict:
    """Differentiate an expression. Example: "x**3" -> "3*x**2"."""
    return _sym_differentiate_impl(expr, symbol, order, timeout)


@mcp.tool()
def sym_integrate(
    expr: str, symbol: str | None = None, bounds: list | None = None, timeout: int = 10
) -> dict:
    """Integrate an expression, indefinitely or over `bounds` as [low, high].

    Example: "x**2" -> "x**3/3"; with bounds [0, 1] -> "1/3".
    """
    return _sym_integrate_impl(expr, symbol, bounds, timeout)


@mcp.tool()
def sym_derivation(
    expr: str, operation: str = "integrate", symbol: str | None = None, timeout: int = 10
) -> dict:
    """Show the steps of a derivation.

    Genuine rule-by-rule tracing exists only for `operation="integrate"`.
    For "differentiate" SymPy has no step engine, so the response returns the
    result with a note saying so rather than pretending to a trace.
    """
    return _sym_derivation_impl(expr, operation, symbol, timeout)


@mcp.tool()
def sym_verify(lhs: str, rhs: str, timeout: int = 10) -> dict:
    """Check whether two expressions are mathematically equal.

    Returns verdict "equal" (proven), "unequal" (a numeric counterexample was
    found), or "unproven". IMPORTANT: "unproven" means the checker could not
    decide -- it is NOT a disproof, and must not be reported as "false".
    """
    return _sym_verify_impl(lhs, rhs, timeout)


@mcp.tool()
def sym_evaluate(
    expr: str, subs: dict | None = None, units: str | None = None, timeout: int = 10
) -> dict:
    """Evaluate an expression numerically, optionally substituting values or
    converting units.

    Substitution: expr "x**2 + 1" with subs {"x": 3} -> 10.
    Units: expr "5" with units "meter/second -> kilometer/hour" -> 18.
    """
    return _sym_evaluate_impl(expr, subs, units, timeout)


# ---------------------------------------------------------------------------
# knowledge graph
# ---------------------------------------------------------------------------


def _kg_neighbors_impl(node: str, limit: int = 20) -> dict:
    from attestation import kg

    with open_db() as conn:
        try:
            found = kg.neighbors(conn, node, limit=min(limit, MAX_LIST_LIMIT))
            if not found:
                return {
                    "ok": False,
                    "message": f"{node!r} is not a concept in the graph",
                    "neighbors": [],
                    "stale": kg.is_stale(conn),
                }
            return {
                "ok": True,
                "message": f"{len(found)} neighbour(s)",
                "neighbors": found,
                "stale": kg.is_stale(conn),
            }
        except Exception:
            log.exception("kg_neighbors failed for node=%s", node)
            return {"ok": False, "message": "internal error", "neighbors": [], "stale": True}


def _kg_path_impl(source: str, target: str) -> dict:
    from attestation import kg

    with open_db() as conn:
        try:
            found = kg.shortest_path(conn, source, target)
            if found is None:
                return {
                    "ok": False,
                    "message": f"no path between {source!r} and {target!r}",
                    "path": None,
                    "stale": kg.is_stale(conn),
                }
            return {
                "ok": True,
                "message": f"{len(found) - 1} hop(s)",
                "path": found,
                "stale": kg.is_stale(conn),
            }
        except Exception:
            log.exception("kg_path failed for %s -> %s", source, target)
            return {"ok": False, "message": "internal error", "path": None, "stale": True}


def _kg_central_impl(metric: str = "degree", limit: int = 10) -> dict:
    from attestation import kg

    with open_db() as conn:
        try:
            ranked = kg.central(conn, metric=metric, limit=min(limit, MAX_LIST_LIMIT))
            return {
                "ok": True,
                "message": f"top {len(ranked)} by {metric}",
                "nodes": ranked,
                "stale": kg.is_stale(conn),
            }
        except ValueError as exc:
            return {"ok": False, "message": str(exc), "nodes": [], "stale": kg.is_stale(conn)}
        except Exception:
            log.exception("kg_central failed for metric=%s", metric)
            return {"ok": False, "message": "internal error", "nodes": [], "stale": True}


def _kg_communities_impl(min_size: int = 3) -> dict:
    from attestation import kg

    with open_db() as conn:
        try:
            found = kg.communities(conn, min_size=min_size)
            return {
                "ok": True,
                "message": f"{len(found)} cluster(s)",
                "communities": found,
                "stale": kg.is_stale(conn),
            }
        except Exception:
            log.exception("kg_communities failed")
            return {"ok": False, "message": "internal error", "communities": [], "stale": True}


def _kg_rebuild_impl(confirm: bool = False) -> dict:
    from attestation import kg

    if not confirm:
        return {
            "ok": False,
            "message": (
                "refusing to rebuild without confirm=true. This replaces the "
                "kg_nodes and kg_edges tables (the graph is derived, so nothing "
                "unrecoverable is lost)."
            ),
            "nodes": 0,
            "edges": 0,
        }
    with open_db() as conn:
        try:
            counts = kg.rebuild(conn)
            return {"ok": True, "message": "graph rebuilt", **counts}
        except Exception:
            log.exception("kg_rebuild failed")
            return {"ok": False, "message": "internal error", "nodes": 0, "edges": 0}


@mcp.tool()
def kg_neighbors(node: str, limit: int = 20) -> dict:
    """Concepts directly adjacent to a given concept in the reading graph.

    The "what else should I read about this" query. Concepts come from the
    tagging pass; a tag used only once is not in the graph.

    Returns only DIRECT neighbours, ranked by co-occurrence weight (how many
    items carry both concepts), strongest first, capped at `limit` (clamped
    to 50). Each row's `weight` is that real edge weight. For questions that
    span more than one hop, use `kg_path`, which answers them exactly.
    """
    return _kg_neighbors_impl(node, limit)


@mcp.tool()
def kg_path(source: str, target: str) -> dict:
    """Shortest chain of concepts connecting two topics in the reading graph.

    Returns ok=false with path=null when the two are in different components —
    that is a legitimate answer meaning "these never co-occur", not an error.
    """
    return _kg_path_impl(source, target)


@mcp.tool()
def kg_central(metric: str = "degree", limit: int = 10) -> dict:
    """Most important concepts. metric="degree" for most-connected,
    "betweenness" for the bridges between otherwise separate clusters."""
    return _kg_central_impl(metric, limit)


@mcp.tool()
def kg_communities(min_size: int = 3) -> dict:
    """Topic clusters in the reading graph, each labelled by its most
    connected member. Useful for seeing what the reading actually splits into.

    Clusters by modularity, so a dense hub does not swallow the graph: a
    concept joins a group only when its links there beat what chance predicts.
    Densely-interconnected corpora still split into real topics -- on the live
    graph, 7 of them, from a machine-learning core down to a small
    quantum-chemistry group.

    Expect overlapping subject matter across clusters rather than clean
    partitions; concepts sit in exactly one group, so a bridging concept lands
    wherever its links are strongest.
    """
    return _kg_communities_impl(min_size)


@mcp.tool()
def kg_rebuild(confirm: bool = False) -> dict:
    """Regenerate the stored kg_nodes/kg_edges tables from current tags.
    Requires confirm=true.

    Every kg_* read tool derives its answer fresh from item_tags on each
    call, so this does not change what those tools return -- it only
    refreshes the stored tables (used for external inspection) and clears
    stale=true. Normally unnecessary since `hermes tag` rebuilds
    automatically; use this after editing the database by hand.
    """
    return _kg_rebuild_impl(confirm)


# ---------------------------------------------------------------------------
# run ledger
# ---------------------------------------------------------------------------

_NO_ROOT = (
    "no workspace configured -- set RESEARCH_ROOT to the directory holding your"
    " projects, or pass root explicitly"
)


def _runs_scan_impl(
    root: str | None = None, project: str | None = None, confirm: bool = False
) -> dict:
    from attestation import ledger

    if not confirm:
        return {
            "ok": False,
            "message": (
                "refusing to scan without confirm=true. This replaces the ledger's"
                " rows for each project scanned (they are re-read from disk, so"
                " nothing unrecoverable is lost)."
            ),
            "scanned": {},
            "empty": [],
        }
    target = ledger.workspace_root(root)
    if target is None:
        return {"ok": False, "message": _NO_ROOT, "scanned": {}, "empty": []}

    with open_db() as conn:
        try:
            out = ledger.scan(conn, target, project=project)
            total = sum(out["scanned"].values())
            return {
                "ok": True,
                "message": f"{total} run(s) across {len(out['scanned'])} project(s)",
                "scanned": out["scanned"],
                "empty": out.get("empty", []),
            }
        except Exception:
            log.exception("runs_scan failed for root=%s", target)
            return {"ok": False, "message": "internal error", "scanned": {}, "empty": []}


def _runs_list_impl(project: str | None = None, family: str | None = None, limit: int = 20) -> dict:
    from attestation import ledger

    with open_db() as conn:
        try:
            found = ledger.list_runs(
                conn, project=project, family=family, limit=min(limit, MAX_LIST_LIMIT)
            )
            if not found:
                return {
                    "ok": False,
                    "message": "no runs recorded -- call runs_scan(confirm=true) first",
                    "runs": [],
                    "families": [],
                }
            return {
                "ok": True,
                "message": f"{len(found)} run(s)",
                "runs": found,
                "families": ledger.families(conn, project=project),
            }
        except Exception:
            log.exception("runs_list failed")
            return {"ok": False, "message": "internal error", "runs": [], "families": []}


def _runs_compare_impl(family: str, metric: str | None = None) -> dict:
    from attestation import ledger

    with open_db() as conn:
        try:
            result = ledger.compare(conn, family, metric=metric)
            if not result["arms"]:
                return {
                    "ok": False,
                    "message": f"no runs in family {family!r}",
                    "family": family,
                    "metric": metric,
                    "arms": [],
                    "winner": None,
                }
            return {"ok": True, "message": f"{len(result['arms'])} arm(s)", **result}
        except ValueError as exc:
            # an undeclared metric direction is a caller-fixable problem, so the
            # reason is surfaced rather than flattened to "internal error"
            return {
                "ok": False,
                "message": str(exc),
                "family": family,
                "metric": metric,
                "arms": [],
                "winner": None,
            }
        except Exception:
            log.exception("runs_compare failed for family=%s", family)
            return {
                "ok": False,
                "message": "internal error",
                "family": family,
                "metric": metric,
                "arms": [],
                "winner": None,
            }


def _runs_detail_impl(project: str, name: str) -> dict:
    from attestation import ledger

    with open_db() as conn:
        try:
            found = ledger.detail(conn, project, name)
            if found is None:
                return {
                    "ok": False,
                    "message": f"no run {name!r} in project {project!r}",
                    "run": None,
                }
            return {"ok": True, "message": f"{len(found['metrics'])} metric row(s)", "run": found}
        except Exception:
            log.exception("runs_detail failed for %s/%s", project, name)
            return {"ok": False, "message": "internal error", "run": None}


@mcp.tool()
def runs_scan(root: str | None = None, project: str | None = None, confirm: bool = False) -> dict:
    """Read experiment runs from artifacts already on disk into the ledger.

    Walks a workspace directory, treating each subdirectory as a project, and
    reads the conventions research repos already use -- `results/`, `logs/`,
    `configs/`, `outputs/`, `benchmarks/` holding JSON, JSONL, YAML or TOML.
    Nothing needs to be instrumented and no project needs to be registered.

    `root` defaults to the RESEARCH_ROOT environment variable. Requires
    `confirm=true` since it replaces each scanned project's rows. Directories
    with nothing recognisable are listed in `empty` rather than omitted, so
    "found nothing" is never mistaken for "nothing was there".
    """
    return _runs_scan_impl(root, project, confirm)


@mcp.tool()
def runs_list(project: str | None = None, family: str | None = None, limit: int = 20) -> dict:
    """Experiment runs in the ledger, with the families they group into.

    A `family` is a set of sibling runs -- the arms of a sweep, or one run's
    checkpoints over training. Use it with `runs_compare` to answer which arm
    won. Also returns the family list, so you can see what is comparable.
    """
    return _runs_list_impl(project, family, limit)


@mcp.tool()
def runs_compare(family: str, metric: str | None = None) -> dict:
    """Rank the arms of an experiment family by a metric.

    The question a sweep exists to answer and that usually lives only in
    filenames: which variant won, on what metric, by how much. Omit `metric` to
    use the one most arms share.

    Refuses to rank a metric whose direction is undeclared rather than guessing
    -- ranking WER as if higher were better would name the worst arm the
    winner. Arms with no value for the metric are listed in `without_metric`
    rather than dropped: an arm that was never evaluated is a finding.
    """
    return _runs_compare_impl(family, metric)


@mcp.tool()
def runs_detail(project: str, name: str) -> dict:
    """One run in full: config shape, every metric, source path, and the
    header comment from its config if it had one.

    That header is often where the hypothesis and the single changed variable
    are written down. It is stored verbatim and never interpreted.
    """
    return _runs_detail_impl(project, name)


# ---------------------------------------------------------------------------
# claim checking
# ---------------------------------------------------------------------------


def _claims_check_impl(path: str | None = None, verdict: str | None = None) -> dict:
    from pathlib import Path

    from attestation import claims, ledger

    target = Path(path).expanduser() if path else ledger.workspace_root()
    if target is None:
        return {"ok": False, "message": _NO_ROOT, "claims": [], "counts": {}, "malformed": []}
    if not target.exists():
        return {
            "ok": False,
            "message": f"no such path: {target}",
            "claims": [],
            "counts": {},
            "malformed": [],
        }

    with open_db() as conn:
        try:
            out = claims.check(conn, target)
            rows = [
                {
                    "verdict": v.verdict,
                    "file": v.claim.path,
                    "line": v.claim.line,
                    "run": f"{v.claim.project}/{v.claim.run}",
                    "metric": v.claim.metric,
                    "claimed": v.claim.value,
                    "actual": v.actual,
                    "message": v.message,
                    "source_path": v.source_path,
                }
                for v in out["verdicts"]
                if verdict is None or v.verdict == verdict
            ]
            summary = ", ".join(f"{n} {k}" for k, n in sorted(out["counts"].items()))
            return {
                "ok": True,
                "message": (
                    f"{out['claims']} claim(s): {summary}" if out["claims"] else "no claims found"
                ),
                "claims": rows,
                "counts": out["counts"],
                "malformed": out["malformed"],
            }
        except Exception:
            log.exception("claims_check failed for path=%s", target)
            return {
                "ok": False,
                "message": "internal error",
                "claims": [],
                "counts": {},
                "malformed": [],
            }


def _claims_coverage_impl(path: str | None = None) -> dict:
    from pathlib import Path

    from attestation import claims, ledger

    target = Path(path).expanduser() if path else ledger.workspace_root()
    if target is None:
        return {"ok": False, "message": _NO_ROOT, "uncovered": [], "numbers": 0}
    if not target.exists():
        return {"ok": False, "message": f"no such path: {target}", "uncovered": [], "numbers": 0}
    try:
        out = claims.coverage(target)
        return {
            "ok": True,
            "message": f"{out['covered']}/{out['numbers']} number(s) covered by a claim",
            "uncovered": out["uncovered"],
            "numbers": out["numbers"],
            "covered": out["covered"],
        }
    except Exception:
        log.exception("claims_coverage failed for path=%s", target)
        return {"ok": False, "message": "internal error", "uncovered": [], "numbers": 0}


@mcp.tool()
def claims_coverage(path: str | None = None) -> dict:
    """Numbers asserted in Markdown that no claim annotation covers.

    The inverse of `claims_check`: that verifies the claims that exist, this
    finds assertions nobody made checkable. A document with zero contradicted
    claims looks healthy while asserting a dozen unverifiable numbers, and
    nothing else surfaces the difference.

    Only decimals count as measurements -- integers in prose are overwhelmingly
    versions, counts and dates. Versions, ISO dates, URLs, package pins and
    anything inside an HTML comment are excluded, since a comment renders as
    nothing and asserts nothing to a reader.
    """
    return _claims_coverage_impl(path)


@mcp.tool()
def claims_check(path: str | None = None, verdict: str | None = None) -> dict:
    """Verify numeric claims written in Markdown against runs in the ledger.

    A claim is an HTML comment beside the prose it describes, so it renders as
    nothing and the document reads exactly as before:

        <!-- claim: project/run metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->

    Five verdicts, and the differences matter. `supported`: a run agrees.
    `contradicted`: a run disagrees — the document or the run is wrong.
    `unsupported`: no run matches, so the claim may still be true but nothing
    backs it. `ambiguous`: a wildcard matched several runs, so which is meant is
    undecidable. `stale`: the value matches but the artifact changed after
    `as_of`, so it is worth re-verifying.

    Filter with `verdict` to answer "what in my documentation is unsupported".
    Read-only: it reports, it never edits a document.
    """
    return _claims_check_impl(path, verdict)


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------


def _ranking_quality(conn, user_id: int) -> dict:
    """How much to trust the ordering, stated up front.

    A digest built from an untrained ranker looks exactly like one built from a
    good one. rank.classifier_probs returns None when a user's clicks are all
    one class (rank.py's single-class guard), so the click-CLASSIFIER term
    never fires -- but rank_items blends in a second, independent term
    (avg_ranks over pref_scores_for_items) whenever n_clicks > 0, regardless of
    the guard. So a single-class history with at least one click is NOT pure
    embedding similarity: the feature-preference term still contributes, only
    the classifier is silent. Naming which terms are actually contributing
    matters more than a blanket "profile-embedding only" claim, which is wrong
    in exactly the case this caveat exists to describe.
    """
    rows = conn.execute(
        "SELECT useful, COUNT(*) n FROM clicks WHERE user_id = ? GROUP BY useful",
        (user_id,),
    ).fetchall()
    counts = {int(r["useful"]): r["n"] for r in rows}
    total = sum(counts.values())
    active = len(counts) > 1
    out = {
        "clicks": total,
        "useful": counts.get(1, 0),
        "not_useful": counts.get(0, 0),
        "classifier_active": active,
    }
    if not active:
        if total > 0:
            out["caveat"] = (
                f"ranking is running WITHOUT its click classifier: {total} click(s), "
                f"all {'useful' if counts.get(1) else 'not-useful'}. Order blends "
                "profile-embedding similarity with a feature-preference term learned "
                "from those clicks -- the classifier term is silent (needs both "
                "useful and not-useful clicks to fire), but the preference term is "
                "still contributing. Mark some items the other way to train the "
                "classifier too."
            )
        else:
            out["caveat"] = (
                "ranking is running WITHOUT its click classifier or any "
                "feature-preference signal: 0 clicks recorded. Order is "
                "profile-embedding similarity only."
            )
    elif total < 20:
        out["caveat"] = f"only {total} clicks: the classifier is active but weakly trained"
    return out


def _digest_impl(user: str, days: int = 7, per_topic: int = 3, limit: int = 30) -> dict:
    from attestation import kg

    empty = {"topics": [], "unclustered": [], "ranking_quality": {}, "window_days": days}
    with open_db() as conn:
        try:
            row = get_user(conn, user)
            if row is None:
                return {"ok": False, "message": f"unknown user {user!r}", **empty}

            items_ranked = _ranked_items(conn, row, min(limit, MAX_LIST_LIMIT), days)
            items = [
                {
                    "item_id": it.item_id,
                    "title": it.title,
                    "url": it.url,
                    "source": it.source,
                    "score": it.score,
                    "tags": it.tags,
                    "content_type": it.content_type,
                }
                for it in items_ranked
            ]
            if not items:
                return {"ok": False, "message": "no unread items to digest", **empty}

            communities = kg.communities(conn, min_size=3)
            members = [(c["label"], set(c["members"])) for c in communities]
            cached = {
                r["item_id"]: r["text"]
                for r in conn.execute(
                    "SELECT item_id, text FROM explanations WHERE user_id = ?", (row["id"],)
                )
            }

            grouped: dict[str, list] = {}
            unclustered: list = []
            for item in items:
                tags = set(item.get("tags") or [])
                # strongest tag overlap; ties break on label so repeated calls agree
                best_label, best_n = None, 0
                for label, concepts in members:
                    overlap = len(tags & concepts)
                    if overlap > best_n or (
                        overlap == best_n and overlap and label < (best_label or "~")
                    ):
                        best_label, best_n = label, overlap
                enriched = dict(item)
                if item["item_id"] in cached:
                    enriched["explanation"] = cached[item["item_id"]]
                if best_n and best_label is not None:
                    grouped.setdefault(best_label, []).append(enriched)
                else:
                    unclustered.append(enriched)

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
                "ok": True,
                "message": f"{len(items)} item(s) in {len(topics)} topic(s)",
                "topics": topics,
                "unclustered": unclustered,
                "ranking_quality": _ranking_quality(conn, row["id"]),
                "window_days": days,
            }
        except Exception:
            log.exception("digest failed for user=%s", user)
            return {"ok": False, "message": "internal error", **empty}


@mcp.tool()
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
    return _digest_impl(user, days, per_topic, limit)


def main() -> None:
    from attestation.llm import load_env

    load_env()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
