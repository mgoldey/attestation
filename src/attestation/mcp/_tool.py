"""The ritual every MCP tool used to repeat, owned in one place.

Before this module, each of the 37 tools opened its own connection, wrapped
its body in `try`, resolved the user by hand, built a success envelope by hand,
and built a failure envelope by hand -- 26 copies of `with open_db()`, 28 broad
`except` blocks, and 10 unknown-user checks.

Hand-maintaining that shape 34 times is how it drifted. `_profile_status_impl`
listed eight keys in its error branch to match its success branch;
`_runs_compare_impl` had three exits each re-listing four keys;
`_explain_item_impl` wrote `"explanation": None` in three places. A caller could
not rely on "the failure shape matches the success shape", because that property
was maintained by hand in every tool separately.

`@tool` makes it structural: the decorated function returns only the fields it
computes, and the envelope -- `ok`, `message`, and the declared empty fields on
failure -- is added here.

**Scope limit.** This owns the *presentation* envelope only: turning an
unexpected exception into `ok: False` so an MCP tool never returns a traceback.
It is NOT a general error-swallowing mechanism. The four inline
`# noqa: BLE001` sites in the codebase stay exactly where they are, because
each implements a specific policy that needs local knowledge -- `rank.py:198`
serves a stale cached profile vector when the embedder is down and raises only
when the cache is cold, a decision this layer cannot make because it does not
know a cache exists.
"""

import contextlib
import functools
import logging
import sqlite3
from collections.abc import Callable

from attestation.db import get_db, resolve_db_path

log = logging.getLogger(__name__)


@contextlib.contextmanager
def open_db(path=None):
    """One connection per tool call: open, yield, always close.

    sqlite3.Connection is not itself a closing context manager -- its
    __enter__/__exit__ manage transactions, not the handle -- so this wraps
    get_db(resolve_db_path(path)) in a real close-on-exit contract.
    """
    conn = get_db(resolve_db_path(path))
    try:
        yield conn
    finally:
        conn.close()


def valid_users(conn: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]


def unknown_user_message(conn: sqlite3.Connection, user: str) -> str:
    names = valid_users(conn)
    return f"unknown user: {user!r}. Valid users: {', '.join(names) if names else '(none seeded)'}"


class ToolError(Exception):
    """A failure the caller should see spelled out, not logged as a bug.

    Raise this for the expected refusals -- unknown item, undeclared metric,
    a confirm gate that was not passed. The message reaches the caller verbatim.
    An uncaught exception of any other type is a bug: it is logged with a
    traceback and the caller gets a generic message instead, because internal
    details are not the caller's to act on.
    """


# What a persona starts with before its reader says anything. Not empty: an
# empty interests string embeds to nothing and ranks on nothing, so a new
# reader would get an arbitrary order and no reason to trust it. The corpus's
# own most-common topics are the honest default -- "here is what this feed is
# about" -- and the first thing the agent does is ask what to change it to.


def tool(
    *,
    empty: dict | None = None,
    needs_user: bool = False,
    needs_db: bool = True,
    label: str | None = None,
    autocreate_user: bool = False,
) -> Callable:
    """Wrap a tool body in the connection, the user lookup, and the envelope.

    The decorated function receives `conn` first when `needs_db`, then
    `user_row` when `needs_user`, then the tool's own arguments. It returns a
    dict of just the fields it computed, or raises ToolError to refuse.

    `empty` names the non-envelope keys and their failure values, so a failed
    call has the same shape as a successful one. That is the property callers
    could not rely on before, and it is now enforced structurally rather than
    by 34 hand-written error branches.
    """
    empty = empty or {}

    def decorate(fn: Callable) -> Callable:
        name = label or getattr(fn, "__name__", "tool")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            def fail(message: str) -> dict:
                return {"ok": False, "message": message, **empty}

            def succeed(result: dict | None) -> dict:
                result = dict(result or {})
                message = result.pop("message", "")
                return {"ok": True, "message": message, **{**empty, **result}}

            try:
                if not needs_db:
                    return succeed(fn(*args, **kwargs))
                with open_db() as conn:
                    if not needs_user:
                        return succeed(fn(conn, *args, **kwargs))
                    user = args[0] if args else kwargs.pop("user", None)
                    rest = args[1:] if args else ()
                    row = _get_user(conn, user)
                    created_with = None
                    if row is None:
                        if not (autocreate_user and user):
                            return fail(unknown_user_message(conn, user))
                        row, created_with = _autocreate_user(conn, user)
                    result = succeed(fn(conn, row, *rest, **kwargs))
                    if created_with is None:
                        return result
                    # Announce it. Creating a reader profile is a real side
                    # effect, and doing it silently turns a typo into a
                    # permanent persona nobody knows exists. Ask what they
                    # want to follow rather than confirming a name: the name
                    # is whatever was passed, and the interests text IS the
                    # profile embedding.
                    result["message"] = (
                        f"created a new reader '{user}', starting from what this"
                        f" feed mostly covers ({created_with}). What topics do you"
                        " want to monitor? I will retune it to those. " + result.get("message", "")
                    ).strip()
                    return result
            except ToolError as exc:
                return fail(str(exc))
            except sqlite3.OperationalError as exc:
                # Contention is transient and the caller can act on it; a bug
                # is neither. Both arrived as "internal error; see server
                # logs", so a click lost to an hourly cron ingest holding the
                # write lock was indistinguishable from a crash -- and the web
                # UI's long-lived connection plus per-call MCP connections make
                # that a live configuration, not a hypothetical.
                #
                # Only "locked"/"busy" is excused. `no such column` is a bug
                # and must not be dressed up as something to retry.
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    return fail(
                        f"{name}: the database is busy (another write is in progress)."
                        " Retry in a moment."
                    )
                # Anything else is a bug and takes the bug path -- logged with
                # a traceback, reported generically. A bare `raise` would
                # escape the whole wrapper, since the generic handler below is
                # a sibling of this clause, not an outer one.
                log.exception("%s failed args=%s kwargs=%s", name, args, kwargs)
                return fail(f"internal error in {name}; see server logs")
            except TypeError as exc:
                # A missing or surplus argument is the CALLER's mistake and
                # the caller can fix it -- but only if told which one. This
                # reached the generic handler and came back as "internal
                # error; see server logs", which reads as a server fault and
                # gives an agent nothing to act on. Re-raised if it did not
                # come from binding, since a TypeError inside a body is a bug.
                message = str(exc)
                if "argument" not in message:
                    raise
                return fail(f"{name}: {message.split('()', 1)[-1].strip()}")
            except Exception:
                log.exception("%s failed args=%s kwargs=%s", name, args, kwargs)
                return fail(f"internal error in {name}; see server logs")

        return wrapper

    return decorate


def _autocreate_user(conn: sqlite3.Connection, name: str):
    """Moved to rank.py so server.py can share it without importing mcp/."""
    from attestation.rank import autocreate_user

    return autocreate_user(conn, name)


def _get_user(conn: sqlite3.Connection, name: str):
    from attestation.rank import get_user

    return get_user(conn, name)
