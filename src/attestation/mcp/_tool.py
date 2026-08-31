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
It is NOT a general error-swallowing mechanism. The inline
`# noqa: BLE001` sites in the codebase stay exactly where they are, because
each implements a specific policy that needs local knowledge -- `rank.py`
serves a stale cached profile vector when the embedder is down and raises only
when the cache is cold, a decision this layer cannot make because it does not
know a cache exists.
"""

import contextlib
import functools
import inspect
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
    """Every persona name in the database, alphabetical."""
    return [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]


def unknown_user_message(conn: sqlite3.Connection, user: str) -> str:
    """The refusal text for a name that does not autocreate -- names the
    personas that do exist, so the caller has something to act on."""
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
    empty: dict | Callable[[dict], dict] | None = None,
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

    `empty` may also be a callable `(kwargs: dict) -> dict`, resolved once per
    call from the CALLER's own arguments -- `conn` excluded (never passed by
    a caller), bound by the caller-facing parameter names so a positional or
    keyword call sees the same dict -- for a tool whose shape genuinely
    branches on an argument, such as `feed.persona_status(user=None)`
    returning `{"users": [...]}` when listing versus the seven per-user
    training keys when given one. A plain dict is still the right choice
    whenever one shape covers every call, which is every other tool on this
    surface; reach for the callable form only when a fixed `empty` would
    force one branch's keys onto the other's envelope.

    For a `needs_user=True` tool, the callable sees `user` (the caller's
    persona name) even though the tool BODY never takes `user` as a
    parameter -- the body receives the resolved `user_row` instead, and that
    contract is unchanged. The dict the callable sees and the arguments the
    body receives are deliberately different: the callable branches on what
    the caller asked for, before the user lookup; the body works with what
    lookup resolved it to.
    """

    def decorate(fn: Callable) -> Callable:
        """Bind `fn` inside the ritual described in `tool`'s own docstring."""
        name = label or getattr(fn, "__name__", "tool")
        # The CALLER's own parameter names -- what a callable `empty` sees --
        # are not simply `fn`'s signature minus its injected leaders: when
        # `needs_user`, the caller passes `user` (a persona name) but `fn`
        # never receives that name as a parameter, since the wrapper resolves
        # it to `user_row` before calling the body. So `user` is spliced back
        # in here as the caller-facing leader `needs_user` implies, and only
        # `conn` (never passed by the caller at all) is stripped from `fn`'s
        # own signature. This list is used ONLY to build the dict a callable
        # `empty` is invoked with; the body still receives exactly what it
        # always has (conn, user_row, then its own remaining arguments).
        body_params = list(inspect.signature(fn).parameters)
        if needs_db:
            body_params = body_params[1:]
        if needs_user:
            body_params = body_params[1:]
        own_params = (["user"] if needs_user else []) + body_params

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """The actual per-call ritual: connection, user lookup, the try
            block whose except clauses are documented individually below."""
            resolved_empty: dict = {}

            def fail(message: str) -> dict:
                """The failure envelope: `empty`'s fields plus why."""
                return {"ok": False, "message": message, **resolved_empty}

            def succeed(result: dict | None) -> dict:
                """The success envelope: the body's own fields layered over
                `empty`'s defaults, so both envelopes share every key."""
                result = dict(result or {})
                message = result.pop("message", "")
                return {"ok": True, "message": message, **{**resolved_empty, **result}}

            try:
                # Resolved INSIDE the try: a callable `empty` is caller-
                # supplied code the decorator now invokes on the body's
                # behalf, and it is the one piece of code in this wrapper
                # that was not protected by the try/except below. A raising
                # callable now takes the same generic-failure path as a
                # raising body -- logged, `resolved_empty` left at `{}` so
                # `fail()` still returns a well-formed envelope -- rather
                # than escaping as a bare traceback.
                if empty is None:
                    resolved_empty = {}
                elif isinstance(empty, dict):
                    resolved_empty = empty
                else:
                    bound = dict(zip(own_params, args, strict=False))
                    bound.update(kwargs)
                    resolved_empty = empty(bound)
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
                    # NOT a bare `raise`: the generic handler below is a
                    # SIBLING of this clause, not an outer one, so raising here
                    # escapes the wrapper and an MCP tool returns a traceback --
                    # the one thing this envelope exists to prevent. Found via
                    # kg.neighbors, whose `min(limit, MAX)` against a None limit
                    # says "'<' not supported", with no "argument" in it. Its
                    # siblings survived only because int()'s message happens to
                    # contain the word.
                    log.exception("%s failed args=%s kwargs=%s", name, args, kwargs)
                    return fail(f"internal error in {name}; see server logs")
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
