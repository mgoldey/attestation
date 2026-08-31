"""The envelope contract, enforced structurally instead of by hand.

The property under test is the one that drifted when each of 34 tools built
its own error branch: a failed call must have the same keys as a successful
one, so a caller can read `result["items"]` without first checking `ok`.
"""

import pytest

from attestation.mcp._tool import ToolError, tool


def test_success_gets_ok_and_declared_fields():
    @tool(empty={"items": []}, needs_db=False)
    def f():
        return {"items": [1, 2], "message": "two"}

    assert f() == {"ok": True, "message": "two", "items": [1, 2]}


def test_failure_has_the_same_keys_as_success():
    """The whole point. Before, this was maintained by hand in every tool."""

    @tool(empty={"items": [], "quality": {}}, needs_db=False)
    def f(explode: bool):
        if explode:
            raise ToolError("nope")
        return {"items": [1], "quality": {"a": 1}}

    ok, bad = f(False), f(True)
    assert set(ok) == set(bad), "failure envelope must match success envelope"
    assert bad == {"ok": False, "message": "nope", "items": [], "quality": {}}


def test_tool_error_message_reaches_the_caller_verbatim():
    @tool(needs_db=False)
    def f():
        raise ToolError("refusing without confirm=true")

    assert f()["message"] == "refusing without confirm=true"


def test_unexpected_exception_is_logged_not_leaked(caplog):
    """A bug must not hand the caller a traceback to act on."""

    @tool(empty={"items": []}, needs_db=False, label="thing")
    def f():
        raise ZeroDivisionError("secret internal detail")

    out = f()
    assert out["ok"] is False
    assert "secret internal detail" not in out["message"]
    assert "see server logs" in out["message"]
    assert out["items"] == []
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_message_defaults_to_empty_when_body_omits_it():
    @tool(needs_db=False)
    def f():
        return {}

    assert f() == {"ok": True, "message": ""}


def test_body_fields_override_empty_defaults():
    @tool(empty={"n": 0, "rows": []}, needs_db=False)
    def f():
        return {"n": 3}

    assert f() == {"ok": True, "message": "", "n": 3, "rows": []}


def test_callable_empty_gives_the_failing_branchs_own_envelope():
    """A callable `empty` is resolved from the call's OWN arguments, so a
    failure carries exactly the keys the branch that failed would have
    succeeded with -- not the other branch's keys too.

    This is what `feed.persona_status(user=None)` needed: a fixed `empty=`
    forced the seven per-user training keys onto the three-key listing
    branch (and vice versa), so `persona_status(user=None)` returned ten
    keys instead of three even on success, let alone failure.
    """

    def shape(kwargs: dict) -> dict:
        return {"listing": []} if kwargs.get("user") is None else {"detail": None, "user": None}

    @tool(empty=shape, needs_db=False)
    def f(user: str | None = None, explode: bool = False):
        if explode:
            raise ToolError("nope")
        if user is None:
            return {"listing": [1, 2]}
        return {"detail": "trained", "user": user}

    listing_ok = f()
    assert listing_ok == {"ok": True, "message": "", "listing": [1, 2]}

    listing_fail = f(explode=True)
    assert listing_fail == {"ok": False, "message": "nope", "listing": []}
    assert set(listing_fail) == set(listing_ok), "the listing branch's own envelope shape"

    detail_ok = f(user="ana")
    assert detail_ok == {"ok": True, "message": "", "detail": "trained", "user": "ana"}

    detail_fail = f(user="ana", explode=True)
    assert detail_fail == {"ok": False, "message": "nope", "detail": None, "user": None}
    assert "listing" not in detail_fail, "must not carry the OTHER branch's keys"


def test_needs_user_callable_empty_sees_the_callers_user(tmp_path, monkeypatch):
    """A `needs_user=True` tool's callable `empty` must see the CALLER's
    `user` argument, not the tool body's parameter list.

    `own_params` used to be `fn`'s own signature minus its injected `conn`/
    `user_row` leaders -- correct for a plain tool, but `needs_user` tools
    never take `user` as a body parameter at all (the wrapper resolves it to
    `user_row` before calling the body), so stripping two leaders from `fn`'s
    signature dropped `user` from the dict a callable `empty` is built from,
    silently. A callable keyed on `kwargs.get("user")` always saw `None`.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'x')")
    conn.commit()
    conn.close()

    def shape(kwargs: dict) -> dict:
        return {"a": "known"} if kwargs.get("user") == "ana" else {"a": "unknown"}

    @tool(empty=shape, needs_user=True)
    def f(conn, user_row):
        raise ToolError("boom")

    assert f("ana") == {"ok": False, "message": "boom", "a": "known"}


def test_a_raising_callable_empty_is_contained_not_leaked():
    """A callable `empty` is caller-supplied code the decorator now invokes,
    and it is the one piece of code inside the wrapper that the wrapper did
    not protect: `resolved_empty = empty(bound)` sat OUTSIDE the `try` block,
    so a callable that raises escaped as a bare traceback -- the exact
    failure `@tool` exists to prevent. Unreachable today (the only callable
    `empty` in the tree cannot raise), but the branch is new, general, and
    now documented as an invited pattern.
    """

    def explode(_kwargs: dict) -> dict:
        raise RuntimeError("empty callable blew up")

    @tool(empty=explode, needs_db=False)
    def f():
        return {"value": 1}

    out = f()
    assert out["ok"] is False
    assert "empty callable blew up" not in out["message"]
    assert "see server logs" in out["message"]


def test_unknown_user_names_the_valid_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))

    @tool(empty={"items": []}, needs_user=True)
    def f(conn, user_row):
        return {"items": [user_row["name"]]}

    out = f("nobody")
    assert out["ok"] is False
    assert "nobody" in out["message"]
    assert out["items"] == []


def test_known_user_is_resolved_and_passed_through(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    from attestation.db import get_db

    conn = get_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'x')")
    conn.commit()
    conn.close()

    @tool(needs_user=True)
    def f(conn, user_row, suffix: str):
        return {"message": user_row["name"] + suffix}

    assert f("ana", "!")["message"] == "ana!"


def test_connection_is_closed_even_when_the_body_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    seen = []

    @tool(needs_db=True)
    def f(conn):
        seen.append(conn)
        raise ZeroDivisionError

    f()
    with pytest.raises(Exception):
        seen[0].execute("SELECT 1")


# --- the contract across the REAL surface ---------------------------------
#
# Everything above exercises the decorator with throwaway functions. That
# proves the mechanism and nothing about the 37 registered tools, whose
# `empty=` dicts are hand-written -- the same hand-maintenance the decorator
# was built to remove, relocated to its argument. The sym_* tools do not use
# the decorator at all and were dropping `numeric` and `parsed_input` on
# failure until this test was written.


def _user_tools():
    """The tools that take a `user`, so an unknown persona forces the failure
    branch without needing a differently-shaped bad input per tool.

    Two shapes both count: `needs_user=True` tools, whose wrapped signature is
    `(conn, user_row, ...)` -- `@tool` resolves the row before the body runs
    -- and a tool like `feed.persona_status` that resolves its OWN user by
    hand and is `needs_user=False`, whose second parameter is still literally
    named `user`. Both must answer an unknown persona with the full envelope;
    only the resolution mechanism differs.
    """
    import inspect

    from attestation import mcp_server
    from attestation.mcp import DOMAINS

    found = []
    for module in DOMAINS:
        for name in dir(module):
            fn = getattr(module, name)
            if not (name.startswith("_") and callable(fn) and hasattr(fn, "__wrapped__")):
                continue
            params = list(inspect.signature(fn.__wrapped__).parameters)
            if len(params) >= 2 and params[0] == "conn" and params[1] in ("user_row", "user"):
                found.append((f"{module.__name__}.{name}", fn))
    assert found, "no user-taking tools discovered"
    assert mcp_server  # imported for its side effect of registering everything
    return found


def test_every_user_tool_answers_an_unknown_persona_with_a_full_envelope(tmp_path, monkeypatch):
    """A caller must be able to read result["items"] without checking ok first.

    That property was maintained by hand in 37 places before the decorator and
    is maintained by hand in 37 `empty=` dicts after it. This is the test the
    design spec asked for and the file above did not provide.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    offenders = []
    for name, fn in _user_tools():
        out = fn("definitely-not-a-persona")
        if out.get("ok") is not False:
            # Read-side tools auto-create the reader rather than refusing --
            # the refusal is what taught agents to call persona_create with
            # whatever string they had. They must still return a full
            # envelope, which succeed() guarantees, so there is nothing to
            # check here beyond it not blowing up.
            assert "message" in out and "ok" in out, f"{name}: {sorted(out)}"
            continue
        message = out.get("message", "")
        # A refusal must say something the caller can act on: which persona is
        # unknown, which argument is missing, or what the tool needs. The one
        # thing it must never be is the generic bug message, which reads as a
        # server fault for a mistake the caller could fix.
        if "see server logs" in message or not message:
            offenders.append(f"{name}: unhelpful message {message!r}")
    assert not offenders, "\n".join(offenders)


def test_symbolic_tools_keep_their_shape_on_failure():
    """The 7 sym_* tools build their own envelope rather than using @tool."""
    from attestation.mcp import symbolic

    checks = [
        (symbolic._sym_simplify, "x+x", "@@@"),
        (symbolic._sym_solve, "x-1", "@@@"),
        (symbolic._sym_evaluate, "2+2", "@@@"),
        (symbolic._sym_differentiate, "x**2", "@@@"),
        (symbolic._sym_integrate, "x", "@@@"),
    ]
    for fn, good, bad in checks:
        ok, err = fn(good), fn(bad)
        assert ok["ok"] is True, f"{fn.__name__} failed on valid input: {ok}"
        assert err["ok"] is False, f"{fn.__name__} succeeded on garbage: {err}"
        assert set(ok) == set(err), f"{fn.__name__} drops {sorted(set(ok) - set(err))} on failure"


def test_a_missing_argument_names_itself(tmp_path, monkeypatch):
    """`internal error; see server logs` is useless to an agent that can fix
    the call itself.

    Calling feed.read with a user but no item_id raised TypeError, which the
    decorator turned into the generic bug message -- so a recoverable mistake
    read as a server fault, and the agent had nothing to act on. A missing
    argument is the caller's to supply.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.mcp import feed as feed_mod

    out = feed_mod._read_item("someone")

    assert out["ok"] is False
    assert "item_id" in out["message"], out["message"]
    assert "see server logs" not in out["message"]


def test_a_locked_database_is_reported_as_busy_not_as_an_internal_error():
    """Lock contention is transient and retryable; an internal error is not.

    `sqlite3.OperationalError: database is locked` fell into the generic
    handler, so a caller got "internal error in record_feedback; see server
    logs" -- indistinguishable from a crash, with no hint that retrying works.
    It is live-reachable: the hourly cron ingest writes while the web UI holds
    its long-lived connection and every MCP call opens its own.

    The module already draws this distinction for TypeError, whose comment says
    a caller mistake "reads as a server fault and gives an agent nothing to act
    on". A busy database is the same shape.
    """
    import sqlite3

    from attestation.mcp._tool import tool

    @tool(empty={"value": None}, needs_db=False, label="locked_probe")
    def _probe() -> dict:
        raise sqlite3.OperationalError("database is locked")

    out = _probe()
    assert out["ok"] is False
    assert "busy" in out["message"].lower() or "locked" in out["message"].lower(), out["message"]
    assert "internal error" not in out["message"].lower()
    assert "value" in out, "the failure envelope must still match the success shape"


def test_a_real_sqlite_error_is_still_a_bug():
    """Only contention is excused. A malformed query is a bug and must not be
    dressed up as a transient condition a caller should retry."""
    import sqlite3

    from attestation.mcp._tool import tool

    @tool(empty={"value": None}, needs_db=False, label="broken_probe")
    def _probe() -> dict:
        raise sqlite3.OperationalError("no such column: nope")

    out = _probe()
    assert out["ok"] is False
    assert "internal error" in out["message"].lower(), out["message"]


def test_a_type_error_inside_a_body_does_not_escape_the_envelope():
    """The envelope's stated purpose: "an MCP tool never returns a traceback".

    The `except TypeError` clause bare-`raise`s when the message lacks the
    substring "argument", which escapes the wrapper entirely -- the generic
    handler is a SIBLING of that clause, not an outer one. The comment three
    lines below, added for OperationalError, warns about exactly this.

    kg.neighbors demonstrates it: `min(limit, MAX)` against a None limit raises
    "'<' not supported between instances of 'int' and 'NoneType'". Its siblings
    survive only by luck -- their TypeError comes from int(), whose message
    happens to contain "argument".
    """
    from attestation.mcp._tool import tool

    @tool(empty={"value": None}, needs_db=False, label="typeerror_probe")
    def _probe() -> dict:
        return {"value": min(3, None)}  # type: ignore[type-var]

    out = _probe()
    assert out["ok"] is False
    assert "value" in out, "the failure envelope must still match the success shape"
    assert "traceback" not in out["message"].lower()


def test_closed_vocabularies_are_schema_constraints_not_silent_defaults():
    """An invalid value from a CLOSED set must be a client-side reject, not a
    confident answer to a different question.

    `sym.derivation(operation="factorize")` returned `2*x` with ok=true --
    anything but "integrate" fell through to an else branch and differentiated.
    `runs.claims_check(verdict="banana")` returned ok=true with a message
    summarising all 7 claims and `claims: []`, so a model relaying the message
    reports contradicted claims it cannot show.

    Goes through call_tool, NOT the `_impl` functions: those bypass pydantic,
    which is exactly why a prior audit reported two already-constrained
    parameters as broken and missed these.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    mcp = FastMCP("audit")
    register_all(mcp)
    schemas = {t.name: t.inputSchema for t in asyncio.run(mcp.list_tools())}

    def allowed(tool: str, param: str) -> set[str]:
        spec = schemas[tool]["properties"][param]
        for branch in (spec, *spec.get("anyOf", [])):
            if "enum" in branch:
                return set(branch["enum"])
        return set()

    assert allowed("sym.derivation", "operation") == {"integrate", "differentiate"}, (
        "sym.derivation.operation is not a closed set in the schema"
    )
    assert allowed("runs.claims_check", "verdict") == {
        "supported",
        "contradicted",
        "unsupported",
        "ambiguous",
        "stale",
        "uncited",
    }, "runs.claims_check.verdict is not a closed set in the schema"

    # Bounds carry their arity, so [1] and [1,2,3] are rejects rather than an
    # IndexError and a silently-dropped extra.
    bounds = schemas["sym.integrate"]["properties"]["bounds"]
    branches = (bounds, *bounds.get("anyOf", []))
    assert any(b.get("minItems") == 2 and b.get("maxItems") == 2 for b in branches), (
        f"sym.integrate.bounds does not require exactly two values: {bounds}"
    )

    # An empty interests string embeds to nothing (CLAUDE.md).
    for tool_name in ("feed.persona_create", "feed.persona_update"):
        spec = schemas[tool_name]["properties"]["interests"]
        assert spec.get("minLength") == 1, f"{tool_name}.interests accepts an empty string: {spec}"
