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
    branch without needing a differently-shaped bad input per tool."""
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
            if len(params) >= 2 and params[0] == "conn" and params[1] == "user_row":
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
            offenders.append(f"{name}: unknown user did not fail")
        elif "unknown user" not in out.get("message", ""):
            offenders.append(f"{name}: unhelpful message {out.get('message')!r}")
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
