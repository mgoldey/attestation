import pathlib
import time

import pytest

from attestation import symbolic


@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').getcwd()",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "lambda: 1",
    ],
)
def test_parse_safe_blocks_code_execution(attack):
    """parse_expr WITHOUT a restricted namespace executes these -- verified.

    sympify does too. The whole module exists to make that impossible, so
    these must raise rather than evaluate.
    """
    with pytest.raises(ValueError):
        symbolic.parse_safe(attack)


def test_module_never_calls_sympify_or_eval():
    """A source-level guard: the safe parser is pointless if someone later
    reaches for sympify because it is more convenient.

    Checks the parsed AST for CALLS rather than the raw text, so the module's
    own docstring can explain why sympify is banned without tripping its own
    guard.
    """
    import ast as _ast

    tree = _ast.parse(pathlib.Path(symbolic.__file__).read_text())
    called = {
        node.func.id
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    } | {
        node.func.attr
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute)
    }
    assert "sympify" not in called
    assert "eval" not in called
    assert "exec" not in called


def test_parse_safe_blocks_the_subclasses_gadget_specifically():
    """Regression guard for the hole namespace-restriction alone leaves open.

    With only local_dict/global_dict pinned, this returns 441 live classes with
    subprocess.Popen reachable -- the first move of a sandbox escape. It needs
    no name lookup at all, which is why the AST screen exists.
    """
    with pytest.raises(ValueError, match="attribute access"):
        symbolic.parse_safe("().__class__.__bases__[0].__subclasses__()")


def test_parse_safe_handles_real_math():
    expr = symbolic.parse_safe("x**2 + sin(x)")
    assert str(expr) == "x**2 + sin(x)"


@pytest.mark.parametrize(
    "text,expected",
    [("2x", "2*x"), ("2 x", "2*x"), ("3sin(x)", "3*sin(x)")],
)
def test_parse_safe_keeps_implicit_multiplication(text, expected):
    """The AST screen normalizes these before checking, so SymPy's implicit
    multiplication must still work after it."""
    assert str(symbolic.parse_safe(text)) == expected


@pytest.mark.parametrize(
    "text",
    ["Matrix([[1,2],[3,4]])", "Rational(1,3)+1"],
)
def test_parse_safe_keeps_whitelisted_calls(text):
    """Regression guard: the literal-magnitude and normalization fixes must
    not disturb ordinary whitelisted-function calls."""
    symbolic.parse_safe(text)  # must not raise


@pytest.mark.parametrize("text", ["1e30000", "1e300000"])
def test_parse_safe_rejects_pathological_literals_quickly(text):
    """parse_expr builds arbitrary-precision numbers straight from literal
    text, outside run_isolated's subprocess boundary entirely -- verified,
    parse_safe('1e300000') cost 7.1s and a 300,002-char result before this
    fix. Asserting on elapsed time (not just the exception) means a future
    regression that moves the cost elsewhere still fails this test, rather
    than merely passing because *a* ValueError happened to be raised.
    """
    start = time.monotonic()
    with pytest.raises(ValueError, match="exponent magnitude"):
        symbolic.parse_safe(text)
    assert time.monotonic() - start < 1.0


def test_parse_safe_accepts_underscore_grouped_integers():
    """1_000 is a valid Python integer literal for 1000. The implicit-
    multiplication normalization used to split it into `1 * _000` and reject
    it via the underscore-identifier check -- a false positive, not a
    security hole (it failed safe), but wrong."""
    assert str(symbolic.parse_safe("1_000")) == "1000"


@pytest.mark.parametrize(
    "text",
    [
        "'just a string'",
        '\'__import__("os").system("touch /tmp/should-not-exist-hermes-test")\'',
    ],
)
def test_parse_safe_rejects_quoted_string_literals(text):
    """A quoted string is a single benign ast.Constant -- no Attribute node, no
    Name, no underscore -- so _screen's original checks let it through and
    parse_expr handed back a live Python str. Downstream ops then called
    sp.simplify()/sp.integrate() on that str, which sympifies it with SymPy's
    UNRESTRICTED default namespace -- the exact escape this module exists to
    ban, reached indirectly. Both the AST screen and the sp.Basic type gate
    must refuse this."""
    with pytest.raises(ValueError):
        symbolic.parse_safe(text)


@pytest.mark.parametrize("text", ["[1, 2]", "(1, 2)", "{1: 2}"])
def test_parse_safe_rejects_container_shaped_results(text):
    """list/dict/tuple results have the same escape shape as a bare str: they
    are not a sp.Basic, so anything relying on parse_safe's return type being
    mathematical must not receive one."""
    with pytest.raises(ValueError):
        symbolic.parse_safe(text)


def test_parse_safe_keeps_eq():
    """sp.Basic (not sp.Expr) is the gate specifically so Eq(x, 1) -- a
    Relational, not an Expr -- still parses."""
    result = symbolic.parse_safe("Eq(x, 1)")
    assert str(result) == "Eq(x, 1)"


def test_truncate_caps_and_marks():
    out = symbolic.truncate("a" * (symbolic.MAX_RESULT_CHARS + 500))
    assert len(out) < symbolic.MAX_RESULT_CHARS + 100
    assert "truncated" in out


def test_truncate_leaves_short_text_alone():
    assert symbolic.truncate("x**2") == "x**2"


def test_caret_is_exponentiation_not_xor():
    """`2^3` parsed to 1 and `2^10` to 8, returned as ok=True with no error.

    Python's grammar makes `^` bitwise XOR, and both results are valid
    expressions, so nothing raised -- the tool answered a different question
    than the one asked and reported success. sym.verify then produced a real
    FALSE DISPROOF for `2^3 == 8`, which its own docstring forbids, and
    `parsed_input` showed `1 == 8`, which does not read as a misparse.

    Caret is how a language model writes exponentiation more often than not.
    Measured on gemma4:e2b: the model hit `x^2`, got a bare TypeError, then
    abandoned the tool and did the calculus in its head.
    """
    from attestation.symbolic import parse_safe

    assert parse_safe("2^3") == 8, "caret is being parsed as XOR"
    assert parse_safe("2^10") == 1024
    assert str(parse_safe("x^2")) == "x**2"
    # The Python spelling must keep working; convert_xor only reinterprets `^`.
    assert parse_safe("2**3") == 8
    assert str(parse_safe("x**2 + 2*x")) == "x**2 + 2*x"


def test_a_refused_memory_cap_is_not_reported_as_the_result(monkeypatch):
    """macOS returns EINVAL for setrlimit(RLIMIT_AS), which Python raises as
    ValueError('current limit exceeds maximum limit') -- and the worker's
    `except ValueError`, meant for parser errors, shipped that as the answer
    to every symbolic call (first macOS CI run, 2026-08-28). The cap is
    best-effort; the subprocess timeout is the bound that always holds."""
    import queue

    def refuse(*_args):
        raise ValueError("current limit exceeds maximum limit")

    monkeypatch.setattr(symbolic.resource, "setrlimit", refuse)
    q = queue.Queue()

    symbolic._worker(q, "op_simplify", {"expr": "x + x"})

    out = q.get_nowait()
    assert out["ok"] is True, out["error"]


def test_a_caret_power_tower_is_refused_by_the_sandbox_not_the_parser():
    """Making `^` mean exponentiation makes `9^9^9` as expensive as `9**9**9`.

    Both hang if parse_safe is called directly -- SymPy tries to build
    9**387420489 -- and both are contained by run_isolated's subprocess
    timeout, which is where every MCP tool actually goes. Asserted so a future
    change that moves parsing outside that boundary fails here rather than
    hanging a caller.
    """
    from attestation.symbolic import run_isolated

    for expr in ("9**9**9", "9^9^9"):
        out = run_isolated("op_simplify", {"expr": expr}, timeout=5)
        assert out["ok"] is False, f"{expr} was not refused"
        assert "exceeded" in (out["error"] or ""), out["error"]
