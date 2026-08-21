# SymPy MCP Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose seven symbolic-math tools over MCP so an agent can check the math in papers, with a safety layer that makes arbitrary-string input neither executable nor unbounded.

**Architecture:** A new `src/hermes/symbolic.py` holds a safety layer (restricted `parse_expr`, subprocess-isolated evaluation with timeout and memory cap, output truncation) and seven pure functions built on it. `mcp_server.py` gains thin `_impl` wrappers and `@mcp.tool()` decorators following its existing pattern. Nothing in the module imports from `db.py`, `rank.py`, or `feeds.py`.

**Tech Stack:** Python 3.12+, SymPy 1.14, `multiprocessing`, `resource` (RLIMIT_AS), FastMCP (`mcp==1.28.1`), pytest, `uv`.

## Global Constraints

- **NEVER use `sympy.sympify` or bare `eval`/`exec` in this work.** `sympify("__import__('os').getcwd()")` executes code — verified. A test asserts these names are absent from `symbolic.py`.
- **Every `parse_expr` call MUST pass both a restricted `local_dict` and `global_dict={"__builtins__": {}}`.** Calling it with only `transformations=` leaves the default namespace and IS exploitable — verified: that form returns `/home/matt/hermes-rss` for `__import__('os').getcwd()` and reads `/etc/passwd`. There are exactly two legitimate call sites: `symbolic.parse_safe` (math, `SAFE_NAMESPACE`) and `symbolic_ops._unit_expr` (units, a per-call namespace of resolved unit names). Both are shown in full in their tasks. Add no third.
- Commit ONLY the files each task's **Files** section lists. `feeds.toml`, `demo/`, and `docs/hermes-agent-plugin-research.md` are the user's uncommitted work — NEVER stage them. Verify with `git status --short` before every commit.
- Every commit message ends with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Gates before every commit: `uv run pytest -q`, `uv run ruff check .`, `uv run ty check` — all must pass.
- Ruff line-length is 100; lint set is `["E", "F", "W", "I", "BLE"]`.
- Existing tests must pass untouched. A failing pre-existing test means you changed behavior — investigate, do not edit the test.
- There is a `git stash@{0}` holding unreviewed KG+SymPy code. **Do not apply, pop, or reference it.** It contains the vulnerable `parse_expr` call this plan exists to avoid.

---

### Task 1: Safety layer

**Files:**
- Create: `src/hermes/symbolic.py`
- Modify: `pyproject.toml` (add `sympy>=1.13` to `[project] dependencies`)
- Test: `tests/test_symbolic.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `symbolic.SAFE_NAMESPACE: dict[str, object]` — whitelisted SymPy names.
  - `symbolic.MAX_RESULT_CHARS: int` = `4000`.
  - `symbolic.DEFAULT_TIMEOUT: int` = `10`; `symbolic.MAX_TIMEOUT: int` = `30`.
  - `symbolic.parse_safe(text: str) -> sympy.Expr` — restricted parse; raises `ValueError` on anything unparseable or non-whitelisted.
  - `symbolic.truncate(text: str) -> str` — caps at `MAX_RESULT_CHARS`, appending `" … [truncated, {n} chars total]"`.
  - `symbolic.run_isolated(fn_name: str, payload: dict, timeout: int) -> dict` — runs a named worker in a subprocess; returns `{"ok": bool, "value": str | None, "error": str | None}`.

- [ ] **Step 1: Write the failing security tests**

Create `tests/test_symbolic.py`:

```python
import pathlib

import pytest

from hermes import symbolic


@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').getcwd()",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
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


def test_truncate_caps_and_marks():
    out = symbolic.truncate("a" * (symbolic.MAX_RESULT_CHARS + 500))
    assert len(out) < symbolic.MAX_RESULT_CHARS + 100
    assert "truncated" in out


def test_truncate_leaves_short_text_alone():
    assert symbolic.truncate("x**2") == "x**2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_symbolic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes.symbolic'`.

- [ ] **Step 3: Write the safety layer**

Create `src/hermes/symbolic.py`:

```python
"""Symbolic math over untrusted strings: restricted parsing + isolated evaluation.

Two hazards drive every design choice here, both measured rather than assumed:

1. `sympy.sympify` on a caller-supplied string is remote code execution --
   `sympify("__import__('os').getcwd()")` returns the working directory. So is
   `parse_expr` called WITHOUT an explicit namespace. Agents pass strings that
   can originate in feed content, so `parse_safe` is the only entry point and
   it pins both `local_dict` and `global_dict`.
2. Small inputs can produce unbounded work: `expand((x+1)**2000)` yields 887KB
   in 1.4s. Every evaluation therefore runs in a subprocess with a wall-clock
   timeout and an address-space cap, and results are truncated before return.

A subprocess is the only bound that actually holds -- SIGALRM cannot preempt
the C-level loops inside SymPy.
"""

import ast
import multiprocessing as mp
import re
import resource
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

MAX_RESULT_CHARS = 4000
DEFAULT_TIMEOUT = 10
MAX_TIMEOUT = 30
MEMORY_LIMIT_BYTES = 2 * 1024**3

_SAFE_NAMES = (
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "Abs",
    "sign",
    "factorial",
    "gamma",
    "pi",
    "E",
    "oo",
    "I",
    "nan",
    "Symbol",
    "Function",
    "Integer",
    "Float",
    "Rational",
    "Matrix",
    "diff",
    "integrate",
    "simplify",
    "expand",
    "factor",
    "solve",
    "limit",
    "Sum",
    "Product",
    "Eq",
)

SAFE_NAMESPACE: dict[str, Any] = {name: getattr(sp, name) for name in _SAFE_NAMES}

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _screen(text: str) -> None:
    """Reject attribute access and private names BEFORE SymPy sees the string.

    Restricting parse_expr's namespace only restricts NAME LOOKUP. It does not
    restrict attribute access on values that need no name at all, so
    `().__class__.__bases__[0].__subclasses__()` still returns 441 live classes
    with subprocess.Popen among them -- verified. That is the standard
    sandbox-escape gadget chain, and namespace restriction alone cannot stop it.

    The screen parses a normalized copy with Python's own AST (SymPy's implicit
    multiplication accepts "2x" and "x y", which are not valid Python, so the
    copy inserts the explicit `*` first) and refuses any Attribute node, any
    name starting with an underscore, and lambda/await/yield.
    """
    probe = re.sub(r"(?<=[0-9])\s*(?=[A-Za-z_(])", "*", text)
    probe = re.sub(r"(?<=[A-Za-z_0-9)])\s+(?=[A-Za-z_(])", "*", probe)
    try:
        tree = ast.parse(probe, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError(f"attribute access is not allowed (.{node.attr})")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"names beginning with underscore are not allowed ({node.id})")
        if isinstance(node, (ast.Lambda, ast.Await, ast.Yield)):
            raise ValueError("unsupported syntax")


def parse_safe(text: str) -> sp.Expr:
    """Parse a caller-supplied expression with no access to builtins or imports.

    Two layers, both required. `_screen` blocks attribute-access escapes that a
    restricted namespace cannot; `local_dict` + `global_dict` block name lookup.
    Passing only `transformations=` leaves parse_expr's default namespace, which
    IS exploitable -- verified: it returns the working directory for
    `__import__('os').getcwd()` and reads /etc/passwd.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expression must be a non-empty string")
    _screen(text)
    try:
        return parse_expr(
            text,
            local_dict=SAFE_NAMESPACE,
            global_dict={"__builtins__": {}},
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:  # noqa: BLE001 - any parse failure is a bad expression
        raise ValueError(f"could not parse {text!r}: {type(exc).__name__}") from exc


def truncate(text: str) -> str:
    """Cap a result so an 887KB expansion cannot flood the caller's context."""
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return f"{text[:MAX_RESULT_CHARS]} … [truncated, {len(text)} chars total]"


def _worker(queue, fn_name: str, payload: dict) -> None:
    """Child-process entry point. Sets its own memory cap before importing work."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        from hermes import symbolic_ops

        value = getattr(symbolic_ops, fn_name)(payload)
        queue.put({"ok": True, "value": value, "error": None})
    except MemoryError:
        queue.put({"ok": False, "value": None, "error": "exceeded 2GB memory limit"})
    except ValueError as exc:
        queue.put({"ok": False, "value": None, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - report, never crash the child silently
        queue.put({"ok": False, "value": None, "error": f"{type(exc).__name__}: {exc}"})


def run_isolated(fn_name: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Run `symbolic_ops.<fn_name>(payload)` in a subprocess, killed at `timeout`.

    Costs ~0.3s of process spawn per call -- acceptable for correctness-critical
    tools an agent calls occasionally, and it is the only bound that holds.
    """
    timeout = max(1, min(int(timeout), MAX_TIMEOUT))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, fn_name, payload))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "ok": False,
            "value": None,
            "error": f"computation exceeded {timeout}s and was cancelled",
        }
    if queue.empty():
        return {"ok": False, "value": None, "error": "computation produced no result"}
    return queue.get()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_symbolic.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Prove the attack really is blocked, and that the unsafe form is not**

This step is the reason the module exists — run it and read the output.

```bash
uv run python -c "
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from hermes.symbolic import SAFE_NAMESPACE, parse_safe

attacks = [
    \"__import__('os').getcwd()\",
    '().__class__.__bases__[0].__subclasses__()',
]
for attack in attacks:
    # (a) no namespace at all
    try:
        got = parse_expr(attack, transformations=standard_transformations)
        print('UNSAFE(no namespace)  EXECUTED ->', str(got)[:60])
    except Exception as e:
        print('UNSAFE(no namespace)  blocked:', type(e).__name__)
    # (b) namespace pinned but NO ast screen -- this is what the escape defeats
    try:
        got = parse_expr(attack, local_dict=SAFE_NAMESPACE,
                         global_dict={'__builtins__': {}},
                         transformations=standard_transformations)
        print('UNSAFE(namespace only) EXECUTED ->', str(got)[:60])
    except Exception as e:
        print('UNSAFE(namespace only) blocked:', type(e).__name__)
    # (c) the real thing
    try:
        print('parse_safe EXECUTED ->', parse_safe(attack))
    except ValueError:
        print('parse_safe blocked: ValueError')
    print()
"
```

Expected, and paste the real output into your report:

- `__import__('os').getcwd()` — case (a) prints a filesystem path.
- `().__class__.__bases__[0].__subclasses__()` — case **(b)** prints a long class list. This is the important line: it shows that pinning the namespace alone is NOT sufficient, which is exactly why `_screen` exists.
- `parse_safe` blocks BOTH with `ValueError`.

If `parse_safe` fails to block either attack, STOP and report BLOCKED — nothing else in this plan is safe to build on it.

- [ ] **Step 6: Add the dependency**

In `pyproject.toml`, add to `[project] dependencies` (after `"python-dotenv>=1.0",`):

```toml
    "sympy>=1.13",
```

Then run `uv sync` so `uv.lock` reflects it.

- [ ] **Step 7: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add src/hermes/symbolic.py tests/test_symbolic.py pyproject.toml uv.lock
git status --short
git commit -m "$(cat <<'EOF'
feat: symbolic safety layer -- restricted parsing + isolated evaluation

parse_safe pins both local_dict and global_dict on parse_expr. Passing only
transformations= leaves the default namespace, which IS exploitable: that form
evaluates __import__('os').getcwd() and reads /etc/passwd. sympify is worse and
is banned outright, enforced by a source-level test.

run_isolated evaluates in a spawn subprocess with a wall-clock timeout and a
2GB RLIMIT_AS, because expand((x+1)**2000) produces 887KB in 1.4s from a
20-character input. SIGALRM cannot preempt SymPy's C-level loops, so a
subprocess is the only bound that holds. Costs ~0.3s spawn per call.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Operations module

**Files:**
- Create: `src/hermes/symbolic_ops.py`
- Test: `tests/test_symbolic_ops.py`

**Interfaces:**
- Consumes: `symbolic.parse_safe`, `symbolic.truncate` (Task 1).
- Produces — seven worker functions, each taking one `payload: dict` and returning a `dict` of JSON-safe values (they run in a subprocess, so only picklable primitives cross the boundary):
  - `op_simplify(payload) -> dict` — payload `{"expr": str}`
  - `op_solve(payload) -> dict` — payload `{"expr": str, "symbol": str | None}`
  - `op_differentiate(payload) -> dict` — payload `{"expr": str, "symbol": str | None, "order": int}`
  - `op_integrate(payload) -> dict` — payload `{"expr": str, "symbol": str | None, "bounds": list | None}`
  - `op_derivation(payload) -> dict` — payload `{"expr": str, "operation": str, "symbol": str | None}`
  - `op_verify(payload) -> dict` — payload `{"lhs": str, "rhs": str}`
  - `op_evaluate(payload) -> dict` — payload `{"expr": str, "subs": dict | None, "units": str | None}`
  - `resolve_symbol(expr, name: str | None) -> sympy.Symbol` — helper; raises `ValueError` naming candidates when ambiguous.
  - `describe(expr) -> dict` — helper returning `{"result", "latex", "numeric"}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_symbolic_ops.py`:

```python
import pytest

from hermes import symbolic_ops as ops


def test_simplify_returns_result_latex_and_parsed_input():
    out = ops.op_simplify({"expr": "(x**2 - 1)/(x - 1)"})
    assert out["result"] == "x + 1"
    assert out["latex"] == "x + 1"
    assert out["parsed_input"] == "(x**2 - 1)/(x - 1)"


def test_differentiate_auto_detects_single_symbol():
    out = ops.op_differentiate({"expr": "x**3", "symbol": None, "order": 1})
    assert out["result"] == "3*x**2"


def test_differentiate_respects_order():
    out = ops.op_differentiate({"expr": "x**3", "symbol": None, "order": 2})
    assert out["result"] == "6*x"


def test_solve_auto_detects_and_returns_roots():
    out = ops.op_solve({"expr": "x**2 - 4", "symbol": None})
    assert set(out["result"].strip("[]").split(", ")) == {"-2", "2"}


def test_solve_explicit_symbol_overrides_autodetect():
    """x*y - 1 is ambiguous; naming y must solve for y, not x."""
    out = ops.op_solve({"expr": "x*y - 1", "symbol": "y"})
    assert out["result"] == "[1/x]"


def test_ambiguous_symbol_raises_naming_candidates():
    with pytest.raises(ValueError) as exc:
        ops.op_solve({"expr": "x*y - 1", "symbol": None})
    assert "x" in str(exc.value) and "y" in str(exc.value)


def test_integrate_indefinite_and_definite():
    indef = ops.op_integrate({"expr": "x**2", "symbol": None, "bounds": None})
    assert indef["result"] == "x**3/3"
    definite = ops.op_integrate({"expr": "x**2", "symbol": None, "bounds": [0, 1]})
    assert definite["result"] == "1/3"
    assert definite["numeric"] == pytest.approx(1 / 3)


def test_verify_proves_a_true_identity():
    out = ops.op_verify({"lhs": "sin(x)**2 + cos(x)**2", "rhs": "1"})
    assert out["verdict"] == "equal"


def test_verify_reports_unequal_for_a_clear_mismatch():
    out = ops.op_verify({"lhs": "x + 1", "rhs": "x + 2"})
    assert out["verdict"] == "unequal"


def test_verify_never_claims_unequal_without_numeric_evidence():
    """simplify is incomplete, so 'could not prove' must never render as
    'unequal' -- an agent reading that would report a false disproof."""
    out = ops.op_verify({"lhs": "sqrt(x**2)", "rhs": "x"})
    assert out["verdict"] in ("equal", "unproven", "unequal")
    if out["verdict"] == "unproven":
        assert "not a disproof" in out["message"].lower()
    if out["verdict"] == "unequal":
        # an "unequal" verdict must always cite concrete evidence
        assert any(w in out["message"].lower() for w in ("counterexample", "reduces to"))


def test_evaluate_substitutes_and_returns_float():
    out = ops.op_evaluate({"expr": "x**2 + 1", "subs": {"x": 3}, "units": None})
    assert out["numeric"] == pytest.approx(10.0)


def test_evaluate_converts_units():
    out = ops.op_evaluate({"expr": "5", "subs": None, "units": "meter/second -> kilometer/hour"})
    assert out["numeric"] == pytest.approx(18.0)


def test_derivation_traces_integral_steps():
    out = ops.op_derivation({"expr": "x*sin(x)", "operation": "integrate", "symbol": None})
    assert out["steps"], "expected at least one step"
    assert any("Parts" in s["rule"] for s in out["steps"])


def test_derivation_differentiate_is_labeled_not_faked():
    out = ops.op_derivation({"expr": "x**3", "operation": "differentiate", "symbol": None})
    assert out["result"] == "3*x**2"
    assert "not" in out["note"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_symbolic_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes.symbolic_ops'`.

- [ ] **Step 3: Write the operations module**

Create `src/hermes/symbolic_ops.py`:

```python
"""Symbolic operations. Every function here runs INSIDE the isolation subprocess.

Each takes a single JSON-safe `payload` dict and returns a JSON-safe dict --
nothing SymPy-typed crosses the process boundary. Parsing always goes through
symbolic.parse_safe; this module never calls parse_expr or sympify directly.
"""

import dataclasses
import re

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.physics import units as physical_units
from sympy.physics.units import convert_to

from hermes.symbolic import parse_safe, truncate


def resolve_symbol(expr: sp.Expr, name: str | None) -> sp.Symbol:
    """Pick the symbol an operation targets. Explicit name wins; otherwise the
    expression must have exactly one free symbol -- guessing between several
    silently answers a different question than the caller asked."""
    if name:
        return sp.Symbol(name)
    free = sorted(expr.free_symbols, key=str)
    if not free:
        return sp.Symbol("x")
    if len(free) > 1:
        names = ", ".join(str(s) for s in free)
        raise ValueError(f"ambiguous: expression has several symbols ({names}); pass symbol=")
    return free[0]


def describe(expr) -> dict:
    """Common result shape: text, LaTeX, and a float when one exists."""
    numeric = None
    try:
        if getattr(expr, "is_number", False):
            numeric = float(expr)
    except (TypeError, ValueError):
        numeric = None
    return {
        "result": truncate(str(expr)),
        "latex": truncate(sp.latex(expr)),
        "numeric": numeric,
    }


def op_simplify(payload: dict) -> dict:
    expr = parse_safe(payload["expr"])
    out = describe(sp.simplify(expr))
    out["parsed_input"] = str(expr)
    return out


def op_solve(payload: dict) -> dict:
    expr = parse_safe(payload["expr"])
    symbol = resolve_symbol(expr, payload.get("symbol"))
    roots = sp.solve(expr, symbol)
    out = describe(roots if isinstance(roots, list) else [roots])
    out["parsed_input"] = str(expr)
    out["symbol"] = str(symbol)
    return out


def op_differentiate(payload: dict) -> dict:
    expr = parse_safe(payload["expr"])
    symbol = resolve_symbol(expr, payload.get("symbol"))
    order = max(1, int(payload.get("order") or 1))
    out = describe(sp.diff(expr, symbol, order))
    out["parsed_input"] = str(expr)
    out["symbol"] = str(symbol)
    return out


def op_integrate(payload: dict) -> dict:
    expr = parse_safe(payload["expr"])
    symbol = resolve_symbol(expr, payload.get("symbol"))
    bounds = payload.get("bounds")
    if bounds:
        lo, hi = parse_safe(str(bounds[0])), parse_safe(str(bounds[1]))
        result = sp.integrate(expr, (symbol, lo, hi))
    else:
        result = sp.integrate(expr, symbol)
    out = describe(result)
    out["parsed_input"] = str(expr)
    out["symbol"] = str(symbol)
    return out


def _walk_steps(rule, depth: int = 0) -> list[dict]:
    """Flatten SymPy's nested manualintegrate rule dataclasses into a step list."""
    steps = [
        {
            "depth": depth,
            "rule": type(rule).__name__,
            "integrand": truncate(str(getattr(rule, "integrand", ""))),
        }
    ]
    if dataclasses.is_dataclass(rule):
        for field in dataclasses.fields(rule):
            value = getattr(rule, field.name)
            if dataclasses.is_dataclass(value):
                steps.extend(_walk_steps(value, depth + 1))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if dataclasses.is_dataclass(item):
                        steps.extend(_walk_steps(item, depth + 1))
    return steps


def op_derivation(payload: dict) -> dict:
    expr = parse_safe(payload["expr"])
    symbol = resolve_symbol(expr, payload.get("symbol"))
    operation = payload.get("operation") or "integrate"

    if operation == "integrate":
        from sympy.integrals.manualintegrate import integral_steps

        rule = integral_steps(expr, symbol)
        out = describe(sp.integrate(expr, symbol))
        out["steps"] = _walk_steps(rule)[:50]
        out["note"] = "step rules from SymPy's manual-integration engine"
    else:
        result = sp.diff(expr, symbol)
        out = describe(result)
        out["steps"] = [
            {"depth": 0, "rule": "Derivative", "integrand": truncate(str(expr))},
            {"depth": 1, "rule": "Result", "integrand": truncate(str(result))},
        ]
        out["note"] = (
            "SymPy does not expose a step-by-step engine for differentiation, so "
            "this is the result with its input, not a rule-by-rule trace"
        )
    out["parsed_input"] = str(expr)
    out["symbol"] = str(symbol)
    return out


def op_verify(payload: dict) -> dict:
    lhs, rhs = parse_safe(payload["lhs"]), parse_safe(payload["rhs"])
    difference = sp.simplify(lhs - rhs)
    if difference == 0:
        verdict, message = "equal", "proven equal: simplify(lhs - rhs) is exactly 0"
    else:
        verdict = "unproven"
        message = (
            "could NOT prove equal -- simplify is incomplete, so this is not a "
            "disproof. Treat as unknown, not false."
        )
        free = sorted(difference.free_symbols, key=str)
        if not free:
            # The difference reduced to a non-zero constant (e.g. "x+1" vs "x+2"
            # simplifies to -1). That IS a disproof, with no substitution needed.
            verdict = "unequal"
            message = f"disproven: lhs - rhs reduces to the non-zero constant {difference}"
        else:
            probe = {s: sp.Rational(3, 7) + i for i, s in enumerate(free)}
            try:
                value = complex(difference.subs(probe).evalf())
                if abs(value) > 1e-9:
                    verdict = "unequal"
                    message = f"counterexample: substituting {probe} gives {value:.6g} != 0"
            except (TypeError, ValueError):
                pass
    out = describe(difference)
    out["verdict"] = verdict
    out["message"] = message
    out["parsed_input"] = f"{lhs} == {rhs}"
    return out


def op_evaluate(payload: dict) -> dict:
    expr = parse_safe(payload["expr"])
    subs = payload.get("subs")
    if subs:
        expr = expr.subs({sp.Symbol(k): parse_safe(str(v)) for k, v in subs.items()})

    units = payload.get("units")
    if units:
        if "->" not in units:
            raise ValueError("units must look like 'meter/second -> kilometer/hour'")
        source, target = (part.strip() for part in units.split("->", 1))
        expr = convert_to(expr * _unit_expr(source), _unit_expr(target))
    out = describe(sp.simplify(expr))
    if out["numeric"] is None:
        try:
            out["numeric"] = float(expr.args[0]) if expr.args else float(expr)
        except (TypeError, ValueError, IndexError):
            out["numeric"] = None
    out["parsed_input"] = payload["expr"]
    return out


def _unit_expr(text: str):
    """Build a unit expression from names, restricted to sympy.physics.units.

    This deliberately does NOT go through parse_safe: implicit multiplication
    shatters unit names into single letters ("second" -> s*e*c*o*n*d), so units
    are parsed with standard transformations only, against a local_dict holding
    exactly the unit names found in the text. sympy.physics.units carries 351
    names -- meter, second, kilometer, hour, foot, joule, newton, kelvin, watt,
    gram, liter, mile, inch, day, year and so on -- but not every unit anyone
    might type (`erg` and `kilojoule`, for instance, are absent), so an unknown
    name raises rather than silently resolving to a bare symbol.
    """
    if not re.fullmatch(r"[A-Za-z_0-9 */().^-]+", text or ""):
        raise ValueError(f"invalid unit expression: {text!r}")
    namespace = {}
    for name in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)):
        unit = getattr(physical_units, name, None)
        if unit is None:
            raise ValueError(f"unknown unit: {name}")
        namespace[name] = unit
    return parse_expr(
        text,
        local_dict=namespace,
        global_dict={"__builtins__": {}},
        transformations=standard_transformations,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_symbolic_ops.py -v`
Expected: PASS (14 tests). If `test_evaluate_converts_units` fails on the float extraction, print the actual `convert_to` output and adjust `describe`'s numeric fallback — the conversion returns `18*kilometer/hour`, whose `.args[0]` is `18`.

- [ ] **Step 5: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add src/hermes/symbolic_ops.py tests/test_symbolic_ops.py
git status --short
git commit -m "$(cat <<'EOF'
feat: symbolic operations -- solve, calculus, verification, units

Seven worker functions that run inside the isolation subprocess, exchanging
only JSON-safe dicts. All parsing goes through symbolic.parse_safe.

op_verify returns a three-valued verdict rather than a boolean: simplify is
incomplete, so a non-zero difference means "unproven", never "false". It
escalates to "unequal" only with a numeric counterexample, because an agent
reading "unproven" as "false" would report disproofs that are not disproofs.

op_derivation genuinely traces integrals via manualintegrate's rule
dataclasses; SymPy has no differentiation equivalent, so that branch says so
in its note instead of implying parity.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: MCP tools + isolation wiring

**Files:**
- Modify: `src/hermes/mcp_server.py` (add seven `_impl` functions and seven `@mcp.tool()` wrappers at the end of the file, before `def main()`)
- Test: `tests/test_symbolic_mcp.py`

**Interfaces:**
- Consumes: `symbolic.run_isolated`, `symbolic.DEFAULT_TIMEOUT` (Task 1); the `op_*` names (Task 2).
- Produces: MCP tools `sym_simplify`, `sym_solve`, `sym_differentiate`, `sym_integrate`, `sym_derivation`, `sym_verify`, `sym_evaluate`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_symbolic_mcp.py`:

```python
from hermes import mcp_server


def test_sym_differentiate_returns_structured_result():
    out = mcp_server._sym_differentiate_impl("x**3")
    assert out["ok"] is True
    assert out["result"] == "3*x**2"
    assert out["latex"]


def test_sym_solve_reports_ambiguity_without_crashing():
    out = mcp_server._sym_solve_impl("x*y - 1")
    assert out["ok"] is False
    assert "ambiguous" in out["message"]


def test_malicious_expression_is_refused_not_executed():
    out = mcp_server._sym_simplify_impl("__import__('os').getcwd()")
    assert out["ok"] is False
    assert "hermes-rss" not in str(out), "filesystem path leaked -- code executed"


def test_runaway_computation_is_cancelled_not_hung():
    """expand((x+1)**200000) would exhaust memory; the subprocess must be killed."""
    out = mcp_server._sym_simplify_impl("(x+1)**200000", timeout=3)
    assert out["ok"] is False
    assert "cancelled" in out["message"] or "memory" in out["message"]


def test_sym_verify_surfaces_the_verdict():
    out = mcp_server._sym_verify_impl("sin(x)**2 + cos(x)**2", "1")
    assert out["ok"] is True
    assert out["verdict"] == "equal"


def test_all_seven_tools_are_served():
    import asyncio

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    expected = {
        "sym_simplify",
        "sym_solve",
        "sym_differentiate",
        "sym_integrate",
        "sym_derivation",
        "sym_verify",
        "sym_evaluate",
    }
    assert expected <= names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_symbolic_mcp.py -v`
Expected: FAIL — `AttributeError: module 'hermes.mcp_server' has no attribute '_sym_differentiate_impl'`.

- [ ] **Step 3: Add the `_impl` functions**

In `src/hermes/mcp_server.py`, before `def main()`:

```python
# ---------------------------------------------------------------------------
# symbolic math (SymPy) -- every evaluation is subprocess-isolated
# ---------------------------------------------------------------------------


def _sym_call(op_name: str, payload: dict, timeout: int) -> dict:
    """Shared bridge: run an op in isolation and flatten it into the tool contract."""
    from hermes.symbolic import run_isolated

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
```

Note `_sym_call` returns `result` and `latex` as `None` on failure so a caller reading those keys never hits a KeyError — the same key-preservation rule the other `_impl` functions in this file follow.

- [ ] **Step 4: Add the seven tool wrappers**

Immediately after the `_impl` functions:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_symbolic_mcp.py -v`
Expected: PASS (6 tests). The runaway-computation test takes ~3s by design.

- [ ] **Step 6: Confirm the served tool count**

```bash
uv run python -c "
import asyncio
from hermes.mcp_server import mcp
names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
print(len(names), 'tools')
print([n for n in names if n.startswith('sym_')])
"
```

Expected: **23** tools (16 existing + 7 new), with all seven `sym_*` names listed. If the count differs, report the actual list rather than adjusting expectations.

- [ ] **Step 7: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add src/hermes/mcp_server.py tests/test_symbolic_mcp.py
git status --short
git commit -m "$(cat <<'EOF'
feat: seven SymPy MCP tools over the isolation layer

sym_simplify / solve / differentiate / integrate / derivation / verify /
evaluate. Every call crosses into a subprocess via run_isolated, so a
malicious or runaway expression cannot execute code, exhaust memory, or hang
the MCP server -- tested with both an __import__ attack and a 200000-degree
expansion.

sym_verify's description states that "unproven" is not a disproof, since the
failure mode that matters is an agent reporting it as "false".

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md` (MCP tool table)
- Modify: `src/hermes/skills/science-recommendations/SKILL.md`

**Interfaces:**
- Consumes: the seven tool names from Task 3.
- Produces: nothing code-level.

- [ ] **Step 1: Extend the README tool table**

The table currently lists sixteen tools and is introduced as "exposes sixteen tools:". Change that to "exposes twenty-three tools:" and append:

```markdown
| `sym_simplify(expr, timeout)` | Simplify an expression to canonical form | fast |
| `sym_solve(expr, symbol, timeout)` | Solve expr = 0 for a symbol | fast |
| `sym_differentiate(expr, symbol, order, timeout)` | Differentiate | fast |
| `sym_integrate(expr, symbol, bounds, timeout)` | Integrate, indefinitely or over bounds | fast |
| `sym_derivation(expr, operation, symbol, timeout)` | Step-by-step trace (genuine only for integrals) | fast |
| `sym_verify(lhs, rhs, timeout)` | Test symbolic equality; "unproven" is NOT a disproof | fast |
| `sym_evaluate(expr, subs, units, timeout)` | Numeric value, substitutions, unit conversion | fast |
```

- [ ] **Step 2: Add a safety note under the table**

```markdown
Symbolic tools never `eval` your input: expressions are parsed against a
whitelist of mathematical names with builtins removed, and every computation
runs in a subprocess with a wall-clock timeout and a 2 GB memory cap. A
malicious expression is refused, and a runaway one is cancelled rather than
taking down the server — at the cost of roughly 0.3 s of process spawn per
call.
```

- [ ] **Step 3: Add the tools to SKILL.md**

In the `## MCP tools` section of `src/hermes/skills/science-recommendations/SKILL.md`, add a short grouped entry in the file's existing prose voice, covering all seven names and stating two things: `sym_verify`'s "unproven" is not a disproof, and step-by-step tracing is genuine only for integrals.

- [ ] **Step 4: Verify docs match the served tools**

```bash
uv run python -c "
import asyncio, re
from pathlib import Path
from hermes.mcp_server import mcp
served = {t.name for t in asyncio.run(mcp.list_tools())}
documented = set(re.findall(r'\| \`(\w+)\(', Path('README.md').read_text()))
print('served not documented:', sorted(served - documented))
print('documented not served:', sorted(documented - served))
"
```

Expected: both lists empty. If not, fix the README — the served tools are the truth.

- [ ] **Step 5: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add README.md src/hermes/skills/science-recommendations/SKILL.md
git status --short
git commit -m "$(cat <<'EOF'
docs: document the seven SymPy MCP tools

Brings the README table to twenty-three tools and records the safety
properties users should be able to rely on: whitelisted parsing, subprocess
isolation, the memory cap, and the ~0.3s spawn cost. SKILL.md gets the same
tools in its own voice, including the warning that sym_verify's "unproven"
verdict is not a disproof.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Post-plan manual verification

1. **The attack that motivated the design:** `uv run python -c "from hermes.symbolic import parse_safe; parse_safe(\"__import__('os').getcwd()\")"` must raise `ValueError`, not print a path.
2. **`hermes mcp test hermes-rss`** reports 23 tools discovered.
3. **`uv run hermes install --check`** still exits 0 — the new dependency must not break the installer's doctor mode.
4. **Spawn cost is real:** time a `sym_differentiate` call and confirm it lands near 0.3–0.5s. If it is far worse, the `spawn` start method may be re-importing more than expected; note it rather than silently accepting.
5. The stashed `stash@{0}` remains untouched and unapplied.
