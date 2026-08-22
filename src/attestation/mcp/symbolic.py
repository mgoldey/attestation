"""Symbolic mathematics tools: the `sym.*` namespace.

Every op runs in an isolated subprocess with its own resource limits -- see
`symbolic.run_isolated`. That boundary is the only complexity in this domain,
and `_call` below is where it is hidden, which is why these tools need no
database, no user, and no envelope decorator: `run_isolated` already returns
the contract.
"""

from attestation.symbolic import run_isolated

# Every key any sym op can return, so a failure carries the same shape as a
# success. These tools predate mcp/_tool.py's @tool decorator and build their
# own envelope, which meant a failed sym_evaluate simply omitted `numeric` and
# `parsed_input` -- a caller reading result["numeric"] to see whether an
# expression evaluated got a KeyError instead of a None.
_EMPTY = {
    "result": None,
    "latex": None,
    "numeric": None,
    "steps": None,
    "equal": None,
    "symbol": None,
    "parsed_input": None,
}


def _call(op_name: str, payload: dict, timeout: int) -> dict:
    """Run an op in isolation and flatten it into the tool contract."""
    outcome = run_isolated(op_name, payload, timeout)
    if not outcome["ok"]:
        return {"ok": False, "message": outcome["error"], **_EMPTY}
    return {"ok": True, "message": "", **_EMPTY, **outcome["value"]}


def register(mcp) -> None:
    """Attach every sym.* tool to the server."""

    @mcp.tool(name="sym.simplify")
    def sym_simplify(expr: str, timeout: int = 10) -> dict:
        """Simplify a mathematical expression to canonical form.

        Example: "(x**2 - 1)/(x - 1)" -> "x + 1". Returns the result as text and
        LaTeX, plus how the input was parsed so a misread is visible.

        """
        return _sym_simplify(expr, timeout)

    @mcp.tool(name="sym.solve")
    def sym_solve(expr: str, symbol: str | None = None, timeout: int = 10) -> dict:
        """Solve expr = 0 for a symbol. Example: "x**2 - 4" -> [-2, 2].

        The symbol is auto-detected when the expression has exactly one; pass
        `symbol` explicitly when there are several (otherwise the call is refused
        rather than guessing which variable you meant).

        """
        return _sym_solve(expr, symbol, timeout)

    @mcp.tool(name="sym.differentiate")
    def sym_differentiate(
        expr: str, symbol: str | None = None, order: int = 1, timeout: int = 10
    ) -> dict:
        """Differentiate an expression. Example: "x**3" -> "3*x**2"."""
        return _sym_differentiate(expr, symbol, order, timeout)

    @mcp.tool(name="sym.integrate")
    def sym_integrate(
        expr: str, symbol: str | None = None, bounds: list | None = None, timeout: int = 10
    ) -> dict:
        """Integrate an expression, indefinitely or over `bounds` as [low, high].

        Example: "x**2" -> "x**3/3"; with bounds [0, 1] -> "1/3".

        """
        return _sym_integrate(expr, symbol, bounds, timeout)

    @mcp.tool(name="sym.derivation")
    def sym_derivation(
        expr: str, operation: str = "integrate", symbol: str | None = None, timeout: int = 10
    ) -> dict:
        """Show the steps of a derivation.

        Genuine rule-by-rule tracing exists only for `operation="integrate"`.
        For "differentiate" SymPy has no step engine, so the response returns the
        result with a note saying so rather than pretending to a trace.

        """
        return _sym_derivation(expr, operation, symbol, timeout)

    @mcp.tool(name="sym.verify")
    def sym_verify(lhs: str, rhs: str, timeout: int = 10) -> dict:
        """Check whether two expressions are mathematically equal.

        Returns verdict "equal" (proven), "unequal" (a numeric counterexample was
        found), or "unproven". IMPORTANT: "unproven" means the checker could not
        decide -- it is NOT a disproof, and must not be reported as "false".

        """
        return _sym_verify(lhs, rhs, timeout)

    @mcp.tool(name="sym.evaluate")
    def sym_evaluate(
        expr: str, subs: dict | None = None, units: str | None = None, timeout: int = 10
    ) -> dict:
        """Evaluate an expression numerically, optionally substituting values or
        converting units.

        Substitution: expr "x**2 + 1" with subs {"x": 3} -> 10.
        Units: expr "5" with units "meter/second -> kilometer/hour" -> 18.

        """
        return _sym_evaluate(expr, subs, units, timeout)


def _sym_simplify(expr: str, timeout: int = 10) -> dict:
    return _call("op_simplify", {"expr": expr}, timeout)


def _sym_solve(expr: str, symbol: str | None = None, timeout: int = 10) -> dict:
    return _call("op_solve", {"expr": expr, "symbol": symbol}, timeout)


def _sym_differentiate(
    expr: str, symbol: str | None = None, order: int = 1, timeout: int = 10
) -> dict:
    return _call("op_differentiate", {"expr": expr, "symbol": symbol, "order": order}, timeout)


def _sym_integrate(
    expr: str, symbol: str | None = None, bounds: list | None = None, timeout: int = 10
) -> dict:
    return _call("op_integrate", {"expr": expr, "symbol": symbol, "bounds": bounds}, timeout)


def _sym_derivation(
    expr: str, operation: str = "integrate", symbol: str | None = None, timeout: int = 10
) -> dict:
    return _call("op_derivation", {"expr": expr, "operation": operation, "symbol": symbol}, timeout)


def _sym_verify(lhs: str, rhs: str, timeout: int = 10) -> dict:
    return _call("op_verify", {"lhs": lhs, "rhs": rhs}, timeout)


def _sym_evaluate(
    expr: str, subs: dict | None = None, units: str | None = None, timeout: int = 10
) -> dict:
    return _call("op_evaluate", {"expr": expr, "subs": subs, "units": units}, timeout)
