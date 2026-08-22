"""Symbolic operations. Every function here runs INSIDE the isolation subprocess.

Each takes a single JSON-safe `payload` dict and returns a JSON-safe dict --
nothing SymPy-typed crosses the process boundary. Parsing always goes through
symbolic.parse_safe; this module never calls parse_expr or sympify directly.
"""

import dataclasses
import re
from typing import cast

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.physics import units as physical_units
from sympy.physics.units import Quantity, convert_to

from attestation.symbolic import parse_safe, truncate


def resolve_symbol(expr: sp.Expr, name: str | None) -> sp.Symbol:
    """Pick the symbol an operation targets. Explicit name wins; otherwise the
    expression must have exactly one free symbol -- guessing between several
    silently answers a different question than the caller asked."""
    if name:
        sym = sp.Symbol(name)
        if expr.free_symbols and sym not in expr.free_symbols:
            raise ValueError(
                f"{name!r} does not appear in {expr}; free symbols are "
                f"{sorted(map(str, expr.free_symbols))}"
            )
        return sym
    free = sorted(expr.free_symbols, key=str)
    if not free:
        return sp.Symbol("x")
    if len(free) > 1:
        names = ", ".join(str(s) for s in free)
        raise ValueError(f"ambiguous: expression has several symbols ({names}); pass symbol=")
    # free_symbols is typed as set[Basic] in SymPy's stubs, but its elements
    # are always Symbol instances -- verified at runtime.
    return cast(sp.Symbol, free[0])


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
    if out["numeric"] is None and units:
        try:
            # A units conversion leaves `18*kilometer/hour`, whose coefficient
            # IS the answer. Scoped to the units path on purpose: the same
            # fallback used to run for every symbolic result, so `x**2 + 1`
            # reported numeric=1.0 and `2*x` reported 2.0 -- the coefficient of
            # an arbitrary arg, handed back as though it were the value. A
            # confidently wrong number is worse than no number.
            out["numeric"] = float(cast(sp.Expr, expr.args[0])) if expr.args else float(expr)
        except (TypeError, ValueError, IndexError):
            out["numeric"] = None

    if out["numeric"] is None:
        out["message"] = _free_symbol_note(sp.simplify(expr))
    out["parsed_input"] = payload["expr"]
    return out


def _free_symbol_note(expr) -> str:
    """Why there is no number, named specifically enough to act on.

    An empty string when nothing is free -- the expression is unevaluatable for
    some other reason and inventing an explanation would be worse than silence.
    """
    free = sorted(str(sym) for sym in expr.free_symbols)
    if not free:
        return ""
    verb = "is" if len(free) == 1 else "are"
    return (
        f"no numeric value: {', '.join(free)} {verb} still free"
        " -- pass subs={...} to substitute them"
    )


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

    Every name is checked two ways before it can enter the namespace: it must
    not start with an underscore (blocks `__builtins__` outright, matching
    _screen's rule in symbolic.py), and the resolved attribute must be an
    actual `Quantity` instance, not just "present on the module" -- the module
    also exports functions (`convert_to`), submodules (`definitions`), and
    classes (`Quantity` itself), none of which are units. The charset also
    excludes `.`, so attribute access like `meter.__class__` cannot appear in
    the text parse_expr eventually sees at all.
    """
    if not re.fullmatch(r"[A-Za-z_0-9 */()^-]+", text or ""):
        raise ValueError(f"invalid unit expression: {text!r}")
    # standard_transformations alone starves the '**' exponent path of the
    # numeric constructors it needs at parse time (parse_expr('meter**2')
    # raises NameError: name 'Integer' is not defined) -- 'meter*meter' works
    # because it never takes that path. These are constructors, not units, so
    # they are injected directly rather than being run through the
    # unit-resolution loop below, which must keep rejecting any name that is
    # not an actual Quantity.
    namespace: dict = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational}
    for name in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)):
        if name.startswith("_"):
            raise ValueError(f"unknown unit: {name}")
        unit = getattr(physical_units, name, None)
        if not isinstance(unit, Quantity):
            raise ValueError(f"unknown unit: {name}")
        namespace[name] = unit
    return parse_expr(
        text,
        local_dict=namespace,
        global_dict={"__builtins__": {}},
        transformations=standard_transformations,
    )
