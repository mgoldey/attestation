"""Symbolic math over untrusted strings: restricted parsing + isolated evaluation.

Two hazards drive every design choice here, both measured rather than assumed:

1. `sympy.sympify` on a caller-supplied string is remote code execution --
   `sympify("__import__('os').getcwd()")` returns the working directory. So is
   `parse_expr` called WITHOUT an explicit namespace. Agents pass strings that
   can originate in feed content, so `parse_safe` is the only entry point and
   it pins both `local_dict` and `global_dict`.
2. Small inputs can produce unbounded work: `expand((x+1)**2000)` yields 887KB
   in 1.4s. Every EVALUATION therefore runs in a subprocess with a wall-clock
   timeout and an address-space cap, and results are truncated before return.
   Numeric-literal PARSING is a separate hazard with its own bound: it happens
   outside that subprocess, inside `parse_expr` itself, so `parse_safe('1e300000')`
   costs 7.1s and builds a 300,002-char number before evaluation ever starts --
   `_screen` rejects literals above a digit/exponent ceiling for exactly this
   reason, cheaply, before parse_expr runs.

A subprocess is the only bound that holds against unbounded EVALUATION --
SIGALRM cannot preempt the C-level loops inside SymPy. Pathological literals
are bounded a different way, by refusing them before parsing.
"""

import ast
import io
import multiprocessing as mp
import re
import resource
import tokenize
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

MAX_RESULT_CHARS = 4000
DEFAULT_TIMEOUT = 10
MAX_TIMEOUT = 30
MEMORY_LIMIT_BYTES = 2 * 1024**3

# `parse_expr` builds arbitrary-precision SymPy numbers directly from literal
# text, entirely outside run_isolated's subprocess/timeout boundary. Cost
# scales with the digit count of the CONSTRUCTED value, not the length of the
# input text: `parse_safe('1e300000')` measured at 7.1s and a 300,002-char
# result from a 9-character string. Measured at the boundary, a 1000-digit /
# 1000-exponent literal parses in well under a millisecond, and no legitimate
# numeric constant in a paper needs anywhere near that many digits of
# precision. An exponent ceiling of 1e6 (as opposed to 1e3) is NOT safe --
# `1e999999` alone was measured to exceed 30s in parse_expr, because cost
# grows with the resulting digit count (~exponent), not with the exponent's
# own digit count.
MAX_LITERAL_DIGITS = 1000
MAX_LITERAL_EXPONENT = 1000

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

# convert_xor is NOT optional here. Without it Python's grammar wins and `^`
# is bitwise XOR, so `2^3` parsed to 1 and `2^10` to 8 -- returned as ok=True
# with no error, because both are valid expressions. sym.verify then produced
# an actual FALSE DISPROOF for `2^3 == 8`, which its own docstring forbids.
#
# Caret is how a language model writes exponentiation more often than not, and
# a symbolic engine that quietly answers a different question than the one
# asked is worse than one that refuses. Measured on gemma4:e2b: the model hit
# `x^2`, got a bare TypeError, abandoned the tool and did the calculus in its
# head -- which is the whole thing this tool exists to prevent.
_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


def _check_literal_magnitude(text: str) -> None:
    """Reject numeric literals that are cheap to write but expensive to build.

    `parse_expr` converts literal text straight into an arbitrary-precision
    SymPy number, and that conversion runs before run_isolated's subprocess
    boundary ever starts -- there is no timeout or memory cap around it.
    Tokenizing the raw text (underscores and all) and bounding digit count and
    exponent magnitude catches this while it is still nearly free: ast.parse
    and Python's own float folding do NOT reproduce the cost, since Python
    silently collapses `1e300000` to `inf` where SymPy instead builds the full
    300,002-digit number.
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        numbers = [tok.string for tok in tokens if tok.type == tokenize.NUMBER]
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return  # malformed input is ast.parse's problem, not this check's
    for raw in numbers:
        digits = raw.replace("_", "")
        digit_count = sum(c.isdigit() for c in digits)
        if digit_count > MAX_LITERAL_DIGITS:
            raise ValueError(
                f"numeric literal has {digit_count} digits, "
                f"exceeds the {MAX_LITERAL_DIGITS}-digit limit"
            )
        exp_match = re.search(r"[eE]([+-]?\d+)", digits)
        if exp_match and abs(int(exp_match.group(1))) > MAX_LITERAL_EXPONENT:
            raise ValueError(
                f"exponent magnitude {exp_match.group(1)} exceeds the {MAX_LITERAL_EXPONENT} limit"
            )


def _screen(text: str) -> None:
    """Reject attribute access, private names, and pathological literals
    BEFORE SymPy sees the string.

    Restricting parse_expr's namespace only restricts NAME LOOKUP. It does not
    restrict attribute access on values that need no name at all, so
    `().__class__.__bases__[0].__subclasses__()` still returns 441 live classes
    with subprocess.Popen among them -- verified. That is the standard
    sandbox-escape gadget chain, and namespace restriction alone cannot stop it.
    Nor does it bound the cost of numeric-literal construction -- see
    `_check_literal_magnitude`.

    The screen parses a normalized copy with Python's own AST (SymPy's implicit
    multiplication accepts "2x" and "x y", which are not valid Python, so the
    copy inserts the explicit `*` first -- but never inside an underscore-
    grouped numeric literal like `1_000`, which must survive as one token) and
    refuses any Attribute node, any name starting with an underscore, and
    lambda/await/yield.
    """
    _check_literal_magnitude(text)
    probe = re.sub(r"(?<=[0-9])\s*(?=[A-Za-z(])|(?<=[0-9])\s*(?=_(?!\d))", "*", text)
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
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise ValueError("string literals are not allowed")


def parse_safe(text: str) -> sp.Expr:
    """Parse a caller-supplied expression with no access to builtins or imports.

    Two layers, both required. `_screen` blocks attribute-access escapes that a
    restricted namespace cannot, and bounds pathological numeric literals (see
    `_check_literal_magnitude`); `local_dict` + `global_dict` block name lookup.
    Passing only `transformations=` leaves parse_expr's default namespace, which
    IS exploitable -- verified: it returns the working directory for
    `__import__('os').getcwd()` and reads /etc/passwd.

    Contract: this function is bounded against pathological LITERALS by
    `_screen`, which runs before any SymPy parsing work -- but that bound
    covers literal MAGNITUDE only, not evaluator cost. Parsing an ordinary-
    looking expression can still hang: `9**9**9` runs past 60s inside
    `parse_expr(evaluate=True)`, before any subprocess exists to bound it.
    Callers must route parse_safe (and anything downstream of it) through
    `run_isolated` rather than assuming this function returns quickly.

    It is NOT bounded against expensive EVALUATION either -- `simplify`,
    `integrate`, `expand`, and friends can still do unbounded work on a
    perfectly ordinary-looking parsed expression (e.g. `expand((x+1)**2000)`).
    That cost belongs inside `run_isolated`, which is the only boundary with a
    wall-clock timeout and a memory cap.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expression must be a non-empty string")
    _screen(text)
    try:
        result = parse_expr(
            text,
            local_dict=SAFE_NAMESPACE,
            global_dict={"__builtins__": {}},
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"could not parse {text!r}: {type(exc).__name__}") from exc
    # sp.Basic is the common ancestor for Expr, Eq/Relational, and booleans,
    # but SymPy's Matrix containers (unlike their individual elements) are
    # NOT Basic subclasses -- MatrixBase is a separate hierarchy rooted at
    # Printable, not Basic. Both must be accepted; str/list/dict/tuple are
    # neither, which is exactly the escape shape this gate closes.
    if not isinstance(result, (sp.Basic, sp.MatrixBase)):
        raise ValueError("parsed expression must be a mathematical object")
    return result


def truncate(text: str) -> str:
    """Cap a result so an 887KB expansion cannot flood the caller's context."""
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return f"{text[:MAX_RESULT_CHARS]} … [truncated, {len(text)} chars total]"


def _worker(queue, fn_name: str, payload: dict) -> None:
    """Child-process entry point. Sets its own memory cap before importing work."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        from attestation import symbolic_ops

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
