# SymPy MCP tools — design

**Date:** 2026-08-06
**Status:** approved (brainstorming dialogue; all sections approved)

## Problem

hermes-rss surfaces papers but cannot check their math. An agent reading an
abstract has no way to simplify an expression, solve for a variable, take a
derivative, or test whether a claimed identity holds — it can only guess,
and local models guess badly at symbolic algebra.

SymPy answers this exactly, and is already installed (1.14.0, transitive via
another dependency). The work is exposing it as MCP tools **safely**, which
is where the entire difficulty lies.

## The two hazards (both verified, not hypothetical)

### 1. `sympify` on agent input is remote code execution

```
>>> sympy.sympify("__import__('os').getcwd()")
/home/matt/hermes-rss
```

That is code execution from a string an agent supplies. An abstract
containing a prompt injection could reach it. `sympify` (and bare `eval`)
is therefore **banned from this module** — a test asserts the name does not
appear in the source.

The safe path, verified:

```python
parse_expr(
    text,
    local_dict=SAFE_NAMESPACE,  # ~30 explicit sympy names
    global_dict={"__builtins__": {}},  # no builtins reachable
    transformations=standard_transformations + (implicit_multiplication_application,),
)
```

Verified behavior: `x**2 + sin(x)` parses correctly, `diff(x**3, x)` gives
`3*x**2`, `solve(x**2 - 4)` gives `[-2, 2]`; both
`__import__('os').getcwd()` and `open("/etc/passwd").read()` raise
`AttributeError`.

`SAFE_NAMESPACE` holds only mathematical names — trig, `exp`, `log`,
`sqrt`, `pi`, `E`, `oo`, `Symbol`, `Function`, `Integer`, `Float`,
`Rational`, `Matrix`, and the calculus entry points. No `__import__`, no
`open`, no `eval`, no module objects.

### 2. Computation is unbounded in both time and memory

Measured on this machine:

| Input | Result |
|---|---|
| `expand((x+1)**2000)` | 887,259 characters in 1.4s |
| `integrate(exp(x**2)*sin(x)/log(x), x)` | 9.6s |

A 20-character input producing 887KB is simultaneously a memory hazard and
a context-flooding hazard: returning that string to an agent would consume
its entire context window. This matters concretely — this machine
hard-rebooted from an Ollama OOM cascade on 2026-08-06, so resource
exhaustion is a demonstrated failure mode here, not a theoretical one.

**Every evaluation runs in a `multiprocessing.Process`** with:

- a wall-clock timeout (default 10s, per-call override capped at 30s),
  enforced by `p.join(timeout)` then `p.terminate()`
- `resource.setrlimit(RLIMIT_AS, 2GB)` set inside the child
- results returned over a `Queue`, capped at 4000 characters with an
  explicit `"… [truncated, N chars total]"` marker

Verified with the real pattern: `diff(x**3, x)` returns in 0.3s;
`integrate(exp(x**2)*sin(x)/log(x), x)` and `expand((x+1)**200000)` are
both killed cleanly at a 5s boundary.

A subprocess is the only bound that actually holds. Signal-based timeouts
(SIGALRM) interrupt between Python bytecodes and cannot preempt the C-level
loops inside SymPy and gmpy, so a pathological input would still hang the
MCP server.

**Cost, stated plainly: ~0.3s of process-spawn overhead per call.** These
are correctness-critical tools called occasionally, not in a hot loop, so
the trade is worth it — but it is real and should not be discovered later.

## Module

New `src/hermes/symbolic.py`, importing nothing from `db.py`, `rank.py`,
`features.py`, or `feeds.py`. It is a pure function library over strings,
independently testable without a database or an MCP server.

`sympy>=1.13` moves from transitive to a declared dependency in
`pyproject.toml` — relying on another package to keep pulling it in is
fragile.

## Tools

Seven tools, each an `@mcp.tool()` wrapper over an `_impl` function, in the
established `mcp_server.py` pattern (structured dict return, `except
Exception: log.exception(...)` guard, success-path keys preserved on the
error path).

| Tool | Signature | Behavior |
|---|---|---|
| `sym_simplify` | `(expr, timeout=10)` | Canonical simplified form. |
| `sym_solve` | `(expr, symbol=None, timeout=10)` | Solve `expr = 0` for `symbol`; auto-detected when omitted. |
| `sym_differentiate` | `(expr, symbol=None, order=1, timeout=10)` | Derivative of the given order. |
| `sym_integrate` | `(expr, symbol=None, bounds=None, timeout=10)` | Indefinite, or definite when `bounds=[lo, hi]`. |
| `sym_derivation` | `(expr, operation, symbol=None, timeout=10)` | Step-by-step trace (see limits). |
| `sym_verify` | `(lhs, rhs, timeout=10)` | Test whether two expressions are equal (see limits). |
| `sym_evaluate` | `(expr, subs=None, units=None, timeout=10)` | Numeric value; optional unit conversion. |

### Symbol handling

Free symbols are auto-detected from the parsed expression and treated as
real by default. `symbol` overrides which one an operation targets — needed
for `solve("x*y - 1", symbol="y")`, where auto-detection would otherwise
pick `x` by sort order. When auto-detection is ambiguous (more than one
free symbol and no explicit `symbol`), the tool returns `ok: false` naming
the candidates rather than guessing.

Symbols are real by default, not positive: assuming positivity would
silently produce wrong answers for expressions valid over negatives
(`sqrt(x**2)` simplifies to `x` only when `x >= 0`).

### Return shape

Every successful result carries:

- `result` — the plain-text form, round-trippable back into another call
- `latex` — `sp.latex(result)`, verified working
- `parsed_input` — how the expression was actually interpreted, so a
  precedence or implicit-multiplication misparse is visible rather than
  silently wrong
- `numeric` — a float when the result evaluates to one, else `null`

## Honest limits (must appear in the tool descriptions)

**`sym_verify` cannot disprove.** It computes `simplify(lhs - rhs) == 0`.
A zero result is a proof of equality. A non-zero result means **"could not
prove equal"** — not "unequal" — because `simplify` is incomplete.
The tool returns a three-valued `verdict`: `"equal"`, `"unproven"`, or
`"unequal"`, where `"unequal"` is used **only** when a random numeric
substitution produces a clear mismatch. Wording matters: an agent that
reads "unproven" as "false" will report false disproofs.

**`sym_derivation` genuinely traces only integrals.** SymPy's
`integral_steps` (verified present) has no differentiation counterpart.
For `operation="differentiate"`, the tool returns the derivative with each
rule application labeled — useful, but not the same machinery. The
description states the asymmetry rather than implying parity.

## Testing

Pure unit tests in `tests/test_symbolic.py`; no database, no MCP server.

**Security (non-negotiable):**
- `__import__('os').getcwd()`, `open('/etc/passwd').read()`, and an
  attribute-traversal attempt each raise rather than evaluate
- a source-level test asserting `sympify` and bare `eval` appear nowhere in
  `symbolic.py`

**Resource bounds:**
- a known-slow computation is killed at the timeout and returns
  `ok: false` with a timeout message, not a hang
- an oversized result is truncated to 4000 chars with the marker present
- the timeout parameter is clamped to 30s

**Correctness:**
- each tool on a textbook case with a known answer
- auto-detected and explicitly-passed symbols agree on single-symbol input
- ambiguous multi-symbol input returns `ok: false` naming candidates
- `sym_verify` returns `"equal"` for a true identity, `"unproven"` (never
  `"unequal"`) for a hard-but-true one, `"unequal"` for a clear mismatch
- `sym_evaluate` converts units correctly (verified: `5 m/s → 18 km/h`)
- `parsed_input` reflects implicit multiplication (`2x` → `2*x`)

All existing tests must pass untouched — this module adds surface without
altering any existing path.

## Out of scope (YAGNI)

Plotting; matrix decompositions beyond what `simplify` reaches; ODE/PDE
solving; assumption declaration (`positive=True`, `integer=True`) — real
symbols only for now; persisting derivations to the database; wiring
symbolic checks into ingest or ranking; LaTeX *input* parsing
(`parse_latex` needs the optional `antlr4` runtime).

## Sequencing note

The implementation plan should build the safety layer first — parser,
subprocess runner, truncation — with its security and resource tests
green before any tool exists. Every tool depends on that layer, and it is
the part that must not be wrong.
