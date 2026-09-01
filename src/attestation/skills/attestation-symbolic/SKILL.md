---
name: attestation-symbolic
description: "Derive, simplify, solve, differentiate, integrate and numerically evaluate expressions exactly, and test two expressions for symbolic equality, so a derivation in a paper is checked rather than eyeballed. SymPy in a sandboxed subprocess; touches no database and no network."
version: 2.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [math, sympy, derivation, algebra, calculus]
    related_skills: [attestation-setup, attestation-provenance]
---

# attestation: symbolic math

Use this when the reader wants algebra or calculus done exactly -- a step in
an appendix checked, an expression simplified, an equation solved, a value
with units -- rather than approximated in prose.

## When NOT to use this

- Numerical simulation, plotting, or anything needing data: this is exact
  symbolic computation on expressions the reader gives you.
- Checking a *number* in a draft against an experiment: the provenance
  agent (`attestation-provenance`).

## Ask the router first

```
sym.ask(expr="x**2 - 4", question="solve")
sym.ask(expr="sin(x)**2 + cos(x)**2", question="simplify")
```

Returns `answer` (relay VERBATIM), `refs`, `caveat`, `options` and
`tool_used`; `ok=false` with `options` means ask the reader which operation
they meant. Specific tools may be hidden from your session; `sym.tools`
explains why and how to reveal them.

## The tools

`^` and `2x` both parse. Every call takes a `timeout`.

- `sym.simplify(expr)` -- canonical form.
- `sym.solve(expr, symbol)` -- solve `expr = 0`; the symbol is auto-detected
  when the expression has exactly one.
- `sym.differentiate(expr, symbol, order)` and `sym.integrate(expr, symbol,
  bounds)` -- indefinitely or over bounds.
- `sym.derivation(expr, operation, symbol)` -- a step-by-step trace.
  **Genuine rule-by-rule tracing exists only for integrals**; the
  differentiate branch returns the result with a note saying so. Do not
  present that note's result as a traced derivation.
- `sym.verify(lhs, rhs)` -- `equal`, `unequal` or `unproven`.
  **`unproven` is NOT a disproof.** Simplification is incomplete, so
  `unproven` means only "could not decide"; say exactly that.
- `sym.evaluate(expr, subs, units)` -- a numeric value with substitutions
  and unit conversion (`units="meter/second -> kilometer/hour"`).
  `numeric` is `null` whenever any symbol is still free, and the message
  names which: an unsubstituted expression has no value, and reporting one
  anyway is how a wrong number reaches a paper.

## Checking a derivation

Take the paper's steps one at a time: `sym.verify(step_n, step_n_plus_1)`
for each consecutive pair. An `unequal` names the step that breaks; an
`unproven` names a step you cannot vouch for, which is a different report.
Show the reader the chain of verdicts, not a summary of it.

## Limits

Input is never `eval`ed: expressions are parsed against a whitelist of
mathematical names with builtins removed, and every computation runs in a
subprocess with a wall-clock timeout and a 2 GB memory cap. A malicious
expression is refused; a runaway one is cancelled rather than taking the
server down, at the cost of roughly 0.3 s of process spawn per call. A
timeout on a hard integral is a limit, not a wrong answer -- raise
`timeout` or simplify the input.

## Mistakes that look reasonable

| Instead of | Do |
|---|---|
| Reading `unproven` as "not equal" | Say it could not be decided |
| Reporting a `numeric` for an expression with a free symbol | Substitute, or say which symbol is free |
| Presenting the differentiate branch as a traced derivation | Say the trace is genuine only for integrals |
| Retrying a timed-out call unchanged | Raise `timeout` or simplify |
| Concluding a missing tool is broken | It is hidden (`sym.tools`) or the server is stale (`attestation-setup`) |
