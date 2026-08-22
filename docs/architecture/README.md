# Architecture diagrams

`agent-flows.html` — the two paths through the system, and the feedback loop
between them that does not close.

Three figures:

1. **A question from the researcher.** Hermes picks one tool from four
   namespaces; `mcp/_tool.py` handles the connection, persona lookup and
   response envelope once rather than 37 times; only search and tagging reach
   Ollama.
2. **The nightly refresh.** `flock`, then ingest (must succeed, no model
   needed), then tag (best-effort, a cold Ollama is a degraded run). The
   asymmetry is the design.
3. **The starved signal.** 70 stated opinions across 5,167 items, all
   positive, so the click classifier never fired for a real account. The
   figure now also records what happened when the two synthetic channels were
   run on `matt`: 128 rows, classifier active, and an `evaluate_user` score of
   1.0 that turned out to be separable by feed rather than by topic.

Numbers come from the live database and the tool registry, not from
estimates. If either changes materially, redraw rather than letting the
figures drift -- a diagram nobody trusts is worse than no diagram.
