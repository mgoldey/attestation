# Knowledge graph + symbolic math demo recording

## What you get

An asciinema recording of `demo.py`, which drives `kg.*` and `sym.*` over
MCP: the concept vocabulary, this reader's most central topics, direct
neighbours of one concept, a connected path between two topics and a real
"no path" answer between two that never co-occur, the reader's research
communities, then seven `sym.*` calls (simplify, solve, differentiate,
integrate with bounds, a rule-by-rule integration derivation, an identity
verification, a units conversion).

`kg.*` reads a real graph, which needs real tagged items — there is no
committed fixture for this one, because a live model's tags are the whole
point (the offline stub's placeholder tags produce a graph with nodes
named "existing"/"vocabulary"/"title", not worth recording). `sym.*` needs
nothing; it is pure SymPy in a sandboxed subprocess.

## Run it

```bash
uv tool install asciinema         # once
cargo install --locked --git https://github.com/asciinema/agg   # once

ollama serve   # or whatever LLM_BASE_URL points at
uv run python seed_kg_db.py /tmp/demo-kg.db     # ~2 min, tags 40 items for real

ATTEST_DB=/tmp/demo-kg.db ./record.sh
```

Writes `../../demo/kg-symbolic.cast` and `.gif` (gitignored, not
committed).
