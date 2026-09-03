# Demo recordings

Four short recordings of `attest` in use, one per major surface. Unlike
`examples/*/`, these are not golden paths: they produce video, not a pinned
output line, and two of the four need a running model server, so
`tests/test_golden_paths.py` does not run them. What's committed here is the
recording *scripts* — real commands against real fixtures, the same ones
`examples/workspace/` and `examples/citations/` already use — not the
video files themselves. Regenerate locally; see each subdirectory's own
notes for what it needs.

| dir | what it shows | needs |
|---|---|---|
| `ledger/` | `attest runs scan`/`list`/`compare`, `attest claims` — the run ledger and claim checker over `examples/workspace/` | `none — pure local computation` |
| `claims/` | `attest claims` plus `cite.*` over MCP — the citation lint over `examples/citations/` | `none — pure local computation` |
| `kg-symbolic/` | `kg.*` and `sym.*` over MCP — the reading graph and symbolic math, neither of which has a CLI command or web page | `kg.*` needs a model server once, to seed real tags; `sym.*` needs nothing |
| `feed/` | the HTMX web UI (`attest serve`) — browsing a persona's feed, marking items useful/not, opening the onboarding form | a model server, to seed real tagged items |

## Recording pipeline

Terminal recordings (`ledger/`, `claims/`, `kg-symbolic/`) use
[asciinema](https://asciinema.org) to capture the real session and
[agg](https://github.com/asciinema/agg) to convert it to a GIF:

```bash
uv tool install asciinema
cargo install --locked --git https://github.com/asciinema/agg
```

The browser recording (`feed/`) uses Playwright, in its own dependency
group so it is never installed by a plain `uv sync`:

```bash
uv sync --group demos
uv run --group demos playwright install chromium
```

Each `record.sh`/`record.py` writes its output to `../../demo/` (the
repo-root `demo/` directory, already gitignored) unless given a path as its
first argument.
