# Demo recordings

Six short recordings of `attest` in use, one per major surface plus one
pair driving a real agent. Unlike `examples/*/`, these are not golden
paths: they produce video, not a pinned output line, and most need a
running model server, so `tests/test_golden_paths.py` does not run them.
What's committed here is the recording *scripts* — real commands against
real fixtures, the same ones `examples/workspace/` and `examples/citations/`
already use — not the video files themselves. Regenerate locally; see each
subdirectory's own notes for what it needs.

| dir | what it shows | needs |
|---|---|---|
| `ledger/` | `attest runs scan`/`list`/`compare`, `attest claims` — the run ledger and claim checker over `examples/workspace/` | `none — pure local computation` |
| `claims/` | `attest claims` plus `cite.*` over MCP — the citation lint over `examples/citations/` | `none — pure local computation` |
| `kg-symbolic/` | `kg.*` and `sym.*` over MCP — the reading graph and symbolic math, neither of which has a CLI command or web page | `kg.*` needs a model server once, to seed real tags; `sym.*` needs nothing |
| `feed/` | the HTMX web UI (`attest serve`) — browsing a persona's feed, marking items useful/not, opening the onboarding form | a model server, to seed real tagged items |
| `hermes/` | a real Hermes Agent (`hermes chat`) calling `runs.ask` and `feed.ask` over MCP — the only pair driving an agent rather than the tools directly | a live Hermes install, and a model server |

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
first argument. `feed/record.py` always writes to `demo/feed.webm` and
ignores that argument.

## Last verified

All six ran green on 2026-09-22 against `main`, and nothing here needed a
code change to keep working:

| demo | how it was checked |
|---|---|
| `ledger/` | `narrate.sh examples/workspace` — 9 runs, 7 claims, coverage lint |
| `claims/` | `narrate.sh examples/citations` — 4 claims, 1 uncited key, `cite.*` over MCP |
| `kg-symbolic/` | seeded (40 items tagged, 0 failed), then `record.sh` → 2.5 KB cast, 152 KB gif |
| `feed/` | seeded + persona, then `record.py` → 628 KB webm |
| `hermes/` provenance | real agent turn: called `runs_ask`, answered `winner: kdsweep_t4` with both caveats |
| `hermes/` feed | real agent turn: called `feed_ask`, returned 4 ranked items with tags |

Two notes for whoever runs these next, both of which cost time to rediscover:

- **`kg-symbolic/demo.py` takes its database from `ATTEST_DB`, not from a
  positional argument.** Passing a path as an argument is silently ignored and
  the demo runs against whatever `resolve_db_path` finds — for the author that
  was the live 12k-item database, whose `kg.communities` output is an
  alphabetical sprawl rather than the four coherent clusters the seeded
  fixture gives.
- **The MCP server's INFO logs go to stderr**, so running `demo.py` by hand
  looks noisy (28 of 91 lines). `record.sh` already redirects them; the
  recording was never polluted.

`hermes/record-feed.sh` reads the `attestation-feed` MCP server from
`~/.hermes/config.yaml`. With no `ATTEST_DB` set there it demos against the
author's live database rather than the seeded fixture, which works but is not
what `feed/seed_feed_db.py` was written for.
