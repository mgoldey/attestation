# Claims + citations demo recording

## What you get

An asciinema recording of `examples/citations/`'s own commands: scan the
workspace fixture, check a draft's numeric claims and citation keys from
the CLI, then the same citation tools (`cite.sources`/`cite.check`/
`cite.lookup`/`cite.search`) driven over MCP by `check_citations.py`.

Note: `cite.check`'s output names the claim file by the path it was given,
which for `check_citations.py` is an absolute path under this machine's
home directory — expected on screen for a few seconds during recording,
not a bug in the demo.

## Run it

```bash
uv tool install asciinema         # once
cargo install --locked --git https://github.com/asciinema/agg   # once
./record.sh
```

Writes `../../demo/claims.cast` and `.gif` (gitignored, not committed).
