# Ledger demo recording

## What you get

An asciinema recording of `examples/workspace/`'s own commands: scan a
two-project workspace into the ledger, list what's there, rank a four-arm
sweep with its caveats, then check a Markdown findings doc's claims against
it — one contradicted on purpose.

## Run it

```bash
uv tool install asciinema         # once
cargo install --locked --git https://github.com/asciinema/agg   # once
./record.sh
```

Writes `../../demo/ledger.cast` and `.gif` (gitignored, not committed).
`narrate.sh` is the script under recording — it runs
`examples/workspace/README.md`'s commands one at a time with an echoed
prompt between them, so the recording reads as a walkthrough.
