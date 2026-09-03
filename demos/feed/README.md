# Feed web UI demo recording

## What you get

A Playwright recording of `attest serve`'s real HTMX page: a persona's
ranked feed, marking one item useful and one not useful (watch the ranking
and the caveat line change), then opening the onboarding form via
"+ new reader".

Like `kg-symbolic/`, this needs real tagged items — `seed_feed_db.py`
shares the same ingest+tag path (`demos/kg-symbolic/seed_kg_db.py`)
against the real chat model, then creates one persona, `demo-reader`.

## Run it

```bash
uv sync --group demos                              # once
uv run --group demos playwright install chromium    # once

ollama serve   # or whatever LLM_BASE_URL points at
uv run python seed_feed_db.py /tmp/demo-feed.db     # ~2 min, tags 40 items for real

ATTEST_DB=/tmp/demo-feed.db uv run --group demos python record.py
```

Writes `../../demo/feed.webm` (gitignored, not committed).
`record.py` starts `attest serve` as a subprocess against `ATTEST_DB`,
drives it headless, and stops the server when done.
