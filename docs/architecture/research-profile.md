# The `research` Hermes profile

A separate Hermes profile for attestation work, so the tuning does not touch
the default one you chat through.

```bash
research chat            # the profile's own agent
hermes chat              # default, unchanged
```

## What differs from default

| | default | research |
|---|---|---|
| Tool schemas | 85.4 KB / 67 tools | **44.8 KB / 27 tools** |
| `filament` plugin | enabled | **disabled** |
| attestation MCP | one combined server, 45 tools | four restricted servers, `feed` enabled |
| Skills | 72 | 70 |

## Why each

**`filament` disabled.** It is the chat gateway plugin and it carries 40 tools
at 39.6 KB -- 46% of the whole tool budget. It stays enabled in the default
profile because that is what delivers Telegram; a research session does not
need it.

**Four restricted MCP servers, one enabled.** `ATTEST_TOOLS` scopes the server
to one namespace. Measured on gemma4:e2b, a model picked the `ask` router 1
time in 26 when the specific tools were visible beside it and 26 in 26 when
they were not, so each server also hides its specifics behind
`<namespace>.tools` unless `ATTEST_EXPAND=1`. Flip `enabled` in
`~/.hermes/profiles/research/config.yaml` to switch namespace.

**`arxiv` and `blogwatcher` disabled.** Their descriptions -- "Search arXiv
papers" and "Monitor blogs and RSS/Atom feeds" -- collide directly with the
feed tools. With four such skills listed alongside, routing fell from 6/6 to
3/6: "check my rss feeds" went to blogwatcher and "recent arxiv papers on rag"
went to arxiv. Their SKILL.md files are renamed rather than deleted, so
re-enabling is a `mv`.

Only these two were disabled. Per-skill attribution at n=6 was noise --
dropping `arxiv` alone scored *worse* than keeping it -- so the aggregate is
the only trustworthy result, and the other 70 skills cost ~70 bytes each with
no evidence they hurt.

## What was deliberately NOT done

Skills were not mass-disabled. They load as a ~70-byte index entry and the
body only loads on invoke: 68 skills cost 7 KB against 85 KB of tool schemas.
Hiding tools is 12x the lever that hiding skills is.
