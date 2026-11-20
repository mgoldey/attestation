# Wire attestation and smeltery into Claude

How do I give Claude Code or Claude Desktop both servers, so it can run a
smeltery screen and then check the result with attestation? Register two stdio
MCP servers, one per package, each spawned by `uvx` (or `pipx`) with nothing
cloned, and point both at one workspace so attestation reads the runs smeltery
writes.

Checked on 2026-11-20: every command below was run, except that the Claude
Desktop file location is quoted from the MCP quickstart (not reachable from
the session that wrote this) and the example conversation is illustrative,
not a recording. Claude Code's syntax was checked against `claude mcp add` in
Claude Code 2.1.329 and the
[Claude Code MCP docs](https://code.claude.com/docs/en/mcp).

## What is on PyPI today, and what is not

| Package | Latest on PyPI | What it lacks |
|---|---|---|
| `attestation` | 0.5.0 | `runs.status` and the core-record reader: it cannot read what smeltery writes. Both land in 0.5.1 (in this repo, not yet published). |
| `smeltery` | 0.1.0 | `smeltery mcp` and the `mcp` extra: the MCP server is not in the published wheel. |

So today both servers install from git. The commands below use the git form;
once the releases above are on PyPI, swap `git+https://github.com/mgoldey/attestation`
for `attestation` and `smeltery[mcp,attestation] @ git+https://github.com/mgoldey/smeltery`
for `smeltery[mcp,attestation]`. Check with `uvx pip index versions attestation`
(or `pip index versions`) before relying on either.

Each server runs in its **own** environment, which is the point of one `uvx`
per server: attestation pins `mcp==1.28.x` and smeltery needs `mcp>=2.3`, so
they cannot share one.

## One-line install, per server

```bash
# attestation: the MCP server (console script `attest-mcp`)
uvx --from git+https://github.com/mgoldey/attestation attest-mcp
pipx run --spec git+https://github.com/mgoldey/attestation attest-mcp

# smeltery: `smeltery mcp`, needing the `mcp` extra; `attestation` adds the
# core run record that attestation reads. smeltery supports Python 3.11 and
# 3.12 only, so say so: uv otherwise may pick 3.13 and fail to resolve.
uvx --python 3.12 --from 'smeltery[mcp,attestation] @ git+https://github.com/mgoldey/smeltery' smeltery mcp
pipx run --python python3.12 --spec 'smeltery[mcp,attestation] @ git+https://github.com/mgoldey/smeltery' smeltery mcp
```

A server started by hand waits on stdin for an MCP client; that is the
expected behavior, and Ctrl-D ends it. `smeltery mcp --list-tools` prints the
tool names and exits, which is the quickest check that the install works.

## Point both at one workspace

smeltery writes each screen to a run directory; attestation scans a directory
for run records. They meet on the filesystem, so three settings matter:

| Variable | Server | Value | Why |
|---|---|---|---|
| `SMELTERY_ROOT` | smeltery | `/data/smeltery` | Workspace holding the job queue and screens. |
| `RESEARCH_ROOT` | attestation | `/data/smeltery/.smeltery/screens` | Where `runs.scan` and `runs.status` look. |
| `ATTEST_DB` | attestation | `/data/attestation/ledger.db` | An absolute path. Unset, the database is `./hermes.db` in whatever directory the client spawned the server from. |

`RESEARCH_ROOT` is `SMELTERY_ROOT` plus `.smeltery/screens`, not
`SMELTERY_ROOT` itself: a screen with no `out_dir` lands in
`ROOT/.smeltery/screens/<campaign>-<8 hex>`, and the attestation scanner skips
hidden directories below its root, so `RESEARCH_ROOT=/data/smeltery` sees
nothing (measured: `0 in flight` there, the submitted screen listed one level
down). Each screen directory is then a project of its own. If you would rather
keep the root at `SMELTERY_ROOT`, pass a visible `out_dir` to
`screen_candidates`.

smeltery's guardrails (`SMELTERY_MAX_CPU_HOURS`, `SMELTERY_MAX_CANDIDATES`)
belong on the same entry; see smeltery's `docs/mcp.md`.

## Claude Code

Put the name first and `--` before the command: `-e` takes several values, so
a name written after it is read as another `KEY=value` and rejected, and
without `--` Claude Code would parse the server's own flags as its own.

```bash
claude mcp add attestation -s user \
  -e ATTEST_TOOLS=provenance -e ATTEST_EXPAND=1 \
  -e ATTEST_DB=/data/attestation/ledger.db \
  -e RESEARCH_ROOT=/data/smeltery/.smeltery/screens \
  -- uvx --from git+https://github.com/mgoldey/attestation attest-mcp

claude mcp add smeltery -s user \
  -e SMELTERY_ROOT=/data/smeltery \
  -- uvx --python 3.12 --from 'smeltery[mcp,attestation] @ git+https://github.com/mgoldey/smeltery' smeltery mcp
```

`-s user` makes them available in every project; the default `local` scope is
this project for you only, and `project` writes a `.mcp.json` the team shares
(Claude Code asks each person to approve it). `claude mcp list` shows both and
`claude mcp get attestation` its configuration.

`ATTEST_TOOLS=provenance` serves only the provenance surface (`runs.*`,
`cite.check`, `cite.support`), which is the part that reads smeltery's runs;
a typo in the surface name raises rather than serving everything.
`ATTEST_EXPAND=1` shows that surface's tools from the start: without it a
surface shows `runs.ask` plus `runs.tools`, and the rest appear on request,
which is progressive disclosure for small models, not what Claude needs. Leave
both unset to serve everything, including the feed, which needs the model
server from the [install guide](install.md).

## Claude Desktop

Claude Desktop reads `claude_desktop_config.json` (macOS
`~/Library/Application Support/Claude/claude_desktop_config.json`, Windows
`%APPDATA%\Claude\claude_desktop_config.json`; open it from Settings,
Developer, Edit Config). The `mcpServers` entries have the same
`command`/`args`/`env` shape Claude Code writes for a stdio server. Restart
Claude Desktop after editing.

```json
{
  "mcpServers": {
    "attestation": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mgoldey/attestation",
        "attest-mcp"
      ],
      "env": {
        "ATTEST_TOOLS": "provenance",
        "ATTEST_EXPAND": "1",
        "ATTEST_DB": "/data/attestation/ledger.db",
        "RESEARCH_ROOT": "/data/smeltery/.smeltery/screens"
      }
    },
    "smeltery": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "smeltery[mcp,attestation] @ git+https://github.com/mgoldey/smeltery",
        "smeltery",
        "mcp"
      ],
      "env": {
        "SMELTERY_ROOT": "/data/smeltery"
      }
    }
  }
}
```

A desktop app often starts with a shorter `PATH` than your shell, so if the
server fails to start with "command not found", replace `"uvx"` with the
absolute path from `which uvx` (typically `~/.local/bin/uvx`). Windows paths
need doubled backslashes in JSON. (Practical advice, not checked here.)

## What a session looks like

Illustrative: the tool names and arguments are real, the transcript is
condensed and the numbers are made up.

```text
You:    Screen these three analogues against pocket.pdb, and tell me the cost first.
Claude: estimate_cost(target="pocket.pdb", ligand_set={...})            [smeltery]
        -> 3 candidates, 36 calculations, about 2.4 CPU-hours
        screen_candidates(...)  -> handle ".../.smeltery/screens/screen-pocket-1f0a9c3d"

You:    Is it running?
Claude: runs.status()                                                   [attestation]
        -> 1 in flight (1 running): screen-pocket-1f0a9c3d, local:<job id>
        run_status(handle)                                              [smeltery]
        -> running; measurement 14 of 36 SCF legs

You:    It finished. What did it find, and can I trust the records?
Claude: runs.scan(confirm=true)                                         [attestation]
        runs.list(), runs.detail(...)
        -> the screen's metrics, its dependencies, and provenance caveats
           (a dirty or inferred code or dependency identity is listed, never
           turned into a verdict)
        explain_drop("analogue-2", handle)                              [smeltery]
        -> why that molecule was dropped, from the run's own record
```

`runs.status` reads each run's `attestation/state.json` and lists runs that
have no record yet; it is evidence (`stale` means nothing was written for a
while, not that the job died). `runs.scan` replaces the scanned projects' rows,
so it asks for `confirm=true`. `runs.compare` ranks the arms of a family and
needs two or more; smeltery emits arm-level metrics only for a run whose
submitter named an `arm` (smeltery's `docs/core-record.md`), and the MCP
`screen_candidates` has no `arm` argument, so for a plain screen read it with
`runs.detail` and let smeltery's `explain_drop` say why a molecule dropped.

smeltery's side of this (the four tools, the guardrails, the refusals) is in
[smeltery's `docs/mcp.md`](https://github.com/mgoldey/smeltery/blob/main/docs/mcp.md).
