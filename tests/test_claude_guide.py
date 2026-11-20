"""docs/guides/claude.md: the commands and the JSON it shows name real entry points.

The guide is the one place a user copies a config from, so it cannot drift from
what the package ships: the console script it spawns must be in
`[project.scripts]`, the surface it restricts to must be in `AGENT_SURFACES`,
the JSON must parse, and every `claude mcp add` line must have the shape the
CLI accepts (name first, `--` before the command, `KEY=value` pairs).
"""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import Path

from attestation.mcp import AGENT_SURFACES

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs" / "guides" / "claude.md").read_text()
SCRIPTS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]


def _fences(lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", GUIDE, re.S)


def _config() -> dict:
    blocks = _fences("json")
    assert len(blocks) == 1, "the guide shows one complete claude_desktop_config.json"
    return json.loads(blocks[0])["mcpServers"]


def test_the_desktop_json_names_real_entry_points():
    servers = _config()
    assert set(servers) == {"attestation", "smeltery"}
    att = servers["attestation"]
    assert att["command"] == "uvx"
    assert att["args"][-1] in SCRIPTS and att["args"][-1] == "attest-mcp"
    assert att["env"]["ATTEST_TOOLS"] in AGENT_SURFACES
    assert Path(att["env"]["ATTEST_DB"]).is_absolute()
    # smeltery's own side is pinned by its repo's test (the `mcp` subcommand and extras).
    sm = servers["smeltery"]
    assert sm["args"][-2:] == ["smeltery", "mcp"]
    assert "smeltery[mcp,attestation]" in " ".join(sm["args"])
    assert "--python" in sm["args"]  # smeltery's floor/ceiling is 3.11-3.12


def test_both_servers_share_the_workspace_the_guide_describes():
    servers = _config()
    root = servers["smeltery"]["env"]["SMELTERY_ROOT"]
    assert servers["attestation"]["env"]["RESEARCH_ROOT"] == f"{root}/.smeltery/screens"


def _add_commands() -> list[list[str]]:
    out = []
    for block in _fences("bash"):
        text = block.replace("\\\n", " ")
        out += [shlex.split(ln) for ln in text.splitlines() if ln.startswith("claude mcp add")]
    return out


def test_every_claude_mcp_add_line_has_the_shape_the_cli_accepts():
    cmds = _add_commands()
    assert {c[3] for c in cmds} == {"attestation", "smeltery"}
    for argv in cmds:
        assert "--" in argv, argv
        before, after = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
        name = argv[3]
        assert not name.startswith("-"), "name must come first: -e would swallow it"
        envs = [before[i + 1] for i, a in enumerate(before) if a == "-e"]
        assert envs and all(re.fullmatch(r"[A-Z_]+=\S+", e) for e in envs), envs
        assert after[0] == "uvx"
    attest = next(c for c in cmds if c[3] == "attestation")
    assert attest[-1] in SCRIPTS


def test_the_one_line_installs_name_the_console_script():
    for text in (GUIDE, (ROOT / "README.md").read_text()):
        assert "attest-mcp" in text
    readme = (ROOT / "README.md").read_text()
    assert "docs/guides/claude.md" in readme
