"""Generating agent configs from AGENT_SURFACES, and detecting drift from them.

The problem this closes was live when it was written. `~/.hermes/config.yaml`
held five attestation entries -- the full server plus four `attestation-<surface>`
entries setting ATTEST_TOOLS -- and `install.py` wrote exactly one of them. The
other four were typed by hand. Nothing in the repo knew they existed, and
`step_mcp_wiring` returned ok on seeing the substring "attestation" in
`hermes mcp list`, so a config with the full server and zero surfaces reported
clean. That is the same false-clean the scheduler check had before it learned
to look for a second entry.
"""

import pytest

from attestation import emit
from attestation.mcp import AGENT_SURFACES


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='attestation'\n")
    return tmp_path


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------


def test_every_surface_gets_an_entry(root):
    servers = emit.hermes_servers(root)
    assert set(servers) == {f"attestation-{name}" for name in AGENT_SURFACES}


def test_entries_carry_their_surface_and_this_checkout(root):
    servers = emit.hermes_servers(root)
    entry = servers["attestation-feed"]
    assert entry["env"]["ATTEST_TOOLS"] == "feed"
    assert str(root) in entry["args"]
    assert entry["command"] == "uv"


def test_surfaces_are_disabled_by_default(root):
    """A surface is chosen at launch by a person -- that is the finding the
    whole split rests on. Five servers all enabled would put every tool back
    into one session and undo it."""
    for entry in emit.hermes_servers(root).values():
        assert entry["enabled"] is False


def test_adding_a_surface_needs_no_edit_here(root, monkeypatch):
    """The point of generating: a new surface appears without hand editing."""
    monkeypatch.setitem(
        AGENT_SURFACES,
        "citations",
        emit.Surface(prefixes=frozenset({"cite"}), summary="s", rationale="r"),
    )
    assert "attestation-citations" in emit.hermes_servers(root)


def test_claude_agent_files_come_from_the_same_table(root):
    """One schema, two consumers, so the definitions cannot drift."""
    agents = emit.claude_agents(root)
    assert set(agents) == {name for name in AGENT_SURFACES}
    body = agents["feed"]
    assert body.startswith("---")
    assert AGENT_SURFACES["feed"].summary in body


def test_both_consumers_describe_the_same_surfaces(root):
    assert set(emit.claude_agents(root)) == {
        n.removeprefix("attestation-") for n in emit.hermes_servers(root)
    }


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def test_a_config_with_no_surfaces_is_not_clean(root):
    """The false-clean this exists to end: the full server present, every
    surface missing, and a substring match calling it ok."""
    findings = emit.check_hermes({"attestation": {"command": "uv"}}, root)
    assert findings, "a config with zero surfaces reported no problem"
    assert all(f.kind == "missing" for f in findings), findings


def test_a_complete_config_is_clean(root):
    assert emit.check_hermes(emit.hermes_servers(root), root) == []


def test_an_entry_pointing_at_another_checkout_is_stale(root, tmp_path):
    servers = emit.hermes_servers(root)
    servers["attestation-feed"]["args"] = ["run", "--project", "/somewhere/else", "attest-mcp"]

    kinds = {f.kind for f in emit.check_hermes(servers, root)}
    assert kinds == {"stale"}


def test_an_entry_bound_to_the_wrong_surface_is_stale(root):
    """The check must compare ATTEST_TOOLS, not only the checkout path.

    That variable IS the mechanism this whole feature exists to keep correct:
    it is what restricts the tool surface, and a plausible-but-wrong binding --
    attestation-feed serving `knowledge` after a copy-paste -- is precisely the
    drift the emitter is for. Comparing only `args` reports it clean, because
    the path is right.
    """
    servers = emit.hermes_servers(root)
    servers["attestation-feed"]["env"]["ATTEST_TOOLS"] = "knowledge"

    findings = emit.check_hermes(servers, root)
    assert [f.kind for f in findings] == ["stale"], findings
    assert "ATTEST_TOOLS" in findings[0].detail


def test_an_entry_with_no_env_at_all_is_stale(root):
    """A surface entry that never got its ATTEST_TOOLS set serves all 50 tools
    under a name promising four -- worse than a missing entry, because it looks
    present."""
    servers = emit.hermes_servers(root)
    del servers["attestation-feed"]["env"]

    assert [f.kind for f in emit.check_hermes(servers, root)] == ["stale"]


def test_an_entry_for_a_surface_that_no_longer_exists_is_orphaned(root):
    """The one a substring check can never find, and the reason this is a
    generator rather than a bigger `in` test. Same shape as the duplicate
    crontab entry: a hand-added thing that used to be right, invisible to the
    tool that owns the domain."""
    servers = emit.hermes_servers(root)
    servers["attestation-citations"] = {
        "command": "uv",
        "args": ["run", "--project", str(root), "attest-mcp"],
        "env": {"ATTEST_TOOLS": "citations"},
        "enabled": False,
    }

    findings = emit.check_hermes(servers, root)
    assert [f.kind for f in findings] == ["orphaned"]
    assert "citations" in findings[0].detail


def test_renaming_a_surface_reports_both_halves(root, monkeypatch):
    """A rename is an orphan plus a missing, and reporting only one of them
    leaves the user with half a repair."""
    before = emit.hermes_servers(root)
    monkeypatch.delitem(AGENT_SURFACES, "symbolic")
    monkeypatch.setitem(
        AGENT_SURFACES,
        "algebra",
        emit.Surface(prefixes=frozenset({"sym"}), summary="s", rationale="r"),
    )

    kinds = sorted(f.kind for f in emit.check_hermes(before, root))
    assert kinds == ["missing", "orphaned"]


def test_the_full_server_entry_is_left_alone(root):
    """`attestation` is written by install.py and is not a surface. Reporting
    it as orphaned would tell the user to delete the working server."""
    servers = emit.hermes_servers(root)
    servers["attestation"] = {"command": "uv", "args": ["run", "attest-mcp"]}
    assert emit.check_hermes(servers, root) == []


def test_unrelated_servers_are_left_alone(root):
    servers = emit.hermes_servers(root)
    servers["filament"] = {"command": "node", "args": ["x.js"]}
    assert emit.check_hermes(servers, root) == []


# --------------------------------------------------------------------------
# Reading what the agent actually has
# --------------------------------------------------------------------------

CONFIG_DUMP = """attestation:
  args:
  - run
  - --project
  - /home/matt/attestation
  - attest-mcp
  command: uv
  enabled: true
attestation-feed:
  args:
  - run
  - --project
  - /home/matt/attestation
  - attest-mcp
  command: uv
  enabled: false
  env:
    ATTEST_TOOLS: feed
"""


def test_parses_the_agents_own_config_dump():
    """`hermes config get mcp_servers` is the only structured view of these
    entries -- `hermes mcp list` truncates args to a fixed column width, which
    would make every entry look stale."""
    servers = emit.parse_config_dump(CONFIG_DUMP)
    assert set(servers) == {"attestation", "attestation-feed"}
    assert servers["attestation-feed"]["args"] == [
        "run",
        "--project",
        "/home/matt/attestation",
        "attest-mcp",
    ]
    assert servers["attestation-feed"]["env"] == {"ATTEST_TOOLS": "feed"}
    assert servers["attestation-feed"]["enabled"] is False
    assert servers["attestation"]["enabled"] is True


def test_an_empty_dump_is_no_servers_not_a_crash():
    """`hermes config get` on an unset key prints nothing. That is a config
    with no MCP servers, which is a finding, not an error."""
    assert emit.parse_config_dump("") == {}
    assert emit.parse_config_dump("null\n") == {}


def test_round_trips_what_this_module_generates(root):
    """The parser and the generator must agree, or the check compares two
    different shapes and reports drift that is not there."""
    import io

    servers = emit.hermes_servers(root)
    rendered = io.StringIO()
    for name, entry in servers.items():
        rendered.write(f"{name}:\n  args:\n")
        for a in entry["args"]:
            rendered.write(f"  - {a}\n")
        rendered.write(f"  command: {entry['command']}\n")
        rendered.write(f"  enabled: {str(entry['enabled']).lower()}\n")
        rendered.write(f"  env:\n    ATTEST_TOOLS: {entry['env']['ATTEST_TOOLS']}\n")

    assert emit.check_hermes(emit.parse_config_dump(rendered.getvalue()), root) == []


def test_install_and_emit_share_one_generator(root, monkeypatch):
    """Two callers of one generator is exactly where this class of bug returns.

    The failure being fixed is two copies of one fact with no check between
    them. Reintroducing it inside its own fix would be an embarrassing way to
    lose the property, so this asserts that the doctor's verdict is computed
    from `emit`, not from a parallel implementation.
    """
    from attestation import install

    calls = []
    real = emit.check_hermes

    def spy(servers, checkout_root):
        calls.append(checkout_root)
        return real(servers, checkout_root)

    monkeypatch.setattr(emit, "check_hermes", spy)
    monkeypatch.setattr(install, "_checkout_root", lambda: root)
    monkeypatch.setattr(
        install,
        "_run",
        lambda cmd, **kw: __import__("types").SimpleNamespace(
            returncode=0, stdout="attestation", stderr=""
        ),
    )

    install.step_mcp_wiring("hermes", check=True)
    assert calls == [root], "the doctor did not go through emit.check_hermes"


# --------------------------------------------------------------------------
# The CLI, which is where "never overwrite" is actually enforced
# --------------------------------------------------------------------------


def _emit(tmp_path, monkeypatch, write):
    import argparse

    from attestation import cli, install

    monkeypatch.setattr(install, "_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(install, "_find_agent_binary", lambda: None)
    return cli.cmd_emit(argparse.Namespace(write=write))


def test_write_creates_missing_agent_files(tmp_path, monkeypatch, capsys):
    assert _emit(tmp_path, monkeypatch, write=True) == 0
    assert (tmp_path / ".claude" / "agents" / "attestation-feed.md").is_file()


def test_write_refuses_to_clobber_a_hand_edited_file(tmp_path, monkeypatch, capsys):
    """The spec's success criterion, and the reason this is a generator rather
    than a formatter: "a hand-edited emitted file is reported, never
    overwritten". A checksum would turn a deliberate edit into a warning the
    user cannot act on; silently rewriting it is worse.

    The realistic path is not someone running --write on a file they just
    edited. It is someone adding a fifth surface, running --write to get it,
    and losing an unrelated edit they made weeks ago as a side effect.
    """
    _emit(tmp_path, monkeypatch, write=True)
    edited = tmp_path / ".claude" / "agents" / "attestation-feed.md"
    edited.write_text(edited.read_text() + "\nMY HAND EDIT\n")

    rc = _emit(tmp_path, monkeypatch, write=True)

    assert "MY HAND EDIT" in edited.read_text(), "a hand edit was silently overwritten"
    assert rc != 0, "clobbering was refused but the exit code said success"
    assert "attestation-feed" in capsys.readouterr().out


def test_write_still_updates_the_files_it_did_not_touch(tmp_path, monkeypatch):
    """One edited file must not block the rest: a user who customised one
    agent still needs the other three regenerated when a surface changes."""
    _emit(tmp_path, monkeypatch, write=True)
    agents = tmp_path / ".claude" / "agents"
    (agents / "attestation-feed.md").write_text("HAND EDITED\n")
    (agents / "attestation-symbolic.md").unlink()

    _emit(tmp_path, monkeypatch, write=True)

    assert (agents / "attestation-feed.md").read_text() == "HAND EDITED\n"
    assert (agents / "attestation-symbolic.md").is_file()


def test_rewriting_an_unmodified_file_is_not_an_error(tmp_path, monkeypatch):
    """Only a DIFFERENCE is protected. Re-running --write on files that match
    what would be generated is a no-op, not a refusal."""
    _emit(tmp_path, monkeypatch, write=True)
    assert _emit(tmp_path, monkeypatch, write=True) == 0


def test_the_generated_header_names_a_real_command(root):
    """The header stamped into every agent file told the reader to run a flag
    that does not exist.

    `attest emit --check` → "unrecognized arguments: --check", exit 2. The real
    flag set is [--write]; bare `attest emit` does the reporting the header
    describes. Self-inflicted: emit.py exists specifically so two copies of one
    fact cannot drift, and its own header drifted from its own CLI.
    """
    from attestation.cli import build_parser

    emit_parser = build_parser()._subparsers._group_actions[0].choices["emit"]  # type: ignore[union-attr]
    flags = {opt for action in emit_parser._actions for opt in action.option_strings}

    import re

    for body in emit.claude_agents(root).values():
        # Only tokens shaped like a real long flag: `--word` or `--two-words`.
        # The frontmatter fence and the HTML comment close are not flags.
        for flag in set(re.findall(r"--[a-z][a-z-]+\b", body)):
            assert flag in flags, f"the generated header names {flag!r}, not in {sorted(flags)}"
