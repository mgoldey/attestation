from argparse import Namespace

from attestation.cli import build_parser, cmd_bootstrap_persona, cmd_eval, main
from attestation.db import get_db


def test_parser_subcommands():
    parser = build_parser()
    for argv in (
        ["ingest"],
        ["serve", "--port", "9000"],
        ["eval", "--user", "matt"],
        ["warmup"],
        ["bootstrap-persona", "bench-chemist", "-k", "10"],
        ["kg-report"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_eval_insufficient_data_message(tmp_path, capsys):
    db = tmp_path / "t.db"
    get_db(db).close()
    rc = main(["eval", "--db", str(db), "--user", "matt"])
    assert rc == 0
    assert "insufficient" in capsys.readouterr().out.lower()


def test_parser_tag_subcommand():
    args = build_parser().parse_args(["tag", "--limit", "5"])
    assert args.command == "tag"
    assert args.limit == 5


def test_parser_install_subcommand_flags():
    args = build_parser().parse_args(["install", "--check", "--yes", "--now"])
    assert args.command == "install"
    assert args.check is True
    assert args.yes is True
    assert args.now is True


def test_parser_install_subcommand_defaults():
    args = build_parser().parse_args(["install"])
    assert args.command == "install"
    assert args.check is False
    assert args.yes is False
    assert args.now is False


def test_install_command_dispatches_to_run_install(monkeypatch):
    import attestation.install

    captured = {}

    def fake_run_install(check=False, yes=False, now=False):
        captured["check"] = check
        captured["yes"] = yes
        captured["now"] = now
        return 0

    monkeypatch.setattr(attestation.install, "run_install", fake_run_install)

    rc = main(["install", "--check", "--yes"])

    assert rc == 0
    assert captured == {"check": True, "yes": True, "now": False}


def test_install_command_returns_nonzero_exit_from_run_install(monkeypatch):
    import attestation.install

    monkeypatch.setattr(
        attestation.install, "run_install", lambda check=False, yes=False, now=False: 1
    )

    assert main(["install"]) == 1


def test_tag_command_prints_stats(tmp_path, capsys, monkeypatch):
    import attestation.features

    db = tmp_path / "t.db"
    get_db(db).close()
    monkeypatch.setattr(
        attestation.features, "run_tagging", lambda conn, limit=None: {"tagged": 0, "failed": 0}
    )
    rc = main(["tag", "--db", str(db)])
    assert rc == 0
    assert "tagged" in capsys.readouterr().out


def test_tag_command_exit_1_on_total_failure(tmp_path, monkeypatch):
    import attestation.features

    db = tmp_path / "t.db"
    get_db(db).close()
    monkeypatch.setattr(
        attestation.features, "run_tagging", lambda conn, limit=None: {"tagged": 0, "failed": 3}
    )
    assert main(["tag", "--db", str(db)]) == 1


def test_warmup_skips_gracefully_on_non_ollama_backend(capsys, monkeypatch):
    import httpx as _httpx

    import attestation.cli

    def boom(*args, **kwargs):
        raise _httpx.ConnectError("no ollama here")

    monkeypatch.setattr(attestation.cli.httpx, "post", boom)
    rc = main(["warmup"])
    assert rc == 0
    assert "skipping" in capsys.readouterr().out.lower()


def test_main_loads_dotenv(tmp_path, monkeypatch, capsys):
    """main() must call llm.load_env() before dispatch: a .env in cwd is visible."""
    import attestation.llm

    (tmp_path / ".env").write_text("CHAT_MODEL=dotenv-model\n")
    monkeypatch.setattr(attestation.llm, "_REPO_ROOT", tmp_path)  # hermetic vs real checkout .env
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    captured = {}

    import attestation.cli

    def fake_warmup():
        from attestation.llm import chat_model

        captured["model"] = chat_model()

    monkeypatch.setattr(attestation.cli, "warmup", fake_warmup)
    assert main(["warmup"]) == 0
    assert captured["model"] == "dotenv-model"


def test_kg_report_runs_on_an_empty_database(tmp_path, capsys):
    """The report must work on a fresh database -- no tags, no graph, no
    division by zero -- since that is when someone first runs it."""
    db = tmp_path / "t.db"
    get_db(db).close()

    rc = main(["kg-report", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "nodes" in out and "singleton_rate" in out


def test_parser_runs_subcommands():
    parser = build_parser()
    for argv, expected in (
        (["runs", "scan"], "scan"),
        (["runs", "list", "--project", "p"], "list"),
        (["runs", "compare", "fam", "--metric", "wer"], "compare"),
        (["runs", "show", "p", "n"], "show"),
    ):
        args = parser.parse_args(argv)
        assert args.command == "runs"
        assert args.runs_command == expected


def test_runs_scan_without_a_root_explains_rather_than_crashing(tmp_path, monkeypatch, capsys):
    """No default workspace is guessed: scanning a directory the user did not
    mean is worse than saying which variable to set."""
    monkeypatch.delenv("RESEARCH_ROOT", raising=False)
    db = tmp_path / "t.db"
    get_db(db).close()

    rc = main(["runs", "--db", str(db), "scan"])

    assert rc == 1
    assert "RESEARCH_ROOT" in capsys.readouterr().out


def test_runs_list_before_scan_directs_the_user(tmp_path, capsys):
    db = tmp_path / "t.db"
    get_db(db).close()

    rc = main(["runs", "--db", str(db), "list"])

    assert rc == 1
    assert "scan" in capsys.readouterr().out


def test_parser_claims_subcommand():
    args = build_parser().parse_args(["claims", "docs/", "--verdict", "unsupported"])
    assert args.command == "claims"
    assert args.verdict == "unsupported"


def test_claims_with_no_annotations_shows_the_format(tmp_path, capsys):
    """A checker that just says "0 claims" teaches nothing; the first run is
    exactly when the format needs explaining."""
    db = tmp_path / "t.db"
    get_db(db).close()
    (tmp_path / "doc.md").write_text("# no claims here\n")

    rc = main(["claims", "--db", str(db), str(tmp_path)])

    assert rc == 0
    assert "<!-- claim:" in capsys.readouterr().out


def test_cmd_eval_directly_with_plain_namespace(tmp_path, capsys):
    """cmd_* handlers are unit-testable on their own -- no parser needed."""
    db = tmp_path / "t.db"
    get_db(db).close()

    rc = cmd_eval(Namespace(db=str(db), user="matt"))

    assert rc == 0
    assert "insufficient" in capsys.readouterr().out.lower()


def test_cmd_bootstrap_persona_directly_with_plain_namespace(tmp_path, capsys, monkeypatch):
    import attestation.rank

    db = tmp_path / "t.db"
    get_db(db).close()
    monkeypatch.setattr(attestation.rank, "bootstrap_persona", lambda conn, embedder, name, k: 7)

    rc = cmd_bootstrap_persona(Namespace(db=str(db), name="bench-chemist", k=10))

    assert rc == 0
    assert "wrote 7 pseudo-clicks for bench-chemist" in capsys.readouterr().out


def test_claims_exits_nonzero_on_a_contradiction(tmp_path, capsys):
    """A document asserting something false should be able to fail a commit."""
    db = tmp_path / "t.db"
    conn = get_db(db)
    conn.execute("INSERT INTO runs(project, name, source_path) VALUES ('p', 'r', '/tmp/x')")
    conn.execute("INSERT INTO run_metrics(run_id, metric, value) VALUES (1, 'wer', 0.9)")
    conn.commit()
    conn.close()
    (tmp_path / "doc.md").write_text("<!-- claim: p/r metric=wer value=0.1 -->\n")

    rc = main(["claims", "--db", str(db), str(tmp_path)])

    assert rc == 1
    assert "contradicted" in capsys.readouterr().out


def test_compare_header_names_the_shared_corpus(tmp_path, capsys, monkeypatch):
    """Naming the corpus says the comparison was checked, not assumed. A
    reader cannot tell those apart from the numbers alone."""
    import json as _json

    from attestation import cli

    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    for tag, loss in (("a", 5.0), ("b", 5.2)):
        (results / f"lm_{tag}.json").write_text(
            _json.dumps({"dataset": "wikitext-2", "best_val_loss": loss})
        )
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    cli.main(["runs", "scan", "--root", str(tmp_path)])
    capsys.readouterr()
    cli.main(["runs", "compare", "lm"])

    out = capsys.readouterr().out
    assert "all arms on wikitext-2" in out, out


from attestation import cli as attest_cli  # noqa: E402


class _Ok:
    def raise_for_status(self):
        return None


def test_warmup_pins_for_a_bounded_time_by_default(monkeypatch):
    """`keep_alive: -1` holds a model in RAM until Ollama restarts.

    On a 23 GB machine that meant 5.4 GB of llama-server pinned permanently
    across two models, and the kernel OOM-killed Chrome, a quantum-chemistry
    job and the terminal. The pin exists so a demo does not stall on a cold
    load, which is a demo-length need, not a forever need.
    """
    sent = []
    monkeypatch.setattr(
        attest_cli.httpx, "post", lambda url, **kw: sent.append((url, kw.get("json", {}))) or _Ok()
    )
    attest_cli.warmup()

    keep_alives = [body.get("keep_alive") for _, body in sent]
    assert keep_alives, "warmup sent nothing"
    assert -1 not in keep_alives, "models are still pinned forever"
    assert all(isinstance(k, str) and k.endswith("m") for k in keep_alives), keep_alives


def test_warmup_keep_alive_is_configurable(monkeypatch):
    """A demo machine and a laptop want different answers."""
    sent = []
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "2h")
    monkeypatch.setattr(
        attest_cli.httpx, "post", lambda url, **kw: sent.append((url, kw.get("json", {}))) or _Ok()
    )
    attest_cli.warmup()
    assert all(body.get("keep_alive") == "2h" for _, body in sent)


def test_reload_finds_and_signals_running_servers(monkeypatch, capsys):
    """`attest reload` restarts every live attest-mcp so edits take effect.

    An MCP server is spawned once per session and never reloads. Both live
    servers were found running code five commits stale -- the Hermes gateway
    from 17:17 and a Claude Code session from 18:13, against a 20:03 commit --
    so every fix landed that afternoon was invisible to the agent using them.

    `hermes mcp test` does not catch this: it spawns a FRESH process, so it
    reports the code on disk rather than what a session is serving.
    """
    signalled = []
    monkeypatch.setattr(attest_cli, "_running_mcp_pids", lambda: [111, 222])
    monkeypatch.setattr(attest_cli.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    attest_cli.cmd_reload(Namespace())

    assert [p for p, _ in signalled] == [111, 222]
    out = capsys.readouterr().out
    assert "111" in out and "222" in out, "each stopped server must be named"
    assert "respawns" in out, (
        "the respawn is lazy -- measured, nothing restarted for ten seconds "
        "and only came back on the next tool call -- so the message must not "
        "imply the server is already fresh"
    )


def test_reload_says_so_when_nothing_is_running(monkeypatch, capsys):
    """Silence would read as success. Nothing running is a normal state --
    no session is open -- and the caller should be told rather than guess."""
    monkeypatch.setattr(attest_cli, "_running_mcp_pids", lambda: [])
    attest_cli.cmd_reload(Namespace())
    assert "no running" in capsys.readouterr().out.lower()


def test_reload_survives_a_server_that_exits_first(monkeypatch, capsys):
    """A watchdog may restart or reap one between listing and signalling.
    That is a race, not an error, and must not abort the remaining kills."""
    import os as _os

    def flaky_kill(pid, sig):
        if pid == 111:
            raise ProcessLookupError
        return None

    monkeypatch.setattr(attest_cli, "_running_mcp_pids", lambda: [111, 222])
    monkeypatch.setattr(attest_cli.os, "kill", flaky_kill)
    attest_cli.cmd_reload(Namespace())
    out = capsys.readouterr().out
    assert "222" in out
    assert _os  # imported for clarity about what is being faked


def test_reload_is_a_real_subcommand():
    parser = build_parser()
    args = parser.parse_args(["reload"])
    assert args.func is attest_cli.cmd_reload
