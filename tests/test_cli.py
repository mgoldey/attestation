import json
from argparse import Namespace

import pytest
from conftest import seeded_db

from attestation.cli import HELP, build_parser, cmd_bootstrap_persona, cmd_eval, main
from attestation.db import get_db


def test_parser_subcommands():
    parser = build_parser()
    for argv in (
        ["ingest"],
        ["serve", "--port", "9000"],
        ["eval", "--user", "researcher"],
        ["warmup"],
        ["bootstrap-persona", "bench-chemist", "-k", "10"],
        ["kg-report"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_eval_insufficient_data_message(tmp_path, capsys):
    """Nonzero: "I could not measure this" is a failure to produce the answer
    the command exists for, and every other CLI failure path exits 1."""
    db = tmp_path / "t.db"
    seeded_db(db).close()
    rc = main(["eval", "--db", str(db), "--user", "researcher"])
    assert rc == 1
    assert "insufficient" in capsys.readouterr().err.lower()


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
    seeded_db(db).close()
    monkeypatch.setattr(
        attestation.features,
        "run_tagging",
        lambda conn, chat_fn, model, limit=None: {"tagged": 0, "failed": 0},
    )
    rc = main(["tag", "--db", str(db)])
    assert rc == 0
    assert "tagged" in capsys.readouterr().out


def test_tag_command_exit_1_on_total_failure(tmp_path, monkeypatch):
    import attestation.features

    db = tmp_path / "t.db"
    seeded_db(db).close()
    monkeypatch.setattr(
        attestation.features,
        "run_tagging",
        lambda conn, chat_fn, model, limit=None: {"tagged": 0, "failed": 3},
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
    division by zero -- since that is when someone first runs it.

    This asserted the metric table was printed, which was only ever how it
    checked the command ran. An empty graph now gets a next step instead of
    ten zeros (see the test below); the no-crash property is what this one is
    for and it is unchanged.
    """
    db = tmp_path / "t.db"
    seeded_db(db).close()

    rc = main(["kg-report", "--db", str(db)])

    # Exit 1 with guidance on stderr: an empty graph is nothing to report, and
    # the test below pins that convention. This one is about not crashing.
    assert rc == 1
    assert capsys.readouterr().err.strip(), "printed nothing at all"


def test_parser_runs_subcommands():
    parser = build_parser()
    for argv, expected in (
        (["runs", "scan"], "scan"),
        (["runs", "list", "--project", "p"], "list"),
        (["runs", "compare", "fam", "--metric", "wer"], "compare"),
        (["runs", "show", "p", "n"], "show"),
        (["runs", "record", "fam", "--arm", "a", "m=1"], "record"),
    ):
        args = parser.parse_args(argv)
        assert args.command == "runs"
        assert args.runs_command == expected


def test_runs_scan_without_a_root_explains_rather_than_crashing(tmp_path, monkeypatch, capsys):
    """No default workspace is guessed: scanning a directory the user did not
    mean is worse than saying which variable to set."""
    monkeypatch.delenv("RESEARCH_ROOT", raising=False)
    db = tmp_path / "t.db"
    seeded_db(db).close()

    rc = main(["runs", "--db", str(db), "scan"])

    assert rc == 1
    assert "RESEARCH_ROOT" in capsys.readouterr().out


def test_runs_list_before_scan_directs_the_user(tmp_path, capsys):
    db = tmp_path / "t.db"
    seeded_db(db).close()

    rc = main(["runs", "--db", str(db), "list"])

    assert rc == 1
    # stderr, not stdout: this is a failure, and the test below pins that
    # convention. The message itself is what this test is about.
    assert "scan" in capsys.readouterr().err


def test_runs_record_dry_run_prints_the_manifest_and_creates_nothing(tmp_path, capsys):
    rc = main(
        [
            "runs",
            "record",
            "asr",
            "--root",
            str(tmp_path),
            "--arm",
            "baseline",
            "wer=0.12",
            "--arm",
            "biglm",
            "wer=0.08",
            "--dry-run",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    manifest = json.loads(out)
    assert set(manifest["files"]) == {
        "results/asr_baseline.json",
        "configs/asr_baseline.yaml",
        "results/asr_biglm.json",
        "configs/asr_biglm.yaml",
    }
    # --dry-run must write NOTHING: not the results/configs files it just
    # printed, and no directory created to hold them either. (tmp_path
    # itself carries the hermetic-env fixture's own pyproject.toml marker,
    # so this checks the record command's targets specifically rather than
    # an empty directory.)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "corpora.toml").exists()


def test_runs_record_refuses_an_undeclared_metric_and_writes_nothing(tmp_path, capsys):
    rc = main(
        [
            "runs",
            "record",
            "opt",
            "--root",
            str(tmp_path),
            "--arm",
            "adam",
            "regret_bound=0.03",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "regret_bound" in err
    assert "refusing to rank" in err
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "configs").exists()


def test_runs_record_writes_and_refuses_to_overwrite_without_force(tmp_path, capsys):
    argv = [
        "runs",
        "record",
        "asr",
        "--root",
        str(tmp_path),
        "--arm",
        "baseline",
        "wer=0.12",
    ]

    rc = main(argv)
    assert rc == 0
    capsys.readouterr()
    assert (tmp_path / "results" / "asr_baseline.json").read_text() == '{\n  "wer": 0.12\n}\n'

    rc = main(argv)
    assert rc == 1
    err = capsys.readouterr().err
    assert "asr_baseline" in err
    # unchanged: the refused second call must not have clobbered the first
    assert (tmp_path / "results" / "asr_baseline.json").read_text() == '{\n  "wer": 0.12\n}\n'

    rc = main([*argv, "--force"])
    assert rc == 0


def test_runs_record_refuses_a_family_that_escapes_root(tmp_path, capsys):
    """CRITICAL 2 (final review, round 2): `family`/arm names become PATH
    SEGMENTS. `../../victim/asr` must refuse BEFORE any write, not walk out
    of --root."""
    victim = tmp_path.parent / "victim"
    rc = main(
        [
            "runs",
            "record",
            "../../victim/asr",
            "--root",
            str(tmp_path),
            "--arm",
            "base",
            "wer=0.12",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "family" in err
    assert not victim.exists(), "must not have written outside --root"
    assert not (tmp_path / "results").exists()


def test_runs_record_refuses_an_arm_name_with_a_slash(tmp_path, capsys):
    rc = main(
        [
            "runs",
            "record",
            "asr",
            "--root",
            str(tmp_path),
            "--arm",
            "../../victim/pwned",
            "wer=0.12",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "arm name" in err
    assert not (tmp_path.parent / "victim").exists()


def test_runs_record_accepts_a_plain_family_and_arm_name(tmp_path, capsys):
    rc = main(
        [
            "runs",
            "record",
            "asr-v2",
            "--root",
            str(tmp_path),
            "--arm",
            "run.1",
            "wer=0.12",
        ]
    )

    assert rc == 0
    assert (tmp_path / "results" / "asr-v2_run.1.json").exists()


def test_parser_claims_subcommand():
    args = build_parser().parse_args(["claims", "docs/", "--verdict", "unsupported"])
    assert args.command == "claims"
    assert args.verdict == "unsupported"


def test_claims_with_no_annotations_shows_the_format(tmp_path, capsys):
    """A checker that just says "0 claims" teaches nothing; the first run is
    exactly when the format needs explaining."""
    db = tmp_path / "t.db"
    seeded_db(db).close()
    (tmp_path / "doc.md").write_text("# no claims here\n")

    rc = main(["claims", "--db", str(db), str(tmp_path)])

    assert rc == 0
    assert "<!-- claim:" in capsys.readouterr().out


def test_cmd_eval_directly_with_plain_namespace(tmp_path, capsys):
    """cmd_* handlers are unit-testable on their own -- no parser needed."""
    db = tmp_path / "t.db"
    seeded_db(db).close()

    rc = cmd_eval(Namespace(db=str(db), user="researcher"))

    assert rc == 1
    assert "insufficient" in capsys.readouterr().err.lower()


def test_cmd_bootstrap_persona_directly_with_plain_namespace(tmp_path, capsys, monkeypatch):
    import attestation.rank

    db = tmp_path / "t.db"
    seeded_db(db).close()
    monkeypatch.setattr(attestation.rank, "bootstrap_persona", lambda conn, embedder, name, k: 7)

    rc = cmd_bootstrap_persona(Namespace(db=str(db), name="bench-chemist", k=10))

    assert rc == 0
    assert "wrote 7 pseudo-clicks for bench-chemist" in capsys.readouterr().out


def test_claims_exits_nonzero_on_a_contradiction(tmp_path, capsys):
    """A document asserting something false should be able to fail a commit."""
    db = tmp_path / "t.db"
    conn = seeded_db(db)
    conn.execute("INSERT INTO runs(project, name, source_path) VALUES ('p', 'r', '/tmp/x')")
    conn.execute("INSERT INTO run_metrics(run_id, metric, value) VALUES (1, 'wer', 0.9)")
    conn.commit()
    conn.close()
    (tmp_path / "doc.md").write_text("<!-- claim: p/r metric=wer value=0.1 -->\n")

    rc = main(["claims", "--db", str(db), str(tmp_path)])

    assert rc == 1
    assert "contradicted" in capsys.readouterr().out


def test_claims_reports_uncited_citation_key(tmp_path, monkeypatch, capsys):
    """`attest claims` must run the citation lint too, not just the numeric
    check -- a claim whose number agrees but whose `cite=` key resolves
    nowhere should not print as plain `supported`."""
    from attestation.claims import VerdictKind

    db = tmp_path / "t.db"
    artifact = tmp_path / "result.json"
    artifact.write_text("{}")
    conn = seeded_db(db)
    conn.execute(
        "INSERT INTO runs(project, name, source_path) VALUES ('p', 'r', ?)", (str(artifact),)
    )
    conn.execute("INSERT INTO run_metrics(run_id, metric, value) VALUES (1, 'wer', 0.1)")
    conn.commit()
    conn.close()
    (tmp_path / "doc.md").write_text("<!-- claim: p/r metric=wer value=0.1 cite=nosuchkey -->\n")

    # No .bib file here, and no Zotero library either: Resolver.from_env()
    # globs *.bib in Path.cwd(), so an empty cwd guarantees nothing resolves
    # `nosuchkey` -- the lint must fire regardless.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    rc = main(["claims", "--db", str(db), str(tmp_path)])

    out = capsys.readouterr().out
    assert VerdictKind.UNCITED.value in out
    assert "nosuchkey" in out
    assert rc == 0  # an uncited lint result is not a contradiction


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


def test_eval_for_an_unknown_user_exits_nonzero(tmp_path, capsys):
    """`--user nobody` produced no measurement and exited 0.

    A script that runs `attest eval` to gate something read that as a pass. It
    is the only failure path in this CLI that did.
    """
    db = tmp_path / "t.db"
    seeded_db(db).close()

    rc = main(["eval", "--db", str(db), "--user", "definitely-not-a-persona"])

    assert rc == 1
    assert "insufficient" in capsys.readouterr().err.lower()


def test_eval_with_a_real_measurement_still_exits_zero(tmp_path, capsys, monkeypatch):
    """Only the no-answer path changed."""
    import attestation.rank

    db = tmp_path / "t.db"
    seeded_db(db).close()
    # evaluate_user returns a labelled dict now, not a bare float: the number
    # covers the click classifier only, and saying so is the point.
    monkeypatch.setattr(
        attestation.rank,
        "evaluate_user",
        lambda conn, uid: {
            "auc": 0.75,
            "n_clicks": 20,
            "n_splits": 4,
            "measures": "the click classifier alone",
        },
    )

    rc = main(["eval", "--db", str(db), "--user", "researcher"])

    assert rc == 0
    assert "0.750" in capsys.readouterr().out


def test_bootstrap_persona_for_an_unknown_name_does_not_traceback(tmp_path, capsys):
    """rank.bootstrap_persona raises ValueError and cli.py did not catch it.

    It was the only user-triggerable traceback in this CLI, on a command the
    README recommends by name. Neighbouring commands print one line naming the
    fix and return 1.
    """
    db = tmp_path / "t.db"
    seeded_db(db).close()

    rc = main(["bootstrap-persona", "--db", str(db), "no-such-persona"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "no-such-persona" in out
    assert "Traceback" not in out
    # names the fix, not just the problem
    assert "persona" in out.lower()


def test_backup_writes_a_restorable_copy(tmp_path, monkeypatch, capsys):
    """`backup_db` was added in round 1 and had no CLI entry point.

    The finding it fixed was that operators reach for `cp hermes.db`, which
    silently drops the WAL -- five such copies were sitting beside the live
    database. A backup function no operator can invoke does not fix that; the
    command is the whole deliverable.
    """
    import sqlite3

    db = tmp_path / "src.db"
    conn = seeded_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ada', 'analysis')")
    conn.commit()
    # The connection stays OPEN. Closing it checkpoints the WAL, which put the
    # data in the main file and made this test pass against the exact bug it
    # names -- swapping VACUUM INTO for a plain file copy kept it green. An
    # operator running `attest backup` while the web UI or a cron ingest holds
    # a connection is the whole scenario, so the test has to hold one too.
    assert (tmp_path / "src.db-wal").exists(), "no WAL to lose; this test would prove nothing"

    dest = tmp_path / "out" / "backup.db"
    rc = main(["backup", "--db", str(db), str(dest)])

    assert rc == 0
    assert dest.is_file()
    names = [r[0] for r in sqlite3.connect(dest).execute("SELECT name FROM users")]
    assert "ada" in names, "the backup lost a committed row still sitting in the WAL"
    conn.close()
    assert str(dest) in capsys.readouterr().out


def test_backup_refuses_an_existing_destination(tmp_path, capsys):
    """Silently replacing the previous backup is one keystroke from having
    none. Exit non-zero and say so."""

    db = tmp_path / "src.db"
    seeded_db(db).commit()
    dest = tmp_path / "b.db"
    dest.write_text("not empty")

    rc = main(["backup", "--db", str(db), str(dest)])

    assert rc == 1
    assert dest.read_text() == "not empty", "an existing file was overwritten"
    assert "exists" in capsys.readouterr().out.lower()


def test_kg_report_on_an_empty_graph_says_what_to_do(tmp_path, capsys):
    """Ten zeros are not an answer to "what does my reading graph look like".

    Flagged in the first review round and still true seven rounds later: a new
    user runs this before ingesting anything and gets `nodes 0 / edges 0 /
    singleton_rate 0.0 ...` with no indication that the graph is DERIVED from
    the tagging pass, so the fix is `attest ingest && attest tag`.

    The house pattern is already set by `attest runs list`, which says "no runs
    recorded -- run `attest runs scan` first". This makes the graph match it.
    """

    db = tmp_path / "t.db"
    seeded_db(db).commit()

    rc = main(["kg-report", "--db", str(db)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "attest tag" in err, err
    # The zeros are noise when there is nothing to report.
    assert "singleton_rate" not in err, "dumped the full metric table for an empty graph"


def test_kg_report_still_reports_a_populated_graph(tmp_path, capsys):
    """The empty-state branch must not swallow a real report."""

    db = tmp_path / "t.db"
    conn = seeded_db(db)
    for i in range(6):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', 's', ?)",
            (f"item {i}", f"h{i}"),
        )
        for tag in ("alpha", "beta"):
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (cur.lastrowid, tag))
    conn.commit()
    conn.close()

    rc = main(["kg-report", "--db", str(db)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "nodes" in out, "a populated graph must still get the metrics"


def test_failures_are_reported_on_stderr_not_stdout(tmp_path, capsys, monkeypatch):
    """A failing command wrote its reason to stdout, so a pipeline consumed it.

    Every CLI failure path exits 1 correctly and prints to stdout, which means
    `attest runs list > runs.txt` writes "no runs recorded -- run `attest runs
    scan` first" into the data file, and `attest runs list | jq` feeds the
    error text to jq. Exit codes are for programs; streams are how the program
    tells data apart from complaint.
    """

    db = tmp_path / "t.db"
    seeded_db(db).commit()
    monkeypatch.setenv("RSS_DB", str(db))

    rc = main(["runs", "list"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "no runs recorded" in captured.err, (
        f"the failure went to stdout: out={captured.out!r} err={captured.err!r}"
    )
    assert not captured.out.strip(), f"stdout should be empty on failure, got {captured.out!r}"


def test_kg_report_on_an_empty_graph_fails_like_its_siblings(tmp_path, capsys, monkeypatch):
    """The empty state I added in an early round predates the stderr/exit
    convention added later, and never adopted it.

    `runs compare` on a missing family exits 1 to stderr; `eval` with too few
    clicks exits 1 to stderr; `kg-report` on an empty graph printed its
    guidance to STDOUT and exited 0. A pipeline cannot tell "no graph yet" from
    "here is your graph", which is the same defect the six other sites were
    fixed for.
    """

    db = tmp_path / "t.db"
    seeded_db(db).commit()
    monkeypatch.setenv("RSS_DB", str(db))

    rc = main(["kg-report"])
    captured = capsys.readouterr()

    assert rc == 1, "an empty graph reported success"
    assert "attest tag" in captured.err, f"guidance went to stdout: {captured.out!r}"
    assert not captured.out.strip()


def test_no_message_recommends_a_subcommand_that_does_not_exist():
    """`attest eval` on an unknown persona advised `attest personas`.

    There is no `personas` subcommand and never has been -- the advice exits 2
    with "invalid choice". A recovery message naming a command that does not
    exist is worse than none: it sends a stuck user somewhere that fails again.

    Guards the class rather than the instance: any `attest <word>` in a
    user-facing string in cli.py must be a real subcommand.
    """
    import ast
    import re
    from pathlib import Path

    from attestation.cli import build_parser

    parser = build_parser()
    real = set(parser._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    # Read STRING LITERALS only, not comments -- the comment recording that
    # `attest personas` was a phantom names it, and a source-wide regex counts
    # that as a fresh recommendation. Multi-word forms like `attest runs scan`
    # name a subcommand plus its own subcommand; only the first word is checked.
    source = (Path(__file__).resolve().parents[1] / "src" / "attestation" / "cli.py").read_text()
    tree = ast.parse(source)
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            named |= set(re.findall(r"attest ([a-z][a-z-]+)", node.value))
    phantom = sorted(named - real - {"mcp"})
    assert not phantom, f"cli.py recommends commands that do not exist: {phantom}"


def test_a_subcommand_help_describes_what_it_prints():
    """`eval --help` promised "leave-last-N-out AUC", which the command stopped
    printing when its scope was corrected -- it reports a click-classifier AUC
    over a shuffled StratifiedKFold. The output and the docstring were fixed
    and `add_parser`'s help string was missed."""
    from attestation.cli import build_parser

    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    eval_help = choices["eval"].description or ""
    for action in parser._subparsers._group_actions[0]._choices_actions:  # type: ignore[union-attr]
        if action.dest == "eval":
            eval_help += " " + (action.help or "")
    assert "leave-last" not in eval_help.lower(), (
        f"eval's help names an approach the command abandoned: {eval_help!r}"
    )


def test_an_unusable_database_path_is_a_sentence_not_a_traceback(tmp_path, capsys, monkeypatch):
    """`RSS_DB=/nonexistent/deep/x.db attest runs list` printed 18 lines of
    traceback ending in "unable to open database file" -- which does not name
    the file, and so does not name the typo. It fired in open_db, before any
    command body ran, which is why it escaped every command's own error
    handling. A directory that does not exist YET is an ordinary first run and
    must still succeed."""
    import sqlite3

    from attestation.cli import main

    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        monkeypatch.setenv("RSS_DB", str(unwritable / "deep" / "x.db"))
        code = main(["runs", "list"])
    finally:
        unwritable.chmod(0o700)
    err = capsys.readouterr().err
    assert code == 1
    assert "Traceback" not in err
    assert "cannot open database at" in err, err
    assert str(unwritable) in err, "the message must name the path that failed"

    # The ordinary first run: parents are created rather than refused.
    fresh = tmp_path / "brand" / "new" / "h.db"
    monkeypatch.setenv("RSS_DB", str(fresh))
    main(["runs", "list"])
    assert fresh.exists(), "a not-yet-existing directory is a first run, not an error"
    assert sqlite3.connect(str(fresh)) is not None


def test_tag_command_names_the_cause_when_the_chat_backend_is_down(tmp_path, capsys, monkeypatch):
    """`{'tagged': 0, 'failed': 2}` is honest but says nothing about why.
    ingest names the cause and points at the doctor; tag does the same, and
    exits non-zero even though no item "failed" -- none was attempted."""
    import attestation.features

    db = tmp_path / "t.db"
    seeded_db(db).close()
    monkeypatch.setattr(
        attestation.features,
        "run_tagging",
        lambda conn, chat_fn, model, limit=None: {"tagged": 0, "failed": 0, "chat_down": True},
    )
    assert main(["tag", "--db", str(db)]) == 1
    streams = capsys.readouterr()
    text = streams.out + streams.err
    assert "unreachable" in text
    assert "install --check" in text


def test_version_flag_reports_the_installed_version(capsys):
    from importlib.metadata import version

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"attest {version('attestation')}" in capsys.readouterr().out


def test_bootstrap_persona_creates_its_demo_persona_on_a_fresh_database(tmp_path, monkeypatch):
    """A fresh database has no personas, so the README's
    `attest bootstrap-persona bench-chemist` must create the demo persona it
    names before writing its pseudo-clicks -- otherwise the documented demo
    command fails on the database `attest install` just created."""
    import attestation.rank
    from attestation.db import SEED_USERS

    db = tmp_path / "t.db"
    get_db(db).close()
    monkeypatch.setattr(attestation.rank, "bootstrap_persona", lambda conn, embedder, name, k: 7)

    assert main(["bootstrap-persona", "--db", str(db), "bench-chemist"]) == 0
    conn = seeded_db(db)
    row = conn.execute("SELECT interests FROM users WHERE name = 'bench-chemist'").fetchone()
    conn.close()
    assert row is not None
    assert row["interests"] == SEED_USERS["bench-chemist"]


def test_bootstrap_persona_unknown_name_names_the_demo_personas(tmp_path, capsys):
    db = tmp_path / "t.db"
    get_db(db).close()
    assert main(["bootstrap-persona", "--db", str(db), "no-such-persona"]) == 1
    out = capsys.readouterr().out
    assert "bench-chemist" in out, "the fix is to name a demo persona; say which exist"


def _leaf_commands(parser):
    """Yield (path, help_text, func) for every leaf subcommand -- `runs`'s
    own sub-subcommands included, so the equality test below covers
    `runs scan` etc. and not just `runs` itself. Walks the live parser tree
    rather than a hand-kept list, so a new subcommand is covered the moment
    it is added."""
    group = parser._subparsers._group_actions[0]  # type: ignore[union-attr]
    for action in group._choices_actions:
        child = group.choices[action.dest]
        func = child.get_default("func")
        if func is not None:
            yield [action.dest], action.help, func
        else:
            for path, help_text, inner_func in _leaf_commands(child):
                yield [action.dest, *path], help_text, inner_func


def test_every_cmd_docstring_is_its_helps_first_line():
    """`build_parser`'s docstring names the design: `help=` is the one
    source, and each `cmd_*` docstring's first line is that same string,
    reused rather than retyped -- so a fix to one (like the `eval --help`
    drift `test_a_subcommand_help_describes_what_it_prints` guards) cannot
    land in the help text without also landing in the docstring, or vice
    versa.
    """
    for path, help_text, func in _leaf_commands(build_parser()):
        first_line = (func.__doc__ or "").splitlines()[0] if func.__doc__ else None
        assert first_line == help_text, (
            f"attest {' '.join(path)}: docstring first line {first_line!r} != help {help_text!r}"
        )


def test_build_parser_help_strings_come_from_HELP():
    """The other half of the one-source contract: `HELP` is not just a table
    the docstrings happen to match, it is what `add_parser(..., help=...)`
    itself reads. Walks the live parser tree the same way the test above
    does, and asserts each subcommand's actual `help=` (what `attest --help`
    prints) equals `HELP[dotted_name]` -- so a literal string slipped back
    into an `add_parser` call, bypassing `HELP` entirely, is caught even
    though it would still happen to match the docstring by coincidence."""
    for path, help_text, _func in _leaf_commands(build_parser()):
        key = ".".join(path)
        assert key in HELP, f"attest {' '.join(path)}: no HELP entry for key {key!r}"
        assert help_text == HELP[key], (
            f"attest {' '.join(path)}: add_parser help={help_text!r} != HELP[{key!r}]={HELP[key]!r}"
        )
