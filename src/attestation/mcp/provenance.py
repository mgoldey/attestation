"""Experiment ledger and claim checking: the `runs.*` namespace.

The ledger reads artifacts already on disk -- it is deliberately not an
experiment tracker and asks for no `log_metric()` calls. See `ledger.py`'s
module docstring for why adoption cost is the design constraint.

Two rules keep it honest and both live in the domain, not here: record what is
unambiguous and refuse to guess the rest, and never rank a metric whose
direction is undeclared. An undeclared direction reaches the caller as a
ToolError with the reason spelled out, because it is caller-fixable.
"""

from typing import Annotated

from pydantic import BaseModel, Field

from attestation import ledger, record
from attestation.mcp._shared import MAX_LIST_LIMIT, Limit, Verdict
from attestation.mcp._tool import ToolError, open_db, tool

# Ten, not twenty. Even without source_path, 20 rows emitted 4576 chars against
# the live ledger. The message already reports what was not shown, so a caller
# who wants more asks for more.
DEFAULT_RUNS_LIMIT = 10

# Metric rows one runs.detail may return. It was uncapped, and its docstring
# promised "every metric" -- so a 429-metric quantum-chemistry run emitted
# 60,680 chars, 49,945 of it metrics, against a 7000 ceiling. 72 of 858 live
# runs were over. The census had it filed as a status-returning tool, which is
# how the largest response on the surface shipped unmeasured.
#
# A caller comparing arms uses runs.compare; a caller reading one run wants to
# see its shape. n_metrics reports the true count either way.
MAX_METRIC_ROWS = 40

# Runs one listing may return. Distinct from _shared.MAX_LIST_LIMIT (16, a feed
# row is cheaper): at 50 this emitted 9965 chars against a 7000 ceiling, and
# its own message tells the caller to "raise limit (max 50)" -- round 11's
# honesty fix advertised the limit that breaks it. Every size guard drove the
# DEFAULT, so the escape moved from a tool nobody measured to an argument
# nobody passed.
MAX_RUNS_LISTED = 25

# Arms one comparison may return. It had no cap and reached 13624 chars on a
# 48-arm family -- larger than the pre-fix runs.detail. Being declared a
# composition tool is exemption from the CONVERSATIONAL budget, not from what
# a caller can hold. Arms are ranked, so this keeps the ones that won.
MAX_ARMS_SHOWN = 20

NO_ROOT = (
    "no workspace configured -- set RESEARCH_ROOT to the directory holding your"
    " projects, or pass root explicitly"
)


class Arm(BaseModel):
    """One arm of `runs.record`'s `arms` argument: a name and its final
    metric values. A real pydantic model rather than a bare `dict` so the
    schema an agent sees states the shape (`name`, `metrics`) instead of an
    unconstrained object -- the same reasoning `mcp/ask.py`'s `Ref`/`Answer`
    already apply to a return shape, extended here to an argument."""

    name: str
    metrics: dict[str, float]


def register(mcp) -> None:
    """Attach every runs.* tool to the server."""

    @mcp.tool(name="runs.scan")
    def runs_scan(
        root: str | None = None, project: str | None = None, confirm: bool = False
    ) -> dict:
        """Read experiment runs from artifacts already on disk into the ledger.

        Walks a workspace directory, treating each subdirectory as a project, and
        reads the conventions research repos already use -- `results/`, `logs/`,
        `configs/`, `outputs/`, `benchmarks/` holding JSON, JSONL, YAML or TOML.
        Nothing needs to be instrumented and no project needs to be registered.

        `root` defaults to the RESEARCH_ROOT environment variable. Requires
        `confirm=true` since it replaces each scanned project's rows. Directories
        with nothing recognisable are listed in `empty` rather than omitted, so
        "found nothing" is never mistaken for "nothing was there".

        """
        return _scan(root, project, confirm)

    @mcp.tool(name="runs.record")
    def runs_record(
        family: str,
        arms: list[Arm],
        corpus: str | None = None,
        directions: dict[str, str] | None = None,
        config: dict[str, str] | None = None,
        root: str | None = None,
        project: str | None = None,
        confirm: bool = False,
    ) -> dict:
        """Write an experiment run's results/config files so `runs.scan` reads them back.

        Derives the ledger's on-disk shape rather than asking you to transcribe it by hand.

        `arms` is `[{"name": ..., "metrics": {metric: value}}, ...]` -- the
        arms of one family (a sweep's variants, or a single run as a
        one-arm family). Without `confirm=true` this writes NOTHING and
        returns the `manifest` it would write (`{relpath: content}`), the
        same preview `attest runs record --dry-run` prints -- call it first
        to see the files before committing to them.

        With `confirm=true` it writes (new files only; a target that already
        exists is a refusal naming every collision, before anything is
        written -- there is no `force` here, unlike the CLI: overwriting a
        result file is the failure this ledger exists to catch), then scans
        the project and returns `compare` for `family`, so one call takes a
        run from numbers to a ranked ledger entry.

        A metric this ledger has no direction for -- built-in or from a
        prior `--direction`/`directions` declaration -- is refused with the
        same sentence `runs.compare` prints, whether or not `confirm` is
        set: this tool never guesses which way a metric should rank.

        `root`/`project` resolve the way `runs.scan`'s do: `root` defaults to
        RESEARCH_ROOT, and `project` names the subdirectory the files are
        written under (the project itself, when omitted).
        """
        return _record(family, arms, corpus, directions, config, root, project, confirm)

    @mcp.tool(name="runs.list")
    def runs_list(
        project: str | None = None,
        family: str | None = None,
        limit: Limit = DEFAULT_RUNS_LIMIT,
    ) -> dict:
        """Experiment runs in the ledger, with the families they group into.

        A `family` is a set of sibling runs -- the arms of a sweep, or one run's
        checkpoints over training. Use it with `runs.compare` to answer which arm
        won. Also returns the family list, so you can see what is comparable.

        """
        return _list(project, family, limit)

    @mcp.tool(name="runs.compare")
    def runs_compare(
        family: str,
        metric: str | None = None,
        project: Annotated[
            str | None,
            Field(description="Required when the family name exists in more than one project."),
        ] = None,
    ) -> dict:
        """Rank the arms of an experiment family by a metric.

        The question a sweep exists to answer and that usually lives only in
        filenames: which variant won, on what metric, by how much. Omit `metric` to
        use the one most arms share.

        `family` is a name from `runs.list`, which needs NO arguments and
        returns every family in the ledger. Call it first when the user names a
        sweep in prose ("my learning-rate sweep") -- the recorded names are
        whatever the directories were called, not what the user calls them.
        Measured: without this, gemma4:e2b pushed the question back at the user
        and hermes3:8b invented a family, a metric and a project.

        Refuses to rank a metric whose direction is undeclared rather than guessing
        -- ranking WER as if higher were better would name the worst arm the
        winner. Arms with no value for the metric are listed in `without_metric`
        rather than dropped: an arm that was never evaluated is a finding.

        A family name is unique per project, not globally. If the same name exists
        in two projects this refuses and names them: arms from different projects
        are not comparable, and picking one silently would be a guess.

        """
        return _compare(family, metric, project)

    @mcp.tool(name="runs.detail")
    def runs_detail(project: str, name: str) -> dict:
        """Everything recorded about one run.

        Config shape, every metric, the source path it was read from, and the
        header comment from its config if it had one. `project` and `name` come
        from `runs.list`, which needs no arguments.

        That header is often where the hypothesis and the single changed variable
        are written down. It is stored verbatim and never interpreted.

        """
        return _detail(project, name)

    @mcp.tool(name="runs.claims_coverage")
    def claims_coverage(path: str | None = None) -> dict:
        """Numbers asserted in Markdown that no claim annotation covers.

        Omit `path` to scan every Markdown file under the configured research
        root.

        The inverse of `runs.claims_check`: that verifies the claims that exist, this
        finds assertions nobody made checkable. A document with zero contradicted
        claims looks healthy while asserting a dozen unverifiable numbers, and
        nothing else surfaces the difference.

        Only decimals count as measurements -- integers in prose are overwhelmingly
        versions, counts and dates. Versions, ISO dates, URLs, package pins and
        anything inside an HTML comment are excluded, since a comment renders as
        nothing and asserts nothing to a reader.

        """
        return _coverage(path)

    @mcp.tool(name="runs.claims_check")
    def claims_check(path: str | None = None, verdict: Verdict = None) -> dict:
        """Verify numeric claims written in Markdown against runs in the ledger.

        Omit `path` to check every Markdown file under the configured research
        root.

        A claim is an HTML comment beside the prose it describes, so it renders as
        nothing and the document reads exactly as before:

        <!-- claim: project/run metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->

        Six verdicts, and the differences matter. `supported`: a run agrees.
        `contradicted`: a run disagrees — the document or the run is wrong.
        `unsupported`: no run matches, so the claim may still be true but nothing
        backs it. `ambiguous`: a wildcard matched several runs, so which is meant is
        undecidable. `stale`: the value matches but the artifact changed after
        `as_of`, so it is worth re-verifying. `uncited`: the claim named a `cite=`
        key no configured bibliographic source has — a lint on the citation, never
        a judgement about what the cited work says. A claim with no `cite=` is
        never linted, and a claim can be both contradicted and uncited.

        Filter with `verdict` to answer "what in my documentation is unsupported".
        Read-only: it reports, it never edits a document.

        """
        return _check(path, verdict)


def _nothing_listed(conn, project: str | None, family: str | None) -> str:
    """Why a listing came back empty -- the two causes are different.

    An empty ledger and a filter that matched nothing both said "no runs
    recorded -- call runs.scan(confirm=true) first". Following that against a
    ledger holding nine runs re-scans a database that is already correct.
    """
    total = ledger.count_runs(conn)
    if not total:
        return "no runs recorded -- call runs.scan(confirm=true) first"
    wanted = ", ".join(f"{k}={v!r}" for k, v in (("project", project), ("family", family)) if v)
    known = [f["family"] for f in ledger.families(conn)[:8]]
    return (
        f"no runs matching {wanted}, though the ledger holds {total}."
        f" Recorded families include: {', '.join(known) or '(none)'}"
    )


@tool(empty={"scanned": {}, "empty": [], "diagnostics": {}}, needs_db=False, label="runs_scan")
def _scan(root: str | None = None, project: str | None = None, confirm: bool = False) -> dict:
    if not confirm:
        raise ToolError(
            "refusing to scan without confirm=true. This replaces the ledger's"
            " rows for each project scanned (they are re-read from disk, so"
            " nothing unrecoverable is lost)."
        )
    target = ledger.workspace_root(root)
    if target is None:
        raise ToolError(NO_ROOT)
    with open_db() as conn:
        out = ledger.scan(conn, target, project=project)
    # ledger.scan sets `message` when the root itself is the problem -- a typo'd
    # path found nothing for a reason the caller can fix. Discarding it and
    # reporting "0 run(s) across 0 project(s)" made a nonexistent directory
    # indistinguishable from a correctly-configured but genuinely empty one,
    # which is the confusion the diagnostics below exist to prevent.
    if not out["scanned"] and out.get("message"):
        raise ToolError(
            f"{out['message']} -- set RESEARCH_ROOT or pass root= to a directory that exists"
        )
    total = sum(out["scanned"].values())
    return {
        "message": f"{total} run(s) across {len(out['scanned'])} project(s)",
        "scanned": out["scanned"],
        "empty": out.get("empty", []),
        # why each empty project was empty: the caller is a model, and a bare
        # "0 run(s)" gives it nothing to tell the user or act on
        "diagnostics": out.get("diagnostics", {}),
    }


def _record_target(target, project: str | None):
    """Where `record.plan`'s manifest is rooted -- the project directory
    itself, matching `record.py`'s `{relpath: content}` paths (`results/...`,
    `configs/...`, root-relative to ONE project), not the workspace `runs.scan`
    walks. `project` omitted writes directly under `target`, the same
    fallback `ledger.scan` uses for a workspace that IS a single project."""
    return target / project if project else target


def _arm_metrics(arms: list) -> dict[str, dict[str, float]]:
    """`{name: metrics}` from `arms`, accepting both shapes: `Arm` instances
    (how they arrive through the real MCP schema, pydantic-coerced) and plain
    dicts (how `_runs_record_impl` -- like every other tool's alias -- is
    called directly by tests and by any caller that skips the MCP transport).
    """
    out: dict[str, dict[str, float]] = {}
    for a in arms:
        if isinstance(a, Arm):
            out[a.name] = a.metrics
        else:
            out[a["name"]] = a["metrics"]
    return out


def _refuse_differing_declared_directions(declared: dict[str, str], known: dict[str, str]) -> None:
    """Raise `ToolError` if `declared` disagrees with `known` for any metric.

    A DIFFERING already-declared direction (round 3 review: including a
    contradiction with the BUILT-IN table, not only a prior file entry --
    `--direction wer=higher_is_better` used to succeed silently, rc=0,
    against `wer`'s built-in `lower_is_better`) is a conflict `plan()`
    itself cannot see: `plan()` omits `metric_direction.toml` from the
    manifest entirely once a metric is already in `known_directions` --
    correctly, since there is nothing NEW to write -- but that means a
    caller who declares a contradicting direction would otherwise sail
    through with no manifest entry to conflict-check, silently ranking on
    the STALE direction instead of refusing. `record.differing_directions`
    is the one shared check (this module and `cli.cmd_runs_record` both
    call it): checked directly against `known`, before `plan()` ever runs.
    """
    conflicting = record.differing_directions(declared, known)
    if not conflicting:
        return
    detail = ", ".join(f"{m}: {old!r} -> {new!r}" for m, (old, new) in conflicting.items())
    raise ToolError(
        "refusing to overwrite an existing [metric_direction] entry without"
        f" --force (CLI-only; this tool has no override): {detail}"
    )


def _validate_record_names(
    family: str,
    arm_metrics: dict[str, dict[str, float]],
    project: str | None,
    corpus: str | None,
    declared: dict[str, str],
) -> None:
    """Raise `ToolError` for any unsafe `family`/arm name/`project`/`corpus`,
    or any `declared` direction that is not one of the two the ledger ranks
    by.

    `family` and an arm's name both become PATH SEGMENTS
    (`record.plan` -> `results/<family>_<arm>.json`), and `project` names the
    subdirectory `_record_target` writes under -- an unvalidated `../../x`
    in any of the three walks a write clean out of the workspace `--root`
    (equivalently `root`) is supposed to confine it to. `corpus` never
    becomes a path segment, but it IS interpolated as a bare TOML table
    name (`[corpus.<name>]`) by `record.plan` -- the same `validate_name`
    grammar (no `/`, no leading `.`) keeps it a safe, unambiguous table
    name, consistent with the CLI's own `--corpus` check. Checked here,
    before `record.plan` ever builds a manifest, using the SAME
    `record.validate_name`/`record.validate_direction` the CLI's
    `_parse_record_args` uses, so both callers refuse identically rather
    than the tool trusting an agent caller more than the CLI trusts a human
    one -- the opposite of what the spec's write-safety rules intend, since
    the tool has no `--force` a mistake could even be undone with.
    """
    try:
        record.validate_name(family, label="family")
        if project is not None:
            record.validate_name(project, label="project")
        if corpus is not None:
            record.validate_name(corpus, label="corpus")
        for name in arm_metrics:
            record.validate_name(name, label="arm name")
        for metric, direction in declared.items():
            record.validate_direction(metric, direction)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@tool(empty={"written": [], "manifest": {}, "compare": None}, needs_db=False, label="runs_record")
def _record(
    family: str,
    arms: list[Arm],
    corpus: str | None = None,
    directions: dict[str, str] | None = None,
    config: dict[str, str] | None = None,
    root: str | None = None,
    project: str | None = None,
    confirm: bool = False,
) -> dict:
    target = ledger.workspace_root(root)
    if target is None:
        raise ToolError(NO_ROOT)

    arm_metrics = _arm_metrics(arms)
    declared = directions or {}
    _validate_record_names(family, arm_metrics, project, corpus, declared)
    known = ledger.metric_directions()
    missing = record.undeclared(arm_metrics, {**known, **declared})
    if missing:
        # Same sentence runs.compare itself raises for one named metric, so
        # an agent that hits this and one that hits a bare runs.compare(...)
        # learn the identical remedy rather than two phrasings of one rule.
        raise ToolError("\n".join(ledger.unknown_direction_message(m) for m in missing))
    _refuse_differing_declared_directions(declared, known)

    manifest = record.plan(
        family,
        arm_metrics,
        corpus=corpus,
        directions=declared,
        config=config,
        known_directions=known,
    )

    if not confirm:
        return {"manifest": manifest, "written": [], "compare": None}
    return _record_confirm(target, project, family, manifest)


def _toml_target(relpath: str, write_root):
    """Where one manifest TOML entry actually gets written -- mirrors
    `cli.py`'s private `_toml_target`, duplicated rather than imported since
    mcp/ may not import a private name from a non-mcp module (see `tests/
    test_architecture.py::test_no_mcp_module_imports_a_private_domain_name`).
    `metric_direction.toml` is NOT root-relative: the ledger always reads it
    from `ledger.metric_direction_path()`, same as `runs.compare`'s own
    refusal names -- writing it under the project instead would declare a
    direction `compare` never looks at. `corpora.toml` is the one genuinely
    root-relative merge target (root here is the PROJECT directory, matching
    `record.plan`'s root-relative paths)."""
    if relpath == "metric_direction.toml":
        return ledger.metric_direction_path()
    return write_root / relpath


def _merge_toml_files(toml_files: dict[str, str], write_root) -> dict:
    """Every manifest TOML entry, merged (never overwritten -- no `force`
    here) into whatever text already exists at its real target, computed
    entirely in memory: `record.merge_toml_table` raises `ValueError` on a
    differing value before this returns anything, so a caller can refuse
    before ANY disk write, the same "refuse before writing" guarantee
    `record.write` gives the per-arm files. Returns `{path: merged_text}`,
    not yet written."""
    merged: dict = {}
    for relpath, content in toml_files.items():
        target_path = _toml_target(relpath, write_root)
        existing_text = target_path.read_text() if target_path.exists() else ""
        for table, entries in record.toml_tables(content):
            existing_text = record.merge_toml_table(existing_text, table, entries, force=False)
        merged[target_path] = existing_text
    return merged


def _record_confirm(target, project: str | None, family: str, manifest: dict[str, str]) -> dict:
    """The `confirm=true` half of `_record`: write the manifest, then scan
    the project back in and compare the family, so one call takes a run from
    numbers to a ranked ledger entry.

    Per-arm results/configs are new-file-only (no `force`, unlike the CLI --
    see the tool's own docstring). `corpora.toml`/`metric_direction.toml`
    merge instead, keeping every foreign entry, and refuse (ToolError) only
    on a DIFFERING value for a key already declared -- the same rule the CLI
    enforces via `record.merge_toml_table`, just with no `--force` escape
    hatch here. Both refusals are computed and raised BEFORE any write: the
    TOML merge is pure text (see `_merge_toml_files`), so a differing
    direction is caught with nothing yet on disk, exactly like a collision on
    a per-arm result file.
    """
    write_root = _record_target(target, project)
    per_arm = {k: v for k, v in manifest.items() if not k.endswith(".toml")}
    toml_files = {k: v for k, v in manifest.items() if k.endswith(".toml")}

    try:
        merged_toml = _merge_toml_files(toml_files, write_root)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    try:
        written = record.write(write_root, per_arm, force=False)
    except FileExistsError as exc:
        raise ToolError(str(exc)) from exc

    for toml_path, text in merged_toml.items():
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        toml_path.write_text(text)
        written.append(toml_path)

    with open_db() as conn:
        ledger.scan(conn, target, project=project)
        try:
            compare = ledger.compare(conn, family, project=project)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return {
        "message": f"wrote {len(written)} file(s)",
        "written": [str(p) for p in written],
        "manifest": {},
        "compare": compare,
    }


@tool(empty={"runs": [], "families": [], "n_families": 0}, label="runs_list")
def _list(
    conn, project: str | None = None, family: str | None = None, limit: int = DEFAULT_RUNS_LIMIT
) -> dict:
    capped = min(limit, MAX_RUNS_LISTED)
    found = ledger.sample_runs(conn, project=project, family=family, limit=capped)
    if not found:
        raise ToolError(_nothing_listed(conn, project, family))
    families = ledger.families(conn, project=project)
    # A sample, not a listing. Fifty of 403 was 73% of this response and an
    # agent could not act on any of it -- a caller asking "which arm won?"
    # got a wall of names ahead of the runs it asked for. A handful shows the
    # SHAPE of what is comparable; the count and the narrowing hint do the
    # rest, and runs.compare's own error names the alternatives when a guess
    # misses.
    FAMILY_SAMPLE = 8
    # Cap the family list. `limit` bounds `runs` only, so a workspace with
    # hundreds of families returned all of them alongside a handful of runs --
    # the field advertised as the bridge to runs_compare was the one that blew
    # the caller's context. Truncation is reported, never silent.
    shown = families[:FAMILY_SAMPLE]
    # Report the runs truncation too. The families truncation below has always
    # been reported exactly, while 212 of 222 runs vanished silently -- and
    # DEFAULT_RUNS_LIMIT was halved to 10 on the stated grounds that this
    # message already said so. It did not. feed.list's wording is the house
    # convention.
    total_runs = ledger.count_runs(conn, project=project, family=family)
    message = f"{len(found)} run(s)"
    if total_runs > len(found):
        # MAX_LIST_LIMIT, not MAX_RUNS_LISTED: the latter is this module's row
        # cap, but the bound an agent must satisfy is Limit's le=16, so "raise
        # limit (max 25)" named a value pydantic rejects -- a dead end, since
        # the rejection is a validation dump. At the ceiling there is no larger
        # limit to ask for, so the advice becomes narrowing instead.
        room = min(MAX_LIST_LIMIT, MAX_RUNS_LISTED)
        more = f"raise limit (max {room})" if len(found) < room else "pass project= to narrow"
        message += f" of {total_runs} -- {more}"
    if len(families) > len(shown):
        message += (
            f"; showing {len(shown)} of {len(families)} families -- pass project= to narrow them"
        )
    # source_path is dropped from a LIST row: it is the most expensive field
    # and the least useful here. Measured on the live ledger, a default
    # runs.list emitted 5926 chars of which 4655 was the runs array, each row
    # carrying a full absolute path. A caller narrowing a list needs project,
    # name and family; runs.detail is the tool that returns the path, and its
    # own message already says how many runs were not shown.
    rows = [{k: v for k, v in run.items() if k != "source_path"} for run in found]
    return {
        "message": message,
        "runs": rows,
        "families": shown,
        "n_families": len(families),
    }


@tool(
    empty={
        "family": None,
        "metric": None,
        "arms": [],
        "winner": None,
        # These four are as load-bearing as the ranking. SKILL.md tells an
        # agent that a comparison with no caveats has earned that silence, so
        # `caveats` missing on failure would be indistinguishable from a clean
        # comparison. Declared here so failure and success have one shape.
        "caveats": [],
        "direction": None,
        "corpus": None,
        "without_metric": [],
    },
    label="runs_compare",
)
def _compare(conn, family: str, metric: str | None = None, project: str | None = None) -> dict:
    try:
        result = ledger.compare(conn, family, metric=metric, project=project)
    except ValueError as exc:
        # an undeclared metric direction is a caller-fixable problem, so the
        # reason is surfaced rather than flattened to "internal error"
        raise ToolError(str(exc)) from exc
    if not result["arms"]:
        # ledger.compare already built the recovery message for this case --
        # it distinguishes "no such family" from "you named a project, not a
        # family" and lists what IS comparable. An earlier version of this
        # wrapper replaced it with a bare "no runs in family 'x'", throwing
        # away the one thing that tells the caller what to do next.
        raise ToolError(result.get("message") or f"no runs in family {family!r}")
    arms = result["arms"]
    total_arms = len(arms)
    result = {**result, "arms": arms[:MAX_ARMS_SHOWN], "n_arms": total_arms}
    message = f"{total_arms} arm(s)"
    if total_arms > MAX_ARMS_SHOWN:
        message += f"; showing the {MAX_ARMS_SHOWN} best"
    return {"message": message, **result}


@tool(empty={"run": None}, label="runs_detail")
def _detail(conn, project: str, name: str) -> dict:
    found = ledger.detail(conn, project, name)
    if found is None:
        raise ToolError(f"no run {name!r} in project {project!r}")

    metrics = found["metrics"]
    total = len(metrics)

    # Collapse a step SERIES to its last value, keeping every distinct metric.
    # Slicing the first N rows instead returned `b_g` forty times -- ledger
    # orders by (metric, step) -- while 32 other metrics vanished from a run
    # whose shape is exactly those 33 names. A caller reading one run wants to
    # know what was measured; runs.compare is where a series belongs.
    distinct = ledger.collapse_to_last(metrics)
    collapsed = distinct[:MAX_METRIC_ROWS]

    found = {**found, "metrics": collapsed, "n_metrics": total}
    message = f"{total} metric row(s)"
    if total > len(collapsed):
        message += (
            f"; showing the last of {len(collapsed)} distinct metric(s)"
            f" of {len(distinct)} -- use runs.compare for a series"
            if len(distinct) > len(collapsed)
            else f"; showing the last value of each of {len(collapsed)} metric(s)"
        )
    return {"message": message, "run": found}


# Re-exported: the claim tools moved to claims_tools.py but are registered
# here, and test_architecture requires every tool's implementation to be
# importable from the module that registers it.
from attestation.mcp.claims_tools import _check, _coverage  # noqa: E402
