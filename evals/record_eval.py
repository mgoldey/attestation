"""Scoring for the attestation-record skill's output, model-free.

A trial in this eval is a scenario -- "you just ran <family> with arms
<a>,<b>; the final metrics are <m>=<v1>,<v2>; the corpus was <c>" -- and the
model's answer is a JSON manifest of files to write:

    {"files": {"results/dit-small-rope-a.json": "{...}", "configs/dit-small-rope-a.yaml": "..."}}

`score_one` writes that manifest into a sandbox workspace, points
`ledger._metric_direction_path()` at any `metric_direction.toml` the manifest
itself wrote (via the real `LEDGER_METRIC_DIRECTION_FILE` env var -- the same
lever a user has), then runs the REAL `ledger_adapters.generic.discover` (via
`ledger.scan`) and `ledger.compare` against what landed on disk. Nothing here
re-implements the ledger's rules; the point is to check that the skill's
CONTENT -- where files land, what they contain, whether a direction got
declared -- produces a ledger that reads correctly, using the exact reader
`attest` uses in production.

Checks, matching the spec's (a)-(d):
    (a) scan finds exactly len(arms) runs for the scenario's family
    (b) one family, grouped: compare does not refuse and names the
        scenario's actual winner (computed from arm values + direction)
    (c) a config file landed under a recognised config dir and contributed
        no metric row to the winning run
    (d) for a metric absent from the built-in METRIC_DIRECTION table, the
        manifest declared a direction (no direction refusal) -- vacuously
        true for a scenario whose metric is already built in
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

from attestation import db, ledger

CASES_PATH = pathlib.Path(__file__).parent / "record_cases.json"

# Paths a manifest may not use -- see _safe_relpath's docstring.
_UNSAFE = ("..",)


def load_cases(path: pathlib.Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _safe_relpath(relpath: str) -> pathlib.PurePosixPath:
    """Validate and normalise one manifest path.

    Rejects an absolute path (escapes the sandbox workspace entirely) and any
    `..` segment (escapes it one directory at a time) -- a manifest is meant
    to describe files *inside* the trial's own workspace, and either shape
    is the model writing somewhere the harness never intended to touch.
    """
    p = pathlib.PurePosixPath(relpath)
    if p.is_absolute():
        raise ValueError(f"manifest path {relpath!r} is absolute")
    if any(part in _UNSAFE for part in p.parts):
        raise ValueError(f"manifest path {relpath!r} escapes the workspace")
    return p


def parse_manifest(answer: str | dict) -> dict[str, str]:
    """`{relpath: content}` from the model's answer, a JSON manifest of
    `{"files": {relpath: content}}`. Raises on a malformed manifest or an
    unsafe path -- the caller decides whether that counts as a failed trial."""
    obj = json.loads(answer) if isinstance(answer, str) else answer
    if not isinstance(obj, dict) or "files" not in obj:
        raise ValueError(f"manifest missing 'files': {obj!r}")
    files = obj["files"]
    if not isinstance(files, dict):
        raise ValueError(f"manifest 'files' must be an object, got {type(files).__name__}")
    out: dict[str, str] = {}
    for relpath, content in files.items():
        safe = _safe_relpath(relpath)
        out[str(safe)] = content if isinstance(content, str) else json.dumps(content)
    return out


PROJECT_NAME = "proj"


def write_manifest(workspace: pathlib.Path, files: dict[str, str]) -> None:
    """Materialise a validated manifest under `workspace`.

    Manifest paths are relative to ONE project directory (`results/...`,
    `configs/...`) -- `workspace` here already IS that project directory,
    named `PROJECT_NAME`, and the scan below reads `workspace.parent` with
    `project=PROJECT_NAME`. Keeping the project fixed means a scenario asks
    the model for file layout, not a project name it would then have to
    parrot back correctly.
    """
    for relpath, content in files.items():
        target = workspace / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _expected_winner(scenario: dict) -> str | None:
    """The arm the scenario's own values+direction say should win, independent
    of what the model wrote -- so the "right winner" check is against ground
    truth, not against a self-report."""
    direction = scenario["direction"]
    values = scenario["values"]  # {arm_name: value}
    if not values:
        return None
    pick = min if direction == "lower_is_better" else max
    return pick(values, key=lambda arm: values[arm])


def score_one(scenario: dict, answer: str | dict, *, workspace: pathlib.Path | None = None) -> dict:
    """Score one trial. Returns per-check booleans plus an overall `pass`.

    `workspace` lets a caller supply its own tmp dir (tests reuse `tmp_path`);
    omitted, a private `TemporaryDirectory` is used and cleaned up here.
    """
    result: dict = {
        "id": scenario["id"],
        "errors": [],
        "checks": {
            "manifest_parses": False,
            "scan_count": False,
            "grouped_and_winner": False,
            "config_not_metric": False,
            "direction_declared": False,
        },
    }

    def _run(root: pathlib.Path) -> None:
        try:
            files = parse_manifest(answer)
        except (ValueError, json.JSONDecodeError) as exc:
            result["errors"].append(f"manifest invalid: {exc}")
            return
        result["checks"]["manifest_parses"] = True
        proj_dir = root / PROJECT_NAME
        write_manifest(proj_dir, files)

        direction_file = proj_dir / "metric_direction.toml"
        env_var = ledger.METRIC_DIRECTION_PATH_ENV
        old = os.environ.get(env_var)
        if direction_file.is_file():
            os.environ[env_var] = str(direction_file)
        else:
            # Point at a path that does not exist, inside this trial's own
            # tmp dir -- so a leftover developer ~/.hermes/metric_direction.toml
            # on this machine cannot leak a declaration into the trial and
            # mask a model that forgot to write one.
            os.environ[env_var] = str(root / "absent-metric_direction.toml")
        try:
            conn = db.get_db(root / "eval.db")
            try:
                scan_out = ledger.scan(conn, root, project=PROJECT_NAME)
                n_arms = len(scenario["arms"])
                scanned = scan_out["scanned"].get(PROJECT_NAME, 0)
                if scanned != n_arms:
                    result["errors"].append(
                        f"scan found {scanned} run(s), expected {n_arms}"
                        f" ({scan_out.get('diagnostics', {}).get(PROJECT_NAME, '')})"
                    )
                else:
                    result["checks"]["scan_count"] = True

                try:
                    cmp = ledger.compare(conn, scenario["family"], project=PROJECT_NAME)
                except ValueError as exc:
                    result["errors"].append(f"compare refused: {exc}")
                    cmp = None

                if cmp is not None:
                    expected_winner = _expected_winner(scenario)
                    n_grouped = len(cmp.get("arms", [])) + len(cmp.get("without_metric", []))
                    if n_grouped < n_arms:
                        result["errors"].append(
                            f"compare grouped only {n_grouped} of {n_arms} arms into one family"
                        )
                    elif cmp.get("winner") != expected_winner:
                        result["errors"].append(
                            f"compare named winner {cmp.get('winner')!r},"
                            f" expected {expected_winner!r}"
                        )
                    else:
                        result["checks"]["grouped_and_winner"] = True
                    # Reaching here at all (no ValueError above) for a
                    # not-built-in metric already proves the manifest
                    # declared a direction -- metric_directions() would
                    # otherwise have had nothing and compare() would have
                    # raised before returning `cmp`.
                    result["checks"]["direction_declared"] = True
                elif scenario.get("built_in", True):
                    # A built-in metric refusing is a real scorer failure
                    # (e.g. cross-project family collision), not evidence
                    # about direction declaration -- leave the check false
                    # and let the `compare refused` error above explain it.
                    pass
                else:
                    result["errors"].append(
                        "not-built-in metric was not declared -- compare refused"
                    )

                # (c): a config file landed under a recognised config dir,
                # and no run in this family recorded a metric the scenario
                # never reported -- the signature of a config misread as a
                # result (see generic.py's RESULTS-FIRST-THEN-CONFIGS note).
                config_dirs = _config_dirs()
                config_dir_present = any(
                    pathlib.PurePosixPath(p).parts
                    and pathlib.PurePosixPath(p).parts[0] in config_dirs
                    for p in files
                )
                recorded_metrics = {
                    row["metric"]
                    for row in conn.execute(
                        "SELECT DISTINCT rm.metric FROM run_metrics rm"
                        " JOIN runs r ON r.id = rm.run_id"
                        " WHERE r.project = ? AND r.family = ?",
                        (PROJECT_NAME, scenario["family"]),
                    )
                }
                extra_metrics = recorded_metrics - set(scenario["metrics"])
                if config_dir_present and not extra_metrics:
                    result["checks"]["config_not_metric"] = True
                elif not config_dir_present:
                    result["errors"].append("no config file found under a recognised config dir")
                else:
                    result["errors"].append(
                        f"a config file contributed metric row(s) {sorted(extra_metrics)}"
                    )
            finally:
                conn.close()
        finally:
            if old is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = old

    if workspace is not None:
        _run(workspace)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            _run(pathlib.Path(tmp))

    result["pass"] = all(result["checks"].values())
    return result


def _config_dirs() -> tuple[str, ...]:
    """`CONFIG_DIRS` from the generic adapter, looked up by name so this
    scorer never hardcodes the convention it is checking against."""
    from attestation.ledger_adapters.generic import CONFIG_DIRS

    return CONFIG_DIRS
