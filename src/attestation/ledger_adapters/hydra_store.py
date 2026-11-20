"""Hydra `--multirun` sweeps, read directly and said as a core run record.

MLflow's adapter is the template (`mlflow_store`), but Hydra records much
less: the composed config (`.hydra/config.yaml`), its own runtime config
(`.hydra/hydra.yaml`: the job name), and the overrides that make this job
differ from the base config (`.hydra/overrides.yaml`). So: job name ->
family, the arm's path under `multirun/` -> run_id, the overrides -> arm
(a `seed=<int>` override -> seeds), the whole config, the arm's metrics file
by the existing convention, every output file by sha256. Hydra records no
outcome, no timing, no git and no curve: status, wall time, code and
history are null and the adapter caveat says so.

Spec: docs/superpowers/specs/2026-11-20-wandb-hydra-adapters-design.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import attestation_schema as schema
import yaml

from attestation.ledger_adapters.found import IMPORT_NAMESPACE, Found, core_metrics, json_safe
from attestation.ledger_adapters.generic import _hydra_arm_metrics

BASE_DIR = "multirun"
ADAPTER = "hydra"
HYDRA_DIR = ".hydra"
SEED_KEY = "seed"
_UNKNOWN_CODE = {
    "git_sha": None,
    "dirty": None,
    "repo": None,
    "version": None,
    "provenance": "unknown",
    "evidence": None,
}


@dataclass
class HydraFound(Found):
    """An arm as the legacy readers' `Found` plus what Hydra recorded that a
    `Found` has no field for: its path, its overrides and its outputs."""

    run_id: str = ""
    overrides: list[str] = field(default_factory=list)
    outputs: list[tuple[str, Path]] = field(default_factory=list)
    config_note: str | None = None


def _yaml(path: Path) -> Any:
    """A YAML document, or None when absent or unreadable."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None


def _dig(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _overrides(hydra_dir: Path, hydra_yaml: Any) -> list[str]:
    """The job's overrides: Hydra's own `overrides.yaml`, else the copy in
    `hydra.yaml` (`hydra.overrides.task`)."""
    listed = _yaml(hydra_dir / "overrides.yaml")
    if not isinstance(listed, list):
        listed = _dig(hydra_yaml, "hydra", "overrides", "task")
    return [str(o) for o in listed] if isinstance(listed, list) else []


def _seed_override(override: str) -> int | None:
    """N for `seed=N` (also `+seed=N`/`++seed=N`) with an integer N."""
    key, sep, value = override.partition("=")
    if not sep or key.lstrip("+") != SEED_KEY:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def arm_and_seeds(overrides: list[str], job_name: str | None) -> tuple[str | None, dict | None]:
    """The arm and seeds Hydra's overrides declare.

    The arm is the overrides sorted and joined with `,` -- Hydra's
    `override_dirname` with its default separators -- minus an integer seed
    override, which becomes `{"seed": N}`. Seeds alone name the base
    configuration (the job). No overrides declare no arm."""
    rest: list[str] = []
    seeds: dict | None = None
    for override in overrides:
        seed = _seed_override(override)
        if seed is not None and seeds is None:
            seeds = {SEED_KEY: seed}
        else:
            rest.append(override)
    if rest:
        return ",".join(sorted(rest)), seeds
    return (job_name, seeds) if seeds and job_name else (None, None)


def _outputs(arm_dir: Path) -> list[tuple[str, Path]]:
    """Every file the job wrote into its output directory; `.hydra/` is
    Hydra's record of the job, not an output."""
    out = []
    for dirpath, dirnames, filenames in os.walk(arm_dir):
        dirnames[:] = sorted(d for d in dirnames if Path(dirpath) != arm_dir or d != HYDRA_DIR)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            out.append((path.relative_to(arm_dir).as_posix(), path))
    return sorted(out)


def _arm_dirs(sweep_dir: Path) -> list[Path]:
    arms = (p for p in sweep_dir.iterdir() if (p / HYDRA_DIR / "config.yaml").is_file())
    return sorted(arms, key=lambda p: p.name)


def _family(arm_dir: Path, sweep_dir: Path) -> str:
    """`hydra.job.name` from the first arm's hydra.yaml -- the driver
    script's stem unless a config overrides it -- else the sweep's time
    directory (an older Hydra, or a directory edited by hand)."""
    name = _dig(_yaml(arm_dir / HYDRA_DIR / "hydra.yaml"), "hydra", "job", "name")
    return str(name) if name not in (None, "") else sweep_dir.name


def _read_arm(root: Path, base: Path, arm_dir: Path, family: str) -> HydraFound | None:
    metrics = _hydra_arm_metrics(arm_dir)
    if not metrics:
        return None  # a spec with no measurement attached -- every reader's rule
    hydra_dir = arm_dir / HYDRA_DIR
    config = _yaml(hydra_dir / "config.yaml")
    found = HydraFound(
        project=root.name,
        name="",
        source_path=str(arm_dir),
        family=family,
        status="recorded",
        config=config if isinstance(config, dict) else None,
        metrics=metrics,
        adapter=ADAPTER,
        run_id=arm_dir.relative_to(base).as_posix(),
        overrides=_overrides(hydra_dir, _yaml(hydra_dir / "hydra.yaml")),
        outputs=_outputs(arm_dir),
    )
    if config is not None and not isinstance(config, dict):
        found.config_note = "its .hydra/config.yaml is not a mapping and was not read"
    elif config is None and (hydra_dir / "config.yaml").stat().st_size:
        found.config_note = "its .hydra/config.yaml is not readable YAML and was not read"
    return found


def hydra_runs(root: Path, seen: set[str]) -> list[HydraFound]:
    """Every arm under `multirun/<date>/<time>/<n>/` that measured something.

    Named `<job.name>/<n>`, qualified with the time directory when a second
    sweep of the same job collides (`<job.name>/<time>/<n>`), as the ledger
    has named Hydra arms since the first reader; a name already in `seen`
    is skipped."""
    base = Path(root) / BASE_DIR
    out: list[HydraFound] = []
    for sweep_dir in sorted(p for p in base.glob("*/*") if p.is_dir()):
        arms = _arm_dirs(sweep_dir)
        if not arms:
            continue
        family = _family(arms[0], sweep_dir)
        for arm_dir in arms:
            found = _read_arm(Path(root), base, arm_dir, family)
            if found is None:
                continue
            found.name = f"{family}/{arm_dir.name}"
            if found.name in seen:
                found.name = f"{family}/{sweep_dir.name}/{arm_dir.name}"
            if found.name in seen:
                continue
            seen.add(found.name)
            out.append(found)
    return out


# ---- the mapping ---------------------------------------------------------------


def _run_caveats(found: HydraFound, lossy: list[str], dropped: list[str]) -> list[str]:
    out = [f"{found.name}: {found.config_note}"] if found.config_note else []
    if lossy:
        out.append(
            f"{found.name}: non-finite value(s) stored as strings, not numbers: {', '.join(lossy)}"
        )
    if dropped:
        out.append(
            f"{found.name}: duplicate metric value(s) dropped, the first of each kept: "
            + ", ".join(dropped)
        )
    return out


def to_core_record(
    found: HydraFound, *, caveats: list[str], replicate: tuple[str, int] | None
) -> schema.RunRecord:
    """The core record for one Hydra arm, validated by `from_dict`.

    The arm and seeds are Hydra's overrides (`arm_and_seeds`); `replicate`
    (the ledger-wide `<config>_seed<N>` naming convention) applies only when
    the overrides declare neither."""
    from attestation import __version__

    arm, seeds = arm_and_seeds(found.overrides, found.family)
    if arm is None and seeds is None and replicate:
        arm, seeds = replicate[0], {SEED_KEY: replicate[1]}
    lossy: list[str] = []
    dropped: list[str] = []
    config = json_safe(found.config or {}, "config", lossy)
    metrics = core_metrics(found, dropped)
    return schema.from_dict(
        {
            "schema": schema.SCHEMA,
            "run_id": found.run_id,
            "family": found.family,
            "arm": arm,
            "status": None,
            "started_at": None,
            "finished_at": None,
            "producer": {"name": f"attestation {ADAPTER}-import", "version": __version__},
            "code": dict(_UNKNOWN_CODE),
            "config": config,
            "data": [
                {"name": rel, "sha256": schema.sha256_file(path), "role": "output"}
                for rel, path in found.outputs
            ],
            "metrics": metrics,
            "history": None,
            "seeds": seeds,
            "dependencies": None,
            "ext": {
                IMPORT_NAMESPACE: {
                    "adapter": ADAPTER,
                    "status": found.status,
                    "notes": None,
                    "corpus": None,
                    "caveats": caveats,
                    "run_caveats": _run_caveats(found, lossy, dropped),
                }
            },
        }
    )


__all__ = ["HydraFound", "arm_and_seeds", "hydra_runs", "to_core_record"]
