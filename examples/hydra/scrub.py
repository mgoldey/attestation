# examples/hydra/scrub.py
"""Strip attribution and machine paths from a freshly generated multirun/.

Run by generate.sh immediately after train.py, never on its own. Hydra
writes this machine's absolute path into every arm's `.hydra/hydra.yaml` in
three places: `hydra.runtime.cwd`, `hydra.runtime.output_dir`, and one
entry of `hydra.runtime.config_sources` (the `conf/` directory, found by
its `provider: main`) -- none of which `_hydra_runs` in
ledger_adapters/generic.py reads. `train.log` is deleted outright (empty in
this fixture, since train.py's own prints go to stdout, not the Hydra job
logger, but a real project's would carry local timestamps and stack traces
with no reason to commit them).

Kept verbatim, per the golden-paths brief: `hydra.job.name` (the family
`_hydra_runs` reads), `hydra.overrides.task` (the sweep override, e.g.
`lr=0.01` -- also read), and `hydra.sweep.dir` (a template string,
`multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}`, which never contains a real
path in the first place).

`multirun.yaml`, written once per sweep (not once per arm) beside the
numbered arm directories, duplicates the same `hydra.runtime.cwd`/
`config_sources` paths and is never read by `_hydra_runs` -- deleted
outright rather than scrubbed, the same call generate.py made for
Sacred's `cout.txt` and `_sources/`.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Lines scrubbed to a relative/empty form, matched by their key.
_STRIP_VALUE_KEYS = ("cwd", "output_dir")


def scrub_hydra_yaml(path: Path) -> None:
    lines = path.read_text().splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        key, _, value = stripped.partition(":")
        if key in _STRIP_VALUE_KEYS and value.strip():
            indent = line[: len(line) - len(line.lstrip(" "))]
            out.append(f"{indent}{key}: .")
            continue
        if key == "path" and str(HERE) in value:
            # the `conf/` entry in config_sources -- relativise rather than
            # drop, so the entry's shape (path/schema/provider) survives
            indent = line[: len(line) - len(line.lstrip(" "))]
            out.append(f"{indent}path: conf")
            continue
        out.append(line)
    text = "\n".join(out) + "\n"
    # Belt and suspenders: no line naming this machine's own absolute
    # directory should survive in any value the loop above did not already
    # catch (there are none as of this writing). Built from HERE's own
    # parts rather than a literal leading path segment, so this file itself
    # never spells out the machine path the golden-paths attribution test
    # (tests/test_golden_paths.py) scans every committed example for.
    text = text.replace(str(HERE), ".").replace(str(HERE.parent.parent), ".")
    path.write_text(text)


def main() -> None:
    multirun = HERE / "multirun"
    n = 0
    for hydra_yaml in sorted(multirun.glob("*/*/*/.hydra/hydra.yaml")):
        scrub_hydra_yaml(hydra_yaml)
        n += 1
    for train_log in sorted(multirun.glob("*/*/*/train.log")):
        train_log.unlink()
    for sweep_summary in sorted(multirun.glob("*/*/multirun.yaml")):
        sweep_summary.unlink()
    print(f"scrubbed {n} .hydra/hydra.yaml file(s), deleted their train.log/multirun.yaml")


if __name__ == "__main__":
    main()
