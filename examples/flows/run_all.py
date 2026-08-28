"""Run every example flow and print one summary.

    uv run --group examples python examples/flows/run_all.py --offline   # CI: stub model
    uv run --group examples python examples/flows/run_all.py --live --write-results

Order: training first (needs no model), then the persona eval, then the
MCP end-to-end. Each flow writes a JSON report; --write-results renders
RESULTS.md from LIVE reports only -- offline numbers are about the stub.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common

FLOWS = ("train_mlflow", "persona_eval", "mcp_e2e")
SCRIPT = {
    "train_mlflow": _common.FLOWS_DIR / "training" / "train_mlflow.py",
    "persona_eval": _common.FLOWS_DIR / "persona_eval.py",
    "mcp_e2e": _common.FLOWS_DIR / "mcp_e2e.py",
}


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _persona_section(pe: dict) -> list[str]:
    out = [
        f"## Persona eval (mode=live, chat={pe['chat_model']}, "
        f"embed={pe['embed_model']}, items={pe['items']})",
        "",
        "Agreement with `corpus/labels.json`; evidence about the flow, not a model benchmark.",
        "",
        "| persona | precision | recall | AUC (signed confidence) | tp/fp/fn/tn | unsure "
        "| rank AUC | classifier AUC | s/reaction |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in pe["personas"]:
        r, k = p["reactions"], p["ranker"]
        out.append(
            f"| {p['persona']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} |"
            f" {_fmt(r['auc'])} | {r['tp']}/{r['fp']}/{r['fn']}/{r['tn']} | {r['n_unsure']} |"
            f" {_fmt(k['rank_auc'])} | {_fmt(k['classifier_auc'])}"
            f" ({k['classifier_n_clicks']} clicks) | {p['seconds_per_reaction']:.2f} |"
        )
    histograms = "; ".join(
        f"{p['persona']} {p['reactions']['confidence_histogram']}" for p in pe["personas"]
    )
    return [*out, "", "Confidence histograms: " + histograms, ""]


def _mcp_section(me: dict) -> list[str]:
    rows = me["rows"]
    failed = [r for r in rows if r["problem"]]
    out = [
        f"## MCP end to end (mode=live, chat={me['chat_model']})",
        "",
        f"{len(rows)} calls over stdio across feed / provenance / knowledge / symbolic / full;"
        f" {len(failed)} failed.",
        "",
        "| surface | tool | result |",
        "|---|---|---|",
    ]
    for r in rows:
        if r["problem"]:
            result = "FAILED: " + r["problem"]
        else:
            result = "refused" if r["ok"] is False else "ok"
        out.append(f"| {r['surface']} | {r['tool']} | {result} |")
    return [*out, ""]


def _training_section(tr: dict) -> list[str]:
    out = [
        f"## Training family `c_sweep` (mlflow-skinny, {tr['seconds']:.1f} s"
        f" for {len(tr['arms'])} arms)",
        "",
        "| C | accuracy | precision | recall | AUC |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {a['C']} | {a['accuracy']:.4f} | {a['precision']:.4f} |"
        f" {a['recall']:.4f} | {a['auc']:.4f} |"
        for a in tr["arms"]
    ]
    return [*out, ""]


def render_results(reports: dict[str, dict], when: str) -> str:
    modes = {r.get("mode") for r in reports.values() if "mode" in r}
    if modes != {"live"}:
        raise ValueError(
            f"RESULTS.md records live numbers only; got modes {sorted(modes)} (offline is the stub)"
        )
    out = [f"# Example flows: results measured {when}", ""]
    if pe := reports.get("persona_eval"):
        out += _persona_section(pe)
    if me := reports.get("mcp_e2e"):
        out += _mcp_section(me)
    if tr := reports.get("train_mlflow"):
        out += _training_section(tr)
    return "\n".join(out)


def _run(name: str, mode_flag: list[str], json_path: Path) -> tuple[int, dict]:
    cmd = [sys.executable, str(SCRIPT[name]), *mode_flag, "--json", str(json_path)]
    print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    report = json.loads(json_path.read_text()) if json_path.exists() else {}
    return rc, report


def _flags(name: str, offline: bool, tmp: str) -> list[str]:
    """train_mlflow needs no model but must not rewrite the committed mlruns/."""
    if name == "train_mlflow":
        out = Path(tmp) / "training"
        out.mkdir()
        return ["--out", str(out)]
    return ["--offline"] if offline else []


def main(argv: list[str] | None = None) -> int:
    assert __doc__ is not None
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--write-results", action="store_true")
    ap.add_argument("--skip", action="append", default=[], choices=FLOWS)
    args = ap.parse_args(argv)

    reports: dict[str, dict] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in FLOWS:
            if name in args.skip:
                continue
            rc, report = _run(name, _flags(name, args.offline, tmp), Path(tmp) / f"{name}.json")
            reports[name] = report
            if rc != 0:
                failures.append(name)
    print("\n=== summary")
    for name in FLOWS:
        if name in reports:
            print(f"{name:<14} {'FAILED' if name in failures else 'ok'}")
    if args.write_results and not failures:
        text = render_results(reports, when=dt.date.today().isoformat())
        (_common.FLOWS_DIR / "RESULTS.md").write_text(text)
        print(f"wrote {_common.FLOWS_DIR / 'RESULTS.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
