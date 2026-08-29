# Golden Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every documented use case becomes a directory under `examples/<name>/` with the same shape (a seven-section README, a `run.sh` that runs the README's own commands, inputs on disk, a test, a catalogue row), the four existing paths are retrofitted, and six third-party integrations (W&B, Sacred, DVC, Hydra, BibTeX citations, non-Ollama model servers) get worked examples whose artifacts come from the real library.

**Architecture:** One test module, `tests/test_golden_paths.py`, discovers every `examples/*/README.md` and enforces the shape (headings, README⇔`run.sh` agreement, prerequisite label, attribution scrub, and — for `none`-prerequisite paths — runs `run.sh` and pins the first line of the README's "What it prints" block). `examples/README.md` is the catalogue. New ledger conventions go into `ledger_adapters/generic.py` as `_sacred_runs`, `_dvc_runs`, `_hydra_runs`, each called from `discover()`, each with fixture-tolerance tests.

**Tech Stack:** Python ≥3.12, bash, `wandb` / `sacred` / `dvc` / `hydra-core` (ephemeral, `uv run --with`, never dependencies), the existing `examples/flows/stub_openai.py`.

**Spec:** `docs/superpowers/specs/2026-08-28-golden-paths-design.md`

## Global Constraints

- Prerequisite labels are exactly one of: `none — pure local computation`, `a model server at LLM_BASE_URL`, `network`.
- README sections, in this order, as `## ` headings: `What you get`, `Prerequisites`, `Run it`, `What it prints`, `What it demonstrates`, `When it goes wrong`, `Next`.
- Every fenced line in the README's `Run it` block that starts with `uv run`, `attest`, `./run.sh`, `export ` or `ATTEST_` must appear verbatim in the path's `run.sh`.
- `run.sh`: `#!/usr/bin/env bash`, `set -euo pipefail`, `cd "$(dirname "$0")"`, `export ATTEST_DB="$(mktemp -d)/attest.db"` (unless the path is about an existing DB), the README's commands, exit 0 when green.
- No committed file under `examples/**` contains `/home/`, a username, `github.com`, `mlflow.user`, or an SSH remote; generated artifacts come from a committed `generate.py`/`generate.sh` that runs the real library and scrubs.
- Nothing under `src/` imports wandb, sacred, dvc, hydra, or mlflow (`tests/test_tag_prompt.py`'s guard gains those names).
- New ledger conventions: final values not curves; no metric-direction inference; `NAMED` stays empty; one function per convention called from `discover()`; participate in `seen` dedup; a shape-tolerance test; the reader docstring names the library version that produced the committed fixture.
- No new `# noqa: BLE001` under `src/`; complexity ratchet holds; line length 100; `*.md` is excluded from ruff.
- Gates after `git add`: `uv run --frozen pre-commit run --all-files`, reading per-hook `Passed/Failed` lines. New test files and new `examples/*` directories go into CLAUDE.md's docs index (an architecture test enforces it).
- Commit by explicit pathspec; message style: a plain sentence, blank line, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Never rewrite `examples/README.md` wholesale — add your row with an exact-string edit, because other tasks add rows concurrently.
- No renames of existing example directories; `evals/` stays where it is.

---

### Task 1: The framework, the catalogue, and the two retrofits that need no new content

**Files:**
- Create: `tests/test_golden_paths.py`
- Create: `examples/workspace/run.sh`, `examples/flows/run.sh`
- Modify: `examples/workspace/README.md` (restructure into the seven sections; keep its prose), `examples/flows/README.md` (same), `examples/README.md` (becomes the catalogue), `CLAUDE.md` (docs index: `test_golden_paths.py`)

**Interfaces:**
- Produces: `test_golden_paths.paths() -> list[Path]` (every `examples/*/README.md` except the catalogue), `test_golden_paths.sections(readme_text) -> list[str]`, `test_golden_paths.prerequisite(readme_text) -> str`, `test_golden_paths.run_commands(readme_text) -> list[str]`, `test_golden_paths.pinned_line(readme_text) -> str` (first non-empty line of the fenced block under `## What it prints`), `test_golden_paths.catalogue_rows() -> dict[str, str]` (path name → prerequisite label from the table in `examples/README.md`). Later tasks add directories; this module discovers them with no edits.

- [ ] **Step 1: Write the failing framework test**

```python
# tests/test_golden_paths.py
"""Every golden path has the same shape, and the documented commands are the
tested commands. Discovery is by directory: adding examples/<name>/README.md
enrols a path in every check here, with no edit to this file.

Spec: docs/superpowers/specs/2026-08-28-golden-paths-design.md."""

import os
import re
import subprocess
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
CATALOGUE = EXAMPLES / "README.md"
SECTIONS = [
    "What you get", "Prerequisites", "Run it", "What it prints",
    "What it demonstrates", "When it goes wrong", "Next",
]
LABELS = {"none — pure local computation", "a model server at LLM_BASE_URL", "network"}
COMMAND_PREFIXES = ("uv run", "attest", "./run.sh", "export ", "ATTEST_")
FORBIDDEN = ("/home/", "github.com", "mlflow.user", "git@")
TEXT_SUFFIXES = {".md", ".sh", ".py", ".toml", ".json", ".yaml", ".yml", ".xml", ".txt", ".bib", ".csv", ".lock", ""}


def paths() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*/README.md"))


def sections(text: str) -> list[str]:
    return re.findall(r"^## (.+?)\s*$", text, re.M)


def _section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def prerequisite(text: str) -> str:
    body = _section(text, "Prerequisites").strip().splitlines()
    return body[0].strip().strip("`") if body else ""


def _fenced(block: str) -> list[str]:
    out, inside = [], False
    for line in block.splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        if inside:
            out.append(line.rstrip())
    return out


def run_commands(text: str) -> list[str]:
    return [l.strip() for l in _fenced(_section(text, "Run it")) if l.strip().startswith(COMMAND_PREFIXES)]


def pinned_line(text: str) -> str:
    lines = [l for l in _fenced(_section(text, "What it prints")) if l.strip()]
    return lines[0].strip() if lines else ""


def catalogue_rows() -> dict[str, str]:
    rows = {}
    for line in CATALOGUE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].startswith("`") and cells[0].endswith("/`"):
            rows[cells[0].strip("`").rstrip("/")] = cells[2].strip("`")
    return rows


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_seven_sections_in_order(readme):
    assert sections(readme.read_text()) == SECTIONS, readme


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_prerequisite_is_one_of_three_honest_labels(readme):
    assert prerequisite(readme.read_text()) in LABELS, readme


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_readme_commands_are_the_run_sh_commands(readme):
    script = readme.parent / "run.sh"
    assert script.is_file() and os.access(script, os.X_OK), f"{script} missing or not executable"
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    commands = run_commands(readme.read_text())
    assert commands, f"{readme}: Run it has no commands"
    for cmd in commands:
        assert cmd in text, f"{readme.parent.name}: README command not in run.sh: {cmd!r}"


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_every_path_has_a_catalogue_row_with_the_same_label(readme):
    rows = catalogue_rows()
    name = readme.parent.name
    assert name in rows, f"{name} has no row in examples/README.md"
    assert rows[name] == prerequisite(readme.read_text()), name


def test_the_catalogue_lists_only_paths_that_exist():
    for name in catalogue_rows():
        assert (EXAMPLES / name / "README.md").is_file(), name


def test_no_committed_example_carries_attribution_or_machine_paths():
    user = os.environ.get("USER", "")
    hits = []
    for f in EXAMPLES.rglob("*"):
        if not f.is_file() or f.suffix not in TEXT_SUFFIXES or "__pycache__" in f.parts:
            continue
        text = f.read_text(errors="replace")
        for needle in FORBIDDEN + ((user,) if len(user) >= 4 else ()):
            if needle in text:
                hits.append(f"{f.relative_to(EXAMPLES)}: {needle}")
    assert not hits, "\n".join(hits)


def _offline(readme: Path) -> bool:
    return prerequisite(readme.read_text()).startswith("none")


@pytest.mark.parametrize("readme", [p for p in paths() if _offline(p)], ids=lambda p: p.parent.name)
def test_an_offline_path_runs_green_and_prints_its_pinned_line(readme, tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "LLM_BASE_URL": "http://127.0.0.1:9/v1"}
    for var in ("ATTEST_DB", "RSS_DB", "RESEARCH_ROOT"):
        env.pop(var, None)
    proc = subprocess.run(
        [str(readme.parent / "run.sh")], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, f"{readme.parent.name} run.sh failed:\n{proc.stdout}\n{proc.stderr}"
    pin = pinned_line(readme.read_text())
    assert pin, f"{readme.parent.name}: What it prints has no fenced line to pin"
    assert pin in proc.stdout, f"{readme.parent.name}: pinned line not in output: {pin!r}"
```

Run: `uv run --frozen pytest tests/test_golden_paths.py -q`
Expected: FAIL for `workspace` and `flows` (no seven sections, no `run.sh`, no catalogue rows).

- [ ] **Step 2: Retrofit `examples/workspace/`**

Rewrite `examples/workspace/README.md` into the seven sections, keeping every paragraph of the current walkthrough (they become *What it demonstrates*). *Prerequisites*: `none — pure local computation`. *Run it*:

```bash
export RESEARCH_ROOT=$PWD
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare kdsweep --metric wer
uv run attest claims speech-distill/FINDINGS.md || true
uv run attest claims speech-distill/FINDINGS.md --coverage
```

(`|| true` because `attest claims` exits 1 on the deliberately contradicted claim, and `run.sh` is `set -e`; say so in *When it goes wrong*.) *What it prints*: the real output of those commands, abridged; its first fenced line is what the test pins — choose a stable one such as `winner: kdsweep_t4`. Note that `run.sh` runs from the path's own directory (`cd "$(dirname "$0")"`), so paths in commands are relative to `examples/workspace`; the repo-root README quickstart keeps its own `--root examples/workspace` form and `tests/test_examples.py::test_the_readme_quickstart_runs_without_a_model_server` still pins that.

`examples/workspace/run.sh`:

```bash
#!/usr/bin/env bash
# The commands examples/workspace/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
export RESEARCH_ROOT=$PWD
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare kdsweep --metric wer
uv run attest claims speech-distill/FINDINGS.md || true
uv run attest claims speech-distill/FINDINGS.md --coverage
```

`chmod +x`. Run it by hand; paste the abridged output into *What it prints*.

- [ ] **Step 3: Retrofit `examples/flows/`**

Restructure `examples/flows/README.md` into the seven sections (its current content maps onto *What you get*, *What it demonstrates*, *Next*). *Prerequisites*: `none — pure local computation` (the offline mode; say that `--live` needs Ollama under *When it goes wrong* / *Next*). *Run it*: `uv run --group examples python run_all.py --offline` (relative to the directory). `run.sh` runs exactly that. *What it prints*: the summary block; pin `mcp_e2e        ok` or the `=== summary` line — whichever is byte-stable.

- [ ] **Step 4: The catalogue**

Replace `examples/README.md` with an index. Keep a two-sentence intro, then the table:

```markdown
# Golden paths

Each directory here is a use case you can run from a clean clone: a README with
the same seven sections, a `run.sh` that runs the README's own commands, and
its inputs on disk. `tests/test_golden_paths.py` runs every path whose
prerequisite is `none` and pins one line of its output, so the docs are what
the suite asserts.

| path | what it shows | prerequisite | runtime |
|---|---|---|---|
| `workspace/` | the run ledger and claim checker over a two-project workspace with three claims deliberately wrong | `none — pure local computation` | ~1 s |
| `flows/` | forty labelled items scored for two personas, every MCP tool driven over stdio, four MLflow arms read back by the ledger | `none — pure local computation` | ~75 s |
```

Ordered by prerequisite (`none` first), then alphabetically. Later tasks add rows with exact-string edits.

- [ ] **Step 5: Run the framework tests, the gates, commit**

Run: `uv run --frozen pytest tests/test_golden_paths.py tests/test_examples.py -q` → all pass (workspace and flows run green; pinned lines found). Add `test_golden_paths.py` to CLAUDE.md's index. `git add`, gates, commit:

```bash
git commit -m "Golden paths have one shape now: seven sections, a run.sh that runs the README's commands, and a test that runs the offline ones" -- tests/test_golden_paths.py examples/README.md examples/workspace examples/flows/README.md examples/flows/run.sh CLAUDE.md
```

---

### Shared recipe for Tasks 2–8 (each task's brief includes this)

For a path `examples/<name>/`:

1. Write `README.md` with the seven sections; the `Prerequisites` first line is the label; *Run it* holds the commands; *What it prints* holds real, abridged output whose first fenced line is stable.
2. Write `run.sh` per the global constraints (executable). Run it by hand; the README's output block comes from that run.
3. If the path has generated artifacts, write `generate.py` (or `.sh`) that runs the real library via `uv run --with <lib> --no-project python generate.py` (or the library's CLI), writes into the path's directory, scrubs attribution and machine paths (delete or rewrite: usernames, `/home/...`, hostnames, git remotes, absolute script paths, timestamps that embed the host), and prints what it wrote. Commit the artifacts. The README says regeneration is deliberate.
4. Add the catalogue row to `examples/README.md` with an exact-string edit (insert after the last row of the same prerequisite group; never rewrite the file).
5. Add the directory to CLAUDE.md's docs index; run `uv run --frozen pytest tests/test_golden_paths.py -q` (your path is auto-discovered), then the gates after `git add`, then commit by pathspec.
6. If the path finds a bug in `src/` (a reader that misses the real layout, a tool that fails on real input), fix it in its own commit with its own failing-then-passing test in the reader's/tool's test module, and say so in the README's *What it demonstrates*.

---

### Task 2: `examples/prompt-evals/` and `examples/agents/`

**Files:** Create `examples/prompt-evals/{README.md,run.sh}`, `examples/agents/{README.md,run.sh}`; modify `examples/README.md`, `CLAUDE.md`.

**prompt-evals** — *Prerequisites*: `a model server at LLM_BASE_URL`. *Run it* (relative to repo root, so `run.sh` does `cd "$(dirname "$0")/../.."`):
```bash
uv run python evals/run_tagging_eval.py --split dev
uv run python evals/transfer_matrix.py --artifact evals/prompts/tagging-2026-08-27.json
```
*What you get*: a score for the shipping tagging prompt and the transfer gate that decides whether a candidate may replace it. *What it demonstrates*: the README § "Prompt evals and the optimizer" paragraphs (move the substance here; leave a two-line pointer in the repo README). *When it goes wrong*: Ollama down (`chat model unreachable`), a model not pulled, the optimizer's group not installed (`uv run --group optimize`). *What it prints*: the eval's per-case lines and summary from a real run — run it once against Ollama (check `nvidia-smi` first; ~2 min) to capture it. This path is not executed by the test (label is not `none`); only the shape and command agreement are checked.

**agents** — *Prerequisites*: `none — pure local computation`. *Run it*:
```bash
uv run attest install --check || true
uv run attest emit
ATTEST_TOOLS=provenance ATTEST_EXPAND=1 uv run python ../flows/mcp_e2e.py --surface provenance --offline
```
(`|| true` because `install --check` exits nonzero with a model server down — the README explains each `[BROKEN]` line as the honest report it is.) *What you get*: what an agent sees: the doctor's report, the generated per-surface agent configs, and one surface driven over stdio the way an agent drives it. *What it demonstrates*: the four surfaces (`AGENT_SURFACES`), progressive disclosure (two tools without `ATTEST_EXPAND`), why `attest emit` never overwrites, `attest reload` and the stale-server problem. *When it goes wrong*: `ATTEST_TOOLS` typo raises; a server running stale code; `hermes mcp test` not catching staleness. *Next*: README § "Launching alongside hermes-agent". Pin a stable line from the `mcp_e2e` matrix (e.g. `provenance  runs.scan`… — check the exact rendering) or `mcp surfaces:`/`no agent binary found` from `attest emit`.

Commit the two paths separately.

---

### Task 3: `examples/citations/`

**Files:** Create `examples/citations/{README.md,run.sh,references.bib,DRAFT.md}`; modify `examples/README.md`, `CLAUDE.md`.

Hand-write `references.bib` with four entries (real, well-known works with correct DOIs are fine — cite them accurately; e.g. Vaswani 2017, Hohenberg–Kohn 1964, Kingma–Ba 2015, Hinton 2015 distillation) keyed `vaswani2017attention` etc. Write `DRAFT.md` with claims carrying `cite=` fields — read `src/attestation/claims.py` for the exact field syntax (`<!-- claim: <project>/<run> metric=… value=… cite=<key> -->`) — where three keys resolve and one (`doe2099imaginary`) does not, plus a copy of `examples/workspace/speech-distill/results/` runs? No: point `RESEARCH_ROOT` at `../workspace` and reference `speech-distill/kdsweep_t4` metrics so the numeric claims are `supported` while the citation lint is what varies. *Prerequisites*: `none — pure local computation`. *Run it* (from the path's directory so `Path.cwd().glob("*.bib")` finds the file — `Resolver.from_env` reads the cwd):
```bash
export RESEARCH_ROOT=$PWD/../workspace
uv run attest runs scan --root ../workspace
uv run attest claims DRAFT.md || true
```
Check what `attest claims` prints for `cite=` keys (`VerdictKind.UNCITED` per CLAUDE.md) and whether a CLI flag exists for citation sources; if the CLI surfaces nothing for citations, drive `cite.sources` / `cite.check` / `cite.lookup` through `examples/flows/mcp_e2e.py`'s spawning pattern instead (a small `check_citations.py` in the path using `mcp.client.stdio` with `ATTEST_TOOLS=knowledge`), and record in *What it demonstrates* that the CLI has no citation command — that is a finding, not a defect to fix here. *What it demonstrates*: the lint semantics ("no source has this key", never "the paper does not support this"), `offline: true` from `cite.sources`, `ATTEST_CITATION_WEB` read at construction. Pin the uncited key's line.

---

### Task 4: `examples/model-servers/`

**Files:** Create `examples/model-servers/{README.md,run.sh}`; modify `examples/README.md`, `CLAUDE.md`.

*What you get*: attestation against any OpenAI-compatible server — the only contract is `POST /v1/chat/completions` with `response_format.json_schema` and `POST /v1/embeddings`. *Prerequisites*: `none — pure local computation` (the runnable example uses `examples/flows/stub_openai.py`). *Run it*:
```bash
uv run python ../flows/stub_openai.py > "$TMPDIR_URL" &
```
— no: background processes in `run.sh` are fragile. Instead write a 30-line `with_server.py` in the path that starts the stub in-process, exports `LLM_BASE_URL`/`CHAT_MODEL`/`EMBED_MODEL`, and runs `attest ingest --feeds ../flows/corpus/feeds.toml`-equivalent (use `_common.write_feeds_toml` from `../flows/_common.py`) then `attest tag --limit 5` as subprocesses, and shuts the stub down. *Run it*: `uv run python with_server.py`. *What it demonstrates*: `LLM_BASE_URL` semantics; a table for vLLM (`http://host:8000/v1`, `--served-model-name`), llama.cpp server (`http://host:8080/v1`), LM Studio (`http://localhost:1234/v1`), Ollama (`/v1` suffix required); `EMBED_DIMS` must match the embedding model and changing it invalidates every stored vector (`get_db` refuses); `reasoning_effort` is sent and a 400 is retried without it. *When it goes wrong*: `embedding model unreachable`, a server that rejects `response_format`, dims mismatch. Pin the ingest stats line.

---

### Task 5: `examples/wandb/`

**Files:** Create `examples/wandb/{README.md,run.sh,generate.py}` and the committed `wandb/` output; possibly modify `src/attestation/ledger_adapters/generic.py` + `tests/test_tracker_adapters.py`; modify `tests/test_tag_prompt.py` (guard gains `wandb`), `examples/README.md`, `CLAUDE.md`, the tracker spec's Status line and the reader docstring (W&B now verified against a real directory).

`generate.py`: `WANDB_MODE=offline`, `wandb.init(project="flows", name="lr_sweep", config={...})` four times with different `lr`, `wandb.log({"train_loss": …}, step=s)` for ten steps, `wandb.summary` set to final `accuracy`/`auc` (compute with sklearn on `load_breast_cancer` like `examples/flows/training/train_mlflow.py`), `wandb.finish()`. Run via `uv run --with wandb --with scikit-learn --no-project python generate.py`. Inspect what it wrote: the run directories are `wandb/offline-run-<ts>-<id>/` (not `run-*`) and `files/` holds `wandb-summary.json`, `config.yaml`, `wandb-metadata.json`, plus `.wandb` binary logs, `logs/`, `tmp/`. Scrub: delete `*.wandb`, `logs/`, `tmp/`, `files/requirements.txt`, `files/output.log`; in `wandb-metadata.json` remove `host`, `username`, `executable`, `root`, `git`, `email`, `codePath` absolute paths; keep `program`, `startedAt`, `args`, `python`. `README.md` *Prerequisites*: `none — pure local computation`; *Run it*: `uv run attest runs scan --root .` / `uv run attest runs list` / `uv run attest runs compare lr_sweep --metric auc`.

Expected finding: `_wandb_runs` (`generic.py:551+`, docstring says `wandb/run-<timestamp>-<id>/files/`) does not see `offline-run-*` directories → the scan reports the project empty. Confirm with the real directory; then fix the reader to accept both `run-*` and `offline-run-*` with a test in `tests/test_tracker_adapters.py` built from the committed fixture, in its own commit, and say so in *What it demonstrates*. Also verify the run name derivation (`program + id`) against a real `wandb-metadata.json`. Retire the "W&B never run against a real directory" caveats (reader docstring, test module docstring, spec Status). Pin `winner:` line.

---

### Task 6: `examples/sacred/` and `_sacred_runs`

**Files:** Create `examples/sacred/{README.md,run.sh,generate.py}` + committed `sacred_runs/` output; modify `src/attestation/ledger_adapters/generic.py` (new `_sacred_runs(root, seen)`, `TRACKER_DIRS` gains the directory name, `discover()` calls it), `tests/test_tracker_adapters.py` (fixture-shape + tolerance tests), `tests/test_tag_prompt.py` (guard gains `sacred`), `docs/superpowers/specs/2026-08-22-tracker-adapters-design.md` (a dated "Sacred" subsection), `examples/README.md`, `CLAUDE.md`.

`generate.py`: a Sacred `Experiment("lr_sweep")` with `FileStorageObserver("sacred_runs")`, config `lr`, `seed`; `_run.log_scalar("train_loss", v, step)` ten times and `_run.log_scalar("auc", final)`; run four times with `ex.run(config_updates={"lr": …})`. Run via `uv run --with sacred --with scikit-learn --no-project python generate.py`. Layout to read (verify against the real output): `sacred_runs/<n>/{config.json, run.json, metrics.json, cout.txt, info.json?}` and `sacred_runs/_sources/`. Scrub: delete `cout.txt` (host output), `_sources/`, and in `run.json` remove `host` (hostname, python path, os, cpu, gpus), `meta.command` absolute paths, `experiment.sources`/`base_dir`, `experiment.repositories`; keep `experiment.name`, `status`, `start_time`, `stop_time`, `result`. Reader: one run per numbered directory whose `run.json` `status == "COMPLETED"`; name `<experiment.name>/<n>`, family `experiment.name`; config from `config.json` (drop `seed`? no — keep all keys, `_SKIP_KEYS` handles metric keys only); metrics from `metrics.json` (`{name: {"steps": [...], "values": [...]}}`) — final value and its last step; `result` in `run.json`, if numeric, as metric `result`. Tolerance: missing `metrics.json` → config-only record with no metrics is skipped like MLflow's; missing `config.json` → no config. Docstring: "run against a real directory produced by sacred <version> on 2026-08-28 (`examples/sacred/generate.py`)". Pin `winner:`.

---

### Task 7: `examples/dvc/` and `_dvc_runs`

**Files:** Create `examples/dvc/{README.md,run.sh,generate.sh,dvc.yaml,params.yaml,train.py}` + committed `dvc.lock` and metrics files; modify `generic.py` (`_dvc_runs`), `tests/test_tracker_adapters.py`, `tests/test_tag_prompt.py` (guard gains `dvc`), the tracker spec, `examples/README.md`, `CLAUDE.md`.

`dvc.yaml`: four stages `train@0.01`, `train@0.1`, `train@1`, `train@10` via a `foreach` over `lr`, each `cmd: python train.py ${item}`, `params: [lr]`, `metrics: [metrics/${item}.json]`. `train.py` writes `metrics/<lr>.json` with `accuracy`, `precision`, `recall`, `auc` (sklearn, breast-cancer, like the MLflow example). `generate.sh`: `uv run --with dvc --with scikit-learn --no-project bash -c 'dvc init --no-scm -q && dvc repro -q'` from the path directory, then `rm -rf .dvc/cache .dvc/tmp` (keep `.dvc/config` if it is small and path-free; else delete `.dvc/` and note that `dvc.yaml` + `dvc.lock` are the record). Verify the layout `dvc repro` produced, especially `dvc.lock`'s per-stage `params` and `metrics` entries. Reader: parse `dvc.yaml` (YAML) for stages declaring `metrics:`; expand `foreach` stages by their `do:`; for each stage read the metric files it lists (JSON, `metrics_from_payload`), config from `params.yaml` keys the stage lists under `params:` plus `dvc.lock`'s recorded values; name `<stage>` (`train@0.1`), family = the stage's base name before `@`. Tolerance: no `dvc.lock` → still read metric files that exist; a listed metric file missing → fewer metrics. Pin `winner:`.

---

### Task 8: `examples/hydra/` and `_hydra_runs`

**Files:** Create `examples/hydra/{README.md,run.sh,generate.sh,train.py,conf/config.yaml}` + committed `multirun/` output; modify `generic.py` (`_hydra_runs`), `tests/test_tracker_adapters.py`, `tests/test_tag_prompt.py` (guard gains `hydra`), the tracker spec, `examples/README.md`, `CLAUDE.md`.

`train.py`: `@hydra.main(config_path="conf", config_name="config", version_base=None)`; trains the same sklearn model with `cfg.lr`; writes `metrics.json` into `os.getcwd()` (Hydra changes cwd per job). `generate.sh`: `uv run --with hydra-core --with scikit-learn --no-project python train.py --multirun lr=0.01,0.1,1,10` from the path directory; Hydra writes `multirun/<YYYY-MM-DD>/<HH-MM-SS>/<n>/{.hydra/{config.yaml,hydra.yaml,overrides.yaml}, metrics.json, train.log}`. Scrub: delete `train.log`; in `.hydra/hydra.yaml` remove `hydra.runtime.cwd`, `hydra.runtime.output_dir`, `hydra.job.config_name` paths and any `/home` occurrences (rewrite to relative); keep `hydra.job.name`, `hydra.overrides.task`, `hydra.sweep.dir`. Reader: sweep root `multirun/*/*/` — each numbered child with a `.hydra/config.yaml` is an arm; name `<job.name>/<date>/<time>/<n>` is unreadable, so name `<job.name>/<n>` with family `job.name` (from `.hydra/hydra.yaml`, fall back to the directory name); config from `.hydra/config.yaml`; metrics from any JSON/CSV file in the arm directory via the existing `metrics_from_payload` readers; two sweeps of the same job dedupe by `seen` (document that only the latest sweep is read? No — read all; name collisions are qualified with the time directory). Tolerance: no `hydra.yaml` → family from the sweep directory; no metrics file → skipped. Pin `winner:`.

---

### Task 9: Front door and closing

**Files:** Modify `README.md` (one line after "Try it in 60 seconds" → the catalogue; shorten "Prompt evals and the optimizer" to a pointer at `examples/prompt-evals/`), `CLAUDE.md` (Key API Patterns: one line on golden paths — the seven sections, README⇔run.sh agreement enforced by `test_golden_paths.py`, the three labels, the attribution guard now covering all of `examples/`), `docs/superpowers/specs/2026-08-28-golden-paths-design.md` (Status: implemented, with any deviations — the wandb `offline-run-*` finding, whatever DVC/Hydra needed).

Run the whole suite and the gates; run every offline `run.sh` once more by hand and record the measured runtimes in the catalogue's `runtime` column. Commit. Push (the user authorised pushing when green) and watch CI.

---

## Self-review

**Spec coverage.** Shape + test + catalogue (Task 1); retrofits: workspace and flows (Task 1), prompt-evals and agents (Task 2); integrations: citations (3), model servers (4), wandb (5), sacred (6), dvc (7), hydra (8); front door and status (9). Convention rules (final values, no direction inference, `NAMED` empty, own function, tolerance test, docstring names the library version) appear in Tasks 6–8 and the global constraints. Attribution guard generalised to `examples/**` (Task 1). `src/` import guard extended (Tasks 5–8).

**Placeholders.** The framework test is complete code; each path task names its files, commands, the reader's exact inputs and naming rule, its scrub list and its pinned line. Real-library layouts are stated as expectations to *verify against the real output*, which is the point of generating them.

**Type consistency.** `prerequisite()` returns the label string the catalogue rows compare against; `pinned_line()` reads the same *What it prints* block every path writes; `run_commands()` prefixes match the global constraint list.
