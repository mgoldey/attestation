"""Build and execute the attestation-provenance demo notebooks.

    uv run python scripts/build_skill_notebooks.py

Cell SOURCES live here; cell OUTPUTS are never written by hand -- each notebook
is run top to bottom in a fresh kernel (nbclient) and saved with whatever that
run printed. RESULTS.md is then rendered from those saved outputs, so the
notebook, the table people read without opening it, and the run all agree.

Churn is designed out: cell ids are fixed, timing metadata is off, the kernel's
Python version is not recorded, and the notebooks print no paths, user names or
times. Re-running on an unchanged repo rewrites byte-identical files.
"""

# ruff: noqa: E501  (the notebook cell sources below are data: long markdown and JSON lines)
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "attestation" / "skills" / "attestation-provenance" / "notebooks"

SETUP_BASE = '''\
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Everything this notebook writes goes to a throwaway folder and a throwaway
# ledger. Your real database and workspace are never read or changed.
WORK = Path(tempfile.mkdtemp(prefix="attest-demo-"))
os.environ["ATTEST_DB"] = str(WORK / "demo.db")
HOME = WORK / "hermes"
os.environ["HERMES_HOME"] = str(HOME)
# Point these at the temp folder rather than just unsetting them: every `attest`
# call reloads a checkout's .env, which would put an unset variable back.
os.environ["LEDGER_METRIC_DIRECTION_FILE"] = str(HOME / "metric_direction.toml")
os.environ["LEDGER_CORPUS_FILE"] = str(HOME / "corpus.toml")
os.environ["RESEARCH_ROOT"] = str(WORK / "workspace")


def _find_attest():
    found = shutil.which("attest", path=str(Path(sys.executable).parent)) or shutil.which("attest")
    if found is None:
        raise RuntimeError(
            "`attest` is not on PATH: run this notebook in the attestation environment"
        )
    return found


_ATTEST = _find_attest()


def attest(command, cwd):
    """Run `attest <command>` offline in `cwd`, print what it prints, return its exit code."""
    print("$ attest " + command)
    done = subprocess.run(
        [_ATTEST, *shlex.split(command)], cwd=cwd, capture_output=True, text=True, timeout=60
    )
    text = (done.stdout + done.stderr).rstrip()
    print(text.replace(str(WORK), "<tmp>").replace(str(Path.home()), "~"))
    print(f"[exit {done.returncode}]")
    return done.returncode
'''

SETUP_EXAMPLE = '''\


def _run(tag, wer, val_loss):
    record = {"tag": tag, "corpus": "librispeech-100h", "wer": wer, "val_loss": val_loss}
    return json.dumps({**record, "n_records": 2620})


FALLBACK = {
    "speech-distill/results/kdsweep_baseline.json": _run("kdsweep_baseline", 0.0731, 2.4),
    "speech-distill/results/kdsweep_t2.json": _run("kdsweep_t2", 0.0688, 2.3),
    "speech-distill/results/kdsweep_t4.json": _run("kdsweep_t4", 0.0642, 2.18),
    "speech-distill/results/kdsweep_t4b.json": _run("kdsweep_t4b", 0.0659, 2.2),
    "speech-distill/FINDINGS.md": """# Distillation notes

The baseline reaches a word error rate of 0.0731.
<!-- claim: speech-distill/kdsweep_baseline metric=wer value=0.0731 -->

Temperature 4 is better, at 0.0642.
<!-- claim: speech-distill/kdsweep_t4 metric=wer value=0.0642 -->

Temperature 2 gives 0.0701 (the run records a different number).
<!-- claim: speech-distill/kdsweep_t2 metric=wer value=0.0701 -->

Temperature 8 gives 0.0600 (there is no such run).
<!-- claim: speech-distill/kdsweep_t8 metric=wer value=0.0600 -->

A claim with no metric at all.
<!-- claim: speech-distill/kdsweep_baseline value=0.0731 -->

Validation loss ended at 2.18 and the model has 41.3M parameters.
""",
}


def find_example_workspace():
    """examples/workspace from a repo checkout, or None. ATTEST_EXAMPLE_WORKSPACE
    points at one explicitly; the value `none` skips the search."""
    override = os.environ.get("ATTEST_EXAMPLE_WORKSPACE")
    if override == "none":
        return None
    if override:
        return Path(override)
    import attestation

    for base in (Path.cwd(), Path(attestation.__file__).resolve().parent):
        for folder in (base, *base.parents):
            candidate = folder / "examples" / "workspace"
            if (candidate / "speech-distill" / "FINDINGS.md").is_file():
                return candidate
    return None


WORKSPACE = WORK / "workspace"
SOURCE = find_example_workspace()
if SOURCE is not None:
    shutil.copytree(SOURCE, WORKSPACE)
    print("Using the example workspace from the attestation repo (copied to a temporary folder).")
else:
    for name, text in FALLBACK.items():
        target = WORKSPACE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print("The repo's examples/workspace is not on this machine, so this notebook")
    print("wrote a small stand-in with the same shape (also in a temporary folder).")
'''

# --------------------------------------------------------------------------
# notebook 1: check a draft's claims
# --------------------------------------------------------------------------
CHECK = [
    (
        "md",
        """# Check a draft's claims

A claim is an HTML comment beside the prose it describes. `attest claims`
re-derives the number from the experiment runs on disk and gives a verdict per
claim. This notebook checks the worked example `speech-distill/FINDINGS.md`,
in which three claims are **deliberately wrong**, and shows what each kind of
miss means.

It runs offline (no model server) against a temporary ledger.

Copy this notebook before editing it: the copy that ships with the skill is
overwritten whenever attestation updates it.""",
    ),
    ("code", SETUP_BASE + SETUP_EXAMPLE),
    (
        "md",
        """## 1. Read the runs into the ledger

Claims are checked against the ledger, so it must hold the runs first. Against
an empty ledger every claim would come back `unsupported`, which would mean
"nothing was scanned", not "the draft is wrong".""",
    ),
    ("code", 'attest("runs scan --root .", cwd=WORKSPACE)'),
    (
        "md",
        """## 2. Check every claim in the draft

The same check from Python, as a table: one row per claim.""",
    ),
    (
        "code",
        """\
import contextlib

from IPython.display import Markdown, display

from attestation import claims
from attestation.db import get_db

os.chdir(WORKSPACE)  # the ledger stores source paths relative to the scan root
with contextlib.closing(get_db(Path(os.environ["ATTEST_DB"]))) as conn:
    result = claims.check(conn, WORKSPACE / "speech-distill" / "FINDINGS.md")

rows = ["| line | claim | verdict | why |", "|---:|---|---|---|"]
for v in result["verdicts"]:
    claim = f"`{v.claim.run} {v.claim.metric}={v.claim.value:g}`"
    rows.append(f"| {v.claim.line} | {claim} | **{v.verdict}** | {v.message} |")
for problem in result["malformed"]:
    where, _, reason = problem.rpartition(": ")
    rows.append(f"| {where.rpartition(':')[2]} | (cannot be read) | **malformed** | {reason} |")
display(Markdown("\\n".join(rows)))
counts = {k.value: n for k, n in result["counts"].items()}
print(counts, "plus", len(result["malformed"]), "malformed")""",
    ),
    (
        "md",
        """## 3. What the three misses mean

They are different problems and need different fixes. Never read one as the other.

| verdict | what it says | what you do |
|---|---|---|
| `contradicted` | a run **disagrees** with the number in the prose | the draft or the run is wrong; correct one of them |
| `unsupported` | **no run matches**: the run name does not exist (as here), or the run has no such metric | the claim may be true, but nothing backs it; find or record the run. It does not mean false |
| `malformed` | the annotation **cannot be read** (here: no `metric` field) | fix the annotation; it is reported rather than skipped so a claim cannot vanish from review |

`supported` means a run agrees within tolerance. Three more verdicts exist and
this draft has none of them: `ambiguous` (a wildcard matched several runs, or
one run holds the metric at several splits or steps and the claim does not say
which: add `split=` or `step=`), `stale` (the value matches a run whose file
changed after `as_of`) and `uncited` (a claim with `cite=` whose key no
configured bibliography source has).""",
    ),
    (
        "md",
        """## 4. The same check from the command line

`attest claims` exits 1 when a claim is contradicted, so it can gate a commit.
The exit code below is the check working, not a crash.""",
    ),
    ("code", 'attest("claims speech-distill/FINDINGS.md", cwd=WORKSPACE)'),
    (
        "md",
        """## 5. Numbers no claim covers

A draft with zero contradicted claims can still assert numbers nothing backs.
`--coverage` lists them.""",
    ),
    ("code", 'attest("claims speech-distill/FINDINGS.md --coverage", cwd=WORKSPACE)'),
]

# --------------------------------------------------------------------------
# notebook 2: which arm won?
# --------------------------------------------------------------------------
ARM = [
    (
        "md",
        """# Which arm won?

`attest runs compare` ranks the arms of a sweep and then says why you may not
trust the ranking. **The caveats are the product, not a disclaimer**: a winner
reported without them misrepresents what was found. This notebook compares the
`kdsweep` distillation sweep and shows its caveats exactly as the tool printed
them.

It runs offline (no model server) against a temporary ledger.

Copy this notebook before editing it: the copy that ships with the skill is
overwritten whenever attestation updates it.""",
    ),
    ("code", SETUP_BASE + SETUP_EXAMPLE),
    (
        "md",
        """## 1. Read the runs into the ledger

A *family* is a shared filename prefix (`kdsweep_t2`, `kdsweep_t4` ... form the
family `kdsweep`), not a project. `runs list` shows the families that exist.""",
    ),
    (
        "code",
        'attest("runs scan --root .", cwd=WORKSPACE)\nattest("runs list", cwd=WORKSPACE)',
    ),
    (
        "md",
        """## 2. Compare by a named metric

Word error rate is better when **lower**. The ledger knows that direction, so
it ranks. Pass the metric explicitly: with none, it falls back to whichever
metric most arms share, which can answer a different question than you asked.""",
    ),
    ("code", 'attest("runs compare kdsweep --metric wer", cwd=WORKSPACE)'),
    (
        "md",
        """## 3. Read the caveats, not just the winner

The two lines starting `caveat:` above are verbatim from the tool. In words:

- **Too close to call.** The top two arms differ by 0.0017 (2.6%). In this
  sweep `kdsweep_t4b` is `kdsweep_t4` at a different seed, so 0.0017 is
  also the size of one seed's swing.
- **No seed replication.** Each arm is one run, so the ranking cannot separate
  the configuration from luck.

The tool's caveat is about the top two arms. Beyond them it says only that
every arm is a single run; it does not rank the gaps further down. A
comparison whose margin is no bigger than its seed variance has not found
anything, so the honest summary is "t4 is nominally ahead and the tool calls
it too close to call", not "t4 won".""",
    ),
    (
        "md",
        """## 4. When it refuses

The ledger **refuses to rank a metric whose direction nobody declared**, rather
than guess: ranking WER as if higher were better would name the worst arm the
winner. `n_records` is a count of records, not a quality score, so nobody has
declared which way is better.""",
    ),
    (
        "code",
        'attest("runs compare kdsweep --metric n_records", cwd=WORKSPACE)',
    ),
    (
        "md",
        """The refusal is the right answer (here the file it names sits in a temporary
home; on your machine it lives in your Hermes home, by default
`~/.hermes/metric_direction.toml`). The fix is a
`[metric_direction]` entry made by the person who knows which way is better, never a guess by the tool or
by an agent reporting it.""",
    ),
]

# --------------------------------------------------------------------------
# notebook 3: make your own claim
# --------------------------------------------------------------------------
OWN = [
    (
        "md",
        """# Make your own claim

Write a tiny results file and a one-claim Markdown note, check it, then change
the number and watch the verdict turn `contradicted`. Nothing here needs the
repo's examples or a model server; it writes everything to a temporary folder
and a temporary ledger.

Copy this notebook before editing it: the copy that ships with the skill is
overwritten whenever attestation updates it.""",
    ),
    ("code", SETUP_BASE + '\n\nprint("Working in a temporary folder with a temporary ledger.")\n'),
    (
        "md",
        """## 1. A run, and a sentence that cites it

The run is a JSON file under `results/`. The ledger reads files like this where
they already are; nothing is registered in advance. The claim names the run
(`project/run`), the metric and the value, and sits in a comment beside the
prose so the note renders unchanged.""",
    ),
    (
        "code",
        """\
OWN = WORK / "own"
(OWN / "mywork" / "results").mkdir(parents=True)
(OWN / "mywork" / "results" / "run_a.json").write_text(
    json.dumps({"tag": "run_a", "corpus": "demo", "accuracy": 0.912, "n_records": 500})
)
note = OWN / "mywork" / "notes.md"
note.write_text(
    "Run A reaches an accuracy of 0.912.\\n"
    "<!-- claim: mywork/run_a metric=accuracy value=0.912 tol=0.001 -->\\n"
)
print(note.read_text())""",
    ),
    ("md", "## 2. Scan, then check"),
    (
        "code",
        'attest("runs scan --root .", cwd=OWN)\nattest("claims mywork/notes.md", cwd=OWN)',
    ),
    (
        "md",
        """## 3. Change the number

Suppose the note is edited to say 0.931 and nobody re-ran anything. The run on
disk still records 0.912.""",
    ),
    (
        "code",
        """\
note.write_text(note.read_text().replace("0.912 tol", "0.931 tol").replace("of 0.912", "of 0.931"))
print(note.read_text())
attest("claims mywork/notes.md", cwd=OWN)""",
    ),
    (
        "md",
        """`contradicted`, and the message names both numbers: the note says 0.931, the
run records 0.912. Either the note or the run is wrong, and the check cannot
tell you which; that is yours to decide. Exit code 1 means the check worked.""",
    ),
    (
        "md",
        """## 4. A claim with nothing behind it

Point the claim at a run that does not exist and the verdict is different:
`unsupported` is "nothing here backs this", not "false".""",
    ),
    (
        "code",
        """\
note.write_text(note.read_text().replace("mywork/run_a", "mywork/run_b").replace("0.931", "0.912"))
attest("claims mywork/notes.md", cwd=OWN)""",
    ),
]

NOTEBOOKS = {
    "check-a-drafts-claims.ipynb": CHECK,
    "which-arm-won.ipynb": ARM,
    "make-your-own-claim.ipynb": OWN,
}


def build(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    for i, (kind, text) in enumerate(cells, 1):
        cell = (
            nbformat.v4.new_markdown_cell(text)
            if kind == "md"
            else nbformat.v4.new_code_cell(text.rstrip("\n"))
        )
        cell["id"] = f"c{i:02d}"
        nb.cells.append(cell)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def execute(nb: nbformat.NotebookNode, cwd: Path) -> None:
    client = NotebookClient(
        nb,
        timeout=60,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(cwd)}},
    )
    client.execute()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    for cell in nb.cells:
        cell.metadata.pop("execution", None)


def render_results() -> str:
    """RESULTS.md: the executed outputs of the saved notebooks, nothing retyped."""
    parts = [
        "# Results of the executed demos\n",
        "Rendered by `scripts/build_skill_notebooks.py` from the outputs saved in the",
        "notebooks next to this file, from a run against the repo's `examples/workspace`.",
        "Nothing below was typed by hand. Open the `.ipynb` files to see the prose around it.\n",
    ]
    for name in NOTEBOOKS:
        nb = nbformat.read(OUT / name, as_version=4)
        title = nb.cells[0].source.splitlines()[0].lstrip("# ")
        parts.append(f"## {title}\n\n`{name}`\n")
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            text = ""
            for out in [*cell.get("outputs", []), None]:
                if out is not None and out.output_type == "stream":
                    text += out.text
                    continue
                if text.strip():
                    parts.append("```text\n" + text.rstrip() + "\n```\n")
                text = ""
                if (
                    out is not None
                    and out.output_type == "display_data"
                    and "text/markdown" in out.data
                ):
                    parts.append(out.data["text/markdown"].rstrip() + "\n")
    return "\n".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = build(cells)
        execute(nb, OUT)
        nbformat.validate(nb)
        nbformat.write(nb, OUT / name)
        print(f"wrote {name}")
    (OUT / "RESULTS.md").write_text(render_results())
    print("wrote RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
