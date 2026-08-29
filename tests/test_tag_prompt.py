"""The tagging prompt is data: one renderer, an optional artifact, no dspy in src.

Before this, the prompt lived in three places -- `features._tag_prompt`, the
eval's `build_messages`, and whatever an optimizer would emit -- and a prompt
measured in one could differ from the one shipped in another. `tag_messages`
is the only renderer; the eval, the transfer test, and `attest tag` all call
it, so a score is always a score OF the prompt that runs.
"""

import json
import re
from pathlib import Path

import pytest
from conftest import seeded_db

from attestation.features import (
    DEFAULT_TAG_INSTRUCTION,
    TagPrompt,
    load_tag_prompt,
    run_tagging,
    tag_messages,
    tag_prompt_from_env,
)

VOCAB = ["quantum-chemistry", "transformers"]


SHIPPED = Path(__file__).resolve().parents[1] / "evals" / "prompts" / "tagging-2026-08-27.json"


def test_the_default_is_the_shipped_artifact_verbatim():
    """The default is embedded in features.py because evals/ is not in the
    wheel; this pins the embedded text to the artifact whose transfer matrix
    justified it, so neither can drift from the other."""
    msgs = tag_messages("T", "S", VOCAB)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == DEFAULT_TAG_INSTRUCTION
    shipped = json.loads(SHIPPED.read_text())
    assert shipped["shipped"] is True
    assert DEFAULT_TAG_INSTRUCTION == shipped["instruction"]
    assert (
        msgs[1]["content"]
        == "Existing vocabulary: quantum-chemistry, transformers\n\nTitle: T\nSummary: S"
    )


def test_no_vocabulary_renders_a_placeholder_not_an_empty_list():
    assert "Existing vocabulary: (none yet)" in tag_messages("T", "S", [])[1]["content"]


def test_the_summary_is_truncated_to_a_thousand_characters():
    msgs = tag_messages("T", "x" * 5000, VOCAB)
    assert msgs[1]["content"].endswith("Summary: " + "x" * 1000)


def test_an_artifact_replaces_the_instruction_and_renders_demos_as_turns():
    prompt = TagPrompt(
        instruction="Label it.",
        demos=(
            {
                "title": "Demo",
                "summary": "About dft",
                "content_type": "paper",
                "tags": ["quantum-chemistry"],
            },
        ),
    )
    msgs = tag_messages("T", "S", VOCAB, prompt=prompt)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "Label it."
    assert msgs[1]["content"] == "Title: Demo\nSummary: About dft"
    assert json.loads(msgs[2]["content"]) == {
        "content_type": "paper",
        "tags": ["quantum-chemistry"],
    }
    # The vocabulary rides on the real query only: repeating 150 tags per demo
    # would spend more prompt on the examples than on the item.
    assert msgs[3]["content"].startswith("Existing vocabulary: quantum-chemistry")


def test_an_instruction_only_artifact_renders_exactly_like_the_default_shape():
    prompt = TagPrompt(instruction="Label it.")
    msgs = tag_messages("T", "S", VOCAB, prompt=prompt)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1] == tag_messages("T", "S", VOCAB)[1]


def write_artifact(path: Path, **overrides) -> Path:
    body = {"instruction": "Label it.", "demos": []}
    body.update(overrides)
    path.write_text(json.dumps(body))
    return path


def test_load_tag_prompt_reads_instruction_and_demos(tmp_path):
    demo = {"title": "D", "summary": "s", "content_type": "blog", "tags": ["cuda"]}
    path = write_artifact(tmp_path / "p.json", demos=[demo], scores={"dev": 0.9})
    prompt = load_tag_prompt(path)
    assert prompt.instruction == "Label it."
    assert prompt.demos == (demo,)
    assert prompt.source == str(path)


@pytest.mark.parametrize(
    "bad",
    [
        {"instruction": ""},
        {"instruction": 3},
        {"demos": [{"title": "D", "summary": "s", "content_type": "paper"}]},  # no tags
        {"demos": [{"title": "D", "summary": "s", "content_type": "poem", "tags": ["x"]}]},
        {"demos": [{"title": "D", "summary": "s", "content_type": "paper", "tags": ["arxiv"]}]},
        {"demos": "not a list"},
    ],
    ids=["empty", "not-a-string", "demo-without-tags", "bad-content-type", "non-topic", "shape"],
)
def test_load_tag_prompt_rejects_a_malformed_artifact(tmp_path, bad):
    path = write_artifact(tmp_path / "p.json", **bad)
    with pytest.raises(ValueError):
        load_tag_prompt(path)


def test_tag_prompt_from_env_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("ATTEST_TAG_PROMPT", raising=False)
    assert tag_prompt_from_env() is None


def test_run_tagging_uses_the_artifact_named_by_the_environment(tmp_path, monkeypatch):
    path = write_artifact(tmp_path / "p.json", instruction="FROM ARTIFACT")
    monkeypatch.setenv("ATTEST_TAG_PROMPT", str(path))
    conn = seeded_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, published, content_hash)"
        " VALUES (NULL, 'T', 'http://x', 'S', datetime('now'), 'h')"
    )
    seen = []

    def chat_fn(messages, schema):
        seen.append(messages)
        return {"content_type": "paper", "tags": ["dft"]}

    stats = run_tagging(conn, chat_fn=chat_fn)
    assert stats["tagged"] == 1
    assert seen[0][0]["content"] == "FROM ARTIFACT"
    # A run must say which prompt produced its tags, the way it says which model.
    assert stats["prompt"] == str(path)


def test_run_tagging_reports_the_default_prompt_when_none_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("ATTEST_TAG_PROMPT", raising=False)
    conn = seeded_db(tmp_path / "t.db")
    stats = run_tagging(conn, chat_fn=lambda m, s: {"content_type": "paper", "tags": ["x"]})
    assert stats["prompt"] == "default"


def test_run_tagging_refuses_a_broken_artifact_before_touching_the_model(tmp_path, monkeypatch):
    path = write_artifact(tmp_path / "p.json", instruction="")
    monkeypatch.setenv("ATTEST_TAG_PROMPT", str(path))
    conn = seeded_db(tmp_path / "t.db")
    calls = []
    with pytest.raises(ValueError):
        run_tagging(conn, chat_fn=lambda m, s: calls.append(1))
    assert calls == []


def test_dspy_never_enters_the_library():
    """The optimizer is an offline dev tool; `attest tag` must run without it.
    The spec's refusal condition: if dspy cannot be kept out of the runtime
    import path, the design is abandoned.

    mlflow, wandb and sacred are the same kind of optional dependency
    (examples/flows/training/train_mlflow.py, examples/wandb/generate.py,
    examples/sacred/generate.py): the ledger READS mlruns/, wandb/ and
    sacred_runs/ directories (ledger_adapters/generic.py's `_mlflow_runs`,
    `_wandb_runs` and `_sacred_runs`, named as such in comments and the
    `adapter="mlflow"`/`adapter="wandb"`/`adapter="sacred"` labels) without
    ever importing the libraries that write them, so this checks for an
    import, not the word."""
    src = Path(__file__).resolve().parents[1] / "src" / "attestation"
    files = list(src.rglob("*.py"))
    for name in ("dspy", "mlflow", "wandb", "sacred"):
        pattern = re.compile(rf"^\s*(import|from)\s+{name}\b", re.MULTILINE)
        offenders = [str(p.relative_to(src)) for p in files if pattern.search(p.read_text())]
        assert offenders == [], f"{name} imported under src/: {offenders}"


def test_dspy_stays_confined_to_the_optimizer_under_evals():
    """`evals/` is not a package and every script there imports its peers
    top-level (see tagging_eval.py's own docstring), so a reaction or
    explanation eval could import dspy by accident -- e.g. copying a helper
    out of optimize_tagging.py -- without src/ ever seeing it. Only the
    optimizer itself may import it."""
    evals_dir = Path(__file__).resolve().parents[1] / "evals"
    pattern = re.compile(r"^\s*(import|from)\s+dspy\b", re.MULTILINE)
    offenders = [
        str(p.relative_to(evals_dir))
        for p in evals_dir.rglob("*.py")
        if p.name != "optimize_tagging.py" and pattern.search(p.read_text())
    ]
    assert offenders == [], f"dspy imported outside optimize_tagging.py: {offenders}"
