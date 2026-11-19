"""The Hugging Face Trainer callback, on real (tiny, CPU) training runs.

Skips without `attestation-schema[hf]`, unless ATTESTATION_REQUIRE_INTEGRATIONS=1
(CI's schema-integrations job), where a missing framework fails. Spec:
docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

from __future__ import annotations

from pathlib import Path

import attestation_schema as s
import pytest
from attestation_schema import conformance
from integration_fixtures import (
    DATASET_ID,
    SPLIT,
    assert_conforms_and_links,
    assert_outputs_are_the_files,
    load_rows,
    make_dataset,
    require,
)

torch = require("torch")
transformers = require("transformers")
from attestation_schema.integrations.huggingface import AttestationCallback  # noqa: E402

nn = torch.nn


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2, 1)

    def forward(self, x, labels=None):
        y = self.l(x).squeeze(-1)
        return {"loss": ((y - labels) ** 2).mean(), "logits": y}


class Rows(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = torch.tensor(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return {"x": self.xs[i], "labels": self.ys[i]}


class Boom(RuntimeError):
    pass


class _RaiseAt(transformers.TrainerCallback):
    def __init__(self, step: int, exc: BaseException):
        self.step, self.exc = step, exc

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            raise self.exc


def _train(tmp_path: Path, cb: AttestationCallback, *extra, out: str = "out", **overrides):
    ds = tmp_path / "datasets" / DATASET_ID / "1"
    train, evals = Rows(*load_rows(ds, "train")), Rows(*load_rows(ds, "eval"))
    kwargs = dict(
        max_steps=6,
        per_device_train_batch_size=4,
        learning_rate=0.05,
        logging_steps=1,
        save_steps=2,
        eval_strategy="steps",
        eval_steps=3,
        report_to=[],
        use_cpu=True,
        seed=11,
        data_seed=5,
        metric_for_best_model="loss",
        disable_tqdm=True,
    )
    kwargs.update(overrides)
    args = transformers.TrainingArguments(str(tmp_path / out), **kwargs)
    trainer = transformers.Trainer(
        Linear(), args, train_dataset=train, eval_dataset=evals, callbacks=[cb, *extra]
    )
    trainer.train()
    return trainer


@pytest.fixture
def dataset(tmp_path: Path) -> s.DatasetRecord:
    return make_dataset(tmp_path / "datasets" / DATASET_ID / "1")


def test_a_training_run_writes_a_conforming_record_linked_to_its_dataset(tmp_path, dataset):
    ds_dir = tmp_path / "datasets" / DATASET_ID / "1"
    with AttestationCallback(datasets=[(ds_dir, SPLIT)], family="toy", arm="lr=0.05") as cb:
        _train(tmp_path, cb)
    out = tmp_path / "out"
    assert cb.run_dir == out / "attestation" / f"run-{cb.recorder.run_id}"
    record = assert_conforms_and_links(cb.run_dir, dataset)
    assert (record.status, record.family, record.arm) == ("complete", "toy", "lr=0.05")
    assert record.producer.name == "attestation-schema hf-trainer"
    assert record.started_at <= record.finished_at

    # seeds: what the Trainer applied, under TrainingArguments' own names
    assert record.seeds == {"seed": 11, "data_seed": 5}
    # config: TrainingArguments verbatim (HF's own serialization)
    assert record.config["learning_rate"] == 0.05 and record.config["max_steps"] == 6
    assert record.config["hub_token"] == "<HUB_TOKEN>"

    # checkpoints: every file of every checkpoint this run saved, by digest
    outputs = assert_outputs_are_the_files(record, out)
    folders = {d.name.split("/")[0] for d in outputs}
    assert folders == {"checkpoint-2", "checkpoint-4", "checkpoint-6"}
    assert any(d.name.endswith("model.safetensors") for d in outputs)

    # metrics: final values at their step; curves in the sidecar
    final = {m.name: m for m in record.metrics}
    curves = {c.name: c for c in s.read_history(cb.run_dir)}
    assert len(curves["loss"].points) == 6
    assert [p.step for p in curves["eval_loss"].points] == [3, 6]
    for name, m in final.items():
        assert (m.value, m.step) == (curves[name].points[-1].value, curves[name].points[-1].step)
    # direction only where the user declared it (metric_for_best_model)
    assert final["eval_loss"].direction == "min"
    assert {m.name for m in record.metrics if m.direction} == {"eval_loss"}

    # dependencies: what the process imported, with provenance and source
    deps = {d.name: d for d in record.dependencies}
    assert {"python", "torch", "transformers", "accelerate"} <= set(deps)
    assert deps["torch"].version == torch.__version__
    assert deps["transformers"].version == transformers.__version__
    for d in deps.values():
        assert d.provenance in ("verified", "inferred")
    assert record.ext["hf-trainer"]["world_size"] == 1
    assert record.ext["hf-trainer"]["global_step"] == 6


def test_no_dataset_given_names_the_data_without_a_digest(tmp_path, dataset):
    with AttestationCallback() as cb:
        _train(tmp_path, cb)
    record = s.read(cb.run_dir)
    named = [(d.name, d.sha256, d.role) for d in record.data if d.role in ("train", "eval")]
    assert named == [("train", None, "train"), ("eval", None, "eval")]
    lineage = s.resolve_lineage(record, [dataset])
    assert lineage.edges == () and lineage.issues == ()  # not linked, and not guessed


def test_rotated_checkpoints_are_named_not_bound(tmp_path, dataset):
    with AttestationCallback(datasets=[dataset]) as cb:
        _train(tmp_path, cb, save_total_limit=1, metric_for_best_model=None)
    record = assert_conforms_and_links_content_only(cb.run_dir, dataset)
    folders = {d.name.split("/")[0] for d in record.data if d.role == "output"}
    assert folders == {"checkpoint-6"}
    assert record.ext["hf-trainer"]["checkpoints_removed"] == ["checkpoint-2", "checkpoint-4"]


def assert_conforms_and_links_content_only(run_dir, dataset):
    assert conformance.check_run_dir(run_dir).ok
    record = s.read(run_dir)
    lineage = s.resolve_lineage(record, [dataset])
    assert lineage.verified and [e.via for e in lineage.edges] == ["content"]
    return record


def test_a_run_that_raises_writes_a_failed_record(tmp_path, dataset):
    ds_dir = tmp_path / "datasets" / DATASET_ID / "1"
    with pytest.raises(Boom):
        with AttestationCallback(datasets=[(ds_dir, SPLIT)]) as cb:
            _train(tmp_path, cb, _RaiseAt(3, Boom("diverged at step 3")))
    record = assert_conforms_and_links(cb.run_dir, dataset)
    assert record.status == "failed"
    assert record.ext["hf-trainer"]["error"] == {"type": "Boom", "message": "diverged at step 3"}
    assert {d.name.split("/")[0] for d in record.data if d.role == "output"} == {"checkpoint-2"}
    # the curve so far is kept: steps 1-2 (step 3 raised in on_step_end, before its log)
    assert [p.step for p in s.read_history(cb.run_dir)[0].points] == [1, 2]


def test_an_interrupt_writes_a_cancelled_record(tmp_path, dataset):
    with pytest.raises(KeyboardInterrupt):
        with AttestationCallback() as cb:
            _train(tmp_path, cb, _RaiseAt(2, KeyboardInterrupt()))
    assert s.read(cb.run_dir).status == "cancelled"


def test_only_global_rank_zero_writes(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(transformers.Trainer, "is_world_process_zero", lambda self: False)
    with AttestationCallback() as cb:
        _train(tmp_path, cb)
    assert cb.run_dir is None and cb.record_path is None
    assert not (tmp_path / "out" / "attestation").exists()


def test_a_run_directory_that_holds_a_record_is_refused_before_training(tmp_path, dataset):
    with AttestationCallback(run_dir=tmp_path / "run") as cb:
        _train(tmp_path, cb)
    with pytest.raises(FileExistsError):
        with AttestationCallback(run_dir=tmp_path / "run") as again:
            _train(tmp_path, again, out="out2")
    assert again.recorder.started_at is None
