"""The PyTorch Lightning callback, on real (tiny, CPU) training runs.

Skips without `attestation-schema[lightning]`, unless
ATTESTATION_REQUIRE_INTEGRATIONS=1 (CI's schema-integrations job), where a
missing framework fails. Spec:
docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import attestation_schema as s
import pytest
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
L = require("lightning")
from attestation_schema.integrations.lightning import AttestationCallback  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402


class Linear(L.LightningModule):
    def __init__(self, lr: float = 0.05, fail_at: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.l = torch.nn.Linear(2, 1)

    def _loss(self, batch):
        x, y = batch
        return ((self.l(x).squeeze(-1) - y) ** 2).mean()

    def training_step(self, batch, i):
        if self.hparams.fail_at is not None and self.global_step >= self.hparams.fail_at:
            raise RuntimeError("diverged")
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, i):
        self.log("val_loss", self._loss(batch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


class Interrupting(Linear):
    def training_step(self, batch, i):
        if self.global_step >= 3:
            raise KeyboardInterrupt
        return super().training_step(batch, i)


def _loaders(ds: Path):
    out = []
    for part in ("train", "eval"):
        xs, ys = load_rows(ds, part)
        tds = torch.utils.data.TensorDataset(torch.tensor(xs), torch.tensor(ys))
        out.append(torch.utils.data.DataLoader(tds, batch_size=4))
    return out


def _fit(tmp_path: Path, cb, model=None, *, logger=True, callbacks=()):
    ds = tmp_path / "datasets" / DATASET_ID / "1"
    train, val = _loaders(ds)
    trainer = L.Trainer(
        default_root_dir=str(tmp_path / "out"),
        max_epochs=2,
        log_every_n_steps=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=logger,
        callbacks=[cb, *callbacks],
    )
    trainer.fit(model or Linear(), train, val)
    return trainer


@pytest.fixture
def dataset(tmp_path: Path) -> s.DatasetRecord:
    return make_dataset(tmp_path / "datasets" / DATASET_ID / "1")


@pytest.fixture(autouse=True)
def _no_global_seed(monkeypatch):
    for name in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS"):
        monkeypatch.delenv(name, raising=False)


def test_a_training_run_writes_a_conforming_record_linked_to_its_dataset(tmp_path, dataset):
    L.seed_everything(7, workers=True)
    ds_dir = tmp_path / "datasets" / DATASET_ID / "1"
    cb = AttestationCallback(datasets=[(ds_dir, SPLIT)], family="toy")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True)
    trainer = _fit(tmp_path, cb, callbacks=[checkpoint])
    root = Path(trainer.log_dir)
    assert cb.run_dir == root / "attestation" / f"run-{cb.recorder.run_id}"
    record = assert_conforms_and_links(cb.run_dir, dataset)
    assert (record.status, record.family, record.arm) == ("complete", "toy", None)
    assert record.producer.name == "attestation-schema lightning"

    assert record.seeds == {"seed_everything": 7}
    assert record.ext["lightning"]["seed_workers"] is True
    assert record.config["hparams"] == {"lr": 0.05, "fail_at": None}
    assert record.config["trainer"]["max_epochs"] == 2

    outputs = assert_outputs_are_the_files(record, root)
    names = {d.name for d in outputs}
    assert "checkpoints/last.ckpt" in names
    assert Path(checkpoint.best_model_path).relative_to(root).as_posix() in names

    final = {m.name: m for m in record.metrics}
    curves = {c.name: c for c in s.read_history(cb.run_dir)}
    assert set(final) == {"train_loss", "val_loss"}
    assert len(curves["train_loss"].points) == 12  # 6 batches x 2 epochs, logged every step
    assert len(curves["val_loss"].points) == 2
    assert final["val_loss"].direction == "min" and final["train_loss"].direction is None
    for name in final:
        assert final[name].value == curves[name].points[-1].value
        assert final[name].step == curves[name].points[-1].step

    deps = {d.name: d for d in record.dependencies}
    assert {"python", "torch", "lightning"} <= set(deps)
    assert deps["lightning"].version == L.__version__
    assert record.ext["lightning"]["world_size"] == 1


def test_no_seed_everything_is_null_not_empty(tmp_path, dataset):
    cb = AttestationCallback()
    _fit(tmp_path, cb)
    record = s.read(cb.run_dir)
    assert record.seeds is None
    assert [(d.name, d.sha256) for d in record.data if d.role in ("train", "eval")] == [
        ("train", None),
        ("eval", None),
    ]


def test_conflicting_declared_directions_leave_it_null(tmp_path, dataset):
    cb = AttestationCallback()
    _fit(
        tmp_path,
        cb,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", mode="max", patience=10),
        ],
    )
    final = {m.name: m for m in s.read(cb.run_dir).metrics}
    assert final["val_loss"].direction is None


def test_without_a_logger_no_curve_is_recorded_and_checkpoints_stay_put(tmp_path, dataset):
    cb = AttestationCallback()
    trainer = _fit(tmp_path, cb, logger=False)
    assert trainer.loggers == []  # the callback did not add itself as the only logger
    assert Path(trainer.checkpoint_callback.dirpath) == tmp_path / "out" / "checkpoints"
    record = s.read(cb.run_dir)
    assert record.history is None
    assert {m.name for m in record.metrics} == {"train_loss", "val_loss"}
    assert all(m.step is None for m in record.metrics)
    assert record.ext["lightning"]["curves"] == "none: the trainer has no logger"


def test_a_run_that_raises_writes_a_failed_record(tmp_path, dataset):
    ds_dir = tmp_path / "datasets" / DATASET_ID / "1"
    cb = AttestationCallback(datasets=[(ds_dir, SPLIT)])
    with pytest.raises(RuntimeError, match="diverged"):
        _fit(tmp_path, cb, Linear(fail_at=8))
    record = assert_conforms_and_links(cb.run_dir, dataset)
    assert record.status == "failed"
    assert record.ext["lightning"]["error"] == {"type": "RuntimeError", "message": "diverged"}
    curves = {c.name: c for c in s.read_history(cb.run_dir)}
    assert [p.step for p in curves["train_loss"].points][-1] == 7


def test_an_interrupt_writes_a_cancelled_record(tmp_path, dataset):
    cb = AttestationCallback()
    with pytest.raises(SystemExit):  # Lightning turns Ctrl-C into exit(1)
        _fit(tmp_path, cb, Interrupting())
    assert s.read(cb.run_dir).status == "cancelled"


def test_only_global_rank_zero_writes(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(L.Trainer, "is_global_zero", property(lambda self: False))
    cb = AttestationCallback()
    trainer = _fit(tmp_path, cb)
    assert cb.run_dir is None and cb.record_path is None
    assert not (Path(trainer.log_dir) / "attestation").exists()
    assert os.listdir(tmp_path / "out")  # training itself ran
