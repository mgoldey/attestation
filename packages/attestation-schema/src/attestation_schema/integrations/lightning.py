"""A PyTorch Lightning callback that writes a core record when `fit` ends.

    from attestation_schema.integrations.lightning import AttestationCallback

    L.seed_everything(7)
    cb = AttestationCallback(datasets=["data/ds/v1"])
    trainer = L.Trainer(max_epochs=2, callbacks=[cb])
    trainer.fit(model, datamodule=dm)
    cb.record_path    # <trainer.log_dir>/attestation/run-<run_id>/attestation/run.json

At `on_fit_end` (global rank 0 only) the record gets: the module's and the
datamodule's `hparams` and the trainer's loop settings as `config`; the
`seed_everything` seed (`PL_GLOBAL_SEED`) as `seeds`, or null when it never
ran; every value Lightning sends to its loggers as a curve, and
`trainer.callback_metrics` as `metrics`; every `ModelCheckpoint` file by
sha256, role `output`; the datasets given (or `train`/`eval` named with no
digest); python/torch/lightning with their provenance and source; the
training script's git identity. `on_exception` writes `failed` (`cancelled`
for Ctrl-C). Uses `lightning.pytorch`; a `pytorch_lightning.Trainer` checks
callbacks against its own base class and refuses this one. Spec:
docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from attestation_schema.integrations import missing_extra
from attestation_schema.integrations._core import (
    DatasetArg,
    Options,
    Recorder,
    dependencies,
    error_ext,
    error_status,
    merge_directions,
    number,
    output_entries,
)
from attestation_schema.record import CodeIdentity, DataDigest

# The framework is an extra, absent from the dev venv and the gates job on
# purpose (spec, "Testing and CI"); ty cannot resolve it there.
try:
    from lightning.pytorch.callbacks import (  # ty: ignore[unresolved-import]
        Callback,
        EarlyStopping,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import Logger  # ty: ignore[unresolved-import]
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise missing_extra("lightning", exc) from exc

NAMESPACE = "lightning"
FRAMEWORK = (
    ("torch", "torch", "torch"),
    ("lightning", "lightning", "lightning"),
)
# Loop settings a Trainer exposes as attributes; what the run was given.
TRAINER_SETTINGS = (
    "max_epochs",
    "min_epochs",
    "max_steps",
    "min_steps",
    "accumulate_grad_batches",
    "gradient_clip_val",
    "gradient_clip_algorithm",
    "precision",
    "check_val_every_n_epoch",
    "val_check_interval",
    "limit_train_batches",
    "limit_val_batches",
    "num_devices",
    "num_nodes",
)


class _Tap(Logger):
    """Receives exactly what Lightning sends its loggers: the curves."""

    def __init__(self, recorder: Recorder):
        super().__init__()
        self._recorder = recorder

    @property
    def name(self) -> str:
        return "attestation"

    @property
    def version(self) -> str:
        return self._recorder.run_id

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        """Hyperparameters reach the record through `config`, not here."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._recorder.log(metrics, 0 if step is None else int(step))


def _hparams(obj: Any) -> dict[str, Any] | None:
    hp = getattr(obj, "hparams", None)
    return dict(hp) if hp else None


class AttestationCallback(Callback):
    """Write `<run_dir>/attestation/run.json` when `Trainer.fit()` ends.

    Arguments as for the Hugging Face callback: every one optional;
    `run_dir` defaults to `<trainer.log_dir>/attestation/run-<run_id>`.
    """

    def __init__(
        self,
        *,
        run_dir: str | os.PathLike[str] | None = None,
        run_id: str | None = None,
        family: str | None = None,
        arm: str | None = None,
        datasets: Sequence[DatasetArg] | None = None,
        dataset_role: str = "train",
        data: Sequence[DataDigest] = (),
        seeds: dict[str, int | list[int]] | None = None,
        code: CodeIdentity | None = None,
        directions: Mapping[str, str] | None = None,
    ):
        options = Options(
            run_dir=run_dir,
            run_id=run_id,
            family=family,
            arm=arm,
            datasets=datasets,
            dataset_role=dataset_role,
            data=data,
            seeds=seeds,
            code=code,
            directions=dict(directions or {}),
        )
        self.recorder = Recorder(options, "lightning", NAMESPACE)
        self._root: Path | None = None
        self._seed: int | None = None
        self._seed_workers: bool | None = None
        self._tapped = False
        self._evaluated = False

    @property
    def run_dir(self) -> Path | None:
        return self.recorder.run_dir

    @property
    def record_path(self) -> Path | None:
        return self.recorder.written

    # ---- hooks --------------------------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        root = trainer.log_dir or trainer.default_root_dir  # broadcasts: every rank
        if not trainer.is_global_zero:
            return
        self._root = Path(root)
        seed = os.environ.get("PL_GLOBAL_SEED")
        self._seed = int(seed) if seed is not None and seed.strip().lstrip("-").isdigit() else None
        workers = os.environ.get("PL_SEED_WORKERS")
        self._seed_workers = None if workers is None else workers == "1"
        self.recorder.begin(self._root)
        # Appended last, and only beside an existing logger: Lightning takes
        # log_dir and the checkpoint directory from loggers[0], and with no
        # logger it logs nothing anywhere (history null is then the truth).
        if trainer.loggers:
            trainer.loggers = [*trainer.loggers, _Tap(self.recorder)]
            self._tapped = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._evaluated = True

    def on_fit_end(self, trainer, pl_module):
        if self.recorder.begun and self.recorder.written is None:
            self._finish(trainer, pl_module, "complete")

    def on_exception(self, trainer, pl_module, exception):
        if self.recorder.begun and self.recorder.written is None:
            try:
                self._finish(trainer, pl_module, error_status(exception), error=exception)
            except (OSError, ValueError):
                self.recorder.abort()  # never mask the training error with ours

    # ---- the record ---------------------------------------------------------

    def _final_metrics(self, trainer) -> dict[str, tuple[int | float, int | None]]:
        """`callback_metrics`: Lightning's current values. A value's step is
        its curve's last step when that point holds the same value; Lightning
        does not say which step an epoch aggregate belongs to."""
        out: dict[str, tuple[int | float, int | None]] = {}
        for name, raw in dict(trainer.callback_metrics).items():
            value = number(raw)
            if value is None:
                continue
            last = self.recorder.last.get(name)
            out[name] = (value, last[1] if last is not None and last[0] == value else None)
        return out

    def _finish(self, trainer, pl_module, status: str, error: BaseException | None = None) -> Path:
        declared = merge_directions(
            (cb.monitor, cb.mode)
            for cb in trainer.callbacks
            if isinstance(cb, ModelCheckpoint | EarlyStopping) and cb.monitor
        )
        paths: list[Path] = []
        for cb in trainer.checkpoint_callbacks:
            names = [*getattr(cb, "best_k_models", {}), cb.best_model_path, cb.last_model_path]
            paths += [Path(p) for p in names if p]
        config: dict[str, Any] = {
            "hparams": _hparams(pl_module),
            "datamodule": _hparams(trainer.datamodule),
            "trainer": {k: getattr(trainer, k, None) for k in TRAINER_SETTINGS},
        }
        ext: dict[str, Any] = {
            "world_size": trainer.world_size,
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
            "seed_workers": self._seed_workers,
            "curves": "recorded" if self._tapped else "none: the trainer has no logger",
        }
        if error is not None:
            ext["error"] = error_ext(error)
        r = self.recorder
        root = self._root or Path(trainer.default_root_dir)
        return r.finish(
            status=status,
            config=config,
            data=r.data(self._evaluated, output_entries(paths, root)),
            metrics=r.metrics(final=self._final_metrics(trainer), declared=declared),
            seeds=None if self._seed is None else {"seed_everything": self._seed},
            dependencies=dependencies(FRAMEWORK),
            ext=ext,
        )
