"""A Hugging Face `Trainer` callback that writes a core record when training ends.

    from attestation_schema.integrations.huggingface import AttestationCallback

    with AttestationCallback(datasets=[("data/ds/v1", "scaffold")]) as cb:
        trainer = Trainer(model, args, train_dataset=..., callbacks=[cb])
        trainer.train()
    cb.record_path    # <output_dir>/attestation/run-<run_id>/attestation/run.json

At `on_train_end` (global rank 0 only) the record gets: `TrainingArguments`
as `config`; `args.seed` (applied by `Trainer.__init__`) and `data_seed` as
`seeds`; every `on_log` value as a curve in the history sidecar and its last
value in `metrics`; the files of every checkpoint this run saved, by sha256,
role `output`; the datasets given (or, if none, `train`/`eval` named with no
digest); python/torch/transformers/accelerate with their provenance and
source; the training script's git identity. HF has no exception hook, so the
`with` form is what writes a `failed` (or, for Ctrl-C, `cancelled`) record
when `train()` raises. Spec:
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
    output_entries,
    output_name,
)
from attestation_schema.record import CodeIdentity, DataDigest

# The framework is an extra, absent from the dev venv and the gates job on
# purpose (spec, "Testing and CI"); ty cannot resolve it there.
try:
    from transformers import TrainerCallback  # ty: ignore[unresolved-import]
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise missing_extra("huggingface", exc) from exc

NAMESPACE = "hf-trainer"
# (record name, module, distribution): what an HF Trainer run runs on.
FRAMEWORK = (
    ("torch", "torch", "torch"),
    ("transformers", "transformers", "transformers"),
    ("accelerate", "accelerate", "accelerate"),
)
CHECKPOINT_PREFIX = "checkpoint"  # transformers.trainer_utils.PREFIX_CHECKPOINT_DIR


class AttestationCallback(TrainerCallback):
    """Write `<run_dir>/attestation/run.json` when `Trainer.train()` ends.

    Every argument is optional. `run_dir` defaults to
    `<output_dir>/attestation/run-<run_id>`; `run_id` to a fresh 12-hex id;
    `family`/`arm` to null. `datasets`: dataset directories, `dataset.json`
    paths or `DatasetRecord`s, each optionally `(dataset, split_name)`, listed
    by content (and split) digest with `dataset_role`; `data`: further
    `DataDigest` entries, verbatim. `seeds`, `code` and `directions` override
    what the callback observes.
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
        self.recorder = Recorder(options, "hf-trainer", NAMESPACE)
        self._args: Any = None
        self._state: Any = None
        self._saved: list[str] = []
        self._evaluated = False

    @property
    def run_dir(self) -> Path | None:
        """The run directory (known from train start, on rank 0)."""
        return self.recorder.run_dir

    @property
    def record_path(self) -> Path | None:
        """`run.json`, once written."""
        return self.recorder.written

    # ---- Trainer hooks ------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        self._args, self._state = args, state
        if state.is_world_process_zero:
            self.recorder.begin(args.output_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.recorder.log(logs, state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        self._evaluated = True

    def on_save(self, args, state, control, **kwargs):
        if self.recorder.begun:
            self._saved.append(f"{CHECKPOINT_PREFIX}-{state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.recorder.begun:
            self._finish("complete")

    # ---- failure: HF calls no hook when train() raises ---------------------

    def __enter__(self) -> AttestationCallback:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        if not self.recorder.begun or self.recorder.written is not None:
            return
        if exc is None:
            self.recorder.abort()  # left the block without train() ending
            return
        try:
            self._finish(error_status(exc), error=exc)
        except (OSError, ValueError):
            self.recorder.abort()  # never mask the training error with ours

    # ---- the record ---------------------------------------------------------

    def _declared_directions(self) -> dict[str, str | None]:
        """HF's best-model criterion: the metric the trainer compared, in the
        direction it compared it (`_determine_best_metric` adds `eval_`)."""
        name = getattr(self._args, "metric_for_best_model", None)
        greater = getattr(self._args, "greater_is_better", None)
        if not name or greater is None:
            return {}
        key = name if name.startswith("eval_") else f"eval_{name}"
        return merge_directions([(key, "max" if greater else "min")])

    def _finish(self, status: str, error: BaseException | None = None) -> Path:
        args, state = self._args, self._state
        root = Path(args.output_dir)
        folders = [root / name for name in self._saved]
        seeds: dict[str, int | list[int]] = {"seed": int(args.seed)}
        if getattr(args, "data_seed", None) is not None:
            seeds["data_seed"] = int(args.data_seed)
        ext: dict[str, Any] = {
            "world_size": getattr(args, "world_size", 1),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "max_steps": state.max_steps,
            "checkpoints_removed": [p.name for p in folders if not p.exists()],
            "best_model_checkpoint": (
                output_name(Path(state.best_model_checkpoint), root)
                if getattr(state, "best_model_checkpoint", None)
                else None
            ),
        }
        if error is not None:
            ext["error"] = error_ext(error)
        r = self.recorder
        return r.finish(
            status=status,
            config=args.to_dict(),
            data=r.data(self._evaluated, output_entries(folders, root)),
            metrics=r.metrics(declared=self._declared_directions()),
            seeds=seeds,
            dependencies=dependencies(FRAMEWORK),
            ext=ext,
        )
