"""An MLflow run-context provider: every run started carries its provenance.

    pip install "attestation-schema[mlflow]"
    # nothing to import: MLflow finds the provider through the
    # `mlflow.run_context_provider` entry point and asks it for tags at every
    # `mlflow.start_run`

The tags (`attestation.format`, `.producer`, `.code`, `.dependencies`, and
`.error` when something could not be observed) are what the run's own
process observed at run start: the script's git checkout with its dirty bit,
and the identity of every already-imported key package through
`identity.package_identity`. `attest mlflow sync` and the exporter read them
back into the core record (`mlflow_mapping.read_tags`).

On by default; `ATTESTATION_MLFLOW_TAGS=0` turns it off. It never raises,
imports nothing the run did not, and writes only `attestation.*` tags. This
is the ONLY module in the package that imports mlflow; everything it does is
`mlflow_mapping` (stdlib). Spec:
docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

from attestation_schema.integrations import missing_extra
from attestation_schema.integrations.mlflow_mapping import enabled, observe_tags

# mlflow is an extra, absent from the default dev venv and the gates job;
# ty cannot resolve it there.
try:
    from mlflow.tracking.context.abstract_context import (  # ty: ignore[unresolved-import]
        RunContextProvider,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise missing_extra("mlflow", exc) from exc


class AttestationRunContext(RunContextProvider):
    """MLflow instantiates this once, when mlflow is imported; construction
    does nothing."""

    def in_context(self) -> bool:
        """True unless `ATTESTATION_MLFLOW_TAGS` is 0/false/no/off. Reads one
        environment variable and does no I/O: MLflow asks it for every run."""
        return enabled()

    def tags(self) -> dict[str, str]:
        """The `attestation.*` tags for the run starting now; never raises."""
        return observe_tags()


__all__ = ["AttestationRunContext"]
