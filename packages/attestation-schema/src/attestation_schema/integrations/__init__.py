"""Training-loop callbacks that write a core record (`attestation.run/1`).

    pip install "attestation-schema[hf]"          # Hugging Face Trainer
    from attestation_schema.integrations.huggingface import AttestationCallback

    pip install "attestation-schema[lightning]"   # PyTorch Lightning
    from attestation_schema.integrations.lightning import AttestationCallback

Importing this package imports no framework: each framework is imported only
by its own module, so `attestation_schema` stays standard-library only and
3.11-compatible. What the callbacks share (the recorder, dependency and code
identity) is `_core`, stdlib only. Spec:
docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

EXTRAS = {"huggingface": "hf", "lightning": "lightning"}


def missing_extra(module: str, exc: ImportError) -> ImportError:
    """The ImportError a framework module raises when its extra is not installed."""
    extra = EXTRAS[module]
    return ImportError(
        f"attestation_schema.integrations.{module} needs its framework ({exc.name or exc}):"
        f' pip install "attestation-schema[{extra}]"'
    )
