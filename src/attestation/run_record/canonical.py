"""RFC 8785 JSON Canonicalization Scheme (JCS) -- a re-export.

The implementation lives in `attestation_schema.canonical`, the one copy:
attestation took a runtime dependency on the schema package when the ledger
began reading core records (docs/superpowers/specs/2026-11-19-ledger-reads-
core-records-design.md), and the core-record spec said the duplicated
serializer would then become this re-export. The signed checker-run record
(`run_record/`) and the core run record therefore canonicalize through the
same function, which is what keeps their bytes comparable.
"""

from attestation_schema.canonical import MAX_SAFE_INTEGER, canonicalize

__all__ = ["MAX_SAFE_INTEGER", "canonicalize"]
