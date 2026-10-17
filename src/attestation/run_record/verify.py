"""Verify a run record with nothing but the record and, optionally, its files.

Deliberately small and dependency-light: the standard library, `cryptography`
(imported only when a signature is checked) and `canonical.py` beside it. No
database, no model, nothing from the rest of Attestation -- a reviewer who was
handed a record and a draft runs

    python -m attestation.run_record.verify record.json --root unpacked/

and gets the same answer `attest record verify` gives.

Every item is reported as `ok`, `FAILED`, `warning` or `not checked`, and a
missing file is never a pass. What no verifier can establish from a record
alone is printed too (`CANNOT_CHECK`), because a valid signature over a
`supported` verdict reads as "the claim is true" to anyone not told otherwise.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from attestation.run_record.canonical import canonicalize

PAYLOAD_TYPE = "application/vnd.in-toto+json"
STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://github.com/mgoldey/attestation/run-record/v1"
SIGNATURE_ALG = "ed25519"

OK, FAILED, WARNING, UNCHECKED = "ok", "FAILED", "warning", "not checked"

CANNOT_CHECK = (
    "that any verdict is correct -- a valid record says what the tool concluded, not that"
    " it was right; re-run the check against the same ledger and library to test that",
    "that the listed papers were read and the queries run -- those are the signer's account",
    "when the run happened -- timestamps are the signer's clock, not a trusted timestamp",
    "who the signer is, unless you pinned a key you already trust (--key or --keyid)",
)


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE v1 Pre-Authentication Encoding: what the signature actually covers."""
    kind = payload_type.encode("utf-8")
    return b"DSSEv1 %d %b %d %b" % (len(kind), kind, len(payload), payload)


def keyid_for(public_raw: bytes) -> str:
    """A key's id: the hex sha256 of its raw 32-byte public key."""
    return hashlib.sha256(public_raw).hexdigest()


def sha256_file(path: Path) -> str:
    """A file's hex sha256, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Item:
    """One thing the verifier looked at, and what it found."""

    what: str
    status: str
    detail: str = ""


@dataclass
class Report:
    """Everything the verifier looked at, plus the decoded Statement when there was one."""

    items: list[Item] = field(default_factory=list)
    statement: dict | None = None

    def add(self, what: str, status: str, detail: str = "") -> None:
        """Record one item."""
        self.items.append(Item(what, status, detail))

    @property
    def ok(self) -> bool:
        """True when nothing FAILED. Unchecked items do not fail a record."""
        return bool(self.items) and all(i.status != FAILED for i in self.items)

    def to_dict(self) -> dict:
        """The report as JSON-ready data, with what no record can show."""
        return {
            "ok": self.ok,
            "items": [vars(i) for i in self.items],
            "cannot_check": list(CANNOT_CHECK),
            "summary": summarize(self.statement) if self.statement else None,
        }

    def lines(self) -> list[str]:
        """The report as terminal lines, ending VERIFIED or NOT VERIFIED."""
        out = [f"  {i.status:12s} {i.what}" + (f" -- {i.detail}" if i.detail else "") for i in self]
        out.append("")
        out.append("A record cannot show:")
        out.extend(f"  - {c}" for c in CANNOT_CHECK)
        out.append("")
        out.append("VERIFIED" if self.ok else "NOT VERIFIED")
        return out

    def __iter__(self):
        return iter(self.items)


def summarize(statement: dict) -> dict:
    """What the record says happened, in a few fields -- for a reader, not a check."""
    pred = statement.get("predicate") or {}
    return {
        "run": pred.get("run"),
        "tool": pred.get("tool"),
        "signer_keyid": (pred.get("signer") or {}).get("keyid"),
        "n_inputs": len(pred.get("inputs") or []),
        "n_papers": len(pred.get("papers") or []),
        "n_queries": len(pred.get("queries") or []),
        "checks": {c.get("check"): c.get("summary") for c in pred.get("checks") or []},
    }


def _b64(text) -> bytes:
    if not isinstance(text, str):
        raise ValueError("not a base64 string")
    return base64.b64decode(text, validate=True)


def _decode_envelope(envelope, report: Report) -> tuple[bytes, dict] | None:
    """The payload bytes and parsed Statement, or None after reporting why not."""
    if not isinstance(envelope, dict) or envelope.get("payloadType") != PAYLOAD_TYPE:
        report.add("envelope", FAILED, f"not a DSSE envelope with payloadType {PAYLOAD_TYPE}")
        return None
    try:
        payload = _b64(envelope.get("payload"))
        statement = json.loads(payload)
    except (ValueError, binascii.Error) as exc:
        report.add("envelope", FAILED, f"payload does not decode: {exc}")
        return None
    if not isinstance(statement, dict):
        report.add("envelope", FAILED, "payload is not a JSON object")
        return None
    report.add("envelope", OK, "DSSE, in-toto payload")
    return payload, statement


def _check_canonical(payload: bytes, statement: dict, report: Report) -> None:
    try:
        canonical = canonicalize(statement) == payload
    except (TypeError, ValueError) as exc:
        report.add("canonical form", WARNING, f"cannot canonicalize: {exc}")
        return
    if canonical:
        report.add("canonical form", OK, "RFC 8785 JCS")
    else:
        report.add("canonical form", WARNING, "payload is not JCS; the signature still covers it")


def _check_statement(statement: dict, report: Report) -> None:
    problems = []
    if statement.get("_type") != STATEMENT_TYPE:
        problems.append(f"_type is not {STATEMENT_TYPE}")
    if statement.get("predicateType") != PREDICATE_TYPE:
        problems.append(f"predicateType is not {PREDICATE_TYPE}")
    subject = statement.get("subject")
    if not isinstance(subject, list) or not subject:
        problems.append("subject is empty")
    elif not all(_digest(d) for d in subject):
        problems.append("a subject has no sha256 digest")
    if not isinstance(statement.get("predicate"), dict):
        problems.append("predicate is missing")
    report.add("statement", FAILED if problems else OK, "; ".join(problems) or "in-toto v1")


def _digest(descriptor) -> str | None:
    if not isinstance(descriptor, dict):
        return None
    value = (descriptor.get("digest") or {}).get("sha256")
    return value if isinstance(value, str) else None


def _embedded_key(statement: dict, report: Report) -> bytes | None:
    signer = (statement.get("predicate") or {}).get("signer") or {}
    try:
        raw = _b64(signer.get("public_key"))
    except (ValueError, binascii.Error):
        report.add("signature", FAILED, "the record embeds no readable public key")
        return None
    if signer.get("alg") != SIGNATURE_ALG or len(raw) != 32:
        report.add("signature", FAILED, f"embedded key is not a 32-byte {SIGNATURE_ALG} key")
        return None
    if signer.get("keyid") != keyid_for(raw):
        report.add("signature", FAILED, "embedded keyid does not match the embedded key")
        return None
    return raw


def _signature_verifies(public_raw: bytes, envelope: dict, payload: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = Ed25519PublicKey.from_public_bytes(public_raw)
    message = pae(envelope["payloadType"], payload)
    for sig in envelope.get("signatures") or []:
        if not isinstance(sig, dict) or sig.get("keyid") not in (None, "", keyid_for(public_raw)):
            continue
        try:
            key.verify(_b64(sig.get("sig")), message)
            return True
        except (InvalidSignature, ValueError, binascii.Error):
            continue
    return False


def _check_signature(envelope, payload, statement, report, pinned_keyid) -> None:
    raw = _embedded_key(statement, report)
    if raw is None:
        return
    keyid = keyid_for(raw)
    if not _signature_verifies(raw, envelope, payload):
        report.add("signature", FAILED, f"no signature verifies under the embedded key {keyid}")
        return
    report.add("signature", OK, f"ed25519, key {keyid}")
    if pinned_keyid is None:
        report.add("signer identity", UNCHECKED, "key is embedded, not pinned; pass --key/--keyid")
    elif pinned_keyid.lower() == keyid:
        report.add("signer identity", OK, "matches the pinned key")
    else:
        report.add("signer identity", FAILED, f"signed by {keyid}, not the pinned {pinned_keyid}")


def _check_summaries(statement: dict, report: Report) -> None:
    for check in (statement.get("predicate") or {}).get("checks") or []:
        name = f"check {check.get('check')!r}"
        tally = dict(Counter(r.get("verdict") for r in check.get("results") or []))
        if tally == (check.get("summary") or {}):
            report.add(name, OK, ", ".join(f"{n} {v}" for v, n in sorted(tally.items())) or "empty")
        else:
            report.add(name, FAILED, f"summary {check.get('summary')} != its results {tally}")


def _descriptors(statement: dict) -> list[tuple[str, dict]]:
    pred = statement.get("predicate") or {}
    seen: set[tuple] = set()
    out = []
    for kind, items in (
        ("subject", statement.get("subject")),
        ("input", pred.get("inputs")),
        ("output", pred.get("outputs")),
    ):
        for d in items or []:
            key = (d.get("name"), _digest(d)) if isinstance(d, dict) else (None, None)
            if key not in seen:
                seen.add(key)
                out.append((kind, d))
    return out


def _check_files(statement: dict, root: Path, report: Report) -> None:
    for kind, descriptor in _descriptors(statement):
        name, want = descriptor.get("name"), _digest(descriptor)
        label = f"{kind} {name}"
        if not isinstance(name, str) or want is None:
            report.add(label, FAILED, "descriptor has no name or sha256")
            continue
        path = root / name
        if not path.is_file():
            report.add(label, UNCHECKED, f"not found under {root}")
        elif sha256_file(path) == want:
            report.add(label, OK, "sha256 matches")
        else:
            report.add(label, FAILED, "sha256 differs: the file changed after the record")


def verify(
    envelope,
    *,
    root: Path | None = None,
    pinned_public_key: bytes | None = None,
    pinned_keyid: str | None = None,
) -> Report:
    """Check a parsed envelope. `root` is where its relative file names resolve."""
    report = Report()
    decoded = _decode_envelope(envelope, report)
    if decoded is None:
        return report
    payload, statement = decoded
    report.statement = statement
    if pinned_public_key is not None:
        pinned_keyid = keyid_for(pinned_public_key)
    _check_signature(envelope, payload, statement, report, pinned_keyid)
    _check_canonical(payload, statement, report)
    _check_statement(statement, report)
    _check_summaries(statement, report)
    _check_files(statement, Path(root) if root is not None else Path.cwd(), report)
    return report


def load_public_key(path: Path) -> bytes:
    """Raw 32 bytes of an ed25519 public key in PEM (SubjectPublicKeyInfo)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{path} is not an ed25519 public key")
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def verify_file(
    path: Path,
    *,
    root: Path | None = None,
    key: Path | None = None,
    keyid: str | None = None,
) -> Report:
    """Read and verify a record file. Raises OSError/ValueError on an unreadable one."""
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    pinned = load_public_key(key) if key is not None else None
    return verify(envelope, root=root, pinned_public_key=pinned, pinned_keyid=keyid)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The verify flags, shared by `attest record verify` and `python -m`."""
    parser.add_argument("record", help="the run record (a DSSE envelope .json)")
    parser.add_argument(
        "--root", default=None, help="where the record's file names resolve (default: cwd)"
    )
    pin = parser.add_mutually_exclusive_group()
    pin.add_argument("--key", default=None, help="require this ed25519 public key (PEM)")
    pin.add_argument("--keyid", default=None, help="require this key id (hex sha256)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")


def run(args: argparse.Namespace) -> int:
    """Verify `args.record` and print the report; 0 only when nothing FAILED."""
    try:
        report = verify_file(
            Path(args.record).expanduser(),
            root=Path(args.root).expanduser() if args.root else None,
            key=Path(args.key).expanduser() if args.key else None,
            keyid=args.keyid,
        )
    except (OSError, ValueError) as exc:
        print(f"cannot read record: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n".join(report.lines()))
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    """`python -m attestation.run_record.verify`: the same as `attest record verify`."""
    parser = argparse.ArgumentParser(prog="attestation.run_record.verify")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
