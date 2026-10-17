"""Where a run record's signing key comes from.

The default is one ed25519 key per install, generated on first use under
`$HERMES_HOME/attestation/keys/`, mode 0600 in a 0700 directory. Whose key
SHOULD sign a record -- the install's, the researcher's, an organisation's --
is an open decision (see the run-record design spec), so the source is a
small protocol rather than a function: a new one is a class with `signer()`.

Two ship. `LocalKeySource` generates its key when there is none, because a
first record should not need a setup step. `ATTEST_RECORD_KEY=<path>` names an
existing key and NEVER generates one: a typo in that variable must fail, not
silently mint a new identity whose records nobody trusts.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from attestation.run_record.verify import SIGNATURE_ALG, keyid_for

ENV_KEY = "ATTEST_RECORD_KEY"
DEFAULT_KEY_NAME = "run-record-ed25519.pem"
KEY_FILE_MODE = 0o600
KEY_DIR_MODE = 0o700


class RecordKeyError(Exception):
    """A key that cannot be used, with what to do about it."""


@dataclass(frozen=True)
class Signer:
    """A private key and the identity a record embeds for it."""

    private_key: Ed25519PrivateKey
    source: str

    @property
    def public_raw(self) -> bytes:
        """The raw 32-byte ed25519 public key."""
        return self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    @property
    def keyid(self) -> str:
        """The hex sha256 of the raw public key."""
        return keyid_for(self.public_raw)

    def identity(self) -> dict:
        """The `predicate.signer` block: inside the signed payload, so swapping
        it for another key breaks the signature."""
        return {
            "alg": SIGNATURE_ALG,
            "keyid": self.keyid,
            "public_key": base64.b64encode(self.public_raw).decode("ascii"),
        }

    def sign(self, message: bytes) -> bytes:
        """An ed25519 signature over `message`."""
        return self.private_key.sign(message)

    def public_pem(self) -> str:
        """The public key as PEM, for handing to a verifier."""
        return (
            self.private_key.public_key()
            .public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            .decode("ascii")
        )


class KeySource(Protocol):
    """Anything that can hand back a `Signer`; one per way of holding a key."""

    def signer(self) -> Signer:
        """The key to sign with; raises `RecordKeyError` when unusable."""
        ...


def _refuse_loose_permissions(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise RecordKeyError(
            f"{path} is readable by other users (mode {mode:o}); run `chmod 600 {path}`"
        )


def _load(path: Path, source: str) -> Signer:
    _refuse_loose_permissions(path)
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (ValueError, TypeError) as exc:
        raise RecordKeyError(f"{path} is not an unencrypted PEM private key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise RecordKeyError(f"{path} is not an ed25519 key")
    return Signer(key, source)


def _pem(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@dataclass(frozen=True)
class LocalKeySource:
    """The per-install default: generated on first use, then reused."""

    path: Path

    def signer(self) -> Signer:
        """The key at `path`, generated first if there is none."""
        if not self.path.exists():
            self._generate()
        return _load(self.path, f"local:{self.path}")

    def _generate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=KEY_DIR_MODE)
        if os.name == "posix":
            self.path.parent.chmod(KEY_DIR_MODE)
        try:
            # O_EXCL: two processes racing on first use must not both write a
            # key -- the loser would sign with a key the file no longer holds.
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, KEY_FILE_MODE)
        except FileExistsError:
            return
        with os.fdopen(fd, "wb") as fh:
            fh.write(_pem(Ed25519PrivateKey.generate()))


@dataclass(frozen=True)
class FileKeySource:
    """An existing key named by path. Never generates."""

    path: Path

    def signer(self) -> Signer:
        """The key at `path`; refuses when it is missing."""
        if not self.path.is_file():
            raise RecordKeyError(f"{ENV_KEY} names {self.path}, which does not exist")
        return _load(self.path, f"file:{self.path}")


def default_key_path() -> Path:
    """Where the per-install key lives under `$HERMES_HOME`."""
    from attestation import paths

    return paths.record_keys_dir() / DEFAULT_KEY_NAME


def resolve_key_source() -> KeySource:
    """`ATTEST_RECORD_KEY` if set, else the per-install key. Read at call time."""
    configured = (os.environ.get(ENV_KEY) or "").strip()
    if configured:
        return FileKeySource(Path(configured).expanduser())
    return LocalKeySource(default_key_path())
