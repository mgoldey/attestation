"""Corpus identity, recovered from artifacts and source without inference.

A comparison of two runs is only meaningful if they saw the same data, and
nothing in the ledger checked that. The corpus is usually not in the results:
in one real project not one of 34 result files records `vocab_size` or
`seq_len`, and the corpus exists only as a call in the driver --
`load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")` with
`get_encoding("gpt2")`.

Those are *literal arguments*, so `ast` recovers them exactly. That matters
more here than anywhere else in the codebase, because a corpus name is a
**join key**: two runs share a corpus only when their identity strings match.

An LLM was measured on this task before this module was written. Asked three
times for the corpus in one file it returned three different identities
("WikiText-2", "WikiText-2 data loading and tokenization", ...), and on a file
that merely *calls* a loader it invented `tokenizer: "load_wikitext2"` -- a
function name -- rather than declining. Either failure makes the guard report
agreement where there is none, which is worse than having no guard. So this
module reads syntax and never asks a model.
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

# Loader calls whose literal arguments name a dataset. Value is the meaning of
# each positional argument, so `load_dataset("a/b", "c")` yields source="a/b",
# config="c" without guessing at argument order.
CORPUS_FILE_ENV = "LEDGER_CORPUS_FILE"

_DATASET_CALLS: dict[str, tuple[str, ...]] = {
    "load_dataset": ("source", "config"),
}
# Calls whose first literal argument names a tokenizer/encoding.
_TOKENIZER_CALLS = ("get_encoding", "from_pretrained")
# Keyword arguments carrying corpus shape wherever they appear.
_SHAPE_KEYWORDS = {"seq_len": "seq_len", "max_seq_len": "seq_len", "block_size": "seq_len"}


@dataclass(frozen=True)
class DetectedCorpus:
    """What a source file *states* about its corpus. Every field optional:
    a partially-known corpus is the normal case and is more honest than none,
    provided the unknowns stay visible as unknowns."""

    source: str | None = None
    config: str | None = None
    tokenizer: str | None = None
    seq_len: int | None = None
    source_path: str | None = None

    def is_empty(self) -> bool:
        """Whether nothing at all was detected -- distinct from `source_path`
        alone being set, which just says where the (empty) detection looked."""
        return not any((self.source, self.config, self.tokenizer, self.seq_len))


log = logging.getLogger(__name__)


def _literal(node: ast.AST) -> str | int | None:
    """A node's value if it is a literal, else None. Deliberately does not
    resolve variables: `load_dataset(name)` states nothing about which corpus
    was used, and reporting the variable's *name* would be a fabrication."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return node.value
    return None


def detect_in_source(path: Path) -> DetectedCorpus | None:
    """Corpus identity stated by one Python file, or None if it states none.

    Never raises: a syntax error in one file of a scanned tree must not abort
    the scan, and an unreadable file states nothing.
    """
    try:
        tree = ast.parse(Path(path).read_text(errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None

    found: dict[str, str | int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in _DATASET_CALLS:
            for meaning, arg in zip(_DATASET_CALLS[name], node.args, strict=False):
                value = _literal(arg)
                if value is not None:
                    found.setdefault(meaning, value)
        elif name in _TOKENIZER_CALLS and node.args:
            value = _literal(node.args[0])
            if value is not None:
                found.setdefault("tokenizer", value)
        for kw in node.keywords:
            field = _SHAPE_KEYWORDS.get(kw.arg or "")
            if field:
                value = _literal(kw.value)
                if isinstance(value, int):
                    found.setdefault(field, value)

    if not found:
        return None
    seq_len = found.get("seq_len")
    return DetectedCorpus(
        source=found.get("source"),  # ty: ignore[invalid-argument-type]
        config=found.get("config"),  # ty: ignore[invalid-argument-type]
        tokenizer=found.get("tokenizer"),  # ty: ignore[invalid-argument-type]
        seq_len=seq_len if isinstance(seq_len, int) else None,
        source_path=str(path),
    )


# Field names under which artifacts already record corpus identity. Extracted
# rather than ranked: `dataset` names what a run read, and reading it as a
# measurement is the over-extraction bug one level up.
_ARTIFACT_FIELDS: dict[str, tuple[str, ...]] = {
    "name": ("dataset", "corpus", "dataset_name", "data", "corpus_name"),
    "source": ("dataset_source", "hf_dataset", "data_source"),
    "config": ("dataset_config", "subset"),
    "tokenizer": ("tokenizer", "tokenizer_name", "encoding"),
    "vocab_size": ("vocab_size", "n_vocab"),
    "seq_len": ("seq_len", "max_seq_len", "block_size", "context_length"),
}
# Every artifact key that names a corpus rather than measuring something.
CORPUS_KEYS = frozenset(k for names in _ARTIFACT_FIELDS.values() for k in names)


def from_payload(payload) -> dict | None:
    """Corpus fields stated by a result payload, or None if it states none."""
    if not isinstance(payload, dict):
        return None
    lowered = {str(k).lower(): v for k, v in payload.items()}
    found: dict = {}
    for field, names in _ARTIFACT_FIELDS.items():
        for key in names:
            if key not in lowered:
                continue
            value = lowered[key]
            if field in ("vocab_size", "seq_len"):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                found[field] = int(value)
            elif isinstance(value, str) and value.strip():
                found[field] = value.strip()
            break
    if not found:
        return None
    # A corpus needs a name to be a join key. Fall back to source, then config.
    if "name" not in found:
        name = found.get("source") or found.get("config")
        if not name:
            return None
        found["name"] = name
    return found


def manifest_path(workspace: Path | None = None) -> Path | None:
    """Where the corpus manifest lives: LEDGER_CORPUS_FILE, then the
    workspace, then the per-user file -- `None` if nothing is there to read.

    Shares `ledger._config_ladder`'s precedence (imported lazily: `ledger`
    imports this module at call time, so a module-level import back would
    cycle). Unlike `ledger._metric_direction_path`, this checks existence
    itself and returns `None` when absent: "nothing declared" must read as
    no manifest, never as a manifest at a path that isn't there.
    """
    from attestation.ledger import _config_ladder

    path = _config_ladder(CORPUS_FILE_ENV, "corpora.toml", workspace)
    return path if path.is_file() else None


def load_manifest(workspace: Path | None = None) -> tuple[dict, dict]:
    """`(corpora, assignments)` from the manifest, or two empty dicts.

    Never raises on a malformed file: a broken manifest must not abort a scan
    that would otherwise succeed from artifacts alone.
    """
    import tomllib

    path = manifest_path(workspace)
    if path is None:
        return {}, {}
    try:
        doc = tomllib.loads(path.read_text())
    except (OSError, ValueError):
        return {}, {}
    corpora = {}
    for name, body in (doc.get("corpus") or {}).items():
        if not isinstance(body, dict):
            continue
        entry = {k: v for k, v in body.items() if k != "splits"}
        entry["name"] = name
        entry["source_path"] = str(path)
        splits = body.get("splits")
        entry["splits"] = splits if isinstance(splits, dict) else {}
        corpora[name] = entry
    assign = doc.get("assign") or {}
    return corpora, {
        "family": dict(assign.get("family") or {}),
        "run": dict(assign.get("run") or {}),
    }


def upsert(conn, entry: dict) -> int | None:
    """Store a corpus by name and return its id, filling gaps without
    overwriting. A declaration must never be silently replaced by a weaker
    value read from an artifact, so an existing non-NULL column stands."""
    name = entry.get("name")
    if not name:
        return None
    columns = ("source", "config", "tokenizer", "vocab_size", "seq_len", "source_path")
    row = conn.execute("SELECT * FROM corpora WHERE name = ?", (str(name),)).fetchone()
    if row is None:
        # INSERT OR IGNORE, not a bare INSERT after a read. The check and the
        # write had nothing around them, so concurrent FIRST scans of one
        # workspace -- ordinary when two projects declare the same corpus --
        # gave 7 of 8 callers an OperationalError and left ZERO rows: they lost
        # their data, not just their race. Only the first scan is exposed;
        # after that the UPDATE path below is safe, which is why it hid behind
        # a populated database.
        conn.execute(
            f"INSERT OR IGNORE INTO corpora(name, {', '.join(columns)})"
            f" VALUES (?, {', '.join('?' * len(columns))})",
            (str(name), *(entry.get(c) for c in columns)),
        )
        conn.commit()
    else:
        # A CONFLICT is not a gap. The docstring below promises a declaration
        # is never silently replaced by a weaker value; a different value was
        # simply ignored, which is worse -- two arms declaring `internal-eval`
        # with different dataset_source collapsed into one row, and compare
        # then vouched for agreement between runs on different data. Mark the
        # name as contested so _corpus_agreement can refuse rather than
        # confirm; inventing a second name would be a guess about which
        # declaration is authoritative.
        conflicts = [
            c
            for c in columns
            if row[c] is not None and entry.get(c) is not None and row[c] != entry.get(c)
        ]
        if conflicts:
            log.warning(
                "corpus %r declared with conflicting %s (%s vs %s); marking it contested",
                name,
                ", ".join(conflicts),
                row[conflicts[0]],
                entry.get(conflicts[0]),
            )
            conn.execute(
                "UPDATE corpora SET source = ? WHERE name = ?",
                (f"CONTESTED: {row['source']} vs {entry.get('source')}", str(name)),
            )
        missing = {c: entry.get(c) for c in columns if row[c] is None and entry.get(c) is not None}
        if missing:
            sets = ", ".join(f"{c} = ?" for c in missing)
            conn.execute(
                f"UPDATE corpora SET {sets} WHERE name = ?", (*missing.values(), str(name))
            )
    got = conn.execute("SELECT id FROM corpora WHERE name = ?", (str(name),)).fetchone()
    corpus_id = got["id"]
    for split, body in (entry.get("splits") or {}).items():
        if not isinstance(body, dict):
            continue
        conn.execute(
            "INSERT OR IGNORE INTO corpus_splits(corpus_id, split, n_tokens, n_records, n_bytes)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                corpus_id,
                str(split),
                body.get("n_tokens"),
                body.get("n_records"),
                body.get("n_bytes"),
            ),
        )
    return corpus_id
