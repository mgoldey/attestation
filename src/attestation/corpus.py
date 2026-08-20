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
from dataclasses import dataclass
from pathlib import Path

# Loader calls whose literal arguments name a dataset. Value is the meaning of
# each positional argument, so `load_dataset("a/b", "c")` yields source="a/b",
# config="c" without guessing at argument order.
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
        return not any((self.source, self.config, self.tokenizer, self.seq_len))


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
