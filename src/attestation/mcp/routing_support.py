"""The `cite.support` route, shared by `runs.ask` and `kg.ask`.

A deployed Hermes session (`ATTEST_TOOLS=provenance` or `knowledge`, without
`ATTEST_EXPAND`) sees only its router and the disclosure tool, so a tool the
router cannot reach is unreachable there however good its docstring is. This
module is the route: a deterministic phrase rule (no model call, like every
router in `routing.py`) plus the argument extraction a router needs when the
question itself carries the sentence and the key.

Its own module because `routing.py` and `ask.py` are at their size caps
(tests/test_architecture.py), and this rule is checked BEFORE either router's
own table -- "is this claim supported by its source" contains "claim", which
`route_runs` sends to `runs.claims_check`.
"""

import re

# "support"/"say" alone is too wide: "are my claims supported by the runs" is
# runs.claims_check, and "what articles support X" is a feed search. The rule
# needs a word that points at a CITED source, and no word that points at runs.
_ASKS_SUPPORT = ("support", "back up", "backs up", "say that", "says that", "actually say")
_CITED_THING = (
    "citation",
    "cited",
    "cite ",
    "cites",
    "paper",
    "source",
    "reference",
    "sentence",
)
_ALWAYS = ("misquot", "misrepresent", "misattribut", "right number, wrong")
_RUNS = re.compile(r"\bruns?\b|\bledger\b|\barms?\b")

# A key the question names: a DOI, an arXiv id, `@key`, or a bib-style key
# (letters, a four-digit year, optional letters: batzner2022nequip).
_KEY = re.compile(
    r"(?<![\w/.])(10\.\d{4,9}/[^\s\"'“”]+[^\s\"'“”.,;:?]"
    r"|\d{4}\.\d{4,5}(?:v\d+)?"
    r"|@[A-Za-z][\w:.-]*\w"
    r"|[A-Za-z][A-Za-z-]*\d{4}[A-Za-z0-9]*)"
)
_QUOTED = re.compile(r"[\"“]([^\"”]{12,})[\"”]")


def asks_support(question: str) -> bool:
    """Whether a question asks if a cited source backs a sentence."""
    q = _QUOTED.sub(" ", question).lower()  # the quoted sentence is data, not intent
    if _RUNS.search(q):
        return False
    if any(p in q for p in _ALWAYS):
        return True
    asks = any(p in q for p in _ASKS_SUPPORT)
    if asks and any(p in q for p in _CITED_THING):
        return True
    # A quoted sentence and a named key ARE the cited source being asked
    # about, so a bare "say" is enough there -- and only there: "what do the
    # papers I read say about X" is a graph question, not this one.
    return (asks or re.search(r"\bsays?\b", q) is not None) and all(sentence_and_key(question))


def sentence_and_key(question: str) -> tuple[str | None, str | None]:
    """The quoted sentence and the cited key a question carries, if any."""
    quoted = _QUOTED.search(question)
    sentence = quoted.group(1).strip() if quoted else None
    outside = _QUOTED.sub(" ", question)
    key = _KEY.search(outside)
    return sentence, (key.group(1).lstrip("@") if key else None)


def _reply(ok: bool, answer: str, *, caveat=None, options=(), tool_used=None) -> dict:
    return {
        "ok": ok,
        "answer": answer,
        "refs": [],
        "caveat": caveat,
        "options": list(options),
        "tool_used": tool_used,
    }


def _summary(out: dict) -> str:
    """The worst rows first, one line each: verdict, key, level, reason, quote."""
    lines = [out.get("message", "").split(". ")[0] + "."]
    for r in out.get("results", []):
        line = f"{r['verdict']} -- {r['key']} [{r['text_level'] or 'no text'}]: {r['reason']}"
        if r.get("quote"):
            line += f' Source: "{r["quote"][:200]}"'
        lines.append(line)
    return "\n".join(lines)


def support_answer(question: str, path: str | None = None) -> dict | None:
    """The Answer for a citation-support question, or None when it is not one.

    `path` (a draft) wins; otherwise the question must quote the sentence and
    name the key, and a question that does neither gets told exactly that.
    """
    if not asks_support(question):
        return None
    from attestation.mcp.citation_support import _support

    sentence, key = sentence_and_key(question)
    if not path and not (sentence and key):
        return _reply(
            False,
            "Which draft or sentence? Pass path= (a Markdown or .tex draft), or quote the"
            ' sentence and name its key, e.g. does batzner2022nequip support "NequIP ...".',
            options=["cite.support"],
        )
    out = _support(path=path, sentence=None if path else sentence, cite=None if path else key)
    if not out.get("ok"):
        return _reply(False, out.get("message", ""), options=["cite.support"])
    notes = "; ".join(out.get("notes") or []) or None
    return _reply(True, _summary(out), caveat=notes, tool_used="cite.support")
