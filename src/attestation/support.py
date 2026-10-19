"""Does the cited paper actually say what the citing sentence says it does?

`cite.check` answers a narrower question -- does the key resolve -- and says
so in every reply, because "the key resolves" was being read as "the paper
backs this". This module answers the wider one, per citing sentence:

    1. resolve the key in the library store (then the disk readers);
    2. take the best text Attestation holds for it -- full text when
       `attest library fulltext` fetched a body, else the abstract, else the
       title alone -- and record which (`text_level`);
    3. rank the source's sentences against the citing sentence (word overlap,
       reranked by embedding similarity when the embedder answers);
    4. judge: a small local model reads the top passages and picks a verdict
       and a quote, and a deterministic check of numbers and names runs
       either way -- it is the whole judge when no model is reachable.

Five verdicts, in the spirit of `claims.py`'s, where the distinctions are the
point:

    supported            a passage states what the sentence states
    partially_supported  a passage backs part of it; a number or name is missing
    contradicted         a passage says otherwise, or the sentence keeps the
                         source's number but changes what it measures
    not_found            the source text available does not discuss it
    source_unavailable   the key resolves to nothing with any text

`not_found` at `text_level: "abstract"` is weaker than at `"full"`: the body
may say it. Every result carries the level so a reader can tell.

**Every quote is an exact substring of the source text**, checked here with
`in`, not assumed: a model's quote that does not match is replaced by the
passage it pointed at, and `quote_from` says which happened. A citation
checker that invented a quotation would be the failure it exists to catch.

**Why the deterministic check overrides the model on numbers.** Measured on
the motivating case (see `support_text`'s module docstring), a 2B judge reads
"three orders of magnitude" in both texts and answers SUPPORTS. Word overlap
agrees with it. Only the phrase the number is attached to disagrees, so that
comparison is code, and a disagreement it finds wins over the model.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from attestation import support_text as st

log = logging.getLogger(__name__)


class SupportKind(StrEnum):
    """The closed set of `cite.support` verdicts (see the module docstring)."""

    SUPPORTED = "supported"
    PARTIAL = "partially_supported"
    CONTRADICTED = "contradicted"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "source_unavailable"


JUDGES = ("auto", "llm", "none")
TOP_PASSAGES = 3  # passages shown to the judge: a 2B model loses track past ~3
PREFILTER = 12  # sentences embedded per citing sentence, after word overlap
MAX_QUOTE_CHARS = 300
MAX_PASSAGE_CHARS = 600
SUPPORTED_COVERAGE = 0.6
PARTIAL_COVERAGE = 0.34
# A negated clause counts against the sentence when it holds a third of the
# sentence's content words ("dispensing with recurrence and convolutions").
NEGATED_SHARE = 0.33


# ---------------------------------------------------------------------------
# the source
# ---------------------------------------------------------------------------


@dataclass
class SourceText:
    """The best text Attestation holds for one cited key.

    `level` is `full`, `abstract`, `title`, or None when nothing resolved.
    `spans` are sentence offsets into `text`, computed once per source.
    """

    key: str
    text: str = ""
    level: str | None = None
    title: str | None = None
    reference: dict | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if self.text and not self.spans:
            self.spans = st.sentence_spans(self.text)


def _store_text(conn, row) -> tuple[str, str]:
    """(text, level) for a library row: full text, else abstract, else title.

    The abstract is prepended to a body that does not already contain it --
    PMC XML often carries only the body sections, and a claim citing a
    paper's headline result most often paraphrases the abstract.
    """
    body = conn.execute(
        "SELECT text FROM reference_fulltext WHERE reference_id = ?", (row["id"],)
    ).fetchone()
    abstract = (row["abstract"] or "").strip()
    if body is not None and (body["text"] or "").strip():
        text = body["text"]
        if abstract and abstract[:200] not in text:
            text = f"{abstract}\n\n{text}"
        return text, "full"
    if abstract:
        return abstract, "abstract"
    return (row["title"] or "").strip(), "title"


def _store_row(conn, key: str | None):
    """`library.lookup_row`, plus the arXiv id inside a DataCite arXiv DOI
    (10.48550/arXiv.NNNN), which a feed row is stored under but which
    lookup_row's arXiv branch does not unwrap when handed the DOI."""
    from attestation import library

    if conn is None or not key:
        return None
    row = library.lookup_row(conn, key)
    inner = library.arxiv_from_doi(key)
    return row if row is not None or not inner else library.lookup_row(conn, inner)


def resolve_source(conn, key: str, resolver=None) -> SourceText:
    """The library row for `key` (by identity, DOI, arXiv id or bib key),
    else whatever the disk readers know, else an empty `SourceText`.

    A key only a `.bib` file knows is retried against the store by the DOI
    and arXiv id that file gives it, so a paper synced from the feed under
    another key still contributes its abstract.
    """
    from attestation import library

    row = _store_row(conn, key)
    ref = None
    if row is None and resolver is not None:
        ref = resolver.lookup(key)
        if ref is not None:
            row = _store_row(conn, ref.doi) or _store_row(conn, ref.arxiv_id)
    if row is not None:
        text, level = _store_text(conn, row)
        reference = library.to_reference(conn, row).to_row()
        return SourceText(key, text, level if text else None, row["title"], reference)
    if ref is not None and ref.title:
        return SourceText(key, ref.title, "title", ref.title, ref.to_row())
    return SourceText(key)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


class Retriever:
    """Ranks a source's sentences against a claim.

    Word overlap picks `PREFILTER` candidates; the embedder (when it answers)
    reorders them. The first embedder failure switches it off for the rest of
    the run and is reported once in `note`, rather than costing a timeout per
    sentence.
    """

    def __init__(self, embedder=None, enabled: bool = True):
        self._embedder = embedder
        self.enabled = enabled
        self.note: str | None = None
        self.used = False
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def mode(self) -> str:
        """`embedding` once the embedder has reranked something, else `lexical`."""
        return "embedding" if self.used and self.note is None else "lexical"

    def _get(self):
        if self._embedder is None:
            from attestation.embed import Embedder

            self._embedder = Embedder()
        return self._embedder

    def _vectors(self, claim: str, source: SourceText, spans) -> list[float] | None:
        import httpx

        from attestation.ports import BackendUnreachable

        try:
            emb = self._get()
            q = emb.embed_query(claim)
            todo = [s for s in spans if (source.key, *s) not in self._cache]
            pairs = [(source.title or "", source.text[s:e]) for s, e in todo]
            for span, vec in zip(todo, emb.embed_documents(pairs), strict=True):
                self._cache[(source.key, *span)] = vec
        except (httpx.HTTPError, BackendUnreachable, ValueError, OSError) as exc:
            self.note = f"embedder unavailable ({type(exc).__name__}); ranked by word overlap"
            log.info("support: %s", self.note)
            return None
        self.used = True
        return [float(np.dot(q, self._cache[(source.key, *s)])) for s in spans]

    def rank(self, claim: str, source: SourceText) -> list[tuple[tuple[int, int], float]]:
        """(span, score) for the source's sentences, best first."""
        scored = [(s, st.coverage(claim, source.text[s[0] : s[1]])) for s in source.spans]
        scored.sort(key=lambda x: -x[1])
        top = scored[:PREFILTER]
        if self.enabled and self.note is None and len(top) > 1:
            sims = self._vectors(claim, source, [s for s, _ in top])
            if sims is not None:
                top = [(s, cov + sim) for (s, cov), sim in zip(top, sims, strict=True)]
                top.sort(key=lambda x: -x[1])
        return top


# ---------------------------------------------------------------------------
# deterministic evidence
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """What code alone can say about one (sentence, source) pair."""

    coverage: float
    best: tuple[int, int] | None
    findings: list[st.QuantityFinding]
    missing_names: list[str]
    negated: list[str] = field(default_factory=list)

    @property
    def problem(self) -> st.QuantityFinding | None:
        """The first number the source uses differently, if any."""
        return next((f for f in self.findings if f.kind in ("meaning", "value")), None)

    @property
    def gaps(self) -> list[str]:
        """Numbers and names in the sentence that the source never states."""
        nums = [f.detail for f in self.findings if f.kind == "missing"]
        names = [f"{n!r} is not named in the source" for n in self.missing_names]
        return nums + names

    def checks(self) -> list[str]:
        """Short tags for what the deterministic pass found."""
        out = [f"number:{f.kind}" for f in self.findings]
        out += [f"negation:{w}" for w in self.negated]
        return out + [f"name-missing:{n}" for n in self.missing_names]


def gather_evidence(claim: str, source: SourceText, ranked) -> Evidence:
    """Overlap of the best one or two passages, plus the number and name checks."""
    best = ranked[0][0] if ranked else None
    cov = st.coverage(claim, source.text[best[0] : best[1]]) if best else 0.0
    if len(ranked) > 1:
        pair = " ".join(source.text[s:e] for (s, e), _ in ranked[:2])
        cov = max(cov, 0.9 * st.coverage(claim, pair))
    findings = st.quantity_findings(claim, source.text, source.spans)
    low = source.text.lower()
    names = [n for n in st.entities(claim) if n.lower() not in low]
    return Evidence(cov, best, findings, names, _negated(claim, source, best))


def _negated(claim: str, source: SourceText, best) -> list[str]:
    """Negating words whose clause in the best passage holds a real share of
    the claim, when the claim itself negates nothing."""
    if best is None or st.negations(claim):
        return []
    scopes = st.negated_scopes(source.text[best[0] : best[1]])
    return sorted({cue for cue, scope in scopes if st.coverage(claim, scope) >= NEGATED_SHARE})


def _cap_partial(verdict: SupportKind, why: str, ev: Evidence, level: str | None):
    """Downgrade `supported` to partial for a gap or a title-only source."""
    if verdict != SupportKind.SUPPORTED:
        return verdict, why
    if ev.gaps:
        return SupportKind.PARTIAL, f"{why}; but {ev.gaps[0]}"
    if ev.negated:
        # Overlap cannot read polarity, and measured on gemma4:e2b neither
        # could the judge: it answered SUPPORTS for "networks require massive
        # training sets" against "challenging the belief that networks require
        # massive training sets", naming the challenge in its own reason.
        cue = ev.negated[0]
        return SupportKind.PARTIAL, f"{why}; but the passage says it with {cue!r} -- read it"
    if level == "title":
        return SupportKind.PARTIAL, f"{why}; only the title is available"
    return verdict, why


Span = tuple[int, int] | None


def deterministic_verdict(ev: Evidence, level: str | None) -> tuple[SupportKind, str, Span]:
    """(verdict, reason, quote span) from overlap and the number/name checks."""
    problem = ev.problem
    if problem is not None:
        return (
            SupportKind.CONTRADICTED,
            f"right number, wrong meaning: {problem.detail}",
            (problem.span or ev.best),
        )
    pct = f"{ev.coverage:.0%} of the sentence's content words"
    if ev.coverage >= SUPPORTED_COVERAGE:
        verdict, why = SupportKind.SUPPORTED, f"the passage shares {pct}"
    elif ev.coverage >= PARTIAL_COVERAGE:
        verdict, why = SupportKind.PARTIAL, f"the passage shares only {pct}"
    else:
        verdict, why = SupportKind.NOT_FOUND, f"no passage shares more than {pct}"
    verdict, why = _cap_partial(verdict, why, ev, level)
    match = next((f.span for f in ev.findings if f.kind == "match" and f.span), None)
    return verdict, why, (match if verdict == SupportKind.SUPPORTED and match else ev.best)


# ---------------------------------------------------------------------------
# the model judge
# ---------------------------------------------------------------------------

LLM_LABELS = {
    "SUPPORTS": SupportKind.SUPPORTED,
    "PARTIAL": SupportKind.PARTIAL,
    "CONTRADICTS": SupportKind.CONTRADICTED,
    "NOT_IN_TEXT": SupportKind.NOT_FOUND,
}

# Evidence before verdict: the passage number and the copied quote come first
# so a small model commits to WHAT it read before deciding what it means.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "passage": {"type": "integer"},
        "quote": {"type": "string"},
        "verdict": {"type": "string", "enum": list(LLM_LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["passage", "quote", "verdict", "reason"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTION = """You check whether a paper supports a sentence that cites it.
Use ONLY the numbered passages from the paper. Do not use outside knowledge.

Pick the passage that best matches the sentence and copy a quote from it word for word.
Then choose one verdict:
SUPPORTS - the passage states what the sentence states.
PARTIAL - the passage backs part of the sentence, but not all of it.
CONTRADICTS - the passage says the opposite, or uses the same number for a different thing.
NOT_IN_TEXT - no passage talks about what the sentence claims.

A passage that denies, rejects or argues against the sentence is CONTRADICTS, even when it
uses the same words. Check numbers carefully: the same number measuring a different
quantity is CONTRADICTS.
Give a reason in one short sentence."""


def judge_messages(claim: str, source: SourceText, spans) -> list[dict]:
    """The ONE renderer of the judge prompt (the tagging and explanation
    prompts follow the same rule), so an eval and the tool send the same text."""
    passages = "\n".join(
        f"[{i}] {source.text[s:e][:MAX_PASSAGE_CHARS]}" for i, (s, e) in enumerate(spans, 1)
    )
    kind = {"full": "full text", "abstract": "abstract", "title": "title only"}[source.level or ""]
    user = (
        f"Sentence: {claim}\n\nPaper: {source.title or source.key} ({kind})\nPassages:\n{passages}"
    )
    return [
        {"role": "system", "content": JUDGE_INSTRUCTION},
        {"role": "user", "content": user},
    ]


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` sits in `text`: exactly, else ignoring case and spacing.

    The second form still returns the SOURCE's own characters (the span is
    of `text`), so the quote reported is always the source verbatim.
    """
    import re

    quote = quote.strip().strip("\"'").strip()
    if len(quote) < 12:
        return None
    at = text.find(quote)
    if at >= 0:
        return at, at + len(quote)
    words = quote.split()
    m = re.search(r"\s+".join(re.escape(w) for w in words), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


class Judge:
    """The model half of the verdict, with its failure policy.

    `mode` is `auto` (use the model, fall back to the deterministic check
    when it is unreachable), `llm` (unreachable is an error) or `none`. The
    first unreachable call in `auto` switches the model off for the rest of
    the run and records why in `note`; a malformed REPLY affects only that
    one sentence.
    """

    def __init__(self, mode: str = "auto", chat_fn: Callable | None = None, label: str = ""):
        if mode not in JUDGES:
            raise ValueError(f"judge must be one of {', '.join(JUDGES)}, not {mode!r}")
        self.mode = mode
        self._chat_fn = chat_fn
        self._label = label
        self.note: str | None = None

    @property
    def active(self) -> bool:
        """Whether the next sentence will be put to the model."""
        return self.mode != "none" and self.note is None

    @property
    def label(self) -> str:
        """`llm:<model>` -- resolved once, on first use."""
        if not self._label:
            from attestation import llm

            try:
                self._label = f"llm:{llm.chat_target().model}"
            except (RuntimeError, OSError, ValueError, KeyError):
                self._label = "llm"
        return self._label

    def _call(self, messages):
        if self._chat_fn is None:
            from attestation.llm import default_chat_fn

            self._chat_fn = default_chat_fn
        return self._chat_fn(messages, JUDGE_SCHEMA)

    def ask(self, claim: str, source: SourceText, spans) -> dict | None:
        """The model's parsed answer, or None (and why, in `note` or the log)."""
        import httpx

        from attestation.ports import BackendUnreachable

        try:
            reply = self._call(judge_messages(claim, source, spans))
        except (httpx.HTTPError, BackendUnreachable, OSError) as exc:
            if self.mode == "llm":
                raise
            self.note = f"model unavailable ({type(exc).__name__}: {str(exc)[:120]})"
            return None
        except (ValueError, KeyError, TypeError) as exc:
            log.info("support: unusable judge reply: %s", exc)
            return None
        return reply if _valid(reply, len(spans)) else None


def _valid(reply, n: int) -> bool:
    return (
        isinstance(reply, dict)
        and reply.get("verdict") in LLM_LABELS
        and isinstance(reply.get("passage"), int)
        and isinstance(reply.get("quote"), str)
        and n > 0
    )


def model_verdict(reply: dict, ev: Evidence, source: SourceText, spans):
    """(verdict, reason, quote span, quote_from) from a valid model reply,
    with the deterministic number check allowed to overrule it."""
    problem = ev.problem
    if problem is not None and LLM_LABELS[reply["verdict"]] != SupportKind.CONTRADICTED:
        why = f"right number, wrong meaning: {problem.detail} (the model said {reply['verdict']})"
        return SupportKind.CONTRADICTED, why, problem.span or ev.best, "check"
    verdict = LLM_LABELS[reply["verdict"]]
    why = str(reply.get("reason") or "").strip()[:200] or reply["verdict"].lower()
    verdict, why = _cap_partial(verdict, why, ev, source.level)
    located = locate_quote(reply["quote"], source.text)
    if located is not None:
        return verdict, why, located, "model"
    pick = min(max(reply["passage"], 1), len(spans)) - 1
    return verdict, why, spans[pick], "passage"


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class SupportResult:
    """One (citing sentence, key) verdict, with the quote that decided it."""

    path: str
    line: int
    sentence: str
    key: str
    verdict: SupportKind
    reason: str
    text_level: str | None = None
    quote: str | None = None
    quote_offset: int | None = None
    quote_verified: bool = False
    quote_from: str | None = None
    judge: str = "deterministic"
    title: str | None = None
    checks: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        """The wire projection `cite.support` and `attest support --json` emit.

        Every free-text field is capped, so a row is bounded whatever the
        draft, the paper or the model wrote (the quote by `MAX_QUOTE_CHARS`).
        """
        return {
            "where": f"{self.path[-80:]}:{self.line}",
            "sentence": self.sentence[:200],
            "key": self.key[:60],
            "title": (self.title or "")[:100] or None,
            "verdict": str(self.verdict),
            "text_level": self.text_level,
            "quote": self.quote,
            "quote_verified": self.quote_verified,
            "reason": self.reason[:200],
            "judge": self.judge[:40],
            "checks": [c[:40] for c in self.checks[:4]],
        }


def _quote(source: SourceText, span) -> tuple[str | None, int | None, bool]:
    """The source text at `span`, clipped at a word boundary, and whether it
    really is a substring of the source -- checked, not assumed."""
    if not span or not source.text:
        return None, None, False
    start, end = span
    if end - start > MAX_QUOTE_CHARS:
        cut = source.text.rfind(" ", start, start + MAX_QUOTE_CHARS)
        end = cut if cut > start else start + MAX_QUOTE_CHARS
    quote = source.text[start:end]
    return quote, start, bool(quote) and quote in source.text


@dataclass
class Checker:
    """The resolver, retriever and judge one run shares, plus a source cache."""

    conn: object = None
    resolver: object = None
    judge: Judge = field(default_factory=Judge)
    retriever: Retriever = field(default_factory=Retriever)
    _sources: dict = field(default_factory=dict)

    def source(self, key: str) -> SourceText:
        """`resolve_source`, once per key per run."""
        if key not in self._sources:
            self._sources[key] = resolve_source(self.conn, key, self.resolver)
        return self._sources[key]

    def check(self, sentence: str, key: str, path: str = "<sentence>", line: int = 1):
        """Verdict for one sentence citing one key."""
        source = self.source(key)
        base = (path, line, sentence, key)
        if not source.text:
            why = (
                f"{key!r} resolves to {source.title!r} but no text is stored for it"
                if source.reference
                else f"no library row or configured source has {key!r}"
            )
            return SupportResult(*base, verdict=SupportKind.UNAVAILABLE, reason=why)
        ranked = self.retriever.rank(sentence, source)
        ev = gather_evidence(sentence, source, ranked)
        spans = [s for s, _ in ranked[:TOP_PASSAGES]]
        reply = self.judge.ask(sentence, source, spans) if self.judge.active and spans else None
        if reply is not None:
            verdict, why, span, quote_from = model_verdict(reply, ev, source, spans)
            judge = self.judge.label
        else:
            verdict, why, span = deterministic_verdict(ev, source.level)
            quote_from, judge = "check", "deterministic"
        quote, offset, verified = _quote(source, span)
        return SupportResult(
            *base,
            verdict=verdict,
            reason=why,
            text_level=source.level,
            quote=quote,
            quote_offset=offset,
            quote_verified=verified,
            quote_from=quote_from,
            judge=judge,
            title=source.title,
            checks=ev.checks(),
        )


# ---------------------------------------------------------------------------
# drafts
# ---------------------------------------------------------------------------

DRAFT_SUFFIXES = (".md", ".markdown", ".tex")


def citing_sentences(path: Path) -> list[st.CitingSentence]:
    """Every citing sentence in a Markdown or LaTeX file, or in every such
    file under a directory (hidden directories skipped, as `claims` does)."""
    path = Path(path).expanduser()
    if path.is_file():
        text = path.read_text(errors="replace")
        reader = st.tex_citations if path.suffix == ".tex" else st.markdown_citations
        return reader(text, str(path))
    out: list[st.CitingSentence] = []
    for f in sorted(p for p in path.rglob("*") if p.suffix in DRAFT_SUFFIXES):
        if not any(part.startswith(".") for part in f.relative_to(path).parts):
            out.extend(citing_sentences(f))
    return out


# Worst first, so a capped listing shows what needs fixing.
SEVERITY = (
    SupportKind.CONTRADICTED,
    SupportKind.NOT_FOUND,
    SupportKind.PARTIAL,
    SupportKind.UNAVAILABLE,
    SupportKind.SUPPORTED,
)


def run(checker: Checker, found: list[st.CitingSentence], limit: int | None = None) -> dict:
    """Check every (sentence, key) pair in `found`, up to `limit` pairs."""
    pairs = [(c, k) for c in found for k in c.keys]
    todo = pairs if limit is None else pairs[:limit]
    results = [checker.check(c.sentence, k, c.path, c.line) for c, k in todo]
    return {
        "n_sentences": len(found),
        "n_citations": len(pairs),
        "n_skipped": len(pairs) - len(todo),
        "counts": dict(Counter(str(r.verdict) for r in results)),
        "results": results,
        "judges": sorted({r.judge for r in results}),
        "retrieval": checker.retriever.mode,
        "notes": [n for n in (checker.judge.note, checker.retriever.note) if n],
    }


def failing(results: list[SupportResult], fail_on) -> int:
    """How many results carry a verdict in `fail_on` (the gate)."""
    return sum(1 for r in results if r.verdict in set(fail_on))


def by_severity(results: list[SupportResult]) -> list[SupportResult]:
    """Results ordered worst verdict first, then by position in the draft."""
    rank = {k: i for i, k in enumerate(SEVERITY)}
    return sorted(results, key=lambda r: (rank[r.verdict], r.path, r.line))
