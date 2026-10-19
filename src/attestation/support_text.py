"""Pure text work for `support.py`: citing sentences, source sentences, quantities.

Nothing here opens a database or calls a model. It is split from `support.py`
so the parts a reviewer most wants to read in isolation -- how a sentence is
found, how a number is read, when two numbers count as "the same claim" --
are testable with strings alone.

**Right number, wrong meaning.** The failure this module exists to catch is a
draft that keeps a source's number and changes what it measures:

    draft:  NequIP outperforms invariant models by up to three orders of magnitude
    source: NequIP outperforms existing models with up to three orders of
            magnitude fewer training data

A word-overlap score calls that a near-perfect match, and so does a 2B judge
reading the two side by side. What differs is the phrase the number is
attached to, so `quantity_findings` compares the words that FOLLOW a number in
the source with the draft sentence: when most of them are absent, the number
is the source's but the claim is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

STOPWORDS = frozenset(
    """a an the and or but if then than that this these those to of in on for at by
    with without from into onto over under as is are was were be been being it its
    we our us they their them he she his her which who whom whose what when where
    while also not no nor so such can could may might must shall should will would
    do does did done has have had having up down out about across after before
    between both each few more most other some any all only own same very just
    via per using use used show shows shown here there one's etc et al""".split()
)

# Words that end a number's attached phrase: a preposition or conjunction
# introduces SCOPE ("12% on QM9", "10x over DFT"), not what was measured, and
# treating the scope as meaning made every benchmark name a false alarm.
_PHRASE_BREAK = STOPWORDS | {"compared", "relative", "versus", "vs", "respectively"}


def stem(word: str) -> str:
    """A crude suffix strip, enough to match `outperforms`/`outperform` and
    `models`/`model`. Not a real stemmer: this only has to make overlap
    counts agree, and a dependency for that would outweigh the gain."""
    w = word.lower()
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "ly", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def content_stems(text: str) -> set[str]:
    """The stemmed non-stopwords of `text`, lowercased."""
    return {stem(w) for w in _WORD.findall(text) if w.lower() not in STOPWORDS}


def coverage(claim: str, passage: str) -> float:
    """Fraction of the claim's content words that the passage contains."""
    want = content_stems(claim)
    if not want:
        return 0.0
    return len(want & content_stems(passage)) / len(want)


# Words that turn a passage against what it mentions. Word overlap cannot see
# them: "challenging the widely held belief that deep neural networks require
# massive training sets" shares every content word with a sentence claiming
# networks DO require them.
_NEGATION = frozenset(
    """not no never without neither nor unlike challenging challenge contrary
    dispensing instead rather fail fails failed cannot lack lacks absence refute
    refutes refuted contradict contradicts""".split()
)
# Stems too, so "dispenses"/"challenges" in a sentence count like the
# "dispensing"/"challenging" in its source -- otherwise a sentence that keeps
# the negation would be read as dropping it.
_NEGATION_STEMS = frozenset(stem(w) for w in _NEGATION if len(w) > 5)


def _negating(word: str) -> bool:
    w = word.lower()
    return w.endswith("n't") or w in _NEGATION or stem(w) in _NEGATION_STEMS


def negations(text: str) -> set[str]:
    """The negating or contrasting words in `text`, lowercased."""
    return {w.lower() for w in _WORD.findall(text) if _negating(w)}


def negated_scopes(text: str) -> list[tuple[str, str]]:
    """(cue, scope) for each negating word: the scope runs to the clause's end.

    The scope, not the sentence, is what is negated: "outperforms existing
    models with ... fewer training data, challenging the belief that networks
    require massive training sets" negates the second clause only, and a claim
    restating the first must not be read as contradicting it.
    """
    out = []
    for m in _WORD.finditer(text):
        word = m.group(0).lower()
        if _negating(word):
            scope = re.split(r"[,.;:()]", text[m.end() : m.end() + 160], maxsplit=1)[0]
            out.append((word, scope))
    return out


def _is_name(word: str, first: bool) -> bool:
    if re.fullmatch(r"[\d.,]+", word) or word.lower() in STOPWORDS:
        return False
    acronym = sum(c.isupper() for c in word) >= 2
    capital = word[0].isupper() and not first
    return acronym or capital or any(c.isdigit() for c in word)


def entities(sentence: str) -> list[str]:
    """Names a claim depends on: acronyms, mid-sentence capitals, and tokens
    carrying digits (`MD17`, `WMT`, `English-to-German`, `rMD17`).

    The first word is skipped unless it is an acronym, since a capital there
    is grammar. Pure numbers are quantities, not names, and are left to
    `quantities`.
    """
    words = [m.group(0) for m in _WORD.finditer(sentence)]
    names = [w for i, w in enumerate(words) if _is_name(w, i == 0)]
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# sentences
# ---------------------------------------------------------------------------

# Abbreviations whose period is not a sentence end. "et al." matters most:
# it is in nearly every citing sentence.
_ABBREV = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs|\bFig|\bFigs|\bEq|\bEqs|\bRef|\bRefs|\bSec"
    r"|\bTab|\bNo|\bpp?|\bapprox|\bresp|\bDr|\bProf|\bSt|\b[A-Z])\.$"
)
_BOUNDARY = re.compile(r"[.!?](?=\s+[A-Z0-9(\[\\@])|\n[ \t]*\n|\n(?=[ \t]*(?:#|[-*+] |\d+\. |\|))")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, as offsets into `text` itself.

    Offsets rather than strings so a quote taken from a span is, by
    construction, an exact substring of the source -- the property
    `support.py` verifies on every result rather than assumes.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end() if m.group(0)[0] in ".!?" else m.start()
        if m.group(0)[0] == "." and _ABBREV.search(text[max(start, end - 12) : end]):
            continue
        spans.append((start, end))
        start = m.end()
    spans.append((start, len(text)))
    return [_trim(text, s, e) for s, e in spans if text[s:e].strip()]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# ---------------------------------------------------------------------------
# citing sentences in a draft
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitingSentence:
    """One sentence of a draft and the keys it cites, markers removed."""

    path: str
    line: int
    sentence: str
    keys: tuple[str, ...]


_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
_CLAIM_CITE = re.compile(r"<!--\s*claim:[^>]*?\bcite=([^\s>]+)[^>]*?-->", re.DOTALL)
_PANDOC_GROUP = re.compile(r"\[[^\[\]]*?@[^\[\]]*\]")
_PANDOC_KEY = re.compile(r"(?<![\w@])-?@(?:\{([^}]+)\}|([A-Za-z0-9_][\w:.#$%&+?<>~/-]*))")
_TEX_CITE = re.compile(r"\\[A-Za-z]*cite[A-Za-z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}")
_TEX_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
_TEX_STYLE = re.compile(r"\\(?:emph|textbf|textit|texttt|mathrm|text)\{([^{}]*)\}")
_TEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?")


def _blank(m: re.Match) -> str:
    """Same-length spaces (newlines kept), so offsets and line numbers survive."""
    return re.sub(r"[^\n]", " ", m.group(0))


def _keys_in(fragment: str) -> list[str]:
    keys = []
    for braced, bare in _PANDOC_KEY.findall(fragment):
        key = braced or bare.rstrip(".,;:")
        if key and key not in keys:
            keys.append(key)
    return keys


def _clean_markdown(sentence: str) -> str:
    out = _PANDOC_GROUP.sub("", sentence)
    out = _PANDOC_KEY.sub("", out)
    out = re.sub(r"[*_`]+", "", out)
    return " ".join(re.sub(r"\s+([.,;:])", r"\1", out).split())


def _clean_tex(sentence: str) -> str:
    out = _TEX_CITE.sub("", sentence).replace("~", " ")
    for _ in range(3):  # nested \emph{\textbf{...}} peels one level per pass
        out = _TEX_STYLE.sub(r"\1", out)
    out = _TEX_COMMAND.sub("", out).replace("{", "").replace("}", "")
    return " ".join(re.sub(r"\s+([.,;:])", r"\1", out).split())


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def markdown_citations(text: str, path: str = "<sentence>") -> list[CitingSentence]:
    """Sentences citing with Pandoc syntax (`[@key]`, `[see @a; @b]`, `@key`)
    or carrying a `<!-- claim: ... cite=key -->` annotation.

    A claim annotation cites the sentence ending just before it -- the
    convention `examples/citations/DRAFT.md` follows, where the comment sits
    on the line after the prose it backs. Code fences and other comments are
    blanked first, so an email address in a code block is not a citation.
    """
    masked = _FENCE.sub(_blank, text)
    claim_cites = [(m.start(), m.group(1)) for m in _CLAIM_CITE.finditer(masked)]
    masked = _COMMENT.sub(_blank, masked)
    spans = sentence_spans(masked)
    found = {i: keys for i, (s, e) in enumerate(spans) if (keys := _keys_in(masked[s:e]))}
    _attach_claim_cites(found, spans, claim_cites)
    return _citing(found, spans, masked, text, path, _clean_markdown)


def _attach_claim_cites(found: dict, spans: list, claim_cites: list) -> None:
    """Add each `claim: ... cite=` key to the sentence ending just before it."""
    for at, key in claim_cites:
        before = [i for i, (_s, e) in enumerate(spans) if e <= at]
        if before and key not in found.setdefault(before[-1], []):
            found[before[-1]].append(key)


def _citing(found: dict, spans: list, masked: str, text: str, path: str, clean):
    """`CitingSentence`s for the sentence indexes in `found`, in order."""
    out = []
    for i, keys in sorted(found.items()):
        s, e = spans[i]
        if cleaned := clean(masked[s:e]):
            out.append(CitingSentence(path, _line(text, s), cleaned, tuple(keys)))
    return out


def tex_citations(text: str, path: str = "<sentence>") -> list[CitingSentence]:
    """Sentences carrying `\\cite{...}` (and natbib/biblatex variants:
    `\\citep`, `\\citet`, `\\parencite`, `\\autocite`, with optional notes)."""
    masked = _TEX_COMMENT.sub(_blank, text)
    spans = sentence_spans(masked)
    found = {i: keys for i, (s, e) in enumerate(spans) if (keys := _tex_keys(masked[s:e]))}
    return _citing(found, spans, masked, text, path, _clean_tex)


def _tex_keys(chunk: str) -> list[str]:
    keys = [k.strip() for group in _TEX_CITE.findall(chunk) for k in group.split(",")]
    return list(dict.fromkeys(k for k in keys if k))


# ---------------------------------------------------------------------------
# quantities
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "thirty": 30,
    "fifty": 50,
    "hundred": 100,
    "thousand": 1000,
    "million": 1e6,
    "billion": 1e9,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "twice": 2,
    "double": 2,
    "half": 0.5,
}
_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_QUANTITY = re.compile(
    rf"(?<![\w.])(?<!\w\()(?P<num>{_NUM})(?![\w.]*\d)"
    r"(?P<unit>\s*%|\s+percent\b|\s*[x×](?![A-Za-z])|-?\s*fold\b"
    r"|\s+orders?\s+of\s+magnitude\b|-(?P<hyph>[a-z]+)\b)?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"^(?:19|20)\d\d$")
# Read as numbers only with a unit: bare, they are grammar ("one of", "per
# second", "the first model", "half the time").
_NEEDS_UNIT = frozenset({"one", "first", "second", "third", "fourth", "fifth", "double", "half"})


@dataclass(frozen=True)
class Quantity:
    """A number in text, normalised: `value` in `unit`, plus where it sits.

    Units are coarse on purpose -- `%`, `x` (fold, times, orders of magnitude
    as a power of ten), a hyphenated suffix (`four-body` is 4 `body`), or ""
    for a bare count. `after` is the phrase the number attaches to.
    """

    value: float
    unit: str
    start: int
    end: int
    after: tuple[str, ...]


def _unit(m: re.Match, value: float) -> tuple[float, str]:
    unit = (m.group("unit") or "").strip().lower()
    if not unit:
        return value, ""
    if unit.endswith("%") or "percent" in unit:
        return value, "%"
    if "magnitude" in unit:
        return 10.0**value, "x"
    if unit in ("x", "×") or "fold" in unit:
        return value, "x"
    return value, (m.group("hyph") or "").lower()


def _after(text: str, end: int, limit: int = 3) -> tuple[str, ...]:
    """Up to `limit` words right after a number, stopping at punctuation or
    a scope word (see `_PHRASE_BREAK`)."""
    words: list[str] = []
    for m in re.finditer(r"\s*([A-Za-z][\w-]*|[^\sA-Za-z])", text[end : end + 80]):
        w = m.group(1)
        if not w[0].isalpha() or w.lower() in _PHRASE_BREAK or len(words) == limit:
            break
        words.append(w.lower())
    return tuple(words)


def quantities(text: str) -> list[Quantity]:
    """Every quantity in `text`. Years and reference numbers are skipped."""
    out = []
    for m in _QUANTITY.finditer(text):
        raw = m.group("num").lower().replace(",", "")
        bare = not m.group("unit")
        glued = m.end() < len(text) and text[m.end()].isalpha()  # 3BPA, ones
        if bare and (glued or _YEAR.match(raw) or raw in _NEEDS_UNIT):
            continue
        value = float(_NUMBER_WORDS.get(raw, 0) or raw)
        value, unit = _unit(m, value)
        if raw == "twice":
            unit = "x"
        out.append(Quantity(value, unit, m.start(), m.end(), _after(text, m.end())))
    return out


def same_value(a: float, b: float) -> bool:
    """Equal within rounding a writer could reasonably apply."""
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)


@dataclass(frozen=True)
class QuantityFinding:
    """What happened to one of the claim's quantities in the source.

    `kind` is `match` (same number, same meaning), `meaning` (same number, a
    different attached phrase or a different named thing), `value` (same unit,
    another number), or `missing`. `span` is the source sentence that decided
    it, so a verdict built from this can quote that sentence.
    """

    kind: str
    claim: Quantity
    span: tuple[int, int] | None
    detail: str


def _meaning_differs(occurrence: Quantity, claim_text: str) -> list[str]:
    """The source's attached words the claim lacks, if they are most of them."""
    have = content_stems(claim_text)
    missing = [w for w in occurrence.after if stem(w) not in have]
    return missing if occurrence.after and 2 * len(missing) > len(occurrence.after) else []


def _entity_elsewhere(
    claim_text: str, q: Quantity, sentence: str, others: list[tuple[str, list[Quantity]]]
) -> str | None:
    """A name the claim pairs with this number that the source pairs with a
    DIFFERENT number of the same unit, in another sentence."""
    low = sentence.lower()
    for name in entities(claim_text):
        if name.lower() in low:
            continue
        for other_text, other_qs in others:
            if other_text == sentence or name.lower() not in other_text.lower():
                continue
            if any(o.unit == q.unit and not same_value(o.value, q.value) for o in other_qs):
                return name
    return None


def _describe(q: Quantity, text: str) -> str:
    return text[q.start : q.end]


_Sentences = list[tuple[tuple[int, int], str, list[Quantity]]]


def _occurrences(q: Quantity, sentences: _Sentences):
    """(span, sentence, quantity) for each place the source states q's number."""
    return [
        (span, text, o)
        for span, text, qs in sentences
        for o in qs
        if o.unit == q.unit and same_value(o.value, q.value)
    ]


def _same_number(q: Quantity, claim_text: str, sentences: _Sentences) -> QuantityFinding | None:
    """`match` for the first occurrence of the claim's number used in the
    claim's sense, else `meaning` for the first one used in another, else None."""
    others = [(t, qs) for _span, t, qs in sentences]
    shown = _describe(q, claim_text)
    problems: list[QuantityFinding] = []
    for span, text, o in _occurrences(q, sentences):
        extra = _meaning_differs(o, claim_text)
        name = None if extra else _entity_elsewhere(claim_text, q, text, others)
        if not extra and not name:
            return QuantityFinding("match", q, span, f"{shown!r} appears with the same sense")
        why = (
            f"the source's {_describe(o, text)!r} is about '{' '.join(o.after)}'"
            if extra
            else f"the source gives {shown!r} for something other than {name}"
        )
        problems.append(QuantityFinding("meaning", q, span, why))
    return problems[0] if problems else None


def _other_value(q: Quantity, claim_text: str, sentences: _Sentences) -> QuantityFinding | None:
    """`value` when the source states a different number in the same unit,
    attached to the same thing: "10% fewer parameters" says nothing about a
    claimed "5% accuracy" gain, so it does not count."""
    if not q.unit:
        return None
    for span, text, qs in sentences:
        for o in qs:
            if o.unit == q.unit and not _meaning_differs(o, claim_text):
                said = _describe(o, text)
                why = f"claim says {_describe(q, claim_text)!r}, the source says {said!r}"
                return QuantityFinding("value", q, span, why)
    return None


def _finding_for(q: Quantity, claim_text: str, sentences: _Sentences) -> QuantityFinding:
    found = _same_number(q, claim_text, sentences) or _other_value(q, claim_text, sentences)
    if found is not None:
        return found
    shown = _describe(q, claim_text)
    return QuantityFinding("missing", q, None, f"{shown!r} does not appear in the source")


def quantity_findings(claim_text: str, source: str, spans: list[tuple[int, int]]):
    """One `QuantityFinding` per quantity in the claim, checked against every
    sentence of the source (not only the best-ranked one: the sentence that
    carries a number is often not the one that shares the most words)."""
    sentences = [((s, e), source[s:e], quantities(source[s:e])) for s, e in spans]
    sentences = [x for x in sentences if x[2]]
    return [_finding_for(q, claim_text, sentences) for q in quantities(claim_text)]
