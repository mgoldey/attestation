"""cite.support / `attest support`: does the cited paper say what the sentence says?

The sources in evals/support_cases.json are real arXiv abstracts. The model is
always a fake here; evals/run_support_eval.py measures a live one.
"""

import json
from pathlib import Path

import httpx
import pytest
from test_response_size import HARD_RESPONSE_CEILING

from attestation import support
from attestation import support_text as st
from attestation.cli import main
from attestation.db import get_db
from attestation.library import ReferenceRecord, lookup_row, upsert

CASES = Path(__file__).resolve().parent.parent / "evals" / "support_cases.json"
DATA = json.loads(CASES.read_text())
ABSTRACTS = {s["key"]: s for s in DATA["sources"]}
NEQUIP = ABSTRACTS["batzner2022nequip"]
WRONG_MEANING = "NequIP outperforms invariant models by up to three orders of magnitude."


def _sources() -> dict:
    return {
        k: support.SourceText(k, s["abstract"], "abstract", s["title"])
        for k, s in ABSTRACTS.items()
    }


def _checker(judge=None, retriever=None) -> support.Checker:
    checker = support.Checker(
        judge=judge or support.Judge("none"),
        retriever=retriever or support.Retriever(enabled=False),
    )
    checker._sources = _sources()
    return checker


def _seed(db: Path, *, fulltext: dict | None = None) -> None:
    conn = get_db(db)
    for s in DATA["sources"]:
        upsert(
            conn,
            ReferenceRecord(
                source="feed",
                source_key=s["arxiv_id"],
                bib_key=s["key"],
                title=s["title"],
                arxiv_id=s["arxiv_id"],
                abstract=s["abstract"],
            ),
        )
    upsert(conn, ReferenceRecord(source="feed", source_key="t", bib_key="titleonly", title="X"))
    for key, text in (fulltext or {}).items():
        rid = lookup_row(conn, key)["id"]
        conn.execute(
            "INSERT INTO reference_fulltext(reference_id, text, source, fetched_at)"
            " VALUES (?, ?, 'pmc-xml', '2026-10-17')",
            (rid, text),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# text: sentences, citations, quantities
# ---------------------------------------------------------------------------


def test_et_al_does_not_end_a_sentence_and_offsets_are_exact():
    text = "As Vaswani et al. showed, attention works. Fig. 2 agrees.\n\nNew para."
    spans = st.sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "As Vaswani et al. showed, attention works.",
        "Fig. 2 agrees.",
        "New para.",
    ]


def test_markdown_citations_read_pandoc_claims_and_skip_code_and_email():
    text = (
        "Intro sentence. NequIP is data efficient"
        " [@batzner2022nequip; see @kingma2015adam, p. 3].\n"
        "@vaswani2017attention introduced the Transformer. Mail me@example.com today.\n\n"
        "```\nnot a cite [@fake]\n```\n\n"
        "Distillation follows Hinton.\n"
        "<!-- claim: p/r metric=wer value=0.1 cite=hinton2015distilling -->\n"
    )
    found = st.markdown_citations(text, "d.md")
    assert [(c.line, c.keys) for c in found] == [
        (1, ("batzner2022nequip", "kingma2015adam")),
        (2, ("vaswani2017attention",)),
        (8, ("hinton2015distilling",)),
    ]
    assert found[0].sentence == "NequIP is data efficient."
    assert found[1].sentence == "introduced the Transformer."


def test_tex_citations_read_natbib_variants_and_ignore_comments():
    text = (
        "% \\cite{commented}\n"
        "SchNet avoids grids~\\citep[e.g.][p.~2]{schutt2017schnet, other}. "
        "Adam \\textit{rarely} needs tuning \\parencite{kingma2015adam}.\n"
    )
    found = st.tex_citations(text, "d.tex")
    assert [c.keys for c in found] == [("schutt2017schnet", "other"), ("kingma2015adam",)]
    assert found[1].sentence == "Adam rarely needs tuning."


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("up to three orders of magnitude fewer", (1000.0, "x")),
        ("a 12.5% gain", (12.5, "%")),
        ("using four-body messages", (4.0, "body")),
        ("a tenfold speedup of 10x", (10.0, "x")),
        ("28.4 BLEU", (28.4, "")),
    ],
)
def test_quantities_normalise_units(text, want):
    q = st.quantities(text)[-1]
    assert (q.value, q.unit) == want


def test_years_equivariance_labels_and_bare_ordinals_are_not_quantities():
    assert st.quantities("In 2014, E(3) models were the first of one kind; 3BPA too.") == []


def test_the_motivating_case_is_right_number_wrong_meaning():
    text = NEQUIP["abstract"]
    (finding,) = st.quantity_findings(WRONG_MEANING, text, st.sentence_spans(text))
    assert finding.kind == "meaning"
    assert "fewer training data" in finding.detail
    fixed = (
        "NequIP outperforms existing models with up to three orders of magnitude"
        " fewer training data."
    )
    assert [f.kind for f in st.quantity_findings(fixed, text, st.sentence_spans(text))] == ["match"]


def test_a_number_attached_to_another_named_task_is_a_meaning_mismatch():
    text = ABSTRACTS["vaswani2017attention"]["abstract"]
    spans = st.sentence_spans(text)
    swapped = "The Transformer achieves 41.8 BLEU on WMT 2014 English-to-German."
    assert st.quantity_findings(swapped, text, spans)[0].kind == "meaning"
    right = "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German."
    assert st.quantity_findings(right, text, spans)[0].kind == "match"


def test_a_scope_word_after_a_number_is_not_its_meaning():
    """'12% on QM9' attaches the number to QM9 as SCOPE; a claim that names a
    different benchmark must not be told the number means something else."""
    source = "The model reduces error by 12% on QM9 relative to SchNet."
    spans = st.sentence_spans(source)
    claim = "The model reduces error by 12%."
    assert st.quantity_findings(claim, source, spans)[0].kind == "match"


# ---------------------------------------------------------------------------
# deterministic verdicts over the real abstracts
# ---------------------------------------------------------------------------


def test_deterministic_judge_on_the_labelled_cases():
    """Pinned so a change that loses accuracy, or calls a contradicted
    sentence supported, is a decision rather than an accident. Measured
    2026-10-17: 15/20, no contradicted case called supported."""
    checker = _checker()
    results = {c["id"]: checker.check(c["sentence"], c["key"]) for c in DATA["cases"]}
    right = [
        cid
        for cid, c in zip(results, DATA["cases"], strict=True)
        if results[cid].verdict == c["label"]
    ]
    assert len(right) >= 15, sorted(set(results) - set(right))
    false_support = [
        c["id"]
        for c in DATA["cases"]
        if c["label"] == "contradicted" and results[c["id"]].verdict == "supported"
    ]
    assert false_support == []
    assert results["nequip-wrong-meaning"].verdict == support.SupportKind.CONTRADICTED
    assert results["nequip-wrong-meaning"].judge == "deterministic"


def test_every_quote_is_a_verified_substring_of_its_source():
    checker = _checker()
    for c in DATA["cases"]:
        r = checker.check(c["sentence"], c["key"])
        assert r.quote and r.quote_verified, c["id"]
        assert ABSTRACTS[c["key"]]["abstract"][r.quote_offset :].startswith(r.quote)
        assert r.text_level == "abstract"


# ---------------------------------------------------------------------------
# the model judge, faked
# ---------------------------------------------------------------------------


class FakeChat:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def __call__(self, messages, schema):
        self.calls.append(messages)
        if self.exc is not None:
            raise self.exc
        return self.reply


def test_the_number_check_overrules_a_model_that_says_supports():
    """Measured: a 2B judge reads the same 'three orders of magnitude' in
    both texts and answers SUPPORTS. The number check must win."""
    chat = FakeChat({"passage": 1, "quote": "x", "verdict": "SUPPORTS", "reason": "same"})
    r = _checker(support.Judge("auto", chat, label="llm:fake")).check(WRONG_MEANING, NEQUIP["key"])
    assert r.verdict == support.SupportKind.CONTRADICTED
    assert "the model said SUPPORTS" in r.reason
    assert r.quote_verified and "fewer training data" in r.quote


def test_a_model_quote_is_located_in_the_source_or_replaced_by_its_passage():
    sentence = "Adam's hyper-parameters typically require little tuning."
    exact = "have intuitive interpretations and typically require little tuning"
    chat = FakeChat({"passage": 1, "quote": exact.upper(), "verdict": "SUPPORTS", "reason": "r"})
    r = _checker(support.Judge("auto", chat, label="llm:fake")).check(sentence, "kingma2015adam")
    assert r.verdict == support.SupportKind.SUPPORTED and r.judge == "llm:fake"
    assert r.quote == exact and r.quote_from == "model" and r.quote_verified

    chat.reply = {
        "passage": 1,
        "quote": "words the paper never wrote at all",
        "verdict": "SUPPORTS",
    }
    chat.reply["reason"] = "r"
    r = _checker(support.Judge("auto", chat)).check(sentence, "kingma2015adam")
    assert r.quote_from == "passage" and r.quote_verified
    assert r.quote in ABSTRACTS["kingma2015adam"]["abstract"]


def test_judge_messages_show_the_level_and_numbered_passages():
    source = _sources()["batzner2022nequip"]
    msgs = support.judge_messages(WRONG_MEANING, source, source.spans[:2])
    assert "(abstract)" in msgs[1]["content"]
    assert "[1] This work presents" in msgs[1]["content"] and "[2] " in msgs[1]["content"]
    assert set(support.JUDGE_SCHEMA["properties"]["verdict"]["enum"]) == set(support.LLM_LABELS)


def test_unreachable_model_falls_back_once_in_auto_and_raises_in_llm_mode():
    chat = FakeChat(exc=httpx.ConnectError("refused"))
    checker = _checker(support.Judge("auto", chat))
    a = checker.check(WRONG_MEANING, NEQUIP["key"])
    b = checker.check("Adam requires little tuning.", "kingma2015adam")
    assert len(chat.calls) == 1, "a dead model must cost one attempt per run, not one per sentence"
    assert a.judge == b.judge == "deterministic"
    assert "model unavailable" in checker.judge.note

    with pytest.raises(httpx.ConnectError):
        _checker(support.Judge("llm", FakeChat(exc=httpx.ConnectError("x")))).check(
            WRONG_MEANING, NEQUIP["key"]
        )


def test_a_malformed_reply_costs_only_that_sentence():
    chat = FakeChat({"verdict": "MAYBE"})
    checker = _checker(support.Judge("auto", chat))
    assert checker.check(WRONG_MEANING, NEQUIP["key"]).judge == "deterministic"
    assert checker.judge.active and checker.judge.note is None


def test_an_unknown_judge_mode_is_refused():
    with pytest.raises(ValueError, match="judge must be one of"):
        support.Judge("gpt")


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_embedder_reranks_and_a_dead_one_falls_back_to_overlap(fake_embedder):
    checker = _checker(retriever=support.Retriever(fake_embedder))
    checker.check(WRONG_MEANING, NEQUIP["key"])
    assert checker.retriever.mode == "embedding"

    class Dead:
        def embed_query(self, text):
            raise httpx.ConnectError("down")

    checker = _checker(retriever=support.Retriever(Dead()))
    r = checker.check(WRONG_MEANING, NEQUIP["key"])
    assert r.verdict == support.SupportKind.CONTRADICTED
    assert checker.retriever.mode == "lexical" and "embedder unavailable" in checker.retriever.note


# ---------------------------------------------------------------------------
# resolving the cited paper: full text, abstract, title, nothing
# ---------------------------------------------------------------------------


def test_text_level_is_full_then_abstract_then_title_then_unavailable(tmp_path):
    db = tmp_path / "t.db"
    body = "Methods. We trained on 50 molecules. NequIP needs far less data than SchNet."
    _seed(db, fulltext={"2101.03164": body})
    conn = get_db(db)
    full = support.resolve_source(conn, "batzner2022nequip")
    assert full.level == "full" and body in full.text and NEQUIP["abstract"] in full.text
    assert support.resolve_source(conn, "10.48550/arXiv.1412.6980").level == "abstract"
    assert support.resolve_source(conn, "titleonly").level == "title"
    checker = support.Checker(
        conn=conn, judge=support.Judge("none"), retriever=support.Retriever(enabled=False)
    )
    assert checker.check("X is good.", "nope2099").verdict == support.SupportKind.UNAVAILABLE
    title = checker.check("X is a thing.", "titleonly")
    assert title.text_level == "title" and title.verdict != support.SupportKind.SUPPORTED
    conn.close()


def test_a_bib_only_key_reaches_the_store_through_its_arxiv_id(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{mykey,\n  title = {Adam},\n  doi = {10.48550/arXiv.1412.6980},\n}\n")
    from attestation import citations

    conn = get_db(db)
    resolver = citations.Resolver([citations.BibtexReader([bib])])
    source = support.resolve_source(conn, "mykey", resolver)
    assert source.level == "abstract" and source.title.startswith("Adam")
    conn.close()


# ---------------------------------------------------------------------------
# CLI and MCP
# ---------------------------------------------------------------------------

DRAFT = (
    "NequIP outperforms invariant models by up to three orders of magnitude"
    " [@batzner2022nequip].\n\n"
    "Adam's hyper-parameters typically require little tuning [@kingma2015adam].\n"
)


@pytest.fixture
def offline_embedder(monkeypatch):
    """The CLI and MCP paths build the real Embedder; keep them off the wire."""
    from conftest import FakeEmbedder

    import attestation.embed

    monkeypatch.setattr(attestation.embed, "Embedder", FakeEmbedder)


def test_cli_gates_on_contradicted_and_honours_fail_on(tmp_path, capsys, offline_embedder):
    db = tmp_path / "t.db"
    _seed(db)
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT)
    assert main(["support", str(draft), "--db", str(db), "--judge", "none"]) == 1
    out = capsys.readouterr().out
    assert "contradicted" in out and "fewer training data" in out
    assert "2 citation(s) checked" in out

    args = ["support", str(draft), "--db", str(db), "--judge", "none", "--fail-on", "not_found"]
    assert main(args) == 0
    assert main([*args[:-1], "bogus"]) == 1


def test_cli_single_sentence_and_json(tmp_path, capsys, offline_embedder):
    db = tmp_path / "t.db"
    _seed(db)
    base = ["support", "--db", str(db), "--judge", "none", "--sentence", WRONG_MEANING]
    assert main(base) == 1, "--sentence without --cite must refuse"
    assert main([*base, "--cite", "2101.03164", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    (row,) = out["results"]
    assert row["verdict"] == "contradicted" and row["quote_verified"] is True
    assert row["text_level"] == "abstract" and row["judge"] == "deterministic"


def test_cite_support_tool_envelope_and_size(tmp_path, monkeypatch, offline_embedder):
    from attestation.mcp import citation_support as citation

    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    _seed(db)
    draft = tmp_path / "draft.md"
    draft.write_text(
        "\n\n".join(f"{c['sentence'][:-1]} [@{c['key']}]." for c in DATA["cases"]) + "\n"
    )
    out = citation._support(path=str(draft), judge="none")
    assert out["ok"] is True and out["checked"] == ["support"]
    assert out["n_citations"] == len(DATA["cases"]) and out["n_checked"] == len(DATA["cases"])
    assert out["results"][0]["verdict"] == "contradicted", "worst first"
    assert len(out["results"]) == citation.MAX_SUPPORT_ROWS
    assert len(json.dumps(out)) < HARD_RESPONSE_CEILING

    one = citation._support(sentence=WRONG_MEANING, cite="batzner2022nequip", judge="none")
    assert one["results"][0]["verdict"] == "contradicted"
    refused = citation._support(sentence=WRONG_MEANING)
    assert refused["ok"] is False and "needs `cite`" in refused["message"]


def test_cite_support_stays_under_the_ceiling_at_every_field_cap(tmp_path, monkeypatch):
    """Measured at the caps, not on friendly data: abstract sentences are short,
    and a full-text quote, a long title and a long model reason are not."""
    from attestation.mcp import citation_support

    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "t.db"))
    huge = support.SupportResult(
        path="/" + "p" * 300 + ".md",
        line=99999,
        sentence="s" * 2000,
        key="k" * 500,
        verdict=support.SupportKind.CONTRADICTED,
        reason="r" * 2000,
        text_level="full",
        quote="q" * support.MAX_QUOTE_CHARS,
        quote_verified=True,
        judge="llm:" + "m" * 60,
        title="t" * 1000,
        checks=["name-missing:" + "x" * 80] * 20,
    )
    rows = [huge] * 30

    def fake_run(checker, found, limit=None):
        return {
            "n_sentences": 30,
            "n_citations": 30,
            "n_skipped": 10,
            "counts": {str(k): 6 for k in support.SupportKind},
            "results": rows,
            "judges": ["deterministic", "llm:" + "m" * 60],
            "retrieval": "lexical",
            "notes": ["n" * 200, "n" * 200],
        }

    monkeypatch.setattr(support, "run", fake_run)
    out = citation_support._support(sentence="x", cite="y", judge="none")
    assert out["ok"] is True and len(out["results"]) == citation_support.MAX_SUPPORT_ROWS
    assert len(json.dumps(out)) < HARD_RESPONSE_CEILING, len(json.dumps(out))


# ---------------------------------------------------------------------------
# reachable from router-only sessions: runs.ask and kg.ask
# ---------------------------------------------------------------------------

from attestation.mcp import routing_support  # noqa: E402


@pytest.mark.parametrize(
    "question",
    [
        'does batzner2022nequip support "NequIP outperforms invariant models"?',
        "check my citations support the draft at paper.md",
        "is this claim supported by its source?",
        "does the cited paper actually say that?",
        "did I misquote any of my references?",
    ],
)
def test_support_questions_route_to_cite_support(question):
    assert routing_support.asks_support(question)


@pytest.mark.parametrize(
    "question",
    [
        "are my claims supported by the runs?",  # runs.claims_check
        "check the numbers in my draft",  # runs.claims_check
        "What articles support these claims about mixture of experts?",  # feed.search
        "which arm won?",
        "which numbers did I forget to cite?",  # runs.claims_coverage
        "what do the papers I read say about equivariance?",  # kg, no sentence quoted
    ],
)
def test_other_questions_keep_their_routes(question):
    assert not routing_support.asks_support(question)


@pytest.mark.parametrize(
    ("question", "want"),
    [
        (f'does batzner2022nequip support "{WRONG_MEANING}"', (WRONG_MEANING, "batzner2022nequip")),
        (f"is “{WRONG_MEANING}” supported by 2101.03164?", (WRONG_MEANING, "2101.03164")),
        (f'does @kingma2015adam say "{WRONG_MEANING}"', (WRONG_MEANING, "kingma2015adam")),
        (f'does 10.48550/arXiv.2101.03164 back up "{WRONG_MEANING}"?', None),
        ("is this claim supported by its source?", (None, None)),
    ],
)
def test_the_sentence_and_key_come_out_of_the_question(question, want):
    sentence, key = routing_support.sentence_and_key(question)
    if want is None:
        assert sentence == WRONG_MEANING and key == "10.48550/arXiv.2101.03164"
    else:
        assert (sentence, key) == want


def _served(monkeypatch, surface):
    import asyncio
    import importlib

    from mcp.server.fastmcp import FastMCP

    import attestation.mcp as mcp_pkg

    monkeypatch.setenv("ATTEST_TOOLS", surface)
    monkeypatch.delenv("ATTEST_EXPAND", raising=False)
    importlib.reload(mcp_pkg)
    server = FastMCP("deployed")
    mcp_pkg.register_all(server)
    names = {t.name for t in asyncio.run(server.list_tools())}
    return server, names


def _call(server, tool, args) -> dict:
    import asyncio

    result = asyncio.run(server.call_tool(tool, args))
    first = result[1] if isinstance(result, tuple) else result
    if isinstance(first, dict):
        return first.get("result", first)
    return json.loads(first[0].text)


@pytest.mark.parametrize(
    ("surface", "router"), [("provenance", "runs.ask"), ("knowledge", "kg.ask")]
)
def test_a_router_only_session_reaches_cite_support(
    tmp_path, monkeypatch, offline_embedder, surface, router
):
    """The deployed session serves the router and nothing under it, so the
    route is the only way there: cite.support itself must stay hidden."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    monkeypatch.setattr(support.Judge, "active", property(lambda self: False))
    _seed(db)
    server, names = _served(monkeypatch, surface)
    assert router in names and "cite.support" not in names
    q = f'does batzner2022nequip support "{WRONG_MEANING}"'
    out = _call(server, router, {"question": q})
    assert out["ok"] is True and out["tool_used"] == "cite.support"
    assert out["answer"].splitlines()[1].startswith("contradicted -- batzner2022nequip [abstract]")
    assert "fewer training data" in out["answer"]


def test_runs_ask_checks_a_draft_by_path_and_asks_back_without_one(
    tmp_path, monkeypatch, offline_embedder
):
    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    monkeypatch.setattr(support.Judge, "active", property(lambda self: False))
    _seed(db)
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT)
    out = routing_support.support_answer("check my citations support the draft", str(draft))
    assert out["ok"] and out["tool_used"] == "cite.support"
    assert "2 citation(s) checked" in out["answer"]
    asked = routing_support.support_answer("is this claim supported by its source?")
    assert asked["ok"] is False and asked["options"] == ["cite.support"]
    assert routing_support.support_answer("which arm won?") is None
