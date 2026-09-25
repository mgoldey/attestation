"""The routers: question in, tool out, deterministically.

These are the 15 turns from the architecture spike, which measured 13/15 for
routing against 8/15 for the flat 37-tool surface and 7.3/15 for an LLM swarm
at twice the latency. Routing here is rules, not a model, so the cases run
without either and guard the number rather than remembering it.

Two findings from that spike are encoded as tests below, because both are easy
to lose and neither is obvious:

  A catch-all destination becomes a magnet. An early routed version had a
  `doctor` tool for "diagnose the system" and three of four remaining misses
  went to it. An ambiguous question must ask back, never default.

  Descriptions and rules have to contain the words a user actually says.
  "which topics are most central or most read about" catches a turn that
  "what is central" does not.
"""

import pytest

from attestation.mcp.ask import route_feed, route_kg, route_runs, route_sym

# (turn, expected tool). Verbatim from the spike.
FEED_CASES = [
    ("what should I read today?", "feed.list"),
    # MEASURED 2026-09-04 over Discord on gemma4:e2b: each of these came back
    # as the "current feed or archive?" ambiguity, the model then called
    # feed.ask with no arguments and told the reader it had no tool for it.
    # "my feed" matched, "my ranked feed" did not -- the adjective broke the
    # phrase -- and nothing named the feed by day at all.
    ("What's in my ranked feed today?", "feed.list"),
    ("Give me my daily ranked feed", "feed.list"),
    ("Give me my daily feed review", "feed.list"),
    ("show me today's feed", "feed.list"),
    ("top papers for me today", "feed.list"),
    ("anything new on retrieval augmented generation?", "feed.search"),
    ("that first one isn't relevant", "feed.rate"),
    ("why did that rank so high?", "feed.explain"),
    ("what have I been reading about lately?", "feed.digest"),
    ("show me my feeds", "feed.sources"),
    ("add arxiv cs.CL to my feeds", "feed.source_add"),
    ("how well trained is my persona?", "feed.persona_status"),
    ("who are the personas?", "feed.persona_status"),
    # Going and looking, not searching what arrived: a named source or venue.
    ("what has been published on protein language models on pubmed?", "feed.research"),
    ("search arxiv for equivariant force fields", "feed.research"),
    ("look up recent papers in Nature Methods on cryo-EM", "feed.research"),
    ("research diffusion models for molecules", "feed.research"),
    # A standing topic, registered as a feed.
    ("track equivariant interatomic potentials for me", "feed.source_add"),
    ("follow graph neural networks for chemistry", "feed.source_add"),
    # Content and feedback rules beat the research router: a loose " journal"
    # phrase must not steal a question about an item already in hand.
    ("summarize the journal article", "feed.read"),
    ("why is this ranked so high in the journal", "feed.explain"),
]
RUNS_CASES = [
    ("which arm of my sweep won?", "runs.compare"),
    ("are the numbers in my draft right?", "runs.claims_check"),
    ("what numbers did I forget to cite?", "runs.claims_coverage"),
    ("what runs do I have recorded?", "runs.list"),
    ("record these results for the ledger", "runs.record"),
    ("write the results to the ledger", "runs.record"),
    ("save these metrics for my sweep", "runs.record"),
    ("re-read the runs, I just added new results", "runs.scan"),
]
KG_CASES = [
    ("how does retrieval connect to transformers?", "kg.path"),
    ("what topics do I read about most?", "kg.central"),
    ("what concepts exist?", "kg.concepts"),
]
SYM_CASES = [
    ("solve x squared minus four", "sym.solve"),
    ("is sin squared plus cos squared equal to one?", "sym.verify"),
    ("simplify this", "sym.simplify"),
]


@pytest.mark.parametrize(("turn", "want"), FEED_CASES)
def test_feed_routing(turn, want):
    assert route_feed(turn).tool == want


@pytest.mark.parametrize(("turn", "want"), RUNS_CASES)
def test_runs_routing(turn, want):
    assert route_runs(turn).tool == want


@pytest.mark.parametrize(("turn", "want"), KG_CASES)
def test_kg_routing(turn, want):
    assert route_kg(turn).tool == want


@pytest.mark.parametrize(("turn", "want"), SYM_CASES)
def test_sym_routing(turn, want):
    assert route_sym(turn).tool == want


def test_routing_beats_the_flat_surface_on_the_spike_cases():
    """The whole claim, in one assertion. 13/15 was the measurement; the
    deterministic router should clear it, since it is not guessing."""
    cases = FEED_CASES + RUNS_CASES + KG_CASES + SYM_CASES
    routers = {"feed": route_feed, "runs": route_runs, "kg": route_kg, "sym": route_sym}
    hits = sum(routers[want.split(".")[0]](turn).tool == want for turn, want in cases)
    assert hits >= 16, f"{hits}/{len(cases)}; the LLM baseline was 8/15"


def test_an_ambiguous_question_asks_rather_than_guessing():
    """A catch-all destination is a magnet. Ask instead."""
    decision = route_feed("tell me about machine learning")
    assert decision.tool is None
    assert decision.question, "an ambiguous route must carry a question"
    assert len(decision.options) >= 2, "and must name the alternatives"


def test_there_is_no_catch_all_destination():
    """No turn may fall through to a default tool -- that is the failure the
    spike measured, where `doctor` absorbed three of four misses."""
    for turn in ("hello", "what?", "do the thing", ""):
        decision = route_feed(turn)
        assert decision.tool is None, f"{turn!r} silently routed to {decision.tool}"


def test_routing_is_deterministic():
    """Same question, same answer, always. This is what a model does not give
    and what makes the 1.3s latency possible."""
    for _ in range(5):
        assert route_feed("what should I read today?").tool == "feed.list"


def test_arguments_are_extracted_not_guessed():
    """A router that picks the right tool with the wrong arguments has not
    helped."""
    assert route_feed("anything new on retrieval augmented generation?").kwargs["query"]
    assert route_sym("solve x**2 - 4").kwargs.get("expr")


def test_research_and_track_decisions_carry_their_arguments():
    """The exact kwargs feed.research and feed.source_add(research:) need."""
    d = route_feed("search arxiv for equivariant force fields")
    assert d.kwargs == {"query": "equivariant force fields", "sources": "arxiv"}
    d = route_feed("what has been published on protein language models on pubmed?")
    assert d.kwargs["query"] == "protein language models" and d.kwargs["sources"] == "pubmed"
    d = route_feed("research diffusion models for molecules")
    assert d.kwargs == {"query": "diffusion models for molecules", "sources": "arxiv,pubmed"}
    d = route_feed("track equivariant interatomic potentials for me")
    assert d.tool == "feed.source_add"
    assert d.kwargs == {"url": "research:arxiv,pubmed?q=equivariant+interatomic+potentials"}
    # a URL keeps the old behaviour: no research kwargs, the caller supplies the url
    assert route_feed("follow https://example.com/rss").kwargs == {}
    # "papers on X" is still the local archive
    assert route_feed("anything new on retrieval augmented generation?").tool == "feed.search"


def test_track_phrases_refuse_idiom_debris_as_a_topic():
    """A track phrase mid-sentence, or one whose 'topic' is idiom debris,
    must not register a research: feed with a garbage query."""
    # "follow" is not at the start -- not an instruction, falls through to
    # feed.source_add's old ask-for-a-URL path, non-mutating.
    d = route_feed("follow up on that paper")
    assert d.tool == "feed.source_add" and d.kwargs == {}
    # "watch" opens the sentence but is not a track instruction here.
    assert route_feed("what should I watch out for").tool is None
    # "monitor" opens the sentence; the extracted topic is stoplisted, so the
    # question falls through to the ordinary feed.list rule ("my feed").
    assert route_feed("monitor my feed").tool == "feed.list"
    # "watch" opens the sentence; the topic is stoplisted, and nothing else
    # claims it either.
    assert route_feed("watch this space").tool is None


# --- the tools themselves -------------------------------------------------


def test_ask_tools_declare_an_output_schema(tmp_path, monkeypatch):
    """The robustness mechanism. No tool declared one before, so every
    response was an untyped dict the model had to interpret and reformat --
    which is where the truncate-apologise-redump loop lived. A typed shape is
    rendered, not rewritten.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.delenv("ATTEST_TOOLS", raising=False)
    mcp = FastMCP("test")
    register_all(mcp)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}

    for name in ("feed.ask", "runs.ask", "kg.ask", "sym.ask"):
        assert name in tools, f"{name} is not served"
        schema = tools[name].outputSchema
        assert schema, f"{name} declares no outputSchema"
        props = schema.get("properties", {})
        assert {"ok", "answer", "caveat"} <= set(props), f"{name}: {sorted(props)}"


def test_an_ask_tool_relays_a_caveat_verbatim(tmp_path, monkeypatch):
    """A composed answer that drops the caveat is worse than the payload it
    replaced, because it reads as confident. ranking_quality's honesty note
    and runs.compare's caveats pass through unabridged."""
    from attestation.db import get_db
    from attestation.mcp.ask import _feed_ask

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'x')")
    conn.commit()
    conn.close()

    out = _feed_ask("ana", "what should I read today?")
    assert "caveat" in out
    # A persona with no clicks at all must be told its ranking is untrained.
    assert out["caveat"], "an untrained ranker answered without a caveat"


def test_an_ambiguous_ask_returns_the_options(tmp_path, monkeypatch):
    from attestation.mcp.ask import _feed_ask

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    out = _feed_ask("ana", "hello")
    assert out["ok"] is False
    assert out["options"], "an ambiguous ask must name the alternatives"
    assert "?" in out["answer"], "and must actually ask"


def test_not_cited_phrasings_all_reach_coverage():
    """ "did I not cite" and "did I forget to cite" are the same question.

    Found by driving the shipped surfaces: the model picked
    runs.claims_check, and the router declined, so neither path worked. A rule
    table only catches the words it lists.
    """
    for turn in (
        "what numbers did I not cite?",
        "what numbers did I forget to cite?",
        "which numbers have no claim?",
        "what is uncovered in my draft?",
    ):
        assert route_runs(turn).tool == "runs.claims_coverage", turn


def test_asking_to_summarise_a_paper_routes_to_read():
    """The turn that exposed the gap. An agent asked to summarise a paper
    answered that it had no tool for the job -- correctly, at the time."""
    for turn in (
        "summarize that paper",
        "what is that paper about?",
        "tell me about item 4043",
        "read me the abstract",
        "what does it say?",
    ):
        assert route_feed(turn).tool == "feed.read", turn


def test_expanding_coverage_routes_to_suggestions_not_a_blind_add():
    """ "What should I subscribe to" is a question, not an instruction.

    It routed to feed.source_add, which needs a URL the reader does not have
    -- so the agent either asked them for one or gave up. Suggestions come
    first: feed.source_suggest scores a curated list against tags this reader
    already liked, and the URL comes out of that.
    """
    for turn in (
        "what feeds should I subscribe to?",
        "suggest some feeds",
        "expand my sources",
        "I want more coverage of chemistry",
        "recommend feeds",
        "am I missing any sources?",
    ):
        assert route_feed(turn).tool == "feed.source_suggest", turn


def test_an_explicit_url_still_adds_directly():
    """A reader who names a feed is not asking for advice about it."""
    for turn in (
        "add https://rss.arxiv.org/rss/cs.CL",
        "subscribe to https://example.com/feed.xml",
    ):
        assert route_feed(turn).tool == "feed.source_add", turn


def test_previewing_before_subscribing_is_reachable():
    """Adding a feed sight-unseen is how a bad feed gets in. Preview is the
    step between suggestion and subscription."""
    for turn in ("what's in that feed?", "preview that feed", "show me what it publishes"):
        assert route_feed(turn).tool == "feed.source_preview", turn


def test_the_which_family_prompt_reports_how_many_it_left_out(tmp_path, monkeypatch):
    """Every other truncation in this repo says it truncated.

    `runs.list` reports "showing 8 of 40 families -- pass project= to narrow
    them". The disambiguation prompt interpolated at most eight names and said
    nothing about the rest, so a caller whose family was the ninth read the
    list as complete and concluded their sweep was not in the ledger.
    `n_families` was already in hand.
    """
    import json

    from attestation.db import get_db
    from attestation.mcp import ask

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()

    ws = tmp_path / "ws" / "proj" / "results"
    ws.mkdir(parents=True)
    for i in range(20):
        (ws / f"sweep{i}_a.json").write_text(json.dumps({"wer": 0.1 + i / 100}))
    monkeypatch.setenv("RESEARCH_ROOT", str(tmp_path / "ws"))

    from attestation.mcp import provenance as prov

    prov._scan(confirm=True)

    out = ask._runs_ask("which arm won?")

    assert out["ok"] is False
    assert "20" in out["answer"], out["answer"]
    assert "of" in out["answer"]


def test_kg_ask_returns_the_answer_not_just_a_count(tmp_path, monkeypatch):
    """Two of kg.ask's five routes named the tool and threw away its result.

    `_RESULT_KEYS` listed items/nodes/concepts/arms/communities/feeds/runs/
    suggestions -- not `path` or `neighbors`, which are the keys kg.path and
    kg.neighbors actually return. So a reader asking how two topics connect got
    "2 hop(s)" and a reader asking what is adjacent got "7 neighbour(s)", with
    the answer sitting unused in the payload.

    _summarise's own docstring says why that is wrong: "top 10 by degree tells
    a reader nothing; the ten concepts do."
    """
    from attestation.db import get_db

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for i in range(1, 7):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', 's', ?)",
            (f"item {i}", f"h{i}"),
        )
    for item, tags in (
        (1, ("retrieval", "ranking")),
        (2, ("retrieval", "ranking")),
        (3, ("ranking", "transformers")),
        (4, ("ranking", "transformers")),
        (5, ("retrieval", "transformers")),
        (6, ("retrieval", "transformers")),
    ):
        for tag in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item, tag))
    conn.commit()
    conn.close()

    from attestation.mcp.ask import _kg_ask

    neighbours = _kg_ask("what is next to retrieval?", source="retrieval")
    assert "ranking" in neighbours["answer"] or "transformers" in neighbours["answer"], (
        f"named the tool and dropped its answer: {neighbours['answer']!r}"
    )

    path = _kg_ask(
        "how does retrieval connect to transformers?",
        source="retrieval",
        target="transformers",
    )
    assert any(c.isalpha() for c in path["answer"].replace("hop", "")), path["answer"]
    assert "retrieval" in path["answer"], f"the path itself is missing: {path['answer']!r}"


def test_runs_ask_uses_the_metric_the_question_named(tmp_path, monkeypatch):
    """`_runs_ask` called `_compare(family)` with no metric even when the
    question named one, so `ledger.compare` fell back to `_pick_metric`'s own
    choice on a family recording more than one metric -- silently answering a
    different question than the one asked, with the wrong numbers in the
    caveat and no refusal to signal it. A real session (2026-09-03) asked
    "compare the kdsweep arms by wer" and got the caveat computed over a
    different metric's spread.

    Two arms here rank OPPOSITE ways depending on which metric wins the
    auto-pick, so a wrong pick is not just a different number -- it is a
    different winner.
    """
    import json

    from attestation.db import get_db

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()

    ws = tmp_path / "ws" / "proj" / "results"
    ws.mkdir(parents=True)
    # wer: sweep_a wins (lower). accuracy: sweep_b wins (higher). accuracy
    # has MORE arms than wer alone, so _pick_metric's "most arms share a
    # directed metric" rule picks accuracy when metric=None.
    (ws / "sweep_a.json").write_text(json.dumps({"wer": 0.1, "accuracy": 0.80}))
    (ws / "sweep_b.json").write_text(json.dumps({"wer": 0.2, "accuracy": 0.95}))
    (ws / "sweep_c.json").write_text(json.dumps({"accuracy": 0.70}))
    monkeypatch.setenv("RESEARCH_ROOT", str(tmp_path / "ws"))

    from attestation.mcp import provenance as prov
    from attestation.mcp.ask import _runs_ask

    prov._scan(confirm=True)

    by_wer = _runs_ask("compare the sweep arms by wer, which won?", family="sweep")
    assert "winner: sweep_a" in by_wer["answer"], (
        f"asked for wer but got a different metric's winner: {by_wer!r}"
    )

    by_accuracy = _runs_ask("compare the sweep arms by accuracy, which won?", family="sweep")
    assert "winner: sweep_b" in by_accuracy["answer"], (
        f"asked for accuracy but got a different metric's winner: {by_accuracy!r}"
    )


def test_runs_ask_metric_argument_wins_over_a_paraphrased_question(tmp_path, monkeypatch):
    """A caller that already extracted the metric should not depend on
    `question` still carrying it: a real Hermes session (2026-09-03, three
    runs) normalised "using the wer metric, compare the kdsweep arms" down
    to `question="which arm won?"` before it ever reached `_runs_ask`, so
    text extraction from `question` alone never saw "wer" -- the explicit
    `metric` parameter is the reliable path."""
    import json

    from attestation.db import get_db

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()

    ws = tmp_path / "ws" / "proj" / "results"
    ws.mkdir(parents=True)
    (ws / "sweep_a.json").write_text(json.dumps({"wer": 0.1, "accuracy": 0.80}))
    (ws / "sweep_b.json").write_text(json.dumps({"wer": 0.2, "accuracy": 0.95}))
    (ws / "sweep_c.json").write_text(json.dumps({"accuracy": 0.70}))
    monkeypatch.setenv("RESEARCH_ROOT", str(tmp_path / "ws"))

    from attestation.mcp import provenance as prov
    from attestation.mcp.ask import _runs_ask

    prov._scan(confirm=True)

    out = _runs_ask("which arm won?", family="sweep", metric="wer")
    assert "winner: sweep_a" in out["answer"], (
        f"explicit metric= was not honoured over the paraphrased question: {out!r}"
    )


def test_feed_research_answer_names_the_papers_and_carries_their_links():
    """`feed.research` routed correctly and then threw its results away.

    MEASURED from a live Discord session 2026-09-18: asked to "research
    orthogonalized mixtures of experts", the agent called feed.ask, got back
    "3 paper(s) from arxiv, pubmed; 3 new in the library" and answered that it
    had "only a summary count rather than the specific details or titles". It
    was not being evasive -- `_compose` really did hand it nothing else.

    Two causes, both here rather than in the tool: `_RESULT_KEYS` listed
    `items` but not `papers`, which is the key feed.research returns, so
    `_summarise` degraded to the bare message; and `refs` is built only from
    rows carrying `item_id`, which a paper row has no reason to have. The
    reader then asked for "the explicit corpus" and got an apology, because
    the ids and urls never crossed the wire.

    This is the same failure as test_kg_ask_returns_the_answer_not_just_a_count
    one route further on -- a route whose result key is absent degrades
    silently -- so it is pinned the same way.
    """
    from attestation.mcp.ask import _compose

    out = {
        "ok": True,
        "message": "3 paper(s) from arxiv, pubmed; 3 new in the library",
        "papers": [
            {
                "title": "Towards a Statistical Understanding of Mixture-of-Experts",
                "url": "https://arxiv.org/abs/2609.03501",
                "arxiv_id": "2609.03501",
                "doi": None,
            },
            {
                "title": "Evidence for Shared Routing Geometry in Sparse MoE",
                "url": "https://arxiv.org/abs/2609.02404",
                "arxiv_id": "2609.02404",
                "doi": None,
            },
        ],
        "n_found": 3,
        "stored": 3,
        "offline": False,
        "errors": [],
    }

    composed = _compose(out, "feed.research")

    assert "Mixture-of-Experts" in composed["answer"], (
        f"named the tool and dropped its papers: {composed['answer']!r}"
    )
    urls = [r.get("url") for r in composed["refs"]]
    assert "https://arxiv.org/abs/2609.03501" in urls, (
        f"the reader cannot reach a paper it was told about: {composed['refs']!r}"
    )


def test_feed_ask_research_survives_the_answer_model_through_the_real_server(tmp_path, monkeypatch):
    """The regression the previous test missed, pinned at the layer it lived in.

    test_feed_research_answer_names_the_papers_and_carries_their_links checked
    `_compose`'s dict and never built `Answer` -- so when paper refs gained a
    `paper_id` and no `item_id`, `Ref(item_id: int)` rejected every one of
    them, and EVERY research question through feed.ask raised a validation
    error. That shipped in 0.2.0. Worse, the papers were stored before the
    failure, so the library changed while the caller was told nothing worked.

    This drives the registered tool on a real FastMCP server, so the Answer
    model and MCP's own output validation both run. Only the network client is
    stubbed.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation import research
    from attestation.db import get_db
    from attestation.mcp import register_all
    from attestation.mcp import research as research_mod

    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    get_db(db).close()

    paper = research.Paper(
        client="arxiv",
        external_id="2609.03501",
        title="Towards a Statistical Understanding of Mixture-of-Experts",
        abstract="",
        authors=("Doe, Jane",),
        published="2026-09-01",
        arxiv_id="2609.03501",
        url="https://arxiv.org/abs/2609.03501",
    )

    class Fixed:
        name, offline = "arxiv", False

        def search(self, query, *, journal=None, since=None, limit=50):
            return [paper]

    server = FastMCP("t")
    register_all(server)
    # AFTER register_all: research.register() rebuilds CLIENTS from the
    # environment, so patching first silently restored the real arXiv client
    # and this test made a live network call (which 406'd).
    monkeypatch.setattr(
        research_mod,
        "CLIENTS",
        {
            "arxiv": Fixed(),
            "pubmed": research.NullClient("pubmed"),
            "crossref": research.NullClient("crossref"),
        },
    )

    async def call():
        result = await server.call_tool(
            "feed.ask", {"user": "matt", "question": "research mixture of experts on arxiv"}
        )
        return result[1] if isinstance(result, tuple) else result

    structured = asyncio.run(call())

    assert structured.get("ok") is True, structured
    assert structured["tool_used"] == "feed.research", structured
    assert "Mixture-of-Experts" in structured["answer"], structured["answer"]
    urls = [ref.get("url") for ref in structured["refs"]]
    assert "https://arxiv.org/abs/2609.03501" in urls, structured["refs"]


# --- 2026-09-25 query battery -------------------------------------------------
# 39 realistic questions (nine of them lifted from real Hermes transcripts that
# went wrong) were driven through the real attest-mcp over stdio against a copy
# of the live database. The first run passed 25/38 -- and five of those "passes"
# were wrong on inspection. Each finding is pinned here at the layer it lived in.


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        # Named subjects that fell through to a clarifier the feed surface
        # cannot act on (real sessions, 2026-09-04).
        ("latest in memory systems for LLMs", "feed.search"),
        ("What is known about orthogonalized mixtures of agents?", "feed.search"),
        ("What articles support these claims about mixture of experts?", "feed.search"),
        # Ownership vs advice vs instruction: "subscribe" matched "subscribed".
        ("what feeds am I subscribed to?", "feed.sources"),
        ("which feeds do I follow?", "feed.sources"),
        ("am I missing any sources?", "feed.source_suggest"),
        ("subscribe me to arxiv cs.CL", "feed.source_add"),
        # "top of my feeds" wants ranked items, not the subscription list.
        ("give me the top of my feeds", "feed.list"),
        ("when was your recent scrape?", "feed.sources"),
        ("why aren't you learning from what I read?", "feed.persona_status"),
        # A bare "learning" must NOT be read as a question about the persona.
        ("find me papers on machine learning", "feed.search"),
    ],
)
def test_battery_feed_routes(question, tool):
    assert route_feed(question).tool == tool


@pytest.mark.parametrize(
    ("question", "route", "tool"),
    [
        ("show me the details of kdsweep_t4", route_runs, "runs.detail"),  # "sweep" in kdsweep
        ("which arm of the sweep won?", route_runs, "runs.compare"),
        (
            "which numbers in this draft are not backed by a claim?",
            route_runs,
            "runs.claims_coverage",
        ),
        ("what are my main research areas?", route_kg, "kg.communities"),
    ],
)
def test_battery_runs_and_kg_routes(question, route, tool):
    assert route(question).tool == tool


def test_digest_claims_and_coverage_answers_name_what_they_found():
    """Each of these came back as a bare count, and the digest with zero refs
    although every nested item had an id and a url."""
    from attestation.mcp.ask import Answer, _compose

    digest = Answer(
        **_compose(
            {
                "ok": True,
                "message": "16 item(s) in 2 topic(s); showing 4",
                "topics": [
                    {"label": "machine-learning", "items": [{"item_id": 1, "url": "u1"}]},
                    {"label": "reasoning", "items": [{"item_id": 2, "url": "u2"}]},
                ],
            },
            "feed.digest",
        )
    )
    assert "machine-learning" in digest.answer and len(digest.refs) == 2

    checked = Answer(
        **_compose(
            {
                "ok": True,
                "message": "2 claim(s): 1 contradicted, 1 supported",
                "claims": [
                    {"verdict": "supported", "line": 18, "message": "wer=0.0731"},
                    {"verdict": "contradicted", "line": 33, "message": "says 0.0701, run 0.0688"},
                ],
            },
            "runs.claims_check",
        )
    )
    # The contradicted claim leads: it is what the reader has to fix.
    assert checked.answer.index("contradicted line 33") < checked.answer.index("supported line")

    covered = Answer(
        **_compose(
            {
                "ok": True,
                "message": "1/2 number(s) covered",
                "uncovered": [{"line": 13, "value": 41.3, "context": "41.3M parameters."}],
            },
            "runs.claims_coverage",
        )
    )
    assert "line 13" in covered.answer


def test_a_failed_research_client_reaches_the_caveat():
    """The headline keeps text before its first ';', so "1 client(s) failed"
    was cut and caveat was None while arXiv returned 406 -- the reader got
    PubMed papers as if they were the arXiv results."""
    from attestation.mcp.ask import Answer, _compose

    answer = Answer(
        **_compose(
            {
                "ok": True,
                "message": "0 paper(s) from arxiv; 0 new in the library; 1 client(s) failed",
                "papers": [],
                "errors": ["arxiv: HTTPStatusError: Client error '406 Not Acceptable'"],
            },
            "feed.research",
        )
    )
    assert answer.caveat and "arxiv" in answer.caveat and "406" in answer.caveat


def test_runs_ask_record_does_not_pretend_to_have_recorded():
    """It fell through to a run listing while reporting tool_used=runs.record."""
    from attestation.mcp.ask import _runs_ask

    out = _runs_ask("record these metrics for my sweep")
    assert out["tool_used"] is None and out["options"] == ["runs.record"]


def test_sym_verify_answers_in_words_not_the_difference():
    """`result` is lhs - rhs, i.e. "0" exactly when the identity HOLDS."""
    from attestation.mcp.ask import _sym_ask

    same = _sym_ask("(x+1)**2 == x**2 + 2*x + 1", "is it equal?")
    assert same["tool_used"] == "sym.verify" and "equal" in same["answer"], same
    assert same["answer"].strip() != "0"


def test_an_unknown_concept_refusal_names_what_the_reader_probably_meant():
    """Told to call kg.concepts() -- 1403 names a 2B model cannot render."""
    from attestation import kg

    members = {"memory", "working-memory", "immune-system", "system-design", "rag"}
    near = kg.nearest("Memory System", members)
    assert near[:2] == ["memory", "working-memory"], near
