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
    ("anything new on retrieval augmented generation?", "feed.search"),
    ("that first one isn't relevant", "feed.rate"),
    ("why did that rank so high?", "feed.explain"),
    ("what have I been reading about lately?", "feed.digest"),
    ("show me my feeds", "feed.sources"),
    ("add arxiv cs.CL to my feeds", "feed.source_add"),
    ("how well trained is my persona?", "feed.persona_status"),
]
RUNS_CASES = [
    ("which arm of my sweep won?", "runs.compare"),
    ("are the numbers in my draft right?", "runs.claims_check"),
    ("what numbers did I forget to cite?", "runs.claims_coverage"),
    ("what runs do I have recorded?", "runs.list"),
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
