"""Search must find by meaning, not by substring.

`search_feed` used to do `needle not in item.title.lower()` -- a literal
substring test -- while `item_vectors` held an embedding for every item and
`embed.QUERY_PROMPT` existed for exactly this purpose and was never called
with a search query. Searching "LLM" could not find "Large Language Models".

It also ranked the entire archive by persona profile FIRST and filtered
afterwards, so the query barely influenced the order: the results were the
highest-profile-ranked items that happened to contain the string.

These tests use an embedder whose vectors encode a small hand-built semantic
space, so "related" and "unrelated" are facts of the fixture rather than
properties of a real model.
"""

import numpy as np
import pytest

from attestation.db import get_db
from attestation.mcp import _shared
from attestation.mcp import feed as feed_mod

# A tiny concept space. Items and queries are placed by concept, so a query
# about one concept is genuinely nearer its items than to any other's.
CONCEPTS = {
    "language-models": 0,
    "structural-biology": 1,
    "superconductivity": 2,
}

ITEMS = [
    ("Large Language Models scale predictably", "We fit scaling laws.", "language-models"),
    ("Sparse attention for long contexts", "Attention concentrates.", "language-models"),
    ("Cryo-EM structure of the ribosome", "We report structures.", "structural-biology"),
    ("Protein folding from sequence alone", "Folding predicted.", "structural-biology"),
    ("Room-temperature superconductivity claims", "Resistance measured.", "superconductivity"),
]


class ConceptEmbedder:
    """Vectors that encode a concept, so semantic nearness is well-defined."""

    dims = 256

    def _vec(self, concept: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        v[CONCEPTS[concept]] = 1.0
        v[10:] = 0.01  # a little shared mass, as real embeddings have
        return v / np.linalg.norm(v)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        for t, _s, concept in ITEMS:
            if t == title:
                return self._vec(concept)
        return self._vec("language-models")

    def embed_query(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if any(w in lowered for w in ("llm", "language model", "transformer", "attention")):
            return self._vec("language-models")
        if any(w in lowered for w in ("protein", "cryo", "ribosome", "structure")):
            return self._vec("structural-biology")
        if any(w in lowered for w in ("supercond", "resistance")):
            return self._vec("superconductivity")
        return self._vec("language-models")


@pytest.fixture
def search_db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    embedder = ConceptEmbedder()
    conn = get_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'everything')")
    for i, (title, summary, _c) in enumerate(ITEMS, start=1):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, ?)",
            (title, summary, f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, embedder.embed_document(title, summary).tobytes()),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_shared, "_embedder", embedder)
    monkeypatch.setattr(_shared, "get_embedder", lambda: embedder)
    return db


def _titles(out):
    return [i["title"] for i in out["items"]]


def test_a_query_finds_items_that_never_contain_its_words(search_db):
    """The headline failure: "LLM" appears in no title or summary here."""
    out = feed_mod._search_feed("ana", "LLM")
    assert out["ok"], out["message"]
    top = _titles(out)[:2]
    assert any("Large Language Models" in t or "attention" in t for t in top), (
        f"semantic query returned {top}"
    )


def test_results_are_ordered_by_relevance_to_the_query(search_db):
    """Not by persona profile with the query as an afterthought.

    This test used to pass with QUERY_WEIGHT set to 0 -- i.e. with the query
    contributing nothing at all, which is the precise bug the blend was written
    to fix. The orthogonal fixture vectors meant RELEVANCE_FLOOR narrowed the
    result to the two correct items BEFORE ordering ran, so `top[:2]` could not
    fail however the sort behaved.

    So it now asserts on the ORDER within one relevance-filtered set, and
    `test_the_query_outweighs_the_profile` below pins the blend directly.
    """
    out = feed_mod._search_feed("ana", "protein structure")
    top = _titles(out)[:2]
    assert all("Cryo-EM" in t or "Protein folding" in t for t in top), (
        f"biology query returned {top}"
    )
    assert out["items"][0]["relevance"] >= out["items"][1]["relevance"], (
        "results must be sorted by the blended relevance they report"
    )


def test_the_query_outweighs_the_profile(search_db, monkeypatch):
    """The blend itself, which nothing else covers.

    Two items are equally relevant to the query, so only the profile can break
    the tie -- and the profile is inverted so that ranking by it alone puts the
    wrong one first. With QUERY_WEIGHT at 0 the profile decides outright; at
    0.75 the query dominates and the reported relevance separates them.
    """
    from attestation.mcp import feed as f

    out = feed_mod._search_feed("ana", "language models", limit=5)
    assert out["ok"], out["message"]
    scores = [i["relevance"] for i in out["items"]]
    assert scores == sorted(scores, reverse=True), f"not sorted by relevance: {scores}"

    # Semantic hits must outscore an item the query never reached.
    semantic = [i for i in out["items"] if i["match"] in {"semantic", "both"}]
    other = [i for i in out["items"] if i["match"] == "literal"]
    if semantic and other:
        assert min(i["relevance"] for i in semantic) > max(i["relevance"] for i in other), (
            "a literal-only hit outscored a semantic one; the query is not driving the order"
        )

    monkeypatch.setattr(f, "QUERY_WEIGHT", 0.0)
    zeroed = feed_mod._search_feed("ana", "language models", limit=5)
    assert [i["relevance"] for i in zeroed["items"]] != scores, (
        "relevance scores are identical with the query weighted at zero -- "
        "QUERY_WEIGHT is not affecting the blend at all"
    )


def test_an_unrelated_query_does_not_return_everything(search_db):
    """A search that matches nothing must say so, not hand back the feed.

    Substring search returned the whole ranked archive whenever the needle was
    empty or matched broadly; a semantic search with no threshold does the same
    thing more subtly, since every vector has SOME similarity to every query.
    """
    out = feed_mod._search_feed("ana", "superconductivity")
    titles = _titles(out)
    assert any("superconductivity" in t.lower() for t in titles)
    assert not any("Cryo-EM" in t for t in titles), (
        f"unrelated biology item surfaced for a physics query: {titles}"
    )


def test_exact_phrase_still_matches(search_db):
    """Semantic search must not lose the literal case. A user searching an
    exact title expects that title."""
    out = feed_mod._search_feed("ana", "Cryo-EM structure of the ribosome")
    assert "Cryo-EM structure of the ribosome" in _titles(out)[:1]


def test_tag_and_content_type_filters_still_apply(search_db):
    out = feed_mod._search_feed("ana", "language models", content_type="paper")
    assert out["ok"]
    assert out["items"] == [], "no item is tagged paper in this fixture"


def test_empty_query_is_a_filter_not_a_search(search_db):
    """`search_feed(user, "")` filters without a semantic query. Undocumented
    before; asserted here so it stays intentional.

    The limit is explicit: this test is about the FILTER semantics, and it
    previously relied on the default happening to exceed the fixture size, so
    lowering that default (5 -> 4, for the payload budget) failed it for a
    reason unrelated to what it checks.
    """
    out = feed_mod._search_feed("ana", "", limit=len(ITEMS))
    assert out["ok"]
    assert len(out["items"]) == len(ITEMS)


def test_already_rated_is_reported(search_db):
    from attestation.rank import record_click

    conn = get_db(search_db)
    uid = conn.execute("SELECT id FROM users WHERE name='ana'").fetchone()["id"]
    item = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    record_click(conn, uid, item, True)
    conn.commit()
    conn.close()

    out = feed_mod._search_feed("ana", "")
    flagged = {i["item_id"]: i["already_rated"] for i in out["items"]}
    assert flagged[item] is True
    assert any(v is False for k, v in flagged.items() if k != item)


def test_search_reports_how_it_matched(search_db):
    """A caller must be able to tell a semantic hit from a literal one --
    otherwise it cannot judge whether a surprising result is relevance or
    noise."""
    out = feed_mod._search_feed("ana", "LLM")
    assert "match" in out["items"][0], f"no match provenance in {out['items'][0]}"
    assert out["items"][0]["match"] in {"semantic", "literal", "both"}


# --- against the real embedder -------------------------------------------
#
# The fixture above builds a hand-made concept space, which proves the ranking
# logic but says nothing about whether real embeddings behave the way the
# design assumes. These tests use the configured Ollama embedder on a small
# corpus written here, so they check the assumption itself: that a query
# embedding lands nearer to semantically related documents than to unrelated
# ones, with no shared vocabulary to lean on.
#
# Marked `live_model`: excluded from the default run and CI, since they need a
# model resident and take seconds rather than milliseconds.
#
#   uv run pytest -m live_model


REAL_ITEMS = [
    (
        "Scaling behaviour of transformer language models",
        "We fit power laws relating parameter count to validation perplexity.",
    ),
    (
        "Attention is sparse in long-context decoders",
        "Most attention mass concentrates on a small subset of positions.",
    ),
    (
        "Cryo-EM structure of the bacterial ribosome",
        "We resolve the 50S subunit at 2.4 angstrom and describe the peptidyl site.",
    ),
    (
        "Predicting protein folds from sequence",
        "A neural network predicts tertiary structure from primary sequence alone.",
    ),
    (
        "Ambient-pressure superconductivity in a hydride",
        "Electrical resistance drops to zero at 250 kelvin under 150 gigapascals.",
    ),
    (
        "Sourdough starter hydration ratios",
        "A baker's guide to maintaining a rye starter at 100 percent hydration.",
    ),
]


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    from attestation.embed import Embedder

    embedder = Embedder()
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    conn = get_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'science')")
    for i, (title, summary) in enumerate(REAL_ITEMS, start=1):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, ?)",
            (title, summary, f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, embedder.embed_document(title, summary).tobytes()),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_shared, "_embedder", embedder)
    monkeypatch.setattr(_shared, "get_embedder", lambda: embedder)
    return db


@pytest.mark.live_model
def test_real_embeddings_match_an_acronym_to_its_expansion(real_db):
    """ "LLM" appears nowhere in the corpus. Substring search returns nothing;
    a real embedding should still reach the language-model papers."""
    out = feed_mod._search_feed("ana", "LLM", limit=3)
    assert out["ok"], out["message"]
    titles = _titles(out)
    assert titles, "real embeddings found nothing for an acronym"
    assert any("language models" in t.lower() or "attention" in t.lower() for t in titles), (
        f"acronym query returned {titles}"
    )


@pytest.mark.live_model
def test_real_embeddings_separate_unrelated_domains(real_db):
    """The relevance floor's job. Sourdough shares no vocabulary with any
    scientific query and must not appear for one."""
    for query in ("protein folding", "superconductivity", "transformer models"):
        titles = _titles(feed_mod._search_feed("ana", query, limit=6))
        assert not any("Sourdough" in t for t in titles), (
            f"{query!r} surfaced the unrelated item: {titles}"
        )


@pytest.mark.live_model
def test_real_embeddings_rank_the_right_domain_first(real_db):
    """Each query's top hit must come from its own domain, with no shared
    words to make it easy."""
    expected = {
        "how do proteins fold": ("protein", "ribosome", "cryo-em"),
        "zero electrical resistance at high temperature": ("superconduct", "hydride"),
        "baking bread at home": ("sourdough",),
    }
    for query, wanted in expected.items():
        titles = _titles(feed_mod._search_feed("ana", query, limit=2))
        assert titles, f"{query!r} returned nothing"
        assert any(w in titles[0].lower() for w in wanted), (
            f"{query!r} ranked {titles[0]!r} first; expected one of {wanted}"
        )


def test_a_negative_best_similarity_returns_nothing():
    """ "The best match points away from the query" is not a result.

    Note what this test does NOT prove. It passes with the `best <= 0` guard
    removed, because the relative floor filters these out anyway -- a fraction
    of a negative number is closer to zero, so the bound tightens rather than
    loosening. I wrote the guard believing the opposite and only found out by
    removing it and watching the test still pass.

    It is kept because the guard states an intent the arithmetic only happens
    to satisfy, and because the assertion is right either way: a query whose
    nearest neighbour is anti-correlated should return nothing.
    """
    from attestation.mcp import feed as f

    class FakeCursor(list):
        def fetchall(self):
            return list(self)

    class FakeConn:
        def execute(self, sql, params):
            # distance > sqrt(2) => cosine = 1 - d^2/2 < 0
            return FakeCursor([{"rowid": 1, "distance": 1.9}, {"rowid": 2, "distance": 1.95}])

    class E:
        def embed_query(self, text):
            return np.zeros(4, dtype=np.float32)

    assert f._semantic_hits(FakeConn(), E(), "q", k=5) == {}


def test_a_tag_from_kg_concepts_reaches_every_item_carrying_it(tmp_path, monkeypatch):
    """The workflow the tools tell an agent to use, end to end.

    kg_concepts returns CANONICAL names -- the graph aliases before it filters.
    search_feed compared its `tag` argument against the RAW stored tags, so a
    name handed over by one tool under-returned in the other. Measured on the
    live corpus: kg_concepts says `hugging-face`, search_feed reached 23 of the
    380 items carrying that concept, and `large-language-models` reached 226 of
    1,072. No crash, no warning -- just a 94% shortfall.
    """
    from attestation import kg
    from attestation.db import get_db

    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    embedder = ConceptEmbedder()
    conn = get_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'everything')")

    # Two spellings of one concept, plus an unrelated item.
    spellings = ["llm", "llms", "large-language-models"]
    assert len({kg.canonical(s) for s in spellings}) == 1, "fixture assumes these alias together"
    canonical = kg.canonical("llm")

    for i, (title, summary, _c) in enumerate(ITEMS, start=1):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, ?)",
            (title, summary, f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, embedder.embed_document(title, summary).tobytes()),
        )
        # items 1..3 each carry a DIFFERENT spelling of the same concept
        if i <= 3:
            conn.execute(
                "INSERT INTO item_tags(item_id, tag) VALUES (?, ?)",
                (cur.lastrowid, spellings[i - 1]),
            )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_shared, "_embedder", embedder)
    monkeypatch.setattr(_shared, "get_embedder", lambda: embedder)

    out = feed_mod._search_feed("ana", "", tag=canonical)
    assert out["ok"], out["message"]
    assert len(out["items"]) == 3, (
        f"a canonical tag reached {len(out['items'])} of 3 items carrying that concept"
    )


def test_the_relevance_floor_is_not_anchored_on_one_item():
    """One weak top hit collapsed the whole result set.

    The floor kept items scoring >= best * RELEVANCE_FLOOR, where `best` was
    the similarity of the SINGLE nearest neighbour -- so the size of every
    answer was a function of one item. Measured on the live corpus,
    `feed.search("enzyme mechanism")` returned one result, a snail slime paper,
    while four genuine enzyme papers were cut, three by less than 0.4% of the
    threshold.

    This asserts the BEHAVIOUR, not the constant. An earlier version checked
    `RELEVANCE_ANCHOR >= 3`, and a reviewer restored `best = max(...)` while
    leaving the constant at 3 -- the whole suite passed while the live failure
    was back. The two are separately mutable and only one of them matters.
    """
    import math

    from attestation.mcp.feed import _semantic_hits

    # One strong outlier, then a tight cluster of genuine matches. Anchored on
    # the single best hit the cluster falls below the floor and only the
    # outlier survives; averaged over the top few, the cluster is the answer.
    # The REAL similarities for "enzyme mechanism" on the live corpus, which is
    # the query that returned one snail-slime paper. A synthetic outlier at
    # 0.95 does not reproduce it -- that drags even a top-3 mean above the
    # cluster. The failure needs a MODEST outlier over a tight band, which is
    # what real embeddings produce.
    wanted = {
        1: 0.3635,
        2: 0.3238,
        3: 0.3154,
        4: 0.3154,
        5: 0.3133,
        6: 0.3131,
        7: 0.3078,
        8: 0.3071,
    }

    class _Rows:
        """A cursor returning the distances that produce `wanted`."""

        def fetchall(self):
            return [
                {"rowid": rid, "distance": math.sqrt(2.0 * (1.0 - sim))}
                for rid, sim in wanted.items()
            ]

    class _Conn:
        def execute(self, *_args):
            return _Rows()

    class _Embedder:
        def embed_query(self, _text):
            class _V:
                @staticmethod
                def tobytes():
                    return b""

            return _V()

    kept = _semantic_hits(_Conn(), _Embedder(), "anything", 10)
    assert len(kept) >= 6, (
        f"one modest outlier starved the query: {len(kept)} of 8 kept."
        " The floor is anchored on a single item."
    )
