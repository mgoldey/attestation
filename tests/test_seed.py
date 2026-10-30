"""The cold start: papers a reader likes, and a first search of their field (seed.py)."""

import pytest
from conftest import FakeEmbedder, seeded_db

from attestation import research, seed
from attestation.db import get_db
from attestation.mcp import research as research_tool
from attestation.mcp.routing import route_feed

DOI = "10.1364/boe.9.000001"


def _paper(i: int, *, doi=None, arxiv_id=None, title=None, client="openalex"):
    return research.Paper(
        client=client,
        external_id=f"W{i}",
        title=title or f"Hyperspectral imaging of blood flow in the chick embryo {i}",
        abstract=f"We image perfusion in the chorioallantoic membrane, study {i}.",
        authors=("Doe, Jane",),
        published="2026-10-20",
        doi=doi,
        arxiv_id=arxiv_id,
        url=f"https://doi.org/{doi}" if doi else f"https://example.org/{i}",
    )


class FakeOpenalex:
    name, offline = "openalex", False

    def __init__(self, by_filter=None, papers=()):
        self.by_filter = by_filter or {}
        self.papers = list(papers)
        self.filters: list[str] = []

    def works(self, filter_expr, *, limit=5):
        self.filters.append(filter_expr)
        return self.by_filter.get(filter_expr, [])[:limit]

    def search(self, query, *, journal=None, since=None, limit=50, arxiv_only=False):
        return self.papers[:limit]


class FakeArxiv:
    name, offline = "arxiv", False

    def __init__(self, papers=()):
        self.papers = list(papers)

    def by_id(self, arxiv_id):
        return [p for p in self.papers if p.arxiv_id == arxiv_id]

    def search(self, query, *, journal=None, since=None, limit=50):
        return []


def _clients(openalex=None, arxiv=None):
    clients = research.null_clients()
    clients["openalex"] = openalex or FakeOpenalex()
    clients["arxiv"] = arxiv or FakeArxiv()
    return clients


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(path))
    seeded_db(path).close()
    from attestation.mcp import _shared

    monkeypatch.setattr(_shared, "_embedder", FakeEmbedder())
    monkeypatch.setattr(research_tool, "CLIENTS", research.null_clients())
    return path


# --- routing --------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "kwargs"),
    [
        (f"I like {DOI}", {"liked": DOI}),
        (
            "I like https://doi.org/10.1364/BOE.9.000001",
            {"liked": "https://doi.org/10.1364/BOE.9.000001"},
        ),
        ("I liked arXiv:2106.02347", {"liked": "arXiv:2106.02347"}),
        (
            "I really like Laser speckle contrast imaging",
            {"liked": "Laser speckle contrast imaging"},
        ),
        ("I like the papers in refs.bib", {"liked_file": "refs.bib"}),
        ("I like everything in Thesis/library.bib", {"liked_file": "Thesis/library.bib"}),
        ("I like zhang2024.pdf", {"liked_file": "zhang2024.pdf"}),
    ],
)
def test_i_like_a_paper_seeds_the_persona(question, kwargs):
    d = route_feed(question)
    assert (d.tool, d.kwargs) == ("feed.persona_update", kwargs)


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("useful", "feed.rate"),
        ("not useful", "feed.rate"),
        ("add it to my bib", "cite.save"),
        ("I would like to follow graph neural networks", "feed.source_add"),
        ("I like papers on optics", "feed.search"),  # a subject, not a paper
        ("I work on spectral imaging of blood flow", "feed.persona_update"),
    ],
)
def test_i_like_does_not_steal_verdicts_follows_or_saves(question, tool):
    assert route_feed(question).tool == tool


@pytest.mark.parametrize("question", ["I like this one", "I like it", "I like #2", "I like 3"])
def test_i_like_a_listed_item_is_never_a_seed(question):
    assert "liked" not in route_feed(question).kwargs


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        (DOI, ("doi", DOI)),
        ("doi:10.1364/BOE.9.000001.", ("doi", DOI)),
        ("10.48550/arXiv.2106.02347", ("arxiv", "2106.02347")),
        ("2106.02347v3", ("arxiv", "2106.02347")),
        ("https://arxiv.org/abs/2106.02347", ("arxiv", "2106.02347")),
        ('"Imaging blood flow in ovo."', ("title", "Imaging blood flow in ovo")),
    ],
)
def test_parse_ref(text, parsed):
    assert seed.parse_ref(text) == parsed


# --- like -----------------------------------------------------------------


def test_a_liked_doi_becomes_a_useful_click_a_library_row_and_profile_text(db):
    oa = FakeOpenalex({f"doi:https://doi.org/{DOI}": [_paper(1, doi=DOI)]})
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    done = seed.like(conn, FakeEmbedder(), _clients(oa), uid, [DOI])

    assert [s["title"] for s in done.seeded] == [_paper(1).title] and not done.unresolved
    click = conn.execute("SELECT useful, source FROM clicks WHERE user_id = ?", (uid,)).fetchone()
    assert (click["useful"], click["source"]) == (1, "agent")
    assert conn.execute('SELECT 1 FROM "references" WHERE doi = ?', (DOI,)).fetchone()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    assert seed.profile_text(conn, user).endswith("Papers I like: " + _paper(1).title)


def test_seeding_the_same_paper_twice_stores_one_item(db):
    oa = FakeOpenalex({f"doi:https://doi.org/{DOI}": [_paper(1, doi=DOI)]})
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    seed.like(conn, FakeEmbedder(), _clients(oa), uid, [DOI])
    seed.like(conn, FakeEmbedder(), _clients(oa), uid, [f"https://doi.org/{DOI}"])
    assert conn.execute("SELECT COUNT(*) n FROM items WHERE doi = ?", (DOI,)).fetchone()["n"] == 1


def test_a_title_is_seeded_only_on_a_confident_match(db):
    typed = "Hyperspectral imaging of blood flow in the chick embryo 7"
    near = FakeOpenalex({f"title.search:{typed}": [_paper(7)]})
    wrong = FakeOpenalex({f"title.search:{typed}": [_paper(8, title="Deep learning for cats")]})
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    assert seed.like(conn, FakeEmbedder(), _clients(wrong), uid, [typed]).unresolved == [typed]
    assert len(seed.like(conn, FakeEmbedder(), _clients(near), uid, [typed]).seeded) == 1


def test_an_arxiv_id_goes_to_arxiv_first(db):
    ax = FakeArxiv([_paper(2, arxiv_id="2106.02347", client="arxiv")])
    oa = FakeOpenalex()
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    done = seed.like(conn, FakeEmbedder(), _clients(oa, ax), uid, ["arXiv:2106.02347"])
    assert len(done.seeded) == 1 and oa.filters == []


def test_a_paper_already_in_the_library_needs_no_network(db):
    from attestation import library

    conn = get_db(db)
    library.upsert(
        conn,
        library.ReferenceRecord(
            source="bib:refs.bib",
            source_key="doe2024",
            title="Imaging blood flow in ovo",
            doi=DOI,
            abstract="Perfusion in the chick CAM.",
        ),
    )
    conn.commit()
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    done = seed.like(
        conn, FakeEmbedder(), research.null_clients(), uid, ["Imaging blood flow in ovo"]
    )
    assert [s["title"] for s in done.seeded] == ["Imaging blood flow in ovo"]


def test_a_bib_file_in_the_workspace_seeds_each_entry(db, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "thesis").mkdir(parents=True)
    (workspace / "thesis" / "refs.bib").write_text(
        f"@article{{a, title={{One}}, doi={{{DOI}}}}}\n"
        "@article{b, title={Missing everywhere}}\n"
    )
    monkeypatch.setenv("ATTEST_WORKSPACE", str(workspace))
    refs, problem = seed.refs_in_file("refs.bib")
    assert refs == [DOI, "Missing everywhere"] and problem is None

    oa = FakeOpenalex({f"doi:https://doi.org/{DOI}": [_paper(1, doi=DOI)]})
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    done = seed.like(conn, FakeEmbedder(), _clients(oa), uid, refs)
    assert len(done.seeded) == 1 and done.unresolved == ["Missing everywhere"]


def test_a_missing_or_wrong_file_says_why():
    assert seed.refs_in_file("nope.bib") == ([], "no file at nope.bib")


def test_a_reset_forgets_seeds(db):
    from attestation.personas import purge_feedback

    oa = FakeOpenalex({f"doi:https://doi.org/{DOI}": [_paper(1, doi=DOI)]})
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    seed.like(conn, FakeEmbedder(), _clients(oa), uid, [DOI])
    purge_feedback(conn, uid)
    conn.commit()
    assert seed.seed_titles(conn, uid) == []
    assert not conn.execute("SELECT 1 FROM engagement WHERE user_id = ?", (uid,)).fetchone()


def test_ranking_embeds_the_seeded_titles(db, monkeypatch):
    from attestation import rank

    oa = FakeOpenalex({f"doi:https://doi.org/{DOI}": [_paper(1, doi=DOI)]})
    conn = get_db(db)
    uid = conn.execute("SELECT id FROM users WHERE name = 'researcher'").fetchone()["id"]
    seed.like(conn, FakeEmbedder(), _clients(oa), uid, [DOI])
    seen = []
    real = rank._profile_vector
    monkeypatch.setattr(
        rank, "_profile_vector", lambda c, e, u, text: seen.append(text) or real(c, e, u, text)
    )
    conn.execute("UPDATE items SET published = datetime('now')")
    conn.execute(
        "INSERT INTO items(title, content_hash, published) VALUES ('other', 'h', datetime('now'))"
    )
    import numpy as np

    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (last_insert_rowid(), ?)",
        (np.ones(256, dtype=np.float32).tobytes(),),
    )
    rank.rank_items(conn, FakeEmbedder(), uid)
    assert seen and "Papers I like: " + _paper(1).title in seen[0]


# --- feed.ask ---------------------------------------------------------------


def test_feed_ask_i_like_a_doi_answers_with_the_paper(db, monkeypatch):
    from attestation.mcp.ask import _feed_ask

    oa = FakeOpenalex({f"doi:https://doi.org/{DOI}": [_paper(1, doi=DOI)]})
    monkeypatch.setattr(research_tool, "CLIENTS", _clients(oa))
    out = _feed_ask("researcher", f"I like {DOI}")
    assert out["ok"] and out["tool_used"] == "feed.persona_update", out
    assert "1 paper(s) added" in out["answer"]
    assert [r["url"] for r in out["refs"]] == [f"https://doi.org/{DOI}"]


def test_feed_ask_i_like_something_unfindable_says_so(db):
    from attestation.mcp.ask import _feed_ask

    out = _feed_ask("researcher", "I like A paper nobody has written")
    assert not out["ok"] and "could not find A paper nobody has written" in out["answer"], out


# --- first search -------------------------------------------------------------


def test_i_work_on_x_adds_recent_papers_from_that_field(db, monkeypatch):
    from attestation.mcp.ask import _feed_ask

    oa = FakeOpenalex(papers=[_paper(i) for i in range(3)])
    monkeypatch.setattr(research_tool, "CLIENTS", _clients(oa))
    out = _feed_ask("owner", "I work on spectral imaging of blood flow in chick embryos")
    assert out["ok"] and "3 recent paper(s)" in out["answer"], out
    conn = get_db(db)
    feed = conn.execute("SELECT * FROM feeds WHERE url LIKE 'research:%'").fetchone()
    assert feed["url"].startswith("research:arxiv,pubmed,openalex?q=spectral+imaging")
    n = conn.execute("SELECT COUNT(*) n FROM items WHERE feed_id = ?", (feed["id"],)).fetchone()
    assert n["n"] == 3


def test_the_first_search_is_skipped_quietly_offline(db):
    from attestation.mcp.ask import _feed_ask

    out = _feed_ask("owner", "I work on perovskite solar cells")
    assert out["ok"] and out["caveat"] is None, out
    assert not get_db(db).execute("SELECT 1 FROM feeds WHERE url LIKE 'research:%'").fetchone()


def test_a_failed_embedder_leaves_the_interests_set(db, monkeypatch):
    import httpx

    from attestation.mcp import _shared
    from attestation.mcp.ask import _feed_ask

    class Down(FakeEmbedder):
        def embed_documents(self, pairs):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(_shared, "_embedder", Down())
    monkeypatch.setattr(research_tool, "CLIENTS", _clients(FakeOpenalex(papers=[_paper(1)])))
    out = _feed_ask("owner", "I work on perovskite solar cells")
    assert out["ok"] and "could not store them" in (out["caveat"] or ""), out
    row = get_db(db).execute("SELECT interests FROM users WHERE name = 'owner'").fetchone()
    assert row["interests"] == "perovskite solar cells"
