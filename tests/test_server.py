import pytest
from fastapi.testclient import TestClient

from attestation.db import get_db
from attestation.server import create_app


@pytest.fixture
def client(tmp_path, fake_embedder):
    db_path = tmp_path / "t.db"
    conn = get_db(db_path)
    for i in range(25):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', 's', ?)",
            (f"item {i}", f"h{i}"),
        )
        vec = fake_embedder.embed_document(f"item {i}", "s")
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, vec.tobytes()),
        )
    conn.commit()
    conn.close()
    app = create_app(db_path, embedder=fake_embedder, chat_fn=lambda m, s: {"text": "why"})
    tc = TestClient(app)
    tc.db_path = db_path
    return tc


def test_index_renders_users_and_feed(client):
    html = client.get("/").text
    assert "bench-chemist" in html and "ml-engineer" in html and "matt" in html
    assert "item 0" in html


def test_list_fragment_per_user_differs(client):
    a = client.get("/list", params={"user": "bench-chemist"}).text
    b = client.get("/list", params={"user": "ml-engineer"}).text
    assert a != b


def test_click_rerenders_without_clicked_item(client):
    html = client.get("/list", params={"user": "matt"}).text
    first_id = html.split('data-item-id="')[1].split('"')[0]
    after = client.post("/clicks", data={"user": "matt", "item_id": first_id, "useful": "1"}).text
    assert f'data-item-id="{first_id}"' not in after


def test_first_click_all_one_class_no_500(client):
    """Regression for the single-class crash blocker."""
    for _ in range(3):
        html = client.get("/list", params={"user": "matt"}).text
        item_id = html.split('data-item-id="')[1].split('"')[0]
        resp = client.post("/clicks", data={"user": "matt", "item_id": item_id, "useful": "1"})
        assert resp.status_code == 200


def test_explanation_endpoint(client):
    html = client.get("/list", params={"user": "matt"}).text
    item_id = html.split('data-item-id="')[1].split('"')[0]
    resp = client.get("/explanation", params={"user": "matt", "item_id": item_id})
    assert resp.status_code == 200
    assert resp.text == "why"


def test_lazy_explanations_limited_to_top_20(client):
    html = client.get("/list", params={"user": "matt"}).text
    assert html.count('hx-get="/explanation') == 20  # 25 items seeded, only top 20 lazy-load


def test_static_htmx_served(client):
    resp = client.get("/static/htmx.min.js")
    assert resp.status_code == 200
    assert "htmx" in resp.text


def test_unknown_user_clicks_returns_404(client):
    """Writes still refuse. A vote under an unknown name is a typo or a stale
    form far more often than a new reader, and creating a persona as the side
    effect of a click is a permanent profile nobody knows exists."""
    resp = client.post("/clicks", data={"user": "nobody", "item_id": 1, "useful": 1})
    assert resp.status_code == 404


def test_unknown_user_explanation_returns_404(client):
    resp = client.get("/explanation", params={"user": "nobody", "item_id": 1})
    assert resp.status_code == 404


def test_xss_title_is_escaped(tmp_path, fake_embedder):
    db_path = tmp_path / "xss.db"
    conn = get_db(db_path)
    payload = '<img src=x onerror="alert(1)">'
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, ?, 'http://x', 's', 'h-xss')",
        (payload,),
    )
    item_id = cur.lastrowid
    vec = fake_embedder.embed_document(payload, "s")
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (item_id, vec.tobytes())
    )
    conn.commit()
    conn.close()
    app = create_app(db_path, embedder=fake_embedder, chat_fn=lambda m, s: {"text": "why"})
    html = TestClient(app).get("/").text
    assert "&lt;img" in html
    assert payload not in html


def test_known_good_title_renders_in_live_markup(client):
    html = client.get("/").text
    # guards against a double-escape regression from the feed_content|safe wiring
    assert "<a href=" in html
    assert "item 0</a>" in html


def test_click_cross_origin_rejected(client):
    html = client.get("/list", params={"user": "matt"}).text
    first_id = html.split('data-item-id="')[1].split('"')[0]
    resp = client.post(
        "/clicks",
        data={"user": "matt", "item_id": first_id, "useful": "1"},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403
    after = client.get("/list", params={"user": "matt"}).text
    assert f'data-item-id="{first_id}"' in after


def test_click_no_origin_still_allowed(client):
    html = client.get("/list", params={"user": "matt"}).text
    first_id = html.split('data-item-id="')[1].split('"')[0]
    resp = client.post("/clicks", data={"user": "matt", "item_id": first_id, "useful": "1"})
    assert resp.status_code == 200


def test_list_renders_tag_badges(tmp_path, fake_embedder):
    db_path = tmp_path / "badge.db"
    conn = get_db(db_path)
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 'tagged item', 'http://x', 's', 'h-badge')"
    )
    item_id = cur.lastrowid
    vec = fake_embedder.embed_document("tagged item", "s")
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (item_id, vec.tobytes())
    )
    conn.execute(
        "INSERT INTO item_features(item_id, content_type, model) VALUES (?, 'paper', 'm')",
        (item_id,),
    )
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'dft')", (item_id,))
    conn.commit()
    conn.close()
    app = create_app(db_path, embedder=fake_embedder, chat_fn=lambda m, s: {"text": "why"})
    resp = TestClient(app).get("/list", params={"user": "matt"})
    assert resp.status_code == 200
    assert '<span class="tag type">paper</span>' in resp.text
    assert '<span class="tag">dft</span>' in resp.text


def test_feed_shows_the_ranking_caveat_when_the_classifier_is_silent(client):
    """The web UI must not present an untrained order as if it were trained.

    CLAUDE.md states the invariant: "_ranking_quality() reports
    classifier_active + caveat | surface it rather than letting a reader assume
    the ranker learned something". Every MCP tool that returns an order carries
    it. The web UI -- the surface a human actually reads -- rendered a bare
    ranked list, so a reader had no way to tell a cold-start order from a
    learned one.
    """
    html = client.get("/list", params={"user": "matt"}).text
    assert "classifier" in html.lower(), "no honesty caveat rendered"


def test_caveat_tracks_the_ranker_state_rather_than_being_static(client):
    """The caveat is a live signal, not decoration.

    A cold profile is told the classifier is silent. After one useful and one
    not-useful click the classifier CAN fire, so the wording has to change --
    to the weakly-trained caveat, not to nothing: two clicks is still two
    clicks, and claiming a trained ranker there would be the same dishonesty
    in the other direction.
    """
    html = client.get("/list", params={"user": "matt"}).text
    assert "WITHOUT its click classifier" in html
    ids = [int(s.split('"')[0]) for s in html.split('data-item-id="')[1:]]
    client.post("/clicks", data={"user": "matt", "item_id": ids[0], "useful": "1"})
    client.post("/clicks", data={"user": "matt", "item_id": ids[1], "useful": "0"})

    after = client.get("/list", params={"user": "matt"}).text
    assert "WITHOUT its click classifier" not in after
    assert "weakly trained" in after


def test_unknown_user_is_created_not_404ed(client):
    """Two front doors onto one database must not disagree about a new user.

    The MCP surface auto-creates on first contact (mcp/_tool.py
    _autocreate_user) because asking an agent to run a setup command first is
    the friction that made people give up. The web UI 404'd the same name. A
    reader who visits /?user=newname before their first agent call gets an
    error page for a state the other front door treats as ordinary.
    """
    resp = client.get("/list", params={"user": "brand-new-reader"})
    assert resp.status_code == 200, resp.text

    from attestation.rank import get_user

    conn = get_db(client.db_path)
    row = get_user(conn, "brand-new-reader")
    assert row is not None
    # Seeded with something, not left blank: the interests text IS the profile
    # embedding, and an empty one ranks nothing.
    assert row["interests"].strip()
