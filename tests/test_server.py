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
    assert "classifier OFF" in html
    ids = [int(s.split('"')[0]) for s in html.split('data-item-id="')[1:]]
    client.post("/clicks", data={"user": "matt", "item_id": ids[0], "useful": "1"})
    client.post("/clicks", data={"user": "matt", "item_id": ids[1], "useful": "0"})

    after = client.get("/list", params={"user": "matt"}).text
    assert "classifier OFF" not in after
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


def test_cross_origin_get_does_not_create_a_persona(client):
    """Creation is a write, even when it happens on a GET.

    `<img src="http://127.0.0.1:8899/?user=typo">` on any page a reader visits
    is enough to reach `reader()`, and `reader()` INSERTs. The blast radius is
    one junk persona row, but that is exactly the "a typo becomes a permanent
    persona nobody knows exists" failure POST /clicks is guarded against --
    and here nothing announces it either.
    """
    resp = client.get(
        "/list",
        params={"user": "drive-by-reader"},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403, resp.text

    from attestation.rank import get_user

    conn = get_db(client.db_path)
    assert get_user(conn, "drive-by-reader") is None, "a cross-origin GET created a persona"


def test_cross_origin_get_for_an_existing_user_still_reads(client):
    """Only creation is guarded. Reading must keep working cross-origin.

    A reader whose persona already exists must not be broken by the guard --
    embedding the feed in another page reads nothing that page could not read
    by asking the reader to visit the URL, and refusing it would make the
    origin check a general browsing restriction rather than a create guard.
    """
    resp = client.get(
        "/list",
        params={"user": "matt"},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 200, resp.text
    assert "item 0" in resp.text


def test_same_origin_get_still_creates_a_persona(client):
    """The ordinary path -- a reader typing their own name into the UI."""
    resp = client.get(
        "/list",
        params={"user": "same-origin-newcomer"},
        headers={"Origin": "http://127.0.0.1:8899"},
    )
    assert resp.status_code == 200, resp.text

    from attestation.rank import get_user

    conn = get_db(client.db_path)
    assert get_user(conn, "same-origin-newcomer") is not None


def test_a_backslash_in_a_reader_name_does_not_break_the_vote_buttons(client):
    """`hx-vals` builds JSON by string interpolation under HTML autoescape.

    Jinja neutralises < > & ' " and NOT backslash, so a reader named `ann\\`
    produces `{"user":"ann\\","item_id":23,...}` -- which htmx cannot parse, so
    both vote buttons silently stop working for that persona. Reachable without
    setup: `reader()` auto-creates a persona from `/?user=`, so a typo is
    enough.

    `| tojson` is the fix: it escapes for the language the value is being
    embedded in, which is JSON, not HTML.
    """
    import json
    import re

    html = client.get("/list", params={"user": "ann\\"}).text
    values = re.findall(r"hx-vals='([^']*)'", html)
    assert values, "no vote buttons rendered"
    for raw in values:
        json.loads(raw.replace("&#34;", '"').replace("&quot;", '"'))


def test_a_hash_in_a_reader_name_does_not_truncate_the_explanation_url(client):
    """The hx-vals bug's sibling, in a URL rather than in JSON.

    `hx-get="/explanation?user={{ user }}&item_id={{ it.item_id }}"` interpolates
    the name into a query string under HTML autoescape, which escapes `&` and
    not `#`. A reader named `a#b` produces a URL whose fragment starts before
    `item_id`, so the explanation request arrives with no item and the "why is
    this here?" panel silently never loads.

    Escape for the language being embedded in: this is a URL, so urlencode.
    """
    import re
    from urllib.parse import parse_qs, urlparse

    html = client.get("/list", params={"user": "a#b"}).text
    urls = re.findall(r'hx-get="([^"]*explanation[^"]*)"', html)
    assert urls, "no explanation panels rendered"
    for raw in urls:
        query = parse_qs(urlparse(raw.replace("&amp;", "&")).query)
        assert query.get("item_id"), f"item_id lost from {raw!r}"
        assert query.get("user") == ["a#b"], f"user mangled in {raw!r}"


def test_concurrent_requests_do_not_share_one_connection(client):
    """The web UI served 500s under ordinary concurrent use.

    `create_app` opened one connection "for the whole app" and `get_db` passes
    check_same_thread=False, but FastAPI runs sync routes in a threadpool -- so
    concurrent requests interleave cursors on one connection. Measured: 11 of
    200 threaded `get_user` calls returned None for a user that exists, and 11
    of 12 parallel `/list` requests against a live server returned 500 with
    four distinct tracebacks, including `get_user` returning None and that None
    reaching autocreate_user.

    This is a NORMAL path, not a stress case: the page fires up to 20 lazy
    /explanation requests per load, so a reload mid-load hits it. Every
    existing server test issues requests serially, which is why seventeen
    rounds missed it.
    """
    import concurrent.futures

    def fetch(i):
        # A NEW reader, not an existing one. This test originally used "matt",
        # the single case that passes: autocreate never fires, so it exercised
        # the interleaved-cursor half and missed the check-then-insert half
        # entirely -- on the first page load for a new reader, which is what
        # autocreate exists to serve.
        return client.get("/list", params={"user": f"fresh{i % 3}"}).status_code

    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        codes = list(pool.map(fetch, range(24)))

    bad = [c for c in codes if c != 200]
    assert not bad, f"{len(bad)} of {len(codes)} concurrent requests failed: {sorted(set(bad))}"


def test_one_hostile_feed_entry_cannot_inflate_the_page(tmp_path, fake_embedder):
    """Autoescape makes third-party text inert; it does nothing about LENGTH.

    feedparser accepts a 10MB <title> without complaint -- it is well-formed
    XML, so bozo is False -- and ingest stores it unbounded. Measured before
    this: one such entry rendered a 10,001,592 byte page. The MCP tools escaped
    it only because they clip independently at MAX_TITLE_CHARS, so the two
    readers of the same row disagreed about whether the row was safe.
    """
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('v', 'science')")
    conn.execute("INSERT INTO feeds(title, url) VALUES (?, 'http://f')", ("F" * 100_000,))
    huge = "A" * 2_000_000
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (1, ?, 'http://x', ?, 'h1')",
        (huge, huge),
    )
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
        (cur.lastrowid, fake_embedder.embed_document("a", "b").tobytes()),
    )
    conn.commit()
    conn.close()

    app = create_app(tmp_path / "t.db", embedder=fake_embedder, chat_fn=lambda m, s: {"text": "w"})
    html = TestClient(app).get("/", params={"user": "v"}).text
    assert len(html) < 50_000, (
        f"a single 2MB title produced a {len(html):,} byte page -- nothing bounds"
        " third-party strings at the render boundary"
    )
    assert huge not in html, "the unclipped title reached the page"


def test_hostile_feed_text_renders_inert(tmp_path, fake_embedder):
    """Titles and summaries come from arbitrary third-party RSS. Verified by
    PARSING the response rather than grepping it: 'onerror=' inside escaped
    text matches a regex and executes nothing, so a grep-based check reports
    a vulnerability that is not there."""
    from html.parser import HTMLParser

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('v', 'science')")
    conn.execute("INSERT INTO feeds(title, url) VALUES ('f', 'http://f')")
    payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        '" onmouseover="alert(2)',
        "<svg/onload=alert(3)>",
    ]
    for i, payload in enumerate(payloads, start=1):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (1, ?, 'javascript:alert(9)', ?, ?)",
            (payload, payload, f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, fake_embedder.embed_document(payload, "s").tobytes()),
        )
    conn.commit()
    conn.close()

    app = create_app(tmp_path / "t.db", embedder=fake_embedder, chat_fn=lambda m, s: {"text": "w"})
    html = TestClient(app).get("/", params={"user": "v"}).text

    injected: list[str] = []

    class Scan(HTMLParser):
        def handle_starttag(self, tag, attrs):
            # The app's own <script src=...> bundle is expected; an injected one
            # would carry a body or come from item text.
            if tag in ("svg", "img"):
                injected.append(f"<{tag}> element from feed content")
            for key, value in attrs:
                if key.lower().startswith("on"):
                    injected.append(f"event attribute {key} on <{tag}>")
                if key.lower() in ("href", "src") and (value or "").lower().startswith(
                    "javascript:"
                ):
                    injected.append(f"javascript: URL in {tag}.{key}")

    Scan().feed(html)
    assert not injected, f"feed content became live markup: {injected}"
