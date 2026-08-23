"""One-page FastAPI + htmx UI. List renders instantly; explanations stream in lazily."""

import sqlite3
from pathlib import Path

import jinja2
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from attestation.db import get_db
from attestation.explain import explain
from attestation.llm import default_chat_fn
from attestation.rank import (
    autocreate_user,
    get_user,
    rank_items,
    ranking_quality,
    record_click,
)

STATIC_DIR = Path(__file__).parent / "static"

EXPLAIN_LIMIT = 20
LIST_LIMIT = 50

# autoescape=True: item titles/sources/tags come from arbitrary third-party RSS
# feeds (ingest.py stores title unsanitized) and render into the same origin
# that owns the /clicks training endpoint — a malicious feed title must not be
# able to inject live HTML/JS here.
env = jinja2.Environment(autoescape=True)


def safe_href(url: str | None) -> str:
    """Only emit http(s) URLs into href — autoescape does not neutralize javascript: URLs."""
    if url and (url.startswith("http://") or url.startswith("https://")):
        return url
    return "#"


env.filters["safe_href"] = safe_href

PAGE = env.from_string("""<!doctype html>
<html><head><title>attestation</title>
<script src="/static/htmx.min.js"></script>
<style>
 body { font-family: system-ui; max-width: 52rem; margin: 2rem auto; }
 .user-btn { margin-right: .5rem; }  .active { font-weight: bold; }
 li { margin-bottom: .8rem; }  .src { color: #888; font-size: .8rem; }
 .why { color: #567; font-size: .85rem; font-style: italic; }
 button.yn { margin-left: .4rem; }
 .tag { background: #eef; color: #446; font-size: .7rem; padding: 0 .35rem;
        border-radius: .5rem; margin-left: .25rem; }
 .tag.type { background: #efe; color: #464; }
</style></head>
<body>
<h1>attestation</h1>
<nav>
{% for u in users %}
 <a class="user-btn {{ 'active' if u == user else '' }}" href="/?user={{ u }}">{{ u }}</a>
{% endfor %}
</nav>
<div id="feed-wrap">
{{ feed_content | safe }}
</div>
</body></html>""")

# The caveat lives INSIDE #feed, not above it. The vote buttons swap #feed
# with outerHTML, so anything outside it survives the swap unchanged -- a
# caveat rendered as a sibling would still be claiming "0 clicks recorded"
# after the click that fixed it.
FRAGMENT = env.from_string("""<div id="feed">
{% if quality and quality.caveat %}
<p class="caveat">{{ quality.caveat }}</p>
{% endif %}
<ol>
{% for it in items %}
 <li data-item-id="{{ it.item_id }}">
  <a href="{{ it.url | safe_href }}">{{ it.title }}</a>
  <span class="src">{{ it.source or '' }} · rank {{ '%.1f' % it.score }}</span>
  {% if it.content_type %}<span class="tag type">{{ it.content_type }}</span>{% endif %}
  {% for t in it.tags %}<span class="tag">{{ t }}</span>{% endfor %}
  <button class="yn" hx-post="/clicks"
      hx-vals='{"user":"{{ user }}","item_id":{{ it.item_id }},"useful":1}'
      hx-target="#feed" hx-swap="outerHTML">✓</button>
  <button class="yn" hx-post="/clicks"
      hx-vals='{"user":"{{ user }}","item_id":{{ it.item_id }},"useful":0}'
      hx-target="#feed" hx-swap="outerHTML">✗</button>
  {% if loop.index <= explain_limit %}
  <div class="why" hx-get="/explanation?user={{ user }}&item_id={{ it.item_id }}"
       hx-trigger="load delay:{{ loop.index }}s" hx-swap="innerHTML"></div>
  {% endif %}
 </li>
{% endfor %}
</ol>
</div>""")


def _is_loopback_origin(value: str) -> bool:
    """True if value is an http(s) origin/referer whose host is 127.0.0.1 or localhost.

    Port is deliberately not checked — the server's port is user-configurable
    (cli.py --port), so any loopback port is accepted.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and parts.hostname in ("127.0.0.1", "localhost")


def require_same_origin(request: Request) -> None:
    """Reject-only-when-present CSRF guard for mutating routes.

    Binding to 127.0.0.1 does not stop cross-origin browser requests: the
    victim's browser, not an attacker's server, is the request origin. A
    plain cross-origin <form method=post> sends no Origin/Referer control
    a strict allowlist could rely on being absent, but browsers DO attach
    Origin (or at least Referer) to real cross-origin requests, while
    curl/CLI/TestClient callers send neither. So: reject only when a
    present Origin/Referer proves the request came from a foreign page;
    allow when both are absent.
    """
    origin = request.headers.get("origin")
    if origin is not None:
        if not _is_loopback_origin(origin):
            raise HTTPException(status_code=403, detail="cross-origin request rejected")
        return
    referer = request.headers.get("referer")
    if referer is not None and not _is_loopback_origin(referer):
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def create_app(db_path: str | Path, embedder=None, chat_fn=None) -> FastAPI:
    if embedder is None:
        from attestation.embed import Embedder

        embedder = Embedder()
    chat_fn = chat_fn or default_chat_fn
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    conn = get_db(db_path)  # single writer connection for the whole app

    def require_user(user_name: str) -> sqlite3.Row:
        """An existing reader, or 404. Used by the routes that WRITE.

        Recording a click or asking for an explanation under an unknown name
        is far more likely a typo or a stale form than a new reader arriving,
        and creating a persona as the side effect of a vote produces exactly
        the "permanent persona nobody knows exists" that mcp/_tool.py warns
        about -- except here nothing announces it. Reads create; writes refuse.
        """
        user = get_user(conn, user_name)
        if user is None:
            raise HTTPException(status_code=404, detail=f"unknown user: {user_name}")
        return user

    def reader(user_name: str) -> sqlite3.Row:
        """An existing reader, created on first sight. Used by the routes that READ.

        Browsing used to 404. Two front doors onto one database disagreeing
        about what a new name means is drift a reader hits before they have
        any way to understand it: they visit /?user=<their name>, get an error
        page, and never learn that any MCP tool with the same name would
        simply have worked. mcp/_tool.py already made this call and recorded
        why -- an unknown-name refusal taught agents to invent personas, so
        the refusal caused the duplicates it was meant to prevent. The web UI
        just never got the same treatment.
        """
        if not user_name.strip():
            raise HTTPException(status_code=400, detail="user name required")
        user = get_user(conn, user_name)
        if user is None:
            user, _ = autocreate_user(conn, user_name)
        return user

    def render_list(user_name: str) -> str:
        user = reader(user_name)
        items = rank_items(conn, embedder, user["id"])[:LIST_LIMIT]
        return FRAGMENT.render(
            items=items,
            user=user_name,
            explain_limit=EXPLAIN_LIMIT,
            quality=ranking_quality(conn, user["id"]),
        )

    @app.get("/", response_class=HTMLResponse)
    def index(user: str = Query("matt")):
        feed_content = render_list(user)  # creates the reader if new; see reader()
        users = [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]
        return PAGE.render(users=users, user=user, feed_content=feed_content)

    @app.get("/list", response_class=HTMLResponse)
    def list_view(user: str = Query("matt")):
        return render_list(user)

    @app.post("/clicks", response_class=HTMLResponse, dependencies=[Depends(require_same_origin)])
    def click(user: str = Form(...), item_id: int = Form(...), useful: int = Form(...)):
        u = require_user(user)
        record_click(conn, u["id"], item_id, bool(useful), source="ui")
        return render_list(user)  # retrain + re-rank happens inside rank_items

    @app.get("/explanation", response_class=PlainTextResponse)
    def explanation(user: str = Query(...), item_id: int = Query(...)):
        u = require_user(user)
        return explain(conn, u["id"], item_id, chat_fn=chat_fn) or ""

    return app
