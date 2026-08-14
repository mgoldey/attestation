"""One-page FastAPI + htmx UI. List renders instantly; explanations stream in lazily."""

import sqlite3
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

from attestation.db import get_db
from attestation.explain import explain
from attestation.llm import default_chat_fn
from attestation.rank import get_user, rank_items, record_click

STATIC_DIR = Path(__file__).parent / "static"

EXPLAIN_LIMIT = 20
LIST_LIMIT = 50

PAGE = Template("""<!doctype html>
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
{{ feed_content }}
</div>
</body></html>""")

FRAGMENT = Template("""<ol id="feed">
{% for it in items %}
 <li data-item-id="{{ it.item_id }}">
  <a href="{{ it.url or '#' }}">{{ it.title }}</a>
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
</ol>""")


def create_app(db_path: str | Path, embedder=None, chat_fn=None) -> FastAPI:
    if embedder is None:
        from attestation.embed import Embedder

        embedder = Embedder()
    chat_fn = chat_fn or default_chat_fn
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    conn = get_db(db_path)  # single writer connection for the whole app

    def require_user(user_name: str) -> sqlite3.Row:
        user = get_user(conn, user_name)
        if user is None:
            raise HTTPException(status_code=404, detail=f"unknown user: {user_name}")
        return user

    def render_list(user_name: str) -> str:
        user = require_user(user_name)
        items = rank_items(conn, embedder, user["id"])[:LIST_LIMIT]
        return FRAGMENT.render(items=items, user=user_name, explain_limit=EXPLAIN_LIMIT)

    @app.get("/", response_class=HTMLResponse)
    def index(user: str = Query("matt")):
        feed_content = render_list(user)  # 404s via require_user if user is unknown
        users = [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]
        return PAGE.render(users=users, user=user, feed_content=feed_content)

    @app.get("/list", response_class=HTMLResponse)
    def list_view(user: str = Query("matt")):
        return render_list(user)

    @app.post("/clicks", response_class=HTMLResponse)
    def click(user: str = Form(...), item_id: int = Form(...), useful: int = Form(...)):
        u = require_user(user)
        record_click(conn, u["id"], item_id, bool(useful), source="ui")
        return render_list(user)  # retrain + re-rank happens inside rank_items

    @app.get("/explanation", response_class=PlainTextResponse)
    def explanation(user: str = Query(...), item_id: int = Query(...)):
        u = require_user(user)
        return explain(conn, u["id"], item_id, chat_fn=chat_fn) or ""

    return app
