"""One-page FastAPI + htmx UI. List renders instantly; explanations stream in lazily."""

import sqlite3
import threading
from pathlib import Path
from urllib.parse import quote

import jinja2
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from attestation.db import get_db
from attestation.explain import explain
from attestation.llm import base_url, default_chat_fn
from attestation.rank import (
    EmbedderUnavailable,
    autocreate_user,
    create_user,
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


# A title long enough to matter is hostile or broken, never useful. The MCP
# surface already clips to MAX_TITLE_CHARS; this is the same bound for the
# other reader. Measured: one RSS entry with a 10MB title rendered a 10,001,592
# byte page, because nothing between feedparser and Jinja bounds a string.
MAX_RENDERED_CHARS = 300


def clip(value, limit: int = MAX_RENDERED_CHARS) -> str:
    """Bound one third-party string at the render boundary.

    Autoescape makes hostile text inert; it does nothing about hostile LENGTH.
    feedparser accepts a 10MB <title> without complaint (it is well-formed XML),
    ingest stores it unbounded, and every reader of that row then pays for it --
    the web UI served a 10MB page for a single item. The MCP tools escaped this
    only because they clip independently.

    Applied per field rather than to the page, so one bad item degrades to a
    truncated line instead of taking the feed down with it.
    """
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def urlarg(value: str | None) -> str:
    """One query-string value, percent-encoded.

    HTML autoescape covers the HTML layer and knows nothing about URL syntax:
    it escapes `&` and leaves `#`, so a reader named `a#b` produced
    `?user=a#b&item_id=2` -- everything after the fragment marker is dropped by
    the browser, and the explanation panel silently never loads. Same shape as
    the hx-vals JSON bug: escape for the language you are embedding in.
    """
    from urllib.parse import quote

    return quote(str(value or ""), safe="")


env.filters["clip"] = clip
env.filters["safe_href"] = safe_href
env.filters["urlarg"] = urlarg

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
 <a class="user-btn" href="/onboard">+ new reader</a>
</nav>
<div id="feed-wrap">
{{ feed_content | safe }}
</div>
</body></html>""")

# The caveat lives INSIDE #feed, not above it. The vote buttons swap #feed
# with outerHTML, so anything outside it survives the swap unchanged -- a
# caveat rendered as a sibling would still be claiming "0 clicks recorded"
# after the click that fixed it.
# `| tojson` on the user, not bare interpolation. hx-vals is JSON inside an
# HTML attribute, and Jinja's autoescape handles the HTML layer only -- it
# neutralises < > & ' " and not backslash. A reader named `ann\` produced
# `{"user":"ann\",...}`, which htmx cannot parse, so both vote buttons
# silently stopped working for that persona. `reader()` auto-creates a persona
# from /?user=, so a typo reaches it with no setup.
# The form a reader fills in to exist. Rendered in the feed slot when the
# database has no personas at all, and at /onboard from the nav otherwise.
# Before this, `/` defaulted to a persona named after the author and created
# it on the spot for whoever opened the page: a stranger's first screen was
# someone else's reading profile. Ranking starts from the interests text
# alone, so that is the one thing worth asking for.
ONBOARD = env.from_string("""<div id="feed">
<h2>Who is reading?</h2>
<form method="post" action="/personas">
 <p><label>Name <input name="name" required autofocus></label></p>
 <p><label>What do you read about?<br>
  <textarea name="interests" rows="3" cols="60" required
   placeholder="e.g. retrieval and ranking, quantum chemistry, evaluation methodology"></textarea>
 </label></p>
 <p><button type="submit">Start reading</button></p>
</form>
</div>""")

# Rendered in the feed slot when the embedder is down and this reader has no
# cached profile vector -- the state every reader is in on a fresh `attest
# serve` before Ollama is up, since rank.py's cache is in-process memory. It
# used to be a bare "Internal Server Error" with a traceback in the log; this
# says what `attest ingest` says for the same condition.
EMBEDDER_DOWN = env.from_string("""<div id="feed">
<p class="caveat">embedding model unreachable at {{ url }} -- is ollama running?
(<code>attest install --check</code> diagnoses this.) The feed cannot be ranked
until it is back; reload once it is.</p>
</div>""")

FRAGMENT = env.from_string("""<div id="feed">
{% if quality and quality.caveat %}
<p class="caveat">{{ quality.caveat }}</p>
{% endif %}
<ol>
{% for it in items %}
 <li data-item-id="{{ it.item_id }}">
  <a href="{{ it.url | safe_href }}">{{ it.title | clip }}</a>
  <span class="src">{{ it.source | clip(60) if it.source else '' }}
   · rank {{ '%.1f' % it.score }}</span>
  {% if it.content_type %}<span class="tag type">{{ it.content_type | clip(40) }}</span>{% endif %}
  {% for t in it.tags %}<span class="tag">{{ t | clip(40) }}</span>{% endfor %}
  <button class="yn" hx-post="/clicks"
      hx-vals='{"user":{{ user | tojson }},"item_id":{{ it.item_id }},"useful":1}'
      hx-target="#feed" hx-swap="outerHTML">✓</button>
  <button class="yn" hx-post="/clicks"
      hx-vals='{"user":{{ user | tojson }},"item_id":{{ it.item_id }},"useful":0}'
      hx-target="#feed" hx-swap="outerHTML">✗</button>
  {% if loop.index <= explain_limit %}
  <div class="why" hx-get="/explanation?user={{ user | urlarg }}&item_id={{ it.item_id }}"
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
    """Build the HTMX web UI as a fresh FastAPI app bound to `db_path`.

    Opens one SQLite connection PER THREAD rather than one for the app (see
    the comment below): FastAPI runs sync routes in a threadpool, and a
    shared connection let concurrent requests interleave cursors, which
    measurably corrupted results under the page's own concurrent load.
    """
    if embedder is None:
        from attestation.embed import Embedder

        embedder = Embedder()
    chat_fn = chat_fn or default_chat_fn
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    # One connection PER THREAD, not one for the app. FastAPI runs sync routes
    # in a threadpool and get_db passes check_same_thread=False, so a shared
    # connection let concurrent requests interleave cursors: 11 of 200 threaded
    # get_user calls returned None for a user that exists, and 11 of 12
    # parallel /list requests 500'd -- one of them because that None reached
    # autocreate_user and collided on the UNIQUE name.
    #
    # Not a stress case: the page fires up to 20 lazy /explanation requests per
    # load, so a reload mid-load is enough. SQLite connections are cheap and
    # WAL lets readers proceed during a write; sharing one bought nothing.
    _local = threading.local()

    def connection() -> sqlite3.Connection:
        """This thread's own SQLite connection, opened once and reused --
        see the comment above `_local` for why one per thread, not one for
        the app."""
        existing = getattr(_local, "conn", None)
        if existing is None:
            existing = get_db(db_path)
            _local.conn = existing
        return existing

    def require_user(user_name: str) -> sqlite3.Row:
        """An existing reader, or 404. Used by the routes that WRITE.

        Recording a click or asking for an explanation under an unknown name
        is far more likely a typo or a stale form than a new reader arriving,
        and creating a persona as the side effect of a vote produces exactly
        the "permanent persona nobody knows exists" that mcp/_tool.py warns
        about -- except here nothing announces it. Reads create; writes refuse.
        """
        conn = connection()
        user = get_user(conn, user_name)
        if user is None:
            raise HTTPException(status_code=404, detail=f"unknown user: {user_name}")
        return user

    def reader(user_name: str, request: Request) -> sqlite3.Row:
        """An existing reader, created on first sight. Used by the routes that READ.

        Browsing used to 404. Two front doors onto one database disagreeing
        about what a new name means is drift a reader hits before they have
        any way to understand it: they visit /?user=<their name>, get an error
        page, and never learn that any MCP tool with the same name would
        simply have worked. mcp/_tool.py already made this call and recorded
        why -- an unknown-name refusal taught agents to invent personas, so
        the refusal caused the duplicates it was meant to prevent. The web UI
        just never got the same treatment.

        The origin check guards CREATION only, not the read. Creating a
        persona is a write, and a write reachable by GET is reachable from
        any page the reader visits -- `<img src="http://127.0.0.1:8899/?
        user=typo">` is enough, since the victim's browser is the request
        origin. Guarding the whole route instead would make this a general
        browsing restriction and would break reading an EXISTING feed
        cross-origin, which discloses nothing the reader could not see by
        visiting the URL themselves. So: reads are always allowed; the
        INSERT is not.
        """
        conn = connection()
        if not user_name.strip():
            raise HTTPException(status_code=400, detail="user name required")
        user = get_user(conn, user_name)
        if user is None:
            require_same_origin(request)
            user, _ = autocreate_user(conn, user_name)
        return user

    def render_list(user_name: str, request: Request) -> str:
        """The ranked-list HTML fragment for one reader, or a cold-embedder
        notice in its place -- shared by the index page and the `/list`
        HTMX endpoint so both render identically."""
        conn = connection()
        user = reader(user_name, request)
        try:
            items = rank_items(conn, embedder, user["id"])[:LIST_LIMIT]
        except EmbedderUnavailable:
            return EMBEDDER_DOWN.render(url=base_url())
        return FRAGMENT.render(
            items=items,
            user=user_name,
            explain_limit=EXPLAIN_LIMIT,
            quality=ranking_quality(conn, user["id"]),
        )

    def persona_names() -> list[str]:
        """Every persona name, for the page's persona picker."""
        return [r["name"] for r in connection().execute("SELECT name FROM users ORDER BY name")]

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, user: str | None = Query(None)):
        """`/` with no `?user`: the first persona if any exist, else the
        onboarding form -- see CLAUDE.md's note that this replaced a
        hardcoded default persona that autocreated for whoever opened it."""
        users = persona_names()
        if user is None:
            if not users:
                return PAGE.render(users=users, user=None, feed_content=ONBOARD.render())
            user = users[0]
        # creates the reader if new, and only from a same-origin page; see reader()
        feed_content = render_list(user, request)
        return PAGE.render(users=persona_names(), user=user, feed_content=feed_content)

    @app.get("/onboard", response_class=HTMLResponse)
    def onboard():
        """The onboarding form on demand, e.g. from a "create another
        persona" link, rather than only when no persona exists yet."""
        return PAGE.render(users=persona_names(), user=None, feed_content=ONBOARD.render())

    @app.post("/personas", dependencies=[Depends(require_same_origin)])
    def create_persona(name: str = Form(...), interests: str = Form(...)):
        """The onboarding form's submit target: create a persona and land on
        its feed. Same-origin only (see `require_same_origin`) since this is
        the one route that both writes and is reachable with no existing
        persona to authenticate the request against."""
        try:
            create_user(connection(), name.strip(), interests.strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(f"/?user={quote(name.strip())}", status_code=303)

    @app.get("/list", response_class=HTMLResponse)
    def list_view(request: Request, user: str = Query(...)):
        """The HTMX fragment `index` embeds -- also fetched directly on a
        persona switch, without reloading the whole page."""
        return render_list(user, request)

    @app.post("/clicks", response_class=HTMLResponse, dependencies=[Depends(require_same_origin)])
    def click(
        request: Request,
        user: str = Form(...),
        item_id: int = Form(...),
        useful: int = Form(...),
    ):
        """Record a useful/not-useful vote and return the re-ranked list --
        `require_user` refuses an unknown name rather than creating one, per
        `require_user`'s own docstring."""
        u = require_user(user)
        record_click(connection(), u["id"], item_id, bool(useful), source="ui")
        # retrain + re-rank happens inside rank_items. require_user already
        # proved the persona exists, so reader() cannot reach its create path.
        return render_list(user, request)

    @app.get("/explanation", response_class=PlainTextResponse)
    def explanation(user: str = Query(...), item_id: int = Query(...)):
        """Lazy-loaded "why this?" text for one item, plain text so the
        page's own JS can drop it straight into the DOM without escaping."""
        u = require_user(user)
        return explain(connection(), u["id"], item_id, chat_fn=chat_fn).text or ""

    return app
