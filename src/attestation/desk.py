"""The Reading desk: a private page of today's ranked papers whose verdicts
come back as clicks.

The page keeps its verdicts in one JSON blob that the hosting platform stores
in a small SQLite file. This module reads that file READ-ONLY and imports the
verdicts through `rank.record_click` before any ranking a person sees -- a
verdict changes nothing until a ranking is computed, so importing then loses
nothing. See docs/superpowers/specs/2026-09-29-reading-desk-design.md.

Configured from the environment (the checkout `.env`, loaded at the entry
points): ATTEST_DESK_STATE (the state file), ATTEST_DESK_USER (the persona the
verdicts belong to), ATTEST_DESK_PUBLISH (a command run after a build).
"""

from __future__ import annotations

import html
import json
import logging
import os
import shlex
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from attestation import paths
from attestation.rank import (
    HUMAN_CLICK_SOURCES,
    get_user,
    rank_items,
    ranking_quality,
    record_click,
)
from attestation.render import clip

log = logging.getLogger(__name__)

STATE_VERSION = 1
# Seconds a read waits on a lock the page host holds mid-save. The host's own
# write is a single-row replace, so anything longer means something is wrong
# and ranking should go ahead on the clicks it already has.
READ_TIMEOUT_S = 5


@dataclass(frozen=True)
class Imported:
    """What one import did. `unchanged` is a verdict already recorded the same
    way; `skipped` is an entry naming no known item or carrying no bool."""

    recorded: int = 0
    unchanged: int = 0
    skipped: int = 0


def desk_config() -> tuple[Path | None, str | None]:
    """(state file, persona name) from the environment; each None when blank."""
    raw_state = (os.environ.get("ATTEST_DESK_STATE") or "").strip()
    raw_user = (os.environ.get("ATTEST_DESK_USER") or "").strip()
    return (Path(raw_state).expanduser() if raw_state else None, raw_user or None)


def read_state(path: str | Path) -> dict:
    """The page's state blob, or {} when it is missing, unreadable, or not v1.

    Opened with `mode=ro` so this can never write, create, or hold a write
    lock on a file another program owns.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(
            path.absolute().as_uri() + "?mode=ro", uri=True, timeout=READ_TIMEOUT_S
        )
        try:
            row = conn.execute("SELECT value FROM state WHERE id = 1").fetchone()
        finally:
            conn.close()
        value = json.loads(row[0]) if row else {}
    except (sqlite3.Error, ValueError) as exc:
        log.warning("desk: cannot read page state %s: %s", path, exc)
        return {}
    if not isinstance(value, dict) or value.get("v") != STATE_VERSION:
        if value:
            log.warning("desk: page state %s is not version %d; ignored", path, STATE_VERSION)
        return {}
    return value


def _utc(value) -> datetime | None:
    """A click's `clicked_at` (SQLite's naive-UTC 'YYYY-MM-DD HH:MM:SS') or a
    page verdict's `at` (ISO 8601 with Z), as an aware UTC datetime."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _newer(clicked_at, at) -> bool:
    """Whether the recorded click is newer than the page's verdict. An
    unreadable time on either side keeps the old rule: the page verdict applies."""
    recorded, given = _utc(clicked_at), _utc(at)
    return recorded is not None and given is not None and recorded > given


def import_verdicts(conn, user_id: int, state: dict) -> Imported:
    """Record each `verdicts[<item_id>] = {"useful": bool, ...}` as a `ui` click.

    Idempotent: a verdict already recorded the same way is left alone, so the
    hourly import and every ranking call can run it without rewriting rows.
    """
    verdicts = state.get("verdicts")
    if not isinstance(verdicts, dict):
        return Imported()
    recorded = unchanged = skipped = 0
    for key, entry in verdicts.items():
        useful = entry.get("useful") if isinstance(entry, dict) else None
        if not (isinstance(key, str) and key.isdecimal()) or not isinstance(useful, bool):
            skipped += 1
            continue
        item_id = int(key)
        if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is None:
            skipped += 1
            continue
        row = conn.execute(
            "SELECT useful, clicked_at FROM clicks WHERE user_id = ? AND item_id = ?",
            (user_id, item_id),
        ).fetchone()
        if row is not None and (bool(row[0]) == useful or _newer(row[1], entry.get("at"))):
            # Same verdict, or a newer one from another surface (chat, the web
            # UI): the last verdict wins, whichever surface gave it.
            unchanged += 1
            continue
        record_click(conn, user_id, item_id, useful, source="ui")
        recorded += 1
    return Imported(recorded, unchanged, skipped)


def import_pending(conn) -> Imported | None:
    """Import the configured page's verdicts; None when unconfigured or failed.

    Never raises: this runs in front of every ranking, and a broken state file
    must degrade to ranking on the clicks already recorded, not to an error.
    Never creates a persona either -- verdicts for a name that does not exist
    yet wait in the state file until it does.
    """
    state_path, name = desk_config()
    if state_path is None or name is None:
        return None
    user = get_user(conn, name)
    if user is None:
        log.warning("desk: ATTEST_DESK_USER=%r is not a persona yet; nothing imported", name)
        return None
    try:
        return import_verdicts(conn, user["id"], read_state(state_path))
    except (sqlite3.Error, ValueError) as exc:
        log.warning("desk: import failed, ranking on existing clicks: %s", exc)
        return None


# --- the page ------------------------------------------------------------------

DEFAULT_DESK_LIMIT = 20
SUMMARY_CHARS = 600
TITLE_CHARS = 300
SOURCE_CHARS = 80
TAG_CHARS = 40
# The page's pruning threshold: under the platform's 32,000-char state cap
# with room for the envelope, so a save never fails on size.
STATE_PRUNE_CHARS = 28_000


def _https_or_none(url: str | None) -> str | None:
    """Only https survives: the page host opens https links behind a confirm
    dialog and drops everything else silently, so any other scheme would be
    a link that does nothing. Plain text is the honest rendering."""
    return url if isinstance(url, str) and url.startswith("https://") else None


def _utc_iso(value: str | None) -> str:
    """`published` as the page's clock reads it: ISO 8601 with a `Z`.

    SQLite's datetime('now') default writes '2026-09-28 08:00:00' and ingest
    writes '2026-09-29T17:06:45', both UTC and neither saying so; a browser
    parses a bare timestamp as LOCAL time, which would age every paper by the
    reader's offset."""
    if not value:
        return ""
    text = str(value).strip().replace(" ", "T", 1)
    return text if text.endswith("Z") or "+" in text[10:] else text + "Z"


def desk_payload(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> dict:
    """What the page shows, as data: the ranked items, the ranking caveat
    while the classifier is off, and how many papers a person has rated."""
    items = rank_items(conn, embedder, user_id)[:limit]
    quality = ranking_quality(conn, user_id)
    sources = sorted(HUMAN_CLICK_SOURCES)
    rated = conn.execute(
        "SELECT COUNT(*) FROM clicks WHERE user_id = ? AND source IN ({})".format(
            ",".join("?" * len(sources))
        ),
        (user_id, *sources),
    ).fetchone()[0]
    ids = [it.item_id for it in items]
    published = dict(
        conn.execute(
            "SELECT id, published FROM items WHERE id IN ({})".format(",".join("?" * len(ids))),
            ids,
        ).fetchall()
        if ids
        else []
    )
    return {
        "items": [
            {
                "id": it.item_id,
                "title": clip(it.title, TITLE_CHARS),
                "url": _https_or_none(it.url),
                "source": clip(it.source or "", SOURCE_CHARS),
                "published": _utc_iso(published.get(it.item_id)),
                "tags": [clip(t, TAG_CHARS) for t in it.tags[:3]],
                "summary": clip((it.summary or "").strip(), SUMMARY_CHARS),
            }
            for it in items
        ],
        "caveat": None if quality.get("classifier_active") else quality.get("caveat"),
        "rated": rated,
    }


def _embed_json(value: dict) -> str:
    """JSON safe inside a <script> element: `<` never appears raw, so no
    string in the data can close the element it lives in."""
    return json.dumps(value).replace("<", "\\u003c")


# Pure functions, no DOM: tests/test_desk.py runs exactly this string under node.
DESK_LOGIC_JS = """
function prune(verdicts, cap) {
  const entries = Object.entries(verdicts).sort((a, b) => (a[1].at < b[1].at ? -1 : 1));
  const size = () => JSON.stringify({v: 1, verdicts: Object.fromEntries(entries)}).length;
  while (entries.length && size() > cap) entries.shift();
  return Object.fromEntries(entries);
}
function judgedHere(items, verdicts) {
  return items.filter(i => verdicts[String(i.id)]).length;
}
function merge(mine, theirs) {
  const out = Object.assign({}, theirs);
  for (const [k, e] of Object.entries(mine)) if (!out[k] || out[k].at < e.at) out[k] = e;
  return out;
}
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function ago(iso, now) {
  const t = Date.parse(iso);
  if (!iso || Number.isNaN(t)) return '';
  const s = Math.max(0, (now - t) / 1000);
  if (s < 60) return 'now';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  if (s < 86400) return Math.floor(s / 3600) + 'h';
  if (s < 7 * 86400) return Math.floor(s / 86400) + 'd';
  const d = new Date(t);
  return MONTHS[d.getUTCMonth()] + ' ' + d.getUTCDate();
}
function failureText(r) {
  // The page host keeps a failed save pending and refuses new ones until its
  // own Retry (in the bar above the page) runs, so that is the only advice.
  const why = r && (r.error || r.message);
  return (why ? why + ' ' : 'Not saved. ') + 'Use Retry at the top of the page.';
}
function mark(source) {
  const m = /[A-Za-z0-9]/.exec(source || '');
  let h = 2166136261;
  for (const c of source || '') { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  return {letter: m ? m[0].toUpperCase() : '?', hue: h % 360};
}
"""

# The page script. `window.agentmarkit` is the page host's bridge (state,
# save, dirty); opened as a plain file it is absent and the page is a list.
# Only textContent/createElement -- never innerHTML -- so no field is markup.
DESK_UI_JS = (
    """
(() => {
  const data = JSON.parse(document.getElementById('desk-data').textContent);
  const am = window.agentmarkit;
  const SVG = 'http://www.w3.org/2000/svg';  // an XML namespace name, never fetched
  let revision = 0, verdicts = {}, toastTimer = 0;
  const $ = s => document.querySelector(s);
  function say(t, sticky) {
    const n = $('#toast');
    n.textContent = t; n.dataset.on = t ? '1' : '';
    clearTimeout(toastTimer);
    if (t && !sticky) toastTimer = setTimeout(() => { n.dataset.on = ''; }, 1600);
  }
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }
  function icon(d) {
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '0 0 16 16'); svg.setAttribute('aria-hidden', 'true');
    const p = document.createElementNS(SVG, 'path');
    p.setAttribute('d', d);
    svg.append(p);
    return svg;
  }
  const CHECK = 'M3 8.5l3.2 3L13 4.5', CROSS = 'M4 4l8 8M12 4l-8 8';
  function buttons(id) {
    const box = el('div', 'verdict');
    const v = verdicts[String(id)];
    for (const [useful, label, d] of [[true, 'Useful', CHECK], [false, 'Not my area', CROSS]]) {
      const b = el('button', useful ? 'yes' : 'no');
      b.type = 'button';
      b.append(icon(d), el('span', '', label));
      b.setAttribute('aria-pressed', String(v ? v.useful === useful : false));
      b.addEventListener('click', () => judge(id, useful));
      box.append(b);
    }
    return box;
  }
  function row(item, full, now) {
    const v = verdicts[String(item.id)];
    const li = el('li', 'paper' + (v ? (v.useful ? ' judged useful' : ' judged notmine') : ''));
    const m = mark(item.source);
    const badge = el('span', 'mark', m.letter);
    badge.style.setProperty('--h', m.hue);
    badge.setAttribute('aria-hidden', 'true');
    const body = el('div', 'body');
    const title = item.url ? el('a', 'title', item.title) : el('span', 'title', item.title);
    if (item.url) { title.href = item.url; title.rel = 'noreferrer'; }
    const meta = el('div', 'meta');
    meta.append(el('span', 'src', item.source));
    const age = ago(item.published, now);
    if (age) meta.append(el('span', 'age', age));
    for (const t of item.tags) meta.append(el('span', 'tag', t));
    body.append(title, meta);
    if (full && item.summary) body.append(el('p', 'summary', item.summary));
    if (am) body.append(buttons(item.id));
    li.append(badge, body);
    return li;
  }
  function render() {
    const now = Date.now();
    for (const [sel, full] of [['#triage', false], ['#read', true]]) {
      $(sel).replaceChildren(...data.items.map(i => row(i, full, now)));
    }
    const total = data.items.length, done = judgedHere(data.items, verdicts);
    $('#progress span').style.width = total ? (100 * done / total) + '%' : '0';
    $('#progress').setAttribute('aria-valuenow', String(done));
    $('#done').textContent = am && total ? done + ' of ' + total + ' judged' : '';
  }
  async function save(retry) {
    const r = await am.save({v: 1, verdicts: prune(verdicts, CAP)}, revision);
    if (r && r.ok) {
      revision = r.state.revision;
      verdicts = (r.state.data && r.state.data.verdicts) || {};
      am.dirty(false); say('Saved'); render(); return;
    }
    if (r && r.conflict && retry) {
      revision = r.state.revision;
      verdicts = merge(verdicts, (r.state.data && r.state.data.verdicts) || {});
      return save(false);
    }
    say(failureText(r), true);
  }
  function judge(id, useful) {
    verdicts[String(id)] = {useful, at: new Date().toISOString()};
    am.dirty(true); render();
    save(true).catch(e => say(failureText(e), true));
  }
  async function load() {
    try {
      const r = await am.state();
      if (r && r.ok) {
        revision = r.state.revision;
        const d = r.state.data || {};
        verdicts = d.v === 1 && d.verdicts ? d.verdicts : {};
      }
    } catch (e) { say(e.message, true); }
    render();
  }
  for (const b of document.querySelectorAll('[data-tab]')) {
    b.addEventListener('click', () => {
      document.body.dataset.show = b.dataset.tab;
      for (const o of document.querySelectorAll('[data-tab]'))
        o.setAttribute('aria-pressed', String(o === b));
    });
  }
  if (am) { load(); addEventListener('agentmarkit-resume', load); } else { render(); }
})();
"""
).replace("CAP", str(STATE_PRUNE_CHARS))

_DESK_CSS = """
:root{--bg:#fff;--fg:#1d2127;--muted:#667080;--line:#e4e7eb;--soft:#f3f5f7;
--yes:#1e7f55;--yes-soft:#e3f2ea;--no:#8a5a2b;--no-soft:#f5ece3;--link:#1d2127;
--mark-l:46%;--mark-s:38%;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e6e8eb;--muted:#9aa3ae;
--line:#252a31;--soft:#191d22;--yes:#5cc095;--yes-soft:#16291f;--no:#d9a36b;
--no-soft:#2a2016;--link:#e6e8eb;--mark-l:62%;--mark-s:42%;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
-webkit-font-smoothing:antialiased}
header{position:sticky;top:0;z-index:2;background:var(--bg);padding:14px 16px 0;
border-bottom:1px solid var(--line)}
.bar{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
h1{font-size:1.35rem;line-height:1.2;margin:0;font-weight:700;letter-spacing:-.01em}
#done{color:var(--muted);font-size:.85rem;font-variant-numeric:tabular-nums}
.sub{color:var(--muted);font-size:.85rem;margin:2px 0 10px}
#progress{height:3px;background:var(--soft);border-radius:3px;overflow:hidden}
#progress span{display:block;height:100%;width:0;background:var(--yes);
transition:width .25s ease}
nav{display:flex;gap:4px;padding:10px 0}
nav button{font:inherit;font-size:.9rem;padding:5px 12px;border-radius:7px;border:0;
background:none;color:var(--muted);cursor:pointer}
nav button[aria-pressed=true]{background:var(--soft);color:var(--fg);font-weight:600}
.caveat{margin:12px 16px 0;padding:10px 12px;border-radius:8px;background:var(--soft);
color:var(--muted);font-size:.85rem}
main{max-width:680px;margin:0 auto}
body[data-show=triage] #read,body[data-show=read] #triage{display:none}
ol{list-style:none;margin:0;padding:0}
.paper{display:grid;grid-template-columns:28px minmax(0,1fr);gap:12px;padding:14px 16px;
border-bottom:1px solid var(--line)}
.mark{width:28px;height:28px;border-radius:7px;display:grid;place-items:center;
font-size:.8rem;font-weight:700;color:#fff;
background:hsl(var(--h) var(--mark-s) var(--mark-l))}
.title{display:block;font-size:1rem;font-weight:600;line-height:1.3;color:var(--link);
text-decoration:none;text-wrap:pretty}
a.title:hover{text-decoration:underline;text-underline-offset:2px}
.meta{display:flex;gap:10px;margin-top:4px;font-size:.8rem;color:var(--muted);
white-space:nowrap;overflow:hidden;mask-image:linear-gradient(90deg,#000 85%,transparent)}
.meta>*{flex:none}
.src{font-weight:600;color:var(--fg);opacity:.75}
.tag{padding:0 7px;border:1px solid var(--line);border-radius:999px}
.summary{margin:10px 0 0;max-width:62ch;
font:1rem/1.6 Charter,"Iowan Old Style","Palatino Linotype",Georgia,serif}
.verdict{display:flex;gap:8px;margin-top:10px}
.verdict button{display:inline-flex;align-items:center;gap:6px;font:inherit;
font-size:.82rem;padding:5px 11px 5px 9px;border-radius:999px;cursor:pointer;
border:1px solid var(--line);background:var(--bg);color:var(--muted)}
.verdict svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round}
.verdict .yes[aria-pressed=true]{background:var(--yes-soft);border-color:transparent;
color:var(--yes);font-weight:600}
.verdict .no[aria-pressed=true]{background:var(--no-soft);border-color:transparent;
color:var(--no);font-weight:600}
.judged .title{font-weight:500;color:var(--muted)}
.judged .mark{opacity:.45}
.judged.notmine .title{text-decoration:line-through;text-decoration-thickness:1px}
body[data-show=triage] .judged{padding-top:10px;padding-bottom:10px}
body[data-show=triage] .judged .meta{display:none}
body[data-show=triage] .judged .verdict{margin-top:6px}
button:focus-visible,a:focus-visible{outline:2px solid var(--yes);outline-offset:2px}
.empty{padding:48px 16px;color:var(--muted);text-align:center}
#toast{position:fixed;left:50%;bottom:16px;transform:translate(-50%,8px);opacity:0;
background:var(--fg);color:var(--bg);font-size:.85rem;padding:7px 14px;border-radius:8px;
transition:opacity .2s,transform .2s;pointer-events:none}
#toast[data-on="1"]{opacity:1;transform:translate(-50%,0)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""


def render_desk(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> str:
    """One self-contained HTML page: no request of its own, safe under the
    page host's CSP (see the spec's "What the platform allows")."""
    payload = desk_payload(conn, embedder, user_id, limit)
    n = len(payload["items"])
    count = f"{n} paper{'s' if n != 1 else ''}, best first" if n else "Nothing new yet"
    rated = payload["rated"]
    sub = f"{count}. You have rated {rated}." if rated else f"{count}."
    empty = "" if n else '<p class="empty">No papers yet: the refresh fetches new ones hourly.</p>'
    caveat = f'<p class="caveat">{html.escape(payload["caveat"])}</p>' if payload["caveat"] else ""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>Reading desk</title><style>{_DESK_CSS}</style></head>"
        '<body data-show="triage"><header>'
        '<div class="bar"><h1>Reading desk</h1><span id="done"></span></div>'
        f'<p class="sub">{html.escape(sub)}</p>'
        '<div id="progress" role="progressbar" aria-label="Papers judged" '
        f'aria-valuemin="0" aria-valuemax="{n}" aria-valuenow="0"><span></span></div>'
        '<nav><button type="button" data-tab="triage" aria-pressed="true">Titles</button>'
        '<button type="button" data-tab="read" aria-pressed="false">Abstracts</button></nav>'
        f"</header><main>{caveat}{empty}"
        '<ol id="triage"></ol><ol id="read"></ol></main>'
        '<div id="toast" role="status"></div>'
        f'<script type="application/json" id="desk-data">{_embed_json(payload)}</script>'
        f"<script>{DESK_LOGIC_JS}{DESK_UI_JS}</script></body></html>"
    )


# --- publishing -------------------------------------------------------------------

PUBLISH_TIMEOUT_S = 60


def desk_output_path() -> Path:
    """Where the configured refresh writes the page."""
    return paths.hermes_home() / "workspace" / "research-desk" / "desk.html"


def publish_argv(command: str) -> list[str]:
    """ATTEST_DESK_PUBLISH as argv: shell-style quoting, `~` expanded per
    argument, and no shell -- the value comes from a file, and nothing in it
    needs pipes or variables."""
    return [os.path.expanduser(arg) for arg in shlex.split(command)]


def publish(command: str) -> subprocess.CompletedProcess:
    """Run ATTEST_DESK_PUBLISH once; the caller reads the return code."""
    return subprocess.run(
        publish_argv(command),
        capture_output=True,
        text=True,
        timeout=PUBLISH_TIMEOUT_S,
        check=False,
    )
