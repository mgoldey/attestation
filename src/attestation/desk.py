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
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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
            "SELECT useful FROM clicks WHERE user_id = ? AND item_id = ?", (user_id, item_id)
        ).fetchone()
        if row is not None and bool(row[0]) == useful:
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
    return {
        "items": [
            {
                "id": it.item_id,
                "title": clip(it.title, TITLE_CHARS),
                "url": _https_or_none(it.url),
                "source": clip(it.source or "", SOURCE_CHARS),
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
function merge(mine, theirs) {
  const out = Object.assign({}, theirs);
  for (const [k, e] of Object.entries(mine)) if (!out[k] || out[k].at < e.at) out[k] = e;
  return out;
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
  let revision = 0, verdicts = {};
  const $ = s => document.querySelector(s);
  const say = t => { $('#status').textContent = t; };
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }
  function buttons(id) {
    const box = el('div', 'verdict');
    for (const [useful, label] of [[true, 'Useful'], [false, 'Not my area']]) {
      const b = el('button', useful ? 'yes' : 'no', label);
      b.type = 'button';
      const v = verdicts[String(id)];
      b.setAttribute('aria-pressed', String(v ? v.useful === useful : false));
      b.addEventListener('click', () => judge(id, useful));
      box.append(b);
    }
    return box;
  }
  function row(item, full) {
    const li = el('li', 'paper');
    const title = item.url ? el('a', 'title', item.title) : el('span', 'title', item.title);
    if (item.url) { title.href = item.url; title.rel = 'noreferrer'; }
    const meta = [item.source, ...item.tags].filter(Boolean).join(' \u00b7 ');
    li.append(title, el('div', 'meta', meta));
    if (full && item.summary) li.append(el('p', 'summary', item.summary));
    if (am) li.append(buttons(item.id));
    return li;
  }
  function render() {
    for (const [sel, full] of [['#triage', false], ['#read', true]]) {
      $(sel).replaceChildren(...data.items.map(i => row(i, full)));
    }
    const mine = Object.keys(verdicts).length;
    const here = mine ? ', ' + mine + ' on this page' : '';
    $('#rated').textContent = data.rated + ' rated so far' + here;
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
    say('Not saved yet. Tap again to retry.');
  }
  function judge(id, useful) {
    verdicts[String(id)] = {useful, at: new Date().toISOString()};
    am.dirty(true); say('Saving\u2026'); render();
    save(true).catch(e => say(e.message));
  }
  async function load() {
    try {
      const r = await am.state();
      if (r && r.ok) {
        revision = r.state.revision;
        const d = r.state.data || {};
        verdicts = d.v === 1 && d.verdicts ? d.verdicts : {};
      }
    } catch (e) { say(e.message); }
    render();
  }
  for (const b of document.querySelectorAll('[data-tab]')) {
    b.addEventListener('click', () => { document.body.dataset.show = b.dataset.tab; });
  }
  if (am) { load(); addEventListener('agentmarkit-resume', load); } else { render(); }
})();
"""
).replace("CAP", str(STATE_PRUNE_CHARS))

_DESK_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--accent:#2f5d50}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;
--line:#2c2b27;--accent:#8cc5b2}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:16px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:720px;margin:0 auto}
h1{font-size:1.3rem;margin:0 0 .25rem}
.sub,.meta,#rated,#status{color:var(--muted);font-size:.9rem}
.caveat{border-left:3px solid var(--accent);padding:.25rem .75rem;margin:.75rem 0}
nav{display:flex;gap:.5rem;margin:1rem 0}
nav button{flex:1;padding:.5rem;border:1px solid var(--line);background:none;color:var(--fg);
border-radius:6px;font:inherit}
body[data-show=triage] nav [data-tab=triage],body[data-show=read] nav [data-tab=read]{
border-color:var(--accent);color:var(--accent)}
body[data-show=triage] #read,body[data-show=read] #triage{display:none}
ol{list-style:none;padding:0;margin:0}
.paper{padding:.75rem 0;border-bottom:1px solid var(--line)}
.title{font-weight:600;color:var(--fg)}
a.title{color:var(--accent)}
.summary{margin:.4rem 0}
.verdict{display:flex;gap:.5rem;margin-top:.5rem}
.verdict button{padding:.35rem .75rem;border:1px solid var(--line);border-radius:999px;
background:none;color:var(--fg);font:inherit;font-size:.9rem}
.verdict button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
color:var(--bg)}
.empty{padding:2rem 0;color:var(--muted)}
"""


def render_desk(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> str:
    """One self-contained HTML page: no request of its own, safe under the
    page host's CSP (see the spec's "What the platform allows")."""
    payload = desk_payload(conn, embedder, user_id, limit)
    empty = (
        ""
        if payload["items"]
        else '<p class="empty">No papers yet: the refresh fetches new ones hourly.</p>'
    )
    caveat = f'<p class="caveat">{html.escape(payload["caveat"])}</p>' if payload["caveat"] else ""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>Reading desk</title><style>{_DESK_CSS}</style></head>"
        '<body data-show="triage"><main>'
        '<h1>Reading desk</h1><p class="sub">Today\'s papers, best first. '
        "Mark what helps and what is not your area; the ranking learns from both.</p>"
        f'{caveat}<p id="rated"></p><p id="status" role="status"></p>'
        '<nav><button type="button" data-tab="triage">Triage</button>'
        '<button type="button" data-tab="read">Read</button></nav>'
        f'{empty}<ol id="triage"></ol><ol id="read"></ol></main>'
        f'<script type="application/json" id="desk-data">{_embed_json(payload)}</script>'
        f"<script>{DESK_LOGIC_JS}{DESK_UI_JS}</script></body></html>"
    )
