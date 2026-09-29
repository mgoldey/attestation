"""Bounds and guards for third-party strings at a render boundary.

Shared by the web UI (server.py) and the Reading desk page (desk.py), so a
fix to either reaches both. No FastAPI, no Jinja: importing this must stay
cheap for the CLI and the MCP server.
"""


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
