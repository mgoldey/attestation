#!/usr/bin/env python
"""Generate `docs/site/specs.md`: every design spec, newest first.

Each spec file name starts with an ISO date (`2026-08-29-package-docs-
design.md`), so sorting by file name is sorting by date -- no reliance on
mtimes, which a fresh clone or CI checkout does not preserve. For each spec,
lists its first `#` heading and its `**Status:**` line (the two facts this
repo already writes at the top of every spec), so the index reads as a
one-line-per-spec table of contents rather than a second copy of the specs.

Run: uv run python scripts/render_spec_index.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "docs" / "superpowers" / "specs"
OUTPUT = ROOT / "docs" / "site" / "specs.md"

HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
STATUS_RE = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$", re.MULTILINE)


def _spec_line(path: Path) -> str:
    """One list entry: the spec's title (linked) and its status line."""
    text = path.read_text()
    heading = HEADING_RE.search(text)
    status = STATUS_RE.search(text)
    title = heading.group(1) if heading else path.stem
    status_text = status.group(1) if status else "(no Status line)"
    return f"- [{title}](../superpowers/specs/{path.name}) -- {status_text}"


def render() -> str:
    """The full design-records index, newest spec first."""
    specs = sorted(SPECS_DIR.glob("*.md"), reverse=True)
    lines = [
        "# Design records",
        "",
        "Every design spec under `docs/superpowers/specs/`, newest first. Generated "
        "by `scripts/render_spec_index.py`; `tests/test_docs_site.py` asserts this file "
        "is a fresh render. Do not hand-edit.",
        "",
    ]
    lines.extend(_spec_line(path) for path in specs)
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> None:
    """Write the rendered index to `docs/site/specs.md`."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
