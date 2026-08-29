#!/usr/bin/env python
"""Generate `docs/reference/cli.md` from the live `attest` parser.

Walks `attestation.cli.build_parser()` and every subparser it holds (`runs`
has sub-subcommands, so this recurses rather than assuming one level) and
emits each command's `--help` text verbatim. The alternative -- a hand-typed
reference page -- drifts the moment a flag changes; this cannot, because
`tests/test_docs_site.py` asserts the committed file equals a fresh render.

`argparse` wraps `format_help()` to the terminal width, reading `COLUMNS`
via `shutil.get_terminal_size()` at CALL time (not at import time -- setting
it after `import argparse` would work just as well), so the same parser
renders different text in a wide terminal and a CI runner unless the width
is pinned. `COLUMNS=80` is set here, before `format_help()` is ever called,
and the test pins the same value.

Run: uv run python scripts/render_cli_reference.py
"""

import os

os.environ["COLUMNS"] = "80"

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "reference" / "cli.md"


def _walk(parser: argparse.ArgumentParser, path: list[str]) -> list[tuple[list[str], str]]:
    """Depth-first (name, help-text) pairs for `parser` and every subcommand.

    A leaf command (no `add_subparsers` of its own) contributes one entry.
    A command with subcommands (`runs`) contributes none for itself -- there
    is nothing to run at `attest runs` alone -- and recurses into each child.
    """
    if parser._subparsers is None:
        return [(path, parser.format_help())]

    subparsers_action = next(
        (
            a
            for a in parser._subparsers._group_actions  # type: ignore[union-attr]
            if isinstance(a, argparse._SubParsersAction)
        ),
        None,
    )
    if subparsers_action is None:
        return [(path, parser.format_help())]

    entries: list[tuple[list[str], str]] = []
    for action in subparsers_action._choices_actions:
        child = subparsers_action.choices[action.dest]
        entries.extend(_walk(child, [*path, action.dest]))
    return entries


def render() -> str:
    """The full CLI reference page: one `## attest <cmd>` section per leaf."""
    from attestation.cli import build_parser

    parser = build_parser()
    lines = [
        "# CLI reference",
        "",
        "Generated from `attestation.cli.build_parser()` by "
        "`scripts/render_cli_reference.py`; `tests/test_docs_site.py` asserts this "
        "file is a fresh render. Do not hand-edit.",
        "",
    ]
    for path, help_text in _walk(parser, []):
        lines.append(f"## attest {' '.join(path)}")
        lines.append("")
        lines.append("```")
        lines.append(help_text.rstrip("\n"))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> None:
    """Write the rendered reference to `docs/reference/cli.md`."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
