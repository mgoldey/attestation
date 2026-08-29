"""mkdocs hook for the site home.

`docs/index.md` is the README, included rather than copied so the two cannot
drift. The README is written for the repository root, so its links into
`docs/` (`[..](docs/guides/x.md)`) would resolve to `docs/docs/...` once
`docs/` is the serving root. This hook performs the include itself, rather
than just rewriting text mkdocs has already inlined, because
`on_page_markdown` runs BEFORE `pymdownx.snippets` expands `--8<--` -- at
the point this hook sees the page, `docs/index.md`'s content is still the
one-line `--8<-- "README.md"` directive, not the README's text. So the hook
reads the README itself, drops the `docs/` prefix from its links, and
substitutes the result in place of the directive; snippets never gets a
chance to inline the unrewritten original. Everything else the README links
(`CONTRIBUTING.md`, `CHANGELOG.md`, `examples/`) already resolves via the
symlinks in `docs/`.
"""

from pathlib import Path

INCLUDE = '--8<-- "README.md"'


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != "index.md" or INCLUDE not in markdown:
        return markdown
    readme = Path(config.config_file_path).with_name("README.md").read_text()
    return markdown.replace(INCLUDE, readme.replace("](docs/", "]("))
