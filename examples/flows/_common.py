"""Shared by the flow scripts. Not a package: scripts insert their own
directory on sys.path; tests load this file by path."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

FLOWS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = FLOWS_DIR / "corpus"
REPO_ROOT = FLOWS_DIR.parents[1]


def load_personas() -> dict[str, str]:
    cfg = tomllib.loads((CORPUS_DIR / "personas.toml").read_text())
    return {p["name"]: p["interests"] for p in cfg["personas"]}


def load_labels() -> dict[str, dict[str, bool]]:
    return json.loads((CORPUS_DIR / "labels.json").read_text())


def corpus_entries() -> list[dict]:
    import feedparser

    parsed = feedparser.parse(str(CORPUS_DIR / "labelled.xml"))
    return [
        {
            "guid": e.get("id"),
            "title": e.get("title", ""),
            "summary": e.get("summary", ""),
            "link": e.get("link", ""),
        }
        for e in parsed.entries
    ]


def write_feeds_toml(directory: Path) -> Path:
    """A feeds.toml whose one feed is the local corpus file.

    feedparser accepts a filesystem path where ingest expects a URL, so the
    fixture goes through run_ingest unchanged -- no parse hook, no INSERT.
    """
    path = Path(directory) / "feeds.toml"
    xml = CORPUS_DIR / "labelled.xml"
    path.write_text(f'[[feeds]]\nurl = "{xml}"\ntitle = "flows fixture"\n')
    return path


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"flows_{name}", FLOWS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
