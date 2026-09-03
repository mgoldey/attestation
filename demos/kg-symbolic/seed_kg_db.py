# demos/kg-symbolic/seed_kg_db.py
"""Build a real, tagged demo database for the kg.* half of demo.py.

Reuses the flows fixture's own ingest path (examples/flows/_common.py)
against the REAL chat model at LLM_BASE_URL, not the --offline stub: the
stub's placeholder tags are schema-shaped filler ("existing", "vocabulary",
"title", one per field name) with no topical structure, so the graph built
from them has nothing worth showing on video. The graph is DERIVED from
tags (attestation.kg), so this is the one demo script that needs a model
server -- everything else in demos/ is pure local computation or the stub.

    ollama serve   # or whatever LLM_BASE_URL points at
    uv run python seed_kg_db.py /path/to/demo.db
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

FLOWS_DIR = Path(__file__).resolve().parents[2] / "examples" / "flows"
sys.path.insert(0, str(FLOWS_DIR))
import _common  # noqa: E402


def seed(db_path: Path) -> dict:
    from attestation.db import get_db
    from attestation.embed import Embedder
    from attestation.features import run_tagging
    from attestation.ingest import run_ingest
    from attestation.llm import default_chat_fn

    conn = get_db(db_path)
    embedder = Embedder()
    with tempfile.TemporaryDirectory() as tmp:
        ingest_stats = run_ingest(conn, embedder, _common.write_feeds_toml(Path(tmp)))
    if ingest_stats.get("embedder_down") or ingest_stats["added"] == 0:
        raise SystemExit(f"ingest added nothing: {ingest_stats} -- is the model server up?")
    from attestation.llm import chat_model

    tag_stats = run_tagging(conn, default_chat_fn, chat_model(), limit=None)
    conn.close()
    return {"ingest": ingest_stats, "tag": tag_stats}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: seed_kg_db.py DB_PATH", file=sys.stderr)
        return 1
    stats = seed(Path(sys.argv[1]))
    print(f"seeded -> {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
