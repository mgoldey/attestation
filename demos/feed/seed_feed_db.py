# demos/feed/seed_feed_db.py
"""A tagged database plus one persona, for record.py to drive attest serve
against. Shares the ingest+tag path with demos/kg-symbolic/seed_kg_db.py:
real items, real tags, from the flows fixture corpus and the real chat
model at LLM_BASE_URL.

    ollama serve   # or whatever LLM_BASE_URL points at
    uv run python seed_feed_db.py /path/to/demo.db
"""

from __future__ import annotations

import sys
from pathlib import Path

KG_DIR = Path(__file__).resolve().parents[1] / "kg-symbolic"
sys.path.insert(0, str(KG_DIR))
from seed_kg_db import seed  # noqa: E402

PERSONA_NAME = "demo-reader"
PERSONA_INTERESTS = "machine learning systems and catalysis chemistry"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: seed_feed_db.py DB_PATH", file=sys.stderr)
        return 1
    db_path = Path(sys.argv[1])
    stats = seed(db_path)
    print(f"seeded -> {stats}")

    from attestation.db import get_db
    from attestation.rank import create_user

    conn = get_db(db_path)
    create_user(conn, PERSONA_NAME, PERSONA_INTERESTS)
    conn.close()
    print(f"persona -> {PERSONA_NAME!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
