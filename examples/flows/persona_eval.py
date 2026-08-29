"""Precision, recall and AUC for the LLM-driven half of the feed.

For each persona in corpus/personas.toml: ingest the labelled corpus through
the real ingest path, ask the chat model to react to EVERY item as that
persona (simulate.simulate_feedback -- one model call per item), and score
the verdicts against corpus/labels.json. Then score the ranker on the same
labels: the cross-validated click-classifier AUC `attest eval` prints, and
the AUC of rank_items' order, which `attest eval` cannot see.

What the numbers mean: agreement with the labels one person wrote for
forty items. Evidence about the flow, not a model benchmark.

    uv run python examples/flows/persona_eval.py --offline      # stub model
    uv run python examples/flows/persona_eval.py                # Ollama at LLM_BASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
import _common

# score_verdicts and rank_auc moved verbatim to evals/reaction_eval.py, which
# `tests/test_reaction_eval.py` covers; this module keeps a thin wrapper so
# there is one definition and `tests/test_flows_scoring.py` still passes.
from reaction_eval import rank_auc, score_verdicts


def prepare_db(db_path: Path, base_url: str, chat_model: str, embed_model: str) -> dict:
    """A fresh database holding the corpus and both personas, via the real paths."""
    from attestation.db import get_db
    from attestation.embed import Embedder
    from attestation.ingest import run_ingest
    from attestation.llm import EmbeddingClient
    from attestation.rank import create_user

    os.environ["LLM_BASE_URL"] = base_url
    os.environ["CHAT_MODEL"] = chat_model
    os.environ["EMBED_MODEL"] = embed_model
    conn = get_db(db_path)
    embedder = Embedder(EmbeddingClient(base_url=base_url, model=embed_model))
    with tempfile.TemporaryDirectory() as tmp:
        stats = run_ingest(conn, embedder, _common.write_feeds_toml(Path(tmp)))
    if stats.get("embedder_down") or stats["added"] == 0:
        raise SystemExit(f"ingest added nothing: {stats} -- is the model server at {base_url} up?")
    users = {
        name: create_user(conn, name, interests)
        for name, interests in _common.load_personas().items()
    }
    guid_to_id = {r["guid"]: r["id"] for r in conn.execute("SELECT id, guid FROM items")}
    conn.close()
    return {"items": stats["added"], "users": users, "guid_to_id": guid_to_id}


def evaluate_persona(
    db_path: Path,
    name: str,
    user_id: int,
    guid_to_id: dict,
    base_url: str,
    chat_model: str,
    embed_model: str,
) -> dict:
    from attestation.db import get_db
    from attestation.embed import Embedder
    from attestation.llm import ChatClient, EmbeddingClient
    from attestation.rank import evaluate_user, rank_items
    from attestation.simulate import simulate_feedback

    labels = {guid_to_id[g]: v[name] for g, v in _common.load_labels().items() if g in guid_to_id}
    conn = get_db(db_path)
    items = conn.execute("SELECT * FROM items ORDER BY id").fetchall()
    chat = ChatClient(base_url=base_url, model=chat_model)
    t0 = time.perf_counter()
    sim = simulate_feedback(conn, chat.chat_json, name, items)
    elapsed = time.perf_counter() - t0
    verdicts = score_verdicts(sim["reactions"], labels)

    embedder = Embedder(EmbeddingClient(base_url=base_url, model=embed_model))
    ranked = rank_items(conn, embedder, user_id, since_days=None, exclude_clicked=False)
    order = [r.item_id for r in ranked]
    clf = evaluate_user(conn, user_id)
    conn.close()
    return {
        "persona": name,
        "reactions": verdicts,
        "simulate_counts": sim["counts"],
        "seconds_per_reaction": elapsed / max(1, len(items)),
        "ranker": {
            "rank_auc": rank_auc(order, labels),
            "classifier_auc": None if clf is None else clf["auc"],
            "classifier_n_clicks": None if clf is None else clf["n_clicks"],
            "provenance_auc": None if clf is None else clf["provenance_auc"],
        },
    }


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render(report: dict) -> str:
    lines = [
        f"persona eval -- mode={report['mode']} chat={report['chat_model']} "
        f"embed={report['embed_model']} items={report['items']}",
        "numbers are agreement with corpus/labels.json (n=40); evidence about the flow,"
        " not a model benchmark",
    ]
    for p in report["personas"]:
        r, k = p["reactions"], p["ranker"]
        lines += [
            "",
            f"[{p['persona']}]  reactions: {p['seconds_per_reaction']:.2f}s each",
            f"  precision {_fmt(r['precision'])}  recall {_fmt(r['recall'])}  auc {_fmt(r['auc'])}"
            f"   (tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}, unsure={r['n_unsure']})",
            f"  confidence histogram {r['confidence_histogram']}"
            + ("   <- inert: AUC undefined" if r["auc"] is None else ""),
            f"  ranker: rank_auc {_fmt(k['rank_auc'])}  classifier_auc {_fmt(k['classifier_auc'])}"
            f" over {k['classifier_n_clicks']} clicks  provenance_auc {_fmt(k['provenance_auc'])}",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="use the stub model server")
    ap.add_argument("--db", type=Path, help="write the database here (default: a temp dir)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)

    from attestation.llm import base_url, chat_model, embed_model, load_env

    server = None
    if args.offline:
        import stub_openai

        server, url = stub_openai.start()
        chat, embed = stub_openai.MODEL, stub_openai.MODEL
    else:
        load_env()
        url, chat, embed = base_url(), chat_model(), embed_model()

    tmp = None if args.db else tempfile.TemporaryDirectory()
    db_path = args.db or Path(tmp.name) / "flows.db"
    try:
        prepared = prepare_db(db_path, url, chat, embed)
        report = {
            "flow": "persona_eval",
            "mode": "offline" if args.offline else "live",
            "chat_model": chat,
            "embed_model": embed,
            "items": prepared["items"],
            "personas": [
                evaluate_persona(db_path, name, uid, prepared["guid_to_id"], url, chat, embed)
                for name, uid in prepared["users"].items()
            ],
        }
    finally:
        if server:
            server.shutdown()
        if tmp:
            tmp.cleanup()
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
