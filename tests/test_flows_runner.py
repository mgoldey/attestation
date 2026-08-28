"""run_all's rendering and its refusal to record offline numbers."""

import importlib.util
from pathlib import Path

import pytest

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _runner():
    spec = importlib.util.spec_from_file_location("flows_run_all", FLOWS / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _reports(mode):
    return {
        "persona_eval": {
            "mode": mode,
            "chat_model": "m",
            "embed_model": "e",
            "items": 40,
            "personas": [
                {
                    "persona": "bench-chemist",
                    "seconds_per_reaction": 1.5,
                    "reactions": {
                        "precision": 0.9,
                        "recall": 0.8,
                        "auc": None,
                        "tp": 9,
                        "fp": 1,
                        "fn": 2,
                        "tn": 28,
                        "n_unsure": 0,
                        "confidence_histogram": {5: 40},
                    },
                    "ranker": {
                        "rank_auc": 0.95,
                        "classifier_auc": 0.7,
                        "classifier_n_clicks": 40,
                        "provenance_auc": None,
                    },
                }
            ],
        },
        "mcp_e2e": {
            "mode": mode,
            "chat_model": "m",
            "rows": [
                {"surface": "feed", "tool": "feed.list", "problem": None, "ok": True},
                {"surface": "full", "tool": "sym.solve", "problem": None, "ok": False},
            ],
        },
        "train_mlflow": {
            "seconds": 4.2,
            "arms": [{"C": 1.0, "auc": 0.99, "precision": 0.98, "recall": 0.97, "accuracy": 0.97}],
        },
    }


def test_results_markdown_names_mode_models_and_every_number():
    text = _runner().render_results(_reports("live"), when="2026-08-28")
    assert "2026-08-28" in text and "live" in text and "chat=m" in text
    assert "0.900" in text and "0.800" in text and "n/a" in text  # precision, recall, inert AUC
    assert "0.950" in text and "feed.list" in text and "4.2" in text


def test_offline_numbers_are_never_written_as_results():
    with pytest.raises(ValueError, match="offline"):
        _runner().render_results(_reports("offline"), when="2026-08-28")
