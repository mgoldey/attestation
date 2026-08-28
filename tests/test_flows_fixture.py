"""The flows' corpus is the ground truth every printed number is scored
against, so its shape is pinned here, model-free."""

import importlib.util
from pathlib import Path

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _common():
    spec = importlib.util.spec_from_file_location("flows_common", FLOWS / "_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_entry_is_labelled_for_every_persona():
    c = _common()
    personas = c.load_personas()
    labels = c.load_labels()
    entries = c.corpus_entries()
    assert len(entries) >= 40, "the spec says about forty items"
    assert set(labels) == {e["guid"] for e in entries}, "labels and entries must match 1:1"
    for guid, per_persona in labels.items():
        assert set(per_persona) == set(personas), f"{guid} is not labelled for every persona"


def test_no_item_is_positive_for_both_personas():
    c = _common()
    for guid, per_persona in c.load_labels().items():
        assert sum(per_persona.values()) <= 1, f"{guid} is useful to more than one persona"


def test_each_persona_has_enough_of_both_classes():
    """evaluate_user stratifies folds on the minority class; ten of each
    keeps n_splits >= 2 after simulated reactions drop a few as unsure."""
    c = _common()
    labels = c.load_labels()
    for persona in c.load_personas():
        positives = sum(1 for v in labels.values() if v[persona])
        negatives = sum(1 for v in labels.values() if not v[persona])
        assert positives >= 10, f"{persona}: {positives} positives"
        assert negatives >= 10, f"{persona}: {negatives} negatives"


def test_entries_are_real_shaped():
    for e in _common().corpus_entries():
        assert e["title"] and len(e["summary"]) >= 80, e["guid"]
        assert e["link"].startswith("https://"), e["guid"]


def test_feeds_toml_points_at_the_corpus(tmp_path):
    c = _common()
    path = c.write_feeds_toml(tmp_path)
    text = path.read_text()
    assert str(c.CORPUS_DIR / "labelled.xml") in text


def test_entries_carry_no_date_so_the_demo_cannot_decay():
    """Ingest sets published to now when the entry has none, and the feed's
    14-day default window would otherwise empty this fixture in September."""
    for e in _common().corpus_entries():
        assert e["updated"] is None, e["guid"]
        assert e["published"] is None, e["guid"]
