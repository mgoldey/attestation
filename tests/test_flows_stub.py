"""The stub is what the offline flows talk to. It must satisfy every schema
the real code asks for, or the offline run proves nothing."""

import importlib.util
import json
import urllib.request
from pathlib import Path

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _stub():
    spec = importlib.util.spec_from_file_location("flows_stub", FLOWS / "stub_openai.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _common():
    spec = importlib.util.spec_from_file_location("flows_common", FLOWS / "_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_embeddings_are_deterministic_and_dims_wide():
    s = _stub()
    a = s.embed_text("photoredox catalysis at the bench", 256)
    b = s.embed_text("photoredox catalysis at the bench", 256)
    c = s.embed_text("distributed training on 64 GPUs", 256)
    assert a == b and len(a) == 256
    assert a != c


def test_similar_texts_embed_closer_than_unrelated_ones():
    import numpy as np

    s = _stub()
    chem1 = np.array(s.embed_text("photoredox C-H alkylation of arenes", 256))
    chem2 = np.array(s.embed_text("photoredox alkylation of unactivated arenes", 256))
    ml = np.array(s.embed_text("gradient all-reduce sharding GPUs", 256))
    assert chem1 @ chem2 > chem1 @ ml


def test_answer_satisfies_reaction_explanation_and_tag_schemas():
    from attestation.explain import Explanation
    from attestation.features import ItemTags
    from attestation.simulate import Reaction

    s = _stub()
    messages = [
        {"role": "system", "content": "You are bench-chemist. Interests: photoredox catalysis."},
        {
            "role": "user",
            "content": "Title: Photoredox alkylation\nSummary: arenes and alkyl bromides",
        },
    ]
    Reaction.model_validate(s.answer(Reaction.model_json_schema(), messages))
    Explanation.model_validate(s.answer(Explanation.model_json_schema(), messages))
    ItemTags.model_validate(s.answer(ItemTags.model_json_schema(), messages))


def test_reaction_verdict_follows_keyword_overlap():
    from attestation.simulate import Reaction

    s = _stub()
    schema = Reaction.model_json_schema()
    on_topic = [
        {"role": "system", "content": "persona interests: photoredox catalysis cross-coupling"},
        {"role": "user", "content": "photoredox cross-coupling of aryl halides"},
    ]
    off_topic = [
        {"role": "system", "content": "persona interests: photoredox catalysis cross-coupling"},
        {"role": "user", "content": "coral reef fish migration"},
    ]
    assert s.answer(schema, on_topic)["verdict"] is True
    assert s.answer(schema, off_topic)["verdict"] is False


def test_server_speaks_both_endpoints():
    s = _stub()
    server, base = s.start(dims=8)
    try:
        req = urllib.request.Request(
            base + "/embeddings",
            data=json.dumps({"model": "stub", "input": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.load(urllib.request.urlopen(req))
        assert len(body["data"][0]["embedding"]) == 8

        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(
                {
                    "model": "stub",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "r",
                            "schema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                    },
                    "reasoning_effort": "none",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.load(urllib.request.urlopen(req))
        assert json.loads(body["choices"][0]["message"]["content"])["text"]
    finally:
        server.shutdown()


def test_the_stub_tags_every_corpus_item_without_failing_ItemTags():
    """`ItemTags` is the schema `run_tagging` validates against, and the stub
    answers it from words taken out of the item -- so a corpus item holding a
    long compound token could produce tags that fail TAG_PATTERN
    (`^[a-z0-9][a-z0-9-]{0,31}$`) and be dropped, or, if every tag failed, fail
    the item outright and leave the offline flow's graph empty.

    Whether that happens is a property of THIS corpus, so it is asserted over
    the real corpus rather than a hand-written string: mcp_e2e.py's kg.* calls
    are only meaningful if tagging populated the graph.
    """
    from attestation.features import ItemTags

    s, c = _stub(), _common()
    entries = c.corpus_entries()
    assert len(entries) >= 40, "the corpus shrank; this test is about its real shape"
    schema = ItemTags.model_json_schema()
    for entry in entries:
        messages = [
            {"role": "system", "content": "Tag this item with two to four topic tags."},
            {
                "role": "user",
                "content": f"Title: {entry['title']}\nSummary: {entry['summary']}",
            },
        ]
        tagged = ItemTags.model_validate(s.answer(schema, messages))
        # Not merely valid: a validator that dropped every tag would still
        # validate (ItemTags keeps the item and returns []), and an item with
        # no tags contributes nothing to the graph.
        assert tagged.tags, f"no usable tags for {entry['title']!r}"
