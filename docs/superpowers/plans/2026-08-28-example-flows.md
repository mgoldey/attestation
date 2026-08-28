# Example Flows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scripts under `examples/flows/` that drive every agent helper end to end — an LLM-driven persona eval printing precision/recall/AUC, a stdio MCP client calling all 46 tools across the four surfaces, and a sub-30-second MLflow training family the ledger scans — runnable offline against a stub model server (CI) and live against Ollama (the recorded numbers).

**Architecture:** One hand-written labelled Atom corpus is ingested through the real `run_ingest` path (feedparser reads a local file); an OpenAI-compatible stub (`stdlib http.server`) stands in for Ollama when `--offline`; each flow is a standalone script with a `main(argv) -> int` and a `--json` report, and `run_all.py` sequences them. Committed `mlruns/` output makes the ledger's MLflow reader testable without mlflow installed.

**Tech Stack:** Python ≥3.12, `mcp` (already a dependency; `mcp.client.stdio`), scikit-learn (already a dependency), `mlflow-skinny` (new `examples` dependency group), stdlib `http.server`, feedparser.

**Spec:** `docs/superpowers/specs/2026-08-28-example-flows-design.md`

## Global Constraints

- Nothing under `src/` may import mlflow (`tests/test_tag_prompt.py` asserts this for dspy; extend the same assertion to mlflow).
- No new `# noqa: BLE001` sites (`tests/test_architecture.py` pins the count at 7).
- Line length 100; ruff lint `E,F,W,I,BLE,RUF100`; `*.md` is excluded from ruff.
- Gates before every commit: `uv run --frozen pre-commit run --all-files` and read the per-hook `Passed/Failed` lines, not the tail.
- The offline run must need no network and no Ollama; the live run needs Ollama at `LLM_BASE_URL` with the models in `.env`.
- Training must finish in under 30 s, enforced by `run_all.py`, not asserted in prose.
- Every printed number that depends on a model names the mode (`offline`/`live`) and the model.
- Scripts under `examples/flows/` are not pytest tests and are not importable as a package: tests load them with `importlib.util.spec_from_file_location`.
- Commit messages follow the repo's style (a sentence saying what changed and why); end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: The labelled fixture and its loader

**Files:**
- Create: `examples/flows/corpus/labelled.xml`
- Create: `examples/flows/corpus/labels.json`
- Create: `examples/flows/corpus/personas.toml`
- Create: `examples/flows/_common.py`
- Test: `tests/test_flows_fixture.py`

**Interfaces:**
- Produces: `_common.CORPUS_DIR: Path`, `_common.load_personas() -> dict[str, str]` (name → interests), `_common.load_labels() -> dict[str, dict[str, bool]]` (guid → persona → label), `_common.corpus_entries() -> list[dict]` (each `{"guid", "title", "summary", "link"}` parsed with feedparser), `_common.write_feeds_toml(directory: Path) -> Path` (writes a `feeds.toml` whose single feed `url` is the absolute path of `labelled.xml`; returns the path), `_common.load_script(name: str)` (imports `examples/flows/<name>.py` by path and returns the module — used by tests and by `run_all.py`).

- [ ] **Step 1: Write the failing fixture test**

```python
# tests/test_flows_fixture.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen pytest tests/test_flows_fixture.py -q`
Expected: FAIL — `FileNotFoundError` for `_common.py`.

- [ ] **Step 3: Write `_common.py`**

```python
# examples/flows/_common.py
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
```

- [ ] **Step 4: Write `personas.toml`**

```toml
# examples/flows/corpus/personas.toml
# Two readers with disjoint interests. The strings are what a person would
# type into the onboarding form; labels.json says which items each finds useful.

[[personas]]
name = "bench-chemist"
interests = """Synthetic organic chemistry: total synthesis, C-H activation, photoredox and
electrochemical catalysis, cross-coupling, flow chemistry, and practical
methodology I can run at the bench. Reaction optimisation and mechanism."""

[[personas]]
name = "ml-engineer"
interests = """Training and serving machine learning systems: distributed training, mixed
precision, quantisation, inference latency, GPU kernels, data loaders,
evaluation harnesses, and reproducibility of model training runs."""
```

- [ ] **Step 5: Write `labelled.xml` and `labels.json`**

Write an Atom feed with **at least 40** entries in four groups, each entry with a stable `<id>` of the form `urn:flows:<group>-<nn>`, a `<title>`, a `<link href="https://example.org/flows/<group>-<nn>"/>`, an `<updated>` in August 2026, and a `<summary>` of 2–4 sentences (≥80 characters). Hand-written, no scraped text.

| group | count | bench-chemist | ml-engineer |
|---|---|---|---|
| `chem` — synthesis methodology, catalysis, flow chemistry, mechanism | 12 | true | false |
| `mlsys` — distributed training, quantisation, inference, data loaders, eval harnesses | 12 | false | true |
| `other` — marine ecology, astronomy, macroeconomics, linguistics | 8 | false | false |
| `bait` — generic "AI for science", "a new dataset", "we release a tool" with no domain content | 8 | false | false |

Skeleton (first entry of each group shown; write the rest to the same shape):

```xml
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>flows fixture</title>
  <id>urn:flows:feed</id>
  <updated>2026-08-20T00:00:00Z</updated>

  <entry>
    <id>urn:flows:chem-01</id>
    <title>Photoredox C–H alkylation of unactivated arenes with alkyl bromides</title>
    <link href="https://example.org/flows/chem-01"/>
    <updated>2026-08-01T09:00:00Z</updated>
    <summary>A visible-light photoredox protocol couples unactivated arenes with primary and secondary alkyl bromides at room temperature. The reaction tolerates esters, free alcohols and basic heterocycles, runs on gram scale in a flow reactor, and the mechanism is supported by radical-clock and Stern–Volmer studies.</summary>
  </entry>

  <entry>
    <id>urn:flows:mlsys-01</id>
    <title>Overlapping gradient all-reduce with backward computation in PyTorch FSDP</title>
    <link href="https://example.org/flows/mlsys-01"/>
    <updated>2026-08-01T10:00:00Z</updated>
    <summary>We measure how much of the backward pass can hide collective communication when sharding a 7B parameter model across 64 GPUs. Bucketing gradients by layer group recovers 18% step time; mixed-precision reduce in bf16 recovers a further 6% with no loss in evaluation accuracy.</summary>
  </entry>

  <entry>
    <id>urn:flows:other-01</id>
    <title>Seasonal migration of coral reef fish tracked with acoustic telemetry</title>
    <link href="https://example.org/flows/other-01"/>
    <updated>2026-08-02T08:00:00Z</updated>
    <summary>Three years of acoustic tagging on a Pacific atoll show that parrotfish move between lagoon and fore-reef habitats on a lunar cycle. The pattern is strongest in adults and weakens where fishing pressure is high, which has consequences for the placement of no-take zones.</summary>
  </entry>

  <entry>
    <id>urn:flows:bait-01</id>
    <title>AI is transforming science: a perspective</title>
    <link href="https://example.org/flows/bait-01"/>
    <updated>2026-08-02T11:00:00Z</updated>
    <summary>Machine learning is reshaping how research is done across every discipline. This perspective surveys the opportunities and challenges that artificial intelligence brings to the scientific enterprise and calls for interdisciplinary collaboration to realise its promise.</summary>
  </entry>
</feed>
```

`labels.json` maps each guid to both personas, exactly as the table says:

```json
{
  "urn:flows:chem-01": {"bench-chemist": true, "ml-engineer": false},
  "urn:flows:mlsys-01": {"bench-chemist": false, "ml-engineer": true},
  "urn:flows:other-01": {"bench-chemist": false, "ml-engineer": false},
  "urn:flows:bait-01": {"bench-chemist": false, "ml-engineer": false}
}
```

- [ ] **Step 6: Run the fixture tests**

Run: `uv run --frozen pytest tests/test_flows_fixture.py -q`
Expected: 5 passed.

- [ ] **Step 7: Gates and commit**

```bash
uv run --frozen pre-commit run --all-files 2>&1 | grep -E '^\S.*\.(Passed|Failed)$'
git add examples/flows tests/test_flows_fixture.py
git commit -m "A labelled corpus for the example flows: forty hand-written items, two personas, every item labelled for both"
```

---

### Task 2: The stub model server

**Files:**
- Create: `examples/flows/stub_openai.py`
- Test: `tests/test_flows_stub.py`

**Interfaces:**
- Produces: `stub_openai.start(dims: int | None = None) -> tuple[http.server.ThreadingHTTPServer, str]` (server + base URL ending in `/v1`, served on a daemon thread; caller calls `server.shutdown()`), `stub_openai.embed_text(text: str, dims: int) -> list[float]` (pure, deterministic), `stub_openai.answer(schema: dict, messages: list[dict]) -> dict` (pure: an object satisfying `schema`), `stub_openai.MODEL = "stub"`. Running the file as a script serves until interrupted and prints the URL.

- [ ] **Step 1: Write the failing stub tests**

```python
# tests/test_flows_stub.py
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
        {"role": "user", "content": "Title: Photoredox alkylation\nSummary: arenes and alkyl bromides"},
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_flows_stub.py -q`
Expected: FAIL — no `stub_openai.py`.

- [ ] **Step 3: Write `stub_openai.py`**

```python
# examples/flows/stub_openai.py
"""An OpenAI-compatible model server that knows nothing.

Serves the two endpoints attestation.llm calls -- POST /v1/embeddings and
POST /v1/chat/completions -- deterministically and instantly, so every
flow can run in CI with no Ollama and no network. Embeddings are a hashed
bag of words (texts sharing vocabulary land near each other, which is what
lets the ranker and the click classifier exercise their real code paths);
chat answers are built from the request's JSON schema and a keyword
overlap between the system and user messages.

It is deliberately dumb. Every number produced against it is about the
stub, and the flows say so on every line that prints one. See
docs/measurement-lessons.md section 3.

    uv run python examples/flows/stub_openai.py      # serve until Ctrl-C
"""

from __future__ import annotations

import hashlib
import http.server
import json
import math
import os
import re
import sys
import threading

MODEL = "stub"
_WORD = re.compile(r"[a-z][a-z0-9\-]+")
_STOP = frozenset(
    "the a an and of to in on for with at by from as is are was be this that we it its".split()
)


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def embed_text(text: str, dims: int) -> list[float]:
    """Hashed bag of words, L2-normalised. Equal texts give equal vectors."""
    vec = [0.0] * dims
    for w in _words(text):
        h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big")
        vec[h % dims] += 1.0 if (h >> 63) else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _placeholder(prop: dict, name: str):
    kind = prop.get("type")
    if "enum" in prop:
        return prop["enum"][0]
    if kind == "string":
        return f"stub {name}"
    if kind == "integer":
        return int(prop.get("minimum", 1))
    if kind == "number":
        return float(prop.get("minimum", 0))
    if kind == "boolean":
        return False
    if kind == "array":
        return [_placeholder(prop.get("items", {"type": "string"}), name)]
    if kind == "object":
        return {}
    return None


def answer(schema: dict, messages: list[dict]) -> dict:
    """An object satisfying `schema`, decided by keyword overlap.

    System/prior text stands for "what the reader wants"; the last user
    message is the item. Overlap of two or more content words is a yes.
    """
    prior = " ".join(m.get("content", "") for m in messages[:-1])
    item = messages[-1].get("content", "") if messages else ""
    overlap = set(_words(prior)) & set(_words(item))
    yes = len(overlap) >= 2
    props = schema.get("properties", {})
    out = {name: _placeholder(prop, name) for name, prop in props.items()}

    if {"reasoning", "verdict", "confidence"} <= set(props):  # simulate.Reaction
        out["reasoning"] = (
            f"shares {len(overlap)} terms with the persona: {', '.join(sorted(overlap))}"
            if overlap
            else "nothing in this item matches the persona's interests"
        )
        out["verdict"] = yes
        out["confidence"] = 5 if len(overlap) >= 3 or not overlap else 3
    if "text" in props and len(props) == 1:  # explain.Explanation
        out["text"] = (
            f"Matches your interests on: {', '.join(sorted(overlap))}."
            if overlap
            else "Nothing here matches your stated interests."
        )
    if "tags" in props and "content_type" in props:  # features.ItemTags
        words = [w for w in _words(item) if len(w) > 4]
        out["tags"] = list(dict.fromkeys(words))[:3] or ["untagged"]
        out["content_type"] = "paper"
    return out


class _Handler(http.server.BaseHTTPRequestHandler):
    dims = 256

    def log_message(self, *_args):  # quiet
        pass

    def _send(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # `attest install --check` probes the native root
        self._send({"status": "stub"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/embeddings"):
            inputs = req.get("input", "")
            texts = inputs if isinstance(inputs, list) else [inputs]
            self._send(
                {
                    "object": "list",
                    "model": MODEL,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": embed_text(t, self.dims)}
                        for i, t in enumerate(texts)
                    ],
                }
            )
            return
        if self.path.endswith("/chat/completions"):
            schema = req.get("response_format", {}).get("json_schema", {}).get("schema", {})
            content = json.dumps(answer(schema, req.get("messages", [])))
            self._send(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [
                        {"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}
                    ],
                }
            )
            return
        self._send({"error": f"unknown path {self.path}"}, status=404)


def start(dims: int | None = None) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Serve on 127.0.0.1:<free port> in a daemon thread. Returns (server, base_url)."""
    handler = type("Handler", (_Handler,), {"dims": dims or int(os.environ.get("EMBED_DIMS", "256"))})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def main() -> int:
    server, base = start()
    print(base, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the stub tests**

Run: `uv run --frozen pytest tests/test_flows_stub.py -q`
Expected: 5 passed. If `test_answer_satisfies_...` fails on `ItemTags` because a tag violates `TAG_PATTERN`, read `features.TAG_PATTERN` and filter `_words` by it in the tagging branch — the fix goes in the stub, never in `features.py`.

- [ ] **Step 5: Prove the stub works with the real client**

Run:
```bash
uv run --frozen python -c "
import sys; sys.path.insert(0, 'examples/flows')
import stub_openai
from attestation.llm import ChatClient, EmbeddingClient
from attestation.simulate import Reaction
srv, base = stub_openai.start()
print(len(EmbeddingClient(base_url=base, model='stub').embed('hello world')))
print(ChatClient(base_url=base, model='stub').chat_json(
    [{'role':'system','content':'interests: photoredox catalysis'},
     {'role':'user','content':'photoredox catalysis of arenes'}], Reaction.model_json_schema()))
srv.shutdown()"
```
Expected: `256` then a dict with `verdict: True`.

- [ ] **Step 6: Gates and commit**

```bash
uv run --frozen pre-commit run --all-files 2>&1 | grep -E '^\S.*\.(Passed|Failed)$'
git add examples/flows/stub_openai.py tests/test_flows_stub.py
git commit -m "A stub model server so the flows run with no Ollama: hashed embeddings, schema-shaped answers, and it says so"
```

---

### Task 3: `persona_eval.py` — precision, recall, AUC from LLM-driven reactions

**Files:**
- Create: `examples/flows/persona_eval.py`
- Test: `tests/test_flows_scoring.py`

**Interfaces:**
- Consumes: `_common.*` (Task 1), `stub_openai.start` (Task 2), `attestation.ingest.run_ingest(conn, embedder, feeds_path)`, `attestation.embed.Embedder(EmbeddingClient(base_url, model))`, `attestation.rank.create_user(conn, name, interests) -> int`, `attestation.simulate.simulate_feedback(conn, chat_fn, user_name, items) -> {"counts", "reactions"}`, `attestation.rank.evaluate_user(conn, user_id) -> dict | None`, `attestation.rank.rank_items(conn, embedder, user_id, since_days=None, exclude_clicked=False) -> list[RankedItem]` (each has `.item_id` and a rank position), `attestation.db.get_db(path)`.
- Produces: `persona_eval.score_verdicts(reactions: list[dict], labels: dict[int, bool]) -> dict` with keys `tp, fp, fn, tn, n_scored, n_unsure, precision, recall, auc, confidence_histogram` (`auc` is `None` when the signed score does not vary); `persona_eval.rank_auc(order: list[int], labels: dict[int, bool]) -> float | None`; `persona_eval.prepare_db(db_path: Path, base_url: str, chat_model: str, embed_model: str) -> dict` (ingests the corpus, creates both personas, returns `{"items": n, "users": {name: id}, "guid_to_id": {...}}`) — Task 4 reuses this; `persona_eval.main(argv) -> int` with `--offline`, `--db PATH`, `--json PATH`.

- [ ] **Step 1: Write the failing scoring tests**

```python
# tests/test_flows_scoring.py
"""The arithmetic behind the printed precision/recall/AUC, on a
hand-computed confusion matrix. Model-free."""

import importlib.util
from pathlib import Path

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _eval():
    spec = importlib.util.spec_from_file_location("flows_eval", FLOWS / "persona_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reaction(item_id, verdict, confidence=5):
    return {"item_id": item_id, "verdict": verdict, "confidence": confidence}


def test_precision_recall_on_a_hand_computed_matrix():
    m = _eval()
    labels = {1: True, 2: True, 3: True, 4: False, 5: False, 6: False}
    reactions = [
        _reaction(1, True), _reaction(2, True), _reaction(3, False),   # tp tp fn
        _reaction(4, True), _reaction(5, False), _reaction(6, False),  # fp tn tn
    ]
    out = m.score_verdicts(reactions, labels)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (2, 1, 1, 2)
    assert out["precision"] == 2 / 3
    assert out["recall"] == 2 / 3
    assert out["n_scored"] == 6 and out["n_unsure"] == 0


def test_unsure_items_are_reported_not_dropped_silently():
    m = _eval()
    labels = {1: True, 2: False, 3: True}
    out = m.score_verdicts([_reaction(1, True), _reaction(2, False)], labels)
    assert out["n_scored"] == 2
    assert out["n_unsure"] == 1  # item 3 never got a verdict


def test_auc_is_none_when_confidence_never_varies():
    m = _eval()
    labels = {1: True, 2: False}
    out = m.score_verdicts([_reaction(1, True, 5), _reaction(2, True, 5)], labels)
    assert out["auc"] is None
    assert out["confidence_histogram"] == {5: 2}


def test_auc_rewards_confident_correct_verdicts():
    m = _eval()
    labels = {1: True, 2: True, 3: False, 4: False}
    perfect = [_reaction(1, True, 5), _reaction(2, True, 4), _reaction(3, False, 4), _reaction(4, False, 5)]
    assert m.score_verdicts(perfect, labels)["auc"] == 1.0


def test_rank_auc_over_an_ordering():
    m = _eval()
    labels = {10: True, 11: True, 12: False, 13: False}
    assert m.rank_auc([10, 11, 12, 13], labels) == 1.0
    assert m.rank_auc([12, 13, 10, 11], labels) == 0.0
    assert m.rank_auc([10, 11], {10: True, 11: True}) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_flows_scoring.py -q`
Expected: FAIL — no `persona_eval.py`.

- [ ] **Step 3: Write `persona_eval.py`**

```python
# examples/flows/persona_eval.py
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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402


def score_verdicts(reactions: list[dict], labels: dict[int, bool]) -> dict:
    """Confusion matrix, precision, recall, and AUC of a signed confidence.

    Items in `labels` with no reaction were skipped as unsure by the model;
    they are counted in n_unsure and excluded from the matrix, never from
    the report.
    """
    by_item = {r["item_id"]: r for r in reactions if r["item_id"] in labels}
    tp = fp = fn = tn = 0
    scores, truth = [], []
    for item_id, label in labels.items():
        r = by_item.get(item_id)
        if r is None:
            continue
        verdict = bool(r["verdict"])
        tp += verdict and label
        fp += verdict and not label
        fn += (not verdict) and label
        tn += (not verdict) and not label
        scores.append(r["confidence"] if verdict else -r["confidence"])
        truth.append(label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    auc = None
    if len(set(scores)) > 1 and len(set(truth)) > 1:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(truth, scores))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_scored": len(scores),
        "n_unsure": len(labels) - len(scores),
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "confidence_histogram": dict(sorted(Counter(r["confidence"] for r in by_item.values()).items())),
    }


def rank_auc(order: list[int], labels: dict[int, bool]) -> float | None:
    """AUC of a ranking: earlier = higher score. None on a single class."""
    truth = [labels[i] for i in order if i in labels]
    if len(set(truth)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    scores = [len(order) - pos for pos, i in enumerate(order) if i in labels]
    return float(roc_auc_score(truth, scores))


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
    users = {name: create_user(conn, name, interests) for name, interests in _common.load_personas().items()}
    guid_to_id = {r["guid"]: r["id"] for r in conn.execute("SELECT id, guid FROM items")}
    conn.close()
    return {"items": stats["added"], "users": users, "guid_to_id": guid_to_id}


def evaluate_persona(db_path: Path, name: str, user_id: int, guid_to_id: dict, base_url: str, chat_model: str, embed_model: str) -> dict:
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
```

- [ ] **Step 4: Run the scoring tests**

Run: `uv run --frozen pytest tests/test_flows_scoring.py -q`
Expected: 5 passed.

- [ ] **Step 5: Run the flow offline, then live**

Run: `uv run --frozen python examples/flows/persona_eval.py --offline`
Expected: two persona blocks with a confusion matrix each; `mode=offline chat=stub`. If `simulate_feedback` writes zero rows, print `sim["counts"]` and check `failed` — a schema the stub does not satisfy shows up here first.

Run: `uv run --frozen python examples/flows/persona_eval.py` (live; ~40 items × 2 personas on gemma4:e2b ≈ 4–6 min; check `nvidia-smi` first).
Expected: the same blocks with `mode=live chat=gemma4:e2b-it-q4_K_M`. Record the output in the scratchpad for `RESULTS.md` (Task 6).

- [ ] **Step 6: Gates and commit**

```bash
uv run --frozen pre-commit run --all-files 2>&1 | grep -E '^\S.*\.(Passed|Failed)$'
git add examples/flows/persona_eval.py tests/test_flows_scoring.py
git commit -m "persona_eval: the model reacts to forty labelled items as each persona, and precision, recall and AUC are printed with their confusion matrix"
```

---

### Task 4: `mcp_e2e.py` — every tool, over stdio, on every surface

**Files:**
- Create: `examples/flows/mcp_e2e.py`
- Test: `tests/test_flows_mcp_plan.py`

**Interfaces:**
- Consumes: `persona_eval.prepare_db` (Task 3), `stub_openai.start` (Task 2), `attestation.mcp.AGENT_SURFACES` (`{name: Surface(prefixes=frozenset[str], ...)}`), `mcp.client.stdio.stdio_client(StdioServerParameters(command, args, env))`, `mcp.ClientSession(read, write)` with `await session.initialize()`, `await session.list_tools()` (`.tools`, each `.name`), `await session.call_tool(name, arguments)` (`.content[0].text` is the JSON envelope, `.isError`).
- Produces: `mcp_e2e.CALLS: list[Call]` where `Call = (tool: str, arguments: dict, expect: Literal["ok", "refused"])` — the scripted argument set, in order; `mcp_e2e.surface_for(tool: str) -> set[str]` (which spawns list it); `mcp_e2e.check_envelope(payload: dict, expect: str) -> str | None` (None when as expected, else a reason); `mcp_e2e.main(argv) -> int` with `--offline`, `--json PATH`, `--surface NAME` (one spawn only).

- [ ] **Step 1: Write the failing plan test**

```python
# tests/test_flows_mcp_plan.py
"""The MCP flow's scripted calls must cover every registered tool and only
registered tools. Model-free: the plan is data; the server is checked in
the flow itself."""

import asyncio
import importlib.util
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from attestation.mcp import register_all

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _flow():
    spec = importlib.util.spec_from_file_location("flows_mcp", FLOWS / "mcp_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _served(monkeypatch, surface=None):
    monkeypatch.setenv("ATTEST_EXPAND", "1")
    if surface:
        monkeypatch.setenv("ATTEST_TOOLS", surface)
    else:
        monkeypatch.delenv("ATTEST_TOOLS", raising=False)
    m = FastMCP("plan")
    register_all(m)
    return {t.name for t in asyncio.run(m.list_tools())}


def test_every_served_tool_is_called_at_least_once(monkeypatch):
    called = {c[0] for c in _flow().CALLS}
    served = _served(monkeypatch)
    for surface in ("feed", "provenance", "knowledge", "symbolic"):
        served |= _served(monkeypatch, surface)
    assert served - called == set(), f"never called: {sorted(served - called)}"
    assert called - served == set(), f"called but not served: {sorted(called - served)}"


def test_every_ask_router_gets_a_question_that_must_disambiguate():
    calls = _flow().CALLS
    for router in ("feed.ask", "runs.ask", "kg.ask", "sym.ask"):
        assert any(c[0] == router and c[2] == "options" for c in calls), router


def test_destructive_tools_are_called_on_entities_the_flow_created():
    calls = _flow().CALLS
    names = [c[0] for c in calls]
    assert names.index("feed.persona_create") < names.index("feed.persona_delete")
    assert names.index("feed.source_add") < names.index("feed.source_remove")
    for tool in ("feed.persona_delete", "feed.persona_reset", "feed.source_remove", "runs.scan"):
        assert any(c[0] == tool and c[1].get("confirm") is True for c in calls), tool


def test_envelope_check_accepts_the_contract_and_rejects_shape_drift():
    m = _flow()
    assert m.check_envelope({"ok": True, "message": "", "items": []}, "ok") is None
    assert m.check_envelope({"ok": False, "message": "no", "items": []}, "refused") is None
    assert m.check_envelope({"ok": False, "message": "boom", "items": []}, "ok")
    assert m.check_envelope({"ok": True, "message": ""}, "refused")
    assert m.check_envelope({"message": "no ok key"}, "ok")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --frozen pytest tests/test_flows_mcp_plan.py -q`
Expected: FAIL — no `mcp_e2e.py`.

- [ ] **Step 3: Write `mcp_e2e.py`**

The scripted `CALLS` list below is the deliverable; the item ids, persona names and paths are resolved at run time by `_resolve(arguments, ctx)` replacing the sentinel strings `"$ITEM"`, `"$ITEM2"`, `"$FEED_ID"`, `"$WORKSPACE"`, `"$FINDINGS"`, `"$CORPUS_XML"`.

```python
# examples/flows/mcp_e2e.py
"""Every MCP tool, called over stdio, on every agent surface.

Spawns `attest-mcp` five times -- once per ATTEST_TOOLS surface with
ATTEST_EXPAND=1, once unrestricted -- with the `mcp` package's stdio client,
lists the tools, and calls each one with a scripted argument set in the
order a person would. This is the path every agent takes and the one
nothing in tests/ exercises: the entry point, the env the server reads at
import, the stdio framing, the schema FastMCP emits, the stale-process
problem.

Prints a matrix of surface x tool x ok/refused/FAILED. Exit 1 if any call
did not do what CALLS says it should.

    uv run python examples/flows/mcp_e2e.py --offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

# (tool, arguments, expectation). Expectation: "ok", "refused", or "options"
# (an .ask router that must return options rather than a default).
CALLS: list[tuple[str, dict, str]] = [
    # --- feed: personas first, since everything else needs one
    ("feed.personas", {}, "ok"),
    ("feed.persona_create", {"name": "flow-temp", "interests": "coral reef ecology and fish telemetry"}, "ok"),
    ("feed.persona_update", {"name": "flow-temp", "interests": "coral reef ecology, fisheries, marine protected areas"}, "ok"),
    ("feed.persona_suggest_interests", {"limit": 5}, "ok"),
    ("feed.persona_status", {"user": "bench-chemist"}, "ok"),
    ("feed.list", {"user": "bench-chemist", "limit": 4}, "ok"),
    ("feed.list", {"user": "bench-chemist", "limit": 13}, "ok"),
    ("feed.search", {"user": "ml-engineer", "query": "quantisation inference latency"}, "ok"),
    ("feed.read", {"user": "bench-chemist", "item_id": "$ITEM"}, "ok"),
    ("feed.explain", {"user": "bench-chemist", "item_id": "$ITEM"}, "ok"),
    ("feed.rate", {"user": "bench-chemist", "item_id": "$ITEM", "useful": True}, "ok"),
    ("feed.rate", {"user": "bench-chemist", "item_id": "$ITEM2", "useful": False}, "ok"),
    ("feed.harvest_engagement", {"user": "bench-chemist"}, "ok"),
    ("feed.simulate_ratings", {"user": "flow-temp", "limit": 3, "confirm": True}, "ok"),
    ("feed.digest", {"user": "ml-engineer", "days": 3650}, "ok"),
    ("feed.ask", {"user": "bench-chemist", "question": "what is new for me this week?"}, "ok"),
    ("feed.ask", {"user": "bench-chemist", "question": "find papers on flow chemistry"}, "ok"),
    ("feed.ask", {"user": "bench-chemist", "question": "hmm"}, "options"),
    ("feed.persona_reset", {"name": "flow-temp", "confirm": True}, "ok"),
    ("feed.persona_delete", {"name": "flow-temp", "confirm": True}, "ok"),
    ("feed.persona_delete", {"name": "never-existed", "confirm": True}, "refused"),
    # --- feed subscriptions
    ("feed.sources", {}, "ok"),
    ("feed.source_preview", {"url": "$CORPUS_XML", "limit": 3}, "ok"),
    ("feed.source_add", {"url": "$CORPUS_XML", "title": "flows fixture again"}, "ok"),
    ("feed.source_suggest", {"user": "ml-engineer", "limit": 3}, "ok"),
    ("feed.source_remove", {"feed_id": "$FEED_ID", "confirm": True}, "ok"),
    # --- provenance
    ("runs.scan", {"root": "$WORKSPACE", "confirm": True}, "ok"),
    ("runs.list", {"limit": 10}, "ok"),
    ("runs.list", {"project": "speech-distill", "family": "kdsweep"}, "ok"),
    ("runs.compare", {"family": "kdsweep", "metric": "wer"}, "ok"),
    ("runs.detail", {"project": "speech-distill", "name": "kdsweep_t4"}, "ok"),
    ("runs.claims_coverage", {"path": "$FINDINGS"}, "ok"),
    ("runs.claims_check", {"path": "$FINDINGS"}, "ok"),
    ("runs.ask", {"question": "which arm of kdsweep won?"}, "ok"),
    ("runs.ask", {"question": "check the claims in $FINDINGS"}, "ok"),
    ("runs.ask", {"question": "hmm"}, "options"),
    # --- knowledge
    ("kg.concepts", {"limit": 10}, "ok"),
    ("kg.central", {"metric": "degree", "limit": 5}, "ok"),
    ("kg.communities", {"min_size": 2}, "ok"),
    ("kg.neighbors", {"node": "$CONCEPT"}, "ok"),
    ("kg.path", {"source": "$CONCEPT", "target": "$CONCEPT2"}, "ok"),
    ("kg.ask", {"question": "what concepts is my reading centred on?"}, "ok"),
    ("kg.ask", {"question": "hmm"}, "options"),
    # --- citations (local sources only; the flow has no .bib, so lookups refuse cleanly)
    ("cite.sources", {}, "ok"),
    ("cite.lookup", {"key": "vaswani2017attention"}, "refused"),
    ("cite.search", {"query": "attention is all you need"}, "ok"),
    ("cite.check", {"path": "$FINDINGS"}, "ok"),
    # --- symbolic (no database)
    ("sym.simplify", {"expr": "(x**2 - 1)/(x - 1)"}, "ok"),
    ("sym.solve", {"expr": "x**2 - 4", "symbol": "x"}, "ok"),
    ("sym.solve", {"expr": "x*y - 1"}, "refused"),
    ("sym.differentiate", {"expr": "x**3", "symbol": "x"}, "ok"),
    ("sym.integrate", {"expr": "x**2", "symbol": "x", "bounds": [0, 1]}, "ok"),
    ("sym.derivation", {"expr": "x**2", "operation": "integrate"}, "ok"),
    ("sym.verify", {"lhs": "sin(x)**2 + cos(x)**2", "rhs": "1"}, "ok"),
    ("sym.evaluate", {"expr": "2*pi", "subs": None}, "ok"),
    ("sym.simplify", {"expr": "(x+1)**200000", "timeout": 3}, "refused"),
    ("sym.ask", {"expr": "x**2 - 9", "question": "solve"}, "ok"),
    ("sym.ask", {"expr": "x**2", "question": "hmm"}, "options"),
    # --- disclosure (only under ATTEST_TOOLS)
    ("feed.tools", {}, "ok"),
    ("runs.tools", {}, "ok"),
    ("kg.tools", {}, "ok"),
    ("sym.tools", {}, "ok"),
]

SURFACES = ("feed", "provenance", "knowledge", "symbolic", None)


def surface_for(tool: str) -> set[str]:
    from attestation.mcp import AGENT_SURFACES

    out = {"full"} if not tool.endswith(".tools") else set()
    for name, surface in AGENT_SURFACES.items():
        if any(tool == p or tool.startswith(p + ".") for p in surface.prefixes):
            out.add(name)
        if tool.endswith(".tools") and tool.split(".")[0] in {p.split(".")[0] for p in surface.prefixes}:
            out.add(name)
    return out


def check_envelope(payload: dict, expect: str) -> str | None:
    if "ok" not in payload or "message" not in payload:
        return "not an envelope: missing ok/message"
    if expect == "options":
        if payload.get("options"):
            return None
        return f"router chose {payload.get('tool_used')!r} instead of asking"
    if expect == "ok" and not payload["ok"]:
        return f"refused: {payload['message']}"
    if expect == "refused" and payload["ok"]:
        return "succeeded but should have refused"
    return None


def _resolve(arguments: dict, ctx: dict) -> dict:
    def sub(v):
        if isinstance(v, str) and v.startswith("$"):
            return ctx[v[1:]]
        if isinstance(v, str) and "$FINDINGS" in v:
            return v.replace("$FINDINGS", ctx["FINDINGS"])
        return v

    return {k: sub(v) for k, v in arguments.items()}


async def run_surface(surface: str | None, env: dict, ctx: dict) -> list[dict]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    spawn_env = {**os.environ, **env, "ATTEST_EXPAND": "1"}
    if surface:
        spawn_env["ATTEST_TOOLS"] = surface
    else:
        spawn_env.pop("ATTEST_TOOLS", None)
    params = StdioServerParameters(
        command="uv", args=["run", "--project", str(_common.REPO_ROOT), "attest-mcp"], env=spawn_env
    )
    label = surface or "full"
    rows = []
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = {t.name for t in (await session.list_tools()).tools}
        for tool, arguments, expect in CALLS:
            if label not in surface_for(tool) or tool not in listed:
                continue
            t0 = time.perf_counter()
            try:
                result = await session.call_tool(tool, _resolve(arguments, ctx))
                payload = json.loads(result.content[0].text) if result.content else {}
                if result.isError and "ok" not in payload:
                    payload = {"ok": False, "message": result.content[0].text if result.content else "error"}
            except Exception as exc:  # noqa: BLE001 -- one bad call is one row, and the row says why
                payload = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
            problem = check_envelope(payload, expect)
            rows.append(
                {
                    "surface": label, "tool": tool, "expect": expect, "ok": payload.get("ok"),
                    "problem": problem, "message": payload.get("message", "")[:160],
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            _learn(tool, payload, ctx)
        rows.append({"surface": label, "tool": "<list_tools>", "expect": "ok", "ok": True,
                     "problem": None, "message": f"{len(listed)} tools", "seconds": 0.0})
    return rows


def _learn(tool: str, payload: dict, ctx: dict) -> None:
    """Ids the later calls need, taken from the earlier ones' answers."""
    if tool == "feed.list" and payload.get("items"):
        ids = [i["item_id"] for i in payload["items"]]
        ctx.setdefault("ITEM", ids[0])
        ctx.setdefault("ITEM2", ids[-1] if len(ids) > 1 else ids[0])
    if tool == "feed.source_add" and payload.get("ok"):
        ctx["FEED_ID"] = payload.get("feed_id") or payload.get("feed", {}).get("id")
    if tool == "kg.concepts" and payload.get("concepts"):
        names = [c["name"] if isinstance(c, dict) else c for c in payload["concepts"]]
        ctx["CONCEPT"] = names[0]
        ctx["CONCEPT2"] = names[1] if len(names) > 1 else names[0]


def _tag_and_click(db_path: Path, base_url: str, chat_model: str, users: dict) -> None:
    """Tags for the graph, simulated clicks for the classifier: what a lived-in DB has."""
    from attestation.db import get_db
    from attestation.features import run_tagging
    from attestation.llm import ChatClient
    from attestation.simulate import simulate_feedback

    conn = get_db(db_path)
    chat = ChatClient(base_url=base_url, model=chat_model)
    run_tagging(conn, chat.chat_json)
    items = conn.execute("SELECT * FROM items ORDER BY id LIMIT 20").fetchall()
    for name in users:
        simulate_feedback(conn, chat.chat_json, name, items)
    conn.close()


def render(rows: list[dict]) -> str:
    width = max(len(r["tool"]) for r in rows)
    out = []
    for r in rows:
        status = "FAILED" if r["problem"] else ("refused" if r["ok"] is False else "ok")
        line = f"{r['surface']:<10} {r['tool']:<{width}} {status:<8} {r['seconds']:>6.2f}s"
        if r["problem"]:
            line += f"  <- {r['problem']}"
        out.append(line)
    failed = sum(1 for r in rows if r["problem"])
    out.append(f"{len(rows)} calls, {failed} failed")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--surface", choices=[s for s in SURFACES if s] + ["full"])
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

    persona_eval = _common.load_script("persona_eval")
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "flows.db"
    rows: list[dict] = []
    try:
        prepared = persona_eval.prepare_db(db_path, url, chat, embed)
        _tag_and_click(db_path, url, chat, prepared["users"])
        ctx = {
            "WORKSPACE": str(_common.REPO_ROOT / "examples" / "workspace"),
            "FINDINGS": str(_common.REPO_ROOT / "examples" / "workspace" / "speech-distill" / "FINDINGS.md"),
            "CORPUS_XML": str(_common.CORPUS_DIR / "labelled.xml"),
        }
        env = {"ATTEST_DB": str(db_path), "LLM_BASE_URL": url, "CHAT_MODEL": chat, "EMBED_MODEL": embed}
        surfaces = [args.surface if args.surface != "full" else None] if args.surface else list(SURFACES)
        for surface in surfaces:
            rows += asyncio.run(run_surface(surface, env, ctx))
    finally:
        if server:
            server.shutdown()
        tmp.cleanup()
    print(f"mcp e2e -- mode={'offline' if args.offline else 'live'} chat={chat} embed={embed}")
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps({"flow": "mcp_e2e", "mode": "offline" if args.offline else "live",
                                         "chat_model": chat, "rows": rows}, indent=2))
    return 1 if any(r["problem"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
```

Note the `# noqa: BLE001` in `run_surface`: it is in `examples/`, not `src/`, so the architecture test's count of 7 is unaffected — confirm by reading `tests/test_architecture.py` before relying on that.

- [ ] **Step 4: Run the plan tests**

Run: `uv run --frozen pytest tests/test_flows_mcp_plan.py -q`
Expected: 4 passed. If `test_every_served_tool_is_called_at_least_once` lists a tool, add a call for it to `CALLS`; if it lists one as "called but not served", the name in `CALLS` is wrong — fix the name, never the test.

- [ ] **Step 5: Run the flow offline and fix what it finds**

Run: `uv run --frozen python examples/flows/mcp_e2e.py --offline`
Expected: a matrix with 0 failed. Work through failures in this order:
1. A `$` sentinel unresolved → `_learn` did not find the id in the earlier payload: print that payload and read the real key names (`feed.list` items carry `item_id`; `feed.source_add`'s key for the new feed id; `kg.concepts`' shape) and fix `_learn`.
2. `refused` where `ok` was expected: read the message. If the flow's arguments are wrong (a persona that does not exist, a family name), fix the arguments. If the tool is wrong, that is a bug: write a failing test in the tool's own test module, fix it in `src/`, commit separately, then re-run the flow.
3. `options` expected but a tool was chosen: read `mcp/routing.py`'s rules for that router and pick a question that is genuinely ambiguous under them.

- [ ] **Step 6: Run the flow live**

Run: `uv run --frozen python examples/flows/mcp_e2e.py` (spawns real servers against Ollama; `feed.explain`, `feed.simulate_ratings` and the tagging in `_tag_and_click` make model calls — expect 3–6 minutes).
Expected: 0 failed. Save the matrix in the scratchpad for `RESULTS.md`.

- [ ] **Step 7: Gates and commit**

```bash
uv run --frozen pre-commit run --all-files 2>&1 | grep -E '^\S.*\.(Passed|Failed)$'
git add examples/flows/mcp_e2e.py tests/test_flows_mcp_plan.py
git commit -m "mcp_e2e: the real server over stdio, every tool on every surface, in the order a person would call them"
```

---

### Task 5: `training/train_mlflow.py` — a real MLflow directory the ledger reads

**Files:**
- Create: `examples/flows/training/train_mlflow.py`
- Create (committed output): `examples/flows/training/mlruns/**`, `examples/flows/training/FINDINGS.md`
- Modify: `pyproject.toml` (add `examples = ["mlflow-skinny>=2.20"]` under `[dependency-groups]`), `uv.lock` (via `uv lock`)
- Modify: `src/attestation/ledger_adapters/generic.py:469-474` (the "NEITHER READER" comment), `tests/test_tracker_adapters.py` module docstring, `docs/superpowers/specs/2026-08-22-tracker-adapters-design.md` status line
- Modify: `tests/test_tag_prompt.py` (the "src mentions no optional dependency" assertion gains `mlflow`)
- Test: `tests/test_examples.py`

**Interfaces:**
- Consumes: `attestation.ledger.scan(conn, root, project=None) -> dict`, `attestation.ledger.compare(conn, family, metric=None, project=None)` (read its signature in `ledger.py` before calling), `attestation.claims.parse_file(path)`, `attestation.claims.check_claim(conn, claim) -> Verdict` (with `.kind` a `VerdictKind`), `attestation.db.get_db`.
- Produces: `train_mlflow.FAMILY = "c_sweep"`, `train_mlflow.ARMS = (0.01, 0.1, 1.0, 10.0)`, `train_mlflow.train(tracking_dir: Path, seed: int = 0) -> list[dict]` (one dict per arm: `{"C", "accuracy", "precision", "recall", "auc", "run_id"}`), `train_mlflow.write_findings(path: Path, results: list[dict], project: str) -> None`, `train_mlflow.main(argv) -> int` with `--out DIR` (default: the script's own directory), `--json PATH`; prints elapsed seconds and returns 1 if over 30.

- [ ] **Step 1: Write the failing ledger test**

Append to `tests/test_examples.py`:

```python
TRAINING = Path(__file__).parents[1] / "examples" / "flows" / "training"


def test_the_committed_mlflow_directory_is_read_as_four_arms_of_one_family(tmp_path):
    """The tracker reader was written against documented layouts and said so
    in capitals. examples/flows/training/mlruns is the output of a real
    mlflow-skinny run (train_mlflow.py, 2026-08-28), committed so this test
    needs no mlflow: this is the first real directory it has read."""
    from attestation import ledger
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    out = ledger.scan(conn, TRAINING.parent, project="training")
    assert out["scanned"] == {"training": 4}, out
    rows = conn.execute(
        "SELECT name, family, adapter FROM runs WHERE project = 'training' ORDER BY name"
    ).fetchall()
    assert {r["family"] for r in rows} == {"c_sweep"}
    assert {r["adapter"] for r in rows} == {"mlflow"}
    metrics = conn.execute(
        "SELECT DISTINCT name FROM run_metrics rm JOIN runs r ON r.id = rm.run_id"
        " WHERE r.project = 'training'"
    ).fetchall()
    assert {m["name"] for m in metrics} >= {"accuracy", "precision", "recall", "auc", "train_loss"}
    steps = conn.execute(
        "SELECT step FROM run_metrics rm JOIN runs r ON r.id = rm.run_id"
        " WHERE r.project = 'training' AND rm.name = 'train_loss'"
    ).fetchall()
    assert all(s["step"] == 9 for s in steps), "final value of a ten-step curve, step recorded"


def test_the_mlflow_family_compares_and_its_findings_carry_one_contradiction(tmp_path):
    from attestation import claims, ledger
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, TRAINING.parent, project="training")
    result = ledger.compare(conn, "c_sweep", metric="auc", project="training")
    assert result["winner"], result
    parsed, errors = claims.parse_file(TRAINING / "FINDINGS.md")
    assert not errors
    verdicts = [claims.check_claim(conn, c) for c in parsed]
    kinds = [v.kind for v in verdicts]
    assert kinds.count(claims.VerdictKind.CONTRADICTED) == 1, kinds
    assert kinds.count(claims.VerdictKind.SUPPORTED) == 4, kinds
```

Read `ledger.compare`'s return shape and `run_metrics`'s column names (`name`, `value`, `step`?) in `db.py`/`ledger.py` first and adjust the SQL and the `result["winner"]` key to what they actually are — the test asserts the real shape, so it must use the real names.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --frozen pytest tests/test_examples.py -q -k mlflow`
Expected: FAIL — `scanned == {}` (no directory yet).

- [ ] **Step 3: Add the dependency group and extend the src guard**

In `pyproject.toml`, after the `optimize` group:

```toml
# mlflow-skinny (no UI stack) for examples/flows/training/train_mlflow.py.
# The ledger READS mlruns/ (ledger_adapters/generic.py); nothing under src/
# imports mlflow, and tests/test_tag_prompt.py asserts it.
examples = ["mlflow-skinny>=2.20"]
```

Run `uv lock` (not `--frozen`) so `uv.lock` records the group; `uv lock --check` must then pass.

In `tests/test_tag_prompt.py`, find the assertion that no file under `src/` mentions `dspy` and make it iterate over `("dspy", "mlflow")`, keeping the message per name.

- [ ] **Step 4: Write `train_mlflow.py`**

```python
# examples/flows/training/train_mlflow.py
"""Four training runs in one MLflow family, in well under thirty seconds.

LogisticRegression on scikit-learn's bundled breast-cancer set (569 rows,
no download), C in {0.01, 0.1, 1, 10}, one fixed stratified split. Each
arm logs its params and its held-out accuracy, precision, recall and AUC
to a local MLflow file store, plus a ten-step train_loss curve so the
ledger's "last line of each metric file" rule meets a real multi-line
file. run_name is the family name: that is how the ledger groups arms.

The mlruns/ this writes is committed beside it: it is the first real MLflow
directory the ledger reader has read, and tests/test_examples.py pins it
without needing mlflow installed. Regenerating changes the run ids and is
a deliberate act.

    uv run --group examples python examples/flows/training/train_mlflow.py
    uv run attest runs scan --root examples/flows --project training
    uv run attest runs compare c_sweep --metric auc
    uv run attest claims examples/flows/training/FINDINGS.md      # exit 1: one is wrong
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

FAMILY = "c_sweep"
ARMS = (0.01, 0.1, 1.0, 10.0)
HERE = Path(__file__).resolve().parent


def train(tracking_dir: Path, seed: int = 0) -> list[dict]:
    import mlflow
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment("flows")
    results = []
    for C in ARMS:
        with mlflow.start_run(run_name=FAMILY) as run:
            mlflow.log_params({"C": C, "seed": seed, "dataset": "sklearn:breast_cancer"})
            # A short curve, so the ledger reads a genuine multi-line metric file.
            sgd = SGDClassifier(loss="log_loss", alpha=1.0 / (C * len(X_tr)), random_state=seed)
            classes = np.unique(y_tr)
            for step in range(10):
                sgd.partial_fit(X_tr, y_tr, classes=classes)
                p = np.clip(sgd.predict_proba(X_tr)[:, 1], 1e-6, 1 - 1e-6)
                loss = float(-np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p)))
                mlflow.log_metric("train_loss", loss, step=step)
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            metrics = {
                "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred)),
                "recall": float(recall_score(y_te, pred)),
                "auc": float(roc_auc_score(y_te, prob)),
            }
            mlflow.log_metrics(metrics)
            results.append({"C": C, "run_id": run.info.run_id, **metrics})
    return results


def write_findings(path: Path, results: list[dict], project: str) -> None:
    """One claim per arm, plus one deliberately stale, under its own heading."""
    best = max(results, key=lambda r: r["auc"])
    lines = [
        "# c_sweep findings",
        "",
        f"Four arms of `LogisticRegression` on scikit-learn's breast-cancer set, C in {list(ARMS)}.",
        f"The best held-out AUC was C={best['C']} at {best['auc']:.4f}.",
        "",
    ]
    for r in results:
        name = f"{FAMILY}/{r['run_id'][:8]}"
        lines.append(
            f"- C={r['C']}: AUC {r['auc']:.4f}, precision {r['precision']:.4f}, recall {r['recall']:.4f}"
            f" <!-- claim: {project}/{name} metric=auc value={r['auc']:.4f} -->"
        )
    stale = best["auc"] - 0.05
    lines += [
        "",
        "### Deliberately wrong claim, for the demo",
        "",
        f"- The best arm reached AUC {stale:.4f} <!-- claim: {project}/{FAMILY}/{best['run_id'][:8]}"
        f" metric=auc value={stale:.4f} -->",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE, help="directory to hold mlruns/ and FINDINGS.md")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tracking = args.out / "mlruns"
    if tracking.exists():
        shutil.rmtree(tracking)
    results = train(tracking)
    write_findings(args.out / "FINDINGS.md", results, project=args.out.name)
    elapsed = time.perf_counter() - t0

    for r in results:
        print(f"C={r['C']:<6} accuracy {r['accuracy']:.4f}  precision {r['precision']:.4f}"
              f"  recall {r['recall']:.4f}  auc {r['auc']:.4f}")
    print(f"{len(results)} runs logged to {tracking} in {elapsed:.1f}s")
    if args.json:
        args.json.write_text(json.dumps({"flow": "train_mlflow", "seconds": elapsed, "arms": results}, indent=2))
    if elapsed > 30:
        print("FAILED: over the 30 s budget the README promises", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run it and inspect what mlflow wrote**

Run: `uv run --group examples python examples/flows/training/train_mlflow.py`
Expected: four lines of metrics and `4 runs logged ... in <n>s` with n well under 30.

Then: `find examples/flows/training/mlruns -type f | head -40` and `cat examples/flows/training/mlruns/*/*/meta.yaml | head -30`. Check: `run_name: c_sweep` present in each run's `meta.yaml` (the ledger derives the family from it — if mlflow put the name only in `tags/mlflow.runName`, teach `_mlflow_runs` to fall back to that file, with a test in `test_tracker_adapters.py`, in its own commit); `metrics/train_loss` has ten lines; `artifact_uri` may be an absolute path — acceptable, the reader ignores it. Delete `mlruns/.trash` and any `models/` directory if present; they are noise.

- [ ] **Step 6: Run the ledger tests**

Run: `uv run --frozen pytest tests/test_examples.py -q -k mlflow`
Expected: 2 passed. If `scanned` is `{"training": 0}` or the fallback picked `mlruns` up as a project of its own, read `ledger.scan`'s candidate logic (lines 209–262) and choose the `--project training` invocation that yields exactly one project; the CLI commands in the docstring must match what the test does.

- [ ] **Step 7: Retire the "never run against a real directory" caveats**

In `src/attestation/ledger_adapters/generic.py` near line 469, replace the capitals with: the MLflow reader was run against a real directory on 2026-08-28 (`examples/flows/training/mlruns`, written by mlflow-skinny 3.x via `train_mlflow.py`) and read four runs with final values and steps; the W&B reader has still not been. Make the same edit in the `tests/test_tracker_adapters.py` module docstring and add a line to the spec's **Status**. Run `uv run --frozen pytest tests/test_tracker_adapters.py -q` — unchanged behaviour, so all pass.

- [ ] **Step 8: Run the CLI sequence the docstring promises**

```bash
uv run --frozen attest runs scan --db /tmp/claude-1000/-home-matt-attestation/420b910a-3c48-4e0d-a54e-3dfe046682b7/scratchpad/flows.db --root examples/flows --project training
uv run --frozen attest runs compare --db <same> c_sweep --metric auc
uv run --frozen attest claims --db <same> examples/flows/training/FINDINGS.md; echo "exit=$?"
```
Expected: 4 runs; a winner with the seed-replication caveat; `exit=1` with exactly one contradicted claim.

- [ ] **Step 9: Gates and commit**

```bash
uv run --frozen pre-commit run --all-files 2>&1 | grep -E '^\S.*\.(Passed|Failed)$'
git add pyproject.toml uv.lock examples/flows/training src/attestation/ledger_adapters/generic.py tests/test_examples.py tests/test_tracker_adapters.py tests/test_tag_prompt.py docs/superpowers/specs/2026-08-22-tracker-adapters-design.md
git commit -m "The MLflow reader meets a real directory: four arms trained in seconds, committed, scanned, compared, and one claim contradicted"
```

---

### Task 6: `run_all.py`, the CI job, README and RESULTS

**Files:**
- Create: `examples/flows/run_all.py`
- Create: `examples/flows/README.md`
- Create: `examples/flows/RESULTS.md` (from a live run)
- Modify: `.github/workflows/ci.yml` (new `flows` job)
- Modify: `README.md` (a short "Demonstrations" pointer to `examples/flows/README.md`, placed after the "Try it in 60 seconds" section)
- Modify: `CLAUDE.md` (docs index gains `examples/flows`; one Key API Patterns line about the flows and the stub)
- Test: `tests/test_flows_runner.py`

**Interfaces:**
- Consumes: each script's `main(argv) -> int` and `--json`; `train_mlflow.main`'s 30 s rule.
- Produces: `run_all.render_results(reports: dict[str, dict], when: str) -> str` (Markdown), `run_all.main(argv) -> int` with `--offline | --live` (exactly one required), `--write-results`, `--skip NAME` (repeatable).

- [ ] **Step 1: Write the failing runner test**

```python
# tests/test_flows_runner.py
"""run_all's rendering and its refusal to record offline numbers."""

import importlib.util
from pathlib import Path

import pytest

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _runner():
    spec = importlib.util.spec_from_file_location("flows_run_all", FLOWS / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reports(mode):
    return {
        "persona_eval": {
            "mode": mode, "chat_model": "m", "embed_model": "e", "items": 40,
            "personas": [{
                "persona": "bench-chemist", "seconds_per_reaction": 1.5,
                "reactions": {"precision": 0.9, "recall": 0.8, "auc": None, "tp": 9, "fp": 1,
                              "fn": 2, "tn": 28, "n_unsure": 0, "confidence_histogram": {5: 40}},
                "ranker": {"rank_auc": 0.95, "classifier_auc": 0.7, "classifier_n_clicks": 40,
                           "provenance_auc": None},
            }],
        },
        "mcp_e2e": {"mode": mode, "chat_model": "m", "rows": [
            {"surface": "feed", "tool": "feed.list", "problem": None, "ok": True},
            {"surface": "full", "tool": "sym.solve", "problem": None, "ok": False},
        ]},
        "train_mlflow": {"seconds": 4.2, "arms": [{"C": 1.0, "auc": 0.99, "precision": 0.98, "recall": 0.97, "accuracy": 0.97}]},
    }


def test_results_markdown_names_mode_models_and_every_number():
    text = _runner().render_results(_reports("live"), when="2026-08-28")
    assert "2026-08-28" in text and "live" in text and "chat=m" in text
    assert "0.900" in text and "0.800" in text and "n/a" in text  # precision, recall, inert AUC
    assert "0.950" in text and "feed.list" in text and "4.2" in text


def test_offline_numbers_are_never_written_as_results():
    with pytest.raises(ValueError, match="offline"):
        _runner().render_results(_reports("offline"), when="2026-08-28")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --frozen pytest tests/test_flows_runner.py -q`
Expected: FAIL — no `run_all.py`.

- [ ] **Step 3: Write `run_all.py`**

```python
# examples/flows/run_all.py
"""Run every example flow and print one summary.

    uv run --group examples python examples/flows/run_all.py --offline   # CI: stub model
    uv run --group examples python examples/flows/run_all.py --live --write-results

Order: training first (needs no model), then the persona eval, then the
MCP end-to-end. Each flow writes a JSON report; --write-results renders
RESULTS.md from LIVE reports only -- offline numbers are about the stub.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

FLOWS = ("train_mlflow", "persona_eval", "mcp_e2e")
SCRIPT = {
    "train_mlflow": _common.FLOWS_DIR / "training" / "train_mlflow.py",
    "persona_eval": _common.FLOWS_DIR / "persona_eval.py",
    "mcp_e2e": _common.FLOWS_DIR / "mcp_e2e.py",
}


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_results(reports: dict[str, dict], when: str) -> str:
    modes = {r.get("mode") for r in reports.values() if "mode" in r}
    if modes != {"live"}:
        raise ValueError(f"RESULTS.md records live numbers only; got modes {sorted(modes)} (offline is the stub)")
    pe, me, tr = reports.get("persona_eval"), reports.get("mcp_e2e"), reports.get("train_mlflow")
    out = [f"# Example flows: results measured {when}", ""]
    if pe:
        out += [f"## Persona eval (mode=live, chat={pe['chat_model']}, embed={pe['embed_model']}, items={pe['items']})", "",
                "Agreement with `corpus/labels.json`; evidence about the flow, not a model benchmark.", "",
                "| persona | precision | recall | AUC (signed confidence) | tp/fp/fn/tn | unsure | rank AUC | classifier AUC | s/reaction |",
                "|---|---|---|---|---|---|---|---|---|"]
        for p in pe["personas"]:
            r, k = p["reactions"], p["ranker"]
            out.append(f"| {p['persona']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['auc'])} |"
                       f" {r['tp']}/{r['fp']}/{r['fn']}/{r['tn']} | {r['n_unsure']} | {_fmt(k['rank_auc'])} |"
                       f" {_fmt(k['classifier_auc'])} ({k['classifier_n_clicks']} clicks) | {p['seconds_per_reaction']:.2f} |")
        out += ["", "Confidence histograms: " + "; ".join(
            f"{p['persona']} {p['reactions']['confidence_histogram']}" for p in pe["personas"]), ""]
    if me:
        rows = me["rows"]
        failed = [r for r in rows if r["problem"]]
        out += [f"## MCP end to end (mode=live, chat={me['chat_model']})", "",
                f"{len(rows)} calls over stdio across feed / provenance / knowledge / symbolic / full; {len(failed)} failed.", ""]
        out += ["| surface | tool | result |", "|---|---|---|"]
        out += [f"| {r['surface']} | {r['tool']} | {'FAILED: ' + r['problem'] if r['problem'] else ('refused' if r['ok'] is False else 'ok')} |" for r in rows]
        out.append("")
    if tr:
        out += [f"## Training family `c_sweep` (mlflow-skinny, {tr['seconds']:.1f} s for {len(tr['arms'])} arms)", "",
                "| C | accuracy | precision | recall | AUC |", "|---|---|---|---|---|"]
        out += [f"| {a['C']} | {a['accuracy']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['auc']:.4f} |" for a in tr["arms"]]
        out.append("")
    return "\n".join(out)


def _run(name: str, mode_flag: list[str], json_path: Path) -> tuple[int, dict]:
    cmd = [sys.executable, str(SCRIPT[name]), *mode_flag, "--json", str(json_path)]
    print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    report = json.loads(json_path.read_text()) if json_path.exists() else {}
    return rc, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--write-results", action="store_true")
    ap.add_argument("--skip", action="append", default=[], choices=FLOWS)
    args = ap.parse_args(argv)

    reports, failures = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for name in FLOWS:
            if name in args.skip:
                continue
            flag = [] if name == "train_mlflow" else (["--offline"] if args.offline else [])
            if name == "train_mlflow":
                out = Path(tmp) / "training"
                out.mkdir()
                flag = ["--out", str(out)]
            rc, report = _run(name, flag, Path(tmp) / f"{name}.json")
            reports[name] = report
            if rc != 0:
                failures.append(name)
    print("\n=== summary")
    for name in FLOWS:
        if name in reports:
            print(f"{name:<14} {'FAILED' if name in failures else 'ok'}")
    if args.write_results and not failures:
        text = render_results(reports, when=dt.date.today().isoformat())
        (_common.FLOWS_DIR / "RESULTS.md").write_text(text)
        print(f"wrote {_common.FLOWS_DIR / 'RESULTS.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `run_all` runs `train_mlflow` into a temp `--out` so a demo run never rewrites the committed `mlruns/`; regenerating the committed one is `python examples/flows/training/train_mlflow.py` on purpose.

- [ ] **Step 4: Run the runner tests, then the whole thing offline**

Run: `uv run --frozen pytest tests/test_flows_runner.py -q` → 2 passed.
Run: `time uv run --group examples python examples/flows/run_all.py --offline` → summary with three `ok`, exit 0, under three minutes.

- [ ] **Step 5: Add the CI job**

Append to `.github/workflows/ci.yml`:

```yaml
  flows:
    name: example flows (offline)
    runs-on: ubuntu-latest
    env:
      UV_PYTHON_PREFERENCE: only-managed
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - name: Sync dependencies (with the examples group)
        run: uv sync --all-extras --dev --group examples
      - name: Every flow, against the stub model server
        # The demonstration in examples/flows, end to end: the MLflow family,
        # the persona eval, and every MCP tool over stdio. No Ollama, no
        # network. Its numbers are about the stub and say so; the point is
        # that the flows cannot rot.
        run: uv run python examples/flows/run_all.py --offline
```

- [ ] **Step 6: Run live and write RESULTS.md**

Check `nvidia-smi` for other jobs first. Run: `uv run --group examples python examples/flows/run_all.py --live --write-results` (10–15 minutes). Expected: three `ok`, `RESULTS.md` written. Read it. If a flow fails live but passed offline, that is a real finding: fix it in its own commit per the spec, re-run.

- [ ] **Step 7: Write `examples/flows/README.md`**

Sections, in this order, each a short paragraph: what is here (the three flows and the fixture); how to run (`--offline` in 2 minutes with nothing installed but `uv sync --group examples`; `--live` against Ollama); what each flow demonstrates and what its numbers mean (persona eval = agreement with forty hand labels; MCP = every tool over stdio on every surface; training = a real MLflow directory read by the ledger, four arms in N seconds — take N from `RESULTS.md`); where the numbers live (`RESULTS.md`, live only, dated, model named); how to regenerate `training/mlruns` and why that is deliberate; and a pointer to `docs/superpowers/specs/2026-08-28-example-flows-design.md`.

Add to the repo `README.md`, after "Try it in 60 seconds", a three-line "Demonstrations" pointer: one command (`uv run --group examples python examples/flows/run_all.py --offline`), one sentence on what it covers, a link to `examples/flows/README.md`.

Add `examples/flows` (and `examples/flows/training`) to the docs index in `CLAUDE.md`, and one line under Key API Patterns: `|Example flows: examples/flows/{persona_eval,mcp_e2e,training/train_mlflow,run_all}.py — --offline uses stub_openai.py (stdlib http.server speaking /v1/embeddings + /v1/chat/completions; numbers are about the stub), --live is Ollama and the only mode that writes RESULTS.md|CI job `flows` runs --offline|mcp_e2e is the ONLY thing that drives attest-mcp over stdio`.

- [ ] **Step 8: Gates, commit, push, watch CI**

```bash
uv run --frozen pre-commit run --all-files 2>&1 | grep -E '^\S.*\.(Passed|Failed)$'
git add examples/flows/run_all.py examples/flows/README.md examples/flows/RESULTS.md tests/test_flows_runner.py .github/workflows/ci.yml README.md CLAUDE.md
git commit -m "run_all and a CI job: the demonstration runs offline on every push, and RESULTS.md records what it measured live"
git push origin main
gh run watch --exit-status   # the new flows job must be green
```

---

## Self-review

**Spec coverage.** Fixture (Task 1); stub (Task 2); persona eval with precision/recall/AUC, unsure-reporting, confidence histogram, ranker AUCs (Task 3); MCP over stdio, every tool, all five spawns, envelope and `options` checks, response-size probe via `feed.list` at 4 and 13 (Task 4 — the probe is a call, the 7000-char ceiling is asserted by the existing `test_response_size.py`, so the flow reports the size in the row's `message` only); training family, committed `mlruns/`, `FINDINGS.md` with one stale claim, 30 s enforced, docstring/spec caveat retired, dependency group, `src/` guard (Task 5); `run_all`, CI job, README, RESULTS live-only, CLAUDE.md (Task 6). Success criteria: offline exits 0 in CI (Task 6 step 5/8); every tool called (Task 4 test); reader caveat retired (Task 5 step 7); no new BLE001 in `src/` (the one in `examples/` is outside the counted tree — Task 4 says to verify); nothing in `src/` imports mlflow (Task 5 step 3).

**Placeholders.** None: every code step carries its code; the corpus entries beyond the four shown are specified by count, group, shape and the tests that pin them.

**Type consistency.** `prepare_db` returns `{"items", "users", "guid_to_id"}` and Task 4 reads exactly those keys; `score_verdicts` keys match `render` and `render_results`; `CALLS` tuples are `(str, dict, str)` in both the flow and its test; `train_mlflow` reports `{"seconds", "arms"}` and `render_results` reads both.
