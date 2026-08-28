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
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                }
            )
            return
        self._send({"error": f"unknown path {self.path}"}, status=404)


def start(dims: int | None = None) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Serve on 127.0.0.1:<free port> in a daemon thread. Returns (server, base_url)."""
    handler = type(
        "Handler", (_Handler,), {"dims": dims or int(os.environ.get("EMBED_DIMS", "256"))}
    )
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
