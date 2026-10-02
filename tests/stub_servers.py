"""Stub OpenAI-compatible servers on a loopback port, with a request log.

`ollama_embedder_only()` is the real failure shape from a hosted Research Desk:
a loopback server that serves /v1/embeddings for ONE embedding model and
answers every chat request with Ollama's actual 404 body for a model it has not
pulled. `chat_server()` stands in for a hosted chat endpoint (NVIDIA NIM): it
requires a Bearer key and records what it was sent, so a test can assert both
that a request arrived and that the key never left the Authorization header.
"""

from __future__ import annotations

import http.server
import json
import threading

OLLAMA_404 = {
    "error": {
        "message": "model 'gemma4:e2b-it-q4_K_M' not found",
        "type": "api_error",
        "param": None,
        "code": None,
    }
}


class StubServer:
    def __init__(
        self, *, embed_model=None, chat_models=(), api_key=None, chat_404_for=None, echo_auth=False
    ):
        self.reply = {"content_type": "paper", "tags": ["graph-neural-networks"]}
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append({"method": "GET", "path": self.path})
                self._send(200, {"status": "stub"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "model": req.get("model"),
                        "auth": self.headers.get("Authorization"),
                        "x_api_key": self.headers.get("x-api-key"),
                        "body": req,
                    }
                )
                presented = {self.headers.get("Authorization"), self.headers.get("x-api-key")}
                echo = (
                    f" (you sent Authorization: {self.headers.get('Authorization')};"
                    f" x-api-key: {self.headers.get('x-api-key')})"
                    if echo_auth
                    else ""
                )
                if api_key and not presented & {f"Bearer {api_key}", api_key}:
                    return self._send(401, {"error": {"message": "Authorization failed" + echo}})
                if self.path.endswith("/embeddings"):
                    if req.get("model") != embed_model:
                        return self._send(404, {"error": {"message": "embed model not served"}})
                    inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
                    vec = [1.0] + [0.0] * 767
                    return self._send(
                        200,
                        {"data": [{"index": i, "embedding": vec} for i in range(len(inputs))]},
                    )
                if self.path.endswith("/chat/completions"):
                    if req.get("model") not in chat_models:
                        body = chat_404_for or {
                            "error": {
                                **OLLAMA_404["error"],
                                "message": f"model '{req.get('model')}' not found" + echo,
                            }
                        }
                        return self._send(404, body)
                    return self._send(
                        200, {"choices": [{"message": {"content": json.dumps(outer.reply)}}]}
                    )
                if self.path.endswith("/responses"):  # the ChatGPT-sign-in (Codex) wire
                    return self._responses(req)
                if self.path.endswith("/messages"):  # Anthropic's native wire
                    return self._messages(req)
                self._send(404, {"error": {"message": "no such path"}})

            def _responses(self, req):
                text = json.dumps(outer.reply)
                item = {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_1",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
                done = {
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": req.get("model"),
                    "output": [item],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
                events = [
                    (
                        "response.created",
                        {"response": {**done, "status": "in_progress", "output": []}},
                    ),
                    ("response.output_item.done", {"output_index": 0, "item": item}),
                    ("response.completed", {"response": done}),
                ]
                body = "".join(
                    f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"
                    for name, data in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _messages(self, req):
                text = json.dumps(outer.reply)
                message = {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": req.get("model"),
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
                if not req.get("stream"):
                    return self._send(200, message)
                # Hermes' Anthropic adapter always streams.
                events = [
                    ("message_start", {"message": {**message, "content": [], "stop_reason": None}}),
                    (
                        "content_block_start",
                        {"index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    (
                        "content_block_delta",
                        {"index": 0, "delta": {"type": "text_delta", "text": text}},
                    ),
                    ("content_block_stop", {"index": 0}),
                    (
                        "message_delta",
                        {
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 1},
                        },
                    ),
                    ("message_stop", {}),
                ]
                body = "".join(
                    f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"
                    for name, data in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def posts(self, suffix: str) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST" and r["path"].endswith(suffix)]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def ollama_embedder_only(embed_model="embeddinggemma") -> StubServer:
    return StubServer(embed_model=embed_model)


def chat_server(
    models=("nvidia/nemotron-3-super-120b-a12b",), api_key="nvapi-FAKE-KEY", echo_auth=False
) -> StubServer:
    return StubServer(chat_models=models, api_key=api_key, echo_auth=echo_auth)
