"""Run `attest ingest` and `attest tag` against a real OpenAI-compatible
server -- here, the repo's own stub, started in-process.

Swap LLM_BASE_URL for vLLM/llama.cpp/LM Studio/Ollama (see README) and the
same subprocess calls exercise attestation.llm's real HTTP client against
that server. No background process, no port race: the stub is a
ThreadingHTTPServer started and shut down in this one script.

    uv run python with_server.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flows"))
import _common
import stub_openai


def main() -> int:
    server, base_url = stub_openai.start()
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["CHAT_MODEL"] = stub_openai.MODEL
    os.environ["EMBED_MODEL"] = stub_openai.MODEL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["ATTEST_DB"] = str(Path(tmp) / "attest.db")
            feeds_path = _common.write_feeds_toml(Path(tmp))
            ingest = subprocess.run(
                ["uv", "run", "attest", "ingest", "--feeds", str(feeds_path)],
                check=True,
            )
            tag = subprocess.run(
                ["uv", "run", "attest", "tag", "--limit", "5"],
                check=True,
            )
    finally:
        server.shutdown()
    return ingest.returncode or tag.returncode


if __name__ == "__main__":
    sys.exit(main())
