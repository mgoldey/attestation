# demos/feed/record.py
"""Drives attest serve's real HTMX UI with Playwright and records video.

Needs a database seeded by seed_feed_db.py (a persona plus real tagged
items) and the `demos` dependency group:

    uv run python seed_feed_db.py /path/to/demo.db   # once; needs a model server
    uv run --group demos playwright install chromium  # once
    ATTEST_DB=/path/to/demo.db uv run --group demos python record.py

Starts `attest serve` as a subprocess against ATTEST_DB, walks the real
page (persona already selected, browse, mark one useful, one not useful,
open the onboarding form), and writes a .webm to ../../demo/feed.webm via
Playwright's own video recording -- not committed, matching the other
demos in demos/.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "demo"
PORT = 8899
BASE_URL = f"http://127.0.0.1:{PORT}"
PERSONA = "demo-reader"


def wait_for_server(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("attest serve exited before it came up")
        try:
            httpx.get(BASE_URL, timeout=1.0)
            return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise TimeoutError(f"attest serve did not answer at {BASE_URL} within {timeout}s")


def run() -> int:
    db_path = os.environ.get("ATTEST_DB")
    if not db_path or not Path(db_path).exists():
        print("ATTEST_DB must point at a database seeded by seed_feed_db.py", file=sys.stderr)
        print("  uv run python seed_feed_db.py /path/to/demo.db", file=sys.stderr)
        return 1

    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(exist_ok=True)
    env = {**os.environ, "ATTEST_DB": db_path}
    proc = subprocess.Popen(
        ["uv", "run", "attest", "serve", "--port", str(PORT)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(proc)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1024, "height": 768},
                record_video_dir=str(OUT_DIR),
                record_video_size={"width": 1024, "height": 768},
            )
            page = context.new_page()

            page.goto(f"{BASE_URL}/?user={PERSONA}")
            page.wait_for_selector("#feed li")
            page.wait_for_timeout(1500)

            first_item = page.locator("#feed li").first
            first_item.locator("button.yn", has_text="✓").click()
            page.wait_for_timeout(1500)

            second_item = page.locator("#feed li").nth(1)
            second_item.locator("button.yn", has_text="✗").click()
            page.wait_for_timeout(1500)

            page.get_by_role("link", name="+ new reader").click()
            page.wait_for_selector("input[name=name]")
            page.wait_for_timeout(1500)

            context.close()
            browser.close()

        recorded = sorted(OUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if recorded:
            final = OUT_DIR / "feed.webm"
            recorded[-1].replace(final)
            print(f"wrote {final}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(run())
