"""Apply a source mutation and CONFIRM it landed, for verifying a guard.

Written after the third time in one session that a `ruff format` reflow made a
string-replace edit silently no-op: the target had been joined onto one line,
sed matched nothing, and the test "passed with the fix removed" when the fix
had never been removed. A mutation that does not apply looks exactly like a
test that does not bite, and the two have opposite meanings.

    uv run python scripts/mutate.py src/attestation/mcp/feed.py \
        'RELEVANCE_ANCHOR = 3' 'RELEVANCE_ANCHOR = 1'
    uv run pytest tests/test_search.py -q -k anchored   # expect FAIL
    uv run python scripts/mutate.py --restore

Exits nonzero if the pattern is absent or ambiguous, so a bad probe is a loud
failure rather than a quiet pass.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

BACKUP = Path(".mutate-backup")


def restore() -> int:
    if not BACKUP.exists():
        print("nothing to restore", file=sys.stderr)
        return 1
    for saved in BACKUP.iterdir():
        target = Path(saved.read_text().split("\n", 1)[0])
        target.write_text(saved.read_text().split("\n", 1)[1])
        print(f"restored {target}")
    shutil.rmtree(BACKUP)
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--restore":
        return restore()
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    path, old, new = Path(argv[0]), argv[1], argv[2]
    text = path.read_text()
    count = text.count(old)
    if count == 0:
        print(f"pattern not found in {path} -- the mutation did NOT apply", file=sys.stderr)
        print("  (a formatter may have reflowed it; check the real text)", file=sys.stderr)
        return 1
    if count > 1:
        print(f"pattern appears {count} times in {path}; be more specific", file=sys.stderr)
        return 1

    BACKUP.mkdir(exist_ok=True)
    (BACKUP / path.name).write_text(f"{path}\n{text}")
    path.write_text(text.replace(old, new))
    print(f"mutated {path}: {old!r} -> {new!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
