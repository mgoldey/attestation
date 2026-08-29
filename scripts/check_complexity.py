#!/usr/bin/env python
"""Per-file complexity ratchet.

A single global threshold does not work here. Setting `xenon --max-absolute D`
so a clean checkout passes also permits any file to grow a D-rated function,
which is most of the regression worth catching -- verified by adding a 17-branch
function to embed.py and watching a global-D gate return 0.

So each file is pinned at ITS OWN measured worst, recorded in BASELINE below.
A file may improve freely; the baseline is then lowered in the same commit. A
file may not get worse without someone editing this list and saying why in the
commit message, which is the point: the ratchet makes the decision visible
rather than silent.

Run: uv run --with radon python scripts/check_complexity.py
"""

import subprocess
import sys

# file -> highest cyclomatic complexity permitted, measured 2026-08-21.
# Lower these when the tree improves. Raising one requires a reason.
BASELINE = {
    "src/attestation/ledger.py": 27,  # _caveats -- lowered 2026-08-29: the
    # onion-seams split moved compare()'s decision logic into _compare/
    # _pick_metric, so compare itself is no longer the worst function here.
    "src/attestation/ledger_adapters/generic.py": 29,  # discover
    "src/attestation/kg.py": 20,  # communities
    "src/attestation/mcp/feed.py": 10,  # _score_matches: relevance blend --
    # lowered 2026-08-29 after the onion-seams split moved the relevance-floor
    # policy into the pure _apply_relevance_floor.
    "src/attestation/corpus.py": 17,  # detect_in_source
    "src/attestation/claims.py": 15,  # check_claim
    "src/attestation/cli.py": 14,  # cmd_claims
    "src/attestation/rank.py": 12,  # _ranking_quality -- lowered 2026-08-29:
    # the onion-seams split moved rank_items's blend into the pure rank_rows,
    # so rank_items is no longer the worst function here.
    # route_feed is a rule table: each intent is one branch, and flattening it
    # into data would hide the ordering that the tests pin (a mutation like
    # "add X to my feeds" must be tested before the listing it mentions).
    "src/attestation/mcp/ask.py": 10,  # route_feed: +1 for the read/summarise rule
}

# Anything not listed must stay at or below this. New code starts strict.
DEFAULT_MAX = 10


def worst_per_file() -> dict[str, tuple[int, str]]:
    out = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "src/attestation", "-s", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    worst: dict[str, tuple[int, str]] = {}
    for path, blocks in json.loads(out.stdout).items():
        for b in blocks:
            if not isinstance(b, dict) or "complexity" not in b:
                continue
            cc = b["complexity"]
            if path not in worst or cc > worst[path][0]:
                worst[path] = (cc, b.get("name", "?"))
    return worst


def main() -> int:
    failures = []
    improvements = []
    for path, (cc, name) in sorted(worst_per_file().items()):
        limit = BASELINE.get(path, DEFAULT_MAX)
        if cc > limit:
            failures.append(f"  {path}: {name} is {cc}, limit {limit}")
        elif path in BASELINE and cc < limit:
            improvements.append(f"  {path}: now {cc}, baseline still {limit} -- lower it")

    for line in improvements:
        print("improved:", line.strip())
    if failures:
        print("\ncomplexity regressed:")
        print("\n".join(failures))
        print("\nSimplify the function, or raise its baseline in")
        print("scripts/check_complexity.py and say why in the commit message.")
        return 1
    print(f"complexity: ok ({len(BASELINE)} files pinned, others capped at {DEFAULT_MAX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
