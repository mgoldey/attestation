#!/usr/bin/env python3
"""Refuse an attestation release whose attestation-schema floor is not on PyPI.

attestation's wheel says `Requires-Dist: attestation-schema>=X`; if X is not on
the index the release installs for nobody. release.yml runs this before the
build (the `pre-build` of reusable-wheel-smoke.yml), so it fails now, with the
fix in the message, rather than after publishing. Stdlib only: it runs on the
runner's bare python3 before anything is installed.

An unreachable PyPI fails the release too ("cannot verify"), never passes it:
a check that passes when it cannot look is no check.
"""

import json
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request

URL = "https://pypi.org/pypi/attestation-schema/json"


def _floor(pyproject: str = "pyproject.toml") -> str:
    with open(pyproject, "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    floors = [
        m.group(1) for d in deps if (m := re.match(r"attestation-schema\s*>=\s*([0-9][^\s,;]*)", d))
    ]
    if len(floors) != 1:
        sys.exit(f"expected one 'attestation-schema>=X' dependency, found {floors}")
    return floors[0]


def _release_first(floor: str) -> str:
    return f"release it first (git tag schema-v{floor} && git push origin schema-v{floor})"


def main() -> None:
    floor = _floor()
    data = None
    err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(URL, timeout=20) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                sys.exit("attestation-schema is not on PyPI at all: " + _release_first(floor))
            err = e
        except (OSError, ValueError) as e:  # network failure or a body that is not JSON
            err = e
        time.sleep(3 * (attempt + 1))
    if data is None:
        sys.exit(
            f"cannot verify that attestation-schema {floor} is on PyPI ({err}); "
            "refusing to release without the check"
        )
    files = data["releases"].get(floor) or []
    if not any(not f.get("yanked") for f in files):
        sys.exit(
            f"attestation-schema>={floor} is not published on PyPI "
            f"(published: {sorted(data['releases'])}): " + _release_first(floor)
        )
    print(f"attestation-schema {floor} is on PyPI")


if __name__ == "__main__":
    main()
