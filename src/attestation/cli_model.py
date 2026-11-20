"""`attest model trace`: a model version back to its training run and data.

Files only, like `attest runs lineage` (no ledger database, no registry
server): the reference is resolved in the local stores under --search, the
runs are read through the ledger's own adapters, and the chain is printed
with every issue. Exits 1 when a link is broken (an unresolvable reference
included), 2 when a registry reference does not parse.
Spec: docs/superpowers/specs/2026-11-20-model-registry-linkage-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attestation.cli import HELP, _documented


def register(sub) -> None:
    """Attach `attest model trace` to the top-level subparsers."""
    sp = sub.add_parser("model", help=HELP["model"])
    model_sub = sp.add_subparsers(dest="model_command", required=True)
    p = model_sub.add_parser("trace", help=HELP["model.trace"])
    p.add_argument(
        "ref",
        help="models:/<name>/<version>, models:/<name>@<alias>, hf:<org>/<name>[@<revision>],"
        " or a directory of saved model files",
    )
    p.add_argument(
        "--search",
        action="append",
        default=[],
        metavar="PATH",
        help="where the registry stores, run and dataset records are (repeatable; default: cwd)",
    )
    p.add_argument("--json", action="store_true", help="print the trace as JSON")
    p.set_defaults(func=cmd_model_trace)


@_documented("model.trace")
def cmd_model_trace(args: argparse.Namespace) -> int:
    """Reads local files only: a registry's local store, a hub cache snapshot
    or a saved-model directory, then every run and dataset record under
    --search. Never contacts a registry."""
    from attestation import model_trace
    from attestation.ledger_adapters import registries

    roots = [Path(p).expanduser() for p in (args.search or ["."])]
    index = model_trace.build_index(roots)
    try:
        found = model_trace.trace(args.ref, index, roots)
    except registries.RefError as exc:
        print(f"attest model trace: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({**found.to_dict(), "skipped": index.skipped}, indent=2))
    else:
        print(found.format())
        for line in index.skipped:
            print(f"skipped (unreadable): {line}")
    return 0 if found.ok else 1
