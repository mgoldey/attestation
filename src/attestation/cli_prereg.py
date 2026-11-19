"""`attest prereg`: register a prediction before the runs, judge it after.

Its own module (the `cli_bib.py` precedent) so cli.py gains only the parser
hook and the HELP entries. Registrations are files in the workspace --
`hypotheses.toml` plus `prereg/` -- so `add` and `list` need no database;
`check` reads the ledger through `ledger.compare`. What a registration does
and does not prove is `prereg.LIMITS`, printed by every command here. See
docs/superpowers/specs/2026-11-19-preregistration-design.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from attestation.cli import HELP, _documented, fail, open_db


def register(sub, add_db) -> None:
    """Attach `attest prereg <command>` to the top-level subparsers."""
    sp = sub.add_parser("prereg", help=HELP["prereg"])
    pre = sp.add_subparsers(dest="prereg_command", required=True)
    root_help = "where hypotheses.toml and prereg/ live (default: $RESEARCH_ROOT, else cwd)"

    p = pre.add_parser("add", help=HELP["prereg.add"])
    p.add_argument("family", help="the experiment family, as `attest runs list` names it")
    p.add_argument("--metric", required=True, help="a metric with a declared direction")
    p.add_argument(
        "--expect",
        required=True,
        help="arms joined by < or > on the metric's raw values, e.g. 'kd_t4 < baseline'",
    )
    p.add_argument(
        "--by", required=True, type=float, help="minimum margin per pair, in the metric's units"
    )
    p.add_argument("--note", default="", help="why you expect it (signed with the rest)")
    p.add_argument(
        "--allow-single-run",
        action="store_true",
        help="let point values (arms without an interval) confirm or refute",
    )
    p.add_argument("--supersedes", metavar="ID", help="the registration this one replaces")
    p.add_argument("--root", help=root_help)
    p.set_defaults(func=cmd_prereg_add)

    p = pre.add_parser("check", help=HELP["prereg.check"])
    add_db(p)
    p.add_argument("family", nargs="?", help="only this family's hypotheses")
    p.add_argument("--root", help=root_help)
    p.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    p.set_defaults(func=cmd_prereg_check)

    p = pre.add_parser("list", help=HELP["prereg.list"])
    p.add_argument("family", nargs="?", help="only this family's hypotheses")
    p.add_argument("--root", help=root_help)
    p.set_defaults(func=cmd_prereg_list)


def _print_limits() -> None:
    from attestation import prereg

    print("\nWhat a preregistration shows:")
    for line in prereg.LIMITS:
        print(f"  - {line}")


@_documented("prereg.add")
def cmd_prereg_add(args: argparse.Namespace) -> int:
    """Refuses a metric with no declared direction, and content already
    registered; writes nothing until the key is known to work."""
    from attestation import prereg
    from attestation.run_record import keys

    root = prereg.workspace(args.root)
    try:
        prereg.refuse_undeclared(args.metric)
        supersedes = prereg.resolve_supersedes(
            args.supersedes, prereg.parse_file(root / prereg.INDEX_NAME)
        )
        hyp = prereg.Hypothesis(
            family=args.family,
            metric=args.metric,
            expect=args.expect,
            by=args.by,
            allow_single_run=args.allow_single_run,
            note=args.note,
            supersedes=supersedes,
        )
        signer = keys.resolve_key_source().signer()
        done = prereg.register(hyp, root, signer)
    except (ValueError, FileExistsError, OSError, keys.RecordKeyError) as exc:
        return fail(f"not registered: {exc}")
    print(f"registered {done.id[: prereg.ID_PREFIX]}  {hyp.describe()}")
    print(f"  registered_at {done.registered_at} (this machine's clock)")
    print(f"  signed by key {done.keyid}")
    print(f"  record {done.record.relative_to(root)}; index {done.index.name}")
    _print_limits()
    return 0


def _judged(conn, regs) -> list[dict]:
    """Each registration judged against a fresh compare on ITS metric."""
    from attestation import ledger, prereg

    cache: dict[tuple[str, str], tuple[dict | None, str | None]] = {}
    out = []
    for reg in regs:
        if reg.hypothesis is None:
            out.append(prereg.judge(reg, None, "the registration could not be read"))
            continue
        key = (reg.hypothesis.family, reg.hypothesis.metric)
        if key not in cache:
            try:
                result = ledger.compare(conn, key[0], metric=key[1])
                cache[key] = (result, None) if result["arms"] else (None, result.get("message"))
            except ValueError as exc:
                cache[key] = (None, str(exc))
        out.append(prereg.judge(reg, *cache[key]))
    return out


def _print_judged(item: dict) -> None:
    flags = f"  [{', '.join(item['flags'])}]" if item["flags"] else ""
    print(f"{item['verdict'].upper():13s} {item['id']}  {item['expect'] or '(unreadable)'}{flags}")
    print(f"  registered_at {item['registered_at'] or '?'}; key {item['keyid'] or '?'}")
    for pair in item["pairs"]:
        arms = f"  ({'; '.join(pair['arms'])})" if pair.get("arms") else ""
        print(f"  {pair['pair']}: {pair['verdict']} -- {pair['why']}{arms}")
    shown = {p["why"] for p in item["pairs"]}
    for reason in item["reasons"]:
        if reason not in shown:
            print(f"  - {reason}")


def _print_report(judged: list[dict]) -> None:
    """Every verdict, the limits, then one tally line."""
    for item in judged:
        _print_judged(item)
    tally = Counter(item["verdict"] for item in judged)
    flagged = Counter(f for item in judged for f in item["flags"])
    summary = ", ".join(f"{n} {v}" for v, n in sorted(tally.items()))
    extra = "".join(f", {n} {f}" for f, n in sorted(flagged.items()))
    _print_limits()
    print(f"\n{len(judged)} hypothes{'i' if len(judged) == 1 else 'e'}s: {summary}{extra}")


def _select(args):
    """The registrations under the workspace, narrowed to `args.family`."""
    from attestation import prereg

    regs = prereg.audit(prereg.workspace(args.root))
    if args.family:
        regs = [r for r in regs if r.hypothesis is None or r.hypothesis.family == args.family]
    return regs


@_documented("prereg.check")
def cmd_prereg_check(args: argparse.Namespace) -> int:
    """Exit 1 on any refuted verdict or altered registration, like `attest
    claims`; post_hoc alone does not fail it -- it is evidence, not proof."""
    from attestation import prereg

    try:
        regs = _select(args)
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    if not regs:
        return fail(f"no hypotheses registered under {prereg.workspace(args.root)}")
    with open_db(args.db) as conn:
        judged = _judged(conn, regs)
    if args.json:
        print(json.dumps({"hypotheses": judged, "limits": list(prereg.LIMITS)}, indent=2))
    else:
        _print_report(judged)
    return 1 if any(prereg.fails(item) for item in judged) else 0


@_documented("prereg.list")
def cmd_prereg_list(args: argparse.Namespace) -> int:
    """Reads the files only (no database): each registration, its signed
    time and key, and anything that no longer matches what was signed."""
    try:
        regs = _select(args)
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    for reg in regs:
        flags = f"  [{', '.join(reg.flags)}]" if reg.flags else ""
        what = reg.hypothesis.describe() if reg.hypothesis else "(unreadable)"
        print(f"{reg.short}  {reg.registered_at or '?':20s}  {what}{flags}")
        for reason in reg.reasons:
            print(f"  - {reason}")
    print(f"{len(regs)} registration(s)")
    _print_limits()
    return 0
