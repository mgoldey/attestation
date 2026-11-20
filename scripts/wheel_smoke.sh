#!/usr/bin/env bash
# The wheel smoke test, run by .github/workflows/ci.yml's `wheel smoke test`
# job through reusable-wheel-smoke.yml: the BUILT wheels, installed with plain
# pip into fresh venvs, from a checkout where dist/ holds both distributions
# this repo builds (attestation-*.whl and attestation_schema-*.whl; the glob
# `attestation-*` cannot match `attestation_schema-*`).
#
#   bash scripts/wheel_smoke.sh [DIST_DIR]
set -euo pipefail

dist="${1:-dist}"
work="$(mktemp -d)"
shopt -s nullglob
app=("$dist"/attestation-*.whl)
schema=("$dist"/attestation_schema-*.whl)
[ "${#app[@]}" -eq 1 ] && [ "${#schema[@]}" -eq 1 ] || {
  echo "::error::expected one attestation and one attestation-schema wheel in $dist, found: ${app[*]} ${schema[*]}"
  exit 1
}

echo "::group::Install both wheels into a clean venv"
# attestation depends on attestation-schema at runtime (the ledger reads core
# records); the schema wheel goes in beside it -- see docs/superpowers/specs/
# 2026-11-19-ledger-reads-core-records-design.md, "Dependency".
python3 -m venv "$work/wheel-venv"
"$work/wheel-venv/bin/pip" install --quiet "${schema[0]}" "${app[0]}"
py="$work/wheel-venv/bin/python"
echo "::endgroup::"

echo "the ledger imports a core record from the installed wheels"
"$py" -c "
import attestation.ledger, attestation.ledger_adapters as a
assert [x.name for x in a.ADAPTERS][0] == 'core'
print('ledger adapters:', [x.name for x in a.ADAPTERS])
"

echo "attest --help exits 0"
"$work/wheel-venv/bin/attest" --help >/dev/null

echo "attestation.server imports cleanly"
"$py" -c "import attestation.server"

# Regression insurance: the advertised `uvx --from git+... attest ingest`
# on-ramp reads attestation/feeds.toml out of the installed package, so a
# repackaging that drops it breaks first-run ingest silently. py.typed: a
# downstream type checker treats a package without it as untyped and ignores
# its annotations, silently, even though this repo checks them with `ty`.
echo "feeds.toml and py.typed are packaged"
"$py" -c "
import importlib.resources as r
for name in ('feeds.toml', 'py.typed'):
    p = r.files('attestation').joinpath(name)
    assert p.is_file(), f'attestation/{name} missing from installed wheel'
    print(name, 'present in wheel:', p)
"

echo "attestation-schema installs alone and round-trips a record"
python3 -m venv "$work/schema-venv"
"$work/schema-venv/bin/pip" install --quiet --no-deps "${schema[0]}"
"$work/schema-venv/bin/python" -c "
import dataclasses, importlib.metadata as m, importlib.resources as r, json, tempfile
import attestation_schema as s
reqs = m.requires('attestation-schema') or []
assert all('extra ==' in r for r in reqs), f'attestation-schema must declare no unconditional dependency: {reqs}'
assert r.files('attestation_schema').joinpath('py.typed').is_file()
schema = json.loads(r.files('attestation_schema').joinpath('schemas/run-1.schema.json').read_text())
assert schema['properties']['schema']['const'] == s.SCHEMA
rec = s.RunRecord(run_id='smoke', family=None, arm=None, status='complete',
    started_at=None, finished_at=None, producer=s.Producer('ci', None),
    code=s.CodeIdentity(None, None, None, None, 'unknown', None), config={}, data=[], metrics=[], history=None,
    seeds=None, dependencies=None, ext={'x': {'y': 1}})
d = tempfile.mkdtemp()
curve = [s.Series('loss', None, {}, [s.Point(0, float('nan'), None), s.Point(1, 0.5, None)])]
rec = dataclasses.replace(rec, history=s.write_history(d, curve))
s.write(rec, d)
assert s.read(d) == rec
assert s.read_history(d)[0].points[1] == s.Point(1, 0.5, None)
assert r.files('attestation_schema').joinpath('schemas/history-1.schema.json').is_file()
print('attestation-schema wheel OK:', s.SCHEMA, s.SCHEMA_STATUS)
"
echo "wheel smoke OK"
