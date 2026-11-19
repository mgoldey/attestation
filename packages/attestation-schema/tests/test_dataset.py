"""`attestation.dataset/1` (draft): reader, writer, content digest, JSON Schema.

Spec: docs/superpowers/specs/2026-11-19-dataset-records-design.md.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil

import attestation_schema as s
import pytest
from attestation_schema import layout
from attestation_schema.canonical import canonicalize
from attestation_schema.record import (
    InvalidFieldError,
    MissingFieldError,
    UnknownFieldError,
    UnsupportedSchemaError,
)
from conformance_fixtures import DATA

SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "src" / "attestation_schema" / "schemas"
DATASET_SCHEMA_FILE = SCHEMAS / "dataset-1.schema.json"
LINEAGE = DATA / "lineage"
FULL = LINEAGE / "dataset"  # every field set: files, splits, source runs, license, ext


def _obj() -> dict:
    return json.loads((FULL / layout.DATASET_JSON).read_bytes())


def _with_content(obj: dict) -> dict:
    """`obj` with content_sha256 recomputed from its files[] (so a test can
    change a file entry and hit the rule it is about, not the digest rule)."""
    obj["content_sha256"] = s.content_digest({f["path"]: f["sha256"] for f in obj["files"]})
    return obj


# ---- round trip and bytes ------------------------------------------------------------


def test_the_committed_record_reads_and_writes_back_byte_for_byte():
    raw = (FULL / layout.DATASET_JSON).read_bytes()
    record = s.loads_dataset(raw)
    assert s.dumps_dataset(record) == raw
    assert s.dataset_digest(record) == hashlib.sha256(raw).hexdigest()
    assert s.dataset_from_dict(record.to_dict()) == record
    assert s.read_dataset(FULL) == record == s.read_dataset(FULL / layout.DATASET_JSON)


def test_input_key_order_does_not_change_the_bytes():
    obj = _obj()
    reordered = dict(reversed(list(obj.items())))
    reordered["files"] = [dict(reversed(list(f.items()))) for f in obj["files"]]
    assert s.dumps_dataset(s.dataset_from_dict(reordered)) == canonicalize(obj)


def test_content_digest_is_the_sha256_of_the_manifest_file():
    record = s.read_dataset(FULL)
    manifest = (FULL / layout.DIGESTS_JSON).read_bytes()
    assert record.content_sha256 == hashlib.sha256(manifest).hexdigest()
    assert s.content_digest(record.files) == record.content_sha256
    assert manifest == canonicalize(layout.build_manifest(FULL))


# ---- required keys, unknown keys --------------------------------------------------------


def _key_paths(obj: dict) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = [(k,) for k in obj]
    for section in ("producer", "code"):
        paths += [(section, k) for k in obj[section]]
    for section in ("files", "columns", "splits", "source_runs"):
        paths += [(section, "0", k) for k in obj[section][0]]
    return paths


@pytest.mark.parametrize("path", _key_paths(_obj()), ids=".".join)
def test_every_required_key_removed_is_refused_naming_its_path(path):
    obj = _obj()
    holder = obj
    for part in path[:-1]:
        holder = holder[int(part)] if part.isdigit() else holder[part]
    del holder[path[-1]]
    with pytest.raises(MissingFieldError) as exc:
        s.dataset_from_dict(obj)
    want = ".".join(path).replace(".0.", "[0].")
    assert exc.value.path == want


@pytest.mark.parametrize("where", ["", "files.0", "splits.0", "source_runs.0", "columns.0"])
def test_unknown_keys_are_refused(where):
    obj = _obj()
    holder = obj
    for part in filter(None, where.split(".")):
        holder = holder[int(part)] if part.isdigit() else holder[part]
    holder["extra"] = 1
    with pytest.raises(UnknownFieldError):
        s.dataset_from_dict(obj)


# ---- the content digest and files ------------------------------------------------------


def test_a_file_digest_changed_without_a_new_content_digest_is_refused():
    obj = _obj()
    obj["files"][0]["sha256"] = "0" * 64
    with pytest.raises(InvalidFieldError) as exc:
        s.dataset_from_dict(obj)
    assert exc.value.path == "content_sha256"
    assert s.dataset_from_dict(_with_content(obj)).files[0].sha256 == "0" * 64  # control


def test_files_are_sorted_by_path_each_once():
    obj = _obj()
    obj["files"].reverse()
    with pytest.raises(InvalidFieldError, match="out of order") as exc:
        s.dataset_from_dict(obj)
    assert exc.value.path == "files[1].path"
    obj = _obj()
    obj["files"] = [obj["files"][0], obj["files"][0]]
    with pytest.raises(InvalidFieldError, match="repeated"):
        s.dataset_from_dict(_with_content(obj))


def test_file_order_is_utf16_code_units_like_jcs_keys():
    # U+FF61 sorts after U+1F600 by code point, before it by UTF-16 unit.
    files = [
        s.DatasetFile("\U0001f600.csv", "1" * 64, 1, None),
        s.DatasetFile("｡.csv", "2" * 64, 1, None),
    ]
    obj = _obj()
    obj.update(splits=[], files=[dataclasses.asdict(f) for f in files])
    s.dataset_from_dict(_with_content(obj))
    obj["files"].reverse()
    with pytest.raises(InvalidFieldError, match="out of order"):
        s.dataset_from_dict(_with_content(obj))


@pytest.mark.parametrize(
    "bad", ["/abs.csv", "../up.csv", "a/../b.csv", "a//b.csv", "./a.csv", "attestation/x.csv", ""]
)
def test_file_paths_are_relative_and_outside_attestation(bad):
    obj = _obj()
    obj["files"][0]["path"] = bad
    with pytest.raises(InvalidFieldError) as exc:
        s.dataset_from_dict(_with_content(obj))
    assert exc.value.path == "files[0].path"


def test_attestation_dot_something_is_an_ordinary_name():
    obj = _obj()
    obj["files"][0]["path"] = "attestation.csv"
    s.dataset_from_dict(_with_content(obj))


@pytest.mark.parametrize(
    "field, value",
    [("bytes", -1), ("bytes", 1.5), ("bytes", True), ("media_type", "Text/CSV"),
     ("media_type", "text/csv; charset=utf-8"), ("media_type", "csv"), ("sha256", "AB" * 32)],
)  # fmt: skip
def test_file_fields_are_checked(field, value):
    obj = _obj()
    obj["files"][0][field] = value
    with pytest.raises(InvalidFieldError) as exc:
        s.dataset_from_dict(_with_content(obj) if field != "sha256" else obj)
    assert exc.value.path == f"files[0].{field}"


def test_a_dataset_has_at_least_one_file():
    obj = _obj()
    obj.update(files=[], splits=[])
    with pytest.raises(InvalidFieldError, match="at least one file"):
        s.dataset_from_dict(_with_content(obj))


def test_media_type_null_means_not_stated():
    obj = _obj()
    obj["files"][0]["media_type"] = None
    assert s.dataset_from_dict(obj).files[0].media_type is None


# ---- splits, source runs, columns, rows --------------------------------------------------


def test_a_split_is_one_of_the_files_with_its_digest():
    obj = _obj()
    obj["splits"][0]["path"] = "elsewhere.json"
    with pytest.raises(InvalidFieldError, match="not one of the files"):
        s.dataset_from_dict(obj)
    obj = _obj()
    obj["splits"][0]["sha256"] = "e" * 64
    with pytest.raises(InvalidFieldError) as exc:
        s.dataset_from_dict(obj)
    assert exc.value.path == "splits[0].sha256"


@pytest.mark.parametrize("seed, ok", [(0, True), ([1, 2], True), (None, True), ([], False),
                                      ("0", False), ([1, "2"], False), (2**60, False)])  # fmt: skip
def test_split_seeds(seed, ok):
    obj = _obj()
    obj["splits"][0]["seed"] = seed
    if ok:
        assert s.dataset_from_dict(obj).splits[0].seed == seed
    else:
        with pytest.raises(InvalidFieldError):
            s.dataset_from_dict(obj)


def test_split_names_are_unique():
    obj = _obj()
    obj["splits"].append(dict(obj["splits"][0]))
    with pytest.raises(InvalidFieldError, match="duplicate split name"):
        s.dataset_from_dict(obj)


def test_source_runs_null_vs_empty():
    obj = _obj()
    obj["source_runs"] = None
    assert s.dataset_from_dict(obj).source_runs is None
    obj["source_runs"] = []
    assert s.dataset_from_dict(obj).source_runs == []


def test_source_runs_are_unique_and_a_digest_appears_once():
    obj = _obj()
    obj["source_runs"].append(dict(obj["source_runs"][0]))
    with pytest.raises(InvalidFieldError, match="duplicate source run"):
        s.dataset_from_dict(obj)
    obj = _obj()
    obj["source_runs"][1]["record_sha256"] = obj["source_runs"][0]["record_sha256"]
    with pytest.raises(InvalidFieldError, match="duplicate digest"):
        s.dataset_from_dict(obj)
    obj = _obj()  # two undigested runs are fine
    for sr in obj["source_runs"]:
        sr["record_sha256"] = None
    assert [r.record_sha256 for r in s.dataset_from_dict(obj).source_runs] == [None, None]


def test_columns_null_vs_empty_and_unique_names():
    obj = _obj()
    obj["columns"] = None
    assert s.dataset_from_dict(obj).columns is None
    obj["columns"] = []
    assert s.dataset_from_dict(obj).columns == []
    obj = _obj()
    obj["columns"].append(dict(obj["columns"][0]))
    with pytest.raises(InvalidFieldError, match="duplicate column name"):
        s.dataset_from_dict(obj)


@pytest.mark.parametrize("rows, ok", [(None, True), (0, True), (-1, False), (1.0, False)])
def test_rows(rows, ok):
    obj = _obj()
    obj["rows"] = rows
    if ok:
        assert s.dataset_from_dict(obj).rows == rows
    else:
        with pytest.raises(InvalidFieldError):
            s.dataset_from_dict(obj)


def test_created_at_is_the_fixed_width_spelling():
    obj = _obj()
    obj["created_at"] = "2026-11-19T10:42:13Z"
    with pytest.raises(InvalidFieldError) as exc:
        s.dataset_from_dict(obj)
    assert exc.value.path == "created_at"
    obj["created_at"] = None
    assert s.dataset_from_dict(obj).created_at is None


# ---- license -------------------------------------------------------------------------------

VALID_LICENSES = [
    "MIT",
    "CC-BY-4.0",
    "MIT OR Apache-2.0",
    "(MIT OR Apache-2.0) AND CC-BY-4.0",
    "GPL-2.0+ WITH Classpath-exception-2.0",
    "GPL-2.0-or-later WITH Classpath-exception-2.0 OR MIT",
    "LicenseRef-proprietary",
    "DocumentRef-spdx-tool-1.2:LicenseRef-MIT-Style-2",
    "((MIT))",
]
INVALID_LICENSES = [
    "NOASSERTION",
    "NONE",
    "MIT OR NONE",
    "MIT and Apache-2.0",
    "MIT OR",
    "(MIT",
    "MIT)",
    "AND",
    "MIT WITH",
    "MIT WITH (x)",
    " MIT",
    "MIT,Apache-2.0",
    "all rights reserved",
]


@pytest.mark.parametrize("expr", VALID_LICENSES)
def test_spdx_expressions_are_accepted(expr):
    obj = _obj()
    obj["license"] = expr
    assert s.dataset_from_dict(obj).license == expr


@pytest.mark.parametrize("expr", INVALID_LICENSES)
def test_non_spdx_licenses_are_refused(expr):
    obj = _obj()
    obj["license"] = expr
    with pytest.raises(InvalidFieldError) as exc:
        s.dataset_from_dict(obj)
    assert exc.value.path == "license"


def test_license_null_means_not_stated():
    obj = _obj()
    obj["license"] = None
    assert s.dataset_from_dict(obj).license is None


# ---- credentials, ext ----------------------------------------------------------------------

_USERINFO = "https://ci:" + "hunter2@example.invalid/hub"


@pytest.mark.parametrize(
    "mutate, path",
    [
        (lambda o: o.update(ext={"hub": {"endpoint": _USERINFO}}), "ext.hub.endpoint"),
        (
            lambda o: o.update(ext={"hub": {"mirrors": ["x", "see " + _USERINFO]}}),
            "ext.hub.mirrors[1]",
        ),
        (lambda o: o.update(name="from " + _USERINFO), "name"),
        (
            lambda o: o["columns"][0].update(type="git+https://tok" + "en@example.invalid/x"),
            "columns[0].type",
        ),
        (lambda o: o["code"].update(repo=_USERINFO), "code.repo"),
    ],
)  # fmt: skip
def test_a_url_with_credentials_is_refused_in_any_string(mutate, path):
    obj = _obj()
    mutate(obj)
    with pytest.raises(InvalidFieldError, match="credentials") as exc:
        s.dataset_from_dict(obj)
    assert exc.value.path == path


@pytest.mark.parametrize(
    "text", ["ssh://git@example.invalid/x.git", "https://example.invalid/x?page=2", "http://[::1"]
)
def test_urls_without_credentials_are_fine(text):
    obj = _obj()
    obj["ext"] = {"hub": {"u": text}}
    s.dataset_from_dict(obj)


def test_ext_namespaces_follow_run_1():
    obj = _obj()
    obj["ext"] = {"attestation.x": {}}
    with pytest.raises(InvalidFieldError, match="reserved"):
        s.dataset_from_dict(obj)
    obj["ext"] = {"tool": []}
    with pytest.raises(InvalidFieldError, match="must be an object"):
        s.dataset_from_dict(obj)


# ---- dispatch ------------------------------------------------------------------------------


def test_a_run_is_not_a_dataset_and_a_dataset_is_not_a_run():
    run = json.loads((LINEAGE / "training" / layout.RUN_JSON).read_bytes())
    with pytest.raises(UnsupportedSchemaError, match="not a dataset record"):
        s.dataset_from_dict(run)
    with pytest.raises(UnsupportedSchemaError, match="not a run record"):
        s.from_dict(_obj())


def test_an_unknown_version_names_the_known_ones():
    obj = _obj()
    obj["schema"] = "attestation.dataset/2"
    with pytest.raises(UnsupportedSchemaError, match=r"\['attestation.dataset/1'\]"):
        s.dataset_from_dict(obj)
    del obj["schema"]
    with pytest.raises(MissingFieldError):
        s.dataset_from_dict(obj)


def test_a_registered_reader_is_used(monkeypatch):
    seen = []
    original = s.DATASET_READERS[s.DATASET_SCHEMA]
    monkeypatch.setitem(
        s.DATASET_READERS,
        "attestation.dataset/9",
        lambda o: seen.append(o) or original(dict(o, schema=s.DATASET_SCHEMA)),
    )
    s.dataset_from_dict(dict(_obj(), schema="attestation.dataset/9"))
    assert len(seen) == 1


def test_parsing_is_strict():
    raw = (FULL / layout.DATASET_JSON).read_bytes()
    assert b'"rows":4' in raw
    with pytest.raises(InvalidFieldError, match="duplicate"):
        s.loads_dataset(b'{"rows":4,' + raw[1:])
    with pytest.raises(InvalidFieldError, match="not a JSON number"):
        s.loads_dataset(raw.replace(b'"rows":4', b'"rows":NaN'))


# ---- writer --------------------------------------------------------------------------------


def _fresh(tmp_path: pathlib.Path) -> tuple[pathlib.Path, s.DatasetRecord]:
    dd = tmp_path / "ds"
    shutil.copytree(FULL, dd)
    record = s.read_dataset(dd)
    (dd / layout.DATASET_JSON).unlink()
    return dd, record


def test_write_dataset_writes_the_canonical_bytes(tmp_path):
    dd, record = _fresh(tmp_path)
    path = s.write_dataset(record, dd)
    assert path.read_bytes() == (FULL / layout.DATASET_JSON).read_bytes()
    with pytest.raises(FileExistsError):
        s.write_dataset(record, dd)
    s.write_dataset(record, dd, overwrite=True)


def test_write_dataset_needs_the_manifest_it_names(tmp_path):
    dd, record = _fresh(tmp_path)
    (dd / layout.DIGESTS_JSON).unlink()
    with pytest.raises(s.DatasetManifestError, match="missing"):
        s.write_dataset(record, dd)
    (dd / "extra.txt").write_text("x")
    s.write_dataset_manifest(dd)
    with pytest.raises(s.DatasetManifestError, match="content_sha256"):
        s.write_dataset(record, dd)
    assert not (dd / layout.DATASET_JSON).exists()


def test_a_run_directory_is_not_a_dataset_directory(tmp_path):
    dd, record = _fresh(tmp_path)
    shutil.copy(LINEAGE / "training" / layout.RUN_JSON, dd / layout.RUN_JSON)
    with pytest.raises(layout.LayoutError, match="run directory"):
        s.write_dataset(record, dd)


def test_the_manifest_is_written_before_the_record(tmp_path):
    dd = tmp_path / "ds"
    shutil.copytree(FULL, dd)
    with pytest.raises(layout.LayoutError, match="before the record"):
        s.write_dataset_manifest(dd)


def test_dataset_files_sizes_media_types_and_symlinks(tmp_path):
    dd = tmp_path / "ds"
    (dd / "sub").mkdir(parents=True)
    (dd / "b.parquet").write_bytes(b"PAR1....PAR1")
    (dd / "sub" / "a.weird").write_bytes(b"x")
    (dd / "attestation").mkdir()
    (dd / "attestation" / "ignored.txt").write_text("not data")
    files = s.dataset_files(dd, media_types={"b.parquet": None})
    assert [(f.path, f.bytes, f.media_type) for f in files] == [
        ("b.parquet", 12, None),
        ("sub/a.weird", 1, None),
    ]
    assert s.dataset_files(dd)[0].media_type == "application/vnd.apache.parquet"
    assert s.suggest_media_type("x/Y.CSV") == "text/csv"
    os.symlink(dd / "b.parquet", dd / "link.parquet")
    with pytest.raises(layout.LayoutError, match="symlink"):
        s.dataset_files(dd)


def test_dataset_lineage_entries():
    record = s.read_dataset(FULL)
    entries = s.dataset_lineage_entries(record, split="scaffold")
    assert [(e.name, e.sha256, e.role) for e in entries] == [
        (record.dataset_id, record.content_sha256, "train"),
        (f"{record.dataset_id}/split-scaffold.json", record.splits[0].sha256, "split"),
    ]
    assert len(s.dataset_lineage_entries(record, role="eval")) == 1
    with pytest.raises(KeyError):
        s.dataset_lineage_entries(record, split="random")


def test_draft_status():
    assert s.DATASET_SCHEMA_STATUS == "draft"
    assert s.DATASET_SCHEMA == "attestation.dataset/1"


# ---- the JSON Schema file ------------------------------------------------------------------

SCHEMA_REFUSES = [
    lambda o: o.pop("license"),
    lambda o: o.update(extra=1),
    lambda o: o.update(content_sha256="x"),
    lambda o: o["files"][0].update(media_type="Text/CSV"),
    lambda o: o["files"][0].update(bytes=-1),
    lambda o: o["files"][0].update(path="../x"),
    lambda o: o["files"][0].update(path="attestation/x"),
    lambda o: o.update(files=[]),
    lambda o: o.update(license="NOASSERTION"),
    lambda o: o.update(created_at="2026-11-19T10:42:13Z"),
    lambda o: o["splits"][0].update(seed=[]),
    lambda o: o["source_runs"][0].update(record_sha256="nope"),
    lambda o: o.update(ext={"attestation": {}}),
    lambda o: o.update(schema="attestation.run/1"),
]
# Rules only the reader can state: the schema accepts these, the reader refuses.
READER_ONLY = [
    lambda o: o["files"][0].update(sha256="0" * 64),  # content digest
    lambda o: o["files"].reverse(),  # order
    lambda o: o["splits"][0].update(sha256="e" * 64),  # split vs file
    lambda o: o.update(license="MIT and"),  # SPDX syntax
    lambda o: o.update(name=_USERINFO),  # credentials
]


@pytest.fixture(scope="module")
def validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(DATASET_SCHEMA_FILE.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_the_json_schema_accepts_every_valid_record(validator):
    for d in [FULL, DATA / "datasets" / "valid" / "minimal"]:
        validator.validate(json.loads((d / layout.DATASET_JSON).read_bytes()))


@pytest.mark.parametrize("mutate", SCHEMA_REFUSES)
def test_the_json_schema_and_the_reader_both_refuse(validator, mutate):
    obj = copy.deepcopy(_obj())
    mutate(obj)
    assert list(validator.iter_errors(obj)), "the JSON Schema accepted it"
    with pytest.raises(s.RecordError):
        s.dataset_from_dict(obj)


@pytest.mark.parametrize("mutate", READER_ONLY)
def test_rules_only_the_reader_can_state(validator, mutate):
    obj = copy.deepcopy(_obj())
    mutate(obj)
    assert not list(validator.iter_errors(obj))
    with pytest.raises(s.RecordError):
        s.dataset_from_dict(obj)
