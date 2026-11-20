"""The one dependency-identity rule, credential stripping, git state, and the
shared durable write (attestation-schema 0.2.0).

Each test under "the rule" pins one decision of
docs/superpowers/specs/2026-11-19-dependency-identity-design.md, where the two
implementations it replaced (smeltery's and the callbacks') disagreed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import attestation_schema as s
import pytest
from attestation_schema import files, identity, layout, record
from attestation_schema.identity import GitState, dependency_identity, strip_credentials

SHA = "a" * 40
OTHER = "b" * 40
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _dep(
    module: str | None = "1.2.0",
    dist: str | None = "1.2.0",
    direct_url=None,
    installer: str | None = "pip",
    **kw,
) -> s.Dependency:
    return dependency_identity(
        "pkg",
        module_version=module,
        dist_version=dist,
        direct_url=direct_url,
        installer=installer,
        **kw,
    )


def _reads(*deps: s.Dependency) -> None:
    """Every shape the rule produces is one the frozen reader accepts."""
    named = [s.Dependency(**{**d.__dict__, "name": f"d{i}"}) for i, d in enumerate(deps)]
    rec = s.RunRecord(
        run_id="r",
        family=None,
        arm=None,
        status="complete",
        started_at=None,
        finished_at=None,
        producer=s.Producer("t", None),
        code=s.UNKNOWN_CODE,
        config={},
        data=[],
        metrics=[],
        history=None,
        seeds=None,
        dependencies=named,
        ext={},
    )
    assert s.loads(s.dumps(rec)) == rec


# ---- the package version ------------------------------------------------------------


def test_version_is_public_and_matches_pyproject():
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert s.__version__ == declared
    assert s._FALLBACK_VERSION == declared  # the source-tree fallback cannot rot


# ---- the rule: index installs (decisions 1, 2, 4) -------------------------------------


def test_an_index_install_the_module_agrees_with_is_verified():
    d = _dep()
    assert (d.version, d.provenance, d.source, d.build_sha, d.dirty) == (
        "1.2.0",
        "verified",
        "index",
        None,
        None,  # no build_sha: nothing for a tree to differ from
    )
    assert _dep(installer="uv\n").source == "index"


def test_an_index_install_without_a_module_version_is_inferred():
    # smeltery called this VERIFIED (index releases are immutable); but nothing
    # showed this process loaded that release
    d = _dep(module=None)
    assert (d.version, d.provenance, d.source) == ("1.2.0", "inferred", "index")
    assert "no module __version__" in d.evidence


def test_versions_that_disagree_are_inferred_with_both_in_evidence():
    d = _dep(module="1.3.0")
    assert (d.version, d.provenance) == ("1.3.0", "inferred")
    assert "1.3.0" in d.evidence and "1.2.0" in d.evidence
    assert _dep(module="1.2.0+a", dist="1.2.0+b").provenance == "inferred"
    assert _dep(module=None, dist=None).provenance == "unknown"


def test_a_local_version_label_the_module_adds_still_agrees():
    # measured: PyPI's torch 2.15.0 wheel says 2.15.0 in METADATA, 2.15.0+cu132 at runtime
    d = _dep(module="2.15.0+cu132", dist="2.15.0")
    assert (d.version, d.provenance) == ("2.15.0+cu132", "verified")
    assert "local label" in d.evidence


def test_a_missing_installer_file_is_no_source_and_never_verified():
    # smeltery read a missing INSTALLER as an index install; `setup.py develop`
    # and hand-copied trees write none either
    d = _dep(installer=None)
    assert (d.source, d.provenance) == (None, "inferred")
    assert "no INSTALLER" in d.evidence
    assert _dep(installer="  ").source is None


def test_the_conda_family_is_conda_and_inferred():
    for tool in ("conda", "mamba", "micromamba", "Micromamba\n"):
        d = _dep(installer=tool)
        assert (d.source, d.provenance) == ("conda", "inferred"), tool


def test_distribution_managers_are_system_and_other_installers_say_nothing():
    assert (_dep(installer="rpm").source, _dep(installer="rpm").provenance) == (
        "system",
        "inferred",
    )
    assert _dep(installer="debian").source == "system"
    # smeltery called every other installer `system`; poetry installs from an index
    other = _dep(installer="poetry")
    assert (other.source, other.provenance) == (None, "inferred")
    assert "INSTALLER=poetry" in other.evidence


# ---- the rule: direct_url.json (decisions 5, 6) ---------------------------------------


def test_a_vcs_install_is_verified_by_its_commit_and_clean():
    d = _dep(
        direct_url={
            "url": "https://user:tok@example.org/pkg.git",
            "vcs_info": {"vcs": "git", "commit_id": SHA},
        }
    )
    assert (d.source, d.build_sha, d.dirty, d.provenance) == (
        "git+https://example.org/pkg.git",
        SHA,
        False,
        "verified",
    )
    prefixed = _dep(direct_url={"url": "git+ssh://git@h/p", "vcs_info": {"vcs": "git"}})
    assert prefixed.source == "git+ssh://git@h/p"  # not git+git+...; ssh user is no secret


def test_a_vcs_install_without_a_commit_names_its_source_but_is_inferred():
    d = _dep(direct_url={"url": "https://h/p", "vcs_info": {"vcs": "hg", "commit_id": "xyz"}})
    assert (d.source, d.build_sha, d.provenance) == ("hg+https://h/p", None, "inferred")
    assert "names no commit" in d.evidence
    bare = _dep(direct_url={"url": "https://h/p", "vcs_info": {"vcs": "git"}})
    assert (bare.source, bare.provenance) == ("git+https://h/p", "inferred")


def test_an_unknown_vcs_has_no_source():
    d = _dep(direct_url={"url": "https://h/p", "vcs_info": {"vcs": "fossil", "commit_id": SHA}})
    assert (d.source, d.build_sha, d.provenance) == (None, None, "inferred")


def test_an_archive_install_is_inferred_with_its_hash_as_evidence():
    d = _dep(
        direct_url={
            "url": "https://u:p@host/pkg-1.2.0.whl?X-Amz-Signature=abc#sha256=00",
            "archive_info": {"hashes": {"sha256": "00"}},
        }
    )
    assert d.source == "url:https://host/pkg-1.2.0.whl#sha256=00"
    assert d.provenance == "inferred" and "(00)" in d.evidence
    local = _dep(direct_url={"url": "file:///w/pkg.whl", "archive_info": {}})
    assert (local.source, local.provenance) == ("file", "inferred")
    assert "no archive hash" in local.evidence


def test_a_directory_install_takes_the_git_probe_and_stays_inferred():
    for editable in (True, False):
        du = {"url": "file:///src/pkg", "dir_info": {"editable": editable}}
        probed = _dep(direct_url=du, git=GitState(SHA, True, None))
        assert (probed.source, probed.build_sha, probed.dirty, probed.provenance) == (
            "file",
            SHA,
            True,
            "inferred",
        )
        assert "now, not at install" in probed.evidence
        bare = _dep(direct_url=du, git=GitState(None, None, None))
        assert (bare.source, bare.build_sha, bare.dirty, bare.provenance) == (
            "file",
            None,
            None,
            "inferred",
        )


# ---- the rule: the build stamp (decision 7) --------------------------------------------


def test_a_build_stamp_is_verified_and_overrides_everything_else():
    d = _dep(module="0.1.0", dist="0.1.0", installer=None, stamp=(SHA, True), stamp_from="ferric")
    assert (d.build_sha, d.dirty, d.provenance, d.source) == (SHA, True, "verified", None)
    assert d.evidence.startswith("ferric.__build__")
    vcs = {"url": "https://h/f", "vcs_info": {"vcs": "git", "commit_id": OTHER}}
    both = _dep(direct_url=vcs, stamp=(SHA, False))
    assert (both.build_sha, both.source) == (SHA, "git+https://h/f")  # source still observed
    assert f"stamp overrides {OTHER}" in both.evidence


def test_build_stamp_takes_only_the_stamp_shape():
    def mod(stamp):
        m = ModuleType("m")
        m.__build__ = stamp
        return m

    assert identity.build_stamp(mod({"git_sha": SHA, "dirty": False})) == (SHA, False)
    assert identity.build_stamp(mod({"git_sha": SHA, "dirty": True, "x": 1})) == (SHA, True)
    for bad in (
        None,  # ferric: the build had no git metadata
        {"git_sha": SHA},  # no dirty flag: never guess one
        {"git_sha": SHA, "dirty": "no"},
        {"git_sha": SHA.upper(), "dirty": False},
        {"git_sha": "abc123", "dirty": False},
        (SHA, False),
    ):
        assert identity.build_stamp(mod(bad)) is None, bad
    assert identity.build_stamp(ModuleType("plain")) is None


def test_every_shape_the_rule_produces_is_one_the_reader_accepts():
    _reads(
        _dep(),
        _dep(module=None),
        _dep(module="1.3.0"),
        _dep(module=None, dist=None),
        _dep(installer=None),
        _dep(installer="mamba"),
        _dep(installer="rpm"),
        _dep(installer="poetry"),
        _dep(direct_url={"url": "https://t@e/p.git", "vcs_info": {"vcs": "git", "commit_id": SHA}}),
        _dep(direct_url={"url": "https://e/p", "vcs_info": {"vcs": "svn", "commit_id": "1234"}}),
        _dep(direct_url={"url": "https://e/p", "vcs_info": {"vcs": "fossil"}}),
        _dep(direct_url={"url": "https://u:p@e/p.whl?q=1#sha256=ab", "archive_info": {}}),
        _dep(direct_url={"url": "not a url", "archive_info": {}}),
        _dep(direct_url={"url": "file:///p", "dir_info": {}}, git=GitState(SHA, False, None)),
        _dep(direct_url={"url": "file:///p", "dir_info": {}}),
        _dep(direct_url={"url": "x"}),
        _dep(stamp=(SHA, True)),
        _dep(module=None, dist=None, stamp=(SHA, False)),
    )


# ---- package_identity, observed --------------------------------------------------------


def _install(
    site: Path, dist: str, version: str, *, installer=None, direct_url=None, module_src=None
):
    """A real installed distribution under `site`: dist-info plus a module."""
    info = site / f"{dist.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n")
    if installer is not None:
        (info / "INSTALLER").write_text(installer + "\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))
    if module_src is not None:
        (site / f"{dist.replace('-', '_')}.py").write_text(module_src)


@pytest.fixture
def site(tmp_path, monkeypatch):
    path = tmp_path / "site"
    path.mkdir()
    monkeypatch.syspath_prepend(str(path))
    importlib.invalidate_caches()
    yield path
    for name in [m for m in sys.modules if m.startswith("idfake_")]:
        del sys.modules[name]


def test_package_identity_reads_a_real_index_install(site):
    _install(site, "idfake-a", "2.0.0", installer="pip", module_src="__version__ = '2.0.0'\n")
    d = identity.package_identity("idfake-a", "idfake_a")
    assert (d.name, d.version, d.source, d.provenance) == ("idfake-a", "2.0.0", "index", "verified")
    meta_only = identity.package_identity("idfake-a", name="a")
    assert (meta_only.name, meta_only.provenance) == ("a", "inferred")
    assert "metadata only" in meta_only.evidence
    _reads(d, meta_only)


def test_package_identity_reads_a_build_stamp(site):
    stamp = f"__version__ = '0.1.0'\n__build__ = {{'git_sha': '{SHA}', 'dirty': False}}\n"
    _install(site, "idfake-b", "0.1.0", installer="pip", module_src=stamp)
    mod = importlib.import_module("idfake_b")
    d = identity.package_identity("idfake-b", mod)
    assert (d.build_sha, d.dirty, d.provenance, d.source) == (SHA, False, "verified", "index")
    assert d.evidence.startswith("idfake_b.__build__")


def test_package_identity_never_raises_for_what_is_missing(site):
    d = identity.package_identity("idfake-none", "idfake_none_module")
    assert (d.version, d.provenance, d.source) == (None, "unknown", None)
    assert "not importable" in d.evidence and "no installed distribution" in d.evidence
    _install(site, "idfake-c", "1.0", direct_url={"not": "a kind"})
    (site / "idfake_c-1.0.dist-info" / "direct_url.json").write_text("{broken")
    assert identity.package_identity("idfake-c").source is None  # unparseable: ignored


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(path: Path, remote: str | None = None) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.org")
    _git(path, "config", "user.name", "t")
    if remote:
        _git(path, "remote", "add", "origin", remote)
    (path / "a.py").write_text("x = 1\n")
    _git(path, "add", "a.py")
    _git(path, "commit", "-q", "-m", "init")
    return path


@needs_git
def test_package_identity_probes_a_directory_install(site, tmp_path):
    src = _repo(tmp_path / "src")
    url = src.resolve().as_uri()
    _install(site, "idfake-d", "0.3.0", installer="uv", direct_url={"url": url, "dir_info": {}})
    d = identity.package_identity("idfake-d")
    assert (d.source, d.build_sha, d.dirty, d.provenance) == (
        "file",
        _git(src, "rev-parse", "HEAD"),
        False,
        "inferred",
    )
    (src / "a.py").write_text("x = 2\n")
    assert identity.package_identity("idfake-d").dirty is True
    assert identity.package_identity("idfake-d", probe_git=False).build_sha is None


@needs_git
def test_git_state(tmp_path):
    repo = _repo(tmp_path / "r", remote="https://bob:s3cret@example.org/x.git")
    st = identity.git_state(repo)
    assert st == GitState(_git(repo, "rev-parse", "HEAD"), False, "https://example.org/x.git")
    (repo / "untracked.txt").write_text("not the code")
    assert identity.git_state(repo).dirty is False
    assert identity.git_state(tmp_path / "nowhere") == GitState(None, None, None)


# ---- strip_credentials == the reader's rule (audit item 2) -----------------------------

URLS = [
    "https://h/x",
    "https://u:p@h/x",
    "https://tok@h/x",
    "https://:p@h/x",
    "http://u@h:8080/x?y=1#z",
    "git+https://tok@github.com/o/r.git",
    "git+https://github.com/o/r.git",
    "git+ssh://git@github.com/o/r.git",
    "git+ssh://u:p@github.com/o/r.git",
    "ssh://git@github.com/o/r.git",
    "ssh://u:p@github.com/o/r.git",
    "svn+svn://u:p@h/r",
    "hg+https://u@h/r",
    "git@github.com:o/r.git",
    "file:///home/x/repo",
    "ftp://anonymous:me@h/p",
    "https://u:p@w@h/x",
    "https://[::1/x",
]


def _reader_accepts_repo(url: str) -> bool:
    try:
        record._repo(url, "repo")
    except s.RecordError:
        return False
    except ValueError:
        return False  # urlsplit refuses it: the reader raises, it does not accept
    return True


def _reader_accepts_source(source: str) -> bool:
    try:
        record._source(source, "source")
    except s.RecordError:
        return False
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("url", URLS)
def test_strip_credentials_strips_exactly_what_the_reader_refuses(url):
    stripped = strip_credentials(url)
    if url == "https://[::1/x":  # not a URL at all: left as is, and the reader refuses it
        assert stripped == url and not _reader_accepts_repo(url)
        return
    assert _reader_accepts_repo(stripped), stripped
    assert (stripped == url) == _reader_accepts_repo(url), (url, stripped)
    assert strip_credentials(stripped) == stripped


SOURCES = [
    "index",
    "index:https://tok@pypi.example/simple",
    "index:internal",
    "url:https://u:p@h/p.whl?X-Amz-Signature=s#sha256=ab",
    "url:https://h/p.whl#sha256=ab",
    "url:https://h/p.whl?token=1",
    "conda:https://u:p@conda.example/c",
    "conda:conda-forge",
    "git+https://tok@h/r",
    "git+ssh://git@h/r",
    "file",
    "system",
]


@pytest.mark.parametrize("source", SOURCES)
def test_strip_credentials_on_a_source_matches_the_reader(source):
    stripped = strip_credentials(source)
    assert _reader_accepts_source(stripped), stripped
    assert (stripped == source) == _reader_accepts_source(source), (source, stripped)
    if source.startswith("url:"):
        assert "?" not in stripped and stripped.endswith("#sha256=ab") == ("#sha256" in source)


def test_strip_credentials_named_cases():
    assert strip_credentials("https://u:p@h/x") == "https://h/x"
    assert strip_credentials("git+https://tok@h/r") == "git+https://h/r"
    assert strip_credentials("ssh://u:p@h/r") == "ssh://h/r"
    assert strip_credentials("ssh://git@h/r") == "ssh://git@h/r"
    assert strip_credentials("git@github.com:o/r.git") == "git@github.com:o/r.git"
    assert (
        strip_credentials("url:https://h/p.whl?sig=1#sha256=ab") == "url:https://h/p.whl#sha256=ab"
    )


# ---- the shared write and digest (audit items 3, 5, 6) ---------------------------------


def test_sha256_file_is_chunked_and_exact(tmp_path, monkeypatch):
    f = tmp_path / "f"
    data = os.urandom(3 * (1 << 20) + 7)
    f.write_bytes(data)
    reads: list[int] = []
    real_open = open

    def spy(path, mode="r", *a, **kw):
        fh = real_open(path, mode, *a, **kw)
        real_read = fh.read

        class Wrapped:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fh.close()

            def read(self, n=-1):
                reads.append(n)
                return real_read(n)

        return Wrapped()

    monkeypatch.setattr(files, "open", spy, raising=False)
    assert s.sha256_file(f) == hashlib.sha256(data).hexdigest()
    assert reads and all(0 < n <= 1 << 20 for n in reads)


def test_atomic_write_exclusive_refuses_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "d" / "x.json"
    files._atomic_write(target, b"one", exclusive=True)
    with pytest.raises(FileExistsError):
        files._atomic_write(target, b"two", exclusive=True)
    assert target.read_bytes() == b"one"
    files._atomic_write(target, b"three", exclusive=False)
    assert target.read_bytes() == b"three"
    assert sorted(p.name for p in target.parent.iterdir()) == ["x.json"]


def test_every_writer_fsyncs(tmp_path, monkeypatch):
    """The manifest writers did not fsync before 0.2.0 while run.json did, so a
    crash could keep a record and lose the manifest it binds (audit item 3)."""
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    run = tmp_path / "run"
    (run / "out.txt").parent.mkdir(parents=True)
    (run / "out.txt").write_text("x")
    for write in (
        lambda: layout.write_manifest(run),
        lambda: s.write_dataset_manifest(tmp_path / "ds"),
        lambda: s.init_state(run, run_id="r", backend="local", job_id="1"),
    ):
        (tmp_path / "ds").mkdir(exist_ok=True)
        before = len(synced)
        write()
        assert len(synced) >= before + 1


def test_manifest_bytes_are_unchanged_by_the_shared_writer(tmp_path):
    run = tmp_path / "run"
    (run / "steps" / "00-a").mkdir(parents=True)
    (run / "steps" / "00-a" / "o.txt").write_text("hello")
    entry = layout.write_manifest(run)
    expected = (
        b'{"files":{"steps/00-a/o.txt":"'
        + hashlib.sha256(b"hello").hexdigest().encode()
        + b'"},"schema":"attestation.digests/1"}'
    )
    assert (run / layout.DIGESTS_JSON).read_bytes() == expected
    assert entry.sha256 == hashlib.sha256(expected).hexdigest()
