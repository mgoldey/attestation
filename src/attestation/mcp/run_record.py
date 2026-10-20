"""Signed run records over MCP: `runs.attest` and `runs.verify_record`.

Under the `runs` namespace, so the provenance surface -- the one whose
answers reach manuscripts -- serves them. Its own module rather than more
lines in provenance.py: the record domain (`attestation.run_record`) shares
nothing with the ledger but the namespace.

`cite.check(record=...)` writes its record through `cite_check_record` here,
so citation.py carries one call rather than the recording logic.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from attestation.mcp._tool import ToolError, tool


class ModelUse(BaseModel):
    """A model the run called."""

    name: str
    role: str = "chat"
    endpoint: str | None = None


class Query(BaseModel):
    """One search the run made, against one source."""

    source: str
    query: str
    hits: Annotated[int, Field(ge=0)]


class Paper(BaseModel):
    """A paper the run read, and how much of it."""

    id: str
    depth: Literal["full", "abstract", "title"]
    title: str | None = None


class FileRef(BaseModel):
    """A file, by path relative to `root`; `role` for inputs (draft, results,
    bib, sources, other)."""

    path: str
    role: str = "other"


class Result(BaseModel):
    """One verdict inside a check."""

    subject: str
    verdict: str
    message: str = ""
    data: dict | None = None


class Check(BaseModel):
    """A checker's results. The summary is derived from them, never supplied."""

    check: str
    target: str | None = None
    results: list[Result] = []


class RecordSpec(BaseModel):
    """What a run did. Every file named is hashed when the record is signed."""

    tool: str
    started_at: str | None = None
    finished_at: str | None = None
    models: list[ModelUse] = []
    queries: list[Query] = []
    papers: list[Paper] = []
    inputs: list[FileRef] = []
    outputs: list[FileRef] = []
    checks: list[Check] = []


def _sign_and_write(recorder, out: str, force: bool) -> dict:
    from pathlib import Path

    from attestation.run_record import build, keys

    try:
        signer = keys.resolve_key_source().signer()
        envelope = recorder.sign(signer)
        path = build.write_record(envelope, Path(out), force=force)
    except (keys.RecordKeyError, FileExistsError, ValueError) as exc:
        raise ToolError(f"run record not written: {exc}") from exc
    return {"path": str(path), "keyid": signer.keyid, "run_id": recorder.run_id}


@tool(empty={"record": None}, needs_db=False, label="runs_attest")
def _attest(spec: RecordSpec, out: str, root: str | None = None, force: bool = False) -> dict:
    from attestation.run_record import build

    try:
        recorder = build.from_spec(spec.model_dump(exclude_none=True), root=root)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    written = _sign_and_write(recorder, out, force)
    return {"message": f"signed run record written to {written['path']}", "record": written}


@tool(empty={"verified": False, "items": [], "cannot_check": []}, needs_db=False)
def _verify(path: str, root: str | None = None, keyid: str | None = None) -> dict:
    from pathlib import Path

    from attestation.run_record import verify

    try:
        report = verify.verify_file(
            Path(path).expanduser(), root=Path(root).expanduser() if root else None, keyid=keyid
        ).to_dict()
    except (OSError, ValueError) as exc:
        raise ToolError(f"cannot read record {path}: {exc}") from exc
    failed = [i["what"] for i in report["items"] if i["status"] == verify.FAILED]
    return {
        "message": (
            "VERIFIED: nothing failed. That says what the tool concluded, not that it was right."
            if report["ok"]
            else "NOT VERIFIED: " + ", ".join(failed)
        ),
        "verified": report["ok"],
        "items": report["items"],
        "cannot_check": report["cannot_check"],
        "summary": report["summary"],
    }


def cite_check_record(root, found, uncited, out: str | None) -> dict:
    """`cite.check(record=out)`: sign a record of that check. {} when not asked."""
    if not out:
        return {}
    from attestation import citations
    from attestation.run_record import adapters

    try:
        recorder = adapters.record_citation_check(
            root, found, uncited, bib_paths=citations.bib_paths_from_env()
        )
    except ValueError as exc:
        raise ToolError(f"run record not written: {exc}") from exc
    return {"record": _sign_and_write(recorder, out, force=False)}


def register(mcp) -> None:
    """Attach the two run-record tools."""

    @mcp.tool(name="runs.attest")
    def runs_attest(
        spec: RecordSpec,
        out: Annotated[str, Field(description="where to write the record (.json)")],
        root: Annotated[
            str | None, Field(description="where spec paths resolve (default: server cwd)")
        ] = None,
        force: bool = False,
    ) -> dict:
        """Write a signed record of a research run: what it searched, read, checked and made.

        Hashes every input and output file, records each query with its hit
        count, each paper with the depth it was read at (full, abstract,
        title), the models used, and every check's verdicts, then signs it
        with this install's ed25519 key. The record is one in-toto/DSSE JSON
        file anyone can check with `runs.verify_record` or `attest record
        verify`, without this database. New files only unless `force`.
        """
        return _attest(spec, out, root, force)

    @mcp.tool(name="runs.verify_record")
    def runs_verify_record(
        path: Annotated[str, Field(description="the record file")],
        root: Annotated[
            str | None, Field(description="where its file names resolve (default: cwd)")
        ] = None,
        keyid: Annotated[str | None, Field(description="require this signer key id")] = None,
    ) -> dict:
        """Check a run record's signature, its check summaries, and the files it names.

        Each item is ok, FAILED or not checked: a file missing under `root`
        is not checked, never a pass. `cannot_check` lists what no record can
        prove -- relay it, because a valid signature over a `supported`
        verdict means the tool said so, not that the claim is true.
        """
        return _verify(path, root, keyid)
