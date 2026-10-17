# Run records: a signed, portable account of one research run

**Date:** 2026-10-17
**Status:** roadmap step 5, built on `feat/run-record`. The default key (a
local ed25519 key per install) is implemented; whether records should instead
be signed by a per-researcher or per-organisation key is a pending decision,
recorded under "Open decisions".
**Depends on:** claims checking (`claims.py`), the citation lint
(`cite.check`, `citations.py`), the paper-research clients
(`2026-09-10-paper-research-design.md`), and AgentMarkit's brief scripts
(`gather_sources.py`, `check_brief.py`), which call the generic API below.

## What this is for

Every check Attestation runs today answers in a terminal or a tool reply and
then evaporates. A reviewer handed a draft cannot tell which papers the agent
actually read, what it searched for, which version of which tool produced
"12 supported, 1 contradicted", or whether the draft they hold is the one that
was checked. A run record is one file that says all of that, that anyone can
check without Attestation's database, and that breaks visibly if a verdict,
an input or the signer is changed afterwards.

It records what happened. It does not prove that what happened was right: a
`supported` verdict in a valid record means "this tool, on these inputs, said
supported", never "the claim is true". The verifier says so in its output.

## What a record holds

One **in-toto Statement v1** (`_type: https://in-toto.io/Statement/v1`) with
`predicateType: https://github.com/mgoldey/attestation/run-record/v1`:

| Field | Holds |
|---|---|
| `subject[]` | The artifacts the record is about, as in-toto ResourceDescriptors (`name`, `digest.sha256`): the run's outputs, or, for a check with no output, the documents it checked. Never empty (in-toto requires one). |
| `predicate.run` | `id` (uuid4), `started_at`, `finished_at` (UTC ISO 8601, the signer's clock). |
| `predicate.tool` | `name` (`attest claims`, `cite.check`, `research-brief`...), `attestation_version`, `git_sha` and `git_dirty` when Attestation runs from a checkout, else null. |
| `predicate.models[]` | `{name, role, endpoint?}` for every model the run called. Empty for the deterministic checks, which call none. |
| `predicate.queries[]` | `{source, query, hits}`: each search, per source (`openalex`, `pubmed`, `library`, `feed`...), with the number of hits it returned. |
| `predicate.papers[]` | `{id, title?, depth}` for each paper actually read, `depth` one of `full`, `abstract`, `title`. |
| `predicate.inputs[]` | ResourceDescriptors with `annotations.role` (`draft`, `results`, `bib`, `sources`, `other`) and `annotations.bytes`: a sha256 of every input file. |
| `predicate.checks[]` | Generic check results (below). |
| `predicate.outputs[]` | ResourceDescriptors for the files the run produced. |
| `predicate.signer` | `{alg: "ed25519", keyid, public_key}`. Inside the signed payload, so a swapped key breaks the signature. |

`name` is a path relative to the run's root (the working directory unless the
caller passes another), so a reviewer who unpacks the same tree anywhere can
recompute every digest.

### The generic check result

Every checker reports the same shape, so a new one (cite.support, a LaTeX
check, an agent's own) plugs in without a schema change:

```json
{"check": "claims", "target": "draft.md",
 "summary": {"supported": 11, "contradicted": 1},
 "results": [{"subject": "draft.md:42", "verdict": "contradicted",
              "message": "run says 0.061", "data": {"metric": "wer", "value": 0.053}}]}
```

`check` names the checker, `target` the input name it judged (optional),
`summary` is derived from `results` when the record is built (never supplied by
the caller, so it cannot disagree with them), and `data` is free-form JSON for
whatever the checker wants a reader to see. `verdict` is a free string: the
claims checker's six verdicts, `uncited` for cite.check, and whatever
cite.support defines, without this module knowing any of them.

## Format: DSSE envelope over a JCS-canonical in-toto Statement

**Chosen:** the record file is a **DSSE envelope** (`payloadType:
application/vnd.in-toto+json`, `payload` = base64 of the Statement,
`signatures: [{keyid, sig}]`), the payload being the Statement serialized as
**RFC 8785 JCS**. Signatures are ed25519 over DSSE's PAE encoding.

Why this and not the alternatives:

- **DSSE over a detached `.sig`.** The ask was "a JSON file plus a detached
  signature". DSSE *is* a detached signature over exact payload bytes -- it
  just travels in the same file. One file is what gets emailed, attached to a
  manuscript or uploaded with a dataset; a pair gets separated, and the half
  that survives is unverifiable. DSSE also binds the payload *type* into what
  is signed (PAE), so a signature over a record cannot be replayed as a
  signature over some other JSON. It is the envelope in-toto, SLSA, Sigstore
  and `cosign attest` use, so standard tooling reads it.
- **in-toto Statement over a home-grown wrapper.** The product is called
  Attestation; the industry's word for "signed statement about artifacts,
  identified by digest" is an in-toto attestation. Using its Statement layer
  (subject + predicateType + predicate) costs nothing and means a policy
  engine that understands in-toto can at least route these records.
- **A custom predicate over SLSA Provenance v1 or in-toto test-result.**
  SLSA Provenance describes a *build* (`buildDefinition`, `builder`); a
  research run's verdicts, queries and reading depth would all be stuffed
  into `byproducts`, and a SLSA verifier would then make claims about a build
  that never happened. `test-result` has only passed/warned/failed test
  names, which loses the verdict vocabulary that is this product's point.
  The predicate reuses in-toto's ResourceDescriptor for every file, so the
  part a generic tool cares about (what files, which digests) is standard.
- **JCS over "sorted keys, no whitespace".** DSSE signs bytes, so
  verification never needs canonical form. Canonical form buys stability:
  the same record content always produces the same bytes, hence the same
  digest, hence a record can be named by its hash and two builds compared
  byte for byte. `json.dumps(sort_keys=True, separators=(",", ":"))` is
  almost JCS but not quite: it sorts by code point where JCS sorts by UTF-16
  code unit, and it prints `1e-07`, `1.0` and `1e+16` where JCS (ES6 number
  formatting) prints `1e-7`, `1` and `10000000000000000`. A verifier written
  in another language would then disagree on the bytes. JCS is ~60 lines of
  stdlib Python here, tested against the RFC's number vectors. Integers
  outside +/-2^53 and non-finite floats are refused rather than rounded.

## Keys

**Default:** an ed25519 key per install, generated on first use at
`$HERMES_HOME/attestation/keys/run-record-ed25519.pem` (PKCS#8 PEM,
unencrypted), file mode `0600` in a `0700` directory, created with
`O_EXCL` so two processes racing on first use cannot both write one. A key
file readable by group or others is refused, the way `ssh` refuses one.
`keyid` is the hex sha256 of the raw 32-byte public key.

`cryptography` is added as a direct dependency. It was already in `uv.lock`
at 50.0.0 as a transitive dependency of the `mcp` stack, so every install
already has it: making it direct pins nothing new and adds no download.
Ed25519 over RSA/ECDSA: deterministic signatures (no nonce to get wrong),
32-byte keys that fit in the record, and the default of minisign, OpenSSH
and Sigstore's ed25519 path.

**Pluggable:** a `KeySource` is anything with `signer() -> Signer`. Two ship:

- `LocalKeySource(path)` -- the default; generates on first use.
- `ATTEST_RECORD_KEY=/path/key.pem` -- an existing PKCS#8 ed25519 key,
  never generated (a typo must not silently mint a new identity).

`attest record key` prints the public key and keyid; `--export pub.pem`
writes the public key for a collaborator.

**What a signature proves:** the public key is embedded in the record, so by
default verification proves *integrity* (nothing changed since signing) and
*consistency* (it was signed by the key it names), not *identity*. Identity
needs the verifier to already trust a key: `attest record verify --key
pub.pem` or `--keyid <hex>` pins it, and a record signed by any other key
fails.

## Verifying

`attest record verify <file> [--root DIR] [--key pub.pem | --keyid HEX]`
(also `python -m attestation.run_record.verify`, and the `runs.verify_record`
tool). `verify.py` imports only the standard library, `cryptography`, and
`canonical.py` -- no database, no model, no `attestation.db` -- so a reviewer
can run it from a bare install.

It reports, per item, one of `ok`, `FAILED`, or `not checked (why)`:

1. The envelope parses and its `payloadType` is in-toto's.
2. The signature verifies against the embedded key (and the pinned key, if
   given).
3. The payload is in canonical (JCS) form -- a non-canonical payload is a
   warning, not a failure: the signature still covers its exact bytes.
4. The Statement has in-toto's `_type`, our `predicateType`, and a non-empty
   subject.
5. Each `checks[].summary` equals the tally of its own `results`.
6. Every input, output and subject found under `--root` (default: the
   current directory) hashes to its recorded digest; a missing file is "not
   checked", never a pass.

And it says what it cannot check: that verdicts are correct (re-running needs
the ledger and library), that the listed papers were really read and the
queries really run (the signer's word), and that the timestamps are true (no
trusted timestamp authority). Exit 0 only when nothing FAILED.

## Surface

- **Python API** (`attestation.run_record`): `RunRecorder` accumulates a run
  (`add_input`, `add_output`, `add_query`, `add_paper`, `add_model`,
  `add_check`) and `sign()`s it into an envelope; `from_spec(dict)` builds
  one from plain JSON, which is what other tools and agents call.
- **CLI**: `attest claims ... --record out.json` (checks `claims` and
  `cite.check`, since `attest claims` runs the citation lint too; inputs are
  the Markdown files, the run artifacts each verdict read, and the `.bib`
  files the resolver consulted). `attest record create --spec run.json --out
  rec.json` for any other tool (AgentMarkit's brief scripts emit the spec).
  `attest record verify`, `attest record key`.
- **MCP**: `cite.check(path, record=...)` writes a record of that check.
  `runs.attest(spec, out)` creates one from a spec; `runs.verify_record(path,
  root)` verifies. Both in a new `mcp/run_record.py` module, registered under
  the `runs` namespace so the provenance surface serves them.

Writing refuses to overwrite an existing record (`--force` to replace),
matching `runs.record`'s new-files-only rule.

## Open decisions (for the user)

1. **Whose key signs.** Default implemented: one key per install. Alternatives,
   each a new `KeySource`: a per-researcher key the researcher holds (portable
   across machines, needs a distribution story); an org key on a hosted
   AgentMarkit machine (records attest "AgentMarkit ran this", not "this
   person"); Sigstore keyless signing (identity from an OIDC login, public
   transparency log -- conflicts with the local-only guarantee unless opt-in).
2. **Trusted time.** Timestamps are the signer's clock. An RFC 3161
   timestamp authority would make them evidence, at the cost of a network
   call.
3. **Where records live.** Today they are written where the caller says.
   An index of records in the DB (so `runs.detail` could list them) is
   deferred until something reads it.
4. **Encrypting the key at rest.** Unencrypted with 0600 perms, like an SSH
   key without a passphrase; a passphrase would break unattended agent runs.

## Testing

`tests/test_run_record.py`: round trip (build, sign, write, verify);
tampering -- an edited verdict, an edited input file, a swapped embedded key,
a pinned wrong key, an edited summary -- each FAILS and names what failed;
JCS stability (RFC 8785 number vectors, UTF-16 key order, the same record
built twice is byte-identical except the run id and time, and key order in the
input does not change the bytes); key file permissions and refusal of a
world-readable key; the verifier module imports no `sqlite3` and no
`attestation.db`; `attest claims --record` and the two MCP tools end to end on
a fixture workspace.
