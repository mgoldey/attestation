# Verifiable claims and citations

Can it check my draft? Yes — a claim is an HTML comment beside the prose it
describes, checked against the run ledger for one of five verdicts, and a
citation key is linted against your configured bibliography the same way.

## Verifiable claims

A README says "MAE 0.353 eV vs experiment". That number was transcribed by hand
and nothing checks it: re-run the benchmark and the document asserts 0.353
forever. A claim is an HTML comment beside the prose it describes, so the
document renders exactly as before:

```markdown
The cut leaves WER essentially unchanged (**0.053 vs 0.043** baseline).
<!-- claim: ablation/stack_4 metric=wer value=0.053 tol=0.001 -->
```

```bash
uv run attest claims ~/projects              # verify every claim
uv run attest claims ~/projects --coverage   # numbers no claim covers
```

Five verdicts, and the distinctions are the design. `supported`: a run agrees.
`contradicted`: a run disagrees — the document or the run is wrong.
`unsupported`: no run matches, so the claim may be true but nothing backs it.
`ambiguous`: a wildcard matched several runs, so which is meant is undecidable.
`stale`: the value matches but the artifact changed after `as_of`.

`unsupported` and `contradicted` never collapse together — one needs a run, the
other needs a correction. `ambiguous` exists because silently taking the first
of several matches is how a checker reports a confident wrong answer.
`attest claims` exits non-zero on a contradiction, so it can gate a commit.

A claim can also carry `cite=<key>`, and that key is linted too — `uncited`
when no configured `.bib` or Zotero source has it — from the CLI (`attest
claims`) as well as the `cite.check` / `runs.claims_check` MCP tools. It is a
lint ("no source has this key"), never "the paper does not support this
claim". See `examples/citations/` for a worked run of both linters together.

`--coverage` is the inverse, and the more useful half for adoption: a document
with zero contradicted claims can still assert a dozen unverifiable numbers.
Only decimals count as measurements — on a real index, 212 numbers reduce to 30
decimals and the decimals are the results.
