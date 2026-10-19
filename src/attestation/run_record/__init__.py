"""Run records: a signed, portable account of one research run.

`build` makes them, `verify` checks them (standard library plus
`cryptography`, no database), `keys` says which key signs, and `canonical`
is the RFC 8785 serializer both sides agree on. See
`docs/superpowers/specs/2026-10-17-run-record-design.md`.

Nothing is re-exported: importing this package must stay as cheap as
importing `verify` alone.
"""
