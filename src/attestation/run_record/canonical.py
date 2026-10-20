"""RFC 8785 JSON Canonicalization Scheme (JCS), standard library only.

A run record is signed over its exact bytes (DSSE), so verification never
needs canonical form. Canonical form buys STABILITY: the same record content
always serializes to the same bytes, so a record can be named by its digest
and two builds compared byte for byte -- by this code or by a verifier
written in another language.

`json.dumps(sort_keys=True, separators=(",", ":"))` is close and not equal:
it orders keys by code point where JCS orders by UTF-16 code unit (they
differ for characters above U+FFFF against U+E000-U+FFFF), and it prints
floats Python's way (`1e-07`, `1.0`, `1e+16`) where JCS prints them the
ECMAScript way (`1e-7`, `1`, `10000000000000000`).

Refused rather than rounded: non-finite floats (JSON has no spelling for
them) and integers outside +/-2^53, which I-JSON -- and every JavaScript
verifier -- would silently round to a different number.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal

MAX_SAFE_INTEGER = 2**53 - 1


def _number(value: float) -> str:
    """ECMAScript Number::toString for a finite double (ES2020 7.1.12.1)."""
    if not math.isfinite(value):
        raise ValueError(f"JCS cannot represent {value!r}")
    if value == 0:
        return "0"  # also -0.0: ES6 prints "0"
    sign = "-" if value < 0 else ""
    # repr() is the shortest round-tripping decimal, which is what ES6 uses too;
    # only the layout differs, so take its digits and re-lay them out.
    _, digit_tuple, exp = Decimal(repr(abs(value))).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    exp = int(exp) + (len(digit_tuple) - len(digits))
    k = len(digits)
    n = k + exp  # position of the decimal point relative to the digits
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{mantissa}e{'+' if e > 0 else '-'}{abs(e)}"


def _string(value: str) -> str:
    # ensure_ascii=False matches ES6 JSON.stringify: only `"`, `\` and C0
    # controls are escaped, with lowercase hex. A lone surrogate is caught
    # by the UTF-8 encode in canonicalize().
    return json.dumps(value, ensure_ascii=False)


def _utf16_key(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def _serialize(value, out: list[str]) -> None:
    if value is None or isinstance(value, bool):
        out.append({None: "null", True: "true", False: "false"}[value])
    elif isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValueError(f"JCS integer out of the IEEE-754 safe range: {value}")
        out.append(str(value))
    elif isinstance(value, float):
        out.append(_number(value))
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, list | tuple):
        _array(value, out)
    elif isinstance(value, dict):
        _object(value, out)
    else:
        raise TypeError(f"JCS cannot serialize {type(value).__name__}")


def _array(value, out: list[str]) -> None:
    out.append("[")
    for i, item in enumerate(value):
        out.append("," if i else "")
        _serialize(item, out)
    out.append("]")


def _object(value: dict, out: list[str]) -> None:
    if not all(isinstance(k, str) for k in value):
        raise TypeError("JCS object keys must be strings")
    out.append("{")
    for i, key in enumerate(sorted(value, key=_utf16_key)):
        out.append(("," if i else "") + _string(key) + ":")
        _serialize(value[key], out)
    out.append("}")


def canonicalize(value) -> bytes:
    """`value` as RFC 8785 canonical JSON, UTF-8 encoded."""
    out: list[str] = []
    _serialize(value, out)
    return "".join(out).encode("utf-8")
