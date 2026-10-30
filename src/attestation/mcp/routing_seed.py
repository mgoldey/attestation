"""The "I like <paper>" rule of feed.ask, split out of routing.py (its size cap).

A reader names a paper they like -- a DOI, an arXiv id, a title -- or a file
of them, and `feed.persona_update` seeds the ranking with it (seed.py). Pure
over the question string, like every rule in routing.py; `route_feed` calls
`route_liked` from `_first_rules`, after "I work on X".
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestation.mcp.routing import Decision

# "I like <paper>" seeds the ranking with a paper the reader names (seed.py);
# "I like the papers in refs.bib" with every paper in a file. Start-anchored,
# and never a pronoun: "I like this one" about a listed item is a verdict.
_LIKED = re.compile(r"^i (?:really )?(?:like|liked|love|loved) ")
_LIKED_FILE = re.compile(r"^(?:the )?(?:papers|refs|references|everything) (?:in|from) ")
_NOT_A_PAPER = re.compile(
    r"^(?:it|this|that|these|those|them|this one|that one|the (?:first|second|third|last)\b"
    r"|#\d+\b|\d+$|papers? (?:on|about)\b|reading\b|to\b|how\b|what\b|when\b)"
)


def route_liked(q: str, question: str) -> Decision | None:
    """`feed.persona_update` with `liked` (a paper) or `liked_file` (a .bib/PDF)."""
    from attestation.mcp.routing import Decision

    if not (m := _LIKED.match(q)) or _NOT_A_PAPER.match(rest := q[m.end() :].strip()):
        return None
    ref = question.strip()[m.end() :].strip().rstrip(".!")
    if f := _LIKED_FILE.match(rest):
        return Decision("feed.persona_update", {"liked_file": ref[f.end() :].strip()})
    if re.search(r"\.(?:bib|pdf)$", rest):
        return Decision("feed.persona_update", {"liked_file": ref})
    return Decision("feed.persona_update", {"liked": ref}) if len(ref) >= 6 else None
