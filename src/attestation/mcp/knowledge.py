"""Reading knowledge graph tools: the `kg.*` namespace.

The graph is derived fresh from item_tags on every read -- `kg.build_graph`
takes the tag assignments and returns adjacency plus edge weights. There is no
stored graph: kg_nodes/kg_edges/kg_meta were deleted on 2026-08-21 after a
characterization run showed all eight kg answers were byte-identical with and
without them.

Concepts are tags used at least MIN_TAG_USES times; edges are co-occurrences of
at least MIN_EDGE_WEIGHT. A tag used once is deliberately not a concept.
"""

from typing import Annotated, Literal

from pydantic import Field

from attestation import kg
from attestation.mcp._shared import MAX_LIST_LIMIT, Limit
from attestation.mcp._tool import ToolError, tool


def register(mcp) -> None:
    """Attach every kg.* tool to the server."""

    @mcp.tool(name="kg.neighbors")
    def kg_neighbors(node: str, limit: Limit = MAX_LIST_LIMIT) -> dict:
        """Concepts directly adjacent to a given concept in the reading graph.

        The "what else should I read about this" query. Concepts come from the
        tagging pass; a tag used only once is not in the graph.

        Returns only DIRECT neighbours, ranked by co-occurrence weight (how many
        items carry both concepts), strongest first, capped at `limit` (clamped
        to 50). Each row's `weight` is that real edge weight. For questions that
        span more than one hop, use `kg.path`, which answers them exactly.

        """
        return _neighbors(node, limit)

    @mcp.tool(name="kg.path")
    def kg_path(source: str, target: str) -> dict:
        """Shortest chain of concepts connecting two topics in the reading graph.

        Returns ok=false with path=null when the two are in different components —
        that is a legitimate answer meaning "these never co-occur", not an error.
        It is given only about names that really are concepts: one that is not is
        refused separately and says so, so a typo never reads as a finding. Use
        `kg.concepts` to see the valid names.

        """
        return _path(source, target)

    @mcp.tool(name="kg.central")
    def kg_central(
        metric: Annotated[
            Literal["degree", "betweenness"], Field(description="Centrality measure.")
        ] = "degree",
        limit: Limit = 10,
    ) -> dict:
        """The concepts this reader's work centres on.

        metric="degree" ranks by most-connected; "betweenness" finds the
        bridges between otherwise separate clusters, which is the more
        interesting answer to "what ties my areas together".
        """
        return _central(metric, limit)

    @mcp.tool(name="kg.communities")
    def kg_communities(min_size: Annotated[int, Field(ge=2, le=100)] = 3) -> dict:
        """What subjects this reader's papers split into.

        Answers "what are my main research areas" and "what themes run through
        my reading". Topic clusters over the concepts extracted from their
        items, each labelled by its most connected member.

        Clusters by modularity, so a dense hub does not swallow the graph: a
        concept joins a group only when its links there beat what chance predicts.
        Densely-interconnected corpora still split into real topics -- on the live
        graph, 7 of them, from a machine-learning core down to a small
        quantum-chemistry group.

        Expect overlapping subject matter across clusters rather than clean
        partitions; concepts sit in exactly one group, so a bridging concept lands
        wherever its links are strongest.

        """
        return _communities(min_size)

    @mcp.tool(name="kg.concepts")
    def kg_concepts(prefix: str | None = None, limit: Limit = MAX_LIST_LIMIT) -> dict:
        """The concept vocabulary the other kg tools accept.

        Names are lowercase and hyphenated (`reinforcement-learning`). The
        lookups now accept spaces and capitals too, but this is how to turn a
        vague phrase into the exact name, via `prefix`.

        Call this when you are not certain a name exists. The others take exact
        names and refuse one they do not have, rather than answering about it:
        `kg.path` reports "no path" only for two REAL concepts that never
        co-occur, which is a fact about the reading, not a typo.

        `prefix` is a case-insensitive SUBSTRING match, so "learn" finds
        `machine-learning`. Sorted, capped at `limit` (at most 16), with
        `n_concepts` giving how many matched so truncation is never silent.

        """
        return _concepts(prefix, limit)


NOT_A_CONCEPT = "{name!r} is not a concept in the graph; call kg.concepts() to list valid names"


@tool(empty={"neighbors": []}, label="kg_neighbors")
def _neighbors(conn, node: str, limit: int = 20) -> dict:
    shown = min(limit, MAX_LIST_LIMIT)
    found = kg.neighbors(conn, node, limit=shown)
    if not found:
        raise ToolError(NOT_A_CONCEPT.format(name=node))

    # The true degree, because kg.neighbors truncates internally and this tool
    # cannot otherwise see what it dropped. This namespace states the rule
    # twice -- "Never silent: a caller that cannot tell a capped list from the
    # whole vocabulary reads a missing name as proof the concept does not
    # exist" -- and kg.concepts and kg.communities both report their totals.
    # This one said "16 neighbour(s)" for a concept with 253.
    adjacency, _edges = kg.build_graph(kg.tag_assignments(conn))
    total = len(adjacency.get(kg.canonical(node), ()))

    message = f"{len(found)} neighbour(s)"
    if total > len(found):
        # NOT "raise limit": Limit is le=MAX_LIST_LIMIT, so a caller obeying
        # that got a pydantic validation dump and no way forward. Measured on
        # gemma4:e2b: the model read the message, sent limit=20, and gave up.
        message = (
            f"{total} neighbour(s); showing {len(found)}, the maximum"
            f" ({MAX_LIST_LIMIT}) -- narrow with a more specific concept"
        )
    return {"message": message, "neighbors": found, "n_neighbors": max(total, len(found))}


@tool(empty={"path": None}, label="kg_path")
def _path(conn, source: str, target: str) -> dict:
    # Membership is checked BEFORE pathfinding: kg.shortest_path returns None
    # both for an absent node and for two nodes in different components, but
    # this tool's docstring promises path=null means the latter. A mistyped
    # name reaching that promise makes the tool assert, confidently, that two
    # topics never co-occur -- about a concept that does not exist.
    adjacency, _ = kg.build_graph(kg.tag_assignments(conn))
    # Canonicalise before checking membership, or the check rejects an alias
    # the graph does in fact contain -- `llm` is the most common tag in the
    # live corpus and maps to a 163-neighbour hub. The refusal is right about
    # unknown names and was wrong about known ones spelled differently.
    for original in (source, target):
        if kg.resolve_query(original, adjacency) not in adjacency:
            raise ToolError(NOT_A_CONCEPT.format(name=original))
    found = kg.shortest_path(conn, source, target)
    if found is None:
        raise ToolError(f"no path between {source!r} and {target!r}")
    return {"message": f"{len(found) - 1} hop(s)", "path": found}


@tool(empty={"nodes": []}, label="kg_central")
def _central(conn, metric: str = "degree", limit: int = 10) -> dict:
    try:
        ranked = kg.central(conn, metric=metric, limit=min(limit, MAX_LIST_LIMIT))
    except ValueError as exc:
        # An unknown metric is the caller's typo, not a bug: name the valid
        # ones rather than logging a traceback they cannot act on.
        raise ToolError(str(exc)) from exc
    return {"message": f"top {len(ranked)} by {metric}", "nodes": ranked}


# How many members of a cluster to name. The live graph returned 13
# communities as 12,055 characters, one of them 201 tags listed
# alphabetically -- the graph's contents rather than a finding. A caller
# wants which clusters exist, how big they are, and what each is about.
MEMBERS_SHOWN = 8


# Groups the tool will show. Members per group were capped and the group COUNT
# was not, which is the same shape as the digest bug: one axis bounded, the
# other open. Measured on the live graph, 16 groups emitted 4039 chars and
# min_size=2 -- allowed by the schema -- reached 6033.
#
# Largest-first ordering means this drops the smallest clusters, and
# n_communities keeps the omission reportable.
MAX_COMMUNITIES_SHOWN = 8


@tool(empty={"communities": []}, label="kg_communities")
def _communities(conn, min_size: int = 3) -> dict:
    found = kg.communities(conn, min_size=min_size)
    # Largest first: a caller reading only the first few should see the
    # clusters that dominate the reading, not whichever the algorithm
    # happened to emit first.
    found.sort(key=lambda c: (-len(c["members"]), c["label"]))
    summarised = [
        {
            "label": c["label"],
            "n_members": len(c["members"]),
            "members": c["members"][:MEMBERS_SHOWN],
        }
        for c in found[:MAX_COMMUNITIES_SHOWN]
    ]
    biggest = summarised[0]["n_members"] if summarised else 0
    message = f"{len(found)} communit(ies), largest first"
    if len(found) > len(summarised):
        message += f"; showing the {len(summarised)} largest"
    if biggest > MEMBERS_SHOWN:
        message += f"; the largest has {biggest} concepts, showing {MEMBERS_SHOWN}"
    return {
        "message": message,
        "communities": summarised,
        "n_communities": len(found),
    }


@tool(empty={"concepts": [], "n_concepts": 0}, label="kg_concepts")
def _concepts(conn, prefix: str | None = None, limit: int = 50) -> dict:
    adjacency, _ = kg.build_graph(kg.tag_assignments(conn))
    names = sorted(adjacency)
    if prefix:
        needle = prefix.lower()
        names = [n for n in names if needle in n.lower()]
    shown = names[: min(max(1, int(limit)), MAX_LIST_LIMIT)]
    message = f"{len(names)} concept(s) " + (f"matching {prefix!r}" if prefix else "in the graph")
    if len(shown) < len(names):
        # Never silent: a caller that cannot tell a capped list from the whole
        # vocabulary reads a missing name as proof the concept does not exist
        # -- the same wrong conclusion this tool exists to prevent.
        message += f"; showing {len(shown)} -- pass prefix= to narrow them"
    return {"message": message, "concepts": shown, "n_concepts": len(names)}
