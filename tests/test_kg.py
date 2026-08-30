import pytest

from attestation import kg
from attestation.db import get_db


def seed(conn, items):
    """items: list of tag-lists, one per synthetic item."""
    for i, tags in enumerate(items, start=1):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"item {i}", f"h{i}"),
        )
        for t in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, t))
    conn.commit()


def test_singleton_tags_are_excluded(tmp_path):
    """86% of real tags are used exactly once and connect to nothing."""
    conn = get_db(tmp_path / "t.db")
    seed(conn, [["alpha", "beta"], ["alpha", "beta"], ["lonely", "alpha"]])

    adjacency, _ = kg.build_graph(kg.tag_assignments(conn))

    assert "lonely" not in adjacency, "a tag used once must not be a node"
    assert "alpha" in adjacency and "beta" in adjacency
    conn.close()


def test_edge_requires_min_weight(tmp_path):
    """A pair sharing only one item is co-incidence, not co-occurrence."""
    conn = get_db(tmp_path / "t.db")
    # alpha+beta share 2 items; alpha+gamma share only 1
    seed(conn, [["alpha", "beta"], ["alpha", "beta"], ["alpha", "gamma"], ["gamma", "beta"]])

    adjacency, weights = kg.build_graph(kg.tag_assignments(conn))

    assert weights.get(("alpha", "beta")) == 2
    assert ("alpha", "gamma") not in weights
    conn.close()


def test_aliases_merge_before_filtering(tmp_path):
    """Order matters: merging lifts variants over the frequency threshold.

    'variant' is used once and 'canon' once, so filtering first would drop
    both. Merging first makes one tag used twice, which survives.
    """
    conn = get_db(tmp_path / "t.db")
    kg.ALIASES["variant"] = "canon"
    try:
        seed(conn, [["variant", "shared"], ["canon", "shared"]])
        adjacency, weights = kg.build_graph(kg.tag_assignments(conn))

        assert "variant" not in adjacency, "the alias must not survive as its own node"
        assert "canon" in adjacency
        assert weights.get(("canon", "shared")) == 2
    finally:
        del kg.ALIASES["variant"]
    conn.close()


def test_canonical_is_identity_for_unknown_tags(tmp_path):
    assert kg.canonical("some-unmapped-tag") == "some-unmapped-tag"


def test_huggingface_hyphenation_variant_is_merged(tmp_path):
    """One of the eight hyphenation variants added to kg_aliases.toml: the
    unhyphenated spelling must not survive as its own node once merged with
    its canonical, hyphenated form."""
    conn = get_db(tmp_path / "t.db")
    seed(
        conn,
        [
            ["huggingface", "shared"],
            ["hugging-face", "shared"],
        ],
    )
    adjacency, _ = kg.build_graph(kg.tag_assignments(conn))

    assert "huggingface" not in adjacency
    assert "hugging-face" in adjacency
    conn.close()


def test_alias_file_maps_only_to_real_canonical_forms():
    """Every alias target should itself be a plausible tag, not a typo.

    Guards against an alias table entry that silently creates a node no item
    ever had.
    """
    for variant, target in kg.ALIASES.items():
        assert variant != target, f"{variant!r} maps to itself"
        assert target not in kg.ALIASES, f"{target!r} is both a target and an alias"


def test_aliases_merge_before_filtering_without_a_database():
    """The alias -> frequency-filter -> co-occurrence order, tested on plain data.

    This is the ordering the module docstring calls load-bearing. "llm" and
    "llms" both alias to large-language-models. Each spelling is used once, so
    if MIN_TAG_USES culled before aliasing, neither would survive and the
    concept would vanish. Merged first, it has two uses and stays.

    It needed a database until build_graph took a connection. Now it does not,
    which is the point of the signature change.
    """
    assignments = [(1, "llm"), (1, "rag"), (2, "llms"), (2, "rag")]
    adjacency, edges = kg.build_graph(assignments)

    merged = "large-language-models"
    assert kg.canonical("llm") == merged and kg.canonical("llms") == merged
    assert merged in adjacency, "aliasing must precede the frequency filter"
    assert adjacency[merged] == {"rag"}
    assert edges[tuple(sorted((merged, "rag")))] == 2


# --- separator and plural folding in canonical() ---------------------------
#
# The alias table can only merge variants someone thought to list. Folding
# handles the open-ended cases: on the live corpus (3955 distinct tags, 71%
# used once) it merged 79 variant pairs the table never named.


@pytest.mark.parametrize(
    ("variant", "canon"),
    [
        ("machinelearning", "machine-learning"),  # separator, via fold_canonical
        ("finetuning", "fine-tuning"),
        ("multi-modal", "multimodal"),
        ("risc-v", "risc-v"),
        ("riscv", "risc-v"),
        ("transformer", "transformers"),  # plural, via the alias table's target
        ("datasets", "dataset"),  # plural, via the default spelling
        ("proteins", "protein"),
        ("stem-cells", "stem-cells"),
        ("hallucinations", "hallucination"),
        ("machine_learning", "machine-learning"),  # underscore separator
    ],
)
def test_canonical_folds_spelling_variants(variant, canon):
    assert kg.canonical(variant) == canon


@pytest.mark.parametrize(
    "tag",
    [
        # -ics field names are not plurals of physic/robotic/genomic.
        "physics",
        "robotics",
        "genomics",
        "ethics",
        "dynamics",
        "statistics",
        # singular nouns that merely end in s
        "series",
        "species",
        "analysis",
        "bias",
        "virus",
        "mass",
        "lens",
        "news",
        "gas",
        "basis",
        "axis",
        # plurals that ARE the term of art, and plurals with no second
        # spelling in the corpus: folding must not rename what it cannot merge
        "mixture-of-experts",
        "scaling-laws",
        "agentic-workflows",
        "neural-networks",
        "diabetes",
        "stochastic-processes",
    ],
)
def test_canonical_leaves_a_tag_it_cannot_merge_alone(tag):
    """Folding merges; it must never rename.

    Two failure modes, both measured on the live corpus. A trailing "s" is
    often not a plural, and stripping it invents words (physics -> physic,
    series -> sery, lens -> len). And where the "s" IS a plural, the plural is
    frequently the term of art (`mixture-of-experts`) or simply the only
    spelling anyone used -- 506 of 588 computed folds had no merge partner at
    all. Every tag here must survive unchanged.
    """
    assert kg.canonical(tag) == tag


def test_canonical_never_merges_distinct_concepts():
    """The constraint that makes an automatic rule safe at all.

    Folding touches separators and a trailing plural "s" only, so it cannot
    reach pairs that differ by a stem letter. `rna`/`dna` is the catastrophic
    case; the rest are near-misses on the live tag list.
    """
    must_stay_distinct = [
        ("rna", "dna"),
        ("attention", "attention-mechanisms"),
        ("physics", "physical"),
        ("bias", "bias-mitigation"),
        ("cell", "cell-free"),
        ("protein", "proteomics"),
        ("graph", "graphics"),
        ("optimization", "optimizers"),
    ]
    for left, right in must_stay_distinct:
        assert kg.canonical(left) != kg.canonical(right), f"{left!r} merged into {right!r}"


def test_folding_merges_a_pair_the_alias_table_never_listed():
    """The point of folding: a variant nobody hand-listed still merges.

    Neither spelling is in kg_aliases.toml, and each is used once, so without
    folding both would be culled by MIN_TAG_USES and the concept would vanish
    from the graph entirely.
    """
    assert "nanocarriers" not in kg.ALIASES and "nanocarrier" not in kg.ALIASES

    assignments = [
        (1, "nanocarriers"),
        (1, "drug-delivery"),
        (2, "nanocarrier"),
        (2, "drug-delivery"),
    ]
    adjacency, _ = kg.build_graph(assignments)

    assert "nanocarrier" in adjacency, "folding must precede the frequency filter"
    assert "nanocarriers" not in adjacency


def test_canonical_is_idempotent():
    """canonical(canonical(t)) == canonical(t) for every tag the table names.

    A fold that moved a tag on each pass would make the graph depend on how
    many times aliasing ran.
    """
    tags = (
        set(kg.ALIASES)
        | set(kg.ALIASES.values())
        | {
            "machinelearning",
            "transformer",
            "datasets",
            "physics",
            "series",
            "huggingface",
        }
    )
    for tag in tags:
        once = kg.canonical(tag)
        assert kg.canonical(once) == once, f"{tag!r} -> {once!r} is not stable"


def test_health_counts_canonical_tags_not_raw_rows(tmp_path):
    """singleton_rate must see the merging it exists to watch.

    Counting raw item_tags rows made every alias invisible to the metric:
    both spellings stayed in the denominator as separate tags. Here one
    concept is spelled two ways across two items -- canonically that is a
    single tag used twice, so nothing is a singleton.
    """
    conn = get_db(tmp_path / "t.db")
    seed(conn, [["machinelearning", "shared"], ["machine-learning", "shared"]])

    out = kg.health(conn)

    assert out["distinct_tags"] == 2, "machinelearning and machine-learning are one tag"
    assert out["singleton_rate"] == 0.0
    conn.close()


def test_folding_never_renames_a_tag_with_no_merge_partner():
    """The guard on the whole mechanism: no partner, no rewrite.

    A computed singular would rewrite all of these. Because canonical() only
    folds onto spellings kg_aliases.toml actually names, a lone tag keeps the
    spelling the corpus gave it -- so `canonical` can be applied to any tag,
    including one no rule anticipated, without corrupting it.
    """
    for tag in ("quantum-dots", "exoplanets", "microrobots", "wildfires"):
        assert tag not in kg.ALIASES
        assert kg.canonical(tag) == tag


def test_an_aliased_concept_is_found_by_the_name_people_write():
    """The graph is built from canonical names, and the lookups were not.

    `llm` is the most common tag in the live corpus (642 uses) and an alias for
    `large-language-models` in kg_aliases.toml. `kg.neighbors('llm')` reported
    "not a concept" while the canonical form had 163 neighbours, and the
    recovery it named made things worse -- kg.concepts(prefix="llm") returns
    code-llm and llm-safety and NOT the hub, so a caller concludes their
    reading is absent. Measured on gemma4:e2b, the model did exactly that and
    told the user their LLM reading did not exist.
    """
    import sqlite3

    from attestation import kg

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # tag_assignments indexes rows by name
    conn.execute("CREATE TABLE item_tags(item_id INTEGER, tag TEXT)")
    # Two items sharing both tags: enough for MIN_TAG_USES and MIN_EDGE_WEIGHT.
    for item_id in (1, 2):
        for tag in ("large-language-models", "transformers"):
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, tag))
    conn.commit()

    canonical_hits = kg.neighbors(conn, "large-language-models")
    assert canonical_hits, "fixture did not build a graph"
    assert kg.neighbors(conn, "llm") == canonical_hits, (
        "an alias returns different neighbours than its canonical form"
    )
    assert kg.shortest_path(conn, "llm", "transformers") is not None, (
        "no path found from an aliased source"
    )
    # A genuinely unknown name must still be unknown.
    assert kg.neighbors(conn, "no-such-concept-xyz") == []
    assert kg.shortest_path(conn, "no-such-concept-xyz", "transformers") is None


def test_resolve_or_raise_reports_the_right_kind_in_its_message():
    """One canonicalise-check-raise helper serves both `_resolved_tag` (feed.py)
    and knowledge.py's `_path` resolve loop, over their own membership sets --
    the message names the kind and the right recovery tool, since a tag and a
    concept point an agent at different next calls (`kg.concepts(prefix=...)`
    vs `kg.concepts()`)."""
    from attestation.kg import resolve_or_raise

    with pytest.raises(ValueError, match="not a tag"):
        resolve_or_raise("nope", {"llm"}, kind="tag")
    with pytest.raises(ValueError, match="not a concept"):
        resolve_or_raise("nope", {"llm"}, kind="concept")
    assert resolve_or_raise("LLM", {"llm"}, kind="tag") == "llm"  # canonicalised
