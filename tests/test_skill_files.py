"""Content checks on the shipped skill files.

No existing test reads setup.sh or SKILL.md content -- test_install.py and
test_install_e2e.py only assert the files *exist*. This file closes that gap
for the failure modes that matter: a skill file invoking a console script
pyproject.toml never declared (which execs silently on machines that happen
to have an unrelated same-named binary on PATH); a namespace or router the
skills forgot to mention (the agent reading them then never learns it
exists); and, since the 2026-08-30 split into one skill per agent surface, a
surface skill naming a tool its session cannot see, or two skills whose
descriptions open with the same word and so compete for the same questions
in Hermes' skill index.

Checked against pyproject.toml's actual [project.scripts] via tomllib and
the live MCP surface via register_all, rather than hardcoded lists, so this
keeps working through the next rename instead of quietly going stale itself.
"""

import re
import tomllib
from pathlib import Path

import attestation.install as install
from attestation import claims
from attestation.mcp import AGENT_SURFACES

_REPO_ROOT = Path(install.__file__).resolve().parent.parent.parent

SETUP_SKILL = "attestation-setup"
# Write-side skills teach an agent to PRODUCE the inputs the read-only tools
# consume; they are not surface skills (there is no "record" or "annotate"
# entry in AGENT_SURFACES) and the surface-scoped tests below must skip them.
WRITE_SIDE_SKILLS = ("attestation-record", "attestation-annotate")
_TOOL_TOKEN = re.compile(r"\b(feed|runs|kg|sym|cite)\.([a-z_]+)\b")


def _declared_console_scripts() -> set[str]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    return set(pyproject["project"]["scripts"])


def _skill_dirs() -> dict[str, Path]:
    dirs = {name: install._skill_source_dir(name) for name in install.SKILL_NAMES}
    for name, path in dirs.items():
        assert path.is_dir(), f"shipped skill source dir must exist: {name}"
    return dirs


def _skill_md(name: str) -> str:
    return (_skill_dirs()[name] / "SKILL.md").read_text()


def _surface_skills() -> dict[str, str]:
    """skill name -> the AGENT_SURFACES key it documents.

    Derived from AGENT_SURFACES rather than "every skill but setup", so a
    write-side skill (no entry in AGENT_SURFACES) is excluded by construction
    instead of by an ever-growing exclusion list.
    """
    return {
        name: name.removeprefix("attestation-")
        for name in install.SKILL_NAMES
        if name.removeprefix("attestation-") in AGENT_SURFACES
    }


def _frontmatter(text: str) -> dict[str, str]:
    """The top-level `key: value` lines between the first two `---` fences."""
    head = text.split("---", 2)[1]
    out = {}
    for line in head.splitlines():
        if line and not line[0].isspace() and ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip().strip('"')
    return out


def _undeclared_exec_targets(text: str, declared: set[str]) -> set[str]:
    """Binary names actually invoked via `uv run [--project <dir>] <bin> ...`
    or `uvx --from <pkg> <bin> ...`, that are not among the declared console
    scripts.

    Deliberately narrow (requires a trailing subcommand/flag like `install`,
    `serve`, `--check`, `--yes`) so it matches only real invocation lines,
    not prose that merely mentions "uv run" or "uvx --from" while explaining
    the command shape (e.g. SKILL.md's "`uvx --from <package>` takes the
    *package*, and the trailing word is the *executable*").
    """
    found = set()
    pattern = re.compile(
        r"\buv run(?:\s+--project\s+\S+)?\s+(\w+)\s+(?:install|serve)\b"
        r"|\buvx --from\s+\S+\s+(\w+)\s+(?:install|serve)\b"
    )
    for match in pattern.finditer(text):
        word = match.group(1) or match.group(2)
        if word not in declared:
            found.add(word)
    return found


def _live_tool_names() -> set[str]:
    """Every tool name the MCP server actually registers.

    Derived from the live surface rather than a literal list. A literal list
    is exactly what let the old SKILL.md drift: it documented the tools
    someone remembered, and `cite.*` -- four tools, a whole namespace -- was
    never written down at all.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("skill-surface-check")
    register_all(server)
    return {t.name for t in asyncio.run(server.list_tools())}


# --------------------------------------------------------------------------
# shape: one directory per skill, frontmatter naming it, surfaces covered
# --------------------------------------------------------------------------


def test_every_surface_has_a_skill_and_setup_has_the_script():
    """The split mirrors AGENT_SURFACES one-for-one, plus the setup skill
    that owns scripts/setup.sh; a surface without a skill is an agent with
    tools and no judgment about them."""
    assert set(_surface_skills().values()) == set(AGENT_SURFACES)
    assert (_skill_dirs()[SETUP_SKILL] / "scripts" / "setup.sh").is_file()


def test_frontmatter_name_matches_the_directory():
    """Hermes indexes a skill by its frontmatter `name`; a directory that
    says one thing and a header that says another is two skills to the
    index and one on disk."""
    for name in install.SKILL_NAMES:
        assert _frontmatter(_skill_md(name))["name"] == name


def test_skill_descriptions_open_with_distinct_verbs():
    """Measured: skill descriptions that name the same TOPIC collide --
    `blogwatcher` ("Monitor blogs and RSS/Atom feeds") took "check my rss
    feeds" from the feed tools, and routing fell 6/6 -> 3/6. Five siblings
    that all opened "Research provenance tools..." would collide with each
    other the same way. Each description leads with its own verb, and never
    with a topic word the collision measurement named."""
    openers = {}
    for name in install.SKILL_NAMES:
        first = _frontmatter(_skill_md(name))["description"].split()[0].lower().strip(",.:")
        openers[name] = first
    assert len(set(openers.values())) == len(openers), openers
    topic_words = {"research", "science", "arxiv", "rss", "feed", "feeds", "attestation"}
    assert not topic_words & set(openers.values()), openers


# --------------------------------------------------------------------------
# what the skills invoke
# --------------------------------------------------------------------------


def test_setup_sh_invokes_only_declared_console_scripts():
    declared = _declared_console_scripts()
    setup_sh = (_skill_dirs()[SETUP_SKILL] / "scripts" / "setup.sh").read_text()

    bad = _undeclared_exec_targets(setup_sh, declared)

    assert not bad, (
        f"setup.sh execs {bad}, not among pyproject's declared [project.scripts] {declared}"
    )


def test_setup_sh_has_no_hermes_install_invocation():
    """Regression guard for the specific incident: both exec paths called
    `hermes install --yes`, a binary pyproject never shipped."""
    setup_sh = (_skill_dirs()[SETUP_SKILL] / "scripts" / "setup.sh").read_text()

    assert "hermes install" not in setup_sh


def test_no_skill_md_invokes_undeclared_console_scripts():
    declared = _declared_console_scripts()
    for name in install.SKILL_NAMES:
        bad = _undeclared_exec_targets(_skill_md(name), declared)
        assert not bad, (
            f"{name}/SKILL.md documents running {bad}, not among pyproject's declared "
            f"[project.scripts] {declared}"
        )


def test_no_skill_md_has_hermes_install_invocation():
    for name in install.SKILL_NAMES:
        assert "hermes install" not in _skill_md(name), name


# --------------------------------------------------------------------------
# what the skills teach: every namespace and router somewhere, and each
# surface skill only its own
# --------------------------------------------------------------------------


def test_bundled_skills_together_document_every_live_namespace():
    """A namespace no skill names is invisible to the agent reading them.

    Four reviewers once found the old single file describing a world that no
    longer existed: `cite.*` undocumented entirely, the `.ask` routers -- the
    measured entry point, 13/15 against 8/15 -- nowhere. The check is over
    the UNION of the bundled skills, since each documents only its surface.

    Asserting on `<ns>.` catches the namespace being named as a tool prefix
    rather than merely as a word: "citations" in prose must not satisfy a
    check for `cite.*`.
    """
    everything = "\n".join(_skill_md(name) for name in _surface_skills())
    namespaces = {name.split(".", 1)[0] for name in _live_tool_names()}

    missing = sorted(ns for ns in namespaces if f"{ns}." not in everything)

    assert not missing, (
        f"no bundled skill documents {missing} tools, but the live MCP surface "
        f"registers that namespace. Add it to the skill for its surface."
    )


def test_bundled_skills_together_teach_every_ask_router():
    """Each router, by name, somewhere in the bundle. Separate from the
    namespace check because a namespace can be documented thoroughly while
    its router is not: the flat tools are what an agent reaches for by
    default, and they measured worse."""
    everything = "\n".join(_skill_md(name) for name in _surface_skills())

    missing = sorted(n for n in _live_tool_names() if n.endswith(".ask") and n not in everything)

    assert not missing, f"no bundled skill mentions {missing}; agents will pick flat tools instead"


def test_each_surface_skill_teaches_its_own_router():
    for name, surface in _surface_skills().items():
        routers = {
            n
            for n in _live_tool_names()
            if n.endswith(".ask") and _allowed(n, AGENT_SURFACES[surface].prefixes)
        }
        for router in routers:
            assert router in _skill_md(name), f"{name} never names {router}"


def _allowed(tool: str, prefixes: frozenset[str]) -> bool:
    return tool in prefixes or tool.split(".", 1)[0] in prefixes


def test_a_surface_skill_names_only_tools_on_its_surface():
    """A session under ATTEST_TOOLS=<surface> cannot see tools outside it --
    they are absent from list_tools, not merely undocumented. A skill that
    names `runs.compare` to a feed session teaches a call that will fail, and
    a model told a tool exists will keep trying it. Tools outside the surface
    are referred to by the AGENT that has them, never by name."""
    live = _live_tool_names()
    for name, surface in _surface_skills().items():
        text = _skill_md(name)
        named = {f"{ns}.{tool}" for ns, tool in _TOOL_TOKEN.findall(text)}
        # `.tools` is the surface's own disclosure tool, allowed everywhere
        # on its surface; anything unknown to the live server is prose
        # (`feed.list`-style hypotheticals do not exist) and is not a tool.
        foreign = sorted(
            t for t in named if t in live and not _allowed(t, AGENT_SURFACES[surface].prefixes)
        )
        assert not foreign, f"{name} names tools its surface cannot see: {foreign}"


def test_a_write_side_skill_names_only_tools_that_exist():
    """Unlike a surface skill, a write-side skill may legitimately name tools
    from more than one surface (record names `runs.*`; annotate names
    `runs.*` and `cite.*`, both hand-offs rather than its own remit) -- there
    is no single AGENT_SURFACES entry to check it against. What still has to
    hold, the no-phantom-tools rule applied to these two: every tool token it
    names must exist on the live, unrestricted MCP surface. A skill that
    teaches a call to a tool that was renamed or removed is teaching a
    guaranteed failure."""
    live = _live_tool_names()
    for name in WRITE_SIDE_SKILLS:
        text = _skill_md(name)
        named = {f"{ns}.{tool}" for ns, tool in _TOOL_TOKEN.findall(text)}
        phantom = sorted(t for t in named if t not in live)
        assert not phantom, f"{name} names tools that do not exist on the live surface: {phantom}"


def test_annotate_description_does_not_lead_with_citation():
    """The research doc's explicit collision rule: `research-paper-writing`
    and `grounded-citations` both own citation-shaped territory already, and
    a description leading with "citation" would land attestation-annotate in
    the same collision those two would have with each other. The skill's own
    content is about citations too (`cite=` keys), which is exactly why the
    verb it leads with has to be the thing that makes it distinct: annotating
    claims, not citations in general."""
    description = _frontmatter(_skill_md("attestation-annotate"))["description"]
    first_word = description.split()[0].lower().strip(",.:")
    assert first_word != "citation"
    assert not description.lower().startswith("citation")


def test_record_config_naming_rule_matches_the_real_pairing(tmp_path):
    """attestation-record's right/wrong config-naming example, fed to the
    REAL pairing logic (`ledger_adapters.generic.discover`), must pair and
    fail to pair exactly the way the skill says.

    Round-2 live eval: the model wrote `configs/asr_baseline_config.yaml`
    (a `_config` suffix) or one shared `configs/config.yaml` beside
    `results/asr_baseline.json`; `discover()` pairs a config to a result by
    EXACT stem equality (see `_result_name`'s `seen` set and the plain
    `cfg.stem` lookup in `discover`'s config-walk) with nothing fuzzier, so
    a stem that doesn't match exactly becomes an unevaluated run of its
    own -- scan found 4 runs for 2 arms, not 2. This is the scorer-
    independent check: it says nothing about what a model writes, only
    that the skill's own taught example is true of the code it describes.
    """
    from attestation.ledger_adapters.generic import discover

    project = tmp_path / "asrproj"
    (project / "results").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "results" / "asr_baseline.json").write_text('{"wer": 0.061}')

    # Right, per the skill: exact stem match, folds into the one run.
    (project / "configs" / "asr_baseline.yaml").write_text("lr: 0.01\n")
    right = discover(project)
    assert len(right) == 1, f"exact-stem config should fold into the result, got {right}"

    # Wrong, per the skill: a `_config` suffix breaks the stem match.
    (project / "configs" / "asr_baseline.yaml").unlink()
    (project / "configs" / "asr_baseline_config.yaml").write_text("lr: 0.01\n")
    wrong_suffix = discover(project)
    assert len(wrong_suffix) == 2, (
        f"a _config-suffixed stem should NOT fold in -- expected the result plus"
        f" an unevaluated spec run, got {wrong_suffix}"
    )

    # Wrong, per the skill: one shared config.yaml matches no result's stem.
    (project / "configs" / "asr_baseline_config.yaml").unlink()
    (project / "configs" / "config.yaml").write_text("lr: 0.01\n")
    wrong_shared = discover(project)
    assert len(wrong_shared) == 2, (
        f"a shared config.yaml should NOT fold in -- expected the result plus"
        f" an unevaluated spec run, got {wrong_shared}"
    )


def test_record_lists_every_built_in_metric_direction():
    """attestation-record's "already known" list is exactly
    `ledger.METRIC_DIRECTION`'s keys, or the skill teaches a stale idea of
    what needs declaring.

    Round-1 live eval: on a NOT-built-in metric (`novelty_rate`,
    `hallucination_score`, ...) the model wrote no `[metric_direction]`
    entry at all, because the skill never said which metrics were already
    covered -- there was nothing to contrast "unfamiliar" against. Fixed by
    naming the built-in table; this guards that the named list doesn't rot
    the way the tool-count docs did, by checking every key from the live
    table appears in the skill body (as a backticked name, so a namespaced
    mention doesn't accidentally satisfy a bare-word check) rather than
    duplicating the list a second time in a test.
    """
    from attestation.ledger import METRIC_DIRECTION

    text = _skill_md("attestation-record")
    missing = sorted(name for name in METRIC_DIRECTION if f"`{name}`" not in text)
    assert not missing, (
        f"attestation-record/SKILL.md's built-in metric list is missing {missing};"
        " ledger.METRIC_DIRECTION grew a key the skill never learned about"
    )


def test_record_claim_grammar_matches_the_parser(tmp_path):
    """attestation-annotate teaches a claim grammar by example; if the
    example does not actually parse, the skill is teaching an unparseable
    annotation, and nothing would notice short of a human running it -- the
    exact docs-drift bug this repo exists to catch. Every CONCRETE example
    (a real project/run, not the `<project>/<run>` placeholder line) is
    fed straight through the real `claims.parse_file` and must come back
    with zero malformed complaints and the fields the prose says it has."""
    text = _skill_md("attestation-annotate")
    claim_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("<!-- claim:") and "<project>/<run>" not in line
    ]
    assert claim_lines, "no concrete claim example found in attestation-annotate/SKILL.md"

    doc = tmp_path / "example.md"
    doc.write_text("\n\n".join(f"Some prose.\n{line}" for line in claim_lines) + "\n")

    found, problems = claims.parse_file(doc)

    assert not problems, f"skill's own claim examples do not parse: {problems}"
    assert len(found) == len(claim_lines)
    for claim in found:
        assert claim.project and claim.run
        assert claim.metric
        assert isinstance(claim.value, float)

    # The two follow-up examples added after the round-1 live eval: a
    # metric disambiguated by split=, and two claims backing two decimals
    # in one paragraph. If either stopped parsing with the field the skill
    # says it has, this would still pass on `found`/`problems` alone --
    # so check the specific fields the prose promises.
    with_split = [c for c in found if c.split is not None]
    assert with_split, "no example teaches split= disambiguation"
    assert {c.value for c in with_split} == {34.1, 31.7}

    two_metric_pairs = [c for c in found if c.project == "cls-two-metrics"]
    assert {c.metric for c in two_metric_pairs} == {"accuracy", "f1"}, (
        "the two-decimals-in-one-paragraph example should carry one claim per metric"
    )


def test_setup_skill_names_every_surface_and_sibling():
    """The setup skill is the one an agent reads first; it has to say which
    sibling carries the judgment for each surface, or the split leaves the
    agent with setup instructions and no map. Scoped to the surface skills
    (plus itself) via `_surface_skills()`: the write-side skills are not part
    of the setup map this test polices -- they teach existing surfaces'
    tools rather than adding one, and are not owned by any ATTEST_TOOLS
    value setup.sh wires up."""
    text = _skill_md(SETUP_SKILL)
    for name in (SETUP_SKILL, *_surface_skills()):
        assert name in text, f"setup skill never mentions {name}"
    for surface in AGENT_SURFACES:
        assert f"`{surface}`" in text, f"setup skill never names the {surface} surface"


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------


def test_the_presentation_example_stays_markdown_not_a_foreign_surface():
    """The gateway converts Markdown; hand-written foreign syntax breaks it.

    A Telegram reader asked for their feed and replied "you didn't give links",
    then "the links weren't clickable". Neither was a rendering bug -- the
    model emitted no links at all, printing `ID: 2385`, an argument for the
    next tool call and useless to a human.

    The first fix misread the transcript as Slack and taught the example
    `<url|title>`. Run against the real sender, that is actively worse: the
    Telegram path (hermes-agent tools/send_message_tool.py `_send_telegram`)
    auto-detects HTML with `re.search(r'<[a-zA-Z/][^>]*>', message)`, so a
    Slack link makes the WHOLE message go out as parse_mode=HTML, where
    `<url|title>` is not a tag. Plain Markdown instead reaches
    `plugins/platforms/telegram/adapter.py::format_message`, which converts it
    to MarkdownV2 -- verified five of five links preserved in a five-item list.

    So: the example must stay Markdown, and must NOT acquire another surface's
    syntax. Both directions are guarded, because both have now been wrong once.
    """
    skill_md = _skill_md("attestation-feed")
    start = skill_md.index("**Present each item as one line")
    block = skill_md[start : skill_md.index("```", skill_md.index("```", start) + 3)]

    assert "](http" in block, (
        "the presentation example lost its Markdown link; the gateway's "
        "markdown->MarkdownV2 converter is what makes urls clickable"
    )
    assert "<http" not in block, (
        "the example shows an angle-bracket link; that trips the Telegram "
        "sender's HTML auto-detect and posts the whole message as HTML"
    )
