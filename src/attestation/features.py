"""Feature extraction: LLM tagging pass + per-key click-preference scores.

Reliability contract mirrors explain.py: tagging is lazy relative to ingest,
validated, and skips (never blocks) on failure. Nothing in this module runs
on the rank path except pure-SQL/numpy preference scoring.
"""

import dataclasses
import itertools
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, field_validator

from attestation.llm import BackendUnreachable, backend_unreachable

log = logging.getLogger(__name__)

CONTENT_TYPES = ("paper", "survey", "announcement", "release", "blog", "other")
TAG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"

# SQLite's SQLITE_LIMIT_VARIABLE_NUMBER default: the max bind parameters in one
# statement. item_ids here can be every item in the archive (via rank_items'
# unconditional call chain when exclude_clicked=False), so IN (...) queries
# must chunk below this limit rather than build one query per item.
_SQL_VAR_CHUNK = 900

# Tags that describe an item's provenance or its post type rather than its
# subject. Both axes are already recorded structurally -- the publication in
# items.feed_id, the post type in item_features.content_type -- so as tags they
# only add edges between items that share a source, not a topic. On the live
# graph these pulled `nature`, `science-feed` and `retraction` into the
# `biology` cluster, making a feed look like a research area. The prompt asks
# the model to avoid them; this enforces it, because a prompt is advisory.
NON_TOPIC_TAGS = frozenset(
    {
        "announcement",
        "arxiv",
        "blog",
        "conference",
        "journal",
        "nature",
        "news",
        "newsletter",
        "preprint",
        "press-release",
        "release",
        "report",
        "retraction",
        "science",
        "science-feed",
        "update",
    }
)


class ItemTags(BaseModel):
    content_type: Literal["paper", "survey", "announcement", "release", "blog", "other"]
    tags: list[str] = Field(min_length=1, max_length=4)

    @field_validator("tags")
    @classmethod
    def _tags_shape(cls, v: list[str]) -> list[str]:
        """Normalize what is recoverable, drop what is not, keep the item.

        TAG_PATTERN is lowercase-only, so every capitalised acronym the
        literature actually uses -- rRNA, DNA, CRISPR, RNA-seq, GPT-4 -- failed
        it. Rejecting the whole item for one such tag meant the item never
        tagged at all and was retried on every cron run forever, discarding the
        two or three perfectly good tags alongside it. One Nature paper sat
        untagged through a full re-tag for exactly this reason: the model
        returned ['biology', 'rRNA', 'exosome', 'nucleolar'] on three
        consecutive attempts and lost all four each time.

        Case and surrounding whitespace carry no meaning here (the graph
        lowercases everything anyway), so folding them is lossless. A tag that
        is still malformed after folding -- truncated, punctuated, too long --
        is genuinely unusable and is dropped on its own. The item survives if
        at least one tag does; only an item with nothing usable fails.
        """
        import re

        well_formed: list[str] = []
        for t in v:
            candidate = t.strip().lower().replace(" ", "-")
            if re.match(TAG_PATTERN, candidate) and candidate not in well_formed:
                well_formed.append(candidate)
        if not well_formed:
            raise ValueError(f"no usable tags in {v!r}")

        # Filtering to empty is NOT a failure: a release note really can be
        # about nothing but its own release. Failing here would leave the item
        # untagged and retried on every cron run forever -- the same trap the
        # all-or-nothing validator fell into. Keep the item with no topic tags;
        # content_type still records what it is, and a tagless item simply
        # contributes nothing to the graph, which is the correct outcome.
        return [t for t in well_formed if t not in NON_TOPIC_TAGS]


def tag_vocabulary(conn: sqlite3.Connection, limit: int = 150) -> list[str]:
    """Most-used tags, canonicalized, to steer the model toward existing vocabulary.

    Counts are summed over `kg.canonical` before ranking, which fixes a
    feedback loop: the raw table ranks each spelling separately, so the
    vocabulary shown to the model listed `machine-learning` (872) AND
    `machinelearning` (463), and `llm` (642) beside `language-models` (212),
    spending three of forty slots re-teaching the model the very variants the
    graph then merges away. Worse, the spellings it taught were the ones
    `canonical()` rewrites, so the model was being steered toward deprecated
    forms. Merging first frees those slots and lets real concepts
    (hugging-face, natural-language-processing, continual-learning) into the
    list instead.

    `limit` is 150 rather than 40 because 40 covered only 59% of tag
    assignments on the live corpus against 67% at 150 -- a model shown 40 tags
    for a 5000-item archive meets an unfamiliar subject constantly and mints a
    new tag when it does. 150 canonical tags is ~1.4KB of prompt, which is
    affordable next to the item summary it accompanies.

    Excludes NON_TOPIC_TAGS: the prompt asks the model not to emit them, so
    suggesting them here would work against it.
    """
    from attestation.kg import canonical

    totals: dict[str, int] = {}
    for row in conn.execute("SELECT tag, COUNT(*) n FROM item_tags GROUP BY tag"):
        name = canonical(row["tag"])
        if name in NON_TOPIC_TAGS:
            continue
        totals[name] = totals.get(name, 0) + row["n"]
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tag for tag, _ in ranked[: max(0, int(limit))]]


# The shipped tagging instruction: evals/prompts/tagging-2026-08-27.json,
# written by GEPA (evals/optimize_tagging.py) and gated by the transfer matrix
# beside it -- not worse than the hand-written prompt on gemma4:e2b
# (0.807/0.807), +0.110 on gemma4:e4b and +0.086 on hermes3:8b, narrower
# spread. It is embedded here rather than read from evals/ because evals/ is
# not in the wheel; tests/test_tag_prompt.py pins this text to the artifact.
# The hand-written prompt it replaced is evals/prompts/hand-written.json.
DEFAULT_TAG_INSTRUCTION = (
    "You are a specialized classifier for science, technology, and software "
    "content. Your goal is to categorize items into specific types and extract "
    "their core technical or scientific domain tags into a JSON format based on "
    "the provided title and summary.\n"
    "\n"
    "### Output Format:\n"
    "Return only a JSON object with the following keys:\n"
    "1. `content_type`: (Must be one of: paper, survey, announcement, release, "
    "blog, other)\n"
    "2. `tags`: A list containing 1-4 short, lowercase, and hyphenated strings "
    "representing the core domain or functionality.\n"
    "\n"
    "### Content Type Definitions:\n"
    "- **paper**: Peer-reviewed scientific research articles describing original "
    'study results or new methodologies (e.g., "A novel approach to...", "We '
    'present a method for...").\n'
    "- **survey**: A comprehensive review, overview, or state-of-the-field "
    "analysis (not a single new discovery).\n"
    "- **release**: Official announcements regarding software updates, library "
    "releases, laboratory achievements, awards, or infrastructure changes.\n"
    "- **blog**: Informal commentary, personal reflections, community discussions, "
    "opinion pieces, or news reports/timelines related to industry events.\n"
    "- **other**: Anything that does not fit the above (e.g., editorial columns, "
    "internal HR announcements, personal career news).\n"
    "\n"
    "### Tagging Rules:\n"
    "1. **Identify Technical Functionality over General Themes**: When content "
    "describes software tools, libraries, or and AI infrastructure, use specific "
    "technical capabilities rather than general conceptual themes. \n"
    "   - **NO** broad descriptive terms like `agentic-workflows`, "
    "`software-development`, or `problem-solving`.\n"
    "   - **YES** specific functional tags such as `tool-use`, `llm-cli`, "
    "`python-tooling`, `command-line-tools`, or `developer-tools`.\n"
    "2. **Domain focus only**: Only tag the primary scientific fields, technical "
    "capabilities, or application domains.\n"
    "   - **NO ENTITIES**: Do not include names of people, organizations, "
    "companies, specific software brands, or social media handles (e.g., NO "
    '"hugging-face", "openai", "meta", "twitter").\n'
    "3. **Specific Science Context over General Terms**: When content relates to "
    "the infrastructure or policy of science, use precise identifiers for the "
    "scientific community rather than general terms like `management`, "
    "`education`, or `policy`.\n"
    "   - Use: `lab-management`, `research-culture`, `mentorship`, "
    "`science-policy`, `academia`, or `scientific-infrastructure`. \n"
    "4. **Specific Domain over General Technique**: Prioritize specific "
    "application domains over broad technical methodologies.\n"
    "   - If a paper uses a deep learning model to solve a finance problem, use "
    "tags like `algorithmic-trading` or `quant-finance` instead of "
    "`deep-learning`, `optimization`, or `machine-learning`.\n"
    "5. **Standardized & Concise**: Use standard industry terms (e.g., "
    "`chemistry`, `genomics`, `computer-vision`). \n"
    "   - Only use broad terms like `machine-learning` if no more specific "
    "sub-domain is applicable.\n"
    "6. **Strict Relevance & No Inference**: Be extremely conservative. Only "
    "include a tag if the summary explicitly highlights that domain or "
    "functionality. Do not infer proximity (e.g., if a math proof describes a "
    "physics problem, use `physics`, do not jump to a related but unmentioned "
    "field like `quantum-computing`).\n"
    "\n"
    "### Constraints:\n"
    "- **Tag Count**: Provide between 1 and 4 tags.\n"
    "- **Format**: JSON only. No extra text.\n"
    '- **Distinctness**: Do not include "filler" tags like `ai` or `research`. If '
    "a specific domain (e.g., `natural-language-processing`) is applicable, use "
    "that instead.\n"
    "\n"
    "### Example Logic for Quality Control:\n"
    "1. A release of a tool for interacting with LLMs $\\rightarrow$ type: "
    "`release`, tags: [`llm-cli`, `tool-use`, `command-line-tools`] (Avoid broad "
    "terms like `software-development`).\n"
    "2. A study on the chemistry of molecules $\\rightarrow$ type: `paper`, tags: "
    "[`chemistry`, `molecular-biology`].\n"
    "3. A paper about trading algorithms $\\rightarrow$ type: `paper`, tags: "
    "[`algorithmic-trading`, `finance`] (Do not use `optimization` or `time-series`).\n"
    "4. An article on how to build a Chrome Extension $\\rightarrow$ type: `blog`, "
    "tags: [`web-development`, `browser-extensions`].\n"
    "\n"
    "### Final Instruction:\n"
    "Produce a valid JSON output only. Do not include any conversational text, "
    "explanations, or notes."
)


@dataclasses.dataclass(frozen=True)
class TagPrompt:
    """A tagging prompt as data: an instruction and optional demonstrations.

    Produced offline by the optimizer (evals/optimize_tagging.py) and loaded
    here; the library never optimizes. `source` names the file it came from
    so a tagging run can report which prompt produced its tags, the way it
    reports which model did.
    """

    instruction: str
    demos: tuple[dict, ...] = ()
    source: str | None = None


def load_tag_prompt(path: str | Path) -> TagPrompt:
    """Read a prompt artifact, validating the parts the renderer relies on.

    A demo is rendered verbatim as an assistant turn, so a malformed one --
    a content_type the schema rejects, a tag the validator would strip -- would
    teach the model the exact output the run then discards. Refuse it here,
    before any model call.
    """
    body = json.loads(Path(path).read_text())
    instruction = body.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{path}: 'instruction' must be a non-empty string")
    demos = body.get("demos", [])
    if not isinstance(demos, list):
        raise ValueError(f"{path}: 'demos' must be a list")
    for i, demo in enumerate(demos):
        if not isinstance(demo, dict):
            raise ValueError(f"{path}: demo {i} is not an object")
        fields = cast(dict[str, Any], demo)
        missing = {"title", "summary", "content_type", "tags"} - set(fields)
        if missing:
            raise ValueError(f"{path}: demo {i} lacks {sorted(missing)}")
        tags = list(fields["tags"])
        parsed = ItemTags.model_validate({"content_type": fields["content_type"], "tags": tags})
        if list(parsed.tags) != tags:
            raise ValueError(f"{path}: demo {i} tags {tags} would be normalized to {parsed.tags}")
    return TagPrompt(instruction=instruction, demos=tuple(demos), source=str(path))


def tag_prompt_from_env() -> TagPrompt | None:
    """`attest tag` uses the artifact ATTEST_TAG_PROMPT names, else the
    hand-written prompt. Artifacts come from evals/optimize_tagging.py."""
    path = os.environ.get("ATTEST_TAG_PROMPT")
    return load_tag_prompt(path) if path else None


def tag_messages(
    title: str, summary: str, vocab: list[str], prompt: TagPrompt | None = None
) -> list[dict]:
    """The ONE renderer of the tagging prompt.

    The eval harness, the optimizer's transfer test and `attest tag` all call
    this, so a score is always a score of the prompt that actually runs. With
    `prompt=None` the output is the hand-written prompt, byte for byte.
    Demonstrations render as prior user/assistant turns without the
    vocabulary line: 150 tags repeated per demo would spend more prompt on
    the examples than on the item.
    """
    instruction = prompt.instruction if prompt else DEFAULT_TAG_INSTRUCTION
    messages: list[dict] = [{"role": "system", "content": instruction}]
    for demo in prompt.demos if prompt else ():
        messages.append(
            {"role": "user", "content": f"Title: {demo['title']}\nSummary: {demo['summary']}"}
        )
        answer = {"content_type": demo["content_type"], "tags": list(demo["tags"])}
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    vocab_line = ", ".join(vocab) if vocab else "(none yet)"
    messages.append(
        {
            "role": "user",
            "content": (
                f"Existing vocabulary: {vocab_line}\n\nTitle: {title}\nSummary: {summary[:1000]}"
            ),
        }
    )
    return messages


def _tag_prompt(item: sqlite3.Row, vocab: list[str], prompt: TagPrompt | None = None) -> list[dict]:
    return tag_messages(item["title"], item["summary"], vocab, prompt)


def tag_one_item(
    conn,
    item: sqlite3.Row,
    chat_fn,
    vocab: list[str],
    model_name: str,
    prompt: TagPrompt | None = None,
) -> bool:
    """One LLM call (plus one retry) -> item_features + item_tags rows.
    False = skipped."""
    parsed = None
    for _ in range(2):  # one retry, per spec
        try:
            out = chat_fn(_tag_prompt(item, vocab, prompt), ItemTags.model_json_schema())
            parsed = ItemTags.model_validate(out)
            break
        except Exception as exc:
            if backend_unreachable(exc):
                # A dead socket is not this item's fault, and retrying it is
                # pointless: hand the run the decision to stop.
                raise BackendUnreachable(str(exc)) from exc
            log.debug("tagging attempt failed for item %s", item["id"], exc_info=True)
    if parsed is None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, ?)",
        (item["id"], parsed.content_type, model_name),
    )
    conn.execute("DELETE FROM item_tags WHERE item_id = ?", (item["id"],))
    conn.executemany(
        "INSERT INTO item_tags(item_id, tag) VALUES (?, ?)",
        [(item["id"], t) for t in dict.fromkeys(parsed.tags)],
    )
    conn.commit()
    return True


def _minted_first_use(conn, item_id: int, parsed_tags) -> bool:
    """Did this item use a tag that exists NOWHERE else in the corpus?

    Only a genuinely new tag can change the steering vocabulary, so only that
    needs a re-read. Comparing against the vocabulary LIST instead was the
    obvious version and it is wrong: tag_vocabulary returns the top 150, so
    every ordinary tag ranked 151st or lower read as "unknown" and refreshed
    anyway -- measured at 188 refreshes across 200 items, which is the per-item
    cost with extra steps. This asks the table, which is the thing that decides.
    """
    if not parsed_tags:
        return False
    placeholders = ",".join("?" * len(list(parsed_tags)))
    row = conn.execute(
        f"SELECT 1 FROM item_tags WHERE tag IN ({placeholders}) AND item_id != ? LIMIT 1",
        (*parsed_tags, item_id),
    ).fetchone()
    return row is None


def run_tagging(conn, chat_fn=None, limit: int | None = None) -> dict:
    """Tag every untagged item, newest first. Returns {"tagged": n, "failed": n}.

    Failed items are skipped (stay untagged) and retried on the next run.
    """
    if chat_fn is None:
        from attestation.llm import default_chat_fn

        chat_fn = default_chat_fn
    from tqdm import tqdm  # lazy: keeps the rank path free of tagging-only imports

    from attestation.llm import chat_model

    sql = (
        "SELECT i.id, i.title, i.summary FROM items i"
        " LEFT JOIN item_features f ON f.item_id = i.id"
        " WHERE f.item_id IS NULL ORDER BY i.published DESC"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()

    # Resolve the model ONCE and report it. Reading it per item let a single
    # run straddle two models if the environment changed underneath, and
    # returning it makes the tagging model visible to callers -- item_features
    # records a model per row but nothing ever surfaced it, so a run against
    # the wrong model (e.g. a standalone script that never called load_env and
    # silently got DEFAULT_CHAT_MODEL) looked identical to a correct one.
    model_name = chat_model()
    # Same reasoning for the prompt: resolved once, reported, and a broken
    # artifact refuses here rather than failing every item against the model.
    prompt = tag_prompt_from_env()
    stats = {
        "tagged": 0,
        "failed": 0,
        "model": model_name,
        "prompt": prompt.source if prompt else "default",
    }
    # disable=None -> bar only on a TTY; cron/pytest logs stay clean
    # Re-read only when this item minted a tag the vocabulary did not have.
    # The per-item call was a full GROUP BY over item_tags plus a canonical()
    # pass, so its cost tracked the whole archive rather than the run:
    # measured on the live corpus at 12.3ms with 20k tag rows and 67.9ms with
    # 163k -- 68 SECONDS of database time in a 1000-item run, on top of
    # inference. A fixed refresh interval was the obvious fix and it is wrong:
    # a tag minted on item 1 must steer item 2, which is what
    # test_new_tags_enter_vocabulary_within_a_run asserts. Refreshing exactly
    # when the answer would change keeps that and makes the query rare.
    vocab = tag_vocabulary(conn)
    with tqdm(rows, desc="tagging", unit="item", disable=None) as bar:
        for item in bar:
            before = conn.total_changes
            try:
                tagged = tag_one_item(conn, item, chat_fn, vocab, model_name, prompt)
            except BackendUnreachable:
                # One dead socket would otherwise cost a retry per item for the
                # whole batch and print `failed: N` with the cause -- Ollama is
                # not running -- nowhere in sight. Stop, and say so once.
                stats["chat_down"] = True
                log.warning(
                    "chat model unreachable; stopping -- untagged items wait for the next run"
                )
                break
            if tagged:
                stats["tagged"] += 1
                if conn.total_changes != before:
                    fresh = [
                        r["tag"]
                        for r in conn.execute(
                            "SELECT tag FROM item_tags WHERE item_id = ?", (item["id"],)
                        )
                    ]
                    if _minted_first_use(conn, item["id"], fresh):
                        vocab = tag_vocabulary(conn)
            else:
                stats["failed"] += 1
            bar.set_postfix(tagged=stats["tagged"], failed=stats["failed"], refresh=False)
    return stats


def _key_stats(conn, user_id: int) -> dict[str, list[int]]:
    """Per-feature-key [useful_count, not_useful_count] over this user's clicks."""
    rows = conn.execute(
        "SELECT 'tag:' || t.tag AS key, c.useful FROM clicks c"
        " JOIN item_tags t ON t.item_id = c.item_id WHERE c.user_id = ?"
        " UNION ALL"
        " SELECT 'type:' || f.content_type AS key, c.useful FROM clicks c"
        " JOIN item_features f ON f.item_id = c.item_id WHERE c.user_id = ?"
        " UNION ALL"
        " SELECT 'source:' || i.feed_id AS key, c.useful FROM clicks c"
        " JOIN items i ON i.id = c.item_id WHERE c.user_id = ? AND i.feed_id IS NOT NULL",
        (user_id, user_id, user_id),
    ).fetchall()
    stats: dict[str, list[int]] = {}
    for r in rows:
        u_n = stats.setdefault(r["key"], [0, 0])
        u_n[0 if r["useful"] else 1] += 1
    return stats


def _score(stats: dict[str, list[int]], key: str) -> float:
    u, n = stats.get(key, (0, 0))
    return (u + 1) / (u + n + 2)  # Laplace-smoothed; 0.5 = neutral


def _item_keys(conn, item_ids: list[int]) -> dict[int, list[str]]:
    keys: dict[int, list[str]] = {i: [] for i in item_ids}
    # Chunk below SQLite's SQLITE_LIMIT_VARIABLE_NUMBER (32766): item_ids can be
    # every item in the archive (via pref_scores_for_items <- rank_items).
    for chunk in itertools.batched(item_ids, _SQL_VAR_CHUNK):
        qmarks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT item_id, 'tag:' || tag AS key FROM item_tags WHERE item_id IN ({qmarks})",
            chunk,
        ):
            keys[r["item_id"]].append(r["key"])
        for r in conn.execute(
            f"SELECT item_id, 'type:' || content_type AS key FROM item_features"
            f" WHERE item_id IN ({qmarks})",
            chunk,
        ):
            keys[r["item_id"]].append(r["key"])
        for r in conn.execute(
            f"SELECT id, 'source:' || feed_id AS key FROM items"
            f" WHERE id IN ({qmarks}) AND feed_id IS NOT NULL",
            chunk,
        ):
            keys[r["id"]].append(r["key"])
    return keys


def pref_scores_for_items(conn, user_id: int, item_ids: list[int]) -> np.ndarray:
    """Mean per-key preference score per item, aligned with item_ids. 0.5 = neutral."""
    out = np.full(len(item_ids), 0.5, dtype=np.float64)
    if not item_ids:
        return out
    stats = _key_stats(conn, user_id)
    keys = _item_keys(conn, item_ids)
    for idx, iid in enumerate(item_ids):
        ks = keys[iid]
        if ks:
            out[idx] = float(np.mean([_score(stats, k) for k in ks]))
    return out
