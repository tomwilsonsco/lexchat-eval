"""
Unit tests for the generated lawyer view of a reference answer, and for the
offline path that regenerates it. No LEX API, no judge, no DB.

The property under test throughout is that `q{id}.md` is a view of the manifest
record, never a second copy of the answer that can drift away from it.
"""

import json

import pytest

from lex_eval.reference.build import render_only, resync
from lex_eval.reference.store import (
    ANSWERS_DIR,
    MANIFEST_NAME,
    answer_section,
    is_built,
    load_manifest,
    render_markdown,
)

pytestmark = pytest.mark.unit


ANSWER = (
    "# Whether the duty applies\n\n"
    "## 1. Summary Answer (BLUF)\n\n**Yes.** See "
    "[section 6](https://www.legislation.gov.uk/ukpga/2018/12/section/6).\n"
)
STATEMENTS = [
    "The duty applies to a controller established in the United Kingdom.",
    "Section 6 defines the terms used in the duty.",
]


def _record(**overrides):
    record = {
        "question_id": 99,
        "question": "Does the duty apply?",
        "research_mode": "legislation_only",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "author": "a tester",
        "plan": {"scope_note": "Covers the duty only.", "steps": []},
        "statements": list(STATEMENTS),
        "research_output": ANSWER,
        "final_answer": ANSWER,
        "tool_sequence": ["search_legislation"],
        "tools_called": [],
        "retrieval_context": [],
        "sources_retrieved": [
            {
                "uri": "http://www.legislation.gov.uk/id/ukpga/2018/12/section/6",
                "title": "Lawfulness of processing",
                "legislation_id": "ukpga/2018/12",
                "extent": ["E+W+S+N.I."],
            }
        ],
        "sources_discovered": [],
        "fallback_used": False,
        "lex_api_calls": 1,
        "review": {"verified": False, "stale": False, "required_citations": []},
    }
    record.update(overrides)
    return record


def _authored(tmp_path, answer=ANSWER, statements=STATEMENTS):
    src = tmp_path / ".authored" / "q99"
    src.mkdir(parents=True)
    (src / "answer.md").write_text(answer, encoding="utf-8")
    (src / "statements.json").write_text(
        json.dumps({"statements": statements}), encoding="utf-8"
    )
    return src


def _seed(tmp_path, record):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps([record], indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The Markdown is a view of the record
# ---------------------------------------------------------------------------


def test_rendered_answer_is_the_record_answer_exactly():
    """A lawyer must be approving the text the judge is scored against."""
    assert answer_section(render_markdown(_record())) == ANSWER.strip()


def test_rendered_statements_are_the_record_statements_in_order():
    markdown = render_markdown(_record())

    positions = [markdown.index(s) for s in STATEMENTS]
    assert positions == sorted(positions)


def test_answer_comes_before_the_research_trail():
    """The reviewer reads the answer first; provenance is an appendix."""
    markdown = render_markdown(_record())

    assert markdown.index("## 2. Reference answer") < markdown.index("## 6. Appendix")


def test_citation_schedule_lists_the_links_the_metric_expects():
    """The schedule is built with Citation Agreement's own parser."""
    markdown = render_markdown(_record())

    assert "ukpga/2018/12/section/6" in markdown
    assert "Lawfulness of processing" in markdown


def test_an_answer_with_no_links_says_so_rather_than_showing_an_empty_table():
    markdown = render_markdown(_record(final_answer="No links at all."))

    assert "no legislation.gov.uk links" in markdown


# ---------------------------------------------------------------------------
# Offline resync
# ---------------------------------------------------------------------------


def test_resync_takes_the_authored_answer_and_statements(tmp_path):
    src = _authored(tmp_path, answer="# Corrected\n\nThe duty does not apply.")

    updated = resync(_record(), src)

    assert updated["final_answer"] == "# Corrected\n\nThe duty does not apply."
    assert updated["research_output"] == updated["final_answer"]
    assert updated["statements"] == STATEMENTS


def test_resync_keeps_the_retrieval_audit(tmp_path):
    """A correction must not silently re-run the searches behind the answer."""
    src = _authored(tmp_path)
    record = _record()

    updated = resync(record, src)

    assert updated["sources_retrieved"] == record["sources_retrieved"]
    assert updated["tool_sequence"] == record["tool_sequence"]


def test_render_only_makes_no_lex_call(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("render-only called LEX")

    monkeypatch.setattr("lex_eval.reference.build.LexTools", explode)
    _authored(tmp_path)
    _seed(tmp_path, _record())

    assert render_only(tmp_path, None) == 0
    assert answer_section((tmp_path / "q99.md").read_text(encoding="utf-8"))


def test_render_only_writes_neither_output_when_an_invariant_breaks(tmp_path):
    """One bad record leaves its manifest entry and its Markdown untouched."""
    _authored(tmp_path, statements=[])
    _seed(tmp_path, _record())

    assert render_only(tmp_path, None) == 1
    assert load_manifest(tmp_path)[0]["final_answer"] == ANSWER
    assert not (tmp_path / "q99.md").exists()


# ---------------------------------------------------------------------------
# The record, not the Markdown, says whether a question is answered
# ---------------------------------------------------------------------------


def test_a_complete_record_is_built_without_its_markdown():
    assert is_built(_record())


def test_a_record_with_no_answer_or_no_statements_is_not_built():
    assert not is_built(_record(final_answer=""))
    assert not is_built(_record(statements=[]))
    assert not is_built(None)


# ---------------------------------------------------------------------------
# The committed set stays in sync
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record", load_manifest(), ids=lambda r: f"q{r['question_id']}"
)
def test_committed_answers_match_their_authored_source_and_markdown(record):
    """Every committed reference: one answer, in all four places it appears.

    Run `python -m lex_eval.reference.build --render-only` if this fails.
    """
    qid = record["question_id"]
    authored = ANSWERS_DIR / ".authored" / f"q{qid}" / "answer.md"
    markdown = ANSWERS_DIR / f"q{qid}.md"

    assert authored.read_text(encoding="utf-8").strip() == record["final_answer"]
    assert record["research_output"] == record["final_answer"]
    assert (
        answer_section(markdown.read_text(encoding="utf-8"))
        == record["final_answer"].strip()
    )
