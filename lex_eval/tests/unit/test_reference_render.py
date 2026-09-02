"""
Unit tests for the generated lawyer view of a reference answer, and for the
offline path that regenerates it. No LEX API, no judge, no DB.

The property under test throughout is that `q{id}.md` is a view of the manifest
record, never a second copy of the answer that can drift away from it.
"""

import json

import pytest

from lex_eval.reference.build import read_review, render_only, resync, write_review
from lex_eval.reference.store import (
    ANSWERS_DIR,
    APPROVE,
    MANIFEST_NAME,
    REVIEW_NAME,
    answer_section,
    apply_review,
    effective_verified,
    is_built,
    load_manifest,
    load_reference_answers,
    new_review_block,
    reference_fingerprint,
    render_markdown,
    review_state,
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


# ---------------------------------------------------------------------------
# The fingerprint covers what an approval rests on
# ---------------------------------------------------------------------------


def _signed(**overrides):
    """A record with a complete, current sign-off."""
    record = _record(**overrides)
    apply_review(
        record,
        {
            "verified": True,
            "verified_by": "A Lawyer",
            "verified_at": "2026-02-01",
            "verdict": APPROVE,
            "citations_reviewed": True,
            "required_citations": [
                "https://www.legislation.gov.uk/ukpga/2018/12/section/6"
            ],
        },
    )
    return record


@pytest.mark.parametrize(
    "change",
    [
        {"question": "Does some other duty apply?"},
        {"final_answer": ANSWER + "\n\nAnd one more thing.\n"},
        {"statements": STATEMENTS[:1]},
        {"statements": list(reversed(STATEMENTS))},
        {"research_mode": "legislation_and_case_law"},
        {"retrieval_context": ["different text entirely"]},
        {"sources_retrieved": []},
    ],
    ids=[
        "question",
        "answer",
        "statements-dropped",
        "statements-reordered",
        "mode",
        "retrieved-text",
        "sources",
    ],
)
def test_changing_approved_material_changes_the_fingerprint(change):
    assert reference_fingerprint(_record(**change)) != reference_fingerprint(_record())


def test_changing_the_required_citations_changes_the_fingerprint():
    """The approved citation set is part of what was approved."""
    other = _signed()
    other["review"]["required_citations"] = ["ukpga/2018/12/section/7"]

    assert reference_fingerprint(other) != reference_fingerprint(_signed())


@pytest.mark.parametrize(
    "change",
    [
        {"generated_at": "2030-01-01T00:00:00+00:00"},
        {"author": "somebody else"},
        {"tool_sequence": ["search_legislation", "search_legislation_sections"]},
        {"sources_discovered": [{"legislation_id": "ukpga/1998/29", "title": "DPA"}]},
        {"lex_api_calls": 99},
    ],
    ids=["written-at", "author", "tool-order", "discovered", "call-count"],
)
def test_presentation_and_provenance_do_not_change_the_fingerprint(change):
    assert reference_fingerprint(_record(**change)) == reference_fingerprint(_record())


def test_reviewer_notes_do_not_change_the_fingerprint():
    noted = _record()
    noted["review"] = {"corrections": "typo in para 3", "notes": "seen by counsel"}

    assert reference_fingerprint(noted) == reference_fingerprint(_record())


def test_citation_urls_and_bare_ids_normalise_to_the_same_fingerprint():
    by_url = _signed()
    by_id = _signed()
    by_id["review"]["required_citations"] = ["ukpga/2018/12/section/6"]

    assert reference_fingerprint(by_id) == reference_fingerprint(by_url)


# ---------------------------------------------------------------------------
# Only a complete, current sign-off counts
# ---------------------------------------------------------------------------


def test_a_current_signed_record_is_verified():
    assert review_state(_signed()) == "Verified"
    assert effective_verified(_signed())


def test_a_record_signed_against_an_older_answer_is_stale():
    record = _signed()
    record["final_answer"] = ANSWER + "\n\nA later correction.\n"

    assert review_state(record) == "Stale"
    assert not effective_verified(record)


@pytest.mark.parametrize("missing", ["verified_by", "verified_at"])
def test_a_sign_off_without_a_reviewer_or_date_is_not_verified(missing):
    record = _signed()
    record["review"][missing] = None

    assert review_state(record) == "Sign-off incomplete"
    assert not effective_verified(record)


def test_a_sign_off_without_the_citation_decision_is_not_verified():
    """Approving an answer is not the same as saying which citations are required."""
    record = _signed()
    record["review"]["citations_reviewed"] = False

    assert not effective_verified(record)


def test_verified_true_typed_in_by_hand_is_not_verified():
    record = _record()
    record["review"] = {"verified": True}

    assert not effective_verified(record)


def test_drafts_load_by_default_and_verified_only_excludes_stale_approvals(tmp_path):
    stale = _signed(question_id=98)
    stale["final_answer"] = "changed after sign-off"
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps([_signed(), stale], indent=2), encoding="utf-8"
    )

    assert list(load_reference_answers(tmp_path, verified_only=True)) == [99]
    assert sorted(load_reference_answers(tmp_path)) == [98, 99]


def test_a_new_approval_is_stamped_with_the_version_in_front_of_it():
    record = _record()

    apply_review(
        record,
        {
            "verified": True,
            "verified_by": "A Lawyer",
            "verified_at": "2026-02-01",
            "citations_reviewed": True,
        },
    )

    assert record["review"]["signed_reference_sha256"] == record["reference_sha256"]
    assert effective_verified(record)


def test_an_existing_signature_is_never_restamped():
    """Re-rendering must not quietly re-approve an answer that moved."""
    record = _signed()
    record["final_answer"] = "changed after sign-off"

    apply_review(record, record["review"])

    assert record["review"]["signed_reference_sha256"] != record["reference_sha256"]
    assert review_state(record) == "Stale"


def test_the_old_answer_hash_and_stale_flag_are_dropped():
    """Staleness is derived now, so a stored `stale: false` cannot override it."""
    record = _signed()
    record["review"]["stale"] = False
    record["review"]["answer_sha256"] = "0000000000000000"
    record["final_answer"] = "changed after sign-off"

    apply_review(record, record["review"])

    assert "stale" not in record["review"]
    assert "answer_sha256" not in record["review"]
    assert review_state(record) == "Stale"


# ---------------------------------------------------------------------------
# The review reaches the lawyer's document
# ---------------------------------------------------------------------------


def test_a_completed_review_is_shown_rather_than_a_blank_form():
    record = _signed()
    record["review"]["corrections"] = "para 3 overstates the duty"
    record["review"]["notes"] = "checked against the 2024 amendments"

    markdown = render_markdown(record)

    assert "A Lawyer" in markdown
    assert "2026-02-01" in markdown
    assert APPROVE in markdown
    assert "para 3 overstates the duty" in markdown
    assert "checked against the 2024 amendments" in markdown
    assert "Accept / Amend" not in markdown


def test_the_approved_citations_are_marked_in_the_schedule():
    markdown = render_markdown(_signed())

    assert "**Required**" in markdown


def test_a_stale_sign_off_says_so_at_the_top():
    record = _signed()
    record["final_answer"] = "changed after sign-off"

    markdown = render_markdown(record)

    assert "**Status: Stale.**" in markdown
    assert "reviewer needs to see it again" in markdown


def test_an_unreviewed_record_says_nobody_has_reviewed_it():
    markdown = render_markdown(_record())

    assert "**Status: Draft.**" in markdown
    assert "_not yet reviewed_" in markdown


# ---------------------------------------------------------------------------
# review.json is where a decision lives
# ---------------------------------------------------------------------------


def test_resync_takes_the_decision_from_review_json(tmp_path):
    src = _authored(tmp_path)
    (src / REVIEW_NAME).write_text(
        json.dumps(
            {
                "verified": True,
                "verified_by": "A Lawyer",
                "verified_at": "2026-02-01",
                "verdict": APPROVE,
                "citations_reviewed": True,
                "required_citations": ["ukpga/2018/12/section/6"],
            }
        ),
        encoding="utf-8",
    )

    updated = resync(_record(), src)

    assert effective_verified(updated)
    assert updated["review"]["verified_by"] == "A Lawyer"


def test_render_only_writes_review_json_for_a_record_that_predates_it(tmp_path):
    _authored(tmp_path)
    _seed(tmp_path, _record())

    assert render_only(tmp_path, None) == 0

    written = json.loads(
        (tmp_path / ".authored" / "q99" / REVIEW_NAME).read_text(encoding="utf-8")
    )
    assert written == new_review_block()


def test_a_partial_review_file_keeps_its_decision_when_it_is_normalised(tmp_path):
    src = _authored(tmp_path)
    (src / REVIEW_NAME).write_text(
        json.dumps({"verified_by": "A Lawyer"}), encoding="utf-8"
    )

    write_review(src, read_review(src))

    assert read_review(src)["verified_by"] == "A Lawyer"


def test_an_approval_is_stamped_on_file_so_a_later_edit_goes_stale(tmp_path):
    """The failure this catches: re-rendering re-approving an answer that moved."""
    src = _authored(tmp_path)
    (src / REVIEW_NAME).write_text(
        json.dumps(
            {
                "verified": True,
                "verified_by": "A Lawyer",
                "verified_at": "2026-02-01",
                "verdict": APPROVE,
                "citations_reviewed": True,
            }
        ),
        encoding="utf-8",
    )
    _seed(tmp_path, _record())
    assert render_only(tmp_path, None) == 0
    assert read_review(src)["signed_reference_sha256"]

    (src / "answer.md").write_text(ANSWER + "\n\nA later correction.\n", "utf-8")
    assert render_only(tmp_path, None) == 0

    assert review_state(load_manifest(tmp_path)[0]) == "Stale"
