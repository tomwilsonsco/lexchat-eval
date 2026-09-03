"""
Unit tests for scoring against a signed-off reference, and for keeping stored
results tied to the reference version that produced them. No LEX API, no judge.
"""

import pytest

from lex_eval.metrics.citation_agreement import (
    CitationAgreementMetric,
    NO_EXPECTED_CITATIONS_REASON,
    expected_citations,
    reference_acts,
)
from lex_eval.reference.store import (
    apply_review,
    effective_verified,
    reference_version,
    render_markdown,
    review_problems,
    review_state,
)
from lex_eval.testcase import LLMTestCase
from lex_eval.utils.db import (
    clear_outdated_eval_results,
    covered_response_ids,
    get_connection,
    init_eval_table,
    insert_eval_result,
)

pytestmark = pytest.mark.unit


SECTION_6 = "https://www.legislation.gov.uk/ukpga/2018/12/section/6"
SECTION_7 = "https://www.legislation.gov.uk/ukpga/2018/12/section/7"


def _reference(**overrides):
    record = {
        "question_id": 1,
        "question": "Does the duty apply?",
        "research_mode": "legislation_only",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "author": "a drafting model",
        "final_answer": f"See {SECTION_6}, and for background {SECTION_7}.",
        "statements": ["The duty applies."],
        "retrieval_context": ["section 6 text"],
        "sources_retrieved": [
            {"uri": SECTION_6, "legislation_id": "ukpga/2018/12"},
            {"uri": SECTION_7, "legislation_id": "ukpga/2018/12"},
        ],
    }
    record.update(overrides)
    apply_review(record, record.get("review"))
    return record


def _signed(required=(SECTION_6,), **overrides):
    record = _reference(**overrides)
    apply_review(
        record,
        {
            "verified": True,
            "verified_by": "A Lawyer",
            "verified_at": "2026-02-01",
            "verdict": "Approve",
            "citations_reviewed": True,
            "required_citations": list(required),
        },
    )
    return record


# ---------------------------------------------------------------------------
# What a response is expected to cite
# ---------------------------------------------------------------------------


def test_a_draft_expects_every_link_in_its_answer():
    expected, mode = expected_citations(_reference())

    assert mode == "draft"
    assert expected == {"ukpga/2018/12/section/6", "ukpga/2018/12/section/7"}


def test_a_signed_reference_expects_only_the_approved_citations():
    """The lawyer's decision, not the author's reading list."""
    expected, mode = expected_citations(_signed())

    assert mode == "approved"
    assert expected == {"ukpga/2018/12/section/6"}


def test_a_stale_sign_off_falls_back_to_the_draft_expectation():
    stale = _signed()
    stale["final_answer"] += " With a later correction."

    _expected, mode = expected_citations(stale)

    assert mode == "draft"


def test_attribution_follows_the_approved_citations():
    """The Acts blame is attributed over come from the same decision."""
    assert reference_acts(_signed()) == {"ukpga/2018/12"}
    assert reference_acts(_signed(required=[])) == set()


def test_a_response_citing_every_required_provision_scores_one():
    metric = CitationAgreementMetric(
        reference_answer="unused",
        threshold=1.0,
        required_citations={"ukpga/2018/12/section/6"},
    )

    metric.measure(LLMTestCase(input="q", actual_output=f"Cited: {SECTION_6}"))

    assert metric.score == 1.0
    assert metric.is_successful()


def test_missing_one_required_provision_fails_at_a_threshold_of_one():
    metric = CitationAgreementMetric(
        reference_answer="unused",
        threshold=1.0,
        required_citations={"ukpga/2018/12/section/6", "ukpga/2018/12/section/7"},
    )

    metric.measure(LLMTestCase(input="q", actual_output=f"Cited: {SECTION_6}"))

    assert metric.score == 0.5
    assert not metric.is_successful()


def test_an_approved_empty_citation_set_is_not_measured_rather_than_zero():
    """ "No citation is mandatory here" is a decision, not a failed response."""
    metric = CitationAgreementMetric(
        reference_answer="unused", required_citations=set()
    )

    metric.measure(LLMTestCase(input="q", actual_output=f"Cited: {SECTION_6}"))

    assert metric.reason == NO_EXPECTED_CITATIONS_REASON


# ---------------------------------------------------------------------------
# Stored results know which reference they came from
# ---------------------------------------------------------------------------


def _row(conn, metric, response_id, reference=None, question_id=1):
    sha, mode = reference_version(reference) if reference else (None, None)
    insert_eval_result(
        conn,
        metric,
        {
            "response_id": response_id,
            "llm_name": "some-model",
            "question_id": question_id,
            "question": "Does the duty apply?",
            "score": 1.0,
            "threshold": 0.3,
            "passed": True,
            "reference_sha256": sha,
            "reference_mode": mode,
        },
    )


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(tmp_path / "test.db")
    init_eval_table(connection, "citation_agreement")
    yield connection
    connection.close()


def test_a_row_scored_against_the_current_reference_covers_its_response(conn):
    reference = _reference()
    _row(conn, "citation_agreement", 1, reference)

    versions = {1: reference_version(reference)}
    assert covered_response_ids(conn, "citation_agreement", versions) == {1}


def test_a_row_scored_against_a_corrected_answer_no_longer_covers_it(conn):
    """The point of the whole exercise: an edited answer forces a re-score."""
    _row(conn, "citation_agreement", 1, _reference())
    corrected = _reference(final_answer=f"Now says something else. {SECTION_6}")

    versions = {1: reference_version(corrected)}
    assert covered_response_ids(conn, "citation_agreement", versions) == set()


def test_signing_a_reference_off_forces_a_re_score(conn):
    _row(conn, "citation_agreement", 1, _reference())

    versions = {1: reference_version(_signed())}
    assert covered_response_ids(conn, "citation_agreement", versions) == set()


def test_rows_written_before_the_columns_existed_are_not_treated_as_current(conn):
    _row(conn, "citation_agreement", 1)

    versions = {1: reference_version(_reference())}
    assert covered_response_ids(conn, "citation_agreement", versions) == set()


def test_a_question_with_no_reference_stays_covered(conn):
    """Nothing to be out of date against, so nothing to re-run."""
    _row(conn, "citation_agreement", 1, question_id=42)

    assert covered_response_ids(conn, "citation_agreement", {}) == {1}


def test_metrics_that_use_no_reference_are_unaffected(conn):
    init_eval_table(conn, "tool_usage")
    _row(conn, "tool_usage", 1)

    assert covered_response_ids(conn, "tool_usage") == {1}


def test_outdated_rows_are_replaced_not_left_beside_the_new_ones(conn):
    _row(conn, "citation_agreement", 1, _reference())
    _row(conn, "citation_agreement", 2, _signed())

    dropped = clear_outdated_eval_results(
        conn, "citation_agreement", {1: reference_version(_signed())}
    )

    rows = conn.execute("SELECT response_id FROM eval_citation_agreement").fetchall()
    assert dropped == 1
    assert [r[0] for r in rows] == [2]


# ---------------------------------------------------------------------------
# What it takes for an approval to count
# ---------------------------------------------------------------------------


def _approved(**review_overrides):
    record = _reference()
    review = {
        "verified": True,
        "verified_by": "A Lawyer",
        "verified_at": "2026-02-01",
        "verdict": "Approve",
        "citations_reviewed": True,
        "required_citations": [SECTION_6],
    }
    review.update(review_overrides)
    apply_review(record, review)
    return record


def test_an_approval_that_says_approve_counts():
    assert review_state(_approved()) == "Verified"
    assert effective_verified(_approved())


def test_verified_true_without_a_decision_does_not_count():
    """`verified: true` alone is somebody's typing, not a lawyer's decision."""
    record = _approved(verdict=None)

    assert review_state(record) == "Sign-off unusable"
    assert not effective_verified(record)
    assert "the decision is missing" in review_problems(record)[0]


def test_changes_required_is_never_treated_as_approval():
    record = _approved(verdict="Changes required")

    assert review_state(record) == "Changes required"
    assert not effective_verified(record)


@pytest.mark.parametrize(
    "citation",
    ["https://example.com/asp/2009/12", "The Data Protection Act 2018", "not/a/thing"],
    ids=["wrong-domain", "prose", "malformed"],
)
def test_a_required_citation_that_is_not_a_provision_blocks_the_approval(citation):
    record = _approved(required_citations=[citation])

    assert not effective_verified(record)
    assert "not a legislation.gov.uk provision" in review_problems(record)[0]


def test_a_required_citation_the_answer_never_makes_blocks_the_approval():
    """The approved expectation has to be something the reference itself does."""
    record = _approved(required_citations=["ukpga/1998/29/section/1"])

    assert not effective_verified(record)
    assert "not cited in the answer" in review_problems(record)[0]


def test_a_required_citation_the_research_never_read_blocks_the_approval():
    record = _reference(
        final_answer=f"See {SECTION_6} and https://www.legislation.gov.uk/ukpga/1998/29",
        sources_retrieved=[{"uri": SECTION_6, "legislation_id": "ukpga/2018/12"}],
    )
    apply_review(
        record,
        {
            "verified": True,
            "verified_by": "A Lawyer",
            "verified_at": "2026-02-01",
            "verdict": "Approve",
            "citations_reviewed": True,
            "required_citations": ["ukpga/1998/29"],
        },
    )

    assert not effective_verified(record)
    assert "never retrieved" in review_problems(record)[0]


JUDGMENT = "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150"


def test_a_judgment_the_research_never_read_blocks_the_approval():
    """No metric scores a case citation, so sign-off is the only place to catch one.

    A case the author never opened, cited as if it had been, would otherwise
    reach the eval as part of an approved reference answer.
    """
    record = _reference(
        research_mode="legislation_and_case_law",
        final_answer=f"See {SECTION_6} and {JUDGMENT}.",
    )
    apply_review(
        record,
        {
            "verified": True,
            "verified_by": "A Lawyer",
            "verified_at": "2026-02-01",
            "verdict": "Approve",
            "citations_reviewed": True,
            "required_citations": [SECTION_6],
        },
    )

    assert not effective_verified(record)
    assert "never read" in review_problems(record)[0]


def test_a_judgment_the_research_read_does_not_block_the_approval():
    record = _reference(
        research_mode="legislation_and_case_law",
        final_answer=f"See {SECTION_6} and {JUDGMENT}.",
        cases_retrieved=[{"ncn": "[2025] EWCA Crim 1150", "url": JUDGMENT}],
    )
    apply_review(
        record,
        {
            "verified": True,
            "verified_by": "A Lawyer",
            "verified_at": "2026-02-01",
            "verdict": "Approve",
            "citations_reviewed": True,
            "required_citations": [SECTION_6],
        },
    )

    assert review_problems(record) == []
    assert effective_verified(record)


def test_an_act_level_requirement_is_met_by_a_section_of_that_act():
    """Requiring the Act does not require a link to the Act's own front page."""
    record = _approved(required_citations=["ukpga/2018/12"])

    assert review_problems(record) == []
    assert effective_verified(record)


def test_an_unusable_approval_falls_back_to_the_draft_citation_expectation():
    """The blocked case: scoring against a baseline the reference cannot support."""
    record = _approved(required_citations=["ukpga/1998/29/section/1"])

    expected, mode = expected_citations(record)

    assert mode == "draft"
    assert expected == {"ukpga/2018/12/section/6", "ukpga/2018/12/section/7"}


def test_the_document_says_why_an_approval_cannot_be_used():
    record = _approved(verdict=None)

    markdown = render_markdown(record)

    assert "**Status: Sign-off unusable.**" in markdown
    assert "The approval on file cannot be used" in markdown
    assert "the decision is missing" in markdown


def test_a_draft_document_tells_the_reviewer_who_drafted_it_and_why_they_matter():
    """A reviewer must not think a draft written by a model is already authority."""
    text = " ".join(render_markdown(_reference()).split())

    assert "drafted from that retrieved text by a drafting model" in text
    assert "Nobody legally qualified has checked it." in text
    assert "Why this is required:" in text
    assert (
        "Your sign-off is what turns this from a draft into a legal benchmark" in text
    )


def test_a_regnal_year_provision_can_be_marked_required():
    """Acts before 1963 are dated by regnal year, not by calendar year.

    Rejecting that id shape would stop a lawyer requiring any provision of, say,
    the Occupiers' Liability Act 1957.
    """
    from lex_eval.reference.store import citation_id

    assert citation_id("ukpga/Eliz2/5-6/31/section/1") == "ukpga/eliz2/5-6/31/section/1"
    assert (
        citation_id("http://www.legislation.gov.uk/id/ukpga/Eliz2/8-9/30")
        == "ukpga/eliz2/8-9/30"
    )
    assert citation_id("not/a/thing") is None
