"""
Compare LexChat responses against the authored reference ("gold") answers.

Three metrics, all anchored to the reference answer for the same question:

  - CitationAgreementMetric      : does the response cite the legislation the
                                   reference answer cites? (no AI judge)
  - ReferenceAnswerAgreementMetric : does the response make the statements
                                     written alongside the reference answer,
                                     and contradict none of them?
  - PlanCoverageMetric            : deep_research only. Does the approved
                                     research plan set out to cover those same
                                     statements, before any research happens?

Signed and unsigned reference answers are both used. An unverified answer's
scores carry a "[DRAFT REFERENCE - unverified]" note, because such a score
measures agreement with the answer's author, not legal correctness. Each row
records which version of the reference it was scored against, so a score taken
against an answer that has since been corrected is re-run, not believed.
"""

import pytest
from lex_eval.testcase import LLMTestCase

from lex_eval.metrics import (
    CitationAgreementMetric,
    PlanCoverageMetric,
    ReferenceAnswerAgreementMetric,
)
from lex_eval.metrics.citation_agreement import (
    MIN_OUTPUT_CHARS,
    NO_EXPECTED_CITATIONS_REASON,
    expected_citations,
    reference_acts,
)
from lex_eval.reference.store import effective_verified, load_reference_answers
from lex_eval.utils.collector import attach_metric
from lex_eval.utils.judge import _judge
from lex_eval.utils.test_helpers import (
    load_records,
    record_id,
    record_to_test_case,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_COVERAGE_THRESHOLD: float = 0.3
# Signed-off references expect only the citations the lawyer marked required,
# so anything short of all of them is a miss. The draft threshold above is low
# for the opposite reason: a draft's expectation is every link in its answer,
# background material included.
_REQUIRED_CITATION_THRESHOLD: float = 1.0
_AGREEMENT_THRESHOLD: float = 0.6
# Same value as _AGREEMENT_THRESHOLD by convention ("at least 3 of 5"), kept as
# its own constant since it's a different question (does the plan set out to
# cover the points, not does the response make them) and may need to diverge.
_PLAN_COVERAGE_THRESHOLD: float = 0.6

# Same threshold as test_groundedness.py's gate, so a non-answer like "Could
# you narrow this down?" is recorded as a capture event here too, not scored
# as a real verdict. Imported rather than repeated because reports/attribution.py
# gates on it too, and the two must not drift.
_MIN_OUTPUT_CHARS: int = MIN_OUTPUT_CHARS

_DRAFT_NOTE = "[DRAFT REFERENCE - unverified]"

# Written at the front of the reason when a question has no reference answer,
# so reports/streamlit_report.py keeps the row out of the mean.
_NO_REFERENCE = "No reference answer for this question;"
_NO_STATEMENTS = "No reference statements for this question;"
_NO_PLAN = "No research plan for this record;"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

records = load_records(read_only=True)

# Signed and unsigned references, which is the default: verified-only loading
# would leave almost every record unscored. A draft's scores are stamped instead.
references = load_reference_answers()

_skip_no_api_key = pytest.mark.skipif(
    _judge is None,
    reason="Configured judge API key not set (check lex_eval/.env)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stamp(reference: dict, reason: str) -> str:
    """Append the draft note when the reference answer is not signed off.

    Appended rather than prefixed because the dashboard identifies rows that
    are not a real quality verdict by the start of the reason.
    """
    if effective_verified(reference):
        return reason
    return f"{reason} {_DRAFT_NOTE}"


def _gate_output_length(request, record, test_name, metric_name, threshold):
    """Fail fast if the output is too short to be meaningful."""
    char_count = len((record.get("actual_output") or "").strip())
    if char_count <= _MIN_OUTPUT_CHARS:
        reason = (
            f"Output too short ({char_count} chars ≤ {_MIN_OUTPUT_CHARS}); "
            f"{metric_name} scored 0"
        )
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=threshold,
            passed=False,
            reason=reason,
        )
        return False, reason
    return True, ""


def _gate_reference(request, record, test_name, metric_name, threshold):
    """Fail fast if no reference answer exists for this question."""
    reference = references.get(record["question_id"])
    if reference and (reference.get("final_answer") or "").strip():
        return reference, ""

    reason = f"{_NO_REFERENCE} {metric_name} not measured"
    _attach_not_measured(request, record, test_name, metric_name, threshold, reason)
    return None, reason


def _gate_statements(request, record, reference, test_name, metric_name, threshold):
    """Fail fast if the reference answer has no statements written for it yet."""
    statements = reference.get("statements") or []
    if statements:
        return statements, ""

    qid = record["question_id"]
    reason = (
        f"{_NO_STATEMENTS} {metric_name} not measured. Write them in "
        f"reference_answers/.authored/q{qid}/statements.json."
    )
    _attach_not_measured(request, record, test_name, metric_name, threshold, reason)
    return None, reason


def _gate_expected_citations(
    request, record, reference, test_name, metric_name, threshold
):
    """Fail fast if the reference cites no legislation for a response to match.

    A `case_law_only` reference is the ordinary case: it cites judgments, and
    this metric reads legislation.gov.uk provisions only. Without this the
    metric scores 0.0 and the test fails, which reads as the response citing
    the wrong law rather than as nothing having been measured.
    """
    expected, expectation = expected_citations(reference)
    if expected:
        return expected, expectation, ""

    _attach_not_measured(
        request, record, test_name, metric_name, threshold, NO_EXPECTED_CITATIONS_REASON
    )
    return None, expectation, NO_EXPECTED_CITATIONS_REASON


def _gate_research_plan(request, record, test_name, metric_name, threshold):
    """Fail fast if this isn't a deep-research response with an approved plan."""
    plan = record.get("research_plan") or {}
    steps = plan.get("steps") or []
    if record.get("chat_mode") == "deep_research" and steps:
        return steps, ""

    reason = f"{_NO_PLAN} {metric_name} not measured"
    _attach_not_measured(request, record, test_name, metric_name, threshold, reason)
    return None, reason


def _attach_not_measured(request, record, test_name, metric_name, threshold, reason):
    """Record a row that carries no verdict, keeping it out of the dashboard mean."""
    attach_metric(
        request,
        record=record,
        test_name=test_name,
        metric_name=metric_name,
        score=0.0,
        threshold=threshold,
        passed=False,
        reason=reason,
        reference=references.get(record["question_id"]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
def test_citation_agreement(request, record):
    """
    The response must cite the legislation the reference answer cites.

    Pre-flight gate: output must be > 50 chars.
    """
    ok, reason = _gate_output_length(
        request,
        record,
        "citation_agreement",
        "Citation Agreement",
        _COVERAGE_THRESHOLD,
    )
    if not ok:
        pytest.skip(reason)

    reference, reason = _gate_reference(
        request,
        record,
        "citation_agreement",
        "Citation Agreement",
        _COVERAGE_THRESHOLD,
    )
    if reference is None:
        pytest.skip(reason)

    # A signed-off reference is scored against the citations the lawyer called
    # required; a draft against every legislation link in its answer.
    expected, expectation, reason = _gate_expected_citations(
        request,
        record,
        reference,
        "citation_agreement",
        "Citation Agreement",
        _COVERAGE_THRESHOLD,
    )
    if expected is None:
        pytest.skip(reason)

    test_case: LLMTestCase = record_to_test_case(record)
    approved = expectation == "approved"
    metric = CitationAgreementMetric(
        reference_answer=reference["final_answer"],
        threshold=_REQUIRED_CITATION_THRESHOLD if approved else _COVERAGE_THRESHOLD,
        expected_acts=reference_acts(reference),
        required_citations=expected if approved else None,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="citation_agreement",
        metric_name="Citation Agreement",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=_stamp(reference, metric.reason or ""),
        reference=reference,
    )

    assert metric.is_successful(), (
        f"Citation Agreement score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@_skip_no_api_key
def test_reference_answer_agreement(request, record):
    """
    The response must make the reference answer's main points, and contradict
    none of them.

    Pre-flight gate: output must be > 50 chars.
    """
    ok, reason = _gate_output_length(
        request,
        record,
        "reference_answer_agreement",
        "Reference Answer Agreement",
        _AGREEMENT_THRESHOLD,
    )
    if not ok:
        pytest.skip(reason)

    reference, reason = _gate_reference(
        request,
        record,
        "reference_answer_agreement",
        "Reference Answer Agreement",
        _AGREEMENT_THRESHOLD,
    )
    if reference is None:
        pytest.skip(reason)

    statements, reason = _gate_statements(
        request,
        record,
        reference,
        "reference_answer_agreement",
        "Reference Answer Agreement",
        _AGREEMENT_THRESHOLD,
    )
    if statements is None:
        pytest.skip(reason)

    test_case: LLMTestCase = record_to_test_case(record)
    metric = ReferenceAnswerAgreementMetric(
        statements=statements,
        model=_judge,
        threshold=_AGREEMENT_THRESHOLD,
    )
    _judge.last_model, _judge.total_usage_tokens = None, None
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="reference_answer_agreement",
        metric_name="Reference Answer Agreement",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=_stamp(reference, metric.reason or ""),
        judge_llm=_judge.last_model,
        judge_tokens=_judge.total_usage_tokens,
        reference=reference,
    )

    assert metric.is_successful(), (
        f"Reference Answer Agreement score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@_skip_no_api_key
def test_plan_coverage(request, record):
    """
    The approved deep-research plan must set out to cover the reference
    answer's main points, before any research happens.

    Pre-flight gates: this must be a deep_research response with an approved
    plan, and the question must have a reference answer with statements. No
    output-length gate, this metric scores the plan, not the response.
    """
    plan_steps, reason = _gate_research_plan(
        request,
        record,
        "plan_coverage",
        "Plan Coverage",
        _PLAN_COVERAGE_THRESHOLD,
    )
    if plan_steps is None:
        pytest.skip(reason)

    reference, reason = _gate_reference(
        request,
        record,
        "plan_coverage",
        "Plan Coverage",
        _PLAN_COVERAGE_THRESHOLD,
    )
    if reference is None:
        pytest.skip(reason)

    statements, reason = _gate_statements(
        request,
        record,
        reference,
        "plan_coverage",
        "Plan Coverage",
        _PLAN_COVERAGE_THRESHOLD,
    )
    if statements is None:
        pytest.skip(reason)

    test_case: LLMTestCase = record_to_test_case(record)
    metric = PlanCoverageMetric(
        plan_steps=plan_steps,
        statements=statements,
        model=_judge,
        threshold=_PLAN_COVERAGE_THRESHOLD,
    )
    _judge.last_model, _judge.total_usage_tokens = None, None
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="plan_coverage",
        metric_name="Plan Coverage",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=_stamp(reference, metric.reason or ""),
        judge_llm=_judge.last_model,
        judge_tokens=_judge.total_usage_tokens,
        reference=reference,
    )

    assert (
        metric.is_successful()
    ), f"Plan Coverage score {metric.score:.2f} < {metric.threshold}: {metric.reason}"
