"""
Compare LexChat responses against the hand written reference ("gold") answers.

Two metrics, both anchored to the reference answer for the same question:

  - CitationAgreementMetric      : does the response cite the legislation the
                                   reference answer cites? (no AI judge)
  - ReferenceAnswerAgreementMetric : does the response make the statements
                                     written alongside the reference answer,
                                     and contradict none of them?

Reference answers are drafts until a lawyer signs one off, so an unverified
answer's scores carry a "[DRAFT REFERENCE - unverified]" note. Such a score
measures agreement with the answer's author, not legal correctness.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics import (
    CitationAgreementMetric,
    ReferenceAnswerAgreementMetric,
)
from lex_eval.reference.store import load_reference_answers
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
_AGREEMENT_THRESHOLD: float = 0.6

# Same threshold as test_groundedness.py's gate, so a non-answer like "Could
# you narrow this down?" is recorded as a capture event here too, not scored
# as a real verdict.
_MIN_OUTPUT_CHARS: int = 50

_DRAFT_NOTE = "[DRAFT REFERENCE - unverified]"

# Written at the front of the reason when a question has no reference answer,
# so reports/streamlit_report.py keeps the row out of the mean.
_NO_REFERENCE = "No reference answer for this question;"
_NO_STATEMENTS = "No reference statements for this question;"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

records = load_records(read_only=True)

# Drafts included deliberately: all six answers are unverified, so verified-only
# loading would leave every record unscored. Their scores are stamped instead.
references = load_reference_answers(verified_only=False)

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
    if (reference.get("review") or {}).get("verified"):
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
            suite="reference",
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
        suite="reference",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@pytest.mark.reference
def test_citation_agreement(request, record):
    """
    The response must cite the legislation the reference answer cites.

    Pre-flight gate: output must be > 50 chars.
    """
    ok, reason = _gate_output_length(
        request, record, "citation_agreement", "Citation Agreement",
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

    test_case: LLMTestCase = record_to_test_case(record)
    metric = CitationAgreementMetric(
        reference_answer=reference["final_answer"],
        threshold=_COVERAGE_THRESHOLD,
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
        suite="reference",
    )

    assert metric.is_successful(), (
        f"Citation Agreement score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@pytest.mark.reference
@_skip_no_api_key
def test_reference_answer_agreement(request, record):
    """
    The response must make the reference answer's main points, and contradict
    none of them.

    Pre-flight gate: output must be > 50 chars.
    """
    ok, reason = _gate_output_length(
        request, record, "reference_answer_agreement", "Reference Answer Agreement",
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
        suite="reference",
    )

    assert metric.is_successful(), (
        f"Reference Answer Agreement score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )
