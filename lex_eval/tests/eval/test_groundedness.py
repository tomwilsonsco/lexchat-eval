"""
Test answer relevancy and groundedness of LexChat responses.

Three custom single-call metrics replace the previous multi-step
FaithfulnessMetric and AnswerRelevancyMetric:

  - LegalAnswerRelevancyMetric   : is the final response relevant to the question?
  - ResponseGroundednessMetric   : is the final response grounded in research output?
  - ResearchGroundednessMetric   : is the research output grounded in retrieval context?

All metrics use a single LLM call per test case. The judge (OpenAI or Gemini)
is configured via JUDGE_PROVIDER in lex_eval/.env.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics import (
    LegalAnswerRelevancyMetric,
    ResponseGroundednessMetric,
    ResearchGroundednessMetric,
)
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

_MIN_OUTPUT_CHARS: int = 50
_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

records = load_records()

_skip_no_api_key = pytest.mark.skipif(
    _judge is None,
    reason="Configured judge API key not set (check lex_eval/.env)",
)


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------


def _gate_output_length(request, record, test_case, test_name, metric_name):
    """Fail fast if the output is too short to be meaningful."""
    char_count = len((test_case.actual_output or "").strip())
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
            threshold=_THRESHOLD,
            passed=False,
            reason=reason,
            suite="groundedness",
        )
        return False, reason
    return True, ""


def _gate_retrieval_context(request, record, test_case, test_name, metric_name):
    """Fail fast if no retrieval context was captured."""
    if not test_case.retrieval_context:
        reason = f"No retrieval context captured; {metric_name} scored 0"
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=_THRESHOLD,
            passed=False,
            reason=reason,
            suite="groundedness",
        )
        return False, reason
    return True, ""


def _gate_research_output(request, record, test_name, metric_name):
    """Fail fast if no research output was captured."""
    if not record.get("research_output", "").strip():
        reason = f"No research output captured; {metric_name} scored 0"
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=_THRESHOLD,
            passed=False,
            reason=reason,
            suite="groundedness",
        )
        return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@pytest.mark.groundedness
@_skip_no_api_key
def test_answer_relevancy(request, record):
    """
    The final response must directly and usefully answer the user's legal question.

    Pre-flight gate: output must be > 50 chars.
    """
    test_case = record_to_test_case(record)

    ok, reason = _gate_output_length(
        request, record, test_case, "answer_relevancy", "Answer Relevancy"
    )
    if not ok:
        pytest.skip(reason)

    metric = LegalAnswerRelevancyMetric(model=_judge, threshold=_THRESHOLD)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="answer_relevancy",
        metric_name="Answer Relevancy",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if getattr(metric, "error", None) else "",
        suite="groundedness",
    )

    assert metric.is_successful(), (
        f"Answer Relevancy score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@pytest.mark.groundedness
@_skip_no_api_key
def test_response_groundedness(request, record):
    """
    The final response must be grounded in the research agent's output.

    Pre-flight gates:
      - Output must be > 50 chars.
      - research_output must be non-empty.
    """
    test_case = record_to_test_case(record)

    ok, reason = _gate_output_length(
        request, record, test_case, "response_groundedness", "Response Groundedness"
    )
    if not ok:
        pytest.skip(reason)

    ok, reason = _gate_research_output(
        request, record, "response_groundedness", "Response Groundedness"
    )
    if not ok:
        pytest.skip(reason)

    metric = ResponseGroundednessMetric(
        research_output=record["research_output"],
        model=_judge,
        threshold=_THRESHOLD,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="response_groundedness",
        metric_name="Response Groundedness",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if getattr(metric, "error", None) else "",
        suite="groundedness",
    )

    assert metric.is_successful(), (
        f"Response Groundedness score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@pytest.mark.groundedness
@_skip_no_api_key
def test_research_groundedness(request, record):
    """
    The research agent's output must be grounded in the raw retrieval context.

    Pre-flight gates:
      - retrieval_context must be non-empty.
      - research_output must be non-empty.
    """
    test_case = record_to_test_case(record)

    ok, reason = _gate_retrieval_context(
        request, record, test_case, "research_groundedness", "Research Groundedness"
    )
    if not ok:
        pytest.skip(reason)

    ok, reason = _gate_research_output(
        request, record, "research_groundedness", "Research Groundedness"
    )
    if not ok:
        pytest.skip(reason)

    metric = ResearchGroundednessMetric(
        research_output=record["research_output"],
        model=_judge,
        threshold=_THRESHOLD,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="research_groundedness",
        metric_name="Research Groundedness",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if getattr(metric, "error", None) else "",
        suite="groundedness",
    )

    assert metric.is_successful(), (
        f"Research Groundedness score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )
