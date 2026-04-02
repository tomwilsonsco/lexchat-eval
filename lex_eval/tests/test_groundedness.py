"""
Test groundedness and answer relevancy of LLM responses against retrieved
legislation.

Uses DeepEval's FaithfulnessMetric and AnswerRelevancyMetric (LLM-as-judge)
with a configurable AI judge (OpenAI gpt-4o-mini by default; set
JUDGE_PROVIDER=gemini to use Gemini instead).  The API key and model are
read from the .env file in lex_eval/.

Output length and retrieval context are used as pre-flight gates rather than
standalone dashboard metrics.  If either gate fails the Faithfulness metric is
scored 0.0 with the failure reason recorded, and the test is skipped so the
more expensive LLM-judge call is avoided.
"""

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

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

# Conservative character budget for retrieval context passed to the judge.
# 128k token limit; reserve ~30k tokens for the prompt, output, and overhead.
# Rough approximation: 1 token ≈ 4 chars.
_MAX_CONTEXT_CHARS: int = (128_000 - 30_000) * 4  # ≈ 392 000 chars


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


def _gate_output_length(request, record, test_case, test_name, metric_name, threshold):
    """
    Fail fast if the output is too short to be meaningful.

    Attaches a 0.0 score and returns False so the caller can skip the test.
    """
    output = (test_case.actual_output or "").strip()
    char_count = len(output)
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
            suite="groundedness",
        )
        return False, reason
    return True, ""


def _truncate_retrieval_context(test_case: "LLMTestCase") -> "LLMTestCase":
    """
    Return a copy of *test_case* whose retrieval_context is trimmed so the
    total character count stays within _MAX_CONTEXT_CHARS.

    Items are included whole in order; the first item that would push the
    total over the budget is dropped along with all subsequent items.
    """
    context = test_case.retrieval_context or []
    kept: list[str] = []
    total = 0
    for item in context:
        if total + len(item) > _MAX_CONTEXT_CHARS:
            break
        kept.append(item)
        total += len(item)
    if len(kept) == len(context):
        return test_case  # nothing to trim
    return LLMTestCase(
        input=test_case.input,
        actual_output=test_case.actual_output,
        retrieval_context=kept,
        tools_called=test_case.tools_called,
    )


def _gate_retrieval_context(
    request, record, test_case, test_name, metric_name, threshold
):
    """
    Fail fast if no retrieval context was captured.

    Attaches a 0.0 score and returns False so the caller can skip the test.
    """
    if not test_case.retrieval_context:
        reason = f"No retrieval context captured; {metric_name} scored 0"
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=threshold,
            passed=False,
            reason=reason,
            suite="groundedness",
        )
        return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@pytest.mark.groundedness
@_skip_no_api_key
def test_faithfulness(request, record):
    """
    The actual output must be faithful to the retrieved legislation.

    Uses DeepEval's FaithfulnessMetric which extracts claims from the output
    and verifies each is supported by the retrieval context.

    Pre-flight gates (do NOT appear as separate dashboard metrics):
      - Output must be > 50 chars (otherwise nothing to judge)
      - Retrieval context must be non-empty (otherwise nothing to check against)
    """
    test_case = record_to_test_case(record)
    _threshold = 0.7

    ok, reason = _gate_output_length(
        request,
        record,
        test_case,
        "faithfulness",
        "Groundedness (AI Judge)",
        _threshold,
    )
    if not ok:
        pytest.skip(reason)

    ok, reason = _gate_retrieval_context(
        request,
        record,
        test_case,
        "faithfulness",
        "Groundedness (AI Judge)",
        _threshold,
    )
    if not ok:
        pytest.skip(reason)

    test_case = _truncate_retrieval_context(test_case)

    metric = FaithfulnessMetric(
        threshold=_threshold,
        model=_judge,
        include_reason=True,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="faithfulness",
        metric_name="Groundedness (AI Judge)",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if metric.error else "",
        suite="groundedness",
    )

    assert metric.is_successful(), (
        f"Groundedness (AI Judge) score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@pytest.mark.groundedness
@_skip_no_api_key
def test_answer_relevancy(request, record):
    """
    The actual output must be relevant to the input question.

    Uses DeepEval's AnswerRelevancyMetric which scores how directly and
    completely the response addresses what was asked.

    Pre-flight gate: output must be > 50 chars.
    """
    test_case = record_to_test_case(record)
    _threshold = 0.7

    ok, reason = _gate_output_length(
        request,
        record,
        test_case,
        "answer_relevancy",
        "Answer Relevancy (AI Judge)",
        _threshold,
    )
    if not ok:
        pytest.skip(reason)

    metric = AnswerRelevancyMetric(
        threshold=_threshold,
        model=_judge,
        include_reason=True,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="answer_relevancy",
        metric_name="Answer Relevancy (AI Judge)",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if metric.error else "",
        suite="groundedness",
    )

    assert metric.is_successful(), (
        f"Answer Relevancy (AI Judge) score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )
