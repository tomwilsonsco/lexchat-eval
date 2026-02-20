"""
Test groundedness of LLM responses against retrieved legislation.

Uses DeepEval's FaithfulnessMetric (LLM-as-judge) to check that claims
in the actual output are supported by the retrieval context rather than
hallucinated.

Requires an LLM judge model (e.g. gpt-4o) configured via OPENAI_API_KEY
or DEEPEVAL_LLM environment variable.
"""

import pytest
from deepeval.metrics import FaithfulnessMetric

from lex_eval.utils.test_helpers import (
    load_test_cases,
    record_to_test_case,
    record_id,
)
from lex_eval.utils.collector import attach_metric


records = load_test_cases()


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@pytest.mark.groundedness
def test_faithfulness(request, record):
    """
    The actual output must be faithful to the retrieved legislation.

    Uses DeepEval's FaithfulnessMetric which extracts claims from the
    output and verifies each is supported by the retrieval context.
    """
    test_case = record_to_test_case(record)

    if not test_case.retrieval_context:
        attach_metric(
            request,
            record=record,
            test_name="faithfulness",
            metric_name="Faithfulness",
            score=0.0,
            threshold=0.7,
            passed=False,
            reason="Skipped — no retrieval context to check against",
        )
        pytest.skip("No retrieval context — nothing to check faithfulness against")

    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="faithfulness",
        metric_name="Faithfulness",
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if metric.error else "",
    )

    assert metric.is_successful(), (
        f"Faithfulness score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@pytest.mark.groundedness
def test_output_not_empty(request, record):
    """The LLM must produce a substantive non-empty answer."""
    test_case = record_to_test_case(record)
    output = (test_case.actual_output or "").strip()
    char_count = len(output)
    passed = char_count > 50

    attach_metric(
        request,
        record=record,
        test_name="output_not_empty",
        metric_name="Output Length",
        score=1.0 if passed else 0.0,
        threshold=1.0,
        passed=passed,
        reason=f"Output is {char_count} chars",
    )

    assert passed, (
        f"Output is too short ({char_count} chars) for Q{record['question_id']} "
        f"with {record['llm_name']}. Expected a substantive legal answer."
    )
