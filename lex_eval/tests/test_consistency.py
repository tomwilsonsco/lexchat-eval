"""
Test consistency of responses when the same LLM answers the same question
multiple times (repeatability).

To generate test data, run gather_responses.py multiple times with --append:
    python lex_eval/gather_responses.py --append
"""

import pytest

from lex_eval.metrics.consistency import ConsistencyMetric
from lex_eval.utils.test_helpers import (
    load_records,
    record_to_test_case,
    group_by_question_and_llm,
)
from lex_eval.utils.collector import attach_metric

# ---------------------------------------------------------------------------
# Same-model repeatability: when the same question was asked to the same
# LLM multiple times (via --append), the answers should be very similar.
# ---------------------------------------------------------------------------


def _same_model_cases():
    """
    Yield (record, other_outputs, test_id) for same-model repeatability.

    Only produces cases when a (question, LLM) pair has more than one
    captured response.
    """
    grouped = group_by_question_and_llm()
    cases = []
    for key, records in sorted(grouped.items()):
        if len(records) < 2:
            continue
        for i, record in enumerate(records):
            others = [
                r["actual_output"] for j, r in enumerate(records) if j != i
            ]
            test_id = f"{key}_run{i + 1}"
            cases.append(pytest.param(record, others, id=test_id))
    return cases


_same_model = _same_model_cases()


@pytest.mark.consistency
@pytest.mark.skipif(
    not _same_model,
    reason="No repeated runs found — re-run gather_responses.py with --append to generate repeatability data",
)
@pytest.mark.parametrize("record, other_outputs", _same_model)
def test_consistency(request, record, other_outputs):
    """
    The same LLM answering the same question repeatedly should produce
    highly consistent answers.

    A higher threshold (0.5) is used because the same model should
    be more self-consistent than different models would be.
    """
    test_case = record_to_test_case(record)
    metric = ConsistencyMetric(
        reference_outputs=other_outputs,
        threshold=0.5,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="consistency",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
        suite="consistency",
    )

    assert metric.is_successful(), metric.reason
