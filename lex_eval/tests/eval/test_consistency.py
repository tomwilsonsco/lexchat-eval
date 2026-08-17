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

_MIN_OUTPUT_CHARS: int = 50
_THRESHOLD: float = 0.5
_METRIC_NAME: str = "Consistency (Cosine)"

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
    grouped = group_by_question_and_llm(read_only=True)
    cases = []
    for key, records in sorted(grouped.items()):
        if len(records) < 2:
            continue
        for i, record in enumerate(records):
            others = [r["actual_output"] for j, r in enumerate(records) if j != i]
            test_id = f"{key}_run{i + 1}"
            cases.append(pytest.param(record, others, id=test_id))
    return cases


_same_model = _same_model_cases()


def _gate_output_length(request, record) -> tuple[bool, str]:
    """Fail fast if the output is too short to be meaningful.

    Two runs that both reply "Could you narrow this down?" are word-for-word
    identical and score 1.000, which reads as perfect consistency. Same gate
    and same wording as test_groundedness.py and test_reference.py, so the
    dashboard's _NON_SCORED_PREFIXES keeps these rows out of the mean.
    """
    char_count = len((record.get("actual_output") or "").strip())
    if char_count > _MIN_OUTPUT_CHARS:
        return True, ""

    reason = (
        f"Output too short ({char_count} chars ≤ {_MIN_OUTPUT_CHARS}); "
        f"{_METRIC_NAME} scored 0"
    )
    attach_metric(
        request,
        record=record,
        test_name="consistency",
        metric_name=_METRIC_NAME,
        score=0.0,
        threshold=_THRESHOLD,
        passed=False,
        reason=reason,
    )
    return False, reason


@pytest.mark.skipif(
    not _same_model,
    reason="No repeated runs found, re-run gather_responses.py with --append to generate repeatability data",
)
@pytest.mark.parametrize("record, other_outputs", _same_model)
def test_consistency(request, record, other_outputs):
    """
    The same LLM answering the same question repeatedly should produce
    highly consistent answers.

    A higher threshold (0.5) is used because the same model should
    be more self-consistent than different models would be.
    """
    proceed, gate_reason = _gate_output_length(request, record)
    if not proceed:
        pytest.skip(gate_reason)

    test_case = record_to_test_case(record)
    metric = ConsistencyMetric(
        reference_outputs=other_outputs,
        threshold=_THRESHOLD,
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
    )

    assert metric.is_successful(), metric.reason
