"""
Test that every captured response used all required legislation tools.

Scores 1/3 per tool (delegate_research, search_legislation, get_legislation_text).
Passes only when all three are present (score == 1.0).
"""

import pytest

from lex_eval.metrics.tool_usage import ToolUsageMetric
from lex_eval.utils.test_helpers import (
    load_test_cases,
    record_to_test_case,
    record_id,
)
from lex_eval.utils.collector import attach_metric

records = load_test_cases()


def _tools_list(test_case):
    """Extract tool names from a test case."""
    if test_case.tools_called:
        return [t.name for t in test_case.tools_called]
    return []


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_tool_usage(request, record):
    """
    All three required tools must be invoked:
        delegate_research, Worker: search_legislation, Worker: get_legislation_text.

    Score = number of tools present / 3 (i.e. 0.33 per tool).
    Passes only when all three are used (score == 1.0).
    """
    test_case = record_to_test_case(record)
    metric = ToolUsageMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="tool_usage",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
        tools_used=_tools_list(test_case),
        suite="tool_usage",
    )

    assert metric.is_successful(), metric.reason
