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


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_retrieval_context_populated(request, record):
    """Tools must have actually retrieved legislation context."""
    test_case = record_to_test_case(record)
    has_context = bool(test_case.retrieval_context)
    reason = (
        f"{len(test_case.retrieval_context)} context item(s) retrieved"
        if has_context
        else "No retrieval context captured"
    )

    attach_metric(
        request,
        record=record,
        test_name="retrieval_context_populated",
        metric_name="Retrieval Context",
        score=1.0 if has_context else 0.0,
        threshold=1.0,
        passed=has_context,
        reason=reason,
        tools_used=_tools_list(test_case),
        suite="tool_usage",
    )

    assert has_context, (
        f"No retrieval context captured for Q{record['question_id']} "
        f"with {record['llm_name']}. Tools may have been called but "
        f"returned no useful content."
    )
