"""
Test that every captured response used all required legislation tools *and*
invoked them in the correct phase order.

Presence, 1/3 per required tool (delegate_research, Worker: search_legislation,
Worker: search_legislation_sections).

Order (legislation_only only), Worker tools must appear in the phase order
mandated by the Worker system prompt, and must not loop back to an earlier
phase once a later one has begun:

    1. Worker: search_legislation          (Phase 1, DISCOVER)
    2. Worker: search_legislation_sections (Phase 2, RETRIEVE PROVISIONS)
    3. Worker: get_legislation_text        (Phase 3, FALLBACK, optional)

Score:
    - 1.0  all three required tools present AND correct order
    - 0.5  all three required tools present BUT wrong order
    - <1.0 one or more required tools missing (proportional)

Passes only when score == 1.0.
"""

import pytest

from lex_eval.metrics.tool_usage import ToolUsageMetric
from lex_eval.utils.test_helpers import (
    load_records,
    record_to_test_case,
    record_id,
)
from lex_eval.utils.collector import attach_metric

records = load_records(read_only=True)


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
    All three required tools must be invoked, in the correct order:

        delegate_research, Worker: search_legislation,
        Worker: search_legislation_sections,
        Worker: get_legislation_text (optional fallback)

    Score = 1.0 only when all three required tools are present AND the Worker
    tools appear in the expected phase order. Wrong order caps the score at
    0.5 (fail). Missing tools score proportionally (fail).
    """
    test_case = record_to_test_case(record)
    research_mode = record.get("research_mode", "legislation_only")
    tool_sequence = record.get("tool_sequence") or []
    metric = ToolUsageMetric(
        threshold=1.0,
        research_mode=research_mode,
        tool_sequence=tool_sequence,
    )
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
