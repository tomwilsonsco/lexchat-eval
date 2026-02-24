"""
Test that the Worker Agent response contains all four mandatory Markdown
headings required by its system prompt:

    **Summary Answer (BLUF):**
    **Detailed Analysis:**
    **Jurisdiction & Status:**
    **References:**

The headings are checked inside the ``delegate_research`` tool-call output.
Records without a ``delegate_research`` call automatically receive a 0.0
failing score.
"""

import pytest

from lex_eval.metrics.structure import (
    CitationPassthroughMetric,
    MandatoryStructureMetric,
)
from lex_eval.utils.collector import attach_metric
from lex_eval.utils.test_helpers import (
    load_test_cases,
    record_id,
    record_to_test_case,
)

records = load_test_cases()


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@pytest.mark.structure
def test_mandatory_structure(request, record):
    """
    The Worker Agent output (returned via ``delegate_research``) must contain
    all four mandatory Markdown headings.

    Records without a ``delegate_research`` tool call automatically score 0.0.
    """
    test_case = record_to_test_case(record)
    metric = MandatoryStructureMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="mandatory_structure",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
        suite="structure",
    )

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@pytest.mark.structure
def test_citation_passthrough(request, record):
    """
    At least one legislation URL from the Worker output must appear in the
    final response delivered to the user.

    Failure A (0.0): no URLs at all in the Worker output.
    Failure B (0.5): Worker output had URLs but none reached the final response.
    Pass    (1.0): at least one Worker URL is present in the final response.
    """
    test_case = record_to_test_case(record)
    metric = CitationPassthroughMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="citation_passthrough",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
        suite="structure",
    )

    assert metric.is_successful(), metric.reason
