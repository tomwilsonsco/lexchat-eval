"""
Test that the Worker Agent response contains all mandatory Markdown
headings required by its system prompt for the given research mode.

For ``legislation_only`` the expected headings are:

    **Summary Answer (BLUF):**   (or **Summary Answer:**, the (BLUF)
                                  qualifier is optional)
    **Detailed Analysis:**
    **Jurisdiction & Status:**
    **References:**

The headings are checked inside the ``delegate_research`` tool-call output.
Records without a ``delegate_research`` call automatically receive a 0.0
failing score.
"""

import pytest

from lex_eval.metrics.structure import (
    CitationDomainMetric,
    CitationGroundingMetric,
    CitationPassthroughMetric,
    GenuineGapMetric,
    MandatoryStructureMetric,
)
from lex_eval.utils.collector import attach_metric
from lex_eval.utils.test_helpers import (
    load_records,
    record_id,
    record_to_test_case,
)

records = load_records(read_only=True)


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_mandatory_structure(request, record):
    """
    The Worker Agent output (returned via ``delegate_research``) must contain
    all mandatory Markdown headings based on the research mode.

    Records without a ``delegate_research`` tool call automatically score 0.0.
    """
    test_case = record_to_test_case(record)
    research_mode = record.get("research_mode", "legislation_only")
    metric = MandatoryStructureMetric(threshold=1.0, research_mode=research_mode)
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
    )

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_citation_passthrough(request, record):
    """
    Every legislation URL from the Worker output must appear in the final
    response delivered to the user.

    Failure A (0.0): no URLs at all in the Worker output.
    Failure B (0.5): Worker output had URLs but one or more didn't reach the
                     final response.
    Pass    (1.0): every Worker URL is present in the final response.
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
    )

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_citation_grounding(request, record):
    """
    Every Act cited in the Worker output must correspond to a legislation_id
    this run's own tool calls actually retrieved via search_legislation,
    search_legislation_sections, or get_legislation_text.

    Records with no delegate_research call automatically score 0.0.
    Records with no citation URLs at all score 1.0 (nothing to ground).
    """
    test_case = record_to_test_case(record)
    metric = CitationGroundingMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="citation_grounding",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
    )

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_citation_domain(request, record):
    """
    Every citation URL in the Worker output must point to legislation.gov.uk,
    the only domain the Worker's system prompt permits.

    Records with no delegate_research call automatically score 0.0.
    Records with no citation URLs at all score 1.0 (nothing to check).
    """
    test_case = record_to_test_case(record)
    metric = CitationDomainMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="citation_domain",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
    )

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_genuine_gap(request, record):
    """
    When retrieval returned no usable legislation section/full-text content,
    the Worker's report must disclose this rather than answering anyway.

    Records with no delegate_research call automatically score 0.0.
    Records outside legislation_only mode score 1.0 (not applicable).
    Records where retrieval succeeded score 1.0 (nothing to disclose).
    """
    test_case = record_to_test_case(record)
    research_mode = record.get("research_mode", "legislation_only")
    metric = GenuineGapMetric(threshold=1.0, research_mode=research_mode)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="genuine_gap",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
    )

    assert metric.is_successful(), metric.reason
