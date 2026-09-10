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

from lex_eval.metrics import ReportIntegrationMetric
from lex_eval.metrics.structure import (
    CitationDomainMetric,
    CitationGroundingMetric,
    CitationPassthroughMetric,
    CitationReadMetric,
    GenuineGapMetric,
    MandatoryStructureMetric,
    StepCompletionMetric,
)
from lex_eval.utils.collector import attach_metric
from lex_eval.utils.applicability import exclusion
from lex_eval.utils.judge import _judge
from lex_eval.utils.test_helpers import (
    load_records,
    record_id,
    record_to_test_case,
)

records = load_records(read_only=True)

# Written at the front of the reason when a record isn't deep_research, so
# reports/streamlit_report.py keeps the row out of the mean (see
# _NON_SCORED_PREFIXES).
_NOT_DEEP_RESEARCH = "Not deep_research; Step Completion not measured"
_NOT_DEEP_RESEARCH_INTEGRATION = "Not deep_research; Report Integration not measured"

# Same idea for conversational runs, where the Worker system prompt tells the
# agent NOT to use the report headings this metric looks for, so a score here
# would measure obedience to an instruction LexChat never gave.
_NOT_CONVERSATIONAL_STRUCTURE = (
    "Not applicable in conversational mode; Research Output Structure not measured"
)

_skip_no_api_key = pytest.mark.skipif(
    _judge is None,
    reason="Configured judge API key not set (check lex_eval/.env)",
)


def _gate_scope(request, record, key):
    reason = exclusion(key, record)
    if reason:
        attach_metric(
            request,
            record=record,
            test_name=key,
            metric_name=key.replace("_", " ").title(),
            score=0.0,
            threshold=1.0,
            passed=False,
            measured=False,
            reason=reason,
        )
        pytest.skip(reason)


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

    Conversational records are not measured: their Worker prompt forbids these
    headings (excluded from the dashboard mean, not scored as a failure).
    """
    if record.get("chat_mode") == "conversational":
        attach_metric(
            request,
            record=record,
            test_name="mandatory_structure",
            metric_name="Research Output Structure",
            score=0.0,
            threshold=1.0,
            passed=False,
            reason=_NOT_CONVERSATIONAL_STRUCTURE,
        )
        pytest.skip(_NOT_CONVERSATIONAL_STRUCTURE)

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

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
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

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
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
    _gate_scope(request, record, "citation_grounding")
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

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_citation_read(request, record):
    """
    Every legislation.gov.uk Act cited in the Worker output must have had its
    text retrieved by search_legislation_sections or get_legislation_text.
    An Act that only appeared as a title in a search_legislation results list
    does not count as read.

    Records with no delegate_research call automatically score 0.0.
    Records with no legislation.gov.uk citations score 1.0 (nothing to check).
    """
    _gate_scope(request, record, "citation_read")
    test_case = record_to_test_case(record)
    metric = CitationReadMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="citation_read",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
    )

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_citation_domain(request, record):
    """
    Every citation URL in the Worker output must point to a domain the
    Worker's system prompt told it to cite: legislation.gov.uk for
    legislation_only, caselaw.nationalarchives.gov.uk for case_law_only,
    and both for legislation_and_case_law.

    Records with no delegate_research call automatically score 0.0.
    Records with no citation URLs at all score 1.0 (nothing to check).
    """
    test_case = record_to_test_case(record)
    metric = CitationDomainMetric(
        threshold=1.0, research_mode=record.get("research_mode", "legislation_only")
    )
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

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
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
    Conversational records accept a plain-English disclosure in full, since
    their Worker prompt mandates no exact wording.
    """
    _gate_scope(request, record, "genuine_gap")
    test_case = record_to_test_case(record)
    research_mode = record.get("research_mode", "legislation_only")
    metric = GenuineGapMetric(
        threshold=1.0,
        research_mode=research_mode,
        chat_mode=record.get("chat_mode", "research"),
    )
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

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
def test_step_completion(request, record):
    """
    Deep research only. Every step whose own tool calls retrieved usable
    legal text must carry a citation from it into that step's own report.

    Records that aren't deep_research are not measured (excluded from the
    dashboard mean, not scored as a failure).
    """
    if record.get("chat_mode") != "deep_research":
        attach_metric(
            request,
            record=record,
            test_name="step_completion",
            metric_name="Step Completion",
            score=0.0,
            threshold=1.0,
            passed=False,
            reason=_NOT_DEEP_RESEARCH,
        )
        pytest.skip(_NOT_DEEP_RESEARCH)

    _gate_scope(request, record, "step_completion")
    test_case = record_to_test_case(record)
    metric = StepCompletionMetric(threshold=1.0)
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="step_completion",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
    )

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize(
    "record",
    records,
    ids=[record_id(r) for r in records],
)
@_skip_no_api_key
def test_report_integration(request, record):
    """
    Deep research only. Every step whose own report carried a real, cited
    finding must have that finding reflected in the final answer, not
    dropped when the Manager condenses several steps into one response.

    Records that aren't deep_research are not measured (excluded from the
    dashboard mean, not scored as a failure).
    """
    if record.get("chat_mode") != "deep_research":
        attach_metric(
            request,
            record=record,
            test_name="report_integration",
            metric_name="Report Integration",
            score=0.0,
            threshold=1.0,
            passed=False,
            reason=_NOT_DEEP_RESEARCH_INTEGRATION,
        )
        pytest.skip(_NOT_DEEP_RESEARCH_INTEGRATION)

    _gate_scope(request, record, "report_integration")
    test_case = record_to_test_case(record)
    metric = ReportIntegrationMetric(model=_judge, threshold=1.0)
    _judge.last_model, _judge.total_usage_tokens = None, None
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="report_integration",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
        judge_llm=_judge.last_model,
        judge_tokens=_judge.total_usage_tokens,
    )

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), metric.reason
