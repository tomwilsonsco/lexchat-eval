"""
Unit tests for ``lex_eval.metrics.consistency.ConsistencyMetric``'s section
citation check, synthetic responses, no DB or LexChat instance needed.

The check is reported in the reason and never decides pass or fail, so these
tests assert on ``metric.reason`` while ``is_successful()`` follows the score.
"""

import pytest
from lex_eval.testcase import LLMTestCase

from lex_eval.metrics.consistency import ConsistencyMetric, _preprocess

pytestmark = pytest.mark.unit


def _test_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual_output)


def test_reports_a_differing_section_without_failing_the_response():
    """Two answers can share almost all of their wording while citing a
    different section, the difference cosine similarity alone can't see. It is
    reported, but the score still decides the result."""
    actual = (
        "The employer must comply with the general duty. "
        "See [Health and Safety at Work Act 1974 - s.2]"
        "(http://www.legislation.gov.uk/ukpga/1974/37/section/2)."
    )
    reference = (
        "The employer must comply with the general duty. "
        "See [Health and Safety at Work Act 1974 - s.3]"
        "(http://www.legislation.gov.uk/ukpga/1974/37/section/3)."
    )
    metric = ConsistencyMetric(reference_outputs=[reference], threshold=0.4)
    metric.measure(_test_case(actual))

    assert "citations differ" in metric.reason
    assert metric.is_successful() is (metric.score >= metric.threshold)
    assert metric.is_successful()


def test_a_differing_section_does_not_rescue_a_low_score():
    """The reported difference is a note, so it must not change the verdict in
    either direction: an unrelated answer still fails on similarity alone."""
    actual = "See [s.2](http://www.legislation.gov.uk/ukpga/1974/37/section/2)."
    reference = (
        "Local authorities must publish an annual report on waste collection "
        "arrangements. See [s.45]"
        "(http://www.legislation.gov.uk/ukpga/1990/43/section/45)."
    )
    metric = ConsistencyMetric(reference_outputs=[reference], threshold=0.4)
    metric.measure(_test_case(actual))

    assert "citations differ" in metric.reason
    assert not metric.is_successful()


def test_passes_when_cited_sections_match():
    actual = (
        "See [s.2](http://www.legislation.gov.uk/ukpga/1974/37/section/2) "
        "for the general duty."
    )
    reference = (
        "The general duty is set out in "
        "[s.2](http://www.legislation.gov.uk/ukpga/1974/37/section/2)."
    )
    metric = ConsistencyMetric(reference_outputs=[reference], threshold=0.1)
    metric.measure(_test_case(actual))

    assert metric.is_successful()


def test_reports_a_differing_alphanumeric_section():
    """Section numbers are not always plain digits, e.g. inserted sections
    like s.10C, so the citation check must recognise those too."""
    actual = "See [s.10C](http://www.legislation.gov.uk/ukpga/1974/37/section/10C)."
    reference = "See [s.10](http://www.legislation.gov.uk/ukpga/1974/37/section/10)."
    metric = ConsistencyMetric(reference_outputs=[reference], threshold=0.1)
    metric.measure(_test_case(actual))

    assert "citations differ" in metric.reason


def test_preprocess_strips_link_target_but_keeps_link_text():
    """Citation URLs share a constant legislation.gov.uk prefix that would
    otherwise inflate cosine similarity with content-free boilerplate."""
    text = (
        "See [Equality Act 2010 - s.149]"
        "(http://www.legislation.gov.uk/ukpga/2010/15/section/149) for detail."
    )
    processed = _preprocess(text)

    assert "http" not in processed
    assert "legislation.gov.uk" not in processed
    assert "equality act 2010 - s.149" in processed


def test_citation_check_skipped_when_no_citations_present():
    """The Worker prompt allows bold-text citation as a fallback, so a
    response with no section URLs should be judged on cosine alone."""
    actual = "The general duty applies to employers under **s.2**."
    reference = "The general duty applies to employers under **s.2**."
    metric = ConsistencyMetric(reference_outputs=[reference], threshold=0.5)
    metric.measure(_test_case(actual))

    assert metric.is_successful()
    assert "citations differ" not in metric.reason
