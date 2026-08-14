"""
Unit tests for the two reference-answer metrics, synthetic answers only, no DB,
judge or LexChat instance needed.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics.citation_agreement import (
    NO_EXPECTED_CITATIONS_REASON,
    CitationAgreementMetric,
)
from lex_eval.metrics.reference_answer_agreement import (
    ReferenceAnswerAgreementMetric,
    _AgreementJudgement,
    _Point,
)

pytestmark = pytest.mark.unit

_REFERENCE = """
Section 6 of the Data Protection Act 2018 defines the controller.
See https://www.legislation.gov.uk/ukpga/2018/12/section/6 and
https://www.legislation.gov.uk/ukpga/2018/12/section/3.
"""


def _test_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual_output)


class _StubJudge:
    """A model stub returning fixed points, so no judge call is made."""

    def __init__(self, points: list[_Point]) -> None:
        self._points = points

    def generate(self, prompt, schema=None):
        return _AgreementJudgement(points=self._points)


class _FailingJudge:
    def generate(self, prompt, schema=None):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Citation Agreement
# ---------------------------------------------------------------------------


def test_citing_every_reference_provision_scores_full_marks():
    metric = CitationAgreementMetric(reference_answer=_REFERENCE)
    metric.measure(
        _test_case(
            "See http://www.legislation.gov.uk/id/ukpga/2018/12/section/6/ and "
            "https://www.legislation.gov.uk/ukpga/2018/12/section/3"
        )
    )

    assert metric.score == 1.0
    assert metric.is_successful()


def test_citing_a_different_section_of_the_same_act_is_a_miss():
    metric = CitationAgreementMetric(reference_answer=_REFERENCE)
    metric.measure(
        _test_case("See https://www.legislation.gov.uk/ukpga/2018/12/section/9")
    )

    assert metric.score == 0.0
    assert not metric.is_successful()
    assert "ukpga/2018/12/section/6" in metric.reason


def test_partial_coverage_is_scored_as_a_share():
    metric = CitationAgreementMetric(reference_answer=_REFERENCE, threshold=0.5)
    metric.measure(
        _test_case("See https://www.legislation.gov.uk/ukpga/2018/12/section/6")
    )

    assert metric.score == 0.5
    assert metric.is_successful()


def test_reference_citing_nothing_is_not_scored():
    metric = CitationAgreementMetric(reference_answer="No links here.")
    metric.measure(_test_case("https://www.legislation.gov.uk/ukpga/2018/12"))

    assert metric.reason == NO_EXPECTED_CITATIONS_REASON
    assert not metric.is_successful()


# ---------------------------------------------------------------------------
# Reference Answer Agreement
# ---------------------------------------------------------------------------


def _labels(*pairs: tuple[str, str]) -> list[_Point]:
    """One _Point per (label, quote), indexed from 1 as the judge must return them."""
    return [
        _Point(index=i + 1, label=label, quote=quote)
        for i, (label, quote) in enumerate(pairs)
    ]


_STATEMENTS = [
    "s.6 qualifies the UK GDPR definition of controller",
    "s.6(2) allocates controllership where an enactment requires the processing",
    "s.6 sits in Part 2, Chapter 2 of the Act",
]


def test_score_is_the_share_of_statements_stated():
    judge = _StubJudge(
        _labels(("stated", "words one"), ("stated", "words two"), ("missing", ""))
    )
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS, model=judge, threshold=0.6
    )
    metric.measure(_test_case("words one and words two"))

    assert metric.score == pytest.approx(2 / 3)
    assert metric.is_successful()


def test_a_contradiction_fails_the_metric_despite_a_high_score():
    judge = _StubJudge(
        _labels(
            ("stated", "words one"),
            ("stated", "words two"),
            ("stated", "words three"),
            ("contradicted", "is not the controller"),
        )
    )
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS + ["s.6 defines the controller"],
        model=judge,
        threshold=0.6,
    )
    metric.measure(
        _test_case("words one, words two, words three, but it is not the controller")
    )

    assert metric.score == 0.75
    assert not metric.is_successful()
    # The reason names the statement that was contradicted, looked up by index.
    assert "s.6 defines the controller" in metric.reason


def test_a_contradiction_quoting_words_not_in_the_response_is_counted_as_missing():
    judge = _StubJudge(
        _labels(
            ("stated", "words one"), ("contradicted", "words the response never used")
        )
    )
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS[:2], model=judge, threshold=0.5
    )
    metric.measure(_test_case("words one"))

    assert metric.score == 0.5
    assert metric.is_successful()
    assert "not in the response" in metric.reason


def test_judge_failure_is_flagged_not_scored_as_a_bad_answer():
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS, model=_FailingJudge()
    )
    metric.measure(_test_case("anything"))

    assert metric.reason.startswith("Judge error:")
    assert not metric.is_successful()


def test_a_wrong_label_count_is_retried_once():
    """The judge returns a short or padded list on roughly 1 call in 22."""

    class _FlakyJudge:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, schema=None):
            self.calls += 1
            if self.calls == 1:
                return _AgreementJudgement(points=_labels(("stated", "words one")))
            return _AgreementJudgement(
                points=_labels(
                    ("stated", "words one"), ("stated", "words two"), ("missing", "")
                )
            )

    judge = _FlakyJudge()
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS, model=judge, threshold=0.6
    )
    metric.measure(_test_case("words one and words two"))

    assert judge.calls == 2
    assert metric.score == pytest.approx(2 / 3)
    assert metric.is_successful()


def test_a_short_label_list_is_a_judge_error_not_a_shrunken_denominator():
    """The whole point of a frozen list is that the denominator cannot move."""
    judge = _StubJudge(_labels(("stated", "words one"), ("stated", "words two")))
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS, model=judge, threshold=0.6
    )
    metric.measure(_test_case("words one and words two"))

    assert metric.score == 0.0
    assert not metric.is_successful()
    assert metric.reason.startswith("Judge error:")
    assert "2 label(s) for 3 statements" in metric.reason
