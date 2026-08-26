"""
Unit tests for the two reference-answer metrics, synthetic answers only, no DB,
judge or LexChat instance needed.
"""

import pytest
from lex_eval.testcase import LLMTestCase

from lex_eval.metrics.citation_agreement import (
    NO_EXPECTED_CITATIONS_REASON,
    CitationAgreementMetric,
)
from lex_eval.metrics.reference_answer_agreement import (
    ReferenceAnswerAgreementMetric,
    _AgreementJudgement,
    _Contradiction,
    _ContradictionJudgement,
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
    """A model stub returning fixed answers, so no judge call is made.

    The metric makes two calls, one labelling the statements and one looking
    only for contradictions, so the stub answers by schema.
    """

    def __init__(
        self, points: list[_Point], contradictions: list[_Contradiction] | None = None
    ) -> None:
        self._points = points
        self._contradictions = contradictions or []

    def generate(self, prompt, schema=None):
        if schema is _ContradictionJudgement:
            return _ContradictionJudgement(findings=self._contradictions)
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


def test_a_contradiction_quoting_words_not_in_the_response_is_ignored():
    """An invented quote cannot fail a record."""
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


def test_a_contradiction_found_only_by_the_second_call_fails_the_metric():
    """The failure this metric's contradiction sweep exists to catch.

    A long answer states every reference point and then, well away from where
    it made them, asserts something that undoes one. The labelling call sees
    the point stated and says so; the contradiction call is what catches it.
    """
    judge = _StubJudge(
        _labels(("stated", "words one"), ("stated", "words two"), ("stated", "three")),
        contradictions=[
            _Contradiction(index=2, contradicted=True, quote="but none of that applies")
        ],
    )
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS, model=judge, threshold=0.6
    )
    metric.measure(_test_case("words one, words two, three, but none of that applies"))

    assert metric.score == 1.0
    assert not metric.is_successful()
    assert _STATEMENTS[1] in metric.reason


def test_an_invented_quote_from_the_second_call_is_ignored():
    judge = _StubJudge(
        _labels(("stated", "words one"), ("stated", "words two"), ("stated", "three")),
        contradictions=[
            _Contradiction(index=2, contradicted=True, quote="never said this at all")
        ],
    )
    metric = ReferenceAnswerAgreementMetric(
        statements=_STATEMENTS, model=judge, threshold=0.6
    )
    metric.measure(_test_case("words one, words two, three"))

    assert metric.score == 1.0
    assert metric.is_successful()


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
            self.calls = 0  # labelling calls only, not the contradiction sweep

        def generate(self, prompt, schema=None):
            if schema is _ContradictionJudgement:
                return _ContradictionJudgement(findings=[])
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
