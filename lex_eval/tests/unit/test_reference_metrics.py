"""
Unit tests for the two reference-answer metrics, synthetic answers only, no DB,
judge or LexChat instance needed.
"""

import json

import pytest
from lex_eval.testcase import LLMTestCase, ToolCall

from lex_eval.metrics.citation_agreement import (
    NO_EXPECTED_CITATIONS_REASON,
    CitationAgreementMetric,
    attribute_missing_acts,
    reference_acts,
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


def _test_case(actual_output: str, tools_called: list | None = None) -> LLMTestCase:
    return LLMTestCase(
        input="q", actual_output=actual_output, tools_called=tools_called or []
    )


_RELIED_ON = {"ukpga/2018/12"}


def _search(*legislation_ids: str) -> ToolCall:
    """A ``search_legislation`` call whose results list *legislation_ids*."""
    return ToolCall(
        name="Worker: search_legislation",
        input_parameters={"query": "q"},
        output=json.dumps(
            {"results": [{"legislation_id": i, "title": i} for i in legislation_ids]}
        ),
    )


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


def test_an_act_no_search_turned_up_is_attributed_to_the_search():
    metric = CitationAgreementMetric(
        reference_answer=_REFERENCE, expected_acts=_RELIED_ON
    )
    metric.measure(_test_case("No citations at all.", [_search("ukpga/1998/46")]))

    assert metric.never_retrieved == ["ukpga/2018/12"]
    assert metric.retrieved_not_cited == []
    assert "No search turned up: ukpga/2018/12." in metric.reason


def test_an_act_a_search_turned_up_is_attributed_to_the_model():
    metric = CitationAgreementMetric(
        reference_answer=_REFERENCE, expected_acts=_RELIED_ON
    )
    metric.measure(_test_case("No citations at all.", [_search("ukpga/2018/12")]))

    assert metric.never_retrieved == []
    assert metric.retrieved_not_cited == ["ukpga/2018/12"]
    assert "Turned up by a search but not cited: ukpga/2018/12." in metric.reason


def test_an_act_the_reference_only_looked_at_is_never_attributed():
    """A source under "Identified but not retrieved" is the author's reading
    list, not something the response has to cite."""
    metric = CitationAgreementMetric(
        reference_answer=_REFERENCE, expected_acts=_RELIED_ON
    )
    metric.measure(
        _test_case(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6",
            [_search("ukpga/2018/12", "ukpga/1998/46")],
        )
    )

    assert metric.score < 1.0, "the prose-linked Act still counts against the score"
    assert metric.never_retrieved == []
    assert metric.retrieved_not_cited == []
    assert "turned up" not in metric.reason.lower()


def test_an_act_cited_somewhere_is_not_attributed_at_all():
    """Citing the wrong section of an Act is a miss, but not a retrieval one."""
    metric = CitationAgreementMetric(
        reference_answer=_REFERENCE, expected_acts=_RELIED_ON
    )
    metric.measure(
        _test_case("See https://www.legislation.gov.uk/ukpga/2018/12/section/9")
    )

    assert metric.score == 0.0
    assert metric.never_retrieved == []
    assert metric.retrieved_not_cited == []
    assert "turned up" not in metric.reason.lower()


def test_without_the_relied_on_acts_nothing_is_attributed():
    metric = CitationAgreementMetric(reference_answer=_REFERENCE)
    metric.measure(_test_case("No citations at all."))

    assert metric.never_retrieved == []
    assert metric.retrieved_not_cited == []
    assert "turned up" not in metric.reason.lower()


def test_a_run_with_no_tool_calls_is_attributed_to_the_search():
    metric = CitationAgreementMetric(
        reference_answer=_REFERENCE, expected_acts=_RELIED_ON
    )
    metric.measure(_test_case("No citations at all."))

    assert metric.never_retrieved == ["ukpga/2018/12"]


def test_attribution_does_not_move_the_score():
    without = CitationAgreementMetric(reference_answer=_REFERENCE)
    without.measure(
        _test_case("https://www.legislation.gov.uk/ukpga/2018/12/section/6")
    )
    with_tools = CitationAgreementMetric(
        reference_answer=_REFERENCE, expected_acts=_RELIED_ON
    )
    with_tools.measure(
        _test_case(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6",
            [_search("ukpga/2018/12")],
        )
    )

    assert without.score == with_tools.score


def test_the_relied_on_acts_are_retrieved_and_cited_both():
    """Retrieved but never cited is law the author ruled out. Cited but never
    retrieved is something they only looked at. Neither counts."""
    reference = {
        "final_answer": (
            "See https://www.legislation.gov.uk/ukpga/2018/12/section/6 and "
            "https://www.legislation.gov.uk/ssi/2015/99."
        ),
        "sources_retrieved": [
            {"legislation_id": "ukpga/2018/12", "uri": "u"},  # retrieved and cited
            {"legislation_id": "ukpga/1985/67", "uri": "u"},  # retrieved, ruled out
            {"uri": "no legislation_id"},
        ],
        "sources_discovered": [{"legislation_id": "ssi/2015/99", "uri": "u"}],
    }

    assert reference_acts(reference) == {"ukpga/2018/12"}


def test_sections_read_count_as_retrieving_the_act():
    """``search_legislation_sections`` names the Act in its arguments."""
    sections = ToolCall(
        name="Worker: search_legislation_sections",
        input_parameters={"legislation_id": "ukpga/2018/12", "query": "controller"},
        output="{}",
    )
    never, retrieved = attribute_missing_acts({"ukpga/2018/12"}, set(), [sections])

    assert never == []
    assert retrieved == ["ukpga/2018/12"]


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
