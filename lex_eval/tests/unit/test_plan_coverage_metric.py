"""
Unit tests for ``lex_eval.metrics.plan_coverage.PlanCoverageMetric``, with a
stub judge model, no API key or DB needed.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics.plan_coverage import PlanCoverageMetric, _CoverageJudgement

pytestmark = pytest.mark.unit


class _StubJudge:
    """Returns canned _CoverageJudgement results in sequence, one per call."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt, schema=None):
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _points(*labels_and_steps):
    return _CoverageJudgement(
        points=[
            {"index": i + 1, "label": label, "step": step}
            for i, (label, step) in enumerate(labels_and_steps)
        ]
    )


_PLAN = [
    {"title": "Retrieve the text", "detail": "Get the section text"},
    {"title": "Check amendments", "detail": "Look for later amendments"},
]
_STATEMENTS = ["Statement one.", "Statement two.", "Statement three."]
_TEST_CASE = LLMTestCase(input="q", actual_output="irrelevant to this metric")


def test_full_coverage_scores_one():
    judge = _StubJudge(_points(("addressed", 1), ("addressed", 2), ("addressed", 1)))
    metric = PlanCoverageMetric(plan_steps=_PLAN, statements=_STATEMENTS, model=judge)
    metric.measure(_TEST_CASE)
    assert metric.score == 1.0
    assert metric.is_successful()


def test_partial_coverage_scores_fraction():
    judge = _StubJudge(
        _points(("addressed", 1), ("not_addressed", 0), ("not_addressed", 0))
    )
    metric = PlanCoverageMetric(plan_steps=_PLAN, statements=_STATEMENTS, model=judge)
    metric.measure(_TEST_CASE)
    assert metric.score == pytest.approx(1 / 3)
    assert not metric.is_successful()  # below default threshold 0.6


def test_invalid_step_number_downgraded_to_not_addressed():
    """The judge can't invent coverage it can't point at: an 'addressed'
    label naming a step outside the plan's range must not count, the same
    anti-hallucination shape as the quote check in ReferenceAnswerAgreement."""
    judge = _StubJudge(
        _points(("addressed", 1), ("addressed", 99), ("not_addressed", 0))
    )
    metric = PlanCoverageMetric(plan_steps=_PLAN, statements=_STATEMENTS, model=judge)
    metric.measure(_TEST_CASE)
    assert metric.score == pytest.approx(1 / 3)
    assert "cited step number doesn't exist" in metric.reason


def test_retries_once_on_wrong_label_count_then_succeeds():
    too_few = _CoverageJudgement(points=[{"index": 1, "label": "addressed", "step": 1}])
    correct = _points(("addressed", 1), ("addressed", 1), ("addressed", 1))
    judge = _StubJudge(too_few, correct)
    metric = PlanCoverageMetric(plan_steps=_PLAN, statements=_STATEMENTS, model=judge)
    metric.measure(_TEST_CASE)
    assert judge.calls == 2
    assert metric.score == 1.0


def test_gives_up_after_max_attempts():
    too_few = _CoverageJudgement(points=[{"index": 1, "label": "addressed", "step": 1}])
    judge = _StubJudge(too_few, too_few)
    metric = PlanCoverageMetric(plan_steps=_PLAN, statements=_STATEMENTS, model=judge)
    metric.measure(_TEST_CASE)
    assert judge.calls == 2
    assert not metric.is_successful()
    assert metric.reason.startswith("Judge error:")


def test_empty_statements_is_a_judge_error_not_a_crash():
    judge = _StubJudge()
    metric = PlanCoverageMetric(plan_steps=_PLAN, statements=[], model=judge)
    metric.measure(_TEST_CASE)
    assert not metric.is_successful()
    assert metric.reason.startswith("Judge error:")
    assert judge.calls == 0


def test_empty_plan_is_a_judge_error_not_a_crash():
    judge = _StubJudge()
    metric = PlanCoverageMetric(plan_steps=[], statements=_STATEMENTS, model=judge)
    metric.measure(_TEST_CASE)
    assert not metric.is_successful()
    assert metric.reason.startswith("Judge error:")
    assert judge.calls == 0


def test_no_contradiction_concept_only_threshold_decides_success():
    """Unlike ReferenceAnswerAgreementMetric, there's no veto label here:
    success is purely score >= threshold."""
    judge = _StubJudge(_points(("addressed", 1), ("addressed", 1), ("addressed", 1)))
    metric = PlanCoverageMetric(
        plan_steps=_PLAN, statements=_STATEMENTS, model=judge, threshold=1.0
    )
    metric.measure(_TEST_CASE)
    assert metric.score == 1.0
    assert metric.is_successful()
