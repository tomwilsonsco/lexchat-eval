"""
Unit tests for ``lex_eval.metrics.response_groundedness.ResponseGroundednessMetric``'s
near-verbatim short-circuit, synthetic responses, no DB, judge, or LexChat instance needed.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics.response_groundedness import (
    ResponseGroundednessMetric,
    _GroundednessJudgement,
)

pytestmark = pytest.mark.unit


def _test_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual_output)


class _JudgeNotInvoked:
    """A model stub that fails the test if the judge is ever called."""

    def generate(self, prompt, schema=None):
        raise AssertionError("Judge should not be invoked for a near-verbatim relay")


class _StubJudge:
    """A model stub that returns a fixed judgement."""

    def __init__(self, score: int, reason: str) -> None:
        self._score = score
        self._reason = reason

    def generate(self, prompt, schema=None):
        return _GroundednessJudgement(
            analysis="stub analysis", score=self._score, reason=self._reason
        )


def test_near_verbatim_relay_skips_judge_and_scores_full_marks():
    research_output = (
        "Section 6 of the Data Protection Act 2018 defines a controller as a "
        "person who determines the purposes and means of processing personal data."
    )
    # Trivial wording tweak only — well above the 0.95 similarity threshold.
    actual_output = (
        "Section 6 of the Data Protection Act 2018 defines a controller as the "
        "person who determines the purposes and means of processing personal data."
    )
    metric = ResponseGroundednessMetric(
        research_output=research_output, model=_JudgeNotInvoked(), threshold=0.7
    )
    metric.measure(_test_case(actual_output))

    assert metric.score == 1.0
    assert metric.is_successful()
    assert "Near-verbatim relay" in metric.reason
    assert "judge not invoked" in metric.reason


def test_divergent_response_still_uses_judge():
    research_output = (
        "The Act requires the controller to demonstrate compliance with the "
        "necessity requirement before relying on section 8(2), and that power "
        "is limited by section 9."
    )
    # Substantially condensed and reworded — well below the threshold.
    actual_output = "The Act lets controllers process data under section 8(2)."
    metric = ResponseGroundednessMetric(
        research_output=research_output,
        model=_StubJudge(score=2, reason="Omits the necessity requirement and the s.9 limitation."),
        threshold=0.7,
    )
    metric.measure(_test_case(actual_output))

    assert metric.score == pytest.approx((2 - 1) / 4)
    assert not metric.is_successful()
    assert "necessity requirement" in metric.reason
