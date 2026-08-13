"""
Unit tests for ``lex_eval.metrics.consistency_llm.LLMConsistencyMetric``'s
handling of judge-call failures, no DB or OpenRouter key needed.

A per-pair judge failure must not be averaged in as a real 0.0: it should
either be excluded from the mean (partial failure) or, if every pairwise
call fails, reported with a reason starting "Judge error:" so
reports/streamlit_report.py's `_NON_SCORED_PREFIXES` check excludes it from
the dashboard's mean/pass-rate instead of scoring it as a bad result.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics.consistency_llm import LLMConsistencyMetric, _ConsistencyJudgement

pytestmark = pytest.mark.unit


def _test_case() -> LLMTestCase:
    return LLMTestCase(input="What does section 6 say?", actual_output="It says X.")


class _StubJudge:
    """Returns queued results/exceptions from `.generate()` in call order."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def generate(self, prompt, schema=None):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_all_judge_calls_failing_is_reported_as_judge_error():
    judge = _StubJudge([RuntimeError("empty content"), RuntimeError("empty content")])
    metric = LLMConsistencyMetric(
        reference_outputs=["ref one", "ref two"], model=judge, threshold=0.7
    )
    metric.measure(_test_case())

    assert metric.reason.startswith("Judge error:")
    assert not metric.is_successful()


def test_partial_judge_failure_excluded_from_mean():
    judge = _StubJudge(
        [
            RuntimeError("empty content"),
            _ConsistencyJudgement(score=1.0, reason="consistent"),
        ]
    )
    metric = LLMConsistencyMetric(
        reference_outputs=["ref one", "ref two"], model=judge, threshold=0.7
    )
    metric.measure(_test_case())

    # Only the successful pair should count — not averaged with the failure's 0.0.
    assert metric.score == 1.0
    assert metric.is_successful()
    assert not metric.reason.startswith("Judge error:")
    assert "excluded due to judge error" in metric.reason


def test_all_judge_calls_succeeding_averages_as_before():
    judge = _StubJudge(
        [
            _ConsistencyJudgement(score=1.0, reason="consistent"),
            _ConsistencyJudgement(score=0.4, reason="scope drift"),
        ]
    )
    metric = LLMConsistencyMetric(
        reference_outputs=["ref one", "ref two"], model=judge, threshold=0.7
    )
    metric.measure(_test_case())

    assert metric.score == pytest.approx(0.7)
    assert "excluded due to judge error" not in metric.reason
