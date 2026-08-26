"""
Helper for attaching metric data to pytest test items.

Tests call ``attach_metric(request, ...)`` to store score/reason data
on the test node. conftest.pytest_runtest_makereport then collects it
and writes results to that metric's own ``eval_<test_name>`` table in the
shared DuckDB database (data/responses.db) at the end of the run.
"""

from typing import Any, Dict, List, Optional

from .db import reason_is_not_measured


def attach_metric(
    request,
    *,
    record: Dict[str, Any],
    test_name: str,
    metric_name: str,
    score: float,
    threshold: float,
    passed: bool,
    reason: str = "",
    error: str = "",
    tools_used: List[str] | None = None,
    judge_llm: Optional[str] = None,
    judge_tokens: Optional[int] = None,
    measured: Optional[bool] = None,
) -> None:
    """
    Attach metric result data to the pytest test item.

    The data is picked up by the ``pytest_runtest_makereport`` hook in
    conftest.py and accumulated for writing to the ``eval_<test_name>``
    DuckDB table, one table per metric.

    Parameters
    ----------
    test_name : str
        The metric's key, e.g. ``"response_groundedness"``. Determines which
        ``eval_<test_name>`` table the result is written to.
    judge_llm, judge_tokens
        Only set by AI-judge metrics, e.g. from ``_judge.last_model`` /
        ``_judge.total_usage_tokens`` after calling ``.generate()``. Left
        ``None`` for deterministic metrics.
    measured
        Whether ``score`` is a real verdict about the response. Left ``None``,
        it is derived from *reason*: a metric that could not score at all, for
        example a deep-research-only metric handed a conversational one, says so
        in its reason and the row is marked unmeasured. Deriving it here rather
        than at each call site is deliberate, so that a metric cannot forget and
        silently put a placeholder 0.0 into a mean. Pass it explicitly only to
        override that.
    """
    request.node._metric_data = {
        "response_id": record.get("response_id"),
        "llm_name": record["llm_name"],
        "question_id": record["question_id"],
        "question": record["question"],
        "test_name": test_name,
        "metric_name": metric_name,
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "reason": reason,
        "error": error,
        "tools_used": tools_used,
        "judge_llm": judge_llm,
        "judge_tokens": judge_tokens,
        "measured": (
            (not reason_is_not_measured(reason)) if measured is None else measured
        ),
    }
