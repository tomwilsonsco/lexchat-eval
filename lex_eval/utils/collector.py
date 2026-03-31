"""
Helper for attaching metric data to pytest test items.

Tests call ``attach_metric(request, ...)`` to store score/reason data
on the test node. conftest.pytest_runtest_makereport then collects it
into per-suite lists which are written to the eval_results table in the
shared DuckDB database (data/responses.db) at the end of the run.
"""

from typing import Any, Dict, List


def attach_metric(
    request,
    *,
    record: Dict[str, Any],
    test_name: str,
    metric_name: str,
    score: float,
    threshold: float,
    passed: bool,
    suite: str,
    reason: str = "",
    error: str = "",
    tools_used: List[str] | None = None,
) -> None:
    """
    Attach metric result data to the pytest test item.

    The data is picked up by the ``pytest_runtest_makereport`` hook in
    conftest.py and accumulated for writing to per-suite JSON files.

    Parameters
    ----------
    suite : str
        Grouping key used to bucket results, e.g. ``"groundedness"``,
        ``"consistency"``, ``"tool_usage"``.  Results are stored in the
        ``eval_results`` DuckDB table under this suite name.
    """
    request.node._metric_data = {
        "suite": suite,
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
    }
