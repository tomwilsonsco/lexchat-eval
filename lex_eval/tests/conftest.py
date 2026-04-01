"""
Shared pytest configuration for the LexChat evaluation suite.

Handles custom markers, sys.path setup, and metric data collection.
Results are written to the `eval_results` DuckDB table
(data/responses.db) at the end of each pytest session.
"""

import pytest
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Accumulated during the session, keyed by suite name.
_metric_records: dict[str, list[dict]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers and ensure imports resolve."""
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    config.addinivalue_line(
        "markers", "tool_usage: tests that check correct tool invocation (fast, offline)"
    )
    config.addinivalue_line(
        "markers", "groundedness: tests that use an LLM judge (require OPENAI_API_KEY)"
    )
    config.addinivalue_line(
        "markers", "consistency: same-model repeatability tests (cosine similarity)"
    )
    config.addinivalue_line(
        "markers",
        "consistency_llm: same-model repeatability tests (AI judge, requires OPENAI_API_KEY)",
    )
    config.addinivalue_line(
        "markers", "structure: tests that check mandatory Worker output structure"
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test reports and collect metric data for per-suite JSON files."""
    outcome = yield
    report = outcome.get_result()

    if call.when == "call" and hasattr(item, "_metric_data"):
        data = item._metric_data
        suite = data.pop("suite", "unknown")
        _metric_records[suite].append(data)


def pytest_sessionfinish(session, exitstatus):
    """Write collected metric data to the eval_results DuckDB table."""
    if not _metric_records:
        return

    from lex_eval.utils.db import (
        DEFAULT_DB,
        get_connection,
        init_eval_results,
        insert_eval_result,
    )

    conn = get_connection(DEFAULT_DB)
    init_eval_results(conn)

    total = 0
    for suite, records in _metric_records.items():
        for record in records:
            insert_eval_result(conn, record, suite=suite)
            total += 1

    conn.commit()
    conn.close()
    print(f"\n📊 {total} eval result(s) written to {DEFAULT_DB}")
