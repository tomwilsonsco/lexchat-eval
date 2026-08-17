"""
Shared pytest configuration for the LexChat evaluation suite.

Handles custom markers, sys.path setup, and metric data collection.
Results are written to per-metric `eval_<test_name>` DuckDB tables
(data/responses.db) at the end of each pytest session.
"""

import pytest
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Accumulated during the session, keyed by metric (test_name).
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
        "markers",
        "unit: unit tests for the lex_eval harness itself (capture, DB, gather), fast, offline, no LLM",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test reports and collect metric data, keyed by metric (test_name)."""
    outcome = yield
    report = outcome.get_result()

    if call.when == "call" and hasattr(item, "_metric_data"):
        data = item._metric_data
        _metric_records[data["test_name"]].append(data)


def pytest_sessionfinish(session, exitstatus):
    """Write collected metric data to each metric's eval_<test_name> DuckDB table.

    Under pytest-xdist, this session is either a worker (one of several
    subprocesses collecting/running a slice of the tests) or the controller
    (the process that dispatches to workers and merges their results). DuckDB
    allows only one read-write connection to a file at a time, so only the
    controller may write here, if every worker wrote independently, their
    near-simultaneous connections would race and lock-error. Workers are
    identified by ``session.config.workerinput``, an attribute xdist sets
    only on worker processes; ship each worker's records to the controller
    via ``workeroutput`` instead, where ``pytest_testnodedown`` (below) picks
    them up.
    """
    if hasattr(session.config, "workerinput"):
        session.config.workeroutput["metric_records"] = dict(_metric_records)
        return

    if not _metric_records:
        return

    from lex_eval.utils.db import (
        DEFAULT_DB,
        get_connection,
        init_eval_table,
        insert_eval_result,
    )

    conn = get_connection(DEFAULT_DB)

    total = 0
    for metric, records in _metric_records.items():
        init_eval_table(conn, metric)
        for record in records:
            insert_eval_result(conn, metric, record)
            total += 1

    conn.commit()
    conn.close()
    print(f"\n📊 {total} eval result(s) written to {DEFAULT_DB}")


def pytest_testnodedown(node, error):
    """Merge a finished pytest-xdist worker's metric records into the controller's.

    Runs only on the controller, once per worker, before the controller's own
    ``pytest_sessionfinish``, so by the time that hook writes to DuckDB, every
    worker's results have already been folded into ``_metric_records``.
    """
    worker_records = (node.workeroutput or {}).get("metric_records", {})
    for metric, records in worker_records.items():
        _metric_records[metric].extend(records)
