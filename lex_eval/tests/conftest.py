"""
Shared pytest configuration for the LexChat evaluation suite.

Handles custom markers, sys.path setup, and metric data collection
for the JSON report consumed by the Streamlit dashboard.
"""

import json
import pytest
import sys
from pathlib import Path
from typing import Any


_REPORTS_DIR = Path(__file__).parent.parent / "reports"
_EVAL_JSON = _REPORTS_DIR / "eval_report.json"

# Accumulated during the session; written to JSON at the end.
_metric_records: list[dict] = []


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers and ensure imports resolve."""
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    config.addinivalue_line(
        "markers", "groundedness: tests that use an LLM judge (require API key)"
    )
    config.addinivalue_line(
        "markers", "consistency: same-model repeatability tests"
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test reports and collect metric data for the JSON report."""
    outcome = yield
    report = outcome.get_result()

    if call.when == "call" and hasattr(item, "_metric_data"):
        _metric_records.append(item._metric_data)


def pytest_sessionfinish(session, exitstatus):
    """Write all collected metric data to eval_report.json."""
    if not _metric_records:
        return
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_EVAL_JSON, "w") as f:
        json.dump(_metric_records, f, indent=2, default=str)
    print(f"\n📊 Eval results written to {_EVAL_JSON}")
