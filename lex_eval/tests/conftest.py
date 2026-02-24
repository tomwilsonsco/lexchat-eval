"""
Shared pytest configuration for the LexChat evaluation suite.

Handles custom markers, sys.path setup, and metric data collection
for per-suite JSON report files consumed by the Streamlit dashboard.

Each test suite writes to its own results file:
  - groundedness_results.json
  - consistency_results.json
  - consistency_llm_results.json
  - tool_usage_results.json
  - structure_results.json
"""

import json
import pytest
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPORTS_DIR = Path(__file__).parent.parent / "reports"

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
        "markers", "groundedness: tests that use an LLM judge (require OPENAI_API_KEY)"
    )
    config.addinivalue_line(
        "markers", "consistency: same-model repeatability tests (Jaccard)"
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
    """Write collected metric data to per-suite JSON files."""
    if not _metric_records:
        return
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for suite, records in _metric_records.items():
        out_path = _REPORTS_DIR / f"{suite}_results.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\n📊 {suite} results written to {out_path}")
