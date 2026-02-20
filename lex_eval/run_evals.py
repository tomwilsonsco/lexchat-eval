#!/usr/bin/env python3
"""
Run LexChat evaluations using pytest + DeepEval.

This script wraps pytest so the full evaluation suite can be launched
from the command line with sensible defaults and optional filters.

Examples
--------
Run everything (tool usage + consistency — groundedness needs an LLM key):
    python lex_eval/run_evals.py

Run only tool-usage checks (fast, no LLM judge needed):
    python lex_eval/run_evals.py --suite tool_usage

Run only groundedness (requires OPENAI_API_KEY):
    python lex_eval/run_evals.py --suite groundedness

Run only consistency:
    python lex_eval/run_evals.py --suite consistency

Exclude slow LLM-judge tests:
    python lex_eval/run_evals.py -m "not groundedness"

Verbose output:
    python lex_eval/run_evals.py -v

Launch Streamlit dashboard after tests:
    python lex_eval/run_evals.py --streamlit

Launch dashboard against existing results (no re-run):
    streamlit run lex_eval/reports/streamlit_report.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"

SUITES = {
    "tool_usage": "test_tool_usage.py",
    "groundedness": "test_groundedness.py",
    "consistency": "test_consistency.py",
}


def run_evals(
    suite: str | None = None,
    markers: str | None = None,
    verbose: bool = False,
    launch_streamlit: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """
    Launch pytest against the evaluation test suite.

    Returns the pytest exit code (0 = all passed).
    """
    cmd: list[str] = [sys.executable, "-m", "pytest"]

    # Test target
    if suite and suite in SUITES:
        cmd.append(str(TESTS_DIR / SUITES[suite]))
    else:
        cmd.append(str(TESTS_DIR))

    # Marker filter
    if markers:
        cmd.extend(["-m", markers])

    # Display
    cmd.extend(["-v" if verbose else "-q", "--tb=short"])

    # Pass-through args
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    reports_dir = Path(__file__).parent / "reports"

    if result.returncode in (0, 1):  # 0=all pass, 1=some fail
        if launch_streamlit:
            _launch_streamlit(reports_dir)
        else:
            print(
                f"\n📊 Results written to {reports_dir / 'eval_report.json'}"
                "\n   View dashboard: streamlit run lex_eval/reports/streamlit_report.py"
            )

    return result.returncode


def _launch_streamlit(reports_dir: Path) -> None:
    """Launch the Streamlit dashboard, blocking until the user exits."""
    app_path = reports_dir / "streamlit_report.py"
    print(f"\n🚀 Launching Streamlit dashboard: {app_path}")
    print("   Press Ctrl+C to stop.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LexChat evaluations (pytest + DeepEval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suites:
  tool_usage     Check tools were invoked correctly (fast, offline)
  groundedness   LLM-as-judge faithfulness check (needs OPENAI_API_KEY)
  consistency    Same-model repeatability checks (fast, offline)

Dashboard:
  Results are written to lex_eval/reports/eval_report.json after each run.
  Launch the Streamlit dashboard at any time:
    streamlit run lex_eval/reports/streamlit_report.py
  Or auto-launch after tests:
    python lex_eval/run_evals.py --streamlit

Markers:
  groundedness   Tests that call an LLM judge
  consistency    Same-model repeatability tests
""",
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()),
        help="Run a specific test suite instead of all",
    )
    parser.add_argument(
        "-m",
        "--markers",
        help="Pytest marker expression (e.g. 'not groundedness')",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        default=False,
        help="Launch the Streamlit dashboard automatically after tests complete",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Additional arguments passed through to pytest",
    )

    args = parser.parse_args()
    return run_evals(
        suite=args.suite,
        markers=args.markers,
        verbose=args.verbose,
        launch_streamlit=args.streamlit,
        extra_args=args.extra,
    )


if __name__ == "__main__":
    sys.exit(main())
