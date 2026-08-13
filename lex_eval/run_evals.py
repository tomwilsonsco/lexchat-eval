#!/usr/bin/env python3
"""
Run LexChat evaluations using pytest + DeepEval.

This script wraps pytest so the full evaluation suite can be launched
from the command line with sensible defaults and optional filters.

Each suite writes results to the shared DuckDB database (data/responses.db)
in the eval_results table.

By default, existing results are preserved: if a (question, LLM) pair
already has 1+ results in the suite's records, the test is skipped.
Use ``--overwrite`` to force re-running all tests and replacing existing
results.

Tests run in parallel via pytest-xdist (``EVAL_WORKERS`` in lex_eval/.env,
default 4). This mainly speeds up the AI-judge suites (groundedness,
consistency_llm), which are otherwise a long serial chain of blocking
OpenRouter calls. Use ``--workers 1`` to disable and run single-process.

Examples
--------
Run everything (skipping already-completed tests):
    python lex_eval/run_evals.py

Run only groundedness (requires OPENROUTER_API_KEY):
    python lex_eval/run_evals.py --suite groundedness

Force re-run (overwrite existing results):
    python lex_eval/run_evals.py --suite groundedness --overwrite

Force re-run a single metric only (leaves the suite's other metrics alone):
    python lex_eval/run_evals.py --suite groundedness --test-name response_groundedness --overwrite

Run only tool-usage checks (fast, no LLM judge needed):
    python lex_eval/run_evals.py --suite tool_usage

Run only consistency:
    python lex_eval/run_evals.py --suite consistency

Exclude slow LLM-judge tests:
    python lex_eval/run_evals.py -m "not groundedness"

Verbose output:
    python lex_eval/run_evals.py -v
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TESTS_DIR = Path(__file__).parent / "tests" / "eval"

SUITES = {
    "tool_usage": "test_tool_usage.py",
    "groundedness": "test_groundedness.py",
    "consistency": "test_consistency.py",
    "consistency_llm": "test_consistency_llm.py",
    "structure": "test_structure.py",
}

# The individual test_name values each suite can write to eval_results, used
# to validate --test-name and to scope --overwrite to just that metric
# instead of clearing the whole suite.
SUITE_TEST_NAMES = {
    "tool_usage": ["tool_usage"],
    "groundedness": ["answer_relevancy", "response_groundedness", "research_groundedness"],
    "consistency": ["consistency"],
    "consistency_llm": ["consistency_llm"],
    "structure": [
        "mandatory_structure",
        "citation_passthrough",
        "citation_grounding",
        "genuine_gap",
    ],
}

_DEFAULT_WORKERS = 4


def _default_workers() -> int:
    """Resolve the pytest-xdist worker count: EVAL_WORKERS env var, else 4."""
    raw = os.getenv("EVAL_WORKERS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_WORKERS
    try:
        return int(raw)
    except ValueError:
        print(
            f"⚠️  EVAL_WORKERS={raw!r} is not a valid integer; "
            f"falling back to {_DEFAULT_WORKERS}"
        )
        return _DEFAULT_WORKERS


def _load_existing_results(suite: str) -> list[dict]:
    """Load existing results for a suite from DuckDB, or return [] if not found."""
    from lex_eval.utils.db import DEFAULT_DB, load_eval_results

    return load_eval_results(DEFAULT_DB, suite=suite)


def _covered_triples(results: list[dict]) -> set[tuple[int, str]]:
    """Return the set of (response_id, test_name) pairs already covered."""
    return {(int(r["response_id"]), r["test_name"]) for r in results}


def _build_deselect_args(suite: str, llm: str | None = None) -> list[str]:
    """
    Build pytest ``--deselect`` arguments for test IDs that already have a
    result for that *specific* response (via ``response_id``).

    If *llm* is given, only records matching that LLM name are considered.

    Returns an empty list if there are no existing results or if the suite file
    doesn't exist.
    """
    existing = _load_existing_results(suite)
    if llm:
        existing = [r for r in existing if r["llm_name"] == llm]
    if not existing:
        return []

    covered = _covered_triples(existing)

    # Pytest appends a numeric suffix (0, 1, …) when multiple records share
    # the same base ID, so we must replicate that here.
    from lex_eval.utils.test_helpers import load_records, record_id

    records = load_records()
    test_file = SUITES[suite]

    # build the same IDs pytest uses: base_id + counter suffix
    base_ids = [record_id(r) for r in records]
    id_counts: dict[str, int] = {}
    pytest_ids: list[str] = []
    for bid in base_ids:
        n = id_counts.get(bid, 0)
        pytest_ids.append(f"{bid}{n}")
        id_counts[bid] = n + 1

    def _covered(record: dict, test_name: str) -> bool:
        return (int(record["response_id"]), test_name) in covered

    deselect_args: list[str] = []
    for record, pid in zip(records, pytest_ids):
        if suite == "groundedness":
            for test_name, fn_name in (
                ("answer_relevancy", "test_answer_relevancy"),
                ("response_groundedness", "test_response_groundedness"),
                ("research_groundedness", "test_research_groundedness"),
            ):
                if _covered(record, test_name):
                    deselect_args.extend(
                        [
                            "--deselect",
                            f"lex_eval/tests/eval/{test_file}::{fn_name}[{pid}]",
                        ]
                    )
        elif suite == "tool_usage":
            if _covered(record, "tool_usage"):
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/eval/{test_file}::test_tool_usage[{pid}]",
                    ]
                )
        elif suite == "consistency":
            if _covered(record, "consistency"):
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/eval/{test_file}::test_consistency[{pid}]",
                    ]
                )
        elif suite == "structure":
            for test_name, fn_name in (
                ("mandatory_structure", "test_mandatory_structure"),
                ("citation_passthrough", "test_citation_passthrough"),
                ("citation_grounding", "test_citation_grounding"),
                ("genuine_gap", "test_genuine_gap"),
            ):
                if _covered(record, test_name):
                    deselect_args.extend(
                        [
                            "--deselect",
                            f"lex_eval/tests/eval/{test_file}::{fn_name}[{pid}]",
                        ]
                    )

    # consistency_llm is parametrized by (question, LLM) group, not individual
    # record, and evaluates one result per group rather than per response — so
    # it stays pair-based by design.
    if suite == "consistency_llm":
        from lex_eval.utils.test_helpers import group_by_question_and_llm

        covered_pairs = {
            (int(r["question_id"]), r["llm_name"]) for r in existing
        }
        deselect_args = []
        for key, grp_records in sorted(group_by_question_and_llm().items()):
            if len(grp_records) < 2:
                continue
            qid = int(grp_records[0]["question_id"])
            rec_llm = grp_records[0]["llm_name"]
            if (qid, rec_llm) in covered_pairs:
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/eval/{test_file}::test_consistency_llm[{key}]",
                    ]
                )

    return deselect_args


def run_evals(
    suite: str | None = None,
    markers: str | None = None,
    verbose: bool = False,
    overwrite: bool = False,
    extra_args: list[str] | None = None,
    llm: str | None = None,
    workers: int | None = None,
    test_name: str | None = None,
) -> int:
    """
    Launch pytest against the evaluation test suite.

    Returns the pytest exit code (0 = all passed).
    """
    suites_to_run = [suite] if suite and suite in SUITES else list(SUITES.keys())
    overall_rc = 0

    for s in suites_to_run:
        from lex_eval.utils.db import (
            DEFAULT_DB,
            clear_eval_results,
            get_connection,
            init_db,
            init_eval_results,
        )

        conn = get_connection(DEFAULT_DB)
        try:
            # Migrate both tables' schemas here, up front, in this single
            # read-write connection. Eval test modules load records/results
            # via read-only connections (safe under parallel pytest-xdist
            # workers) and skip migration themselves, so it must happen once
            # before pytest starts.
            init_db(conn)
            init_eval_results(conn)  # Ensure table exists first
            if overwrite:
                # Scoped to test_name when given, so re-running one metric
                # with --overwrite never wipes its sibling metrics' results.
                clear_eval_results(conn, suite=s, test_name=test_name)
            conn.commit()  # Commit after init and potential clear
        finally:
            conn.close()

        cmd: list[str] = [sys.executable, "-m", "pytest"]
        cmd.append(str(TESTS_DIR / SUITES[s]))

        if markers:
            cmd.extend(["-m", markers])

        # filter to a single LLM and/or a single test_name via a combined
        # pytest keyword expression (both are plain substrings, so "and"
        # narrows to their intersection; LLM names containing ":" are fine
        # unquoted here, same as the single-filter case below)
        keyword_filters = [f for f in (llm, test_name) if f]
        if keyword_filters:
            cmd.extend(["-k", " and ".join(keyword_filters)])

        # skip logic: deselect tests that already have results
        if not overwrite:
            deselect = _build_deselect_args(s, llm=llm)
            if deselect:
                cmd.extend(deselect)
                n_skipped = deselect.count("--deselect")
                print(
                    f"ℹ️  {s}: skipping {n_skipped} test(s) with existing results "
                    f"(use --overwrite to force)"
                )

        # parallelise via pytest-xdist unless disabled (--workers 1); applied
        # uniformly across suites so any future AI-judge suite benefits with
        # no extra wiring, and fast/offline suites just pay a small
        # worker-startup cost
        n_workers = workers if workers is not None else _default_workers()
        if n_workers != 1:
            cmd.extend(["-n", str(n_workers)])

        # display
        cmd.extend(["-v" if verbose else "-q", "--tb=short"])

        # pass-through args
        if extra_args:
            cmd.extend(extra_args)

        print(f"\n{'='*60}")
        print(f"Running suite: {s}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)

        if result.returncode > overall_rc:
            overall_rc = result.returncode

    if overall_rc in (0, 1):
        print(
            "\n📊 Results written to data/responses.db (eval_results table)"
            "\n   View dashboard: streamlit run lex_eval/reports/streamlit_report.py"
        )

    return overall_rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LexChat evaluations (pytest + DeepEval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suites:
  tool_usage        Check tools were invoked correctly (fast, offline)
  groundedness      LLM-as-judge faithfulness + relevancy checks (needs OPENROUTER_API_KEY)
  consistency       Same-model repeatability checks (fast, cosine similarity)
  consistency_llm   Same-model repeatability checks (AI judge, needs OPENROUTER_API_KEY)
  structure         Worker output structure + citation checks (fast, offline)

Results:
  All suites write to the eval_results table in data/responses.db.

  By default, tests are skipped if results already exist for that
  (question, LLM) pair.  Use --overwrite to force re-running.
  Use --llm to restrict evaluation to a single model.

Dashboard:
  Launch the Streamlit dashboard at any time:
    streamlit run lex_eval/reports/streamlit_report.py
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
        "--llm",
        metavar="LLM_NAME",
        help="Only evaluate this LLM (e.g. 'gpt-oss:120b-cloud')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results instead of skipping completed tests",
    )
    parser.add_argument(
        "--test-name",
        metavar="TEST_NAME",
        help=(
            "Only run/overwrite this one metric within --suite (e.g. "
            "response_groundedness). With --overwrite, scopes the DB clear "
            "to this test_name instead of the whole suite. Requires --suite."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel pytest-xdist workers for AI-judge calls (default: "
            "EVAL_WORKERS env var, or 4). Use --workers 1 to disable "
            "parallelism, e.g. for easier-to-read debugging output."
        ),
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

    if args.test_name:
        if not args.suite:
            parser.error("--test-name requires --suite")
        valid = SUITE_TEST_NAMES[args.suite]
        if args.test_name not in valid:
            parser.error(
                f"--test-name {args.test_name!r} is not valid for --suite "
                f"{args.suite!r}; choose from {valid}"
            )

    return run_evals(
        suite=args.suite,
        markers=args.markers,
        verbose=args.verbose,
        overwrite=args.overwrite,
        extra_args=args.extra,
        llm=args.llm,
        workers=args.workers,
        test_name=args.test_name,
    )


if __name__ == "__main__":
    sys.exit(main())
