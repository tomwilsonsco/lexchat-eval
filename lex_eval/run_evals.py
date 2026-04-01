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

Examples
--------
Run everything (skipping already-completed tests):
    python lex_eval/run_evals.py

Run only groundedness (requires OPENAI_API_KEY):
    python lex_eval/run_evals.py --suite groundedness

Force re-run (overwrite existing results):
    python lex_eval/run_evals.py --suite groundedness --overwrite

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
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TESTS_DIR = Path(__file__).parent / "tests"
REPORTS_DIR = Path(__file__).parent / "reports"

SUITES = {
    "tool_usage": "test_tool_usage.py",
    "groundedness": "test_groundedness.py",
    "consistency": "test_consistency.py",
    "consistency_llm": "test_consistency_llm.py",
    "structure": "test_structure.py",
}


def _load_existing_results(suite: str) -> list[dict]:
    """Load existing results for a suite from DuckDB, or return [] if not found."""
    from lex_eval.utils.db import DEFAULT_DB, load_eval_results

    return load_eval_results(DEFAULT_DB, suite=suite)


def _covered_pairs(results: list[dict]) -> set[tuple[int, str]]:
    """
    Return the set of (question_id, llm_name) pairs that already have
    at least one result in the `eval_results` DuckDB table.
    """
    pairs: set[tuple[int, str]] = set()
    for r in results:
        pairs.add((int(r["question_id"]), r["llm_name"]))
    return pairs


def _covered_triples(results: list[dict]) -> set[tuple[int, str, str]]:
    """
    Return the set of (question_id, llm_name, test_name) triples that already
    have a result. Used for suites with multiple test functions per pair so that
    a partially-run pair is not fully skipped.
    """
    triples: set[tuple[int, str, str]] = set()
    for r in results:
        triples.add((int(r["question_id"]), r["llm_name"], r["test_name"]))
    return triples


def _build_deselect_args(suite: str, llm: str | None = None) -> list[str]:
    """
    Build pytest ``--deselect`` arguments for test IDs whose (question_id, llm_name)
    pairs already have results.

    If *llm* is given, only records matching that LLM name are considered.

    Returns an empty list if there are no existing results or if the suite file
    doesn't exist.
    """
    existing = _load_existing_results(suite)
    if llm:
        existing = [r for r in existing if r["llm_name"] == llm]
    if not existing:
        return []

    covered = _covered_pairs(existing)
    if not covered:
        return []

    covered_triples = _covered_triples(existing)

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

    deselect_args: list[str] = []
    for record, pid in zip(records, pytest_ids):
        qid = int(record["question_id"])
        rec_llm = record["llm_name"]
        if (qid, rec_llm) in covered:
            # deselect all test functions in this suite file for this parametrize ID
            if suite == "groundedness":
                # Check per-test-function so a partially-run pair isn't fully skipped
                if (qid, rec_llm, "faithfulness") in covered_triples:
                    deselect_args.extend(
                        [
                            "--deselect",
                            f"lex_eval/tests/{test_file}::test_faithfulness[{pid}]",
                        ]
                    )
                if (qid, rec_llm, "answer_relevancy") in covered_triples:
                    deselect_args.extend(
                        [
                            "--deselect",
                            f"lex_eval/tests/{test_file}::test_answer_relevancy[{pid}]",
                        ]
                    )
            elif suite == "tool_usage":
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/{test_file}::test_tool_usage[{pid}]",
                    ]
                )
            elif suite == "consistency":
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/{test_file}::test_consistency[{pid}]",
                    ]
                )
            elif suite == "structure":
                # check per-test-function so a partially-run pair isn't fully skipped
                if (qid, rec_llm, "mandatory_structure") in covered_triples:
                    deselect_args.extend(
                        [
                            "--deselect",
                            f"lex_eval/tests/{test_file}::test_mandatory_structure[{pid}]",
                        ]
                    )
                if (qid, rec_llm, "citation_passthrough") in covered_triples:
                    deselect_args.extend(
                        [
                            "--deselect",
                            f"lex_eval/tests/{test_file}::test_citation_passthrough[{pid}]",
                        ]
                    )

    # consistency_llm is parametrized by (question, LLM) group, not individual record
    if suite == "consistency_llm":
        from lex_eval.utils.test_helpers import group_by_question_and_llm

        deselect_args = []
        for key, grp_records in sorted(group_by_question_and_llm().items()):
            if len(grp_records) < 2:
                continue
            qid = int(grp_records[0]["question_id"])
            rec_llm = grp_records[0]["llm_name"]
            if (qid, rec_llm) in covered:
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/{test_file}::test_consistency_llm[{key}]",
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
            init_eval_results,
        )

        conn = get_connection(DEFAULT_DB)
        try:
            init_eval_results(conn)  # Ensure table exists first
            if overwrite:
                clear_eval_results(conn, suite=s)
            conn.commit()  # Commit after init and potential clear
        finally:
            conn.close()

        cmd: list[str] = [sys.executable, "-m", "pytest"]
        cmd.append(str(TESTS_DIR / SUITES[s]))

        if markers:
            cmd.extend(["-m", markers])

        # filter to a single LLM via pytest keyword expression
        if llm:
            cmd.extend(["-k", llm])

        # skip logic: deselect tests that already have results
        if not overwrite:
            deselect = _build_deselect_args(s, llm=llm)
            # The connection for _build_deselect_args is opened and closed within load_eval_results
            if deselect:
                cmd.extend(deselect)
                n_skipped = deselect.count("--deselect")
                print(
                    f"ℹ️  {s}: skipping {n_skipped} test(s) with existing results "
                    f"(use --overwrite to force)"
                )

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
  groundedness      LLM-as-judge faithfulness + relevancy checks (needs OPENAI_API_KEY)
  consistency       Same-model repeatability checks (fast, Jaccard)
  consistency_llm   Same-model repeatability checks (AI judge, needs OPENAI_API_KEY)
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
        overwrite=args.overwrite,
        extra_args=args.extra,
        llm=args.llm,
    )


if __name__ == "__main__":
    sys.exit(main())
