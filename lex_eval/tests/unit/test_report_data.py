"""Dashboard regressions, using stored verdicts and temporary databases."""

import pytest

from lex_eval.reports.data import (
    aggregate_metrics,
    outcome_counts,
    question_groups,
    read_database,
)
from lex_eval.utils.db import get_connection, init_db, insert_response

pytestmark = pytest.mark.unit


def score(rid, value=0.6, passed=False, **kwargs):
    return dict(
        response_id=rid,
        test_name="reference_answer_agreement",
        metric_name="Reference Answer Agreement",
        score=value,
        threshold=0.6,
        passed=passed,
        measured=True,
        **kwargs,
    )


def test_contradiction_veto_survives_aggregation():
    result = aggregate_metrics([score(96), score(108)])[0]
    assert result["score"] == 0.6
    assert result["pass_count"] == 0
    assert result["state"] == "Stable fail"
    assert not result["passed"]


def test_repeats_keep_mixed_verdicts_and_exclude_unmeasured():
    missing = {**score(3, 0.0), "measured": False}
    result = aggregate_metrics([score(1, 1.0, True), score(2), missing])[0]
    assert result["score"] == 0.8
    assert result["state"] == "Mixed"
    assert result["measured_count"] == 2
    assert result["not_scored_count"] == 1


def test_rescoring_does_not_increase_response_count():
    result = aggregate_metrics([score(1, 0.0, id=1), score(1, 1.0, True, id=2)])[0]
    assert result["n_runs"] == 1
    assert result["passed"]


def test_modes_and_question_wording_are_separate():
    base = dict(
        question_id=1,
        llm_name="model",
        chat_mode="research",
        research_mode="legislation_only",
        question="original",
    )
    records = [
        base,
        {**base, "research_mode": "case_law_only"},
        {**base, "question": "changed"},
    ]
    assert len(question_groups(records)) == 3


def test_error_only_database_is_visible_and_not_migrated(tmp_path):
    path = tmp_path / "responses.db"
    conn = get_connection(path)
    init_db(conn)
    insert_response(
        conn,
        dict(
            question_id=1,
            question="q",
            llm_name="m",
            timestamp="2026-09-07",
            error="timeout",
        ),
    )
    conn.close()
    before = path.read_bytes()
    records, rows = read_database(path, [])
    assert not rows
    assert outcome_counts(records)["Error"] == 1
    assert len(question_groups(records)) == 1
    assert path.read_bytes() == before


def test_clarification_is_an_attempt_not_an_answer():
    result = outcome_counts(
        [
            {"needs_clarification": True},
            {"actual_output": "answer", "max_turns_halted": 1},
        ]
    )
    assert result["Attempts"] == 2
    assert result["Answer"] == result["Clarification"] == result["Turn-cap flags"] == 1


def test_case_law_scope_excludes_legacy_assurances():
    from lex_eval.reports.data import apply_scope

    rows = [
        dict(test_name=k, research_mode="case_law_only", measured=True)
        for k in (
            "citation_read",
            "citation_grounding",
            "genuine_gap",
            "step_completion",
            "report_integration",
            "tool_usage",
        )
    ]
    assert all(not r["measured"] for r in apply_scope(rows))
    mixed = apply_scope(
        [
            dict(
                test_name="citation_read",
                research_mode="legislation_and_case_law",
                measured=True,
            )
        ]
    )[0]
    assert mixed["measured"]
    assert "Legislation only" in mixed["scope_note"]


@pytest.mark.parametrize(
    "tool", ["Worker: search_case_law", "Worker: get_case_law_text"]
)
def test_case_law_tools_allow_search_or_named_judgment_lookup(tool):
    from lex_eval.metrics.tool_usage import ToolUsageMetric
    from lex_eval.testcase import LLMTestCase, ToolCall

    metric = ToolUsageMetric(research_mode="case_law_only")
    metric.measure(
        LLMTestCase(
            input="q",
            actual_output="a",
            tools_called=[ToolCall(name="delegate_research"), ToolCall(name=tool)],
        )
    )
    assert metric.is_successful()
