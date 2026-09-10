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


def test_run_numbers_come_from_the_group_not_the_metric_rows():
    """One numbering for the question, whatever order a metric's rows arrive in."""
    from lex_eval.reports.data import run_numbers

    group = [
        {"response_id": 80, "timestamp": "2026-09-03T20:04:57"},
        {"response_id": 72, "timestamp": "2026-09-03T20:01:24"},
    ]
    numbers = run_numbers(group)
    assert numbers == {72: 1, 80: 2}
    # Reversed input is the same numbering, and a metric that scored only the
    # second attempt cannot rename it Run 1.
    assert run_numbers(list(reversed(group))) == numbers
    assert numbers[80] == 2


def test_non_verdict_states_are_distinguished():
    from lex_eval.reports.data import (
        NOT_APPLICABLE,
        NOT_COMPARABLE,
        NOT_MEASURED,
        apply_scope,
        unmeasured_state,
    )

    conversational = dict(
        test_name="mandatory_structure",
        measured=False,
        reason="Not applicable in conversational mode; Research Output Structure not measured",
    )
    assert unmeasured_state(conversational) == NOT_APPLICABLE
    assert unmeasured_state(dict(reason="Judge error: timeout")) == NOT_MEASURED
    assert unmeasured_state(dict(not_comparable=True)) == NOT_COMPARABLE
    # apply_scope's own exclusions are scope decisions, not failed measurements.
    scoped = apply_scope(
        [dict(test_name="genuine_gap", research_mode="case_law_only", measured=True)]
    )[0]
    assert unmeasured_state(scoped) == NOT_APPLICABLE


def test_incompatible_scoring_versions_are_not_comparable():
    from lex_eval.reports.data import NOT_COMPARABLE, unmeasured_state

    result = aggregate_metrics(
        [score(1, 1.0, True, metric_version="v1"), score(2, metric_version="v2")]
    )[0]
    assert result["state"] == NOT_COMPARABLE
    assert not result["scored"]
    assert all(unmeasured_state(r) == NOT_COMPARABLE for r in result["not_scored_rows"])


def _consistency_row(rid, score_value=0.373, cohort=2):
    return dict(
        response_id=rid,
        score=score_value,
        threshold=0.5,
        passed=False,
        reason=f"Mean cosine similarity: {score_value:.3f} (across {cohort} responses, threshold: 0.5)",
    )


def test_consistency_rows_are_one_comparison_not_two_verdicts():
    from lex_eval.reports.data import consistency_cohort

    cohort = consistency_cohort([_consistency_row(72), _consistency_row(80)])
    assert cohort == {
        "response_ids": [72, 80],
        "score": 0.373,
        "threshold": 0.5,
        "passed": False,
    }


@pytest.mark.parametrize(
    "rows",
    [
        # A different score is a different comparison.
        [_consistency_row(72), _consistency_row(80, 0.9)],
        # Three responses in the cohort but only two rows here: membership is
        # not established, so the rows stay as stored.
        [_consistency_row(72, cohort=3), _consistency_row(80, cohort=3)],
        # No recorded cohort size at all.
        [
            dict(response_id=72, score=0.0, threshold=0.5, passed=False, reason="x"),
            dict(response_id=80, score=0.0, threshold=0.5, passed=False, reason="x"),
        ],
    ],
)
def test_unrelated_consistency_rows_are_never_merged(rows):
    from lex_eval.reports.data import consistency_cohort

    assert consistency_cohort(rows) is None


def test_review_markdown_keeps_identity_beside_each_finding():
    from lex_eval.reports.review_export import review_markdown

    pack = {
        "question": {
            "id": 13,
            "question": "list the instruments",
            "metadata_source": "x",
        },
        "selection": {"Model": "glm", "Scoring selection": "Latest stored"},
        "responses": [
            {
                "response_id": 72,
                "run_label": "Response 72 · Run 1 of 2",
                "timestamp": "2026-09-03T20:01:24",
                "outcome": "Response received · Research limit reached",
                "research_limit_reached": True,
                "actual_output": "could you narrow this down",
            }
        ],
        "checks": [
            {
                "metric_name": "Reference Answer Agreement",
                "state": "Stable fail",
                "scored": True,
                "results": [
                    {
                        "response_id": 72,
                        "run_label": "Response 72 · Run 1 of 2",
                        "scored": True,
                        "status": "Failed",
                        "score": 0.0,
                        "reason": "States 0 of 4 reference points.",
                    }
                ],
            },
            {
                "metric_name": "Research Output Structure",
                "state": "Not measured",
                "scored": False,
                "results": [
                    {
                        "response_id": 72,
                        "run_label": "Response 72 · Run 1 of 2",
                        "scored": False,
                        "status": "Not applicable",
                        "score": None,
                        "reason": "Not applicable in conversational mode",
                    }
                ],
            },
        ],
        "review": {"observed_problem": "answered the wrong Act"},
    }
    document = review_markdown(pack)
    assert "# Review: Q13 list the instruments" in document
    assert "answered the wrong Act" in document
    # A failure and a check with no verdict are in different sections, and each
    # finding names the response it belongs to.
    assert document.index("### Failed") < document.index("### No verdict")
    assert "Response 72 · Run 1 of 2: Failed, score 0.000." in document
    assert "Response 72 · Run 1 of 2: Not applicable, no score." in document
    assert "Research limit reached" in document
    assert "could you narrow this down" in document


def test_deep_research_checks_do_not_apply_to_other_chat_modes():
    """The run's mode decides, not the wording a metric happened to write."""
    from lex_eval.reports.data import (
        NOT_APPLICABLE,
        NOT_MEASURED,
        apply_scope,
        unmeasured_state,
    )

    def row(metric, chat_mode, reason):
        return dict(
            test_name=metric, chat_mode=chat_mode, measured=False, reason=reason
        )

    # Plan Coverage said "No research plan"; Step Completion said "Not
    # deep_research". Both are the same expected absence on a conversational run.
    scoped = apply_scope(
        [
            row("plan_coverage", "conversational", "No research plan for this record;"),
            row("step_completion", "conversational", "Not deep_research;"),
            row("report_integration", "research", "Not deep_research;"),
        ]
    )
    assert [unmeasured_state(r) for r in scoped] == [NOT_APPLICABLE] * 3
    assert all(r["stored_reason"] for r in scoped)

    # A deep research run that genuinely has no plan is still a measurement gap.
    missing = apply_scope(
        [row("plan_coverage", "deep_research", "No research plan for this record;")]
    )[0]
    assert unmeasured_state(missing) == NOT_MEASURED
    # An unrecorded chat mode is not evidence that the check did not apply.
    unknown = apply_scope(
        [row("plan_coverage", None, "No research plan for this record;")]
    )[0]
    assert unmeasured_state(unknown) == NOT_MEASURED
