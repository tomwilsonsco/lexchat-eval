"""Dashboard navigation and matched comparisons without a live LexChat service."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from lex_eval.reports.comparison import (
    PASS_FREQUENCY_CHANGE,
    PASS_RATE_DELTA,
    change_counts,
    compare,
    matched_entries,
    metric_summary,
    overview_markdown,
    overview_rows,
    shared_cohorts,
    up_to_date,
)
from lex_eval.reports.data import coverage
from lex_eval.reports.diagnostics import searches, search_summary
from lex_eval.utils.db import get_connection, init_db, insert_response

pytestmark = pytest.mark.unit


def test_error_only_dashboard_renders(tmp_path):
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
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom lex_eval.reports import streamlit_report as r\nr.RESPONSES_DB=Path({str(path)!r})\nr.main()"
    )
    app.session_state["view"] = "Review questions"
    app.run(timeout=30)
    assert not app.exception
    assert any("Q1" in str(frame.value) for frame in app.dataframe)
    # The answer and the error are on the page from the first render, in the
    # closed "Response to user" panel, with no widget to find first.
    assert any("timeout" in e.value for e in app.error)
    # A lone run opens on its own; several stay closed so they can be compared
    # without scrolling through the first. The label names the response, so the
    # same run number never points at two different responses.
    runs = [e for e in app.expander if e.label.startswith("Response 1 · Run 1 of 1")]
    assert len(runs) == 1 and runs[0].proto.expanded
    # A rerun must not migrate or otherwise write to the database.
    app.checkbox[0].set_value(True).run(timeout=30)
    assert not app.exception
    assert path.read_bytes() == before


def record(rid, **overrides):
    return (
        dict(
            response_id=rid,
            question_id=1,
            question="same wording",
            chat_mode="research",
            research_mode="legislation_only",
            actual_output="answer",
        )
        | overrides
    )


def verdict(rid, passed, version="v1", **overrides):
    return (
        dict(
            response_id=rid,
            question_id=1,
            test_name="reference_answer_agreement",
            metric_name="Agreement",
            metric_version=version,
            threshold=0.6,
            score=0.6,
            passed=passed,
            measured=True,
        )
        | overrides
    )


def test_matched_comparison_preserves_veto_and_excludes_unmatched():
    summary, changes, counts = compare(
        [record(1), record(3, question_hash="changed")],
        [record(2)],
        [verdict(1, False), verdict(2, True)],
    )
    assert summary == {
        "Matched questions and modes": 1,
        "Baseline only": 1,
        "Candidate only": 0,
    }
    assert changes[0]["Change"] == "More passes"
    assert changes[0]["Baseline"].startswith("0/1")
    assert counts[0]["Attempts"] == 1


def test_incompatible_scoring_cannot_imply_improvement():
    _, changes, _ = compare(
        [record(1)], [record(2)], [verdict(1, False), verdict(2, True, "v2")]
    )
    assert changes[0]["Change"].startswith("Not comparable")


def test_missing_measurement_stays_visible():
    _, changes, _ = compare(
        [record(1)], [record(2), record(3)], [verdict(1, False), verdict(2, True)]
    )
    assert changes[0]["Change"] == "Incomplete measurements"
    assert "1 unmeasured or missing" in changes[0]["Candidate"]


def test_metric_summary_totals_the_shared_questions():
    baseline = [record(1), record(3, question_id=2, question="second")]
    candidate = [record(2), record(4, question_id=2, question="second")]
    rows = [
        verdict(1, False, score=0.4),
        verdict(2, True, score=0.8),
        verdict(3, True, question_id=2, score=1.0),
        verdict(4, True, question_id=2, score=0.6),
    ]
    row = metric_summary(baseline, candidate, rows)[0]
    assert row["Questions compared"] == 2 and row["Not compared"] == 0
    assert (row["Baseline passes"], row["Baseline measured"]) == (1, 2)
    assert (row["Candidate passes"], row["Candidate measured"]) == (2, 2)
    # The means are equal while the pass rate rose, by one run, which is
    # within normal variation.
    assert row["Baseline mean"] == row["Candidate mean"] == 0.7
    assert row[PASS_RATE_DELTA] == 50
    assert row[PASS_FREQUENCY_CHANGE] == "Within normal variation"


def test_change_counts_account_for_every_check():
    baseline = [record(1), record(3, question_id=2, question="second")]
    candidate = [record(2), record(4, question_id=2, question="second")]
    rows = [
        verdict(1, False),
        verdict(2, True),
        verdict(3, True, question_id=2, test_name="tool_usage", metric_name="Tool"),
        verdict(4, True, question_id=2, test_name="tool_usage", metric_name="Tool"),
    ]
    totals = metric_summary(baseline, candidate, rows)
    counts = change_counts(totals)
    assert counts["Checks"] == len(totals) == 2
    assert counts["Within normal variation"] == 2
    assert sum(counts[state] for state in counts if state != "Checks") == 2


def test_metric_summary_excludes_incompatible_versions_from_the_totals():
    baseline = [record(1), record(3, question_id=2, question="second")]
    candidate = [record(2), record(4, question_id=2, question="second")]
    rows = [
        verdict(1, False),
        verdict(2, True),
        verdict(3, False, question_id=2),
        verdict(4, True, "v2", question_id=2),
    ]
    row = metric_summary(baseline, candidate, rows)[0]
    assert row["Questions compared"] == 1 and row["Not compared"] == 1
    assert (row["Baseline passes"], row["Baseline measured"]) == (0, 1)
    assert (row["Candidate passes"], row["Candidate measured"]) == (1, 1)


def test_metric_summary_reports_unmeasured_runs_beside_the_totals():
    row = metric_summary(
        [record(1)],
        [record(2), record(3)],
        [verdict(1, True), verdict(2, True)],
    )[0]
    assert row["Unmeasured runs"] == 1


def test_evidence_gets_the_rows_the_totals_were_built_from():
    """The shared scoring version, not the newest row either side happens to hold.

    The baseline was rescored under v2 and the candidate was not, so "latest
    stored" would read v2 against v1. The comparison selects v1 on both sides,
    and the evidence panel has to be handed that same v1 row.
    """
    baseline = [record(1)]
    candidate = [record(2)]
    shared = verdict(1, False, score=0.4, id=1, run_at="2026-09-01")
    rescored = verdict(1, True, "v2", score=0.9, id=3, run_at="2026-09-08")
    rows = [shared, rescored, verdict(2, True, score=0.8, id=2, run_at="2026-09-02")]

    entry = matched_entries(baseline, candidate, rows)[0]
    assert [[r["id"] for r in side] for side in entry["selected"]] == [[1], [2]]
    assert entry["totals"][0]["raw_results"] == [shared]
    # The same selection the per question table reports, so an individual
    # verdict below always belongs to the counts above it.
    _, changes, _ = compare(baseline, candidate, rows)
    assert changes[0]["Baseline"].startswith("0/1")
    assert changes[0]["Change"] == "More passes"


def test_evidence_for_an_uncomparable_check_carries_its_reason():
    entry = matched_entries(
        [record(1)], [record(2)], [verdict(1, False), verdict(2, True, "v2")]
    )[0]
    assert entry["excluded"].startswith("Not comparable")
    assert "totals" not in entry


def test_shared_questions_need_the_same_wording_and_snapshot():
    baseline = [record(1), record(3, question_id=2, question="second")]
    candidate = [
        record(2),
        record(4, question_id=2, question="second", question_hash="changed"),
    ]
    assert {key[0] for key in shared_cohorts(baseline, candidate)} == {1}


def test_coverage_counts_stored_results_and_skips_checks_that_do_not_apply():
    metrics = [
        ("tool_usage", "Tool Usage", ""),
        ("plan_coverage", "Plan Coverage (Deep research only)", ""),
    ]
    records = [
        record(1, chat_mode="conversational"),
        record(2, chat_mode="conversational"),
    ]
    table, pending = coverage(
        records, [verdict(1, True, test_name="tool_usage")], metrics
    )
    assert table[0]["Responses"] == 2 and table[0]["Answered"] == 2
    # One of the two responses has a Tool Usage row, neither has a deep
    # research one, and a deep research check does not cover these runs.
    assert table[0]["Tool Usage"] == "1"
    assert table[0]["Plan Coverage (Deep research only)"] == "n/a"
    assert pending == ["tool_usage"]


def test_experiment_info_lists_an_unscored_experiment_and_its_command(tmp_path):
    from lex_eval.utils.versioning import start_gather, link_response

    path = tmp_path / "unscored.db"
    conn = get_connection(path)
    init_db(conn)
    exp, gather = start_gather(
        conn, label="probe", config={"label": "probe"}, question_ids=[1]
    )
    rid = insert_response(
        conn,
        dict(
            question_id=1,
            question="q",
            llm_name="m",
            timestamp="2026-09-07",
            actual_output="answer",
        ),
    )
    link_response(conn, rid, gather, exp, {"id": 1, "question": "q"})
    conn.close()
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom lex_eval.reports import streamlit_report as r\nr.RESPONSES_DB=Path({str(path)!r})\nr.main()"
    )
    app.run(timeout=30)
    app.radio[0].set_value("Experiment info").run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].options == [f"probe ({exp[:8]})"]
    assert any("Tool Usage" in frame.value.columns for frame in app.dataframe)
    assert app.code[0].value.startswith(
        f"python lex_eval/run_evals.py --experiment {exp} --metrics "
    )


def test_search_recovery_and_errors_are_distinct():
    def tool(raw, **flags):
        return dict(
            name="search_case_law",
            args={"query": "Evans"},
            raw_result=json.dumps(raw),
            **flags,
        )

    audit = {
        "delegations": [
            {
                "step": 1,
                "tools": [
                    tool({"results": []}),
                    tool({"results": [{"title": "Evans"}]}),
                    tool({"error": "HTTP 503"}),
                    tool({"results": []}, truncated=True),
                ],
            }
        ]
    }
    rows = searches(record(1, audit_json=audit))
    assert [r["Outcome"] for r in rows] == [
        "Empty",
        "Results returned",
        "Error",
        "Unknown / incomplete",
    ]
    assert rows[0]["Later nonempty search in this step"]
    assert rows[2]["Returned"] is None


def test_search_summary_groups_calls_and_keeps_unknown_counts_blank():
    def tool(name, raw, **flags):
        return dict(
            name=name, args={"query": "Evans"}, raw_result=json.dumps(raw), **flags
        )

    audit = {
        "delegations": [
            {
                "step": 1,
                "tools": [
                    tool("search_case_law", {"results": [{"title": "Evans"}]}),
                    tool("search_case_law", {"error": "HTTP 503"}),
                ],
            },
            {
                "step": 2,
                "tools": [
                    tool(
                        "search_case_law", {"results": [{"title": "A"}, {"title": "B"}]}
                    ),
                    tool("search_legislation", {"results": []}),
                ],
            },
        ]
    }
    summary = search_summary(searches(record(1, audit_json=audit)))
    assert summary == [
        {
            "Response": 1,
            "Tool": "search_case_law",
            "Outcome": "Error",
            "Searches": 1,
            # An errored call returns no count, so summing it would invent one.
            "Items returned": None,
        },
        {
            "Response": 1,
            "Tool": "search_case_law",
            "Outcome": "Results returned",
            "Searches": 2,
            "Items returned": 3,
        },
        {
            "Response": 1,
            "Tool": "search_legislation",
            "Outcome": "Empty",
            "Searches": 1,
            "Items returned": 0,
        },
    ]


def test_comparison_view_renders_recorded_experiments(tmp_path):
    from lex_eval.utils.db import init_eval_table, insert_eval_result
    from lex_eval.utils.versioning import start_gather, link_response, start_scoring

    path = tmp_path / "comparison.db"
    conn = get_connection(path)
    init_db(conn)
    init_eval_table(conn, "tool_usage")
    for label, passed in (("baseline", False), ("candidate", True)):
        exp, gather = start_gather(
            conn, label=label, config={"label": label}, question_ids=[1]
        )
        rid = insert_response(
            conn,
            dict(
                question_id=1,
                question="q",
                llm_name="m",
                timestamp="2026-09-07",
                actual_output=f"{label} answer",
            ),
        )
        link_response(conn, rid, gather, exp, {"id": 1, "question": "q"})
        run = start_scoring(conn, dict(source_version="v1", judge={}))
        insert_eval_result(
            conn,
            "tool_usage",
            dict(
                response_id=rid,
                question_id=1,
                question="q",
                llm_name="m",
                score=float(passed),
                threshold=1.0,
                passed=passed,
                scoring_run_id=run,
                metric_version="v1",
            ),
        )
    conn.close()
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom lex_eval.reports import streamlit_report as r\nr.RESPONSES_DB=Path({str(path)!r})\nr.main()"
    )
    app.run(timeout=30)
    assert not app.exception
    app.radio[0].set_value("Compare experiments").run(timeout=30)
    assert not app.exception
    assert any(
        "Matched questions and modes" in frame.value.columns for frame in app.dataframe
    )
    # The per check summary reads without opening a question, and the per
    # question detail is still there behind its expander.
    assert any("Questions compared" in frame.value.columns for frame in app.dataframe)
    detail = app.table[0].value
    assert list(detail.columns) == ["Question", "Baseline", "Candidate", "Change"]
    assert detail["Baseline"].iloc[0].startswith("0/1")
    assert detail["Candidate"].iloc[0].startswith("1/1")
    panels = [e for e in app.expander if e.label.startswith("Inspect evidence")]
    assert len(panels) == 2  # Both the failing baseline and passing candidate.
    for panel, label, other in zip(
        panels, ("baseline", "candidate"), ("candidate", "baseline"), strict=True
    ):
        text = " ".join(m.value for m in panel.markdown)
        assert f"{label} answer" in text
        assert f"{other} answer" not in text
    # The evidence panel is on the page, and it selects a check and a question.
    labels = [box.label for box in app.selectbox]
    assert labels[2:5] == ["Chat mode", "Check", "Question"]
    assert any("Failed" in str(item.value) for item in app.markdown)


def test_comparison_offers_only_the_questions_both_experiments_answered(tmp_path):
    from lex_eval.utils.db import init_eval_table, insert_eval_result
    from lex_eval.utils.versioning import start_gather, link_response, start_scoring

    path = tmp_path / "shared.db"
    conn = get_connection(path)
    init_db(conn)
    init_eval_table(conn, "tool_usage")
    # The baseline asked both questions, the candidate only the first, which is
    # the shape that used to empty the experiment list and reset the pickers.
    for label, question_ids in (("baseline", [1, 2]), ("candidate", [1])):
        exp, gather = start_gather(
            conn, label=label, config={"label": label}, question_ids=question_ids
        )
        for qid in question_ids:
            rid = insert_response(
                conn,
                dict(
                    question_id=qid,
                    question=f"q{qid}",
                    llm_name="m",
                    timestamp="2026-09-07",
                    actual_output="answer",
                ),
            )
            link_response(conn, rid, gather, exp, {"id": qid, "question": f"q{qid}"})
            run = start_scoring(conn, dict(source_version="v1", judge={}))
            insert_eval_result(
                conn,
                "tool_usage",
                dict(
                    response_id=rid,
                    question_id=qid,
                    question=f"q{qid}",
                    llm_name="m",
                    score=1.0,
                    threshold=1.0,
                    passed=True,
                    scoring_run_id=run,
                    metric_version="v1",
                ),
            )
    # A third experiment nothing has been scored against has nothing to
    # compare, so the pickers must leave it out.
    exp, gather = start_gather(
        conn, label="unscored", config={"label": "unscored"}, question_ids=[1]
    )
    rid = insert_response(
        conn,
        dict(
            question_id=1,
            question="q1",
            llm_name="m",
            timestamp="2026-09-07",
            actual_output="answer",
        ),
    )
    link_response(conn, rid, gather, exp, {"id": 1, "question": "q1"})
    conn.close()
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom lex_eval.reports import streamlit_report as r\nr.RESPONSES_DB=Path({str(path)!r})\nr.main()"
    )
    app.run(timeout=30)
    app.radio[0].set_value("Compare experiments").run(timeout=30)
    assert not app.exception
    labels = [box.label for box in app.selectbox]
    assert "Baseline experiment" in labels and "Candidate experiment" in labels
    assert "Model" not in labels and "Research mode" not in labels
    assert not any("unscored" in option for option in app.selectbox[0].options)
    assert len(app.selectbox[0].options) == 2
    questions = app.multiselect[0]
    assert questions.options == ["Q1"]
    chosen = [box.value for box in app.selectbox[:2]]
    questions.set_value(["Q1"]).run(timeout=30)
    assert not app.exception
    # Choosing a question leaves both experiments as they were.
    assert [box.value for box in app.selectbox[:2]] == chosen
    # Experiment info lists every experiment, including the unscored one.
    app.radio[0].set_value("Experiment info").run(timeout=30)
    assert not app.exception
    assert any("unscored" in option for option in app.selectbox[0].options)


def _states_fixture(path):
    """One question, two attempts, and one stored row of every non-verdict kind.

    Synthetic, not an observation: the scores here are written to exercise the
    display, and every row states which response it belongs to.
    """
    from lex_eval.utils.db import init_eval_table, insert_eval_result

    conn = get_connection(path)
    init_db(conn)
    first = insert_response(
        conn,
        dict(
            question_id=1,
            question="does the display say what happened",
            llm_name="m",
            timestamp="2026-09-01T10:00:00",
            actual_output="FIRST ANSWER: the research agent could not finish, could you narrow this down",
            chat_mode="conversational",
            max_turns_halted=1,
        ),
    )
    second = insert_response(
        conn,
        dict(
            question_id=1,
            question="does the display say what happened",
            llm_name="m",
            timestamp="2026-09-01T11:00:00",
            actual_output="SECOND ANSWER: a different answer to the same question entirely",
            chat_mode="conversational",
        ),
    )

    def store(metric, response_id, **overrides):
        init_eval_table(conn, metric)
        insert_eval_result(
            conn,
            metric,
            {
                "response_id": response_id,
                "llm_name": "m",
                "question_id": 1,
                "question": "does the display say what happened",
                "score": 0.0,
                "threshold": 1.0,
                "passed": False,
                **overrides,
            },
        )

    # Deliberately stored second attempt first: run numbering must come from
    # the question group, not from the order a metric's rows are read in.
    store("tool_usage", second, score=1.0, passed=True, reason="All tools used.")
    store("tool_usage", first, reason="delegate_research never called.")
    for rid in (first, second):
        store(
            "mandatory_structure",
            rid,
            measured=False,
            reason="Not applicable in conversational mode; Research Output Structure not measured",
        )
        store(
            "consistency",
            rid,
            score=0.373,
            threshold=0.5,
            reason="Mean cosine similarity: 0.373 (across 2 responses, threshold: 0.5)",
        )
    store(
        "response_groundedness",
        first,
        measured=False,
        reason="Judge error: the judge returned no verdict",
    )
    store("response_groundedness", second, score=1.0, passed=True, reason="Grounded.")
    # Only one of the two responses was ever scored for this check.
    store("citation_passthrough", first, reason="Failure A: no reference links.")
    # A stored failure that scores well above its threshold, the shape a
    # contradiction veto leaves behind. The verdict decides, not the number.
    store(
        "claim_support",
        first,
        score=0.92,
        threshold=0.6,
        passed=False,
        reason="Vetoed: the answer contradicts a point it otherwise supports.",
    )
    # Two thresholds means two different checks; they cannot be averaged.
    store("citation_grounding", first, score=1.0, passed=True, threshold=0.5)
    store("citation_grounding", second, score=1.0, passed=True, threshold=0.9)
    conn.close()
    return first, second


def _run_dashboard(path):
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom lex_eval.reports import streamlit_report as r\nr.RESPONSES_DB=Path({str(path)!r})\nr.main()"
    )
    # The page opens on the overview; these tests are about the question review.
    app.session_state["view"] = "Review questions"
    app.run(timeout=60)
    assert not app.exception
    return app


def test_every_non_verdict_state_says_which_one_it_is(tmp_path):
    first, second = _states_fixture(tmp_path / "states.db")
    app = _run_dashboard(tmp_path / "states.db")
    labels = [e.label for e in app.expander]

    def label_for(name):
        return next(label for label in labels if name in label)

    # An expected exclusion is not reported as a measurement that failed to
    # happen, and never as a judge or capture error.
    assert "Not applicable" in label_for("Research Output Structure")
    assert not any("judge error or capture gate" in label for label in labels)
    # A real judge error is a measurement gap beside a measured pass.
    groundedness = label_for("Response Groundedness")
    assert "Passed" in groundedness and "1 not measured" in groundedness
    # Incompatible thresholds are not averaged into a verdict, and neither row
    # inside that group is shown as the pass its own stored verdict claims.
    assert "Not comparable" in label_for("Citation Grounding")
    assert not any(
        "Not comparable" in v and "Passed" in v for v in (m.value for m in app.markdown)
    )
    # Nothing on the page offers a bare N/A, which read as a pass in colour.
    assert not any("N/A" in m.value for m in app.markdown)
    # A high score with a stored failure is a failure, not a near pass.
    claim_support = label_for("Claim Support")
    assert "0.920" in claim_support and "Failed" in claim_support
    assert ":green[" not in claim_support


def test_run_numbers_are_the_same_response_in_every_panel(tmp_path):
    first, second = _states_fixture(tmp_path / "states.db")
    app = _run_dashboard(tmp_path / "states.db")
    values = [m.value for m in app.markdown]
    # Tool Usage stored the second attempt first; the failure it reports still
    # belongs to run 1, and the pass to run 2.
    assert any(f"Response {first} · Run 1 of 2" in v and "Failed" in v for v in values)
    assert any(f"Response {second} · Run 2 of 2" in v and "Passed" in v for v in values)
    # A metric with no row for the second response cannot renumber the first.
    assert any("No stored result" in v and f"Response {second}" in v for v in values)


def test_repeat_similarity_is_one_comparison_naming_both_responses(tmp_path):
    first, second = _states_fixture(tmp_path / "states.db")
    app = _run_dashboard(tmp_path / "states.db")
    summary = next(m.value for m in app.markdown if "Answer-text similarity" in m.value)
    assert f"Response {first} (run 1)" in summary
    assert f"Response {second} (run 2)" in summary
    assert "0.373" in summary


def test_evidence_shows_the_answer_the_score_was_about(tmp_path):
    first, second = _states_fixture(tmp_path / "states.db")
    app = _run_dashboard(tmp_path / "states.db")
    panels = [
        (e.label, " ".join(m.value for m in e.markdown))
        for e in app.expander
        if e.label.startswith("Inspect evidence")
    ]
    for label, joined in panels:
        wanted = "FIRST ANSWER" if f"Response {first} " in label else "SECOND ANSWER"
        unwanted = "SECOND ANSWER" if wanted == "FIRST ANSWER" else "FIRST ANSWER"
        assert wanted in joined and unwanted not in joined
    # Attached where there is something to check, not to every row. Repeating
    # every answer and Worker report once per check made the page many times
    # larger than the reading it supports. Response 2 passes Tool Usage and
    # Response Groundedness, and its shared consistency row is shown without a
    # panel, so only its two non-verdict rows carry one.
    assert sum(f"Response {second} " in label for label, _ in panels) == 2


def test_received_text_is_not_reported_as_completed_research(tmp_path):
    first, _ = _states_fixture(tmp_path / "states.db")
    app = _run_dashboard(tmp_path / "states.db")
    heading = next(
        e.label for e in app.expander if e.label.startswith(f"Response {first} ")
    )
    assert "Response received" in heading
    assert "Research limit reached" in heading
    assert any(
        "Response received" in str(frame.value.columns.tolist())
        for frame in app.dataframe
    )


def test_comparison_empty_state_returns_to_repeated_response_review(tmp_path):
    """Without two experiments, the reader is sent where repeats can be read."""
    _states_fixture(tmp_path / "states.db")
    app = _run_dashboard(tmp_path / "states.db")
    app.radio[0].set_value("Compare experiments").run(timeout=60)
    assert not app.exception
    assert app.button[0].label == "Review repeated responses instead"
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.radio[0].value == "Review questions"
    assert any(e.label.startswith("Compare two responses") for e in app.expander)


def test_export_says_the_same_thing_as_the_screen():
    """The document a reviewer sends on must not contradict the page."""
    from lex_eval.reports import streamlit_report as sr
    from lex_eval.reports.data import aggregate_metrics

    identity = {
        1: {
            "run": 1,
            "label": "Response 1 · Run 1 of 2",
            "short": "Response 1 (run 1)",
        },
        2: {
            "run": 2,
            "label": "Response 2 · Run 2 of 2",
            "short": "Response 2 (run 2)",
        },
    }

    def row(rid, **overrides):
        return {
            "response_id": rid,
            "test_name": "consistency",
            "metric_name": "Consistency (Cosine)",
            "score": 0.373,
            "threshold": 0.5,
            "passed": False,
            "measured": True,
            "reason": "Mean cosine similarity: 0.373 (across 2 responses, threshold: 0.5)",
            **overrides,
        }

    # One shared comparison is one entry naming both responses, not one each.
    shared = sr._check_results(aggregate_metrics([row(1), row(2)])[0], identity)
    assert len(shared) == 1
    assert shared[0]["run_label"] == "Response 1 (run 1), Response 2 (run 2)"
    assert shared[0]["status"] == "Failed"

    # An expected exclusion is headed the way the metric row heads it.
    excluded = aggregate_metrics(
        [
            row(
                1,
                test_name="mandatory_structure",
                metric_name="Research Output Structure",
                measured=False,
                reason="Not applicable in conversational mode; not measured",
            )
        ]
    )[0]
    assert sr._headline_state(excluded) == "Not applicable"
    assert sr._check_results(excluded, identity)[0]["status"] == "Not applicable"

    # A row excluded as incomparable is exported as incomparable, not as the
    # pass its own stored verdict claims.
    incomparable = aggregate_metrics(
        [
            row(
                1,
                test_name="citation_grounding",
                metric_name="Citation Grounding",
                score=1.0,
                passed=True,
                threshold=0.5,
                reason="ok",
            ),
            row(
                2,
                test_name="citation_grounding",
                metric_name="Citation Grounding",
                score=1.0,
                passed=True,
                threshold=0.9,
                reason="ok",
            ),
        ]
    )[0]
    assert sr._headline_state(incomparable) == "Not comparable"
    assert {r["status"] for r in sr._check_results(incomparable, identity)} == {
        "Not comparable"
    }


def test_review_notes_survive_leaving_the_question_and_do_not_follow_it(tmp_path):
    """Streamlit drops a widget's value when the widget stops being drawn.

    A note is authored work, so it is kept for the session and keyed by the
    whole question group, not by the widget.
    """
    from lex_eval.utils.db import init_eval_table, insert_eval_result

    path = tmp_path / "notes.db"
    conn = get_connection(path)
    init_db(conn)
    init_eval_table(conn, "tool_usage")
    for qid in (1, 2):
        rid = insert_response(
            conn,
            dict(
                question_id=qid,
                question=f"question {qid}",
                llm_name="m",
                timestamp="2026-09-01T10:00:00",
                actual_output="an answer",
            ),
        )
        insert_eval_result(
            conn,
            "tool_usage",
            dict(
                response_id=rid,
                llm_name="m",
                question_id=qid,
                question=f"question {qid}",
                score=0.0,
                threshold=1.0,
                passed=False,
                reason="delegate_research never called.",
            ),
        )
    conn.close()
    app = _run_dashboard(path)

    def pick(question_id):
        app.multiselect[0].set_value([question_id]).run(timeout=60)
        assert not app.exception

    pick(1)
    app.text_area[0].set_value("Cited the wrong Act.").run(timeout=60)
    assert app.text_area[0].value == "Cited the wrong Act."
    pick(2)
    assert app.text_area[0].value == ""
    pick(1)
    assert app.text_area[0].value == "Cited the wrong Act."


def test_comparison_names_each_response_own_check_results():
    from lex_eval.reports import streamlit_report as sr
    from lex_eval.reports.data import aggregate_metrics

    def score(rid, metric, passed, **overrides):
        return {
            "response_id": rid,
            "test_name": metric,
            "metric_name": metric.replace("_", " ").title(),
            "score": float(passed),
            "threshold": 1.0,
            "passed": passed,
            "measured": True,
            "reason": "",
            **overrides,
        }

    metrics = [
        aggregate_metrics(
            [score(1, "tool_usage", False), score(2, "tool_usage", True)]
        )[0],
        aggregate_metrics(
            [
                score(1, "citation_read", True),
                score(
                    2, "citation_read", False, measured=False, reason="Judge error: x"
                ),
            ]
        )[0],
    ]
    assert "Failed: Tool Usage" in sr._response_state_line(1, metrics)
    assert "Passed: 1" in sr._response_state_line(1, metrics)
    # The second response failed nothing but has a gap, and says so.
    second = sr._response_state_line(2, metrics)
    assert "Failed" not in second
    assert "Passed: 1" in second and "No verdict: 1" in second


def test_markdown_export_carries_scoring_provenance():
    from lex_eval.reports import streamlit_report as sr
    from lex_eval.reports.data import aggregate_metrics
    from lex_eval.reports.review_export import review_markdown

    identity = {
        1: {"run": 1, "label": "Response 1 · Run 1 of 1", "short": "Response 1 (run 1)"}
    }
    m = aggregate_metrics(
        [
            {
                "response_id": 1,
                "test_name": "reference_answer_agreement",
                "metric_name": "Reference Answer Agreement",
                "score": 0.0,
                "threshold": 0.6,
                "passed": False,
                "measured": True,
                "reason": "States 0 of 4 reference points.",
                "run_at": "2026-09-07T07:54:23.827275+00:00",
                "judge_llm": "openai/gpt-4o",
            }
        ]
    )[0]
    document = review_markdown(
        {"checks": [{**m, "results": sr._check_results(m, identity)}]}
    )
    # A reader given only the report can still say which scoring event and which
    # judge produced the finding.
    assert "scored 2026-09-07T07:54:23" in document
    assert "judge: openai/gpt-4o" in document
    assert "legacy scorer version unknown" in document


def test_overview_counts_consistency_per_question_and_shows_cost():
    rows = [
        verdict(1, True, test_name="consistency", metric_name="Consistency"),
        verdict(2, True, test_name="consistency", metric_name="Consistency"),
        verdict(3, False, test_name="consistency", metric_name="Consistency"),
        verdict(4, False, test_name="consistency", metric_name="Consistency"),
        verdict(1, True),
        verdict(2, False),
        verdict(3, True),
        verdict(4, True),
    ]
    matched, (base, cand) = overview_rows(
        [
            [record(1, total_cost_usd=0.1), record(2, total_cost_usd=0.3)],
            [
                record(3, total_ms=4000),
                record(4, total_ms=8000),
                record(5, question_id=9),
            ],
        ],
        rows,
    )
    assert matched == 1
    assert base["checks"] == {"Consistency": "1/1", "Agreement": "1/2"}
    assert cand["checks"] == {"Consistency": "0/1", "Agreement": "2/2"}
    # Question 9 was asked by one side only, so it is in neither total.
    assert cand["Responses"] == 2
    assert base["Total cost ($)"] == 0.4 and base["Mean time (s)"] is None
    assert cand["Slowest (s)"] == 8


def test_overview_needs_a_scoring_version_every_side_shares():
    _, totals = overview_rows(
        [[record(1)], [record(2)], [record(3)]],
        [verdict(1, True), verdict(2, True), verdict(3, True, "v2")],
    )
    assert {t["checks"]["Agreement"] for t in totals} == {"Not comparable"}


def test_overview_starts_with_experiments_scored_on_every_current_version():
    rows = [
        verdict(1, True, "v2", experiment_id="current", run_at="2026-10-14"),
        verdict(2, True, "v1", experiment_id="old", run_at="2026-09-01"),
        verdict(3, True, "v2", experiment_id="partial", run_at="2026-10-14"),
        verdict(1, True, "t1", experiment_id="current", test_name="tool_usage"),
        # Consistency stores nothing for a question asked once, so its absence
        # does not make an experiment out of date.
        verdict(1, True, "c1", experiment_id="current", test_name="consistency"),
    ]
    assert up_to_date(["current", "old", "partial"], rows) == ["current"]
    asked_once = verdict(3, True, "t1", experiment_id="partial", test_name="tool_usage")
    assert up_to_date(["current", "partial"], [*rows, asked_once]) == [
        "current",
        "partial",
    ]


def test_a_gap_wider_than_normal_variation_is_better_or_worse():
    def asked(rid, qid):
        return record(rid, question_id=qid, question=f"q{qid}")

    def judged(rid, qid, passed):
        return verdict(rid, passed, question_id=qid)

    baseline = [asked(q, q) for q in range(1, 5)]
    candidate = [asked(10 + q, q) for q in range(1, 5)]
    rows = [judged(q, q, False) for q in range(1, 5)] + [
        judged(10 + q, q, True) for q in range(1, 5)
    ]
    row = metric_summary(baseline, candidate, rows)[0]
    assert row[PASS_FREQUENCY_CHANGE] == "Better" and row[PASS_RATE_DELTA] == 100
    row = metric_summary(candidate, baseline, rows)[0]
    assert row[PASS_FREQUENCY_CHANGE] == "Worse"


def test_comparison_counts_consistency_per_question():
    def similar(rid, passed):
        return verdict(rid, passed, test_name="consistency", metric_name="Consistency")

    row = metric_summary(
        [record(1), record(2)],
        [record(3), record(4)],
        [similar(1, True), similar(2, True), similar(3, False), similar(4, False)],
    )[0]
    assert (row["Baseline passes"], row["Baseline measured"]) == (1, 1)
    assert (row["Candidate passes"], row["Candidate measured"]) == (0, 1)


def test_comparison_never_adds_chat_modes_together():
    baseline = [record(1), record(3, question_id=2, chat_mode="deep_research")]
    candidate = [record(2), record(4, question_id=2, chat_mode="deep_research")]
    modes = {key[2] for key in shared_cohorts(baseline, candidate)}
    assert modes == {"research", "deep_research"}


def test_overview_markdown_is_the_findings_tables():
    _, totals = overview_rows(
        [[record(1, total_cost_usd=0.5)], [record(2)]],
        [verdict(1, False), verdict(2, True)],
    )
    entries = [
        {**t, "Column": column, "Experiment": column, "Model": "m"}
        for column, t in zip(("A (1)", "B (2)"), totals, strict=True)
    ]
    text = overview_markdown(
        "research",
        1,
        entries,
        {"Agreement": "Agreement", "Step Completion": "Step Completion"},
        "2026-10-15",
    )
    assert text.startswith("# Overview, research (2026-10-15)")
    assert "| Check | A (1) | B (2) |" in text
    assert "| Agreement | 0/1 | 1/1 |" in text
    # A check no experiment has a result for is left out, and a missing cost
    # reads n/a rather than None.
    assert "Step Completion" not in text and "None" not in text


def test_a_check_that_does_not_apply_leaves_no_unmeasured_runs():
    def skipped(rid):
        return verdict(
            rid,
            False,
            measured=False,
            not_applicable=True,
            reason="Not applicable: deep research only.",
        )

    row = metric_summary([record(1)], [record(2)], [skipped(1), skipped(2)])[0]
    assert row["Unmeasured runs"] == 0
    _, changes, _ = compare([record(1)], [record(2)], [skipped(1), skipped(2)])
    assert changes[0]["Change"] == "Not applicable"


def test_a_check_one_experiment_never_stored_does_not_block_the_others():
    """Consistency writes nothing for a question asked once."""

    def similar(rid, passed):
        return verdict(rid, passed, test_name="consistency", metric_name="Consistency")

    _, (once, twice) = overview_rows(
        [[record(1)], [record(2), record(3)]],
        [similar(2, True), similar(3, True)],
    )
    assert once["checks"]["Consistency"] == "No stored result"
    # Nothing to set the other side's result against, so it is not totalled.
    assert twice["checks"]["Consistency"] == "Not compared"
    # A side whose rows carry no known version still cannot be compared.
    _, totals = overview_rows(
        [[record(1)], [record(2)]],
        [verdict(1, True, None), verdict(2, True)],
    )
    assert {t["checks"]["Agreement"] for t in totals} == {"Not comparable"}


def test_a_question_scored_on_one_side_only_is_in_no_total():
    """Baseline 1/4, candidate scored only on the question both passed.

    Totalling each side over its own questions read this as Better by 75
    points, a change made entirely of missing evaluations.
    """
    baseline = [record(q, question_id=q, question=f"q{q}") for q in range(1, 5)]
    candidate = [record(10 + q, question_id=q, question=f"q{q}") for q in range(1, 5)]
    rows = [verdict(1, True, question_id=1), verdict(11, True, question_id=1)] + [
        verdict(q, False, question_id=q) for q in range(2, 5)
    ]
    row = metric_summary(baseline, candidate, rows)[0]
    assert (row["Baseline passes"], row["Baseline measured"]) == (1, 1)
    assert (row["Candidate passes"], row["Candidate measured"]) == (1, 1)
    assert row[PASS_FREQUENCY_CHANGE] == "Within normal variation"
    assert row["Questions compared"] == 1 and row["Not compared"] == 3
    _, (base, cand) = overview_rows([baseline, candidate], rows)
    assert (
        base["checks"]["Agreement"]
        == cand["checks"]["Agreement"]
        == ("1/1 (3 not compared)")
    )
    _, changes, _ = compare(baseline, candidate, rows)
    assert {c["Change"] for c in changes if c["Question"] != "Q1"} == {
        "Not compared: no stored result on one side"
    }


def test_evidence_for_a_check_one_side_never_stored_does_not_crash(tmp_path):
    """Consistency between a once-asked baseline and a repeated candidate."""
    from lex_eval.utils.db import init_eval_table, insert_eval_result
    from lex_eval.utils.versioning import start_gather, link_response, start_scoring

    path = tmp_path / "one_sided.db"
    conn = get_connection(path)
    init_db(conn)
    for metric in ("tool_usage", "consistency"):
        init_eval_table(conn, metric)
    run = start_scoring(conn, dict(source_version="v1", judge={}))

    def store(metric, rid, reason="ok"):
        insert_eval_result(
            conn,
            metric,
            dict(
                response_id=rid,
                question_id=1,
                question="q",
                llm_name="m",
                score=1.0,
                threshold=0.5,
                passed=True,
                reason=reason,
                scoring_run_id=run,
                metric_version="v1",
            ),
        )

    for label, runs in (("baseline", 1), ("candidate", 2)):
        exp, gather = start_gather(
            conn, label=label, config={"label": label}, question_ids=[1]
        )
        for _ in range(runs):
            rid = insert_response(
                conn,
                dict(
                    question_id=1,
                    question="q",
                    llm_name="m",
                    timestamp="2026-09-07",
                    actual_output=f"{label} answer",
                ),
            )
            link_response(conn, rid, gather, exp, {"id": 1, "question": "q"})
            store("tool_usage", rid)
            if runs > 1:
                store(
                    "consistency",
                    rid,
                    "Mean cosine similarity: 0.9 (across 2 responses)",
                )
    conn.close()
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom lex_eval.reports import streamlit_report as r\nr.RESPONSES_DB=Path({str(path)!r})\nr.main()"
    )
    app.run(timeout=30)
    assert not app.exception
    app.radio[0].set_value("Compare experiments").run(timeout=30)
    check = next(box for box in app.selectbox if box.label == "Check")
    check.set_value("Consistency (Cosine)").run(timeout=30)
    assert not app.exception
    assert any(
        "no stored result on one side" in str(item.value) for item in app.markdown
    )


def test_a_check_that_failed_to_score_on_both_sides_is_not_measured():
    def failed(rid):
        return verdict(
            rid, False, measured=False, reason="Not measured: judge call failed."
        )

    _, changes, _ = compare([record(1)], [record(2)], [failed(1), failed(2)])
    assert changes[0]["Change"] == "Not measured"
