"""Dashboard navigation and matched comparisons without a live LexChat service."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from lex_eval.reports.comparison import compare
from lex_eval.reports.diagnostics import searches
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
    app.run(timeout=30)
    assert not app.exception
    assert any("Q1" in str(frame.value) for frame in app.dataframe)
    app.toggle[0].set_value(True).run(timeout=30)
    assert not app.exception
    assert any("timeout" in e.value for e in app.error)
    assert path.read_bytes() == before


def record(rid, **overrides):
    return dict(
        response_id=rid,
        question_id=1,
        question="same wording",
        chat_mode="research",
        research_mode="legislation_only",
        actual_output="answer",
        **overrides,
    )


def verdict(rid, passed, version="v1", **overrides):
    return dict(
        response_id=rid,
        question_id=1,
        test_name="reference_answer_agreement",
        metric_name="Agreement",
        metric_version=version,
        threshold=0.6,
        score=0.6,
        passed=passed,
        measured=True,
        **overrides,
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
                actual_output="answer",
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
    assert any("Change" in frame.value.columns for frame in app.dataframe)
