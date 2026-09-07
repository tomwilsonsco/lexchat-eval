"""Runner integration with simulated pytest, without model calls."""

import json
from types import SimpleNamespace

import pytest

from lex_eval import run_evals
from lex_eval.utils import db, versioning

pytestmark = pytest.mark.unit


def setup_database(tmp_path, monkeypatch):
    path = tmp_path / "responses.db"
    monkeypatch.setattr(db, "DEFAULT_DB", path)
    conn = db.get_connection(path)
    db.init_db(conn)
    rid = db.insert_response(
        conn,
        dict(
            question_id=1,
            question="q",
            llm_name="glm",
            timestamp="2026-09-07",
            actual_output="answer",
        ),
    )
    conn.close()
    return path, rid


def test_preview_is_read_only(tmp_path, monkeypatch):
    path, rid = setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(
        versioning,
        "scoring_config",
        lambda metrics, ids: dict(
            source_version="v1", judge={}, references={}, response_ids=sorted(ids)
        ),
    )
    before = path.read_bytes()
    assert run_evals.run_evals(metrics=["tool_usage"], dry_run=True) == 0
    assert path.read_bytes() == before


def test_compatible_results_skip_and_changed_code_appends(tmp_path, monkeypatch):
    path, rid = setup_database(tmp_path, monkeypatch)
    config = dict(source_version="v1", judge={}, references={}, response_ids=[rid])
    monkeypatch.setattr(versioning, "scoring_config", lambda metrics, ids: dict(config))
    commands = []

    def fake_pytest(cmd, env):
        commands.append(cmd)
        assert json.loads(env["LEX_EVAL_RESPONSE_IDS"]) == [rid]
        if "--deselect" in cmd:
            assert any(f"response{rid}]" in arg for arg in cmd)
            return SimpleNamespace(returncode=5)
        conn = db.get_connection(path)
        row = dict(
            response_id=rid,
            question_id=1,
            question="q",
            llm_name="glm",
            score=1.0,
            threshold=1.0,
            passed=True,
            scoring_run_id=env["LEX_EVAL_SCORING_RUN_ID"],
            metric_version=versioning.metric_version(config, "tool_usage"),
        )
        db.insert_eval_result(conn, "tool_usage", row)
        conn.close()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_evals.subprocess, "run", fake_pytest)
    assert run_evals.run_evals(metrics=["tool_usage"], workers=1) == 0
    assert run_evals.run_evals(metrics=["tool_usage"], workers=1) == 0
    config["source_version"] = "v2"
    assert run_evals.run_evals(metrics=["tool_usage"], workers=1) == 0
    conn = db.get_connection(path, read_only=True)
    assert conn.execute("SELECT count(*) FROM eval_tool_usage").fetchone()[0] == 2
    assert (
        conn.execute(
            "SELECT count(*) FROM scoring_runs WHERE status='finished'"
        ).fetchone()[0]
        == 3
    )
    conn.close()


def test_deployment_copy_preserves_history_and_sparse_eval_ids(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "deploy.db"
    conn = db.get_connection(source)
    db.init_db(conn)
    exp, gather = versioning.start_gather(
        conn, label="baseline", config={}, question_ids=[1]
    )
    rid = db.insert_response(
        conn,
        dict(
            question_id=1,
            question="q",
            llm_name="glm",
            timestamp="2026-09-07",
            actual_output="answer",
        ),
    )
    versioning.link_response(conn, rid, gather, exp, {"id": 1})
    db.init_eval_table(conn, "tool_usage")
    config = dict(source_version="v1", judge={})
    run = versioning.start_scoring(conn, config)
    row = dict(
        response_id=rid,
        question_id=1,
        question="q",
        llm_name="glm",
        score=1.0,
        threshold=1.0,
        passed=True,
        scoring_run_id=run,
        metric_version="v1",
    )
    db.insert_eval_result(conn, "tool_usage", row)
    db.insert_eval_result(conn, "tool_usage", row)
    conn.execute("DELETE FROM eval_tool_usage WHERE id=1")
    conn.execute("DELETE FROM eval_versions WHERE eval_id=1")
    conn.close()
    before = source.read_bytes()
    db.make_deploy_db(source, target)
    assert source.read_bytes() == before
    conn = db.get_connection(target, read_only=True)
    assert conn.execute(
        "SELECT e.id, v.scoring_run_id FROM eval_tool_usage e JOIN eval_versions v ON v.eval_id=e.id"
    ).fetchall() == [(2, run)]
    assert conn.execute("SELECT experiment_id FROM response_runs").fetchone()[0] == exp
    conn.close()
