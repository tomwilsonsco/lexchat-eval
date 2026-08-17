"""
Unit tests for ``lex_eval.utils.db``, DuckDB storage layer.

Tests focus on the ``init_db`` migration exception handling:
  - CatalogException (column already exists) → logged at DEBUG, skipped
  - Unexpected exceptions (syntax error, etc.) → logged at WARNING, not
    silently swallowed
"""

import logging

import duckdb
import pytest

from lex_eval.utils.db import (
    clear_eval_results,
    covered_response_ids,
    init_db,
    init_eval_table,
    insert_eval_result,
    load_eval_results,
    _CREATE_TABLE,
    _eval_table_name,
)

# Mark every test in this module as a unit test (fast, offline, no LLM).
pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: wrapper connection for simulating failures
# ---------------------------------------------------------------------------


class _FailingConnection:
    """Wrapper around a DuckDB connection that raises a custom exception
    for the first ALTER TABLE statement matching a pattern.

    This is needed because DuckDB's C extension connection has read-only
    attributes, so ``unittest.mock.patch.object`` cannot be used directly.
    """

    def __init__(self, conn, fail_on_alter=True, exception=None):
        self._conn = conn
        self._fail_on_alter = fail_on_alter
        self._exception = exception
        self._already_failed = False

    def execute(self, stmt, *args, **kwargs):
        if (
            self._fail_on_alter
            and not self._already_failed
            and "ALTER TABLE" in str(stmt)
            and self._exception is not None
        ):
            self._already_failed = True
            raise self._exception
        return self._conn.execute(stmt, *args, **kwargs)

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Tests, init_db migration exception handling
# ---------------------------------------------------------------------------


class TestInitDbMigrationHandling:
    """Test that init_db distinguishes expected vs unexpected migration errors."""

    def test_catalog_exception_logged_at_debug(self, caplog):
        """CatalogException (column already exists) should be logged at DEBUG.

        When the full table already exists (created by _CREATE_TABLE), the
        migration statements will raise CatalogException because the columns
        already exist. These should be logged at DEBUG, and NO WARNING logs
        should appear (regression test for the NOT NULL constraint issue).
        """
        conn = duckdb.connect(":memory:")
        # Create the full table schema (same as init_db would on a fresh DB)
        conn.execute(_CREATE_TABLE)

        # Run init_db again, migrations will try to add columns that
        # already exist, raising CatalogException
        with caplog.at_level(logging.DEBUG, logger="lex_eval.utils.db"):
            init_db(conn)

        # Should have DEBUG logs about migration skipped (for CatalogException)
        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        skipped_msgs = [r for r in debug_msgs if "Migration skipped" in r.message]
        assert len(skipped_msgs) > 0, (
            "Expected at least one DEBUG 'Migration skipped' log for "
            "CatalogException (column already exists)"
        )

        # Should NOT have any WARNING logs, all migration statements should
        # either succeed or raise CatalogException (column already exists).
        # This is a regression test for the NOT NULL constraint issue where
        # ParserException was raised instead of CatalogException.
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 0, (
            f"Expected no WARNING logs on existing database, but got: "
            f"{[r.message for r in warning_msgs]}"
        )

        conn.close()

    def test_unexpected_exception_logged_at_warning(self, caplog):
        """Non-CatalogException errors should be logged at WARNING, not DEBUG."""
        conn = duckdb.connect(":memory:")
        conn.execute(_CREATE_TABLE)

        # Wrap the connection so the first ALTER TABLE raises a
        # non-CatalogException (ParserException simulates a syntax error)
        failing_conn = _FailingConnection(
            conn,
            fail_on_alter=True,
            exception=duckdb.ParserException("Simulated syntax error"),
        )

        with caplog.at_level(logging.DEBUG, logger="lex_eval.utils.db"):
            init_db(failing_conn)

        # Should have a WARNING log about migration failure
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        failed_msgs = [r for r in warning_msgs if "Migration failed" in r.message]
        assert len(failed_msgs) > 0, (
            "Expected at least one WARNING 'Migration failed' log for "
            "non-CatalogException"
        )

        # The warning should include exc_info for diagnostics
        assert failed_msgs[0].exc_info is not None

        conn.close()

    def test_migration_continues_after_unexpected_error(self, caplog):
        """init_db should continue applying remaining migrations after a failure."""
        conn = duckdb.connect(":memory:")
        conn.execute(_CREATE_TABLE)

        # Wrap so only the first ALTER fails; subsequent ones should pass
        # through to the real connection (which will raise CatalogException
        # since columns already exist, logged at DEBUG)
        failing_conn = _FailingConnection(
            conn,
            fail_on_alter=True,
            exception=duckdb.ParserException("Simulated syntax error"),
        )

        with caplog.at_level(logging.DEBUG, logger="lex_eval.utils.db"):
            init_db(failing_conn)

        # Should have both a WARNING (for the simulated failure) and
        # DEBUG logs (for subsequent CatalogException "already exists" cases)
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]

        assert any("Migration failed" in r.message for r in warning_msgs)
        assert any(
            "Migration skipped" in r.message for r in debug_msgs
        ), "Subsequent migrations should have continued after the first failure"

        conn.close()


# ---------------------------------------------------------------------------
# Tests, per-metric eval tables
# ---------------------------------------------------------------------------


def _sample_record(**overrides):
    record = {
        "response_id": 1,
        "llm_name": "test-llm",
        "question_id": 1,
        "question": "What does section 6 say?",
        "score": 0.8,
        "threshold": 0.6,
        "passed": True,
        "reason": "States 4 of 5 reference points.",
        "error": "",
        "tools_used": ["search_legislation"],
        "judge_llm": None,
        "judge_tokens": None,
    }
    record.update(overrides)
    return record


class TestEvalTableName:
    def test_rejects_invalid_metric_name(self):
        with pytest.raises(ValueError):
            _eval_table_name("tool_usage; DROP TABLE responses")

    def test_accepts_snake_case_metric_name(self):
        assert _eval_table_name("response_groundedness") == "eval_response_groundedness"


class TestEvalTableRoundTrip:
    """init_eval_table / insert_eval_result / clear_eval_results / covered_response_ids
    against an in-memory connection (these take a connection directly, unlike
    load_eval_results which opens its own file-based connection)."""

    def test_insert_and_covered_response_ids(self):
        conn = duckdb.connect(":memory:")
        init_eval_table(conn, "tool_usage")

        insert_eval_result(conn, "tool_usage", _sample_record(response_id=1))
        insert_eval_result(conn, "tool_usage", _sample_record(response_id=2))

        assert covered_response_ids(conn, "tool_usage") == {1, 2}
        conn.close()

    def test_judge_metadata_stored_for_judge_metrics(self):
        conn = duckdb.connect(":memory:")
        init_eval_table(conn, "response_groundedness")
        insert_eval_result(
            conn,
            "response_groundedness",
            _sample_record(judge_llm="openai/gpt-4o", judge_tokens=1234),
        )
        row = conn.execute(
            "SELECT judge_llm, judge_tokens, run_at FROM eval_response_groundedness"
        ).fetchone()
        assert row[0] == "openai/gpt-4o"
        assert row[1] == 1234
        assert row[2]  # run_at is populated
        conn.close()

    def test_clear_eval_results_only_affects_its_own_table(self):
        conn = duckdb.connect(":memory:")
        init_eval_table(conn, "tool_usage")
        init_eval_table(conn, "consistency")
        insert_eval_result(conn, "tool_usage", _sample_record())
        insert_eval_result(conn, "consistency", _sample_record())

        clear_eval_results(conn, "tool_usage")

        assert covered_response_ids(conn, "tool_usage") == set()
        assert covered_response_ids(conn, "consistency") == {1}
        conn.close()


class TestLoadEvalResults:
    def test_round_trip_via_file(self, tmp_path):
        from lex_eval.utils.db import get_connection

        db_path = tmp_path / "scratch.db"
        conn = get_connection(db_path)
        init_eval_table(conn, "citation_agreement")
        insert_eval_result(
            conn, "citation_agreement", _sample_record(response_id=1, score=0.5)
        )
        conn.commit()
        conn.close()

        results = load_eval_results(db_path, metric="citation_agreement")
        assert len(results) == 1
        assert results[0]["response_id"] == 1
        assert results[0]["score"] == 0.5
        assert results[0]["judge_llm"] is None

    def test_missing_table_returns_empty_list_read_only(self, tmp_path):
        from lex_eval.utils.db import get_connection

        db_path = tmp_path / "scratch.db"
        # Ensure the file exists but the metric's table doesn't.
        get_connection(db_path).close()

        assert load_eval_results(db_path, metric="genuine_gap", read_only=True) == []

    def test_requires_metric(self):
        with pytest.raises(ValueError):
            load_eval_results(metric=None)
