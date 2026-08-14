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

from lex_eval.utils.db import init_db, _CREATE_TABLE

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
