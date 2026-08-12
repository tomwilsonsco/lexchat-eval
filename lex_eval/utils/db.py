"""
DuckDB storage layer for LexChat evaluation responses and metric results.

A single file-based database (responses.db) stores both tables so the
Streamlit dashboard only needs one file.  Top-level fields are stored as
proper columns; complex nested structures are stored as JSON columns.

Schema
------
responses
    id                SEQUENCE primary key
    question_id       INTEGER
    question          TEXT
    llm_name          TEXT
    timestamp         TEXT
    actual_output     TEXT        (empty string if not captured)
    retrieval_context JSON       (list of context strings; includes legislation section text and case law references)
    tools_called      JSON        (list of tool-call dicts)
    is_error          BOOLEAN     (True when the capture failed)
    error_message     TEXT        (error description, NULL on success)
    research_mode     TEXT        (legislation_only | case_law_only | legislation_and_case_law)
    case_law_context  JSON       (list of {title, ncn, court, date, url} dicts from search_case_law)
    tool_sequence     JSON       (ordered list of worker tool names called, e.g. [search_legislation, search_legislation_sections])
    fallback_used     BOOLEAN     (True when get_legislation_text was invoked)
    summarisation_llm TEXT        (model used for summarisation; equals llm_name when no separate model is configured)
    chat_mode         TEXT        (research | conversational | deep_research)
    research_plan     JSON        (deep_research only: the plan from POST /api/research/plan, NULL otherwise)

eval_results
    id          SEQUENCE primary key
    suite       TEXT    (grouping key: groundedness, tool_usage, etc.)
    llm_name    TEXT
    question_id INTEGER
    question    TEXT
    test_name   TEXT    (individual test function name)
    metric_name TEXT
    score       DOUBLE
    threshold   DOUBLE
    passed      BOOLEAN
    reason      TEXT
    error       TEXT
    tools_used  JSON    (list of tool name strings, or null)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DB = DATA_DIR / "responses.db"

_CREATE_TABLE = """
CREATE SEQUENCE IF NOT EXISTS responses_id_seq START 1;

CREATE TABLE IF NOT EXISTS responses (
    id                INTEGER DEFAULT nextval('responses_id_seq') PRIMARY KEY,
    question_id       INTEGER  NOT NULL,
    question          TEXT     NOT NULL,
    llm_name          TEXT     NOT NULL,
    timestamp         TEXT     NOT NULL,
    actual_output     TEXT     NOT NULL DEFAULT '',
    retrieval_context JSON,
    tools_called      JSON,
    research_output   TEXT     NOT NULL DEFAULT '',
    is_error          BOOLEAN  NOT NULL DEFAULT FALSE,
    error_message     TEXT,
    research_mode     TEXT     NOT NULL DEFAULT 'legislation_only',
    case_law_context  JSON,
    tool_sequence     JSON,
    fallback_used     BOOLEAN  NOT NULL DEFAULT FALSE,
    summarisation_output JSON,
    summarisation_used BOOLEAN DEFAULT FALSE,
    summarisation_llm  TEXT,
    chat_mode         TEXT     NOT NULL DEFAULT 'research',
    provider          TEXT,
    total_cost_usd    DOUBLE,
    total_ms          INTEGER,
    reformatted       BOOLEAN  NOT NULL DEFAULT FALSE,
    local_cache_hits  INTEGER  NOT NULL DEFAULT 0,
    memo_hits         INTEGER  NOT NULL DEFAULT 0,
    audit_schema_version INTEGER,
    audit_json        JSON,
    research_plan     JSON
);
"""

# Columns added after the initial schema; applied to existing databases via init_db.
# NOTE: Do NOT use IF NOT EXISTS here. DuckDB's ADD COLUMN IF NOT EXISTS silently
# resets all existing row values to the column DEFAULT instead of raising an error.
# Without IF NOT EXISTS, DuckDB raises CatalogException when the column already
# exists, which the exception handler in init_db catches and skips. This preserves
# existing data.
#
# NOTE: DuckDB does not support ADD COLUMN with NOT NULL constraints — it raises
# ParserException ("Adding columns with constraints not yet supported"). DEFAULT
# alone is fine, but we omit it here for consistency. Columns are added without
# constraints; the application code provides defaults via dict.get() with fallback
# values, and load_records() applies fallbacks when reading.
_MIGRATE_RESPONSES = [
    "ALTER TABLE responses ADD COLUMN research_mode TEXT",
    "ALTER TABLE responses ADD COLUMN case_law_context JSON",
    "ALTER TABLE responses ADD COLUMN tool_sequence JSON",
    "ALTER TABLE responses ADD COLUMN fallback_used BOOLEAN",
    "ALTER TABLE responses ADD COLUMN summarisation_output JSON",
    "ALTER TABLE responses ADD COLUMN summarisation_used BOOLEAN",
    "ALTER TABLE responses ADD COLUMN summarisation_llm TEXT",
    # --- audit event migration (LexChat commit da3070d) ---
    "ALTER TABLE responses ADD COLUMN chat_mode TEXT",
    "ALTER TABLE responses ADD COLUMN provider TEXT",
    "ALTER TABLE responses ADD COLUMN total_cost_usd DOUBLE",
    "ALTER TABLE responses ADD COLUMN total_ms INTEGER",
    "ALTER TABLE responses ADD COLUMN reformatted BOOLEAN",
    "ALTER TABLE responses ADD COLUMN local_cache_hits INTEGER",
    "ALTER TABLE responses ADD COLUMN memo_hits INTEGER",
    "ALTER TABLE responses ADD COLUMN audit_schema_version INTEGER",
    "ALTER TABLE responses ADD COLUMN audit_json JSON",
    # --- deep_research plan capture (POST /api/research/plan) ---
    "ALTER TABLE responses ADD COLUMN research_plan JSON",
]

_INSERT_RESPONSE = """
INSERT INTO responses (
    question_id, question, llm_name, timestamp, actual_output,
    retrieval_context, tools_called, research_output, is_error, error_message,
    research_mode, case_law_context, tool_sequence, fallback_used,
    summarisation_output, summarisation_used, summarisation_llm,
    chat_mode, provider, total_cost_usd, total_ms, reformatted,
    local_cache_hits, memo_hits, audit_schema_version, audit_json, research_plan
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def get_connection(
    path: Path = DEFAULT_DB, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the file if it doesn't exist.

    Pass ``read_only=True`` for read-only access. DuckDB's single-file format
    allows multiple concurrent read-only connections but only one read-write
    connection at a time — read-only mode is required for callers that may run
    alongside other processes reading the same file (e.g. pytest-xdist workers
    collecting tests in parallel).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the responses table and sequence if they don't already exist.

    Also applies column migrations so existing databases gain new fields.
    """
    conn.execute(_CREATE_TABLE)
    for stmt in _MIGRATE_RESPONSES:
        try:
            conn.execute(stmt)
        except duckdb.CatalogException:
            # Column already exists — roll back the aborted statement so the
            # connection remains usable, then skip.
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            col_hint = (
                stmt.split("ADD COLUMN")[-1].strip() if "ADD COLUMN" in stmt else stmt
            )
            logger.debug("Migration skipped (column may already exist): %s", col_hint)
        except Exception:
            # Unexpected migration failure (syntax error, type mismatch, etc.)
            # — roll back and surface it at WARNING so real failures aren't
            # silently hidden at the default INFO log level.
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            col_hint = (
                stmt.split("ADD COLUMN")[-1].strip() if "ADD COLUMN" in stmt else stmt
            )
            logger.warning("Migration failed for column: %s", col_hint, exc_info=True)


def clear_responses(conn: duckdb.DuckDBPyConnection) -> None:
    """Delete all rows from the responses table."""
    conn.execute("DELETE FROM responses")


def insert_response(conn: duckdb.DuckDBPyConnection, record: Dict[str, Any]) -> None:
    """
    Insert one record into the responses table.

    *record* is the flat dict produced by ``gather_responses.process_combination``
    with top-level keys: ``actual_output``, ``retrieval_context``, ``tools_called``.
    An ``error`` key signals a failed capture.
    """
    is_error = "error" in record

    conn.execute(
        _INSERT_RESPONSE,
        [
            record["question_id"],
            record["question"],
            record["llm_name"],
            record["timestamp"],
            record.get("actual_output", "") if not is_error else "",
            json.dumps(record.get("retrieval_context") or []),
            json.dumps(record.get("tools_called") or []),
            record.get("research_output", "") if not is_error else "",
            is_error,
            record.get("error") if is_error else None,
            record.get("research_mode", "legislation_only"),
            json.dumps(record.get("case_law_context") or []),
            json.dumps(record.get("tool_sequence") or []),
            record.get("fallback_used", False),
            # Pass None through as SQL NULL; only JSON-encode when there is real content.
            (
                json.dumps(record["summarisation_output"])
                if record.get("summarisation_output") is not None
                else None
            ),
            record.get("summarisation_used", False),
            record.get("summarisation_llm") or None,
            record.get("chat_mode", "research"),
            record.get("provider") or None,
            record.get("total_cost_usd") or None,
            record.get("total_ms") or None,
            record.get("reformatted", False),
            record.get("local_cache_hits", 0),
            record.get("memo_hits", 0),
            record.get("audit_schema_version") or None,
            record.get("audit_json") or None,
            (
                json.dumps(record["research_plan"])
                if record.get("research_plan") is not None
                else None
            ),
        ],
    )


def load_records(
    path: Optional[Path] = None,
    include_errors: bool = False,
    read_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load responses from the database and return them as flat record dicts::

        {question_id, question, llm_name, timestamp,
         actual_output, retrieval_context, tools_called}

    Error rows are excluded unless *include_errors* is True.

    Pass ``read_only=True`` when this may run concurrently with other readers
    of the same file (e.g. pytest-xdist workers collecting tests in
    parallel). Read-only connections can't run schema migrations, so callers
    that pass it are responsible for having already migrated the schema via a
    prior read-write connection (``run_evals.py`` does this once, up front).
    """
    path = path or DEFAULT_DB
    if not path.exists():
        return []

    conn = get_connection(path, read_only=read_only)
    try:
        if not read_only:
            # Ensure the schema is migrated (adds new columns to existing DBs)
            init_db(conn)
        where = "" if include_errors else "WHERE NOT is_error"
        rows = conn.execute(f"""
            SELECT id, question_id, question, llm_name, timestamp,
                   actual_output, retrieval_context, tools_called, research_output,
                   research_mode, case_law_context, tool_sequence, fallback_used,
                   summarisation_output, summarisation_used, summarisation_llm,
                   chat_mode, provider, total_cost_usd, total_ms, reformatted,
                   local_cache_hits, memo_hits, audit_schema_version, audit_json,
                   research_plan
            FROM responses
            {where}
            ORDER BY id
            """).fetchall()
    finally:
        conn.close()

    records = []
    for (
        response_id,
        qid,
        question,
        llm_name,
        timestamp,
        actual_output,
        retrieval_context_json,
        tools_called_json,
        research_output,
        research_mode,
        case_law_context_json,
        tool_sequence_json,
        fallback_used,
        summarisation_output_json,
        summarisation_used,
        summarisation_llm,
        chat_mode,
        provider,
        total_cost_usd,
        total_ms,
        reformatted,
        local_cache_hits,
        memo_hits,
        audit_schema_version,
        audit_json,
        research_plan_json,
    ) in rows:
        retrieval_context = (
            json.loads(retrieval_context_json) if retrieval_context_json else []
        )
        tools_called = json.loads(tools_called_json) if tools_called_json else []
        case_law_context = (
            json.loads(case_law_context_json) if case_law_context_json else []
        )
        tool_sequence = json.loads(tool_sequence_json) if tool_sequence_json else []
        summarisation_output = (
            json.loads(summarisation_output_json) if summarisation_output_json else []
        )
        research_plan = json.loads(research_plan_json) if research_plan_json else None
        records.append(
            {
                "response_id": response_id,
                "question_id": qid,
                "question": question,
                "llm_name": llm_name,
                "timestamp": timestamp,
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "tools_called": tools_called,
                "research_output": research_output or "",
                "research_mode": research_mode or "legislation_only",
                "case_law_context": case_law_context,
                "tool_sequence": tool_sequence,
                "fallback_used": bool(fallback_used),
                "summarisation_output": summarisation_output,
                "summarisation_used": bool(summarisation_used),
                "summarisation_llm": summarisation_llm or "",
                "chat_mode": chat_mode or "research",
                "provider": provider,
                "total_cost_usd": total_cost_usd,
                "total_ms": total_ms,
                "reformatted": bool(reformatted),
                "local_cache_hits": local_cache_hits or 0,
                "memo_hits": memo_hits or 0,
                "audit_schema_version": audit_schema_version,
                "audit_json": audit_json,
                "research_plan": research_plan,
            }
        )
    return records


def group_by_question_and_llm(
    path: Optional[Path] = None,
    read_only: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return records grouped by ``'Q{question_id}_{llm_name}'`` key.

    Excludes error rows.
    """
    records = load_records(path, read_only=read_only)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        key = f"Q{r['question_id']}_{r['llm_name']}"
        grouped.setdefault(key, []).append(r)
    return grouped


def clean_incomplete_responses(
    path: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """
    Delete rows where:
    - actual_output is empty or whitespace-only,
    - the row is an error (is_error = TRUE), OR
    - no context was captured (retrieval_context is '[]' or NULL).

    Args:
        path:    Path to the database file. Defaults to DEFAULT_DB.
        dry_run: If True, print what would be deleted without deleting.

    Returns:
        Number of rows deleted (or that would be deleted in dry_run mode).
    """
    path = path or DEFAULT_DB
    conn = get_connection(path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE TRIM(actual_output) = '' OR is_error "
            "OR retrieval_context = '[]' OR retrieval_context IS NULL"
        ).fetchone()[0]

        if dry_run:
            rows = conn.execute("""
                SELECT id, question_id, llm_name, is_error, retrieval_context
                FROM responses
                WHERE TRIM(actual_output) = '' OR is_error 
                OR retrieval_context = '[]' OR retrieval_context IS NULL
                ORDER BY question_id, llm_name
                """).fetchall()
            print(f"Dry run — {count} row(s) would be deleted:")
            for row in rows:
                rid, qid, llm, is_err, ctx = row
                if is_err:
                    tag = "error"
                elif ctx in ("[]", None):
                    tag = "no context"
                else:
                    tag = "empty output"
                print(f"  id={rid}  Q{qid}  {llm}  [{tag}]")
        else:
            conn.execute(
                "DELETE FROM responses WHERE TRIM(actual_output) = '' OR is_error "
                "OR retrieval_context = '[]' OR retrieval_context IS NULL"
            )
            conn.commit()
            print(f"Deleted {count} incomplete/error/no-context row(s).")
    finally:
        conn.close()

    return count


def completeness_report(path: Optional[Path] = None) -> None:
    """Print a summary of complete (non-empty) responses per question/LLM pair."""
    path = path or DEFAULT_DB
    if not path.exists():
        print("Database not found:", path)
        return

    conn = get_connection(path)
    try:
        rows = conn.execute("""
            SELECT
                question_id,
                llm_name,
                COUNT(*) AS total_runs, -- Total number of runs for this Q/LLM pair
                SUM(CASE WHEN TRIM(actual_output) != '' AND NOT is_error THEN 1 ELSE 0 END) AS complete_runs,
                SUM(CASE WHEN TRIM(actual_output) != '' AND NOT is_error THEN LENGTH(actual_output) ELSE 0 END) AS total_actual_output_chars,
                SUM(
                    CASE
                        WHEN TRIM(actual_output) != '' AND NOT is_error
                        THEN COALESCE(LENGTH(LIST_AGGR(JSON_EXTRACT_STRING(retrieval_context, '$[*]'), 'string_agg')), 0)
                        ELSE 0
                    END
                ) AS total_retrieval_context_chars
            FROM responses
            GROUP BY question_id, llm_name
            ORDER BY question_id, llm_name
            """).fetchall()
    finally:
        conn.close()

    print(
        f"{'Q':>3}  {'LLM':<35}  {'total':>5}  {'comp':>4}  {'out_chars':>9}  {'ctx_chars':>9}  {'ok':>4}"
    )
    print("-" * 85)
    for qid, llm, total, complete, out_chars, ctx_chars in rows:
        ok = "YES" if complete >= 2 else "NO "
        print(
            f"{qid:>3}  {llm:<35}  {total:>5}  {complete:>4}  {out_chars:>9}  {ctx_chars:>9}  {ok}"
        )

    total_pairs = len(rows)
    ready = sum(1 for _, _, _, complete, _, _ in rows if complete >= 2)
    print(f"\n{ready}/{total_pairs} pairs have >= 2 complete responses")


# ----------------------------
# EVAL

_CREATE_EVAL_RESULTS_TABLE = """
CREATE SEQUENCE IF NOT EXISTS eval_results_id_seq START 1;

CREATE TABLE IF NOT EXISTS eval_results (
    id          INTEGER DEFAULT nextval('eval_results_id_seq') PRIMARY KEY,
    suite       TEXT    NOT NULL,
    llm_name    TEXT    NOT NULL,
    question_id INTEGER NOT NULL,
    question    TEXT    NOT NULL,
    test_name   TEXT    NOT NULL,
    metric_name TEXT    NOT NULL,
    score       DOUBLE  NOT NULL,
    threshold   DOUBLE  NOT NULL,
    passed      BOOLEAN NOT NULL,
    reason      TEXT,
    error       TEXT,
    tools_used  JSON,
    response_id INTEGER
);
"""

# Columns added after the initial schema; applied to existing databases via
# init_eval_results, mirroring the _MIGRATE_RESPONSES pattern above.
_MIGRATE_EVAL_RESULTS = [
    # Foreign key to responses.id. NULL on rows written before this migration —
    # there is no reliable way to backfill which response they scored.
    "ALTER TABLE eval_results ADD COLUMN response_id INTEGER",
]

_INSERT_EVAL_RESULT = """
INSERT INTO eval_results (
    suite, llm_name, question_id, question, test_name, metric_name,
    score, threshold, passed, reason, error, tools_used, response_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def init_eval_results(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the eval_results table and sequence if they don't already exist.

    Also applies column migrations so existing databases gain new fields.
    """
    conn.execute(_CREATE_EVAL_RESULTS_TABLE)
    for stmt in _MIGRATE_EVAL_RESULTS:
        try:
            conn.execute(stmt)
        except duckdb.CatalogException:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            col_hint = (
                stmt.split("ADD COLUMN")[-1].strip() if "ADD COLUMN" in stmt else stmt
            )
            logger.debug("Migration skipped (column may already exist): %s", col_hint)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            col_hint = (
                stmt.split("ADD COLUMN")[-1].strip() if "ADD COLUMN" in stmt else stmt
            )
            logger.warning("Migration failed for column: %s", col_hint, exc_info=True)


def insert_eval_result(
    conn: duckdb.DuckDBPyConnection, record: Dict[str, Any], suite: str
) -> None:
    """Insert one eval result record into the eval_results table."""
    conn.execute(
        _INSERT_EVAL_RESULT,
        [
            suite,
            record["llm_name"],
            int(record["question_id"]),
            record["question"],
            record["test_name"],
            record["metric_name"],
            float(record["score"]),
            float(record["threshold"]),
            bool(record["passed"]),
            record.get("reason") or None,
            record.get("error") or None,
            json.dumps(record.get("tools_used")),
            record.get("response_id"),
        ],
    )


def clear_eval_results(
    conn: duckdb.DuckDBPyConnection, suite: Optional[str] = None
) -> None:
    """Delete eval results, optionally filtered to a specific suite."""
    if suite:
        conn.execute("DELETE FROM eval_results WHERE suite = ?", [suite])
    else:
        conn.execute("DELETE FROM eval_results")


def load_eval_results(
    path: Optional[Path] = None,
    suite: Optional[str] = None,
    read_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load eval results from the database.  Returns a list of dicts loaded from
    the `eval_results` DuckDB table.

    Optionally filter to a single suite (e.g. ``"groundedness"``).

    Pass ``read_only=True`` when this may run concurrently with other readers
    of the same file (see ``load_records`` for why).
    """
    path = path or DEFAULT_DB
    if not path.exists():
        return []

    conn = get_connection(path, read_only=read_only)
    try:
        if not read_only:
            # Ensure the schema is migrated (adds new columns to existing DBs)
            init_eval_results(conn)
        if suite:
            rows = conn.execute(
                "SELECT llm_name, question_id, question, test_name, metric_name, "
                "score, threshold, passed, reason, error, tools_used, response_id "
                "FROM eval_results WHERE suite = ? ORDER BY id",
                [suite],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT llm_name, question_id, question, test_name, metric_name, "
                "score, threshold, passed, reason, error, tools_used, response_id "
                "FROM eval_results ORDER BY id"
            ).fetchall()
    finally:
        conn.close()

    results = []
    for (
        llm_name,
        question_id,
        question,
        test_name,
        metric_name,
        score,
        threshold,
        passed,
        reason,
        error,
        tools_used_json,
        response_id,
    ) in rows:
        results.append(
            {
                "llm_name": llm_name,
                "question_id": question_id,
                "question": question,
                "test_name": test_name,
                "metric_name": metric_name,
                "score": score,
                "threshold": threshold,
                "passed": passed,
                "reason": reason or "",
                "error": error or "",
                "tools_used": (
                    json.loads(tools_used_json)
                    if tools_used_json and tools_used_json != "null"
                    else None
                ),
                "response_id": response_id,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Deploy copy
# ---------------------------------------------------------------------------

_DEPLOY_CONTEXT_CHARS = 2_000  # per context item


def make_deploy_db(
    source_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Write a deploy copy of the database with ``retrieval_context`` trimmed to
    ``_DEPLOY_CONTEXT_CHARS`` characters per item.

    All other data (``actual_output``, ``tools_called``, ``eval_results``) is
    copied verbatim.  The source database is never modified.

    Args:
        source_path: Path to the source DB (default: ``data/responses.db``).
        output_path: Destination path (default: ``data/deploy.db``).

    Returns:
        The path of the written deploy database.
    """
    source_path = source_path or DEFAULT_DB
    output_path = output_path or (DATA_DIR / "deploy.db")

    if not source_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    src = get_connection(source_path)
    dst = get_connection(output_path)
    try:
        # Ensure source schema is migrated (adds new columns to existing DBs)
        init_db(src)
        init_eval_results(src)
        # Recreate schema in the destination
        init_db(dst)
        init_eval_results(dst)

        # Copy responses with trimmed retrieval_context
        rows = src.execute(
            "SELECT question_id, question, llm_name, timestamp, actual_output, "
            "retrieval_context, tools_called, research_output, is_error, error_message, "
            "research_mode, case_law_context, tool_sequence, fallback_used, "
            "summarisation_output, summarisation_used, summarisation_llm, "
            "chat_mode, provider, total_cost_usd, total_ms, reformatted, "
            "local_cache_hits, memo_hits, audit_schema_version, audit_json, research_plan "
            "FROM responses ORDER BY id"
        ).fetchall()

        trimmed_count = 0
        for row in rows:
            (
                question_id,
                question,
                llm_name,
                timestamp,
                actual_output,
                ctx_json,
                tools_json,
                research_output,
                is_error,
                error_message,
                research_mode,
                case_law_context_json,
                tool_sequence_json,
                fallback_used,
                summarisation_output_json,
                summarisation_used,
                summarisation_llm,
                chat_mode,
                provider,
                total_cost_usd,
                total_ms,
                reformatted,
                local_cache_hits,
                memo_hits,
                audit_schema_version,
                audit_json,
                research_plan_json,
            ) = row

            ctx: list = json.loads(ctx_json) if ctx_json else []
            trimmed = [item[:_DEPLOY_CONTEXT_CHARS] for item in ctx]
            if trimmed != ctx:
                trimmed_count += 1

            dst.execute(
                _INSERT_RESPONSE,
                [
                    question_id,
                    question,
                    llm_name,
                    timestamp,
                    actual_output,
                    json.dumps(trimmed),
                    tools_json,
                    research_output,
                    is_error,
                    error_message,
                    research_mode or "legislation_only",
                    (
                        case_law_context_json
                        if case_law_context_json is not None
                        else "[]"
                    ),
                    tool_sequence_json if tool_sequence_json is not None else "[]",
                    bool(fallback_used),
                    summarisation_output_json,
                    bool(summarisation_used),
                    summarisation_llm,
                    chat_mode or "research",
                    provider,
                    total_cost_usd,
                    total_ms,
                    bool(reformatted),
                    local_cache_hits or 0,
                    memo_hits or 0,
                    audit_schema_version,
                    audit_json,
                    research_plan_json,
                ],
            )

        # Copy eval_results verbatim
        eval_rows = src.execute(
            "SELECT suite, llm_name, question_id, question, test_name, metric_name, "
            "score, threshold, passed, reason, error, tools_used, response_id "
            "FROM eval_results ORDER BY id"
        ).fetchall()
        for er in eval_rows:
            dst.execute(_INSERT_EVAL_RESULT, list(er))

        dst.execute("CHECKPOINT")
    finally:
        src.close()
        dst.close()

    before = source_path.stat().st_size / 1024 / 1024
    after = output_path.stat().st_size / 1024 / 1024
    print(
        f"Deploy DB written to {output_path}\n"
        f"  Source : {before:.1f} MB\n"
        f"  Deploy : {after:.1f} MB ({trimmed_count} row(s) trimmed)"
    )
    return output_path


if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(
        description="DuckDB responses database utilities"
    )
    _parser.add_argument(
        "--clean", action="store_true", help="Delete incomplete/error responses"
    )
    _parser.add_argument(
        "--dry-run", action="store_true", help="Preview what --clean would delete"
    )
    _parser.add_argument(
        "--deploy-db",
        metavar="OUTPUT",
        help="Write a deploy copy with retrieval_context trimmed (default: data/deploy.db)",
        nargs="?",
        const="",  # sentinel: use default path
    )
    _args = _parser.parse_args()

    if _args.deploy_db is not None:
        _out = Path(_args.deploy_db) if _args.deploy_db else None
        make_deploy_db(output_path=_out)
    elif _args.clean or _args.dry_run:
        clean_incomplete_responses(dry_run=_args.dry_run)
        if not _args.dry_run:
            completeness_report()
    else:
        completeness_report()
