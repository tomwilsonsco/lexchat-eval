"""
gather_responses.py

Sends every question from questions.json through the LexChat /api/system/chat
SSE endpoint and writes the structured results to responses.db.

This is Step 2 of the eval pipeline.

Usage:
    python -m lex_eval.gather_responses
    python -m lex_eval.gather_responses --overwrite
    python -m lex_eval.gather_responses --debug-events
    python -m lex_eval.gather_responses --verbose-capture
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lex_eval.utils.audit_capture import audit_capture
from lex_eval.utils.db import get_connection, insert_response, init_db, clear_responses
from lex_eval.utils.get_llm import get_active_model
from lex_eval.utils.lexchat_client import get_authenticated_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load questions from the JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def process_question(
    question_id: int,
    question: str,
    research_mode: str,
    model_name: str,
    max_retries: int,
    debug_events_file: Optional[Path] = None,
    verbose_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single question through audit_capture.

    Returns a dict suitable for passing to ``save_record``.
    """
    client = get_authenticated_client()
    try:
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                logger.info("Retry %d/%d for Q%d", attempt, max_retries, question_id)

            # Set up debug event callback if requested
            on_event = None
            if debug_events_file:
                def _make_callback(fh):
                    def _callback(data):
                        fh.write(json.dumps(data, default=str) + "\n")
                    return _callback
                on_event = _make_callback(debug_events_file)

            # Build per-attempt verbose log path (retries get separate files)
            attempt_log_path = None
            if verbose_log_path:
                if attempt == 1:
                    attempt_log_path = verbose_log_path
                else:
                    # Insert _retryN before the extension
                    stem = verbose_log_path.stem
                    attempt_log_path = verbose_log_path.with_stem(
                        f"{stem}_retry{attempt - 1}"
                    )

            capture_result = audit_capture(
                client,
                question,
                model_name,
                research_mode=research_mode,
                on_event=on_event,
                verbose_log_path=attempt_log_path,
            )

            actual_output = capture_result.get("actual_output", "")
            if actual_output:
                return {
                    "question_id": question_id,
                    "question": question,
                    "llm_name": model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "actual_output": actual_output,
                    "retrieval_context": capture_result.get("retrieval_context", []),
                    "tools_called": capture_result.get("tools_called", []),
                    "research_output": capture_result.get("research_output", ""),
                    "research_mode": research_mode,
                    "case_law_context": capture_result.get("case_law_context", []),
                    "tool_sequence": capture_result.get("tool_sequence", []),
                    "fallback_used": capture_result.get("fallback_used", False),
                    "summarisation_output": capture_result.get("summarisation_output", []),
                    "summarisation_used": capture_result.get("summarisation_used", False),
                    "is_error": False,
                    "error_message": "",
                }
            else:
                logger.warning(
                    "Empty actual_output for Q%d on attempt %d", question_id, attempt
                )
                if attempt == max_retries:
                    return {
                        "question_id": question_id,
                        "question": question,
                        "llm_name": model_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "actual_output": "",
                        "retrieval_context": [],
                        "tools_called": [],
                        "research_output": "",
                        "research_mode": research_mode,
                        "case_law_context": [],
                        "tool_sequence": [],
                        "fallback_used": False,
                        "summarisation_output": [],
                        "summarisation_used": False,
                        "error": "Empty actual_output after retries",
                    }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather evaluation responses from LexChat"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing responses before gathering (default: append)",
    )
    parser.add_argument(
        "--debug-events",
        action="store_true",
        help="Write raw SSE events to lex_eval/data/debug_events.jsonl",
    )
    parser.add_argument(
        "--verbose-capture",
        action="store_true",
        help="Write per-question verbose audit logs to lex_eval/data/verbose_logs/",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of concurrent threads (default: 10)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per question if actual_output is empty (default: 3)",
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=None,
        help="Run only the question with this ID (default: all questions)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    base = Path(__file__).parent
    questions_path = base / "data" / "questions.json"
    db_path = base / "data" / "responses.db"
    debug_events_path = base / "data" / "debug_events.jsonl"
    verbose_logs_dir = base / "data" / "verbose_logs"

    if not questions_path.exists():
        logger.error("questions.json not found at %s", questions_path)
        sys.exit(1)

    # ------------------------------------------------------------------
    # init DB
    # ------------------------------------------------------------------
    db_conn = get_connection(db_path)
    init_db(db_conn)

    if args.overwrite:
        clear_responses(db_conn)
        logger.info("Cleared existing responses (--overwrite)")

    # ------------------------------------------------------------------
    # Get active LLM from LexChat API
    # ------------------------------------------------------------------
    model_name, _ = get_active_model()
    logger.info("Active LLM: %s", model_name)

    # ------------------------------------------------------------------
    # Load questions
    # ------------------------------------------------------------------
    questions = load_questions(questions_path)
    logger.info("Loaded %d questions", len(questions))

    if args.question_id is not None:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            logger.error("No question found with id=%d", args.question_id)
            sys.exit(1)
        logger.info("Filtered to question ID %d", args.question_id)

    # Filter out already-gathered questions (unless --overwrite)
    pending = []
    # Simple skip logic - just add all questions since should_skip is not available
    for q in questions:
        if not args.overwrite:
            logger.info("Processing Q%d (skipping existing check)", q["id"])
        pending.append(q)

    if not pending:
        logger.info("All questions already gathered. Use --overwrite to re-gather.")
        return

    logger.info("Gathering responses for %d pending questions", len(pending))

    # ------------------------------------------------------------------
    # Open debug events file if requested
    # ------------------------------------------------------------------
    debug_fh = None
    if args.debug_events:
        debug_fh = open(str(debug_events_path), "a", encoding="utf-8")
        logger.info("Writing debug events to %s", debug_events_path)

    # ------------------------------------------------------------------
    # Create verbose logs directory if requested
    # ------------------------------------------------------------------
    if args.verbose_capture:
        verbose_logs_dir.mkdir(exist_ok=True)
        logger.info("Writing verbose capture logs to %s", verbose_logs_dir)

    try:
        # ------------------------------------------------------------------
        # Gather responses concurrently
        # ------------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {}
            for q in pending:
                # Build verbose log path for this question (per-question, not per-attempt)
                verbose_log_path = None
                if args.verbose_capture:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    verbose_log_path = verbose_logs_dir / f"Q{q['id']}_{ts}.log"

                futures[executor.submit(
                    process_question,
                    q["id"],
                    q["question"],
                    q.get("research_mode", "legislation_only"),
                    model_name,
                    args.retries,
                    debug_fh,
                    verbose_log_path,
                )] = q

            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    insert_response(db_conn, result)
                    status = "OK" if not result.get("is_error") else "ERROR"
                    logger.info(
                        "Q%d (%s): %s [actual_output=%d chars]",
                        q["id"],
                        q.get("research_mode", "legislation_only"),
                        status,
                        len(result.get("actual_output", "")),
                    )
                except Exception as exc:
                    logger.error("Q%d failed: %s", q["id"], exc)
                    insert_response(
                        db_conn,
                        {
                            "question_id": q["id"],
                            "question": q["question"],
                            "llm_name": model_name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "actual_output": "",
                            "retrieval_context": [],
                            "tools_called": [],
                            "research_output": "",
                            "research_mode": q.get("research_mode", "legislation_only"),
                            "case_law_context": [],
                            "tool_sequence": [],
                            "fallback_used": False,
                            "summarisation_output": [],
                            "summarisation_used": False,
                            "error": str(exc),
                        },
                    )
    finally:
        if debug_fh:
            debug_fh.close()
        db_conn.close()

    logger.info("Done gathering responses.")


if __name__ == "__main__":
    main()