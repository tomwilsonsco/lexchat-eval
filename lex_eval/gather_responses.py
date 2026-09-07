"""
gather_responses.py

Sends every question from questions.json through the LexChat /api/system/chat
SSE endpoint and writes the structured results to responses.db.

This is Step 2 of the eval pipeline.

A question may carry its own "chat_mode", in which case it wins over the
--chat-mode flag, so a file mixing modes can be gathered in a single run.

Usage:
    python -m lex_eval.gather_responses
    python -m lex_eval.gather_responses --overwrite
    python -m lex_eval.gather_responses --debug-events
    python -m lex_eval.gather_responses --verbose-capture
    python -m lex_eval.gather_responses --chat-mode deep_research
    python -m lex_eval.gather_responses --questions data/questions_new.json --question-id 7 8 9
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

from lex_eval.utils.audit_capture import audit_capture
from lex_eval.utils.db import get_connection, insert_response, init_db, clear_responses
from lex_eval.utils.get_llm import get_active_model, get_summarisation_model
from lex_eval.utils.lexchat_client import get_authenticated_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# LexChat sometimes returns a perfectly normal, non-empty message whose content is
# an infrastructure failure rather than an answer, e.g. "I am currently experiencing
# a connection error with the research database and cannot complete the search".
# Retrying is the right response to those, and scoring them as legal research is
# wrong, so they are treated exactly as an empty response is.
TRANSPORT_FAILURE_PHRASES = (
    "connection error",
    "error connecting",
    "unable to connect",
    "service unavailable",
    "temporarily unavailable",
    "request timed out",
    "please try again later",
)


def transport_failure_phrase(text: str) -> Optional[str]:
    """Return the first transport failure phrase found in *text*, else None."""
    lowered = (text or "").lower()
    return next((p for p in TRANSPORT_FAILURE_PHRASES if p in lowered), None)


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load questions from the JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def process_question(
    question_id: int,
    question: str,
    research_mode: str,
    model_name: str,
    summarisation_llm: str,
    max_retries: int,
    chat_mode: str = "research",
    debug_events_file: Optional[IO[str]] = None,
    verbose_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single question through audit_capture.

    Returns a dict suitable for passing to ``save_record``.
    """
    client = get_authenticated_client()
    try:
        # ------------------------------------------------------------------
        # Deep Research: obtain plan before streaming (two-phase flow)
        # ------------------------------------------------------------------
        deep_research_plan: Optional[dict] = None
        if chat_mode == "deep_research":
            plan_response = client.post(
                "/api/research/plan",
                json={
                    "messages": [{"role": "user", "content": question}],
                    "model": model_name,
                    "research_mode": research_mode,
                },
            )
            plan_response.raise_for_status()
            plan_data = plan_response.json()

            if plan_data.get("needs_clarification"):
                logger.warning(
                    "Q%d: Deep Research plan needs clarification (%s), skipping",
                    question_id,
                    plan_data.get("question", ""),
                )
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
                    "summarisation_llm": summarisation_llm,
                    "chat_mode": "deep_research",
                    "research_plan": deep_research_plan,
                    "provider": None,
                    "total_cost_usd": None,
                    "total_ms": None,
                    "max_turns_halted": None,
                    "react_turns_max": None,
                    "local_cache_hits": 0,
                    "memo_hits": 0,
                    "reformatted": False,
                    "audit_schema_version": None,
                    "audit_json": None,
                    "attempts": 0,
                    "needs_clarification": True,
                    "clarification_question": plan_data.get("question", ""),
                }

            deep_research_plan = plan_data.get("plan")

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
                chat_mode=chat_mode,
                deep_research_plan=deep_research_plan,
                on_event=on_event,
                verbose_log_path=attempt_log_path,
            )

            actual_output = capture_result.get("actual_output", "")
            capture_is_error = capture_result.get("is_error", False)
            capture_error_message = capture_result.get("error_message", "")
            failure_phrase = transport_failure_phrase(actual_output)

            if actual_output and not failure_phrase:
                result = {
                    "question_id": question_id,
                    "question": question,
                    "llm_name": model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "actual_output": actual_output,
                    "retrieval_context": capture_result.get("retrieval_context", []),
                    "tools_called": capture_result.get("tools_called", []),
                    "research_output": capture_result.get("research_output", ""),
                    "research_mode": capture_result.get("research_mode", research_mode),
                    "case_law_context": capture_result.get("case_law_context", []),
                    "tool_sequence": capture_result.get("tool_sequence", []),
                    "fallback_used": capture_result.get("fallback_used", False),
                    "summarisation_output": capture_result.get(
                        "summarisation_output", []
                    ),
                    "summarisation_used": capture_result.get(
                        "summarisation_used", False
                    ),
                    "summarisation_llm": summarisation_llm,
                    "is_error": capture_is_error,
                    "error_message": capture_error_message,
                    "chat_mode": capture_result.get("chat_mode", chat_mode),
                    "research_plan": deep_research_plan,
                    "provider": capture_result.get("provider"),
                    "total_cost_usd": capture_result.get("total_cost_usd"),
                    "total_ms": capture_result.get("total_ms"),
                    "max_turns_halted": capture_result.get("max_turns_halted"),
                    "react_turns_max": capture_result.get("react_turns_max"),
                    "local_cache_hits": capture_result.get("local_cache_hits", 0),
                    "memo_hits": capture_result.get("memo_hits", 0),
                    "reformatted": capture_result.get("reformatted", False),
                    "audit_schema_version": capture_result.get("audit_schema_version"),
                    "audit_json": capture_result.get("audit_json"),
                    "attempts": attempt,
                }
                # If the capture layer observed an error (e.g. the audit event
                # carried an error), add the "error" key so insert_response
                # treats this as an error row consistently.
                if capture_is_error and capture_error_message:
                    result["error"] = capture_error_message
                return result
            else:
                if failure_phrase:
                    logger.warning(
                        "Q%d attempt %d reported a transport failure (%r), retrying",
                        question_id,
                        attempt,
                        failure_phrase,
                    )
                else:
                    logger.warning(
                        "Empty actual_output for Q%d on attempt %d",
                        question_id,
                        attempt,
                    )
                if attempt == max_retries:
                    # Preserve the specific error_message from the capture
                    # layer if available, falling back to the generic message.
                    # A transport failure names itself, since insert_response
                    # blanks actual_output on an error row.
                    if failure_phrase:
                        error_msg = (
                            f"Transport failure after {max_retries} attempts "
                            f"({failure_phrase!r}): {actual_output[:300]}"
                        )
                    else:
                        error_msg = (
                            capture_error_message
                            if capture_error_message
                            else "Empty actual_output after retries"
                        )
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
                        "summarisation_llm": summarisation_llm,
                        "chat_mode": chat_mode,
                        "research_plan": deep_research_plan,
                        "provider": None,
                        "total_cost_usd": None,
                        "total_ms": None,
                        "max_turns_halted": None,
                        "react_turns_max": None,
                        "local_cache_hits": 0,
                        "memo_hits": 0,
                        "reformatted": False,
                        "audit_schema_version": None,
                        "audit_json": None,
                        "attempts": attempt,
                        "error": error_msg,
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
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds a response stream may go silent before it is abandoned "
        "(default: 300). Deep Research can stream for longer than that and "
        "still be working, so raise it for deep_research runs.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per question when the response is empty or reports a "
        "transport failure (default: 3)",
    )
    parser.add_argument(
        "--question-id",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Run only the question(s) with these IDs, e.g. --question-id 1 2 4 "
        "(default: all questions)",
    )
    parser.add_argument(
        "--chat-mode",
        default="research",
        choices=["research", "conversational", "deep_research"],
        help="Chat mode to pass to /api/system/chat, for questions that do not "
        "set their own chat_mode (default: research)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the questions JSON file (default: lex_eval/data/questions.json)",
    )
    args = parser.parse_args()

    # Read by get_authenticated_client(), which takes no arguments and is called
    # once per worker thread, so the value is passed through the environment.
    if args.timeout is not None:
        os.environ["LEXCHAT_TIMEOUT"] = str(args.timeout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    base = Path(__file__).parent
    questions_path = args.questions or (base / "data" / "questions.json")
    db_path = base / "data" / "responses.db"
    debug_events_path = base / "data" / "debug_events.jsonl"
    verbose_logs_dir = base / "data" / "verbose_logs"

    if not questions_path.exists():
        logger.error("Questions file not found at %s", questions_path)
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
    if model_name is None:
        logger.error(
            "No active model found in LexChat. Set one in the admin portal before "
            "running the eval."
        )
        sys.exit(1)
    logger.info("Active LLM (manager/worker): %s", model_name)

    summ_model_name, _ = get_summarisation_model()
    if summ_model_name is None:
        summ_model_name = model_name
    if summ_model_name == model_name:
        logger.info("Summarisation LLM: %s (same as active model)", summ_model_name)
    else:
        logger.info("Summarisation LLM: %s", summ_model_name)

    # ------------------------------------------------------------------
    # Load questions
    # ------------------------------------------------------------------
    questions = load_questions(questions_path)
    logger.info("Loaded %d questions", len(questions))

    if args.question_id is not None:
        wanted = set(args.question_id)
        questions = [q for q in questions if q["id"] in wanted]
        found = {q["id"] for q in questions}
        missing = wanted - found
        if missing:
            logger.error("No question found with id(s)=%s", sorted(missing))
            sys.exit(1)
        logger.info("Filtered to question ID(s) %s", sorted(found))

    # No per-question skip check exists, every question is (re-)gathered on each
    # run. --overwrite clears prior responses first; otherwise runs are appended.
    pending = questions

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

                futures[
                    executor.submit(
                        process_question,
                        q["id"],
                        q["question"],
                        q.get("research_mode", "legislation_only"),
                        model_name,
                        summ_model_name,
                        args.retries,
                        q.get("chat_mode", args.chat_mode),
                        debug_fh,
                        verbose_log_path,
                    )
                ] = q

            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    insert_response(db_conn, result)
                    # Error records are signalled by an "error" key (and may not
                    # set is_error), so check both to avoid logging failures as OK.
                    # A clarification request is a valid outcome, not an error.
                    is_error = result.get("is_error") or "error" in result
                    if result.get("needs_clarification"):
                        status = "CLARIFICATION"
                    elif is_error:
                        status = "ERROR"
                    else:
                        status = "OK"
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
                            "summarisation_llm": summ_model_name,
                            "chat_mode": q.get("chat_mode", args.chat_mode),
                            "research_plan": None,
                            "provider": None,
                            "total_cost_usd": None,
                            "total_ms": None,
                            "max_turns_halted": None,
                            "react_turns_max": None,
                            "local_cache_hits": 0,
                            "memo_hits": 0,
                            "reformatted": False,
                            "audit_schema_version": None,
                            "audit_json": None,
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
