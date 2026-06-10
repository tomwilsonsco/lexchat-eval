#!/usr/bin/env python3
"""
Gather LexChat responses for evaluation.

This script runs questions through the active LLM configured in LexChat's admin
portal and captures responses for later evaluation using DeepEval metrics.

The model used for responses is always controlled via LexChat's Admin Portal —
there is no --llm flag. To evaluate a different model, change it in the Admin
Portal and re-run.
"""

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from lex_eval.utils.audit_capture import audit_capture
from lex_eval.utils.db import (
    get_connection,
    init_db,
    clear_responses,
    insert_response,
)
from lex_eval.utils.get_llm import get_active_model
from lex_eval.utils.lexchat_client import get_authenticated_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_questions(
    questions_file: Path, question_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load questions from JSON file, optionally filtering by question_id.

    Args:
        questions_file: Path to questions.json file
        question_id: Optional question ID to filter for

    Returns:
        List of question dictionaries
    """
    logger.info(f"Loading questions from {questions_file}")

    with open(questions_file, "r") as f:
        questions = json.load(f)

    if question_id is not None:
        questions = [q for q in questions if q.get("id") == question_id]
        if not questions:
            raise ValueError(f"Question with id={question_id} not found")
        logger.info(f"Filtered to question {question_id}")
    else:
        logger.info(f"Loaded {len(questions)} questions")

    return questions


def serialize_test_case(test_case) -> Dict[str, Any]:
    """
    Serialize a DeepEval LLMTestCase to a JSON-serializable dictionary.

    Args:
        test_case: LLMTestCase object from audit_capture

    Returns:
        Dictionary representation of the test case
    """
    try:
        # Use Pydantic's model_dump method
        return test_case.model_dump(mode="json", exclude_none=True)
    except Exception as e:
        logger.warning(f"Failed to use model_dump, falling back to dict(): {e}")
        # Fallback to dict() method
        data = test_case.dict()
        # Convert any non-serializable objects
        return json.loads(json.dumps(data, default=str))


def gather_responses(
    questions: List[Dict[str, Any]],
    llm_name: str,
    output_file: Path,
    overwrite: bool = False,
    max_workers: int = 10,
    debug_events_file: Optional[Path] = None,
) -> None:
    """
    Gather responses from the active LLM for all questions and save to DuckDB.

    Runs concurrently using a thread pool, with each thread maintaining its own
    authenticated HTTP client. Results are written incrementally as each question
    completes, making the process crash-resilient.

    Args:
        questions: List of question dictionaries
        llm_name: Name of the active LLM (from LexChat's admin portal)
        output_file: Path to the DuckDB database file
        overwrite: If True, clear table first; if False, add to existing rows
        max_workers: Maximum number of concurrent threads
    """
    total_questions = len(questions)

    logger.info(
        f"Starting evaluation: {total_questions} questions × {llm_name} "
        f"(max_workers={max_workers})"
    )

    # Prepare database
    output_file.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(output_file)
    init_db(conn)

    if overwrite:
        clear_responses(conn)
        logger.info(f"Cleared existing responses, writing fresh to: {output_file}")
    else:
        logger.info(f"Appending to existing database: {output_file}")

    # Per-thread client management
    thread_local = threading.local()
    clients_lock = threading.Lock()
    all_clients: List = []

    # Debug event file writer (thread-safe)
    debug_lock = threading.Lock() if debug_events_file else None
    _debug_fh = None

    def _write_debug_event(event_dict: dict) -> None:
        """Thread-safe append of a single event dict as JSON line to the debug file."""
        nonlocal _debug_fh
        if debug_lock:
            with debug_lock:
                if _debug_fh is None and debug_events_file:
                    _debug_fh = open(debug_events_file, "w")
                if _debug_fh:
                    _debug_fh.write(json.dumps(event_dict, default=str) + "\n")
                    _debug_fh.flush()

    def get_client():
        """Return (or lazily create) an authenticated client for the current thread."""
        if not hasattr(thread_local, "client"):
            client = get_authenticated_client()
            thread_local.client = client
            with clients_lock:
                all_clients.append(client)
        return thread_local.client

    MAX_ATTEMPTS = 3

    def process_question(
        question_data: Dict[str, Any], index: int
    ) -> Optional[Dict[str, Any]]:
        """Run a single question and write the result to the output file.

        Retries up to MAX_ATTEMPTS times. Returns None if a complete response
        (non-empty actual_output, no error) is never obtained — nothing is
        written to the database in that case.
        """
        question_id = question_data.get("id")
        question_text = question_data.get("question")
        research_mode = question_data.get("research_mode", "legislation_only")

        logger.info(
            f"[{index}/{total_questions}] Q{question_id} × {llm_name} (mode={research_mode})"
        )

        client = get_client()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                capture_result = audit_capture(
                    client=client,
                    question=question_text,
                    model_name=llm_name,
                    research_mode=research_mode,
                    on_event=_write_debug_event if debug_events_file else None,
                )
                test_case = capture_result["test_case"]
                research_output = capture_result["research_output"]
                test_case_data = serialize_test_case(test_case)

                actual_output = test_case_data.get("actual_output", "")
                if not actual_output or not actual_output.strip():
                    logger.warning(
                        f"↻ Q{question_id} × {llm_name} attempt {attempt}/{MAX_ATTEMPTS}: "
                        "empty actual_output, retrying…"
                    )
                    continue
                result = {
                    "question_id": question_id,
                    "question": question_text,
                    "llm_name": llm_name,
                    "timestamp": datetime.now().isoformat(),
                    "actual_output": actual_output,
                    "retrieval_context": test_case_data.get("retrieval_context") or [],
                    "tools_called": test_case_data.get("tools_called") or [],
                    "research_output": research_output,
                    "research_mode": capture_result.get("research_mode", research_mode),
                    "case_law_context": capture_result.get("case_law_context") or [],
                    "tool_sequence": capture_result.get("tool_sequence") or [],
                    "fallback_used": capture_result.get("fallback_used", False),
                    "summarisation_output": capture_result.get("summarisation_output") or [],
                    "summarisation_used": capture_result.get("summarisation_used", False),
                }
                logger.info(
                    f"✓ Q{question_id} × {llm_name}: "
                    f"{len(test_case_data.get('tools_called', []))} tools, "
                    f"{len(test_case_data.get('retrieval_context', []))} context items"
                )
                return result
            except Exception as e:
                logger.warning(
                    f"↻ Q{question_id} × {llm_name} attempt {attempt}/{MAX_ATTEMPTS}: {e}",
                    exc_info=attempt == MAX_ATTEMPTS,
                )

        logger.error(
            f"✗ Q{question_id} × {llm_name}: no complete response after "
            f"{MAX_ATTEMPTS} attempts — skipping"
        )
        return None

    completed = 0
    success_count = 0
    error_count = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_question, q, i + 1): q
                for i, q in enumerate(questions)
            }
            for future in as_completed(futures):
                record = future.result()
                completed += 1
                if record is None:
                    error_count += 1
                else:
                    insert_response(conn, record)
                    conn.commit()
                    success_count += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Completed {completed} questions → {output_file}")
        logger.info(f"  Success: {success_count}, Errors: {error_count}")
        logger.info(f"{'='*80}")

    finally:
        conn.close()
        for client in all_clients:
            try:
                client.close()
            except Exception:
                pass
        logger.info(f"Closed {len(all_clients)} client connection(s)")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Gather LexChat responses for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all questions on the active OpenRouter model (default):
  python gather_responses.py

  # Run on the active Ollama model:
  python gather_responses.py --provider ollama

  # Run a specific question:
  python gather_responses.py --question-id 1

  # Overwrite existing results (start fresh):
  python gather_responses.py --overwrite

  # Debug: dump every raw SSE event for inspection:
  python gather_responses.py --question-id 1 --debug-events
  # -> writes lex_eval/data/debug_events.jsonl
        """,
    )

    parser.add_argument(
        "--question-id",
        type=int,
        help="Specific question ID to run (if not specified, runs all questions)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "responses.db",
        help="Output DuckDB database path (default: data/responses.db)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing responses in the output database before writing new ones",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Maximum number of concurrent threads (default: 10)",
    )

    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path(__file__).parent / "data" / "questions.json",
        help="Questions file path (default: data/questions.json)",
    )

    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter"],
        default="openrouter",
        help="Provider to use (default: openrouter). The model is set in LexChat's admin portal.",
    )

    parser.add_argument(
        "--debug-events",
        action="store_true",
        help="Dump every raw SSE event to lex_eval/data/debug_events.jsonl for inspection",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        questions = load_questions(args.questions_file, args.question_id)

        logger.info("Fetching active LLM from LexChat API...")
        llm_name = get_active_model(args.provider)
        if not llm_name:
            raise ValueError(
                f"No active {args.provider} model found in /api/models. "
                f"Set a model in LexChat's admin portal."
            )
        logger.info(
            "Using active %s model from admin portal: %s", args.provider, llm_name
        )

        debug_events_path = None
        if args.debug_events:
            debug_events_path = Path(__file__).parent / "data" / "debug_events.jsonl"
            logger.info(f"Debug events will be written to: {debug_events_path}")

        gather_responses(
            questions=questions,
            llm_name=llm_name,
            output_file=args.output,
            overwrite=args.overwrite,
            max_workers=args.workers,
            debug_events_file=debug_events_path,
        )

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())