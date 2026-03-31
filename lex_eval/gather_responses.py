#!/usr/bin/env python3
"""
Gather LexChat responses for evaluation.

This script runs questions through different LLMs and captures their responses
for later evaluation using DeepEval metrics.
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
    DEFAULT_DB,
)
from lex_eval.utils.get_llms import get_llms
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


def validate_llm(llm_name: str, available_llms: List[str]) -> None:
    """
    Validate that the specified LLM is available.

    Args:
        llm_name: Name of LLM to validate
        available_llms: List of available LLM names

    Raises:
        ValueError: If LLM is not in available list
    """
    if llm_name not in available_llms:
        raise ValueError(
            f"LLM '{llm_name}' not found. Available LLMs: {', '.join(available_llms)}"
        )


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
    llm_names: List[str],
    output_file: Path,
    append: bool = False,
    max_workers: int = 10,
) -> None:
    """
    Gather responses from LLMs for all questions and save to a DuckDB database.

    Combinations are executed concurrently using a thread pool, with each thread
    maintaining its own authenticated HTTP client. Results are written
    incrementally as each combination completes, making the process
    crash-resilient.

    Args:
        questions: List of question dictionaries
        llm_names: List of LLM names to test
        output_file: Path to the DuckDB database file
        append: If True, add to existing rows; if False, clear table first
        max_workers: Maximum number of concurrent threads
    """
    total_combinations = len(questions) * len(llm_names)

    logger.info(
        f"Starting evaluation: {len(questions)} questions × {len(llm_names)} LLMs = "
        f"{total_combinations} combinations (max_workers={max_workers})"
    )

    # Prepare database
    output_file.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(output_file)
    init_db(conn)

    if append:
        logger.info(f"Appending to existing database: {output_file}")
    else:
        clear_responses(conn)
        logger.info(f"Cleared existing responses, writing fresh to: {output_file}")

    # Per-thread client management
    thread_local = threading.local()
    clients_lock = threading.Lock()
    all_clients: List = []

    def get_client():
        """Return (or lazily create) an authenticated client for the current thread."""
        if not hasattr(thread_local, "client"):
            client = get_authenticated_client()
            thread_local.client = client
            with clients_lock:
                all_clients.append(client)
        return thread_local.client

    MAX_ATTEMPTS = 3

    def process_combination(
        question_data: Dict[str, Any], llm_name: str, index: int
    ) -> Optional[Dict[str, Any]]:
        """Run a single question/LLM combination and write the result to the output file.

        Retries up to MAX_ATTEMPTS times. Returns None if a complete response
        (non-empty actual_output, no error) is never obtained — nothing is
        written to the database in that case.
        """
        question_id = question_data.get("id")
        question_text = question_data.get("question")

        logger.info(f"[{index}/{total_combinations}] Q{question_id} × {llm_name}")

        client = get_client()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                test_case = audit_capture(
                    client=client,
                    question=question_text,
                    model_name=llm_name,
                )
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
                    "actual_output": test_case_data.get("actual_output", ""),
                    "retrieval_context": test_case_data.get("retrieval_context") or [],
                    "tools_called": test_case_data.get("tools_called") or [],
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

    # Build flat list of all (question, llm, index) combinations
    combinations = [
        (q, llm, i + 1)
        for i, (q, llm) in enumerate((q, llm) for q in questions for llm in llm_names)
    ]

    completed = 0
    success_count = 0
    error_count = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_combination, q, llm, idx): (q, llm)
                for q, llm, idx in combinations
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
        logger.info(f"✓ Completed {completed} combinations → {output_file}")
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
  # Run all questions with all LLMs:
  python gather_responses.py
  
  # Run specific question:
  python gather_responses.py --question-id 1
  
  # Run with specific LLM:
  python gather_responses.py --llm "gpt-oss:120b-cloud"
  
  # Append to existing results (for incremental testing):
  python gather_responses.py --question-id 2 --append
        """,
    )

    parser.add_argument(
        "--question-id",
        type=int,
        help="Specific question ID to run (if not specified, runs all questions)",
    )

    parser.add_argument(
        "--llm",
        type=str,
        help="Specific LLM to use (if not specified, uses all available LLMs)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "responses.db",
        help="Output DuckDB database path (default: data/responses.db)",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting",
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

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load questions
        questions = load_questions(args.questions_file, args.question_id)

        # Get available LLMs
        logger.info("Fetching available LLMs...")
        available_llms = get_llms()

        # Determine which LLMs to use
        if args.llm:
            validate_llm(args.llm, available_llms)
            llm_names = [args.llm]
            logger.info(f"Using specified LLM: {args.llm}")
        else:
            llm_names = available_llms
            logger.info(f"Using all {len(llm_names)} available LLMs")

        # Gather responses
        gather_responses(
            questions=questions,
            llm_names=llm_names,
            output_file=args.output,
            append=args.append,
            max_workers=args.workers,
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
