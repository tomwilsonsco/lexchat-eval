#!/usr/bin/env python3
"""
Gather LexChat responses for evaluation.

This script runs questions through different LLMs and captures their responses
for later evaluation using DeepEval metrics.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from utils.audit_capture import audit_capture
from utils.get_llms import get_llms
from utils.lexchat_client import get_authenticated_client

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
    deep_research: bool = False,
    append: bool = False,
) -> None:
    """
    Gather responses from LLMs for all questions and save to JSONL file.

    Results are written incrementally as each question/LLM combination completes,
    making the process crash-resilient and memory-efficient.

    Args:
        questions: List of question dictionaries
        llm_names: List of LLM names to test
        output_file: Path to save results (JSONL format)
        deep_research: Whether to enable deep research mode
        append: If True, append to existing file; if False, overwrite
    """
    total_combinations = len(questions) * len(llm_names)
    completed = 0
    success_count = 0
    error_count = 0

    logger.info(
        f"Starting evaluation: {len(questions)} questions × {len(llm_names)} LLMs = {total_combinations} combinations"
    )

    # Prepare output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if append else "w"

    if append and output_file.exists():
        logger.info(f"Appending to existing file: {output_file}")
    else:
        logger.info(f"Writing to new file: {output_file}")

    # Create client once and reuse
    client = get_authenticated_client()

    try:
        # Open file once for all writes (append or overwrite)
        with open(output_file, file_mode) as f:
            for question_data in questions:
                question_id = question_data.get("id")
                question_text = question_data.get("question")

                logger.info(f"\n{'='*80}")
                logger.info(f"Question {question_id}: {question_text}")
                logger.info(f"{'='*80}")

                for llm_name in llm_names:
                    completed += 1
                    logger.info(
                        f"\n[{completed}/{total_combinations}] Testing with {llm_name}"
                    )

                    try:
                        # Capture the response using audit_capture
                        test_case = audit_capture(
                            client=client,
                            question=question_text,
                            model_name=llm_name,
                            deep_research=deep_research,
                        )

                        # Serialize the test case
                        test_case_data = serialize_test_case(test_case)

                        # Create result entry
                        result = {
                            "question_id": question_id,
                            "question": question_text,
                            "llm_name": llm_name,
                            "timestamp": datetime.now().isoformat(),
                            "deep_research": deep_research,
                            "test_case": test_case_data,
                        }

                        # Write result immediately to JSONL file
                        f.write(json.dumps(result) + "\n")
                        f.flush()  # Ensure it's written to disk immediately

                        success_count += 1
                        logger.info(
                            f"✓ Captured response ({len(test_case_data.get('tools_called', []))} tools, "
                            f"{len(test_case_data.get('retrieval_context', []))} context items)"
                        )

                    except Exception as e:
                        logger.error(
                            f"✗ Failed to capture response: {e}", exc_info=True
                        )
                        # Store error result
                        error_result = {
                            "question_id": question_id,
                            "question": question_text,
                            "llm_name": llm_name,
                            "timestamp": datetime.now().isoformat(),
                            "deep_research": deep_research,
                            "error": str(e),
                        }
                        f.write(json.dumps(error_result) + "\n")
                        f.flush()
                        error_count += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Successfully saved {completed} results to {output_file}")
        logger.info(f"  Success: {success_count}, Errors: {error_count}")
        logger.info(f"{'='*80}")

    finally:
        client.close()
        logger.info("Closed client connection")


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
        "--deep-research", action="store_true", help="Enable deep research mode"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "responses.jsonl",
        help="Output file path in JSONL format (default: data/responses.jsonl)",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting",
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
            deep_research=args.deep_research,
            append=args.append,
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
    sys.exit(main())
