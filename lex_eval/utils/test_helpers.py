"""
Helpers for loading captured JSONL responses and converting them into
DeepEval LLMTestCase objects for evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.test_case import LLMTestCase, ToolCall

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RESPONSES = DATA_DIR / "responses.jsonl"


def load_records(
    filepath: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load all records from a responses JSONL file.

    Each record has the shape produced by gather_responses.py:
        {question_id, question, llm_name, timestamp, deep_research, test_case}

    Silently skips lines that contain an ``error`` key (failed captures).
    """
    filepath = filepath or DEFAULT_RESPONSES
    records: List[Dict[str, Any]] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "error" in record:
                continue
            records.append(record)
    return records


def record_to_test_case(record: Dict[str, Any]) -> LLMTestCase:
    """
    Convert a single JSONL record into a DeepEval ``LLMTestCase``.

    Handles the serialisation format produced by
    ``gather_responses.serialize_test_case`` (Pydantic model_dump).
    """
    tc = record["test_case"]

    tools_called = []
    for tool_dict in tc.get("tools_called", []):
        tools_called.append(
            ToolCall(
                name=tool_dict.get("name", ""),
                input_parameters=tool_dict.get("input_parameters")
                or tool_dict.get("inputParameters", {}),
                output=tool_dict.get("output", ""),
            )
        )

    return LLMTestCase(
        input=tc.get("input", ""),
        actual_output=tc.get("actual_output", ""),
        retrieval_context=tc.get("retrieval_context", []),
        tools_called=tools_called,
    )


def load_test_cases(
    filepath: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load records and return them as a list suitable for
    ``@pytest.mark.parametrize``.

    Each element carries the full record *plus* a human-readable ``id``
    string for pytest output.
    """
    return load_records(filepath)


def group_by_question(
    records: Optional[List[Dict[str, Any]]] = None,
    filepath: Optional[Path] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group records by ``question_id``.

    Useful for consistency testing across LLMs or across repeated runs
    of the same question.
    """
    if records is None:
        records = load_records(filepath)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        qid = r["question_id"]
        grouped.setdefault(qid, []).append(r)
    return grouped


def group_by_question_and_llm(
    records: Optional[List[Dict[str, Any]]] = None,
    filepath: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group records by (question_id, llm_name) tuple key.

    Useful for testing repeatability of the same LLM on the same question
    when ``gather_responses.py`` is run with ``--append``.
    """
    if records is None:
        records = load_records(filepath)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        key = f"Q{r['question_id']}_{r['llm_name']}"
        grouped.setdefault(key, []).append(r)
    return grouped


def record_id(record: Dict[str, Any]) -> str:
    """Return a short pytest-friendly identifier for a record."""
    return f"Q{record['question_id']}_{record['llm_name']}"
