"""
Helpers for loading captured responses (from DuckDB) and converting them into
DeepEval LLMTestCase objects for evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.test_case import LLMTestCase, ToolCall

from .db import load_records as _db_load_records
from .db import DEFAULT_DB

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RESPONSES = DEFAULT_DB


def load_records(
    filepath: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load all records from the DuckDB responses database.

    Each record is a flat dict:
        {question_id, question, llm_name, timestamp, deep_research,
         actual_output, retrieval_context, tools_called}

    Error rows are excluded.
    """
    return _db_load_records(path=filepath)


def record_to_test_case(record: Dict[str, Any]) -> LLMTestCase:
    """
    Convert a flat record dict into a DeepEval LLMTestCase.

    Handles the serialisation format produced by
    ``gather_responses.serialize_test_case`` (Pydantic model_dump).
    """
    tools_called = []
    for tool_dict in record.get("tools_called", []):
        tools_called.append(
            ToolCall(
                name=tool_dict.get("name", ""),
                input_parameters=tool_dict.get("input_parameters")
                or tool_dict.get("inputParameters", {}),
                output=tool_dict.get("output", ""),
            )
        )

    return LLMTestCase(
        input=record.get("question", ""),
        actual_output=record.get("actual_output", ""),
        retrieval_context=record.get("retrieval_context", []),
        tools_called=tools_called,
    )



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
        from .db import group_by_question_and_llm as _db_group
        return _db_group(path=filepath)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        key = f"Q{r['question_id']}_{r['llm_name']}"
        grouped.setdefault(key, []).append(r)
    return grouped


def record_id(record: Dict[str, Any]) -> str:
    """Return a short pytest-friendly identifier for a record."""
    return f"Q{record['question_id']}_{record['llm_name']}"
