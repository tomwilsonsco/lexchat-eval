"""
Unit tests for ``audit_capture`` — the SSE stream parser that derives the
result dict from the structured ``audit`` event emitted by LexChat
(commit da3070d+).

These tests use a mock client that simulates the SSE stream.  They do not
require a running LexChat server or a populated database.

Test cases (per implementation plan §4.2):
  - Standard legislation_only trace
  - Deep Research trace (multiple delegations)
  - Failed run (audit carries an error)
  - Missing ``audit`` event → RuntimeError
  - Unknown ``schema_version`` → RuntimeError
"""

import json
from contextlib import contextmanager

import pytest

from lex_eval.utils.audit_capture import audit_capture


# ---------------------------------------------------------------------------
# Mock SSE client
# ---------------------------------------------------------------------------


class _MockResponse:
    """Simulates an ``httpx`` streaming response."""

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for line in self._lines:
            yield line


class _MockClient:
    """Simulates an ``httpx.Client`` with a ``.stream()`` context manager."""

    def __init__(self, lines):
        self._lines = lines

    @contextmanager
    def stream(self, method, url, **kwargs):
        yield _MockResponse(self._lines)

    def close(self):
        pass


def _sse(data_dict):
    """Build an SSE ``data: ...`` line from a dict."""
    return f"data: {json.dumps(data_dict)}"


def _sse_lines(*events):
    """Build a list of SSE lines from event dicts, ending with ``[DONE]``."""
    lines = [_sse(evt) for evt in events]
    lines.append("data: [DONE]")
    return lines


# ---------------------------------------------------------------------------
# Fixture audit dicts (per implementation plan §4.1)
# ---------------------------------------------------------------------------

AUDIT_LEGISLATION = {
    "type": "audit",
    "schema_version": 1,
    "request_id": "aabbccdd",
    "chat_mode": "research",
    "research_mode": "legislation_only",
    "provider": "openrouter",
    "model": "openai/gpt-4o",
    "answer": "The Housing Act 1985 requires landlords to...",
    "delegations": [
        {
            "id": "d1d1d1d1",
            "kind": "delegation",
            "step": None,
            "title": None,
            "brief": "Research the right to repair under the Housing Act 1985",
            "report": (
                "**Summary Answer (BLUF):**\n"
                "Landlords must repair.\n"
                "**Detailed Analysis:**\n"
                "Section 11 applies."
            ),
            "reformatted": False,
            "error": None,
            "started_at": 0.5,
            "duration_s": 12.0,
            "tools": [
                {
                    "id": "t1",
                    "name": "search_legislation",
                    "args": {"query": "Housing Act 1985 repair"},
                    "raw_result": "...",
                    "final_result": "...",
                    "summarised": False,
                    "local_cache_hit": False,
                    "memo_hit": False,
                    "api_calls": [
                        {
                            "url": "https://lex.../legislation/search",
                            "method": "POST",
                            "request": {},
                            "status": 200,
                            "response": {
                                "results": [
                                    {"title": "Housing Act 1985", "year": 1985}
                                ]
                            },
                        }
                    ],
                },
                {
                    "id": "t2",
                    "name": "search_legislation_sections",
                    "args": {
                        "legislation_id": "ukpga/1985/68",
                        "query": "repair",
                    },
                    "raw_result": "...",
                    "final_result": "Section 11: ...",
                    "summarised": False,
                    "local_cache_hit": False,
                    "memo_hit": False,
                    "api_calls": [
                        {
                            "url": "https://lex.../legislation/section/search",
                            "method": "POST",
                            "request": {},
                            "status": 200,
                            "response": [
                                {"title": "Section 11", "content": "The landlord..."}
                            ],
                        }
                    ],
                },
            ],
        }
    ],
    "timings": {"total_ms": 14200, "llm_calls": 3, "total_cost_usd": 0.05},
    "error": None,
}

AUDIT_DEEP_RESEARCH = {
    "type": "audit",
    "schema_version": 1,
    "request_id": "eeff0011",
    "chat_mode": "deep_research",
    "research_mode": "legislation_only",
    "provider": "openrouter",
    "model": "openai/gpt-4o",
    "answer": "Based on multi-step research...",
    "delegations": [
        {
            "id": "d1",
            "kind": "deep_research_step",
            "step": 1,
            "title": "Statutory framework",
            "brief": "Identify the key statutes",
            "report": "**Summary Answer (BLUF):**\nStep 1 report.",
            "reformatted": False,
            "error": None,
            "started_at": 0.5,
            "duration_s": 10.0,
            "tools": [
                {
                    "id": "t1",
                    "name": "search_legislation",
                    "args": {"query": "Housing Act"},
                    "raw_result": "...",
                    "final_result": "...",
                    "summarised": False,
                    "local_cache_hit": False,
                    "memo_hit": False,
                    "api_calls": [],
                },
            ],
        },
        {
            "id": "d2",
            "kind": "deep_research_step",
            "step": 2,
            "title": "Case law analysis",
            "brief": "Identify key cases",
            "report": "**Summary Answer (BLUF):**\nStep 2 report.",
            "reformatted": True,
            "error": None,
            "started_at": 11.0,
            "duration_s": 8.0,
            "tools": [
                {
                    "id": "t2",
                    "name": "search_legislation_sections",
                    "args": {
                        "legislation_id": "ukpga/1985/68",
                        "query": "repair",
                    },
                    "raw_result": "...",
                    "final_result": "Section 11 text",
                    "summarised": True,
                    "local_cache_hit": True,
                    "memo_hit": False,
                    "api_calls": [
                        {
                            "url": "https://lex.../legislation/section/search",
                            "method": "POST",
                            "request": {},
                            "status": 200,
                            "response": [
                                {"title": "Section 11", "content": "The landlord..."}
                            ],
                        }
                    ],
                },
            ],
        },
    ],
    "timings": {"total_ms": 20000, "llm_calls": 5, "total_cost_usd": 0.12},
    "error": None,
}

AUDIT_FAILED = {
    "type": "audit",
    "schema_version": 1,
    "request_id": "failed123",
    "chat_mode": "research",
    "research_mode": "legislation_only",
    "provider": "openrouter",
    "model": "openai/gpt-4o",
    "answer": "",
    "delegations": [
        {
            "id": "d1",
            "kind": "delegation",
            "step": None,
            "title": None,
            "brief": "Research something",
            "report": "",
            "reformatted": False,
            "error": "Worker agent timed out",
            "started_at": 0.5,
            "duration_s": 30.0,
            "tools": [],
        }
    ],
    "timings": {"total_ms": 30000, "llm_calls": 1, "total_cost_usd": 0.01},
    "error": "Worker agent timed out",
}

AUDIT_WRONG_SCHEMA = {
    "type": "audit",
    "schema_version": 2,
    "request_id": "wrong456",
    "chat_mode": "research",
    "research_mode": "legislation_only",
    "provider": "openrouter",
    "model": "openai/gpt-4o",
    "answer": "Some answer",
    "delegations": [],
    "timings": {"total_ms": 1000, "llm_calls": 1, "total_cost_usd": 0.01},
    "error": None,
}


# ---------------------------------------------------------------------------
# Tests — standard legislation_only
# ---------------------------------------------------------------------------


class TestStandardLegislation:
    """Test the standard legislation_only audit event."""

    def test_actual_output_from_audit_answer(self):
        """actual_output should prefer audit['answer'] over token stream."""
        lines = _sse_lines(
            {"type": "token", "content": "partial..."},
            AUDIT_LEGISLATION,
        )
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["actual_output"] == AUDIT_LEGISLATION["answer"]

    def test_research_output_from_delegation_report(self):
        """research_output should be the delegation's report."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        expected = AUDIT_LEGISLATION["delegations"][0]["report"]
        assert result["research_output"] == expected

    def test_tool_sequence_includes_delegate_and_worker_tools(self):
        """tool_sequence should contain delegate_research and Worker: prefixed tools."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert "delegate_research" in result["tool_sequence"]
        assert "Worker: search_legislation" in result["tool_sequence"]
        assert "Worker: search_legislation_sections" in result["tool_sequence"]

    def test_tools_called_structure(self):
        """tools_called should have delegate_research and Worker: entries."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        names = [t["name"] for t in result["tools_called"]]
        assert "delegate_research" in names
        assert "Worker: search_legislation" in names
        assert "Worker: search_legislation_sections" in names

    def test_retrieval_context_from_sections(self):
        """retrieval_context should include section text from search_legislation_sections."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert len(result["retrieval_context"]) > 0
        ctx_str = " ".join(result["retrieval_context"])
        assert "Section 11" in ctx_str

    def test_retrieval_context_from_legislation_search(self):
        """retrieval_context should include legislation search metadata."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        ctx_str = " ".join(result["retrieval_context"])
        assert "Housing Act 1985" in ctx_str

    def test_fallback_used_false(self):
        """fallback_used should be False when get_legislation_text is not called."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["fallback_used"] is False

    def test_new_fields_populated(self):
        """New fields from the audit event should be populated."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["chat_mode"] == "research"
        assert result["provider"] == "openrouter"
        assert result["total_cost_usd"] == 0.05
        assert result["total_ms"] == 14200
        assert result["audit_schema_version"] == 1
        assert result["audit_json"] is not None

    def test_is_error_false(self):
        """is_error should be False for a successful run."""
        lines = _sse_lines(AUDIT_LEGISLATION)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["is_error"] is False
        assert result["error_message"] == ""


# ---------------------------------------------------------------------------
# Tests — deep research
# ---------------------------------------------------------------------------


class TestDeepResearch:
    """Test the deep research audit event with multiple delegations."""

    def test_research_output_concatenates_all_reports(self):
        """research_output should join all delegation reports."""
        lines = _sse_lines(AUDIT_DEEP_RESEARCH)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert "Step 1 report" in result["research_output"]
        assert "Step 2 report" in result["research_output"]
        assert "\n\n" in result["research_output"]

    def test_tool_sequence_has_delegate_per_delegation(self):
        """tool_sequence should contain delegate_research repeated per delegation."""
        lines = _sse_lines(AUDIT_DEEP_RESEARCH)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["tool_sequence"].count("delegate_research") == 2

    def test_summarisation_detected(self):
        """summarisation_used should be True when a tool has summarised=True."""
        lines = _sse_lines(AUDIT_DEEP_RESEARCH)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["summarisation_used"] is True
        assert result["summarisation_output"] is not None
        assert len(result["summarisation_output"]) > 0

    def test_local_cache_hits_counted(self):
        """local_cache_hits should count tools with local_cache_hit=True."""
        lines = _sse_lines(AUDIT_DEEP_RESEARCH)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["local_cache_hits"] == 1

    def test_reformatted_detected(self):
        """reformatted should be True when any delegation has reformatted=True."""
        lines = _sse_lines(AUDIT_DEEP_RESEARCH)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["reformatted"] is True

    def test_chat_mode_deep_research(self):
        """chat_mode should be deep_research."""
        lines = _sse_lines(AUDIT_DEEP_RESEARCH)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["chat_mode"] == "deep_research"


# ---------------------------------------------------------------------------
# Tests — failed run
# ---------------------------------------------------------------------------


class TestFailedRun:
    """Test a failed run where the audit event carries an error."""

    def test_is_error_true(self):
        """is_error should be True when audit['error'] is set."""
        lines = _sse_lines(AUDIT_FAILED)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["is_error"] is True

    def test_error_message_from_audit(self):
        """error_message should be populated from audit['error']."""
        lines = _sse_lines(AUDIT_FAILED)
        result = audit_capture(_MockClient(lines), "test question", "test-model")
        assert result["error_message"] == "Worker agent timed out"


# ---------------------------------------------------------------------------
# Tests — missing audit event
# ---------------------------------------------------------------------------


class TestMissingAuditEvent:
    """Test that a stream without an audit event raises RuntimeError."""

    def test_raises_runtime_error(self):
        """audit_capture should raise RuntimeError naming commit da3070d."""
        lines = _sse_lines(
            {"type": "token", "content": "some text"},
            {"type": "result", "message": {"content": "final answer"}},
        )
        with pytest.raises(RuntimeError, match="da3070d"):
            audit_capture(_MockClient(lines), "test question", "test-model")


# ---------------------------------------------------------------------------
# Tests — unknown schema version
# ---------------------------------------------------------------------------


class TestUnknownSchemaVersion:
    """Test that an audit event with wrong schema_version raises RuntimeError."""

    def test_raises_runtime_error(self):
        """audit_capture should raise RuntimeError naming the version."""
        lines = _sse_lines(AUDIT_WRONG_SCHEMA)
        with pytest.raises(RuntimeError, match="schema_version"):
            audit_capture(_MockClient(lines), "test question", "test-model")