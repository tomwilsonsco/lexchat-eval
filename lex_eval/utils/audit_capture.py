"""
audit_capture.py

Runs a single question through the LexChat /api/system/chat SSE endpoint
and returns a structured result dict.

This is the core capture layer — all the stream parsing logic lives here.
Produces a dict with these keys:

    actual_output, retrieval_context, tools_called, research_output,
    research_mode, case_law_context, tool_sequence, fallback_used,
    summarisation_output, summarisation_used, is_error, error_message
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERBOSE_TRUNCATE_CHARS = 500


def _trunc(s: str, n: int = VERBOSE_TRUNCATE_CHARS) -> str:
    """Truncate a string for display in verbose logs."""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"…[truncated {len(s) - n} chars]"


def _vlog(f, msg: str) -> None:
    """Write a line to the verbose log file if *f* is not None."""
    if f is not None:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_capture(
    client,
    question: str,
    model_name: str,
    research_mode: str = "legislation_only",
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stream the SSE endpoint for *question* and return a structured result dict.

    Parameters
    ----------
    client:
        Authenticated ``httpx.Client`` from :func:`get_authenticated_client`.
    question:
        The natural-language legal question to evaluate.
    model_name:
        The LLM model name as returned by ``/api/models``.
    research_mode:
        ``legislation_only``, ``case_law_only``, or ``legislation_and_case_law``.
    on_event:
        Optional callback invoked for every raw SSE ``data`` dict *after*
        the event has been processed.  Used by ``--debug-events``.
    verbose_log_path:
        Optional path to write a verbose human-readable audit log for this
        capture call.  When ``None`` (default), no verbose log is written.

    Returns
    -------
    dict
        Keys: ``actual_output``, ``retrieval_context``, ``tools_called``,
        ``research_output``, ``research_mode``, ``case_law_context``,
        ``tool_sequence``, ``fallback_used``, ``summarisation_output``,
        ``summarisation_used``, ``is_error``, ``error_message``.
    """

    # --- Verbose log file setup ------------------------------------------------
    _vf = open(verbose_log_path, "w", encoding="utf-8") if verbose_log_path else None
    _started_ts = datetime.now(timezone.utc)
    _token_count = 0  # rolling count of token events (not written per-event)

    _vlog(_vf, "=== AUDIT CAPTURE ===")
    _vlog(_vf, f'question:      "{question}"')
    _vlog(_vf, f"model:         {model_name}")
    _vlog(_vf, f"mode:          {research_mode}")
    _vlog(_vf, f"started:       {_started_ts.isoformat()}")
    _vlog(_vf, f"log_path:      {verbose_log_path}")
    _vlog(_vf, "=====================")
    _vlog(_vf, "")

    # ------------------------------------------------------------------
    # Per-call mutable state
    # ------------------------------------------------------------------
    chat_payload = {
        "messages": [{"role": "user", "content": question}],
        "model": model_name,
        "stream": True,
        "research_mode": research_mode,
    }

    actual_output: str = ""
    retrieval_context: List[str] = []
    case_law_context: List[Dict[str, Any]] = []
    tools_captured: List[ToolCall] = []
    tool_sequence: List[str] = []  # ordered tool names
    fallback_used: bool = False
    research_output: str = ""
    # tool_stack stores dicts: {"name": str, "input_parameters": dict, "output": any}
    tool_stack: List[Dict[str, Any]] = []
    summarisation_output: List[str] = []
    summarisation_used: bool = False
    _in_summarisation: bool = False
    # LIFO stack matching each api_call_start to its api_call_end
    _pending_api_entries: List[Dict[str, Any]] = []
    is_error: bool = False
    error_message: str = ""

    seq = 0  # monotonic sequence counter for verbose log

    def _seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    try:
        with client.stream("POST", "/api/system/chat", json=chat_payload, timeout=300) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line
                try:
                    if isinstance(decoded, bytes):
                        decoded = decoded.decode("utf-8")
                except Exception:
                    continue

                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:]
                if data_str == "[DONE]":
                    _vlog(_vf, "[SEQ ---] [DONE]")
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    _vlog(_vf, f"[SEQ ---] JSON decode error: {_trunc(data_str)}")
                    logger.warning("Failed to parse JSON: %s", data_str[:200])
                    continue

                event_type = data.get("type", "")

                # tool name used in tool_start / tool_end / tool_result events
                tool_name = data.get("tool", "")

                logger.debug(
                    "EVENT %-16s tool=%-35s stack=%s",
                    event_type,
                    tool_name or data.get("url", ""),
                    [t["name"] for t in tool_stack],
                )

                # ----------------------------------------------------------
                # tool_call  (OpenRouter / OpenAI format)
                # ----------------------------------------------------------
                if event_type == "tool_call":
                    func_names = []
                    stack_before = [t["name"] for t in tool_stack]
                    for tc in data.get("tool_calls", []):
                        func = tc.get("function", {})
                        raw_args = func.get("arguments", {})
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {}
                        elif not isinstance(raw_args, dict):
                            raw_args = {}
                        func_name = func.get("name", "Unknown")
                        func_names.append(func_name)
                        tool_stack.append({
                            "name": func_name,
                            "input_parameters": raw_args,
                            "output": None,
                        })
                        # tool_call fires first with the real function name
                        # (e.g. "delegate_research"), before any tool_start event.
                        tool_sequence.append(func_name)

                    seq_n = _seq()
                    _vlog(_vf, f"[SEQ {seq_n:03d}] tool_call")
                    _vlog(_vf, f"  raw (truncated): {_trunc(data_str)}")
                    _vlog(_vf, f"  func_names:      {func_names}")
                    _vlog(_vf, f"  stack_before:    {stack_before}")
                    _vlog(_vf, f"  action:          push {func_names} → tool_stack; append to tool_sequence")
                    _vlog(_vf, f"  stack_after:     {[t['name'] for t in tool_stack]}")
                    _vlog(_vf, "")

                # ----------------------------------------------------------
                # tool_start  (Ollama format)
                # ----------------------------------------------------------
                elif event_type == "tool_start":
                    stack_before = [t["name"] for t in tool_stack]
                    seq_n = _seq()

                    # Summarisation wrapper: skip pushing to tool_stack
                    if tool_name == "Extracting the relevant sections from a large document":
                        _vlog(_vf, f"[SEQ {seq_n:03d}] tool_start")
                        _vlog(_vf, f"  tool:            {tool_name!r}")
                        _vlog(_vf, f"  _in_summarisation: {_in_summarisation} → True")
                        _vlog(_vf, f"  stack_before:    {stack_before}")
                        _vlog(_vf, f"  action:          SET _in_summarisation=True; skip stack push (continue)")
                        _vlog(_vf, f"  stack_after:     {[t['name'] for t in tool_stack]}")
                        _vlog(_vf, "")
                        _in_summarisation = True
                        continue

                    if _in_summarisation:
                        # Progress message inside summarisation; skip the stack
                        _vlog(_vf, f"[SEQ {seq_n:03d}] tool_start (inside summarisation — skip)")
                        _vlog(_vf, f"  tool:            {tool_name!r}")
                        _vlog(_vf, "")
                        continue

                    seq_before_seq = list(tool_sequence)
                    tool_stack.append({
                        "name": tool_name,
                        "input_parameters": {},
                        "output": None,
                    })
                    # Record Worker tool invocations in start order.
                    # Skip "Research Agent" — it's the server-internal wrapper
                    # for "delegate_research", already recorded via tool_call.
                    if tool_name != "Research Agent":
                        tool_sequence.append(tool_name)

                    _vlog(_vf, f"[SEQ {seq_n:03d}] tool_start")
                    _vlog(_vf, f"  tool:            {tool_name!r}")
                    _vlog(_vf, f"  _in_summarisation: {_in_summarisation}")
                    _vlog(_vf, f"  stack_before:    {stack_before}")
                    _vlog(_vf, f"  action:          push {tool_name!r} → tool_stack")
                    if tool_name != "Research Agent":
                        _vlog(_vf, f"  tool_sequence:   {seq_before_seq} → {list(tool_sequence)}")
                    else:
                        _vlog(_vf, f"  note:            Research Agent — NOT added to tool_sequence")
                    _vlog(_vf, f"  stack_after:     {[t['name'] for t in tool_stack]}")
                    _vlog(_vf, "")

                # ----------------------------------------------------------
                # api_call_start
                # ----------------------------------------------------------
                elif event_type == "api_call_start":
                    url = data.get("url", "")
                    method = data.get("method", "")
                    seq_n = _seq()

                    if tool_stack:
                        tool_stack[-1]["input_parameters"] = {
                            "url": url,
                            "method": method,
                            "payload": data.get("payload", {}),
                        }
                        # Push onto pending LIFO stack so api_call_end pops in order
                        _pending_api_entries.append(tool_stack[-1])
                        action_desc = f"set input_parameters on {tool_stack[-1]['name']!r}; pushed to _pending_api_entries"
                    else:
                        action_desc = "tool_stack empty — skipped"

                    _vlog(_vf, f"[SEQ {seq_n:03d}] api_call_start")
                    _vlog(_vf, f"  url:             {url}")
                    _vlog(_vf, f"  method:          {method}")
                    _vlog(_vf, f"  top_of_stack:    {tool_stack[-1]['name'] if tool_stack else '(empty)'}")
                    _vlog(_vf, f"  action:          {action_desc}")
                    _vlog(_vf, f"  _pending_api_cnt: {len(_pending_api_entries)}")
                    _vlog(_vf, "")

                # ----------------------------------------------------------
                # api_call_end
                # ----------------------------------------------------------
                elif event_type == "api_call_end":
                    resp = data.get("response", {})
                    seq_n = _seq()

                    api_entry = (
                        _pending_api_entries.pop()
                        if _pending_api_entries
                        else (tool_stack[-1] if tool_stack else None)
                    )
                    current_tool = api_entry["name"] if api_entry else ""

                    resp_type = "unknown"
                    items_added_ctx = 0
                    items_added_case_law = 0

                    if isinstance(resp, dict) and "full_text" in resp:
                        # get_legislation_text fallback — capture full statutory text
                        retrieval_context.append(resp["full_text"])
                        fallback_used = True
                        items_added_ctx = 1
                        resp_type = "get_legislation_text (full_text)"

                    elif "search_legislation_sections" in current_tool:
                        # Primary retrieval path — capture actual section text
                        if isinstance(resp, list):
                            sections = resp
                        elif isinstance(resp, dict):
                            sections = resp.get("sections") or resp.get("results") or []
                        else:
                            sections = []

                        for sec in sections:
                            if not isinstance(sec, dict):
                                continue
                            content = (
                                sec.get("content")
                                or sec.get("text")
                                or sec.get("excerpt")
                                or ""
                            )
                            sec_title = sec.get("title") or sec.get("section_title") or ""
                            if content:
                                retrieval_context.append(
                                    f"{sec_title}: {content}" if sec_title else content
                                )
                                items_added_ctx += 1
                        resp_type = f"legislation sections ({items_added_ctx} items)"

                    elif isinstance(resp, dict) and (
                        "search_case_law" in current_tool
                        or (
                            "results" in resp
                            and resp["results"]
                            and isinstance(resp["results"], list)
                            and len(resp["results"]) > 0
                            and isinstance(resp["results"][0], dict)
                            and "ncn" in resp["results"][0]
                        )
                    ):
                        # Case law results
                        for r in resp.get("results", []):
                            if not isinstance(r, dict):
                                continue
                            title = r.get("title", "")
                            ncn = r.get("ncn", "")
                            court = r.get("court", "")
                            date = r.get("date", "")
                            url_r = r.get("url", "")
                            case_law_context.append({
                                "title": title,
                                "ncn": ncn,
                                "court": court,
                                "date": date,
                                "url": url_r,
                            })
                            parts = [p for p in [ncn, court, date] if p]
                            retrieval_context.append(
                                f"{title} ({' | '.join(parts)})" if parts else title
                            )
                            items_added_case_law += 1
                        resp_type = f"case_law ({items_added_case_law} results)"

                    elif isinstance(resp, dict) and "results" in resp:
                        # Legislation search metadata
                        for r in resp["results"]:
                            if not isinstance(r, dict):
                                continue
                            title = r.get("title", "")
                            year = str(r.get("year", "")) if r.get("year") else ""
                            status = r.get("status", "")
                            parts = [p for p in [year, status] if p]
                            retrieval_context.append(
                                f"{title} ({', '.join(parts)})" if parts else title
                            )
                            items_added_ctx += 1
                        resp_type = f"legislation search metadata ({items_added_ctx} items)"

                    else:
                        resp_type = f"other / unrecognised ({type(resp).__name__})"

                    if api_entry is not None:
                        api_entry["output"] = json.dumps(resp, default=str)

                    _vlog(_vf, f"[SEQ {seq_n:03d}] api_call_end")
                    _vlog(_vf, f"  url:             {data.get('url', '')}")
                    _vlog(_vf, f"  current_tool:    {current_tool!r}")
                    _vlog(_vf, f"  resp_type:       {resp_type}")
                    _vlog(_vf, f"  items_retrieval: +{items_added_ctx}")
                    _vlog(_vf, f"  items_case_law:  +{items_added_case_law}")
                    _vlog(_vf, f"  fallback_used:   {fallback_used}")
                    _vlog(_vf, f"  _pending_api_cnt_after: {len(_pending_api_entries)}")
                    _vlog(_vf, "")

                # ----------------------------------------------------------
                # tool_end
                # ----------------------------------------------------------
                elif event_type == "tool_end":
                    stack_before = [t["name"] for t in tool_stack]
                    seq_n = _seq()

                    # Summarisation wrapper: capture result and skip tool_stack
                    if tool_name == "Extracting the relevant sections from a large document":
                        summarised_text = str(data.get("result", "")).strip()
                        excluded = summarised_text.lower() in ("done", "none", "null", "")
                        if summarised_text and not excluded:
                            summarisation_output.append(summarised_text)
                            summarisation_used = True
                            _vlog(_vf, f"[SEQ {seq_n:03d}] tool_end")
                            _vlog(_vf, f"  tool:            {tool_name!r}")
                            _vlog(_vf, f"  result_len:      {len(summarised_text)}")
                            _vlog(_vf, f"  result_preview:  {_trunc(summarised_text)}")
                            _vlog(_vf, f"  action:          summarisation_output.append(text)  [now {len(summarisation_output)} item(s)]")
                            _vlog(_vf, f"  _in_summarisation: True → False")
                            _vlog(_vf, "")
                        else:
                            _vlog(_vf, f"[SEQ {seq_n:03d}] tool_end")
                            _vlog(_vf, f"  tool:            {tool_name!r}")
                            _vlog(_vf, f"  result:          {'excluded value' if excluded else 'None/empty'} — skipped")
                            _vlog(_vf, f"  _in_summarisation: True → False")
                            _vlog(_vf, "")
                        _in_summarisation = False
                        continue

                    if tool_stack:
                        completed = tool_stack.pop()

                        # Skip "Research Agent" completion — meaningful output is
                        # already captured via the tool_result handler.
                        if completed["name"] == "Research Agent":
                            _vlog(_vf, f"[SEQ {seq_n:03d}] tool_end")
                            _vlog(_vf, f"  tool:            {tool_name!r}")
                            _vlog(_vf, f"  popped:          {completed['name']!r}")
                            _vlog(_vf, f"  action:          SKIP — Research Agent handled by tool_result")
                            _vlog(_vf, f"  stack_after:     {[t['name'] for t in tool_stack]}")
                            _vlog(_vf, "")
                            continue

                        output_source = "api_call_end (already set)"
                        if not completed["output"]:
                            fallback_result = str(data.get("result", "Done")).strip()
                            if fallback_result.lower() in ("done", "none", "null", ""):
                                completed["output"] = json.dumps({
                                    "status": "no_results",
                                    "message": "API returned no results or empty response",
                                }, default=str)
                                output_source = "fallback no_results sentinel"
                            else:
                                completed["output"] = fallback_result
                                output_source = "fallback result text"

                        completed_name = completed["name"]
                        if completed_name == "get_legislation_text":
                            fallback_used = True

                        tools_captured.append(ToolCall(
                            name=completed_name,
                            input_parameters=completed["input_parameters"],
                            output=completed["output"],
                        ))

                        _vlog(_vf, f"[SEQ {seq_n:03d}] tool_end")
                        _vlog(_vf, f"  tool:            {tool_name!r}")
                        _vlog(_vf, f"  stack_before:    {stack_before}")
                        _vlog(_vf, f"  popped:          {completed_name!r}")
                        _vlog(_vf, f"  output_source:   {output_source}")
                        output_str = str(completed["output"]) if completed["output"] is not None else ""
                        _vlog(_vf, f"  output_len:      {len(output_str)}")
                        _vlog(_vf, f"  tools_captured:  {len(tools_captured)} total")
                        _vlog(_vf, f"  stack_after:     {[t['name'] for t in tool_stack]}")
                        _vlog(_vf, "")
                    else:
                        _vlog(_vf, f"[SEQ {seq_n:03d}] tool_end (stack empty — skipped)")
                        _vlog(_vf, f"  tool:            {tool_name!r}")
                        _vlog(_vf, "")

                # ----------------------------------------------------------
                # tool_result  (used for delegate_research / Research Agent)
                # ----------------------------------------------------------
                elif event_type == "tool_result":
                    result_text = str(data.get("result", ""))
                    input_params: Dict[str, Any] = {}
                    stack_before = [t["name"] for t in tool_stack]
                    seq_n = _seq()

                    # Determine the effective tool name by popping from tool_stack
                    if tool_stack:
                        delegation = tool_stack.pop()
                        input_params = delegation.get("input_parameters", {})
                        effective_tool_name = delegation.get("name", tool_name)
                    else:
                        effective_tool_name = tool_name

                    # Fix 1: the server may label Worker output as "Research Agent"
                    # in tool_result events — treat it as delegate_research.
                    _is_research_agent = (
                        effective_tool_name == "Research Agent"
                        or tool_name == "Research Agent"
                    )
                    if _is_research_agent:
                        research_output = result_text
                        effective_tool_name = "delegate_research"
                    elif effective_tool_name == "delegate_research":
                        research_output = result_text

                    tools_captured.append(ToolCall(
                        name=effective_tool_name,
                        input_parameters=input_params,
                        output=result_text,
                    ))

                    _vlog(_vf, f"[SEQ {seq_n:03d}] tool_result")
                    _vlog(_vf, f"  tool (event):    {tool_name!r}")
                    _vlog(_vf, f"  stack_before:    {stack_before}")
                    _vlog(_vf, f"  effective_name:  {effective_tool_name!r}")
                    _vlog(_vf, f"  is_research_agent: {_is_research_agent}")
                    _vlog(_vf, f"  research_output_set: {bool(research_output)}" + (f" ({len(result_text)} chars)" if research_output else ""))
                    _vlog(_vf, f"  stack_after:     {[t['name'] for t in tool_stack]}")
                    _vlog(_vf, "")

                # ----------------------------------------------------------
                # token  — streaming token from final LLM response
                # ----------------------------------------------------------
                elif event_type == "token":
                    actual_output += data.get("content", "")
                    _token_count += 1
                    # Don't log per-token — too noisy; summarised in 'result' block

                # ----------------------------------------------------------
                # result  — final complete message
                # ----------------------------------------------------------
                elif event_type == "result":
                    message = data.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if content:
                            actual_output = content
                    elif isinstance(message, str) and message:
                        actual_output = message

                    seq_n = _seq()
                    _vlog(_vf, f"[SEQ {seq_n:03d}] result")
                    _vlog(_vf, f"  actual_output:   {len(actual_output)} chars")
                    _vlog(_vf, f"  token_events:    {_token_count}")
                    _vlog(_vf, "")

                # ----------------------------------------------------------
                # error
                # ----------------------------------------------------------
                elif event_type == "error":
                    err_msg = data.get("error", "Unknown error")
                    seq_n = _seq()
                    _vlog(_vf, f"[SEQ {seq_n:03d}] error")
                    _vlog(_vf, f"  WARN: SSE error event: {err_msg}")
                    _vlog(_vf, "")
                    logger.warning("Stream error: %s", err_msg)

                # ----------------------------------------------------------
                # Post-event callback (--debug-events)
                # ----------------------------------------------------------
                if on_event:
                    on_event(data)

    except Exception as exc:
        _vlog(_vf, f"ERROR: audit_capture raised exception: {exc}")
        logger.error("audit_capture failed: %s", exc)
        is_error = True
        error_message = str(exc)

    # ------------------------------------------------------------------
    # Post-stream: flush tools left on the stack without a completion event
    # ------------------------------------------------------------------
    tool_stack_leftovers = 0
    for leftover in tool_stack:
        leftover_name = leftover["name"]
        if leftover_name == "Research Agent":
            continue
        tool_stack_leftovers += 1
        warn_msg = (
            f"tool_stack not empty at stream end — {leftover_name!r} started but no "
            "completion event received → flushed as no_completion_event ToolCall"
        )
        _vlog(_vf, f"WARN: {warn_msg}")
        logger.warning(
            "tool_stack not empty at stream end — '%s' started but no completion event received",
            leftover_name,
        )
        tools_captured.append(ToolCall(
            name=leftover_name,
            input_parameters=leftover.get("input_parameters", {}),
            output=json.dumps({
                "status": "no_completion_event",
                "message": "Tool started but its result event was never received from the stream",
            }, default=str),
        ))

    # ------------------------------------------------------------------
    # Fix 2: retroactive repair — if research_output is still empty, scan
    # tools_captured for any entry whose output contains the research signature.
    # ------------------------------------------------------------------
    retroactive_repair = False
    if not research_output:
        _RESEARCH_SIGNATURE = "[Research Agent Result]"
        for i, tc in enumerate(tools_captured):
            tc_output = str(tc.output) if tc.output else ""
            if _RESEARCH_SIGNATURE in tc_output:
                info_msg = (
                    f"retroactive repair: tool {tc.name!r} (index {i}) contains "
                    "research output signature → renamed to 'delegate_research'"
                )
                _vlog(_vf, f"INFO: {info_msg}")
                logger.info(
                    "retroactive repair: tool '%s' (index %d) contains research output "
                    "signature — renaming to 'delegate_research'",
                    tc.name,
                    i,
                )
                research_output = tc_output
                tools_captured[i] = ToolCall(
                    name="delegate_research",
                    input_parameters=tc.input_parameters if hasattr(tc, "input_parameters") else {},
                    output=tc_output,
                )
                retroactive_repair = True
                break

    # ------------------------------------------------------------------
    # Ensure actual_output is always a string
    # ------------------------------------------------------------------
    if not isinstance(actual_output, str):
        actual_output = str(actual_output) if actual_output else ""

    # ------------------------------------------------------------------
    # Verbose log: final state section
    # ------------------------------------------------------------------
    _ended_ts = datetime.now(timezone.utc)
    _vlog(_vf, "=== FINAL STATE ===")
    _vlog(_vf, f"actual_output:          {len(actual_output)} chars")
    _vlog(_vf, f"research_output:        {len(research_output)} chars" if research_output else "research_output:        (empty)")
    _vlog(_vf, f"retrieval_context:      {len(retrieval_context)} items")
    _vlog(_vf, f"tools_captured:         {[t.name for t in tools_captured]}")
    _vlog(_vf, f"tool_sequence:          {list(tool_sequence)}")
    _vlog(_vf, f"summarisation_output:   {len(summarisation_output)} item(s)  (total {sum(len(s) for s in summarisation_output)} chars)")
    _vlog(_vf, f"summarisation_used:     {summarisation_used}")
    _vlog(_vf, f"fallback_used:          {fallback_used}")
    _vlog(_vf, f"case_law_context:       {len(case_law_context)} items")
    _vlog(_vf, f"tool_stack_leftovers:   {tool_stack_leftovers}")
    _vlog(_vf, f"_pending_api_leftovers: {len(_pending_api_entries)}")
    _vlog(_vf, f"retroactive_repair:     {retroactive_repair}")
    _vlog(_vf, f"is_error:               {is_error}")
    if error_message:
        _vlog(_vf, f"error_message:          {error_message}")
    _vlog(_vf, f"ended:                  {_ended_ts.isoformat()}")
    _vlog(_vf, f"duration_s:             {(_ended_ts - _started_ts).total_seconds():.1f}")
    _vlog(_vf, "==================")

    if _vf:
        _vf.close()

    # Deduplicate retrieval_context (preserving order)
    retrieval_context = list(dict.fromkeys(retrieval_context))

    return {
        "actual_output": actual_output,
        "retrieval_context": retrieval_context,
        "tools_called": [
            {
                "name": tc.name,
                "input_parameters": tc.input_parameters if hasattr(tc, "input_parameters") else {},
                "output": tc.output,
            }
            for tc in tools_captured
        ],
        "research_output": research_output,
        "research_mode": research_mode,
        "case_law_context": case_law_context,
        "tool_sequence": tool_sequence,
        "fallback_used": fallback_used,
        "summarisation_output": summarisation_output,
        "summarisation_used": summarisation_used,
        "is_error": is_error,
        "error_message": error_message,
    }
