from deepeval.test_case import LLMTestCase, ToolCall
import json
import logging

logger = logging.getLogger(__name__)


def audit_capture(
    client, question, model_name, research_mode="legislation_only", on_event=None
):
    chat_payload = {
        "messages": [
            {"role": "user", "content": question},
        ],
        "model": model_name,
        "stream": True,
        "research_mode": research_mode,
    }

    print(f"\u23f3 Auditing research for: '{question}' (mode={research_mode})")

    actual_output = ""
    retrieval_context = []
    case_law_context = []  # structured {title, ncn, court, date, url} dicts
    tools_captured = []
    tool_sequence = []  # ordered worker tool names (excludes delegate_research)
    fallback_used = False  # True if get_legislation_text was called
    research_output = ""
    tool_stack = []
    summarisation_output = []  # ordered list of summarised texts
    summarisation_used = False  # True if any summarisation occurred
    _in_summarisation = False  # True while inside a summarisation wrapper
    _pending_api_entries = (
        []
    )  # LIFO stack matching each api_call_start to its api_call_end
    _event_count = 0

    with client.stream("POST", "/api/system/chat", json=chat_payload) as response:
        for line in response.iter_lines():
            if on_event:
                _event_count += 1
                on_event(
                    {
                        "seq": _event_count,
                        "raw": line,
                        "question": question,
                        "model": model_name,
                    }
                )
            if not line.startswith("data: "):
                continue
            json_str = line[6:]
            if json_str == "[DONE]":
                break

            try:
                data = json.loads(json_str)
                event_type = data.get("type")

                tool_name = data.get("tool", "")
                logger.debug(
                    "EVENT %-16s tool=%-35s stack=%s",
                    event_type,
                    tool_name or data.get("url", ""),
                    [t["name"] for t in tool_stack],
                )

                if event_type == "tool_call":
                    for tc in data.get("tool_calls", []):
                        func = tc.get("function", {})
                        raw_args = func.get("arguments", {})
                        # OpenRouter/OpenAI send arguments as a JSON string;
                        # Ollama/other providers may send them as a dict.
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {}
                        elif not isinstance(raw_args, dict):
                            raw_args = {}
                        tool_stack.append(
                            {
                                "name": func.get("name", "Unknown"),
                                "input_parameters": raw_args,
                                "output": None,
                            }
                        )

                elif event_type == "tool_start":
                    tool_name = data.get("tool", "")
                    # Summarisation wrapper: skip pushing to tool_stack —
                    # progress events within it have no matching tool_end.
                    if tool_name == "Extracting the relevant sections from a large document":
                        _in_summarisation = True
                        continue
                    if _in_summarisation:
                        # Progress message inside summarisation; skip the stack
                        continue
                    tool_stack.append(
                        {
                            "name": tool_name,
                            "input_parameters": {},
                            "output": None,
                        }
                    )

                elif event_type == "api_call_start":
                    if tool_stack:
                        tool_stack[-1]["input_parameters"] = {
                            "url": data.get("url", ""),
                            "method": data.get("method", ""),
                            "payload": data.get("payload", {}),
                        }
                        # Push the owning entry onto the pending LIFO stack so
                        # api_call_end pops it back in order, even when multiple
                        # concurrent calls fire api_call_start before any
                        # api_call_end arrives (tool_start events may interleave).
                        _pending_api_entries.append(tool_stack[-1])

                elif event_type == "api_call_end":
                    resp = data.get("response", {})
                    api_entry = (
                        _pending_api_entries.pop()
                        if _pending_api_entries
                        else (tool_stack[-1] if tool_stack else None)
                    )
                    current_tool = api_entry["name"] if api_entry else ""

                    if isinstance(resp, dict) and "full_text" in resp:
                        # get_legislation_text fallback — capture the full statutory text
                        retrieval_context.append(resp["full_text"])
                    elif "search_legislation_sections" in current_tool:
                        # Primary retrieval path — capture actual section text
                        # resp can be a list directly or a dict with "sections"/"results"
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
                            sec_title = (
                                sec.get("title") or sec.get("section_title") or ""
                            )
                            if content:
                                retrieval_context.append(
                                    f"{sec_title}: {content}" if sec_title else content
                                )
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
                        # Case law results — store structured and add plain refs to context
                        for r in resp.get("results", []):
                            if not isinstance(r, dict):
                                continue
                            title = r.get("title", "")
                            ncn = r.get("ncn", "")
                            court = r.get("court", "")
                            date = r.get("date", "")
                            url = r.get("url", "")
                            case_law_context.append(
                                {
                                    "title": title,
                                    "ncn": ncn,
                                    "court": court,
                                    "date": date,
                                    "url": url,
                                }
                            )
                            parts = [p for p in [ncn, court, date] if p]
                            retrieval_context.append(
                                f"{title} ({' | '.join(parts)})" if parts else title
                            )
                    elif isinstance(resp, dict) and "results" in resp:
                        # Legislation search metadata — record title + year/status for context
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

                    if api_entry is not None:
                        api_entry["output"] = json.dumps(resp, default=str)

                elif event_type == "tool_end":
                    tool_name = data.get("tool", "")
                    # Summarisation wrapper: capture the result and skip tool_stack
                    if tool_name == "Extracting the relevant sections from a large document":
                        summarised_text = str(data.get("result", "")).strip()
                        if summarised_text and summarised_text.lower() not in ("done", "none", "null", ""):
                            summarisation_output.append(summarised_text)
                            summarisation_used = True
                        _in_summarisation = False
                        continue

                    if tool_stack:
                        completed = tool_stack.pop()
                        if not completed["output"]:
                            fallback_result = str(data.get("result", "Done")).strip()
                            if fallback_result.lower() in ("done", "none", "null", ""):
                                completed["output"] = json.dumps(
                                    {
                                        "status": "no_results",
                                        "message": "API returned no results or empty response",
                                    },
                                    default=str,
                                )
                            else:
                                completed["output"] = fallback_result
                        completed_name = completed["name"]
                        # Track worker tool order (skip manager-level delegation wrapper)
                        if completed_name != "delegate_research":
                            tool_sequence.append(completed_name)
                            if completed_name == "get_legislation_text":
                                fallback_used = True
                        tools_captured.append(
                            ToolCall(
                                name=completed_name,
                                input_parameters=completed["input_parameters"],
                                output=completed["output"],
                            )
                        )

                elif event_type == "tool_result":
                    result_text = str(data.get("result", ""))
                    input_params = {}

                    # ── Determine the effective tool name ─────────────────
                    # Prefer the name from the tool_stack (tool_start), but
                    # the LexChat server sometimes sends tool_start under
                    # "delegate_research" and tool_result under
                    # "Research Agent".  Reconcile the mismatch here.
                    if tool_stack:
                        delegation = tool_stack.pop()
                        input_params = delegation.get("input_parameters", {})
                        effective_tool_name = delegation.get("name", tool_name)
                    else:
                        effective_tool_name = tool_name

                    # Fix 1: the server may label the Worker output as
                    # "Research Agent" in tool_result events.  When we see
                    # that (or a tool_stack name mismatch resolves to it),
                    # treat the result as the delegate_research output.
                    _is_research_agent = (
                        effective_tool_name == "Research Agent"
                        or tool_name == "Research Agent"
                    )
                    if _is_research_agent:
                        research_output = result_text
                        # Normalise the captured tool name so downstream
                        # metrics can find it under "delegate_research".
                        effective_tool_name = "delegate_research"
                    elif effective_tool_name == "delegate_research":
                        research_output = result_text

                    tools_captured.append(
                        ToolCall(
                            name=effective_tool_name,
                            input_parameters=input_params,
                            output=result_text,
                        )
                    )

                elif event_type == "token":
                    actual_output += data.get("content", "")

                elif event_type == "result":
                    message = data.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if content:
                            actual_output = content
                    elif isinstance(message, str) and message:
                        actual_output = message

                elif event_type == "error":
                    logger.warning(
                        "Stream error: %s", data.get("error", "Unknown error")
                    )

            except json.JSONDecodeError:
                continue

    # Flush any tools that started but whose completion event was never received.
    # This happens when the server emits tool_start for delegate_research but
    # the corresponding tool_result/tool_end is missing from the stream (e.g.
    # Run 2 behaviour where a worker sub-tool fires tool_result instead).
    for leftover in tool_stack:
        leftover_name = leftover["name"]
        logger.warning(
            "tool_stack not empty at stream end — '%s' started but no completion event received",
            leftover_name,
        )
        tools_captured.append(
            ToolCall(
                name=leftover_name,
                input_parameters=leftover.get("input_parameters", {}),
                output=json.dumps(
                    {
                        "status": "no_completion_event",
                        "message": "Tool started but its result event was never received from the stream",
                    },
                    default=str,
                ),
            )
        )

    # ensure actual_output is always a string
    if not isinstance(actual_output, str):
        actual_output = str(actual_output) if actual_output else ""

    # Fix 2: retroactive repair — if research_output is still empty, scan
    # tools_captured for any entry whose output contains the research
    # signature ("[Research Agent Result]").  The LexChat server sometimes
    # sends the Worker output under a non-standard tool name, so the
    # inline handler (Fix 1) may have missed it if the event wasn't a
    # tool_result type.  When found, rename the tool to "delegate_research"
    # and set research_output so downstream metrics can locate it.
    if not research_output:
        _RESEARCH_SIGNATURE = "[Research Agent Result]"
        for i, tc in enumerate(tools_captured):
            tc_output = str(tc.output) if tc.output else ""
            if _RESEARCH_SIGNATURE in tc_output:
                logger.info(
                    "retroactive repair: tool '%s' (index %d) contains research output "
                    "signature — renaming to 'delegate_research'",
                    tc.name,
                    i,
                )
                research_output = tc_output
                tools_captured[i] = ToolCall(
                    name="delegate_research",
                    input_parameters=tc.input_parameters,
                    output=tc_output,
                )
                break

    return {
        "test_case": LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=list(dict.fromkeys(retrieval_context)),
            tools_called=tools_captured,
        ),
        "research_output": research_output,
        "research_mode": research_mode,
        "case_law_context": case_law_context,
        "tool_sequence": tool_sequence,
        "fallback_used": fallback_used,
        "summarisation_output": summarisation_output,
        "summarisation_used": summarisation_used,
    }
