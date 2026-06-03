from deepeval.test_case import LLMTestCase, ToolCall
import json
import logging

logger = logging.getLogger(__name__)


def audit_capture(client, question, model_name, research_mode="legislation_only"):
    chat_payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an automated legal research agent. You MUST use your available tools to search legislation and provide a final answer. Do NOT ask the user for clarification. If a query is broad, make reasonable assumptions and summarise the most relevant statutory process.",
            },
            {"role": "user", "content": question},
        ],
        "model": model_name,
        "stream": True,
        "research_mode": research_mode,
    }

    print(f"\u23f3 Auditing research for: '{question}' (mode={research_mode})")

    actual_output = ""
    retrieval_context = []
    case_law_context = []   # structured {title, ncn, court, date, url} dicts
    tools_captured = []
    tool_sequence = []      # ordered worker tool names (excludes delegate_research)
    fallback_used = False   # True if get_legislation_text was called
    research_output = ""
    tool_stack = []

    with client.stream("POST", "/api/system/chat", json=chat_payload) as response:
        for line in response.iter_lines():
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
                        tool_stack.append(
                            {
                                "name": func.get("name", "Unknown"),
                                "input_parameters": func.get("arguments", {}),
                                "output": None,
                            }
                        )

                elif event_type == "tool_start":
                    tool_stack.append(
                        {
                            "name": data.get("tool", "Unknown"),
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

                elif event_type == "api_call_end":
                    resp = data.get("response", {})
                    current_tool = tool_stack[-1]["name"] if tool_stack else ""

                    if "full_text" in resp:
                        # get_legislation_text fallback — capture the full statutory text
                        retrieval_context.append(resp["full_text"])
                    elif current_tool == "search_legislation_sections":
                        # Primary retrieval path — capture actual section text
                        sections = resp.get("sections") or resp.get("results") or []
                        for sec in sections:
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
                    elif current_tool == "search_case_law" or (
                        "results" in resp
                        and resp["results"]
                        and "ncn" in resp["results"][0]
                    ):
                        # Case law results — store structured and add plain refs to context
                        for r in resp.get("results", []):
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
                    elif "results" in resp:
                        # Legislation search metadata — record title + year/status for context
                        for r in resp["results"]:
                            title = r.get("title", "")
                            year = str(r.get("year", "")) if r.get("year") else ""
                            status = r.get("status", "")
                            parts = [p for p in [year, status] if p]
                            retrieval_context.append(
                                f"{title} ({', '.join(parts)})" if parts else title
                            )

                    if tool_stack:
                        tool_stack[-1]["output"] = json.dumps(resp, default=str)

                elif event_type == "tool_end":
                    if tool_stack:
                        completed = tool_stack.pop()
                        if not completed["output"]:
                            completed["output"] = str(data.get("result", "Done"))
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
                    effective_tool_name = tool_name
                    if tool_stack:
                        delegation = tool_stack.pop()
                        input_params = delegation.get("input_parameters", {})
                        effective_tool_name = delegation.get("name", tool_name)

                    if effective_tool_name == "delegate_research":
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

    # ensure actual_output is always a string
    if not isinstance(actual_output, str):
        actual_output = str(actual_output) if actual_output else ""

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
    }
