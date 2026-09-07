"""Facts from audit traces, without additional scores or judge calls."""

import json


def searches(record: dict) -> list[dict]:
    audit = record.get("audit_json") or {}
    if isinstance(audit, str):
        audit = json.loads(audit)
    rows = []
    for index, step in enumerate(audit.get("delegations", []), 1):
        for tool in step.get("tools", []):
            name = tool.get("name", "")
            if not name.startswith("search_"):
                continue
            raw = tool.get("raw_result")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError):
                parsed = None
            items = (
                parsed
                if name == "search_legislation_sections"
                else parsed.get("results") if isinstance(parsed, dict) else None
            )
            error = tool.get("error") or (
                parsed.get("error") if isinstance(parsed, dict) else None
            )
            if error:
                state, count = "Error", None
            elif tool.get("truncated") or tool.get("budget_blocked"):
                state, count = "Unknown / incomplete", None
            elif isinstance(items, list):
                count = len(items)
                state = "Empty" if not count else "Results returned"
            else:
                state, count = "Unknown / incomplete", None
            rows.append(
                {
                    "Response": record["response_id"],
                    "Step": step.get("step", index),
                    "Tool": name,
                    "Arguments": json.dumps(tool.get("args") or {}, ensure_ascii=False),
                    "Outcome": state,
                    "Returned": count,
                    "Cache reused": bool(
                        tool.get("memo_hit") or tool.get("local_cache_hit")
                    ),
                    "Error": str(error or ""),
                }
            )
    for index, row in enumerate(rows):
        row["Later nonempty search in this step"] = row["Outcome"] == "Empty" and any(
            later["Step"] == row["Step"]
            and later["Tool"] == row["Tool"]
            and (later["Returned"] or 0) > 0
            for later in rows[index + 1 :]
        )
    return rows


def plan_steps(record: dict) -> list[dict]:
    audit = record.get("audit_json") or {}
    if isinstance(audit, str):
        audit = json.loads(audit)
    return [
        {
            "Step": d.get("step", i),
            "Title": d.get("title", ""),
            "Tools": len(d.get("tools", [])),
            "Report": d.get("report") or "",
            "Error": d.get("error"),
            "Reformatted": bool(d.get("reformatted")),
        }
        for i, d in enumerate(audit.get("delegations", []), 1)
    ]
