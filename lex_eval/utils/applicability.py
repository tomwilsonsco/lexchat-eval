"""Explicit source coverage for the existing checks."""

LEGISLATION_CHECKS = {
    "citation_grounding",
    "citation_read",
    "step_completion",
    "report_integration",
}


def exclusion(metric: str, record: dict) -> str | None:
    mode = record.get("research_mode", "legislation_only")
    if metric == "genuine_gap" and mode != "legislation_only":
        return "Not measured: Genuine Gap checks legislation-only research."
    if metric in LEGISLATION_CHECKS and mode == "case_law_only":
        return "Not measured: this check covers legislation, not judgment evidence."
    return None


def scope_note(metric: str, record: dict) -> str | None:
    if record.get(
        "research_mode"
    ) == "legislation_and_case_law" and metric in LEGISLATION_CHECKS | {
        "tool_usage",
        "citation_agreement",
    }:
        return "Legislation only; case-law behavior is not checked by this result."
    return None
