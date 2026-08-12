"""
Metrics that validate Worker Agent output quality.

MandatoryStructureMetric  — checks the 4-part Markdown heading structure.
CitationPassthroughMetric — checks that Worker references reach the final response.

Both metrics inspect the ``delegate_research`` tool-call output, which is where
the Worker Agent's response is surfaced.
"""

import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

_DELEGATE_TOOL_NAME = "delegate_research"

# Matches http(s) URLs; stops at whitespace or markdown link-closing chars.
_URL_RE = re.compile(r"https?://[^\s\)\]>,\"']+")

# Bare keywords to search for (case-insensitive, colon optional).  Tolerates
# variation in bold markers, numbering, and trailing colons, e.g.:
#   "**Summary Answer (BLUF):**"        ✓
#   "### **1. Summary Answer (BLUF):**" ✓
#   "### 1. **Summary Answer (BLUF):**" ✓
#   "### **3. Jurisdiction & Status**"  ✓  (no colon)
#
# Each entry may be either a single string or a list of acceptable
# alternatives. The summary heading accepts both "Summary Answer (BLUF)"
# and "Summary Answer" — the (BLUF) qualifier is a stylistic hint in the
# Worker system prompt (see LexChat/server_py/src/config.py), not a
# semantic requirement, so either form passes. Likewise "Jurisdiction &
# Status"/"Jurisdiction & Currency" accept the spelled-out "and" — models
# routinely paraphrase the prompt's literal "&" this way.
REQUIRED_HEADINGS = {
    "legislation_only": [
        ["Summary Answer (BLUF)", "Summary Answer"],
        "Detailed Analysis",
        ["Jurisdiction & Status", "Jurisdiction and Status"],
        "References",
    ],
    "case_law_only": [
        ["Summary Answer (BLUF)", "Summary Answer"],
        "Key Cases",
        "Analysis",
        ["Jurisdiction & Currency", "Jurisdiction and Currency"],
        "References",
    ],
    "legislation_and_case_law": [
        ["Summary Answer (BLUF)", "Summary Answer"],
        "Statutory Framework",
        "Key Cases",
        ["Jurisdiction & Status", "Jurisdiction and Status"],
        "References",
    ],
}

# A heading match must sit at the start of its line, after only "decoration"
# characters (whitespace, #, *, digits, '.', '-', ':') — this is what lets a
# bare substring check for something like "References" tell a real heading
# apart from the word appearing mid-sentence in ordinary legal prose (e.g.
# "references to the 1978 Act..."), without requiring a literal Markdown
# '#' that real Worker output doesn't always use.
_HEADING_LINE_PREFIX = r"[\s#*\d.\-:]*"


def _get_delegate_output(test_case: LLMTestCase) -> str | None:
    """Return the ``delegate_research`` tool-call output, or None if absent."""
    if test_case.tools_called:
        for tool in test_case.tools_called:
            if tool.name == _DELEGATE_TOOL_NAME:
                raw = tool.output
                return raw if isinstance(raw, str) else str(raw)
    return None


class MandatoryStructureMetric(BaseMetric):
    """
    Ensures the Worker Agent strictly adhered to the mandatory Markdown structure
    mandated by its system prompt for the given research mode.

    Looks for the headings inside the ``delegate_research`` tool-call output
    rather than the top-level actual_output, because the Worker's response is
    surfaced as the return value of that tool.

    Matching is case-insensitive and ignores surrounding bold markers /
    numbering so minor formatting variations don't cause false failures.

    Score:
        1.0  — all mandatory headings present (pass)
        0.0  — one or more headings missing, or no delegate_research call found
    """

    def __init__(
        self, threshold: float = 1.0, research_mode: str = "legislation_only"
    ) -> None:
        self.threshold = threshold
        self.research_mode = research_mode
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        dr_output = _get_delegate_output(test_case)

        if dr_output is None:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "structure cannot be verified."
            )
            return self.score

        lowered = dr_output.lower()
        headings = REQUIRED_HEADINGS.get(
            self.research_mode, REQUIRED_HEADINGS["legislation_only"]
        )

        def _heading_present(heading) -> bool:
            variants = heading if isinstance(heading, list) else [heading]
            return any(
                re.search(
                    rf"(?m)^{_HEADING_LINE_PREFIX}{re.escape(v.lower())}", lowered
                )
                for v in variants
            )

        missing = [h for h in headings if not _heading_present(h)]

        if missing:
            self.score = 0.0
            self.success = False
            display = [h[0] if isinstance(h, list) else h for h in missing]
            self.reason = f"Missing mandatory headings: {', '.join(display)}"
        else:
            self.score = 1.0
            self.success = True
            self.reason = "All mandatory Markdown headings present in Worker output."

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Research Output Structure"


class CitationPassthroughMetric(BaseMetric):
    """
    Checks that at least one reference link from the Worker output is present
    in the final response delivered to the user.

    Score:
        0.0  — Failure A: no URLs found in Worker output at all.
        0.5  — Failure B: Worker output contains URLs but none appear in the
                          final response (citation links were dropped).
        1.0  — Pass: at least one Worker URL is present in the final response.

    Threshold defaults to 1.0, so both failure modes are recorded as fails.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        dr_output = _get_delegate_output(test_case)

        if dr_output is None:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "citations cannot be verified."
            )
            return self.score

        worker_links = set(_URL_RE.findall(dr_output))

        if not worker_links:
            self.score = 0.0
            self.success = False
            self.reason = "Failure A: no reference links found in Worker output."
            return self.score

        actual = test_case.actual_output or ""
        passed_through = [link for link in worker_links if link in actual]

        if not passed_through:
            self.score = 0.5
            self.success = False
            self.reason = (
                f"Failure B: {len(worker_links)} link(s) in Worker output "
                "but none present in final response."
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"Pass: {len(passed_through)} of {len(worker_links)} Worker "
                "link(s) present in final response."
            )

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Reference Links"
