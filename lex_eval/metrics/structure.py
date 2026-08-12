"""
Metrics that validate Worker Agent output quality.

MandatoryStructureMetric  — checks the 4-part Markdown heading structure.
CitationPassthroughMetric — checks that Worker references reach the final response.
CitationGroundingMetric   — checks that Worker citations were actually retrieved.
CitationDomainMetric      — checks that Worker citation URLs are on legislation.gov.uk.

All four metrics inspect the ``delegate_research`` tool-call output, which is where
the Worker Agent's response is surfaced.
"""

import json
import re
from urllib.parse import urlparse

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
    Checks that every reference link from the Worker output is present in
    the final response delivered to the user.

    Score:
        0.0  — Failure A: no URLs found in Worker output at all.
        0.5  — Failure B: one or more Worker links are missing from the
                          final response (citation links were dropped).
        1.0  — Pass: every Worker URL is present in the final response.

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
        passed_through = {link for link in worker_links if link in actual}
        missing = worker_links - passed_through

        if missing:
            self.score = 0.5
            self.success = False
            self.reason = (
                f"Failure B: {len(missing)} of {len(worker_links)} Worker "
                "link(s) missing from final response."
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"Pass: all {len(worker_links)} Worker link(s) present in "
                "final response."
            )

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Reference Links"


def _legislation_id_from_url(url: str) -> str:
    """
    Derive the Act-level legislation_id (e.g. ``ukpga/1978/29``) from a
    legislation.gov.uk URL, whether it points at the Act itself or a specific
    section/schedule within it (e.g. ``ukpga/1978/29/section/10C``).

    Mirrors how the LexChat server builds legislation_id from a search result
    URI (``LexChat/server_py/src/agent/tools/lex.py::_slim_search_results``:
    URL path, minus a leading ``id/`` segment), then keeps only the first
    three path segments (type/year/number) since that's the granularity
    search_legislation_sections and get_legislation_text are called at.
    """
    path = urlparse(url).path.lstrip("/")
    if path.startswith("id/"):
        path = path[3:]
    return "/".join(path.split("/")[:3])


def _retrieved_legislation_ids(test_case: LLMTestCase) -> set:
    """
    Return the set of legislation_ids the run's own tool calls actually
    retrieved: results returned by ``search_legislation``, plus the
    legislation_id argument passed to ``search_legislation_sections`` /
    ``get_legislation_text``.
    """
    ids: set = set()
    if not test_case.tools_called:
        return ids

    for tool in test_case.tools_called:
        if tool.name == "Worker: search_legislation":
            raw = tool.output
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
                for r in (data or {}).get("results", []):
                    lid = r.get("legislation_id")
                    if lid:
                        ids.add(lid)
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        elif tool.name in (
            "Worker: search_legislation_sections",
            "Worker: get_legislation_text",
        ):
            params = tool.input_parameters or {}
            lid = params.get("legislation_id")
            if lid:
                ids.add(lid)

    return ids


class CitationGroundingMetric(BaseMetric):
    """
    Checks that every Act cited in the Worker's report was actually retrieved
    by this run's own tool calls, rather than invented from pattern-matching.

    Catches fabrication (a citation to something never retrieved), not
    wrongness (a citation to a real, retrieved Act that doesn't actually
    answer the question, which is a substantive-correctness question this
    rule-based check can't make).

    Score:
        0.0  — no delegate_research call found; citations cannot be verified.
        1.0  — no legislation.gov.uk citation URLs in Worker output (nothing
               to falsely ground).
        0.0  — one or more cited Acts were never retrieved by search_legislation,
               search_legislation_sections, or get_legislation_text in this run.
        1.0  — every cited Act was retrieved by this run.

    No partial credit: unlike Reference Links, where "some links survived" is
    a meaningfully different failure from "none did", one fabricated citation
    is a full failure regardless of how many others were genuine.
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
                "citation grounding cannot be verified."
            )
            return self.score

        cited_urls = set(_URL_RE.findall(dr_output))
        cited_ids = {
            lid for lid in (_legislation_id_from_url(u) for u in cited_urls) if lid
        }

        if not cited_ids:
            self.score = 1.0
            self.success = True
            self.reason = (
                "No legislation.gov.uk citations found in Worker output; "
                "nothing to ground."
            )
            return self.score

        retrieved_ids = _retrieved_legislation_ids(test_case)
        fabricated = cited_ids - retrieved_ids

        if fabricated:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Fabricated citation(s): {sorted(fabricated)} cited in Worker "
                "output but never retrieved by search_legislation, "
                "search_legislation_sections, or get_legislation_text in this run."
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"All {len(cited_ids)} cited Act(s) were retrieved by this "
                "run's tool calls."
            )

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Grounding"


_ALLOWED_CITATION_DOMAIN = "legislation.gov.uk"


class CitationDomainMetric(BaseMetric):
    """
    Checks that every citation URL in the Worker's report points to
    legislation.gov.uk, the only domain the Worker's system prompt permits
    ("Do not invent URLs for domains other than legislation.gov.uk").

    Score:
        0.0  — no delegate_research call found; domains cannot be verified.
        1.0  — no citation URLs in Worker output (nothing to check).
        0.0  — one or more citation URLs point to a different domain.
        1.0  — every citation URL is on legislation.gov.uk.

    No partial credit, same reasoning as Citation Grounding: one invented
    domain is a full failure regardless of how many other citations are fine.
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
                "citation domains cannot be verified."
            )
            return self.score

        cited_urls = set(_URL_RE.findall(dr_output))

        if not cited_urls:
            self.score = 1.0
            self.success = True
            self.reason = "No citation URLs found in Worker output; nothing to check."
            return self.score

        def _on_allowed_domain(url: str) -> bool:
            netloc = urlparse(url).netloc.lower()
            return netloc == _ALLOWED_CITATION_DOMAIN or netloc.endswith(
                f".{_ALLOWED_CITATION_DOMAIN}"
            )

        off_domain = {u for u in cited_urls if not _on_allowed_domain(u)}

        if off_domain:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Off-domain citation(s): {sorted(off_domain)} do not point to "
                f"{_ALLOWED_CITATION_DOMAIN}, the only domain the Worker's "
                "system prompt permits."
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"All {len(cited_urls)} citation URL(s) are on "
                f"{_ALLOWED_CITATION_DOMAIN}."
            )

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Domain"
