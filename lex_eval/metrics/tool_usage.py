"""
Custom metric to validate that the LLM used all expected legislation tools
*and* invoked them in the correct phase order.

Presence scoring — 1/3 for each of the three required tools:
    - delegate_research
    - Worker: search_legislation
    - Worker: search_legislation_sections

Order scoring (legislation_only mode only) — the Worker tools must appear in
the phase order mandated by the Worker system prompt
(see ``LexChat/server_py/src/config.py``):

    1. Worker: search_legislation          (Phase 1 — DISCOVER)
    2. Worker: search_legislation_sections (Phase 2 — RETRIEVE PROVISIONS)
    3. Worker: get_legislation_text        (Phase 3 — FALLBACK, optional)

Final score:
    - 1.0  all three required tools present AND correct order
    - 0.5  all three required tools present BUT wrong order
    - <1.0 one or more required tools missing (proportional; order is reported
           for transparency but does not cap the score further)
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from typing import List, Optional, Set

# The three tools that must all be present for a full score
REQUIRED_TOOLS: list[str] = [
    "delegate_research",
    "Worker: search_legislation",
    "Worker: search_legislation_sections",
]

PER_TOOL_SCORE = round(1 / len(REQUIRED_TOOLS), 10)

# Expected invocation order of Worker legislation tools, per the Worker system
# prompt phases. `get_legislation_text` is the optional fallback (Phase 3) and
# is NOT a required tool, but if it is called it must come after
# `search_legislation_sections`.
EXPECTED_TOOL_ORDER: list[str] = [
    "Worker: search_legislation",
    "Worker: search_legislation_sections",
    "Worker: get_legislation_text",
]

# Score assigned when all required tools are present but invoked out of order.
ORDER_VIOLATION_SCORE = 0.5


def _first_occurrence(tool_sequence: List[str], name: str) -> Optional[int]:
    """Return the index of the first occurrence of *name* in *tool_sequence*, or None."""
    for i, t in enumerate(tool_sequence):
        if t == name:
            return i
    return None


def _check_tool_order(
    tool_sequence: Optional[List[str]], research_mode: str
) -> tuple[bool, str]:
    """Validate that Worker legislation tools appear in the expected phase order.

    Only the ``Worker:``-prefixed entries are considered — ``delegate_research``
    is a Manager-level call and is excluded from the phase ordering.

    Returns ``(order_ok, detail)`` where *detail* is a short human-readable
    description of the observed order (or the first violation).
    """
    if research_mode != "legislation_only":
        return True, "skipped (non-legislation mode)"
    if not tool_sequence:
        return True, "n/a (no tool_sequence captured)"

    worker_seq = [t for t in tool_sequence if t.startswith("Worker:")]
    if not worker_seq:
        return True, "n/a (no Worker tools in sequence)"

    present: list[tuple[str, int]] = []
    for name in EXPECTED_TOOL_ORDER:
        idx = _first_occurrence(worker_seq, name)
        if idx is not None:
            present.append((name.replace("Worker: ", ""), idx))

    order_ok = all(present[i][1] < present[i + 1][1] for i in range(len(present) - 1))

    if order_ok:
        return True, " → ".join(name for name, _ in present)

    # Report the first inversion.
    for i in range(len(present) - 1):
        if present[i][1] >= present[i + 1][1]:
            return False, f"{present[i + 1][0]} called before {present[i][0]}"
    return False, "order violation"


class ToolUsageMetric(BaseMetric):
    """
    Scores tool usage by awarding 1/3 for each required tool present, and
    (for ``legislation_only`` mode) additionally validates that the Worker
    tools were invoked in the correct phase order.

    Score:
        - 1.0  all three required tools present AND correct order
        - 0.5  all three required tools present BUT wrong order
        - <1.0 one or more required tools missing (proportional; order is
               reported for transparency but does not cap the score further)

    Passes when ``score >= threshold`` (default threshold 1.0), so any order
    violation or missing tool is a fail.

    Args:
        threshold: Minimum score to pass (default 1.0).
        research_mode: ``legislation_only`` (default), ``case_law_only``, or
            ``legislation_and_case_law``. The order check only runs for
            ``legislation_only``.
        tool_sequence: Optional ordered list of tool names as captured by
            ``audit_capture`` (e.g. ``["delegate_research", "Worker:
            search_legislation", ...]``). When omitted, only presence is
            scored.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        research_mode: str = "legislation_only",
        tool_sequence: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.research_mode = research_mode
        self.tool_sequence = tool_sequence
        self.score = 0.0
        self.reason = ""
        self.success = False

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        tools_used: Set[str] = set()
        # Tools whose completion event was never received from the stream (server-side
        # stream fault — the LLM did call them, but the result was never streamed back).
        incomplete_tools: Set[str] = set()

        if test_case.tools_called:
            for tool in test_case.tools_called:
                tools_used.add(tool.name)
                output = tool.output or ""
                if isinstance(output, str) and "no_completion_event" in output:
                    incomplete_tools.add(tool.name)

        present = [t for t in REQUIRED_TOOLS if t in tools_used]
        missing = [t for t in REQUIRED_TOOLS if t not in tools_used]

        all_required_present = len(present) == len(REQUIRED_TOOLS)

        # --- Presence score (1/3 per required tool) ---
        presence_score = len(present) / len(REQUIRED_TOOLS)

        # --- Order check (legislation_only only, when tool_sequence supplied) ---
        order_ok, order_detail = _check_tool_order(
            self.tool_sequence, self.research_mode
        )

        # --- Final score ---
        if all_required_present and not order_ok:
            # All tools present but wrong order → cap at ORDER_VIOLATION_SCORE.
            self.score = ORDER_VIOLATION_SCORE
        else:
            self.score = presence_score

        self.success = self.score >= self.threshold

        # --- Reason ---
        parts = [f"{t}: {'✓' if t in tools_used else '✗'}" for t in REQUIRED_TOOLS]
        self.reason = (
            f"Score {self.score:.3f} ({len(present)}/{len(REQUIRED_TOOLS)} tools used). "
            + " | ".join(parts)
        )
        if missing:
            self.reason += f" | Missing: {missing}"

        # Order section — always shown for legislation_only when a sequence
        # was supplied, even on missing-tool fails, for transparency.
        if self.research_mode == "legislation_only" and self.tool_sequence:
            order_icon = "✓" if order_ok else "✗"
            self.reason += f" | Order: {order_icon} {order_detail}"
            if all_required_present and not order_ok:
                self.reason += f" (expected: {' → '.join(n.replace('Worker: ', '') for n in EXPECTED_TOOL_ORDER)})"

        if incomplete_tools:
            self.reason += (
                f" | WARNING — stream incomplete (no result event received) for: "
                f"{sorted(incomplete_tools)}. LLM called the tool correctly; "
                f"server failed to return the completion event."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Tool Usage"
