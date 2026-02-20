"""
Custom metric to validate that the LLM used all expected legislation tools.

Scores 1/3 for each of the three required tools:
    - delegate_research
    - Worker: search_legislation
    - Worker: get_legislation_text

A score of 1.0 means all three were used; anything less is a fail.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from typing import Set


# The three tools that must all be present for a full score
REQUIRED_TOOLS: list[str] = [
    "delegate_research",
    "Worker: search_legislation",
    "Worker: get_legislation_text",
]

PER_TOOL_SCORE = round(1 / len(REQUIRED_TOOLS), 10)


class ToolUsageMetric(BaseMetric):
    """
    Scores tool usage by awarding 1/3 for each required tool present.

    Score = (number of required tools used) / 3
    Passes when score == 1.0 (all three tools were used).

    Args:
        threshold: Minimum score to pass (default 1.0).
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    async def a_measure(
        self, test_case: LLMTestCase, *args, **kwargs
    ) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        tools_used: Set[str] = set()
        if test_case.tools_called:
            tools_used = {tool.name for tool in test_case.tools_called}

        present = [t for t in REQUIRED_TOOLS if t in tools_used]
        missing = [t for t in REQUIRED_TOOLS if t not in tools_used]

        self.score = len(present) / len(REQUIRED_TOOLS)
        self.success = self.score >= self.threshold

        parts = [f"{t}: {'✓' if t in tools_used else '✗'}" for t in REQUIRED_TOOLS]
        self.reason = (
            f"Score {self.score:.3f} ({len(present)}/{len(REQUIRED_TOOLS)} tools used). "
            + " | ".join(parts)
        )
        if missing:
            self.reason += f" | Missing: {missing}"

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Tool Usage"
