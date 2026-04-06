"""
Response groundedness metric.

Scores whether the final response to the user is strictly grounded in
the research agent's output, with no hallucinated facts.
"""

from __future__ import annotations

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class _GroundednessJudgement(BaseModel):
    is_grounded: bool
    score: int   # 1–5
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to determine if a final response is strictly grounded in the provided research output.

Research Output:
{research_output}

Final Response:
{actual_output}

Evaluate the response against these criteria:
1. Factual alignment: Does the final response contain any facts, claims, or legal assertions not present in the research output?
2. Hallucination check: Has the model invented new information?
3. Accuracy: Does the response misrepresent or contradict the research output?

Provide your evaluation in strict JSON format exactly like this:
{{
    "is_grounded": true or false,
    "score": <a number from 1 to 5, where 5 is perfectly grounded>,
    "reason": "<A one-sentence explanation of why it passed or failed>"
}}
"""


class ResponseGroundednessMetric(BaseMetric):
    """
    Evaluates whether the final response is grounded in the research
    agent's output, with no hallucinated or invented facts.

    research_output is not a standard LLMTestCase field so it is passed
    via the constructor, following the same pattern as LLMConsistencyMetric.

    Args:
        research_output: The research agent's synthesised output for this question.
        model:           A DeepEval-compatible judge model.
        threshold:       Minimum normalised score to pass (default 0.6).
    """

    def __init__(
        self, research_output: str, model, threshold: float = 0.6
    ) -> None:
        self.research_output = research_output
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        prompt = _PROMPT_TEMPLATE.format(
            research_output=self.research_output,
            actual_output=test_case.actual_output or "",
        )
        try:
            result = self.model.generate(prompt, schema=_GroundednessJudgement)
            if isinstance(result, _GroundednessJudgement):
                raw_score = float(result.score)
                self.reason = result.reason
            else:
                data = json.loads(str(result))
                raw_score = float(data["score"])
                self.reason = data["reason"]
        except Exception as exc:
            raw_score = 1.0
            self.reason = f"Judge error: {exc}"

        self.score = (raw_score - 1) / 4   # normalise 1–5 → 0.0–1.0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Response Groundedness"