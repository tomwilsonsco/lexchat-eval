"""
Research groundedness metric.

Scores whether the research agent's output is strictly grounded in the
raw legal text retrieved from the API, with no extrapolation.
"""

from __future__ import annotations

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

_MAX_CONTEXT_CHARS: int = (128_000 - 30_000) * 4  # ≈ 392 000 chars


class _GroundednessJudgement(BaseModel):
    is_grounded: bool
    score: int   # 1–5
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to determine if a research agent's output is completely grounded in the raw legal text retrieved from the API.

Raw Retrieval Context:
{retrieval_context}

Research Agent Output:
{research_output}

Evaluate the research output against these criteria:
1. Sourcing: Can every factual point in the research output be traced directly back to the raw retrieval context?
2. Distillation accuracy: Did the research agent accurately summarise the legal text without altering its meaning?
3. Extrapolation: Has the agent added external knowledge or assumed facts not explicitly stated in the context?

Provide your evaluation in strict JSON format exactly like this:
{{
    "is_grounded": true or false,
    "score": <a number from 1 to 5, where 5 is perfectly grounded>,
    "reason": "<A one-sentence explanation of why it passed or failed>"
}}
"""


class ResearchGroundednessMetric(BaseMetric):
    """
    Evaluates whether the research agent's output is grounded in the raw
    retrieval context, with no extrapolation or invented facts.

    retrieval_context comes from LLMTestCase (joined to a single string).
    research_output is passed via the constructor as it is not a standard
    LLMTestCase field.

    Args:
        research_output: The research agent's synthesised output.
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
        # Join and truncate retrieval context to stay within token budget
        context_items = test_case.retrieval_context or []
        kept: list[str] = []
        total = 0
        for item in context_items:
            if total + len(item) > _MAX_CONTEXT_CHARS:
                break
            kept.append(item)
            total += len(item)
        retrieval_context_str = "\n\n".join(kept)

        prompt = _PROMPT_TEMPLATE.format(
            retrieval_context=retrieval_context_str,
            research_output=self.research_output,
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
        return "Research Groundedness"