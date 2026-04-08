"""
Legal answer relevancy metric.

Scores how directly and usefully the final response answers the user's
legal question. Uses a single LLM call with a domain-specific prompt
rather than deepeval's multi-step AnswerRelevancyMetric.
"""

from __future__ import annotations

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class _RelevancyJudgement(BaseModel):
    analysis: str
    score: int   # 1–5
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to score how directly and usefully the response answers the user's legal question.

User Question:
{input}

Generated Response:
{actual_output}

Before scoring, identify:
- Parts of the question that are NOT answered or are answered only vaguely.
- Any content in the response that is off-topic or does not contribute to answering the question.
- Aspects of the question that are addressed incompletely.

Then assign a score using this rubric:
1 - Fails to answer the question; response is off-topic or addresses a different question entirely.
2 - Partially answers but misses the main point or omits critical aspects of the question.
3 - Answers the main question but is incomplete, vague, or includes significant irrelevant content.
4 - Answers the question well with only minor gaps or minor irrelevant content.
5 - Answers the question completely, directly, and without unnecessary waffle.

Provide your evaluation in strict JSON format exactly like this:
{{
    "analysis": "<A short paragraph explicitly identifying any unanswered parts, off-topic content, or incompleteness you found above>",
    "score": <integer 1–5>,
    "reason": "<One sentence citing the specific gap or strength that determined the score>"
}}
"""


class LegalAnswerRelevancyMetric(BaseMetric):
    """
    Evaluates whether the final response directly and usefully answers
    the user's legal question.

    Uses a single domain-specific LLM call rather than deepeval's
    multi-step AnswerRelevancyMetric.

    Args:
        model:     A DeepEval-compatible judge model (OpenAIJudge or GeminiJudge).
        threshold: Minimum normalised score to pass (default 0.7).
                   Scores are normalised from 1–5 to 0.0–1.0.
    """

    def __init__(self, model, threshold: float = 0.7) -> None:
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        prompt = _PROMPT_TEMPLATE.format(
            input=test_case.input or "",
            actual_output=test_case.actual_output or "",
        )
        try:
            result = self.model.generate(prompt, schema=_RelevancyJudgement)
            if isinstance(result, _RelevancyJudgement):
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
        return "Answer Relevancy"