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
    is_relevant: bool
    score: int   # 1–5
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to determine if the provided response directly and usefully answers the user's question.

User Question:
{input}

Generated Response:
{actual_output}

Evaluate the response based on the following criteria:
1. Directness: Does it answer the specific legal question asked?
2. Completeness: Does it address all parts of the user's query?
3. Conciseness: Does it avoid rambling or providing irrelevant legal trivia?

Provide your evaluation in strict JSON format exactly like this:
{{
    "is_relevant": true or false,
    "score": <a number from 1 to 5, where 5 is perfectly relevant>,
    "reason": "<A one-sentence explanation of why it passed or failed>"
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
        threshold: Minimum normalised score to pass (default 0.6).
                   Scores are normalised from 1–5 to 0.0–1.0.
    """

    def __init__(self, model, threshold: float = 0.6) -> None:
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