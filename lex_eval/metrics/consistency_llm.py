"""
AI-judge consistency metric.

Uses an LLM to detect substantive contradictions or omissions between
repeated responses to the same question, going beyond surface-level token
overlap (cf. the Jaccard-based ConsistencyMetric in consistency.py).

The judge is prompted to ignore stylistic differences and focus only on
material factual or legal divergences.

Score:
    Mean judge score across all (actual, reference) pairs.
    Each pair is scored 0.0–1.0 where 1.0 = perfectly consistent.
"""

from __future__ import annotations

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class _ConsistencyJudgement(BaseModel):
    score: float
    reason: str


_PROMPT_TEMPLATE = """You are a legal consistency auditor. Compare two responses to the same legal question for substantive contradictions or critical omissions.

Question: {question}

Response A (reference): {reference}

Response B (to evaluate): {actual}

Criteria:
1. Ignore stylistic differences (tone, length, formatting, citation style).
2. Score 0.0 if Response B directly contradicts a material fact or legal conclusion in Response A.
3. Score 0.5 if Response B omits a critical warning, requirement, or legal caveat present in Response A.
4. Score 1.0 if both responses are substantively consistent (same legal conclusions and key facts).
5. Intermediate scores are acceptable for partial omissions or minor inconsistencies.

Respond with a JSON object:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation>"
}}"""


class LLMConsistencyMetric(BaseMetric):
    """
    Evaluates consistency between repeated responses to the same question
    using an AI judge to detect substantive contradictions or omissions.

    Unlike ConsistencyMetric (Jaccard / token-overlap), this metric
    understands the *meaning* of the responses.

    Args:
        reference_outputs: Other answers to the same question to compare against.
        model:             A DeepEval-compatible judge model (e.g. OpenAIJudge()).
        threshold:         Minimum mean score to pass (default 0.7).
    """

    def __init__(
        self,
        reference_outputs: list[str],
        model,
        threshold: float = 0.7,
    ) -> None:
        self.threshold = threshold
        self.reference_outputs = reference_outputs
        self.model = model
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if not self.reference_outputs:
            self.score = 1.0
            self.success = True
            self.reason = "No reference outputs to compare against."
            return self.score

        question = test_case.input or ""
        actual = test_case.actual_output or ""

        scores: list[float] = []
        reasons: list[str] = []

        for i, ref in enumerate(self.reference_outputs, 1):
            prompt = _PROMPT_TEMPLATE.format(
                question=question,
                reference=ref,
                actual=actual,
            )
            try:
                result = self.model.generate(prompt, schema=_ConsistencyJudgement)
                if isinstance(result, _ConsistencyJudgement):
                    pair_score = float(result.score)
                    pair_reason = result.reason
                else:
                    # Fallback: raw string → parse JSON manually
                    data = json.loads(str(result))
                    pair_score = float(data["score"])
                    pair_reason = data["reason"]
            except Exception as exc:
                pair_score = 0.0
                pair_reason = f"Judge error: {exc}"

            scores.append(pair_score)
            reasons.append(f"vs ref {i}: {pair_reason}")

        self.score = sum(scores) / len(scores)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Mean consistency score: {self.score:.3f} "
            f"({len(scores)} comparison(s), threshold: {self.threshold}). "
            + " | ".join(reasons)
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Consistency (AI Judge)"
