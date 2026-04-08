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
    analysis: str
    score: int   # 1–5
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to score whether a research agent's output is strictly grounded in the raw legal text retrieved from the API.

Raw Retrieval Context:
{retrieval_context}

Research Agent Output:
{research_output}

Before scoring, explicitly identify:
- Any factual claim in the research output that CANNOT be traced to a specific passage in the retrieval context.
- Any place where the agent has altered the meaning, overstated, or understated what the legal text says.
- Any external knowledge, assumptions, or inferences not supported by the retrieved text.

Then assign a score using this rubric:
1 - Multiple fabricated or unsupported claims; output cannot be trusted.
2 - Several claims lack grounding or meaningfully distort the source text.
3 - Mostly grounded but contains at least one unsupported claim or notable distortion.
4 - Only minor wording differences; all substantive claims traceable to the context.
5 - Every claim is directly and accurately traceable to the retrieval context.

Provide your evaluation in strict JSON format exactly like this:
{{
    "analysis": "<A short paragraph explicitly identifying any ungrounded claims, distortions, or external inferences you found above>",
    "score": <integer 1–5>,
    "reason": "<One sentence citing the specific ungrounded claim or confirming full traceability>"
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
        threshold:       Minimum normalised score to pass (default 0.7).
    """

    def __init__(
        self, research_output: str, model, threshold: float = 0.7
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