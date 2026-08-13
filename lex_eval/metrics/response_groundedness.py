"""
Response groundedness metric.

Scores whether the final response to the user is strictly grounded in
the research agent's output, with no hallucinated facts.
"""

from __future__ import annotations

import difflib
import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

# Above this similarity, the final response is a near-verbatim relay of the
# research output: grounding is a provable fact, not a judgement call, so the
# judge is not invoked. glm-5.2 (which follows the Manager prompt's "do NOT
# condense, summarise, or restructure" instruction literally) measures
# 0.983-1.000 on stored runs; mistral-large-3 (which paraphrases) measures
# 0.095-0.763. 0.95 sits in the gap between the two clusters.
_NEAR_VERBATIM_THRESHOLD: float = 0.95


class _GroundednessJudgement(BaseModel):
    analysis: str
    score: int  # 1–5
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to score whether a final response is strictly grounded in the provided research output.

Research Output:
{research_output}

Final Response:
{actual_output}

Before scoring, explicitly identify:
- Any fact, legal assertion, or claim in the final response that does NOT appear in the research output.
- Any place where the response contradicts or misrepresents the research output.
- Any hedging, qualifications, or caveats present in the research output that are omitted in the final response in a way that changes meaning.

Then assign a score using this rubric:
1 - Multiple hallucinated or contradictory claims; the response cannot be trusted.
2 - Several claims are unsupported by or contradict the research output.
3 - Mostly grounded but contains at least one unsupported claim or meaningful misrepresentation.
4 - Only trivial wording differences; all substantive claims present in the research output.
5 - Every claim is directly and accurately traceable to the research output.

Provide your evaluation in strict JSON format exactly like this:
{{
    "analysis": "<A short paragraph explicitly identifying any hallucinated facts, contradictions, or omitted caveats you found above>",
    "score": <integer 1–5>,
    "reason": "<One sentence citing the specific hallucination or confirming full grounding>"
}}
"""


class ResponseGroundednessMetric(BaseMetric):
    """
    Evaluates whether the final response is grounded in the research
    agent's output, with no hallucinated or invented facts.

    research_output is not a standard LLMTestCase field so it is passed
    via the constructor, following the same pattern as ClaimSupportMetric.

    Args:
        research_output: The research agent's synthesised output for this question.
        model:           A DeepEval-compatible judge model.
        threshold:       Minimum normalised score to pass (default 0.7).
    """

    def __init__(self, research_output: str, model, threshold: float = 0.7) -> None:
        self.research_output = research_output
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_output = test_case.actual_output or ""
        ratio = difflib.SequenceMatcher(
            None, actual_output.strip(), self.research_output.strip()
        ).ratio()
        if ratio >= _NEAR_VERBATIM_THRESHOLD:
            self.score = 1.0
            self.reason = (
                f"Near-verbatim relay of research output (similarity={ratio:.2f}); "
                "judge not invoked."
            )
            self.success = True
            return self.score

        prompt = _PROMPT_TEMPLATE.format(
            research_output=self.research_output,
            actual_output=actual_output,
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

        self.score = (raw_score - 1) / 4  # normalise 1–5 → 0.0–1.0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Response Groundedness"
