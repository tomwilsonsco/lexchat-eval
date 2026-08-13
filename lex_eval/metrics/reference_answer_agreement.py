"""
Reference answer agreement metric.

Scores a response against the hand written reference ("gold") answer for the
same question: how many of the reference answer's main points does the
response also make, and does it contradict any of them.
"""

from __future__ import annotations

import json
import re
from typing import List, Literal

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

_MAX_POINTS = 8


class _Point(BaseModel):
    point: str
    label: Literal["stated", "contradicted", "missing"]
    quote: str


class _AgreementJudgement(BaseModel):
    points: List[_Point]


_PROMPT_TEMPLATE = """You are comparing two answers to the same UK legal question. One is a reference answer researched and written by a person. The other is a response from the system under test.

Your task is to decide, point by point, whether the response says what the reference answer says.

Do NOT use your own knowledge of the law. Do not judge whether the reference answer is right. Only compare the two documents in front of you.

Question:
{input}

Reference Answer:
{reference_answer}

Response Under Test:
{actual_output}

First identify the main legal points the reference answer makes, at most {max_points} of them. Prefer the points a reader would need to have got right: what the provision says, what it requires or permits, who it applies to, and the answer's overall conclusion. Ignore differences of wording, ordering, formatting and level of detail.

Then label each point:
- "stated" if the response makes the same point, even in different words.
- "contradicted" if the response asserts something incompatible with the point.
- "missing" if the response neither makes the point nor contradicts it.

For "stated" and "contradicted", quote the words from the RESPONSE UNDER TEST that justify the label, copied exactly. For "missing", leave the quote empty.

Provide your evaluation in strict JSON format exactly like this:
{{
    "points": [
        {{"point": "<the point, in one sentence>", "label": "stated|contradicted|missing", "quote": "<exact words from the response, or empty>"}}
    ]
}}
"""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


class ReferenceAnswerAgreementMetric(BaseMetric):
    """
    Evaluates how much of a hand written reference answer's substance the
    response reproduces.

    This is the only metric that compares the response against an answer a
    person researched, so it is the only one that can catch a response that is
    faithful to its own retrieval but wrong about the law.

    A contradiction fails the metric outright, whatever the score, because a
    confidently wrong statement of law is worse than a missing one. A
    contradiction whose supporting quote is not actually in the response is
    downgraded to a missing point, so an invented quote cannot fail a record.

    Args:
        reference_answer: The reference answer's text for this question.
        model:            A DeepEval-compatible judge model.
        threshold:        Minimum share of the reference's points the response
                          must state (default 0.6).
    """

    def __init__(
        self, reference_answer: str, model, threshold: float = 0.6
    ) -> None:
        self.reference_answer = reference_answer
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        prompt = _PROMPT_TEMPLATE.format(
            input=test_case.input or "",
            reference_answer=self.reference_answer,
            actual_output=test_case.actual_output or "",
            max_points=_MAX_POINTS,
        )

        try:
            result = self.model.generate(prompt, schema=_AgreementJudgement)
            if isinstance(result, _AgreementJudgement):
                points = result.points
            else:
                data = json.loads(str(result))
                points = [_Point(**p) for p in data["points"]]
            if not points:
                raise ValueError("judge returned no points")
        except Exception as exc:
            self.score = 0.0
            self.success = False
            self.reason = f"Judge error: {exc}"
            return self.score

        return self._score_points(points, test_case.actual_output or "")

    def _score_points(self, points: List[_Point], actual_output: str) -> float:
        response = _normalise(actual_output)

        stated = [p for p in points if p.label == "stated"]
        contradicted = [
            p
            for p in points
            if p.label == "contradicted" and _normalise(p.quote) in response
        ]
        unevidenced = sum(
            1
            for p in points
            if p.label == "contradicted" and _normalise(p.quote) not in response
        )

        self.score = len(stated) / len(points)
        self.success = self.score >= self.threshold and not contradicted

        if contradicted:
            self.reason = (
                f"Contradicts the reference answer: {contradicted[0].point} "
                f"(states {len(stated)} of {len(points)} reference points)."
            )
        else:
            self.reason = (
                f"States {len(stated)} of {len(points)} reference points."
            )
        if unevidenced:
            self.reason += (
                f" {unevidenced} further contradiction(s) counted as missing "
                "because the quoted words are not in the response."
            )

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Reference Answer Agreement"
