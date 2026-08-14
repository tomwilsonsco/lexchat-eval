"""
Reference answer agreement metric.

Scores a response against the statements written alongside the hand written
reference ("gold") answer for the same question: how many of them does the
response also state, and does it contradict any.
"""

from __future__ import annotations

import json
import re
from typing import List, Literal

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

# Attempts allowed for the judge to return one label per statement, no more and
# no fewer. See ReferenceAnswerAgreementMetric._label.
_MAX_LABEL_ATTEMPTS = 2


class _Point(BaseModel):
    index: int
    label: Literal["stated", "contradicted", "missing"]
    quote: str


class _AgreementJudgement(BaseModel):
    points: List[_Point]


_PROMPT_TEMPLATE = """You are checking a response to a UK legal question against a list of statements a correct answer has to make. The statements were written by the person who researched the question.

Do NOT use your own knowledge of the law. Do not judge whether the statements are right. Only compare the statements against the response in front of you.

Question:
{input}

Statements:
{statements}

Response Under Test:
{actual_output}

Label each statement by its index:
- "stated" if the response makes the same point, even in different words.
- "contradicted" if the response asserts something incompatible with the statement.
- "missing" if the response neither makes the point nor contradicts it.

Ignore differences of wording, ordering, formatting and level of detail. A response that makes the point in passing has still made it.

For "stated" and "contradicted", quote the words from the RESPONSE UNDER TEST that justify the label, copied exactly. For "missing", leave the quote empty.

Return exactly {n} labels, one per statement, in index order.

Provide your evaluation in strict JSON format exactly like this:
{{
    "points": [
        {{"index": 1, "label": "stated|contradicted|missing", "quote": "<exact words from the response, or empty>"}}
    ]
}}
"""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


class ReferenceAnswerAgreementMetric(BaseMetric):
    """
    Evaluates how many of a question's reference statements the response makes.

    The statements are written once, alongside the hand written reference answer,
    and stored with it. The judge is given that fixed list and labels each entry
    stated, contradicted or missing. It never chooses the list itself, which is
    what makes the metric repeatable: when the judge picked the points on every
    run, 48% of calls disagreed with their own record's usual labelling and the
    denominator wandered between 6 and 9 for the same answer. Labelling a fixed
    list, that fell to 7%. See docs/ai-judge-review.md.

    This is the only metric that compares the response against material a person
    researched, so it is the only one that can catch a response that is faithful
    to its own retrieval but wrong about the law.

    A contradiction fails the metric outright, whatever the score, because a
    confidently wrong statement of law is worse than a missing one. A
    contradiction whose supporting quote is not actually in the response is
    downgraded to a missing point, so an invented quote cannot fail a record.

    Args:
        statements: The reference statements for this question, in order. With
                    the usual 5, the score can only be 0.0, 0.2, 0.4, 0.6, 0.8
                    or 1.0, and the default threshold means at least 3 of the 5.
        model:      A DeepEval-compatible judge model.
        threshold:  Minimum share of the statements the response must state
                    (default 0.6).
    """

    def __init__(
        self, statements: List[str], model, threshold: float = 0.6
    ) -> None:
        self.statements = list(statements)
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        numbered = "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(self.statements)
        )
        prompt = _PROMPT_TEMPLATE.format(
            input=test_case.input or "",
            statements=numbered,
            actual_output=test_case.actual_output or "",
            n=len(self.statements),
        )

        try:
            if not self.statements:
                raise ValueError("no reference statements supplied")
            points = self._label(prompt)
        except Exception as exc:
            self.score = 0.0
            self.success = False
            self.reason = f"Judge error: {exc}"
            return self.score

        return self._score_points(points, test_case.actual_output or "")

    def _label(self, prompt: str) -> List[_Point]:
        """One label per statement, retried once if the judge returns the wrong set.

        A short or padded label list would silently move the denominator, which
        is the failure this metric exists to remove, so it is rejected rather
        than scored. Observed on roughly 1 call in 22, and it has not survived a
        retry, so one is enough.
        """
        expected = list(range(1, len(self.statements) + 1))
        last: Exception | None = None
        for _ in range(_MAX_LABEL_ATTEMPTS):
            result = self.model.generate(prompt, schema=_AgreementJudgement)
            if isinstance(result, _AgreementJudgement):
                points = result.points
            else:
                data = json.loads(str(result))
                points = [_Point(**p) for p in data["points"]]
            if sorted(p.index for p in points) == expected:
                return points
            last = ValueError(
                f"judge returned {len(points)} label(s) for "
                f"{len(self.statements)} statements"
            )
        raise last  # type: ignore[misc]

    def _score_points(self, points: List[_Point], actual_output: str) -> float:
        response = _normalise(actual_output)
        ordered = sorted(points, key=lambda p: p.index)

        stated = [p for p in ordered if p.label == "stated"]
        contradicted = [
            p
            for p in ordered
            if p.label == "contradicted" and _normalise(p.quote) in response
        ]
        unevidenced = sum(
            1
            for p in ordered
            if p.label == "contradicted" and _normalise(p.quote) not in response
        )

        self.score = len(stated) / len(ordered)
        self.success = self.score >= self.threshold and not contradicted

        if contradicted:
            self.reason = (
                "Contradicts the reference answer: "
                f"{self.statements[contradicted[0].index - 1]} "
                f"(states {len(stated)} of {len(ordered)} reference points)."
            )
        else:
            self.reason = (
                f"States {len(stated)} of {len(ordered)} reference points."
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
