"""
AI-judge consistency metric.

Uses an LLM to detect substantive contradictions or omissions between
repeated responses to the same question, going beyond surface-level token
overlap (cf. the cosine similarity-based ConsistencyMetric in consistency.py).

The judge is prompted to ignore stylistic differences and focus only on
material factual or legal divergences.

Score:
    The judge picks the single numbered rule that applies to each pair (see
    _PROMPT_TEMPLATE); the score is derived from that rule via _RULE_SCORES,
    not asked for directly. Mean of the derived scores across all
    (actual, reference) pairs, 0.0-1.0 where 1.0 = perfectly consistent.
"""

from __future__ import annotations

import json
from typing import Literal

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class _ConsistencyJudgement(BaseModel):
    matched_rule: Literal[1, 2, 3, 4, 5]
    reason: str


# Score is derived from matched_rule in code, not asked for directly, so the
# judge can't state one rule in its reasoning and a different score in the
# JSON — there is only one number for it to get right.
_RULE_SCORES: dict[int, float] = {1: 0.0, 2: 0.2, 3: 0.4, 4: 0.7, 5: 1.0}


_PROMPT_TEMPLATE = """You are evaluating whether two responses to the same legal question are consistent with each other. Consistency means the responses are similar in both their conclusions AND their scope — not just that they don't contradict each other.

Judge the pair symmetrically: your score must not depend on which response is labelled "1" and which is "2" — swapping the two labels below must not change your score.

Question: {question}

Response 1: {reference}

Response 2: {actual}

Rules — apply the LOWEST matching rule:

1. One response directly contradicts a material fact or legal conclusion stated in the other.
2. One response omits a critical warning, requirement, or legal caveat that the other includes; OR one response asks a clarifying question instead of answering while the other gives a substantive legal answer.
3. One response covers significantly more ground than the other (e.g., discusses additional sections, provisions, or legal concepts the other doesn't mention) such that a user would receive a substantially different impression of the topic. A superset response is NOT automatically consistent.
4. Minor differences in depth or emphasis, but both responses address the same legal provisions and reach the same key conclusions.
5. Both responses address the same provisions and reach the same conclusions and are similar in scope. This includes both responses asking the same kind of clarifying question instead of answering — that is consistent behaviour.

Important: "one response covers everything the other covers, plus much more" should match rule 3, not rule 5. Consistency requires similar scope, not just agreement on shared content.

Respond with a JSON object:
{{
    "matched_rule": <the number, 1-5, of the rule above that applies>,
    "reason": "<brief explanation citing specific scope or content differences>"
}}"""


class LLMConsistencyMetric(BaseMetric):
    """
    Evaluates consistency between repeated responses to the same question
    using an AI judge to detect substantive contradictions or omissions.


    Unlike ConsistencyMetric (Jaccard / token-overlap), this metric
    understands the *meaning* of the responses.


    Args:
        reference_outputs: Other answers to the same question to compare against.
        model: A DeepEval-compatible judge model (e.g. OpenAIJudge()).
        threshold: Minimum mean score to pass (default 0.7).
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

        # Judge failures on individual pairs are kept separate from genuine
        # verdicts so a failed call is excluded from the mean rather than
        # silently averaged in as a real 0.0 (see docs/ai-judge-review.md).
        scores: list[float] = []
        reasons: list[str] = []
        failures: list[str] = []

        for i, ref in enumerate(self.reference_outputs, 1):
            prompt = _PROMPT_TEMPLATE.format(
                question=question,
                reference=ref,
                actual=actual,
            )
            try:
                result = self.model.generate(prompt, schema=_ConsistencyJudgement)
                if isinstance(result, _ConsistencyJudgement):
                    pair_score = _RULE_SCORES[result.matched_rule]
                    pair_reason = result.reason
                else:
                    # Fallback: raw string → parse JSON manually
                    data = json.loads(str(result))
                    pair_score = _RULE_SCORES[int(data["matched_rule"])]
                    pair_reason = data["reason"]
                scores.append(pair_score)
                reasons.append(f"vs ref {i}: {pair_reason}")
            except Exception as exc:
                failures.append(f"vs ref {i}: Judge error: {exc}")

        if not scores:
            # Every pairwise judge call failed — this is a harness failure,
            # not a quality verdict. The "Judge error:" prefix matches what
            # the other judge metrics already write on failure, which
            # reports/streamlit_report.py recognises and excludes from every
            # mean/pass-rate, showing N/A instead.
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Judge error: all {len(failures)} judge call(s) failed "
                "(" + "; ".join(failures) + ")"
            )
            return self.score

        self.score = sum(scores) / len(scores)
        self.success = self.score >= self.threshold
        excluded_note = (
            f" {len(failures)} comparison(s) excluded due to judge error "
            f"(not counted in the mean): {'; '.join(failures)}."
            if failures
            else ""
        )
        self.reason = (
            f"Mean consistency score: {self.score:.3f} "
            f"({len(scores)} comparison(s), threshold: {self.threshold})."
            f"{excluded_note} " + " | ".join(reasons)
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Consistency (AI Judge)"
