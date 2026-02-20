"""
Custom metric to measure consistency across multiple responses to the same
question, either from the same LLM (repeatability) or across different LLMs
(cross-model agreement).

Uses Jaccard similarity on key legal terms/phrases to avoid penalising
stylistic differences while catching substantive divergence.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from typing import List
import re


def _extract_terms(text: str) -> set:
    """
    Extract normalised tokens from text, keeping legal references intact.

    Strips markdown formatting and lowercases everything so that
    superficial style differences don't affect the score.
    """
    # Remove markdown formatting
    text = re.sub(r"[*_#>`\[\]()]", " ", text)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Split into tokens (words and number groups)
    tokens = set(re.findall(r"\b\w+\b", text))
    # Remove very common stop words that add noise
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "to", "for", "with", "on", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when",
        "while", "where", "how", "what", "which", "who", "whom",
        "this", "that", "these", "those", "it", "its", "they",
        "them", "their", "we", "our", "you", "your", "he", "she",
        "his", "her", "i", "me", "my",
    }
    return tokens - stop


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


class ConsistencyMetric(BaseMetric):
    """
    Measures how consistent a response is compared to one or more
    reference responses to the same question.

    The score is the mean Jaccard similarity (on extracted terms) between
    the actual output and each reference output.

    Args:
        reference_outputs: Other answers to compare against.
        threshold:         Minimum mean similarity to pass (default 0.4).
    """

    def __init__(
        self,
        reference_outputs: List[str],
        threshold: float = 0.4,
    ):
        self.threshold = threshold
        self.reference_outputs = reference_outputs
        self.score = 0.0
        self.reason = ""
        self.success = False

    async def a_measure(
        self, test_case: LLMTestCase, *args, **kwargs
    ) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_terms = _extract_terms(test_case.actual_output or "")

        similarities = []
        for ref in self.reference_outputs:
            ref_terms = _extract_terms(ref)
            sim = jaccard_similarity(actual_terms, ref_terms)
            similarities.append(sim)

        self.score = (
            sum(similarities) / len(similarities) if similarities else 0.0
        )
        self.success = self.score >= self.threshold
        total_results = len(similarities) + 1  # references + the current response
        self.reason = (
            f"Mean Jaccard similarity: {self.score:.3f} "
            f"(across {total_results} results, "
            f"threshold: {self.threshold})"
        )
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Consistency"
