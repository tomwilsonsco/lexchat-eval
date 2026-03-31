"""
Custom metric to measure consistency across multiple responses to the same
question, either from the same LLM (repeatability) or across different LLMs
(cross-model agreement).

Uses TF vectorisation (no IDF) with cosine similarity. Skipping IDF ensures
that shared legal terminology is not down-weighted when comparing a small
number of responses, giving more meaningful scores.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np
import re


def _preprocess(text: str) -> str:
    """Strip markdown formatting and normalise whitespace."""
    text = re.sub(r"[*_#>`\[\]()]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _vectorize(texts: List[str]):
    """
    Fit a TF vectorizer on *texts*, falling back to no stop-word removal
    if all tokens in a document are stop words (which would otherwise raise
    a ValueError).  Returns None if vectorization is not possible.
    """
    for stop_words in ("english", None):
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            use_idf=False,
            sublinear_tf=True,
        )
        try:
            return vectorizer.fit_transform(texts)
        except ValueError:
            continue
    return None


class ConsistencyMetric(BaseMetric):
    """
    Measures how consistent a response is compared to one or more
    reference responses to the same question.

    Responses are vectorised with TF (no IDF) so that shared legal
    terminology retains its full weight rather than being penalised for
    appearing across the small comparison corpus. Cosine similarity then
    captures directional agreement independent of response length.

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

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual = _preprocess(test_case.actual_output or "")
        refs = [_preprocess(r) for r in self.reference_outputs]

        if not actual:
            self.score = 0.0
            self.success = False
            self.reason = "Actual output is empty."
            return self.score

        if not refs:
            self.score = 0.0
            self.success = False
            self.reason = "No reference outputs provided."
            return self.score

        # Vectorise with TF only (use_idf=False) so shared domain terms keep
        # their weight. Bigrams capture legal phrases like "good faith".
        all_texts = [actual] + refs
        tfidf_matrix = _vectorize(all_texts)

        if tfidf_matrix is None:
            self.score = 0.0
            self.success = False
            self.reason = "Could not vectorize responses (possibly empty inputs)."
            return self.score

        actual_vec = tfidf_matrix[0]
        ref_vecs = tfidf_matrix[1:]

        sims = cosine_similarity(actual_vec, ref_vecs)[0]
        self.score = float(np.mean(sims))
        self.success = self.score >= self.threshold
        self.reason = (
            f"Mean cosine similarity: {self.score:.3f} "
            f"(across {len(all_texts)} responses, "
            f"threshold: {self.threshold})"
        )
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Consistency (Cosine)"
