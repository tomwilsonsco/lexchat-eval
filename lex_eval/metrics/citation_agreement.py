"""
Citation agreement metric.

Compares the legislation a response cites against the legislation the hand
written reference ("gold") answer for the same question cites.
"""

from __future__ import annotations

from .base import BaseMetric
from ..testcase import LLMTestCase

from .structure import _URL_RE, provision_id_from_url

_LEGISLATION_DOMAIN = "legislation.gov.uk"

# Written at the front of the reason when the reference answer cites nothing,
# so reports/streamlit_report.py::_NON_SCORED_PREFIXES can keep the row out of
# the mean. Nothing was measured, which is not the same as scoring zero.
NO_EXPECTED_CITATIONS_REASON = (
    "No reference answer citations to compare against; nothing measured."
)


def cited_provisions(text: str) -> set[str]:
    """Provision ids (e.g. ``ukpga/2018/12/section/6``) cited in *text*."""
    return {
        provision_id_from_url(url)
        for url in _URL_RE.findall(text or "")
        if _LEGISLATION_DOMAIN in url.lower()
    }


def _is_covered(expected: str, actual: set[str]) -> bool:
    """Whether *expected* is cited in *actual*.

    An Act-level citation (``ukpga/2018/12``) is covered by a citation to any
    provision inside that Act, since citing section 6 of an Act does cite it.
    """
    if expected in actual:
        return True
    is_act_level = len(expected.split("/")) <= 3
    return is_act_level and any(a.startswith(f"{expected}/") for a in actual)


class CitationAgreementMetric(BaseMetric):
    """
    Checks how much of the legislation cited by the reference answer is also
    cited by the response.

    Catches an answer that reaches a plausible conclusion without ever citing
    the provisions the question turns on. It does not check whether the
    response uses them correctly, which is Reference Answer Agreement's job.

    Sections are compared, not just Acts: citing section 3 of an Act when the
    reference cites section 6 of it is a miss.

    Args:
        reference_answer: The reference answer's text for this question.
        threshold:        Minimum share of the reference's citations that the
                          response must also cite (default 0.3).

    The threshold is deliberately low because a reference answer cites
    everything its author consulted, including background provisions a good
    response need not repeat. See docs/metrics.md. Replace this with the
    lawyer's `required_citations` once the reference answers are signed off.
    """

    def __init__(self, reference_answer: str, threshold: float = 0.3) -> None:
        self.reference_answer = reference_answer
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = cited_provisions(self.reference_answer)

        if not expected:
            self.score = 0.0
            self.success = False
            self.reason = NO_EXPECTED_CITATIONS_REASON
            return self.score

        actual = cited_provisions(test_case.actual_output or "")
        covered = {e for e in expected if _is_covered(e, actual)}
        missing = sorted(expected - covered)

        self.score = len(covered) / len(expected)
        self.success = self.score >= self.threshold

        if missing:
            shown = ", ".join(missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            self.reason = (
                f"Cited {len(covered)} of {len(expected)} provisions the "
                f"reference answer cites; missing {shown}{more}."
            )
        else:
            self.reason = (
                f"Cited all {len(expected)} provisions the reference answer cites."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Agreement"
