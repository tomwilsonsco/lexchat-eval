"""
Citation agreement metric.

Compares the legislation a response cites against the legislation the hand
written reference ("gold") answer for the same question cites, and says where
the blame for a miss lies: with the search, or with the model.
"""

from __future__ import annotations

from .base import BaseMetric
from ..testcase import LLMTestCase

from .structure import _URL_RE, _retrieved_legislation_ids, provision_id_from_url

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


def act_of(provision: str) -> str:
    """The Act-level id (``ukpga/2018/12``) a provision id belongs to."""
    return "/".join(provision.split("/")[:3])


def reference_acts(reference: dict) -> set[str]:
    """The Acts a reference answer actually relied on, for attribution.

    Read from the answer's own ``sources_retrieved``, not from the links in its
    prose. A reference answer also links the sources its author looked at and
    set aside, under an "Identified but not retrieved" heading, and blaming a
    response for not citing those is blaming it for the author's reading list.
    Those live in ``sources_discovered`` and are excluded here.
    """
    return {
        s["legislation_id"]
        for s in (reference.get("sources_retrieved") or [])
        if s.get("legislation_id")
    }


# A response shorter than this is a clarification request or a capture gap, not
# an answer. tests/eval/test_reference.py gates every metric in it on the same
# number, imported from here so there is one definition.
MIN_OUTPUT_CHARS = 50


def attributable(reference: dict, actual_output: str) -> bool:
    """Whether blame can honestly be attributed for this response.

    The three conditions are exactly what Citation Agreement needs before it
    will score at all: an answer long enough to be an answer, a reference that
    cites legislation in its prose, and a record of what that reference relied
    on. Sharing them is what stops the dashboard's flag contradicting the
    metric row underneath it, which it did while the two were gated separately.
    """
    return (
        len((actual_output or "").strip()) > MIN_OUTPUT_CHARS
        and bool(cited_provisions(reference.get("final_answer") or ""))
        and bool(reference_acts(reference))
    )


def attribute_missing_acts(
    expected_acts: set[str], actual: set[str], tools_called: list
) -> tuple[list[str], list[str]]:
    """Split the Acts the response never cited by whether retrieval found them.

    *expected_acts* is Act-level, from :func:`reference_acts`. Returns
    ``(never_retrieved, retrieved_not_cited)``, both sorted lists of Act-level
    ids. An Act in the first list never reached the model, so the fault is
    upstream of it. An Act in the second was handed to the model, which then
    left it out.

    Attribution is at Act level, not section level, because that is the
    granularity the retrieval tools are called at. A response that cites the
    wrong section of an Act it did retrieve still counts against the score
    above, it is simply not a retrieval failure.
    """
    uncited = set(expected_acts) - {act_of(a) for a in actual}
    retrieved = _retrieved_legislation_ids(tools_called)
    return sorted(uncited - retrieved), sorted(uncited & retrieved)


def _listed(ids: list[str], limit: int = 3) -> str:
    """Up to *limit* ids, then a count of the rest."""
    shown = ", ".join(ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def _attribution_note(
    never_retrieved: list[str], retrieved_not_cited: list[str]
) -> str:
    """Where the blame for the uncited Acts lies, in two plain sentences.

    Says what the run's own tool calls did, which is all this can see. Whether
    "no search turned it up" means a bad query, a corpus gap or a filter is
    settled outside this metric, in reports/attribution.py.
    """
    note = ""
    if never_retrieved:
        note += f" No search turned up: {_listed(never_retrieved)}."
    if retrieved_not_cited:
        note += f" Turned up by a search but not cited: {_listed(retrieved_not_cited)}."
    return note


class CitationAgreementMetric(BaseMetric):
    """
    Checks how much of the legislation cited by the reference answer is also
    cited by the response.

    Catches an answer that reaches a plausible conclusion without ever citing
    the provisions the question turns on. It does not check whether the
    response uses them correctly, which is Reference Answer Agreement's job.

    The reason also says where the blame for a miss lies. An Act the response
    never cited is reported as either one no search turned up, so the model
    never had it, or one a search did turn up and the answer left out. Only the
    score decides pass or fail, the attribution is there to be read.

    Attribution needs *expected_acts*, the Acts the reference answer relied on
    (:func:`reference_acts`). Without it the reason carries no attribution at
    all, which is deliberate: staying silent is better than blaming a response
    for not citing something the reference author only looked at.

    Sections are compared, not just Acts: citing section 3 of an Act when the
    reference cites section 6 of it is a miss.

    Args:
        reference_answer: The reference answer's text for this question.
        threshold:        Minimum share of the reference's citations that the
                          response must also cite (default 0.3).
        expected_acts:    The Acts the reference answer relied on, for the
                          attribution only. Never affects the score.

    The threshold is deliberately low because a reference answer cites
    everything its author consulted, including background provisions a good
    response need not repeat. See docs/metrics.md. Replace this with the
    lawyer's `required_citations` once the reference answers are signed off.
    """

    def __init__(
        self,
        reference_answer: str,
        threshold: float = 0.3,
        expected_acts: set[str] | None = None,
    ) -> None:
        self.reference_answer = reference_answer
        self.threshold = threshold
        self.expected_acts = expected_acts
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.never_retrieved: list[str] = []
        self.retrieved_not_cited: list[str] = []

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

        if self.expected_acts:
            self.never_retrieved, self.retrieved_not_cited = attribute_missing_acts(
                self.expected_acts, actual, test_case.tools_called or []
            )

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

        self.reason += _attribution_note(self.never_retrieved, self.retrieved_not_cited)

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Agreement"
