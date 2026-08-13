"""
Unit tests for the Claim Support metric, synthetic claims only, no DB, judge or
LexChat instance needed.
"""

import pytest
from deepeval.test_case import LLMTestCase

from lex_eval.metrics.claim_support import (
    ClaimSupportMetric,
    _Claim,
    _ClaimSupportJudgement,
)
from lex_eval.utils.test_helpers import agent_visible_context

pytestmark = pytest.mark.unit

_CONTEXT = [
    "Section 6: The controller in relation to personal data is the person who "
    "determines the purposes and means of the processing.",
    "Section 3: Terms relating to the processing of personal data have the "
    "meaning given in this Part.",
]


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        input="q", actual_output="answer", retrieval_context=list(_CONTEXT)
    )


class _StubJudge:
    """A model stub returning fixed claims, so no judge call is made."""

    def __init__(self, claims: list[_Claim]) -> None:
        self._claims = claims

    def generate(self, prompt, schema=None):
        return _ClaimSupportJudgement(claims=self._claims)


class _FailingJudge:
    def generate(self, prompt, schema=None):
        raise RuntimeError("boom")


def _supported(claim: str, quote: str) -> _Claim:
    return _Claim(claim=claim, label="supported", quote=quote)


def _unsupported(claim: str) -> _Claim:
    return _Claim(claim=claim, label="unsupported", quote="")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_every_claim_supported_scores_full_marks():
    claims = [
        _supported("The controller determines the purposes of processing.",
                   "determines the purposes and means of the processing"),
        _supported("Section 3 defines processing terms.",
                   "Terms relating to the processing of personal data"),
    ]
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge(claims))
    metric.measure(_test_case())

    assert metric.score == 1.0
    assert metric.is_successful()
    assert "Traced 2 of 2 claims" in metric.reason


def test_one_unsupported_claim_in_four_fails_the_threshold():
    claims = [
        _supported("The controller determines the purposes of processing.",
                   "determines the purposes and means of the processing"),
        _supported("Section 3 defines processing terms.",
                   "Terms relating to the processing of personal data"),
        _supported("The controller is the person who decides how data is used.",
                   "the person who determines the purposes"),
        _unsupported("Section 209 concerns intelligence services processing."),
    ]
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge(claims))
    metric.measure(_test_case())

    assert metric.score == 0.75
    assert not metric.is_successful()
    # The reason names the claim, so a reviewer can act on it without re-reading
    # the whole report.
    assert "Section 209" in metric.reason


def test_quote_not_in_retrieval_context_does_not_earn_support():
    """A judge that invents its own evidence cannot pass a record."""
    claims = [
        _supported("The controller determines the purposes of processing.",
                   "determines the purposes and means of the processing"),
        _supported("The maximum fine is 20 million euros.",
                   "the maximum penalty is 20 million euros"),
    ]
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge(claims))
    metric.measure(_test_case())

    assert metric.score == 0.5
    assert not metric.is_successful()
    assert "1 further claim(s) counted unsupported" in metric.reason


def test_quote_matching_ignores_whitespace_and_case():
    claims = [
        _supported("The controller determines the purposes of processing.",
                   "DETERMINES   the purposes\nand means of the processing"),
    ]
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge(claims))
    metric.measure(_test_case())

    assert metric.score == 1.0


def test_a_quote_trimmed_mid_sentence_still_counts():
    """
    Judges copy faithfully but cut the passage short. Measured on real runs,
    39 of 42 rejected quotes were genuine text trimmed like this.
    """
    claims = [
        _supported("The controller determines the purposes of processing.",
                   "The controller in relation to personal data is the person who "
                   "determines the purposes and means of the proces"),
    ]
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge(claims))
    metric.measure(_test_case())

    assert metric.score == 1.0


def test_a_quote_with_different_punctuation_still_counts():
    claims = [
        _supported("The controller determines the purposes of processing.",
                   "Section 6 -- the controller, in relation to personal data, is the person"),
    ]
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge(claims))
    metric.measure(_test_case())

    assert metric.score == 1.0


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_judge_error_is_not_recorded_as_a_fabrication_verdict():
    """
    The reason must carry the 'Judge error:' prefix the dashboard's
    _NON_SCORED_PREFIXES uses to keep infrastructure failures out of the mean.
    """
    metric = ClaimSupportMetric(research_output="report", model=_FailingJudge())
    metric.measure(_test_case())

    assert metric.score == 0.0
    assert not metric.is_successful()
    assert metric.reason.startswith("Judge error:")


def test_judge_returning_no_claims_is_a_judge_error():
    metric = ClaimSupportMetric(research_output="report", model=_StubJudge([]))
    metric.measure(_test_case())

    assert metric.reason.startswith("Judge error:")


# ---------------------------------------------------------------------------
# Which text the agent actually saw
# ---------------------------------------------------------------------------

_RAW = ["the full 400,000 character Act text"]
_TOOLS = [
    {"name": "delegate_research", "output": "the finished report"},
    {"name": "Worker: search_legislation_sections", "output": "summary of section 6"},
    {"name": "Worker: get_legislation_text", "output": "summary of the Act"},
]


def test_unsummarised_run_is_judged_against_the_raw_retrieval():
    record = {"summarisation_used": False, "retrieval_context": _RAW, "tools_called": _TOOLS}
    assert agent_visible_context(record) == _RAW


def test_summarised_run_is_judged_against_what_the_agent_saw():
    """The agent never saw the full Act, so the full Act must not be judged against."""
    record = {"summarisation_used": True, "retrieval_context": _RAW, "tools_called": _TOOLS}
    assert agent_visible_context(record) == [
        "summary of section 6",
        "summary of the Act",
    ]


def test_the_report_itself_is_never_used_as_context():
    """delegate_research's output is the report; including it would let the
    report support its own claims."""
    record = {"summarisation_used": True, "retrieval_context": _RAW, "tools_called": _TOOLS}
    assert "the finished report" not in agent_visible_context(record)


def test_summarised_run_with_no_captured_tool_output_falls_back_to_raw():
    record = {
        "summarisation_used": True,
        "retrieval_context": _RAW,
        "tools_called": [{"name": "Worker: search_legislation", "output": "  "}],
    }
    assert agent_visible_context(record) == _RAW


# ---------------------------------------------------------------------------
# Judge schema
# ---------------------------------------------------------------------------


def test_strict_schema_sets_additional_properties_false_everywhere():
    """
    OpenAI models reject a strict schema unless every object, including nested
    definitions, sets additionalProperties: false. Pydantic does not emit it.
    """
    from lex_eval.utils.judge import OpenRouterJudge

    schema = OpenRouterJudge._strict_schema(_ClaimSupportJudgement)

    def objects_missing_flag(node, path="root"):
        missing = []
        if isinstance(node, dict):
            if node.get("type") == "object" and node.get("additionalProperties") is not False:
                missing.append(path)
            for key, value in node.items():
                missing += objects_missing_flag(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                missing += objects_missing_flag(value, f"{path}[{i}]")
        return missing

    assert objects_missing_flag(schema) == []
    # the nested claim definition is the one Pydantic nests under $defs
    assert schema["$defs"]["_Claim"]["additionalProperties"] is False
