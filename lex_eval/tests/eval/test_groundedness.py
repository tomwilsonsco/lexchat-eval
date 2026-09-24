"""
Test the groundedness of LexChat responses.

Two custom single-call metrics cover the two hops from retrieved legal text to
the answer the user sees:

  - ResponseGroundednessMetric : is the final response grounded in research output?
                                 Pass or fail, and a near-verbatim relay passes
                                 without calling the judge at all.
  - ClaimSupportMetric         : are the research output's legal claims traceable
                                 to the retrieval context?

Each uses at most a single LLM call per test case. The judge is configured in
lex_eval/.env.

Whether the response actually answers the question is measured by the
`reference` suite, against the hand written reference answers.
"""

import re

import pytest

from lex_eval.metrics import (
    ClaimSupportMetric,
    ResponseGroundednessMetric,
)
from lex_eval.metrics.structure import _model_words
from lex_eval.utils.collector import attach_metric
from lex_eval.utils.judge import _judge
from lex_eval.utils.test_helpers import (
    agent_visible_context,
    load_records,
    record_id,
    record_to_test_case,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_OUTPUT_CHARS: int = 50

# LexChat returns this in place of a report when its agent loop hits the ReAct
# turn cap (server_py/src/agent/ollama_client.py and openrouter_client.py),
# so the field is populated but holds no research. A report that is nothing
# but this sentinel is treated as no research output at all. One that merely
# contains it had a single step halt among several and is still scored on the
# rest, which is why this matches the sentence rather than the halted flag.
_HALTED_RESEARCH_RE = re.compile(r"\[Research halted:[^\]]*\]", re.IGNORECASE)
# Response Groundedness is a pass/fail verdict, so its score is 0.0 or 1.0 and
# nothing between the two is reachable.
_RESPONSE_GROUNDEDNESS_THRESHOLD: float = 1.0
# Claim Support is a share of claims, not a verdict: at most one unsupported
# claim in five.
_CLAIM_SUPPORT_THRESHOLD: float = 0.8


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

records = load_records(read_only=True)

_skip_no_api_key = pytest.mark.skipif(
    _judge is None,
    reason="Configured judge API key not set (check lex_eval/.env)",
)


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------


def _gate_output_length(
    request,
    record,
    test_case,
    test_name,
    metric_name,
    threshold=_RESPONSE_GROUNDEDNESS_THRESHOLD,
):
    """Fail fast if the output is too short to be meaningful."""
    char_count = len((test_case.actual_output or "").strip())
    if char_count <= _MIN_OUTPUT_CHARS:
        reason = (
            f"Output too short ({char_count} chars ≤ {_MIN_OUTPUT_CHARS}); "
            f"{metric_name} scored 0"
        )
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=threshold,
            passed=False,
            reason=reason,
        )
        return False, reason
    return True, ""


def _gate_retrieval_context(
    request,
    record,
    test_case,
    test_name,
    metric_name,
    threshold=_RESPONSE_GROUNDEDNESS_THRESHOLD,
):
    """Fail fast if no retrieval context was captured."""
    if not test_case.retrieval_context:
        reason = f"No retrieval context captured; {metric_name} scored 0"
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=threshold,
            passed=False,
            reason=reason,
        )
        return False, reason
    return True, ""


def _gate_research_output(
    request,
    record,
    test_name,
    metric_name,
    threshold=_RESPONSE_GROUNDEDNESS_THRESHOLD,
):
    """Fail fast if no research output was captured.

    "No research output" covers an empty field, one holding only LexChat's
    halted-research sentinel, and one holding only its appended instruction
    blocks: none gives the judge anything to score, and asking it to find
    claims in LexChat's own instructions to the model makes it invent them.
    """
    own_words = _model_words(record.get("research_output") or "")
    if not _HALTED_RESEARCH_RE.sub("", own_words).strip():
        reason = f"No research output captured; {metric_name} scored 0"
        attach_metric(
            request,
            record=record,
            test_name=test_name,
            metric_name=metric_name,
            score=0.0,
            threshold=threshold,
            passed=False,
            reason=reason,
        )
        return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@_skip_no_api_key
def test_response_groundedness(request, record):
    """
    The final response must be grounded in the research agent's output.

    Pre-flight gates:
      - Output must be > 50 chars.
      - research_output must be non-empty.
    """
    test_case = record_to_test_case(record)
    # LexChat emits its search-scope disclosure TWICE, in two renderings that
    # do not agree in detail: an agent-facing [SEARCH SCOPE] block in the
    # report, and a reader-facing italic footer on the answer. That confounds
    # this check, which compares the two texts. Stripping neither is worst,
    # because the judge then compares LexChat's two renderings and reports
    # their disagreements as the model misrepresenting its research. Stripping
    # both sides is used here as the least wrong, and it is still wrong on a
    # model that weaves the disclosure into the answer body rather than
    # confining it to the footer, which Gemini does. Removing body prose would
    # need a prose detector, so it is not attempted, and this metric should not
    # be read as a model verdict on a build that emits the disclosure.

    ok, reason = _gate_output_length(
        request, record, test_case, "response_groundedness", "Response Groundedness"
    )
    if not ok:
        pytest.skip(reason)

    ok, reason = _gate_research_output(
        request, record, "response_groundedness", "Response Groundedness"
    )
    if not ok:
        pytest.skip(reason)

    test_case.actual_output = _model_words(test_case.actual_output)

    metric = ResponseGroundednessMetric(
        research_output=_model_words(record["research_output"]),
        model=_judge,
        threshold=_RESPONSE_GROUNDEDNESS_THRESHOLD,
        scope_note=(record.get("research_plan") or {}).get("scope_note"),
        research_mode=record.get("research_mode"),
    )
    # Reset here, not just inside generate(): a near-verbatim response passes
    # without calling the judge at all, so without this the row would wrongly
    # inherit judge_llm/judge_tokens left over from a previous test's call.
    _judge.last_model, _judge.total_usage_tokens = None, None
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="response_groundedness",
        metric_name="Response Groundedness",
        details=getattr(metric, "details", None),
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if getattr(metric, "error", None) else "",
        judge_llm=_judge.last_model,
        judge_tokens=_judge.total_usage_tokens,
    )

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), (
        f"Response Groundedness score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )


@pytest.mark.parametrize("record", records, ids=[record_id(r) for r in records])
@_skip_no_api_key
def test_claim_support(request, record):
    """
    The research agent's legal claims must be traceable to the text it saw.

    Where LexChat summarised a tool result before returning it to the research
    agent, the agent only ever saw the summary, so that is what its report is
    judged against (see ``agent_visible_context``).

    Pre-flight gates:
      - the agent-visible context must be non-empty.
      - research_output must be non-empty.
    """
    test_case = record_to_test_case(record)
    test_case.retrieval_context = agent_visible_context(record)

    ok, reason = _gate_retrieval_context(
        request,
        record,
        test_case,
        "claim_support",
        "Claim Support",
        threshold=_CLAIM_SUPPORT_THRESHOLD,
    )
    if not ok:
        pytest.skip(reason)

    ok, reason = _gate_research_output(
        request,
        record,
        "claim_support",
        "Claim Support",
        threshold=_CLAIM_SUPPORT_THRESHOLD,
    )
    if not ok:
        pytest.skip(reason)

    metric = ClaimSupportMetric(
        research_output=_model_words(record["research_output"]),
        model=_judge,
        threshold=_CLAIM_SUPPORT_THRESHOLD,
    )
    _judge.last_model, _judge.total_usage_tokens = None, None
    metric.measure(test_case)

    attach_metric(
        request,
        record=record,
        test_name="claim_support",
        metric_name="Claim Support",
        details=getattr(metric, "details", None),
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason or "",
        error=str(metric.error) if getattr(metric, "error", None) else "",
        judge_llm=_judge.last_model,
        judge_tokens=_judge.total_usage_tokens,
    )

    if metric.reason.startswith("Not measured:"):
        pytest.skip(metric.reason)
    assert metric.is_successful(), (
        f"Claim Support score {metric.score:.2f} < {metric.threshold}: "
        f"{metric.reason}"
    )
