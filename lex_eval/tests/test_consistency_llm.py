"""
AI-judge consistency test — checks that the same LLM produces substantively
consistent answers when asked the same question multiple times.

Requires repeated runs (gather_responses.py 2X without --overwrite) AND an OpenAI
API key for the judge model.

Run this suite independently to avoid waiting for groundedness checks:
    python lex_eval/run_evals.py --suite consistency_llm
"""

from __future__ import annotations

import pytest

from lex_eval.metrics.consistency_llm import LLMConsistencyMetric
from lex_eval.utils.collector import attach_metric
from lex_eval.utils.openai_judge import OPENAI_API_KEY as _OPENAI_API_KEY, OpenAIJudge
from lex_eval.utils.test_helpers import group_by_question_and_llm

# ---------------------------------------------------------------------------
# Judge setup
# ---------------------------------------------------------------------------

_judge = OpenAIJudge() if _OPENAI_API_KEY else None

_skip_no_api_key = pytest.mark.skipif(
    _OPENAI_API_KEY is None,
    reason="OPENAI_API_KEY not set (check lex_eval/.env)",
)

# ---------------------------------------------------------------------------
# Parametrize: one test per (question, LLM) group that has ≥2 runs
# ---------------------------------------------------------------------------


def _multi_run_groups():
    """
    Yield pytest.param(records, id=key) for every (question, LLM) group
    that has more than one captured response.
    Each element in the tuple is the full list of records for that group.
    """
    grouped = group_by_question_and_llm()
    cases = []
    for key, records in sorted(grouped.items()):
        if len(records) >= 2:
            cases.append(pytest.param(records, id=key))
    return cases


_groups = _multi_run_groups()


@pytest.mark.consistency_llm
@_skip_no_api_key
@pytest.mark.skipif(
    not _groups,
    reason="No repeated runs found — re-run gather_responses.py",
)
@pytest.mark.parametrize("records", _groups)
def test_consistency_llm(request, records):
    """
    The same LLM should produce substantively consistent answers across
    multiple runs of the same question, as judged by an AI judge.

    One result is recorded per (question, LLM) group (not per individual run).
    The test case for the judge is built from the first run; all other runs
    are passed as reference outputs.
    """
    # Use the first record as the primary test case; compare against the rest
    primary = records[0]
    reference_outputs = [r["actual_output"] for r in records[1:]]

    from lex_eval.utils.test_helpers import record_to_test_case

    test_case = record_to_test_case(primary)

    metric = LLMConsistencyMetric(
        reference_outputs=reference_outputs,
        model=_judge,
        threshold=0.7,
    )
    metric.measure(test_case)

    attach_metric(
        request,
        record=primary,
        test_name="consistency_llm",
        metric_name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
        suite="consistency_llm",
    )

    assert metric.is_successful(), metric.reason
