"""
Unit tests for ``lex_eval.metrics.tool_usage``'s order check, in particular
the ``chat_mode == "deep_research"`` behaviour: per-step segmentation and
dropping the no-revisit ("loop-back") rule.

The loop-back rule is real and correct for single-shot ``research`` mode, but
was measured against six real deep-research responses to fail 13 of 25 real
step-segments, because a single deep-research step routinely covers more than
one Act, discovering and retrieving each in whatever order the Worker finds
useful. These tests fix that behaviour in place with synthetic tool
sequences, no DB or LexChat instance needed.
"""

import pytest

from lex_eval.metrics.tool_usage import ToolUsageMetric, _check_tool_order

pytestmark = pytest.mark.unit

_DELEGATE = "delegate_research"
_SEARCH = "Worker: search_legislation"
_SECTIONS = "Worker: search_legislation_sections"
_FALLBACK = "Worker: get_legislation_text"


def test_single_shot_loop_back_still_fails():
    """Unchanged behaviour for ordinary single-shot research: revisiting
    search_legislation after search_legislation_sections has started is a
    violation, chat_mode defaults to "research"."""
    sequence = [_DELEGATE, _SEARCH, _SECTIONS, _SEARCH, _SECTIONS]
    ok, detail = _check_tool_order(sequence, "legislation_only")
    assert not ok, detail


def test_deep_research_step_with_multi_act_loop_back_passes():
    """The exact real-world shape found in gathered deep-research data: one
    step searches Act A's sections, then discovers and retrieves Act B, all
    within a single plan step. This must not be flagged, chat_mode=deep_research
    checks first-occurrence order per step, not the no-revisit rule."""
    sequence = [
        _DELEGATE,
        _SEARCH,
        _SECTIONS,
        _SEARCH,  # a second Act, discovered after the first Act's sections
        _SECTIONS,
    ]
    ok, detail = _check_tool_order(sequence, "legislation_only", chat_mode="deep_research")
    assert ok, detail


def test_deep_research_step_skipping_discovery_still_fails():
    """A step that jumps straight to retrieval before any discovery is still
    a real violation in deep-research mode, first-occurrence order is
    relaxed on revisits, not dropped entirely."""
    sequence = [_DELEGATE, _SECTIONS, _SEARCH]
    ok, detail = _check_tool_order(sequence, "legislation_only", chat_mode="deep_research")
    assert not ok
    assert "step 1" in detail


def test_deep_research_step_with_no_discovery_at_all_still_fails():
    """A step whose Worker tools are only search_legislation_sections, with
    no search_legislation call anywhere in that step, has nothing to compare
    an index against and passed trivially before require_prerequisites was
    added. Global presence (search_legislation used somewhere in the whole
    run) doesn't cover this: it's checked per plan step here."""
    sequence = [_DELEGATE, _SECTIONS]
    ok, detail = _check_tool_order(sequence, "legislation_only", chat_mode="deep_research")
    assert not ok
    assert "step 1" in detail


def test_deep_research_multiple_steps_each_checked_independently():
    """Step 1 is fine; step 2 skips discovery. The second step's violation
    must be caught even though the first step was clean."""
    sequence = [
        _DELEGATE,
        _SEARCH,
        _SECTIONS,
        _DELEGATE,
        _SECTIONS,  # step 2: retrieval with no discovery first
        _SEARCH,
    ]
    ok, detail = _check_tool_order(sequence, "legislation_only", chat_mode="deep_research")
    assert not ok
    assert "step 2" in detail


def test_tool_usage_metric_threads_chat_mode_through():
    """ToolUsageMetric itself (not just the module-level helper) must pass a
    multi-Act deep-research step that the plain "research" mode would fail."""
    sequence = [_DELEGATE, _SEARCH, _SECTIONS, _SEARCH, _SECTIONS, _FALLBACK]

    deep_research_metric = ToolUsageMetric(
        research_mode="legislation_only",
        tool_sequence=sequence,
        chat_mode="deep_research",
    )
    research_metric = ToolUsageMetric(
        research_mode="legislation_only",
        tool_sequence=sequence,
        chat_mode="research",
    )

    class _FakeToolCall:
        def __init__(self, name):
            self.name = name
            self.output = ""

    class _FakeTestCase:
        tools_called = [_FakeToolCall(t) for t in sequence]

    deep_research_metric.measure(_FakeTestCase())
    research_metric.measure(_FakeTestCase())

    assert deep_research_metric.is_successful(), deep_research_metric.reason
    assert not research_metric.is_successful()
