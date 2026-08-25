"""
Unit tests for the per-step scoping shared by ``StepCompletionMetric`` and
``ReportIntegrationMetric``: a step's report must cite one of the Acts its
own tool calls retrieved, not merely cite some legislation.gov.uk URL, or a
step could get credit for citing a sibling step's Act. No DB, judge, or
LexChat instance needed (ReportIntegrationMetric's judge stub asserts it is
never invoked for an out-of-scope step).
"""

import pytest
from lex_eval.testcase import LLMTestCase, ToolCall

from lex_eval.metrics.report_integration import ReportIntegrationMetric
from lex_eval.metrics.structure import StepCompletionMetric

pytestmark = pytest.mark.unit


def _search_legislation(legislation_id: str) -> ToolCall:
    return ToolCall(
        name="Worker: search_legislation",
        input_parameters={},
        output=(
            '{"results": [{"legislation_id": "%s", "title": "An Act", '
            '"url": "http://www.legislation.gov.uk/%s"}], "total": 1}'
            % (legislation_id, legislation_id)
        ),
    )


def _search_sections(legislation_id: str) -> ToolCall:
    return ToolCall(
        name="Worker: search_legislation_sections",
        input_parameters={"legislation_id": legislation_id},
        output=f"Section 1 of {legislation_id} says...",
    )


def _step(delegate_report: str, *tools: ToolCall) -> list:
    return [
        ToolCall(name="delegate_research", input_parameters={}, output=delegate_report)
    ] + list(tools)


class _JudgeNotInvoked:
    def generate(self, prompt, schema=None):
        raise AssertionError("Judge should not be invoked for an out-of-scope step")


class TestStepCompletionRequiresOwnCitation:
    def test_citing_a_sibling_steps_act_still_fails(self):
        """Step 1 retrieves Act A and cites nothing of its own; step 2
        retrieves Act B. Step 1's report links to Act B (a sibling's Act,
        not its own), which must not count as completing step 1."""
        step1_report = "See [the Act](http://www.legislation.gov.uk/ukpga/2000/7)."
        tools = _step(
            step1_report,
            _search_legislation("asp/2021/3"),
            _search_sections("asp/2021/3"),
        ) + _step(
            "some other finding",
            _search_legislation("ukpga/2000/7"),
            _search_sections("ukpga/2000/7"),
        )
        case = LLMTestCase(input="q", actual_output="final", tools_called=tools)
        metric = StepCompletionMetric()
        metric.measure(case)
        assert not metric.is_successful()
        assert "1" in metric.reason

    def test_citing_own_act_passes(self):
        step1_report = "See [the Act](http://www.legislation.gov.uk/asp/2021/3)."
        tools = _step(
            step1_report,
            _search_legislation("asp/2021/3"),
            _search_sections("asp/2021/3"),
        )
        case = LLMTestCase(input="q", actual_output="final", tools_called=tools)
        metric = StepCompletionMetric()
        metric.measure(case)
        assert metric.is_successful(), metric.reason


class TestReportIntegrationScopingRequiresOwnCitation:
    def test_step_citing_only_a_sibling_act_is_out_of_scope(self):
        """Each step cites the other step's Act, not its own: neither has a
        finding of its own to check, so the judge must never be invoked."""
        step1_report = "See [the Act](http://www.legislation.gov.uk/ukpga/2000/7)."
        step2_report = "See [the Act](http://www.legislation.gov.uk/asp/2021/3)."
        tools = _step(
            step1_report,
            _search_legislation("asp/2021/3"),
            _search_sections("asp/2021/3"),
        ) + _step(
            step2_report,
            _search_legislation("ukpga/2000/7"),
            _search_sections("ukpga/2000/7"),
        )
        case = LLMTestCase(
            input="q",
            actual_output="final answer with no citations",
            tools_called=tools,
        )
        metric = ReportIntegrationMetric(model=_JudgeNotInvoked())
        metric.measure(case)
        assert metric.is_successful()
        assert "nothing to check" in metric.reason
