"""
Unit tests for ``lex_eval.metrics.structure.MandatoryStructureMetric``'s
heading matching, synthetic Worker output, no DB or LexChat instance needed.
"""

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from lex_eval.metrics.structure import (
    CitationGroundingMetric,
    GenuineGapMetric,
    MandatoryStructureMetric,
    _retrieved_usable_content,
)

pytestmark = pytest.mark.unit


_VALID_REPORT = """### **Summary Answer (BLUF)**
Some answer text.

### **Detailed Analysis**
Analysis text.

### **Jurisdiction & Status**
Applies to Scotland.

### **References**
- [Act](http://legislation.gov.uk/example)
"""


def _test_case(*delegate_outputs: str) -> LLMTestCase:
    return LLMTestCase(
        input="q",
        actual_output="final answer",
        tools_called=[
            ToolCall(
                name="delegate_research", input_parameters={}, output=delegate_output
            )
            for delegate_output in delegate_outputs
        ],
    )


def test_prose_use_of_references_does_not_count_as_heading():
    """A missing References *section* must still fail, even when the word
    "references" appears earlier in ordinary prose (real failure mode found
    in captured glm-5.2:cloud output, see structure.py)."""
    output = """### **Summary Answer (BLUF)**
Some answer text.

### **Detailed Analysis**
Section 21(1) provides that references in ss.76, 77, 78 include references
to the 1978 Act.

### **Jurisdiction & Status**
Applies to Scotland.
"""
    metric = MandatoryStructureMetric(research_mode="legislation_only")
    metric.measure(_test_case(output))
    assert not metric.is_successful()
    assert "References" in metric.reason


def test_real_references_heading_after_prose_use_still_passes():
    output = """### **Summary Answer (BLUF)**
Some answer text.

### **Detailed Analysis**
This section references the 1978 Act throughout.

### **Jurisdiction & Status**
Applies to Scotland.

### **References**
- [Act](http://legislation.gov.uk/example)
"""
    metric = MandatoryStructureMetric(research_mode="legislation_only")
    metric.measure(_test_case(output))
    assert metric.is_successful()


def test_jurisdiction_and_status_spelled_out_passes():
    """Model paraphrasing the prompt's literal "&" as "and" is compliant,
    not a missing-heading failure (real failure mode found in captured
    mistral-large-3 output, see structure.py)."""
    output = """### **Summary Answer (BLUF)**
Some answer text.

### **Detailed Analysis**
Analysis text.

#### **3. Jurisdiction and Status**
Applies to Scotland.

### **References**
- [Act](http://legislation.gov.uk/example)
"""
    metric = MandatoryStructureMetric(research_mode="legislation_only")
    metric.measure(_test_case(output))
    assert metric.is_successful(), metric.reason


def test_deep_research_all_steps_present_passes():
    """A deep-research run has one delegate_research entry per plan step;
    every step's report has the mandatory headings here, so it should pass
    exactly like a single-shot run would."""
    metric = MandatoryStructureMetric(research_mode="legislation_only")
    metric.measure(_test_case(_VALID_REPORT, _VALID_REPORT, _VALID_REPORT))
    assert metric.is_successful(), metric.reason


def test_deep_research_bad_second_step_fails():
    """A bad step 2 must not hide behind a good step 1 (the bug this fix
    addresses: the metric used to only look at the first delegate_research
    output, so a deep-research run's later steps were invisible to it)."""
    broken_step = """### **Summary Answer (BLUF)**
Some answer text.

### **Detailed Analysis**
Analysis text.
"""
    metric = MandatoryStructureMetric(research_mode="legislation_only")
    metric.measure(_test_case(_VALID_REPORT, broken_step))
    assert not metric.is_successful()
    assert "step 2" in metric.reason


class TestCitationGroundingParsesSearchResults:
    """``search_legislation`` returns JSON followed by a plain-text "[NEXT STEP:]"
    hint. A plain json.loads of the whole string raises "Extra data", which
    previously left every search result invisible and made correctly cited Acts
    look fabricated."""

    @staticmethod
    def _case(cited_id: str) -> LLMTestCase:
        search_output = (
            '{"results": [{"legislation_id": "asp/2021/3", '
            '"title": "An Act", "url": "http://www.legislation.gov.uk/asp/2021/3"}], '
            '"total": 1}\n\n'
            '[NEXT STEP: Call search_legislation_sections with the relevant '
            'legislation_id(s) below:\n  - legislation_id: "asp/2021/3"]'
        )
        report = f"See [the Act](http://www.legislation.gov.uk/id/{cited_id})."
        return LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="Worker: search_legislation",
                    input_parameters={},
                    output=search_output,
                ),
                ToolCall(
                    name="delegate_research", input_parameters={}, output=report
                ),
            ],
        )

    def test_act_returned_by_search_is_grounded(self):
        metric = CitationGroundingMetric()
        metric.measure(self._case("asp/2021/3"))
        assert metric.score == 1.0
        assert metric.is_successful()

    def test_act_never_retrieved_is_still_flagged(self):
        metric = CitationGroundingMetric()
        metric.measure(self._case("asp/2000/7"))
        assert metric.score == 0.0
        assert "asp/2000/7" in metric.reason

    def test_non_json_tool_output_is_skipped_not_fatal(self):
        """LEX API errors come back as prose, not JSON."""
        case = LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="Worker: search_legislation",
                    input_parameters={},
                    output='Error executing tool: {"detail": "Internal server error"}',
                ),
                ToolCall(name="delegate_research", input_parameters={}, output="No links."),
            ],
        )
        metric = CitationGroundingMetric()
        metric.measure(case)
        assert metric.score == 1.0


class TestRetrievedUsableContentRejectsToolErrors:
    """A LEX tool failure comes back as non-empty prose ("Error executing
    tool: ..."), which truthiness alone would count as usable retrieval,
    incorrectly excusing GenuineGapMetric's disclosure requirement."""

    def test_error_string_is_not_usable_content(self):
        tool = ToolCall(
            name="Worker: search_legislation_sections",
            input_parameters={},
            output='Error executing tool: {"detail": "Internal server error"}',
        )
        assert _retrieved_usable_content([tool]) is False

    def test_real_content_is_usable(self):
        tool = ToolCall(
            name="Worker: search_legislation_sections",
            input_parameters={},
            output="Section 6: ...",
        )
        assert _retrieved_usable_content([tool]) is True

    def test_genuine_gap_requires_disclosure_after_a_tool_error(self):
        """A step whose only search_legislation_sections call failed must be
        judged as empty retrieval, not excused because the failure text was
        non-empty."""
        report = _VALID_REPORT  # confidently answers, no gap disclosure
        case = LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="delegate_research", input_parameters={}, output=report
                ),
                ToolCall(
                    name="Worker: search_legislation_sections",
                    input_parameters={},
                    output='Error executing tool: {"detail": "Internal server error"}',
                ),
            ],
        )
        metric = GenuineGapMetric(research_mode="legislation_only")
        metric.measure(case)
        assert not metric.is_successful(), metric.reason
