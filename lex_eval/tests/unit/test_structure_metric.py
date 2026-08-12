"""
Unit tests for ``lex_eval.metrics.structure.MandatoryStructureMetric``'s
heading matching — synthetic Worker output, no DB or LexChat instance needed.
"""

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from lex_eval.metrics.structure import MandatoryStructureMetric

pytestmark = pytest.mark.unit


def _test_case(delegate_output: str) -> LLMTestCase:
    return LLMTestCase(
        input="q",
        actual_output="final answer",
        tools_called=[
            ToolCall(name="delegate_research", input_parameters={}, output=delegate_output)
        ],
    )


def test_prose_use_of_references_does_not_count_as_heading():
    """A missing References *section* must still fail, even when the word
    "references" appears earlier in ordinary prose (real failure mode found
    in captured glm-5.2:cloud output — see structure.py)."""
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
    mistral-large-3 output — see structure.py)."""
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
