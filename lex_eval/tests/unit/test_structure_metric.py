"""
Unit tests for ``lex_eval.metrics.structure.MandatoryStructureMetric``'s
heading matching, synthetic Worker output, no DB or LexChat instance needed.
"""

import json

import pytest
from lex_eval.testcase import LLMTestCase, ToolCall

from lex_eval.metrics.structure import (
    CitationGroundingMetric,
    CitationReadMetric,
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
            "[NEXT STEP: Call search_legislation_sections with the relevant "
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
                ToolCall(name="delegate_research", input_parameters={}, output=report),
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
                ToolCall(
                    name="delegate_research", input_parameters={}, output="No links."
                ),
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
                ToolCall(name="delegate_research", input_parameters={}, output=report),
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


class TestCitationReadRequiresTextNotJustATitle:
    """Citation Grounding accepts an Act that merely turned up in a
    ``search_legislation`` results list, which returns titles and links but no
    legal text. Citation Read is the check that the Worker actually opened it."""

    _SEARCH_OUTPUT = (
        '{"results": [{"legislation_id": "ssi/2015/99", '
        '"title": "Commencement Order", '
        '"url": "http://www.legislation.gov.uk/ssi/2015/99"}], "total": 1}\n\n'
        "[NEXT STEP: Call search_legislation_sections]"
    )

    @staticmethod
    def _case(report: str, *tools: ToolCall) -> LLMTestCase:
        return LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                *tools,
                ToolCall(name="delegate_research", input_parameters={}, output=report),
            ],
        )

    @staticmethod
    def _cite(*legislation_ids: str) -> str:
        return " ".join(
            f"See [it](http://www.legislation.gov.uk/id/{lid})."
            for lid in legislation_ids
        )

    def test_act_whose_sections_were_read_passes(self):
        tool = ToolCall(
            name="Worker: search_legislation_sections",
            input_parameters={"legislation_id": "ssi/2015/99"},
            output='[{"text": "Section 1) This Order comes into force..."}]',
        )
        metric = CitationReadMetric()
        metric.measure(self._case(self._cite("ssi/2015/99"), tool))
        assert metric.score == 1.0
        assert metric.is_successful()

    def test_act_seen_only_in_a_search_result_fails(self):
        tool = ToolCall(
            name="Worker: search_legislation",
            input_parameters={},
            output=self._SEARCH_OUTPUT,
        )
        metric = CitationReadMetric()
        metric.measure(self._case(self._cite("ssi/2015/99"), tool))
        assert metric.score == 0.0
        assert "ssi/2015/99" in metric.reason
        # ...where Citation Grounding, seeing the same run, is satisfied.
        grounding = CitationGroundingMetric()
        grounding.measure(self._case(self._cite("ssi/2015/99"), tool))
        assert grounding.score == 1.0

    def test_failed_text_retrieval_does_not_count_as_read(self):
        tool = ToolCall(
            name="Worker: get_legislation_text",
            input_parameters={"legislation_id": "ssi/2015/99"},
            output='Error executing tool: {"detail": "Internal server error"}',
        )
        metric = CitationReadMetric()
        metric.measure(self._case(self._cite("ssi/2015/99"), tool))
        assert metric.score == 0.0

    def test_score_is_the_fraction_of_cited_acts_read(self):
        tools = [
            ToolCall(
                name="Worker: search_legislation_sections",
                input_parameters={"legislation_id": lid},
                output='[{"text": "Section 1) ..."}]',
            )
            for lid in ("asp/2015/1", "asp/2014/9", "ukpga/1978/29")
        ]
        report = self._cite("asp/2015/1", "asp/2014/9", "ukpga/1978/29", "ssi/2015/99")
        metric = CitationReadMetric()
        metric.measure(self._case(report, *tools))
        assert metric.score == 0.75
        assert not metric.is_successful()

    def test_case_law_citations_are_ignored(self):
        """Case law links have no legislation retrieval to check them against."""
        report = (
            "See [the case]"
            "(https://caselaw.nationalarchives.gov.uk/ewca/civ/2010/123)."
        )
        metric = CitationReadMetric()
        metric.measure(self._case(report))
        assert metric.score == 1.0
        assert "nothing to check" in metric.reason

    def test_no_citations_at_all_scores_one(self):
        metric = CitationReadMetric()
        metric.measure(self._case("No links in this report."))
        assert metric.score == 1.0

    def test_no_delegate_research_call_is_not_a_quality_verdict(self):
        """The reason must carry the prefix streamlit_report.py's
        _NON_SCORED_PREFIXES uses to keep the row out of the mean."""
        case = LLMTestCase(input="q", actual_output="final answer", tools_called=[])
        metric = CitationReadMetric()
        metric.measure(case)
        assert metric.score == 0.0
        assert metric.reason.startswith("No 'delegate_research' tool call found;")


class TestCitationReadSiblingStepDiagnostic:
    """In a deep research run a step can cite an Act a *sibling* step read but
    it did not, i.e. assert ahead of its own evidence. Reported in the reason,
    never scored, since a step's References section lists Acts it did not
    read."""

    @staticmethod
    def _two_step_case() -> LLMTestCase:
        """Step 1 reads only asp/2015/1 but cites ssi/2015/99; step 2 reads
        ssi/2015/99."""

        def _read(lid):
            return ToolCall(
                name="Worker: search_legislation_sections",
                input_parameters={"legislation_id": lid},
                output='[{"text": "Section 1) ..."}]',
            )

        def _cite(*lids):
            return " ".join(
                f"[x](http://www.legislation.gov.uk/id/{lid})" for lid in lids
            )

        return LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="delegate_research",
                    input_parameters={},
                    output=_cite("asp/2015/1", "ssi/2015/99"),
                ),
                _read("asp/2015/1"),
                ToolCall(
                    name="delegate_research",
                    input_parameters={},
                    output=_cite("ssi/2015/99"),
                ),
                _read("ssi/2015/99"),
            ],
        )

    def test_sibling_read_is_reported(self):
        metric = CitationReadMetric()
        metric.measure(self._two_step_case())
        assert "Diagnostic, not scored" in metric.reason
        assert "step 1" in metric.reason
        assert "ssi/2015/99" in metric.reason

    def test_sibling_read_does_not_affect_the_score(self):
        """The run as a whole read both Acts, so the score stays 1.0 and the
        record still passes. This is the behaviour the diagnostic replaces."""
        metric = CitationReadMetric()
        metric.measure(self._two_step_case())
        assert metric.score == 1.0
        assert metric.is_successful()

    def test_single_step_run_gets_no_diagnostic(self):
        case = LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="delegate_research",
                    input_parameters={},
                    output="[x](http://www.legislation.gov.uk/id/asp/2015/1)",
                ),
                ToolCall(
                    name="Worker: search_legislation_sections",
                    input_parameters={"legislation_id": "asp/2015/1"},
                    output='[{"text": "Section 1) ..."}]',
                ),
            ],
        )
        metric = CitationReadMetric()
        metric.measure(case)
        assert "Diagnostic" not in metric.reason

    def test_act_no_step_read_still_fails_and_is_not_a_diagnostic(self):
        """An Act nobody read is the scored failure, not the diagnostic."""
        case = LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="delegate_research",
                    input_parameters={},
                    output="[x](http://www.legislation.gov.uk/id/ssi/2015/99)",
                ),
                ToolCall(
                    name="Worker: search_legislation_sections",
                    input_parameters={"legislation_id": "asp/2015/1"},
                    output='[{"text": "Section 1) ..."}]',
                ),
                ToolCall(name="delegate_research", input_parameters={}, output="none"),
            ],
        )
        metric = CitationReadMetric()
        metric.measure(case)
        assert metric.score == 0.0
        assert "Cited without reading" in metric.reason
        assert "Diagnostic" not in metric.reason


class TestRegnalYearActsAreNotFabricated:
    """Acts before 1963 are identified by regnal year, e.g. ``ukpga/Edw7/4/31``
    is the Shop Hours Act 1904.

    Two things used to make every one of them look fabricated: the Act id was
    taken as the first three path segments, which cuts a four-segment regnal id
    in half, and the API sends ids cased while a citation URL arrives
    lowercased. Both were live, and together they scored two "shop definitions"
    runs 0.0 for citing Acts their own searches had returned.
    """

    @staticmethod
    def _case(retrieved_id: str, cited_path: str) -> LLMTestCase:
        search_output = json.dumps(
            {
                "results": [
                    {
                        "legislation_id": retrieved_id,
                        "title": "An Act",
                        "url": f"http://www.legislation.gov.uk/{retrieved_id}",
                    }
                ],
                "total": 1,
            }
        )
        report = f"See [the Act](http://www.legislation.gov.uk/id/{cited_path})."
        return LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="Worker: search_legislation",
                    input_parameters={},
                    output=search_output,
                ),
                ToolCall(name="delegate_research", input_parameters={}, output=report),
            ],
        )

    @pytest.mark.parametrize(
        "retrieved_id, cited_path",
        [
            # Four-segment regnal ids: the truncation bug.
            ("ukpga/Edw7/4/31", "ukpga/Edw7/4/31"),
            ("ukpga/Geo6/12-13-14/25", "ukpga/Geo6/12-13-14/25/section/3"),
            ("ukla/Eliz2/1-2/27", "ukla/Eliz2/1-2/27"),
            # Three-segment regnal id: case alone was enough to break it.
            ("ukpga/1-2Geo5/54", "ukpga/1-2Geo5/54/section/14"),
            # Underscored regnal id.
            ("ukpga/26Geo5_1Edw8/28/1936", "ukpga/26Geo5_1Edw8/28/1936"),
            # Modern ids must keep working.
            ("ukpga/1978/29", "ukpga/1978/29/section/10C"),
            ("ssi/2008/216", "ssi/2008/216/regulation/4"),
        ],
    )
    def test_a_retrieved_act_is_grounded(self, retrieved_id, cited_path):
        metric = CitationGroundingMetric()
        metric.measure(self._case(retrieved_id, cited_path))
        assert metric.score == 1.0, metric.reason
        assert "Fabricated" not in metric.reason

    def test_an_act_that_was_never_retrieved_still_fails(self):
        """The fix must not make everything pass."""
        metric = CitationGroundingMetric()
        metric.measure(self._case("ukpga/Edw7/4/31", "ukpga/Geo6/14/28"))
        assert metric.score == 0.0
        assert "ukpga/geo6/14/28" in metric.reason

    @pytest.mark.parametrize(
        "suffix",
        ["section/8", "schedule/1", "regulation/4", "article/2", "made", "enacted"],
    )
    def test_provision_and_version_suffixes_are_stripped(self, suffix):
        metric = CitationGroundingMetric()
        metric.measure(self._case("ssi/2008/216", f"ssi/2008/216/{suffix}"))
        assert metric.score == 1.0, metric.reason

    def test_an_extent_suffix_after_a_section_is_stripped(self):
        metric = CitationGroundingMetric()
        metric.measure(
            self._case("ukpga/1990/43", "ukpga/1990/43/section/79/england+wales")
        )
        assert metric.score == 1.0, metric.reason


class TestMalformedLinksAreNotFabricatedCitations:
    """A model sometimes writes prose into a link body, e.g.
    ``.../id/[UNCLEAR: no document reference provided]``. That is a broken
    link, not a claim about an Act, so calling it a fabricated citation
    accuses the run of the wrong thing. Reported, not scored.
    """

    @staticmethod
    def _case(*cited_paths: str) -> LLMTestCase:
        search_output = json.dumps(
            {"results": [{"legislation_id": "asp/2021/3", "title": "An Act"}]}
        )
        report = " ".join(
            f"See [it](http://www.legislation.gov.uk/id/{p})." for p in cited_paths
        )
        return LLMTestCase(
            input="q",
            actual_output="final answer",
            tools_called=[
                ToolCall(
                    name="Worker: search_legislation",
                    input_parameters={},
                    output=search_output,
                ),
                ToolCall(name="delegate_research", input_parameters={}, output=report),
            ],
        )

    @pytest.mark.parametrize(
        "junk",
        [
            "[UNCLEAR:%20no%20document%20reference%20provided]",
            "10%20&%2011%20Geo.%205.%20c.%2058",
            "s.i.%201950%20No.%201133%20(S.%2080)",
        ],
    )
    def test_a_malformed_link_does_not_fail_the_metric(self, junk):
        metric = CitationGroundingMetric()
        metric.measure(self._case("asp/2021/3", junk))
        assert metric.score == 1.0, metric.reason
        assert "Fabricated" not in metric.reason

    def test_a_malformed_link_is_still_reported(self):
        metric = CitationGroundingMetric()
        metric.measure(self._case("asp/2021/3", "[UNCLEAR:%20none%20provided]"))
        assert "Diagnostic, not scored" in metric.reason
        assert "not a legislation id" in metric.reason

    def test_a_genuine_fabrication_alongside_junk_still_fails(self):
        metric = CitationGroundingMetric()
        metric.measure(self._case("ukpga/1999/99", "[UNCLEAR:%20none]"))
        assert metric.score == 0.0
        assert "ukpga/1999/99" in metric.reason

    def test_a_truncated_act_link_is_not_a_fabrication(self):
        """A bare type/year link has no chapter number, so it names no Act.
        Response 95 wrote one alongside two correct links to
        ukpga/1988/41/section/65, and the Act itself had been retrieved."""
        metric = CitationGroundingMetric()
        metric.measure(self._case("asp/2021/3", "ukpga/1988"))
        assert metric.score == 1.0, metric.reason
        assert "Diagnostic, not scored" in metric.reason
