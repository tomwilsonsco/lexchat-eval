"""
Unit tests for the attribution helper, synthetic records only, no DB.
"""

import json

import pytest

from lex_eval.metrics.citation_agreement import MIN_OUTPUT_CHARS
from lex_eval.reports.attribution import (
    MODEL,
    NO_LAW_LOST,
    NOT_ATTRIBUTABLE,
    SEARCH,
    TECH,
    attribution_for_record,
    caveat,
    worst_attribution,
)

pytestmark = pytest.mark.unit

_REFERENCE = {
    "question_id": 1,
    "final_answer": (
        "See https://www.legislation.gov.uk/ukpga/2018/12/section/6 and "
        "https://www.legislation.gov.uk/ukpga/1998/46/section/53."
    ),
    # What the author relied on. The prose also links ssi/2015/99, which they
    # only looked at, and attribution must ignore it.
    "sources_retrieved": [
        {"legislation_id": "ukpga/2018/12"},
        {"legislation_id": "ukpga/1998/46"},
    ],
    "sources_discovered": [{"legislation_id": "ssi/2015/99"}],
}
_REFERENCES = {1: _REFERENCE}


def _search(*legislation_ids: str) -> dict:
    return {
        "name": "Worker: search_legislation",
        "input_parameters": {"query": "q"},
        "output": json.dumps(
            {"results": [{"legislation_id": i, "title": i} for i in legislation_ids]}
        ),
    }


def _record(**overrides) -> dict:
    record = {
        "question_id": 1,
        "question": "q",
        "llm_name": "m",
        "actual_output": "",
        "retrieval_context": [],
        "tools_called": [],
        "is_error": False,
        "max_turns_halted": 0,
    }
    record.update(overrides)
    return record


def test_a_halted_run_is_a_tech_problem():
    verdict = attribution_for_record(_record(max_turns_halted=2), _REFERENCES)

    assert verdict["stage"] == TECH
    assert "tool call limit" in verdict["detail"]


def test_a_stored_error_run_is_a_tech_problem():
    """A run that errored outright. load_records excludes these by default, so
    the dashboard has to ask for them, or the clearest terminal failure there
    is becomes the one attribution cannot see."""
    record = _record(is_error=True, error_message="timed out")
    verdict = attribution_for_record(record, _REFERENCES)

    assert verdict["stage"] == TECH
    assert verdict["detail"] == "timed out"


def test_an_error_run_outranks_a_clean_one_in_the_same_group():
    clean = _record(
        actual_output=(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6 "
            "https://www.legislation.gov.uk/ukpga/1998/46/section/53"
        ),
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46")],
    )
    errored = _record(is_error=True, error_message="timed out")

    assert worst_attribution([clean, errored], _REFERENCES)["stage"] == TECH


def test_a_tool_call_api_error_is_a_tech_problem():
    record = _record(
        tools_called=[
            {
                "name": "Worker: search_legislation",
                "input_parameters": {},
                "output": 'Error executing tool: {"detail": "boom"}',
            }
        ]
    )

    assert attribution_for_record(record, _REFERENCES)["stage"] == TECH


def test_an_act_no_tool_call_turned_up_is_attributed_to_the_search():
    record = _record(
        actual_output="https://www.legislation.gov.uk/ukpga/2018/12/section/6",
        tools_called=[_search("ukpga/2018/12")],
    )
    verdict = attribution_for_record(record, _REFERENCES)

    assert verdict["stage"] == SEARCH
    assert verdict["ids"] == ["ukpga/1998/46"]


def test_an_act_a_search_turned_up_but_uncited_is_attributed_to_the_model():
    record = _record(
        actual_output="https://www.legislation.gov.uk/ukpga/2018/12/section/6",
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46")],
    )
    verdict = attribution_for_record(record, _REFERENCES)

    assert verdict["stage"] == MODEL
    assert verdict["ids"] == ["ukpga/1998/46"]


def test_an_answer_too_short_to_judge_is_not_attributable():
    """The gate Citation Agreement scores behind. Without it a clarification
    request reads as a response that cited none of the reference's law."""
    record = _record(actual_output="Could you narrow this down?")
    assert len(record["actual_output"]) <= MIN_OUTPUT_CHARS

    verdict = attribution_for_record(record, _REFERENCES)

    assert verdict["stage"] == NOT_ATTRIBUTABLE


def test_a_reference_the_metric_cannot_measure_is_not_attributable():
    """A reference answer with sources but no citations in its prose is one
    Citation Agreement records as unmeasured, so attribution must stay silent too."""
    references = {
        1: {
            "question_id": 1,
            "final_answer": "Prose with no legislation links at all.",
            "sources_retrieved": [{"legislation_id": "ukpga/2018/12"}],
        }
    }
    record = _record(actual_output="A long enough answer that cites nothing at all.")

    assert attribution_for_record(record, references)["stage"] == NOT_ATTRIBUTABLE


def test_a_question_with_no_reference_answer_is_not_attributable():
    verdict = attribution_for_record(_record(question_id=99), _REFERENCES)

    assert verdict["stage"] == NOT_ATTRIBUTABLE


def test_an_act_the_reference_only_looked_at_is_not_attributed_to_anyone():
    """ssi/2015/99 is linked in the reference prose but sits in
    sources_discovered, so no response has to cite it."""
    record = _record(
        actual_output=(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6 "
            "https://www.legislation.gov.uk/ukpga/1998/46/section/53"
        ),
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46", "ssi/2015/99")],
    )

    assert attribution_for_record(record, _REFERENCES)["stage"] == NO_LAW_LOST


def test_no_law_lost_when_every_relied_on_act_was_found_and_cited():
    """Not a pass. It only means no step lost law, and it is a labelled verdict
    rather than silence, because silence was read as approval."""
    record = _record(
        actual_output=(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6 "
            "https://www.legislation.gov.uk/ukpga/1998/46/section/53"
        ),
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46")],
    )

    verdict = attribution_for_record(record, _REFERENCES)

    assert verdict["stage"] == NO_LAW_LOST
    assert caveat(NO_LAW_LOST) == ""


def test_the_search_verdict_outranks_the_model_across_repeat_runs():
    good = _record(
        actual_output="https://www.legislation.gov.uk/ukpga/2018/12/section/6",
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46")],
    )
    bad = _record(
        actual_output="https://www.legislation.gov.uk/ukpga/2018/12/section/6",
        tools_called=[_search("ukpga/2018/12")],
    )

    assert worst_attribution([good, bad], _REFERENCES)["stage"] == SEARCH


def test_repeat_runs_that_all_lose_nothing_report_no_law_lost():
    record = _record(
        actual_output=(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6 "
            "https://www.legislation.gov.uk/ukpga/1998/46/section/53"
        ),
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46")],
    )

    assert worst_attribution([record, record], _REFERENCES)["stage"] == NO_LAW_LOST


def test_only_the_stages_that_excuse_the_scores_carry_a_caveat():
    assert caveat(SEARCH)
    assert caveat(TECH)
    assert caveat(MODEL) == ""
    assert caveat(NO_LAW_LOST) == ""
    assert caveat(NOT_ATTRIBUTABLE) == ""


def test_a_measured_run_outranks_one_that_could_not_be_measured():
    """A group holding a clarification request and a real answer reports the
    real answer, not "cannot say"."""
    measured = _record(
        actual_output=(
            "https://www.legislation.gov.uk/ukpga/2018/12/section/6 "
            "https://www.legislation.gov.uk/ukpga/1998/46/section/53"
        ),
        tools_called=[_search("ukpga/2018/12", "ukpga/1998/46")],
    )
    clarification = _record(actual_output="Could you narrow this down?")

    assert (
        worst_attribution([clarification, measured], _REFERENCES)["stage"]
        == NO_LAW_LOST
    )
