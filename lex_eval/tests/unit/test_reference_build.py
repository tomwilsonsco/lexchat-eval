"""
Unit tests for the authored-statement rules in
``lex_eval.reference.build``. No LEX API, no judge, no DB.
"""

import json

import pytest

from lex_eval.reference.build import MAX_STATEMENTS, read_statements

pytestmark = pytest.mark.unit


def _write(tmp_path, statements):
    (tmp_path / "statements.json").write_text(
        json.dumps({"statements": statements}), encoding="utf-8"
    )
    return tmp_path


def test_fewer_than_the_cap_is_allowed(tmp_path):
    """The cap is not a quota: a narrow question may only have two points."""
    src = _write(tmp_path, ["The Ministers appoint the chair.", "No fixed term."])

    assert read_statements(src) == [
        "The Ministers appoint the chair.",
        "No fixed term.",
    ]


def test_order_is_preserved(tmp_path):
    """Most important first, and the judge is shown them in that order."""
    src = _write(tmp_path, ["first", "second", "third"])

    assert read_statements(src) == ["first", "second", "third"]


def test_more_than_the_cap_is_rejected(tmp_path):
    src = _write(tmp_path, [f"statement {i}" for i in range(MAX_STATEMENTS + 1)])

    with pytest.raises(ValueError, match="between 1 and"):
        read_statements(src)


def test_no_statements_is_rejected(tmp_path):
    """Nothing to label means nothing to score, so catch it at authoring time."""
    src = _write(tmp_path, ["   "])

    with pytest.raises(ValueError, match="between 1 and"):
        read_statements(src)


# ---------------------------------------------------------------------------
# Research mode gating
# ---------------------------------------------------------------------------


class TestModeGating:
    """A question may only be researched with the tools its own brief allows."""

    def test_case_law_search_is_rejected_for_a_legislation_only_question(self):
        from lex_eval.reference.build import check_tools

        with pytest.raises(ValueError, match="search_case_law"):
            check_tools(
                [{"tool": "search_case_law", "args": {"query": "x"}}],
                "legislation_only",
            )

    def test_legislation_search_is_rejected_for_a_case_law_only_question(self):
        from lex_eval.reference.build import check_tools

        with pytest.raises(ValueError, match="search_legislation"):
            check_tools(
                [{"tool": "search_legislation", "args": {"query": "x"}}],
                "case_law_only",
            )

    def test_both_are_allowed_in_the_combined_mode(self):
        from lex_eval.reference.build import check_tools

        check_tools(
            [
                {"tool": "search_legislation", "args": {"query": "x"}},
                {"tool": "search_case_law", "args": {"query": "y"}},
            ],
            "legislation_and_case_law",
        )

    def test_an_unknown_tool_name_is_rejected(self):
        from lex_eval.reference.build import check_tools

        with pytest.raises(ValueError, match="serch_legislation"):
            check_tools([{"tool": "serch_legislation", "args": {}}], "legislation_only")


class TestScaffold:
    """A fresh question gets the searches template its research mode needs."""

    def _tools_in_template(self, tmp_path, mode):
        from lex_eval.reference.build import scaffold

        scaffold(tmp_path, {"id": 1, "question": "q", "research_mode": mode})
        return [e["tool"] for e in json.loads((tmp_path / "searches.json").read_text())]

    def test_case_law_only_scaffolds_case_law_searches(self, tmp_path):
        assert self._tools_in_template(tmp_path, "case_law_only") == [
            "search_case_law",
            "get_case_law_text",
        ]

    def test_legislation_only_scaffolds_legislation_searches(self, tmp_path):
        assert self._tools_in_template(tmp_path, "legislation_only") == [
            "search_legislation",
            "search_legislation_sections",
        ]

    def test_combined_mode_scaffolds_both(self, tmp_path):
        assert self._tools_in_template(tmp_path, "legislation_and_case_law") == [
            "search_legislation",
            "search_legislation_sections",
            "search_case_law",
            "get_case_law_text",
        ]

    def test_every_supported_mode_has_a_template(self):
        from lex_eval.reference.build import SUPPORTED_MODES, _SEARCHES_BY_MODE

        assert set(_SEARCHES_BY_MODE) == SUPPORTED_MODES
