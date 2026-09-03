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


class TestRetrievedDump:
    """What the author reads before writing the answer."""

    def _dump(self, tmp_path, monkeypatch, status):
        """Render retrieved.md for one search_case_law call that returned *status*."""
        from lex_eval.reference import build as build_mod
        from lex_eval.reference.lex_client import ApiCall

        class _StubTools:
            def __init__(self):
                self.api_calls = [
                    ApiCall(
                        "search_case_law",
                        "https://caselaw.nationalarchives.gov.uk/atom.xml",
                        {"query": "privilege", "court": "csih"},
                        status,
                        10,
                        {},
                    )
                ]

            def execute(self, name, args):
                return "{}"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

        monkeypatch.setattr(build_mod, "LexTools", _StubTools)
        build_mod.retrieve(
            tmp_path,
            [{"tool": "search_case_law", "args": {"query": "privilege"}}],
            "case_law_only",
        )
        return (tmp_path / "retrieved.md").read_text()

    def test_a_failed_search_is_not_reported_as_no_judgments(
        self, tmp_path, monkeypatch
    ):
        """A dead service must not read as a search that found nothing.

        An author who takes it that way writes a reference answer resting on a
        false absence of case law.
        """
        dump = self._dump(tmp_path, monkeypatch, 400)

        assert "failed with HTTP 400" in dump
        assert "No judgments matched" not in dump

    def test_a_successful_search_with_no_hits_still_says_so(
        self, tmp_path, monkeypatch
    ):
        dump = self._dump(tmp_path, monkeypatch, 200)

        assert "No judgments matched" in dump
        assert "failed with HTTP" not in dump


class TestRefetch:
    """`--refetch` re-runs the searches for an answer that already exists."""

    def _authored(self, tmp_path, answer):
        """An `.authored/q1/` whose retrieval and answer are already in place."""
        from lex_eval.reference.build import authored_dir

        src = authored_dir(tmp_path, 1)
        src.mkdir(parents=True)
        (src / "searches.json").write_text(
            json.dumps([{"tool": "search_legislation", "args": {"query": "duty"}}]),
            encoding="utf-8",
        )
        (src / "retrieved.md").write_text("the previous retrieval", encoding="utf-8")
        (src / "answer.md").write_text(answer, encoding="utf-8")
        (src / "plan.json").write_text(
            json.dumps({"scope_note": "The duty.", "steps": []}), encoding="utf-8"
        )
        (src / "statements.json").write_text(
            json.dumps({"statements": ["The duty applies."]}), encoding="utf-8"
        )
        return src

    def _run(self, tmp_path, monkeypatch, answer):
        """process() one question with refetch on, recording what it did."""
        from lex_eval.reference import build as build_mod

        self._authored(tmp_path, answer)
        did = {}
        monkeypatch.setattr(
            build_mod, "retrieve", lambda *a, **k: did.setdefault("retrieved", 1)
        )
        monkeypatch.setattr(
            build_mod,
            "build",
            lambda *a, **k: did.setdefault("built", True)
            and {"sources_retrieved": [], "cases_retrieved": [], "tool_sequence": []},
        )
        message, record = build_mod.process(
            {"id": 1, "question": "Does the duty apply?"},
            tmp_path,
            "an author",
            None,
            refetch=True,
        )
        return message, record, did

    def test_refetch_rebuilds_an_answer_that_is_already_written(
        self, tmp_path, monkeypatch
    ):
        """Otherwise the documented command silently does nothing.

        Stopping after the retrieval leaves the author to re-run with
        --overwrite to get the rebuild their edit to searches.json was for.
        """
        message, record, did = self._run(tmp_path, monkeypatch, "The duty applies.")

        assert did == {"retrieved": 1, "built": True}
        assert message.startswith("BUILT")
        assert "re-researched in 1 call(s)" in message
        assert record is not None

    def test_refetch_still_stops_at_the_retrieval_for_an_unwritten_answer(
        self, tmp_path, monkeypatch
    ):
        """The one stage at a time flow for a new question is unchanged."""
        from lex_eval.reference.build import _TODO

        message, record, did = self._run(tmp_path, monkeypatch, _TODO)

        assert did == {"retrieved": 1}
        assert message.startswith("RETRIEVED")
        assert record is None

    def test_a_build_with_no_refetch_says_nothing_about_re_research(
        self, tmp_path, monkeypatch
    ):
        from lex_eval.reference import build as build_mod

        self._authored(tmp_path, "The duty applies.")
        monkeypatch.setattr(
            build_mod,
            "build",
            lambda *a, **k: {
                "sources_retrieved": [],
                "cases_retrieved": [],
                "tool_sequence": [],
            },
        )
        message, _ = build_mod.process(
            {"id": 1, "question": "Does the duty apply?"},
            tmp_path,
            "an author",
            None,
            refetch=False,
        )

        assert message.startswith("BUILT")
        assert "re-researched" not in message

    def test_refetch_does_not_skip_a_question_that_already_has_an_answer(
        self, tmp_path, monkeypatch
    ):
        """The completed-answer guard has to let --refetch through to process()."""
        from lex_eval.reference import build as build_mod

        questions = tmp_path / "questions.json"
        questions.write_text(
            json.dumps([{"id": 1, "question": "Does the duty apply?"}]),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            build_mod,
            "load_manifest",
            lambda answers_dir: [
                {
                    "question_id": 1,
                    "final_answer": "The duty applies.",
                    "statements": ["The duty applies."],
                }
            ],
        )
        reached = []
        monkeypatch.setattr(
            build_mod,
            "process",
            lambda q, *a, **k: (reached.append(k["refetch"]), ("BUILT q1.md", None))[1],
        )

        build_mod.main(
            [
                "--questions",
                str(questions),
                "--answers-dir",
                str(tmp_path),
                "--question-id",
                "1",
                "--refetch",
            ]
        )

        assert reached == [True]
