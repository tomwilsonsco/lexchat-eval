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
