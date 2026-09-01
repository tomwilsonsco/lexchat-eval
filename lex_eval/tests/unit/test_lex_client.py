"""
Unit tests for the LexChat behaviour mirrored in ``reference/lex_client.py``.
No LEX API.
"""

import pytest

from lex_eval.reference.lex_client import matches_jurisdiction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "extent, expected",
    [
        # What LexChat's filter is written for.
        (["E+W+S+NI"], True),
        (["S"], True),
        (["E+W"], False),
        # What the LEX API actually sends. Every one of these is dropped, which
        # is TOM_TO_DO.md finding 41: a Scotland filter that keeps no Scottish
        # Act. Measured 1 Sep 2026 at 0 of 75 Scottish results surviving.
        (["Scotland"], False),
        (["United Kingdom"], False),
        ([""], False),
        # An unknown extent is kept, but only when the list is genuinely empty.
        ([], True),
    ],
)
def test_scotland_filter_against_real_and_expected_extents(extent, expected):
    assert matches_jurisdiction(extent, "scotland") is expected


def test_uk_wide_needs_all_four_territories():
    assert matches_jurisdiction(["E+W+S+NI"], "uk_wide") is True
    assert matches_jurisdiction(["E+W"], "uk_wide") is False
    assert matches_jurisdiction([""], "uk_wide") is False


def test_an_unrecognised_jurisdiction_keeps_everything():
    assert matches_jurisdiction(["Scotland"], "atlantis") is True
