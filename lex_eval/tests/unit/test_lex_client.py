"""
Unit tests for the LexChat behaviour mirrored in ``reference/lex_client.py``.
No LEX API.
"""

import json

import httpx
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


# ---------------------------------------------------------------------------
# Case law parsing and the derived audit fields
# ---------------------------------------------------------------------------

# One entry from a real Find Case Law Atom feed, trimmed to the elements the
# parser reads. The TNA elements sit on the bare host namespace; reading them
# from the /terms/v1 URI instead is what once returned an empty ncn on every hit.
_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:tna="https://caselaw.nationalarchives.gov.uk">
  <entry>
    <title>Graham Andrew Evans v R</title>
    <link rel="alternate"
          href="https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150"/>
    <published>2025-09-05T00:00:00+00:00</published>
    <tna:identifier slug="ewca/crim/2025/1150" type="ukncn">[2025] EWCA Crim 1150</tna:identifier>
  </entry>
</feed>"""

_JUDGMENT = """<?xml version="1.0" encoding="utf-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
            xmlns:uk="https://caselaw.nationalarchives.gov.uk/akn">
  <judgment>
    <meta><identification><FRBRWork>
      <FRBRname value="Graham Andrew Evans v R"/>
    </FRBRWork></identification>
    <proprietary><uk:cite>[2025] EWCA Crim 1150</uk:cite></proprietary></meta>
    <judgmentBody><p>The appeal is dismissed.</p></judgmentBody>
  </judgment>
</akomaNtoso>"""


def test_atom_entry_parses_into_title_ncn_court_date_url():
    from lex_eval.reference.lex_client import parse_case_law_atom

    assert parse_case_law_atom(_ATOM) == [
        {
            "title": "Graham Andrew Evans v R",
            "ncn": "[2025] EWCA Crim 1150",
            "court": "ewca/crim",
            "date": "2025-09-05",
            "url": "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150",
        }
    ]


def test_unparseable_atom_returns_no_entries():
    from lex_eval.reference.lex_client import parse_case_law_atom

    assert parse_case_law_atom("not xml at all") == []


def test_judgment_parses_into_title_ncn_and_text():
    from lex_eval.reference.lex_client import parse_judgment

    url = "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150"
    parsed = parse_judgment(_JUDGMENT, url)
    assert parsed["title"] == "Graham Andrew Evans v R"
    assert parsed["ncn"] == "[2025] EWCA Crim 1150"
    assert "The appeal is dismissed." in parsed["text"]
    assert parsed["url"] == url


def _tools_with_case_calls():
    """A LexTools with case law calls already recorded, making no HTTP request."""
    from lex_eval.reference.lex_client import ApiCall, LexTools, parse_case_law_atom

    tools = LexTools.__new__(LexTools)
    tools.api_calls = []
    tools.runs = []
    results = parse_case_law_atom(_ATOM) + [
        {
            "title": "Some Other Case",
            "ncn": "[2020] EWCA Civ 1",
            "court": "ewca/civ",
            "date": "2020-01-01",
            "url": "https://caselaw.nationalarchives.gov.uk/ewca/civ/2020/1",
        }
    ]
    tools.api_calls.append(
        ApiCall(
            "search_case_law",
            "https://caselaw.nationalarchives.gov.uk/atom.xml",
            {"query": "Evans"},
            200,
            10,
            {"results": results, "total": 2},
        )
    )
    tools.api_calls.append(
        ApiCall(
            "get_case_law_text",
            "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150/data.xml",
            {},
            200,
            10,
            {
                "url": "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150",
                "title": "Graham Andrew Evans v R",
                "ncn": "[2025] EWCA Crim 1150",
                "text": "The appeal is dismissed.",
            },
        )
    )
    return tools


def test_cases_retrieved_takes_court_and_date_from_the_search_hit():
    """The judgment XML carries no court or date; the search result does."""
    assert _tools_with_case_calls().cases_retrieved() == [
        {
            "url": "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/1150",
            "title": "Graham Andrew Evans v R",
            "ncn": "[2025] EWCA Crim 1150",
            "court": "ewca/crim",
            "date": "2025-09-05",
        }
    ]


def test_cases_discovered_is_what_search_found_but_nobody_read():
    discovered = _tools_with_case_calls().cases_discovered()
    assert [c["ncn"] for c in discovered] == ["[2020] EWCA Civ 1"]


def test_retrieval_context_holds_the_search_hits_and_the_judgment_text():
    context = _tools_with_case_calls().retrieval_context()
    assert "[2025] EWCA Crim 1150" in " ".join(context)
    assert any("The appeal is dismissed." in item for item in context)


def test_reading_a_judgment_is_not_the_full_act_fallback():
    """fallback_used tracks get_legislation_text only."""
    assert _tools_with_case_calls().fallback_used() is False


@pytest.mark.parametrize(
    "mode, tool, allowed",
    [
        ("legislation_only", "search_legislation", True),
        ("legislation_only", "search_case_law", False),
        ("case_law_only", "search_case_law", True),
        ("case_law_only", "search_legislation", False),
        ("legislation_and_case_law", "search_legislation", True),
        ("legislation_and_case_law", "get_case_law_text", True),
    ],
)
def test_each_mode_permits_only_its_own_tools(mode, tool, allowed):
    from lex_eval.reference.lex_client import TOOLS_BY_MODE

    assert (tool in TOOLS_BY_MODE[mode]) is allowed


# ---------------------------------------------------------------------------
# Case law failure handling
# ---------------------------------------------------------------------------


def _tools_on(handler, monkeypatch):
    """A LexTools whose HTTP goes to *handler*, with the retry backoff removed."""
    from lex_eval.reference import lex_client
    from lex_eval.reference.lex_client import LEX_API_URL, LexTools

    monkeypatch.setattr(lex_client, "_BASE_BACKOFF_S", 0.0)
    tools = LexTools.__new__(LexTools)
    tools.base_url = LEX_API_URL
    tools._client = httpx.Client(transport=httpx.MockTransport(handler))
    tools.api_calls = []
    tools.runs = []
    return tools


def test_a_failing_search_is_reported_as_an_error_not_as_no_judgments(monkeypatch):
    """A 503 must not read as "this search found no case law".

    Left unraised it becomes {"total": 0} and the zero-results nudge, and the
    author writes a reference answer resting on a false absence of case law.
    """

    def handler(request):
        return httpx.Response(503, text="upstream unavailable")

    tools = _tools_on(handler, monkeypatch)
    result = json.loads(tools.execute("search_case_law", {"query": "detention"}))

    assert "error" in result
    assert result.get("total") != 0
    assert "returned 0 results" not in json.dumps(result)


def test_a_failing_search_is_retried_and_still_recorded(monkeypatch):
    """The retry budget is LexChat's, and the failed call stays in the audit."""
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(503, text="upstream unavailable")

    tools = _tools_on(handler, monkeypatch)
    tools.execute("search_case_law", {"query": "detention"})

    assert len(seen) == 4  # one attempt plus three retries
    assert [c.status for c in tools.api_calls] == [503]
    assert len(tools.tools_called()) == 1  # calls and outputs stay aligned


def test_a_bad_court_filter_is_answered_rather_than_raised(monkeypatch):
    """The one status search_case_law handles itself, as executor.py does."""

    def handler(request):
        return httpx.Response(400, text="bad court")

    tools = _tools_on(handler, monkeypatch)
    result = json.loads(
        tools.execute("search_case_law", {"query": "x", "court": "nonsense"})
    )

    assert result["results"] == []
    assert result["total"] == 0
    assert "Invalid court filter" in result["error"]


def test_a_judgment_that_cannot_be_fetched_returns_an_explicit_error(monkeypatch):
    """Mirrors executor.py, which answers a failed fetch with an empty judgment."""

    def handler(request):
        return httpx.Response(404, text="not found")

    tools = _tools_on(handler, monkeypatch)
    url = "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/9999"
    result = json.loads(tools.execute("get_case_law_text", {"url": url}))

    assert result == {
        "error": "HTTP 404 fetching judgment",
        "url": url,
        "text": "",
    }


def test_an_unfetched_judgment_is_not_counted_as_read(monkeypatch):
    """No text means no retrieval evidence, so nothing may cite it as read."""

    def handler(request):
        return httpx.Response(404, text="not found")

    tools = _tools_on(handler, monkeypatch)
    tools.execute(
        "get_case_law_text",
        {"url": "https://caselaw.nationalarchives.gov.uk/ewca/crim/2025/9999"},
    )

    assert tools.cases_retrieved() == []
    assert tools.retrieval_context() == []


def test_a_network_failure_does_not_shift_later_outputs_onto_the_wrong_call(
    monkeypatch,
):
    """A search that never reached the service must not steal the next one's result.

    The failed run records no request, so pairing outputs to requests by
    position would file the second search's result against the first search
    and drop the second, misrepresenting which search found what.
    """
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= 4:  # one attempt plus three retries, all refused
            raise httpx.ConnectError("no route to host")
        return httpx.Response(200, text=_ATOM)

    tools = _tools_on(handler, monkeypatch)
    tools.execute("search_case_law", {"query": "first"})
    tools.execute("search_case_law", {"query": "second"})

    called = tools.tools_called()
    assert [t["input_parameters"]["query"] for t in called] == ["first", "second"]
    assert "ConnectError" in called[0]["output"]
    assert "Evans" in called[1]["output"]
    assert tools.tool_sequence() == [
        "Worker: search_case_law",
        "Worker: search_case_law",
    ]
