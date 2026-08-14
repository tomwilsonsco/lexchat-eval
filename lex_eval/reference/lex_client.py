"""The three legislation tools, calling the LEX API as LexChat's Worker does.

Mirrors `LexChat/server_py/src/agent/tools/executor.py` and `.../tools/lex.py`: the
same endpoints, the same request payloads, the same response slimming and the same
Phase-2 nudge. The retrieved text is therefore byte-for-byte what LexChat would have
received for the same queries.

Tool results are recorded in full. LexChat summarises anything over a size
threshold, but a reference answer should rest on the primary text.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

LEX_API_URL = os.getenv("LEX_API_URL", "https://lex.lab.i.ai.gov.uk").rstrip("/")

# LexChat's production values (executor.py), with no research filters active.
SEARCH_LIMIT = 5
SECTION_LIMIT = 10

TOOLS = ("search_legislation", "search_legislation_sections", "get_legislation_text")

_RETRY_STATUS = {429, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 8.0


# ---------------------------------------------------------------------------
# Response shaping, copied from LexChat/server_py/src/agent/tools/lex.py
# ---------------------------------------------------------------------------


def slim_search_results(resp_json: dict) -> dict:
    """Strip a `search_legislation` response to the fields LexChat passes on.

    `description` is excluded upstream on purpose, and `legislation_id` is derived
    from the URI so it can be used directly in a section search.
    """
    slimmed = []
    for item in resp_json.get("results", []):
        uri = item.get("uri", "")
        legislation_id = urlparse(uri).path.lstrip("/") if uri else ""
        if legislation_id.startswith("id/"):
            legislation_id = legislation_id[3:]
        slimmed.append(
            {
                "legislation_id": legislation_id,
                "title": item.get("title", ""),
                "url": uri,
                "status": item.get("status", ""),
                "year": item.get("year"),
                "extent": item.get("extent", []),
            }
        )
    return {"results": slimmed, "total": resp_json.get("total", len(slimmed))}


def _extract_ids(resp_json: dict) -> List[Tuple[str, str]]:
    return [
        (item["legislation_id"], item.get("title", ""))
        for item in resp_json.get("results", [])
        if item.get("legislation_id")
    ]


def _phase2_nudge(slimmed: dict) -> str:
    """The `[NEXT STEP: ...]` note LexChat appends after every `search_legislation`."""
    id_pairs = _extract_ids(slimmed)
    if not id_pairs:
        return ""
    lines = "\n".join(f'  - legislation_id: "{i}"  ({t})' for i, t in id_pairs[:5])
    return (
        "\n\n[NEXT STEP: Call search_legislation_sections with the relevant "
        "legislation_id(s) below to retrieve the actual legal text before "
        f"composing your answer:\n{lines}]"
    )


def _section_text(section: dict) -> str:
    return (
        section.get("content")
        or section.get("text")
        or section.get("section_text")
        or ""
    )


def _sections_of(response: Any) -> List[dict]:
    """Normalise a section-search response, which may be a list or a wrapper dict."""
    if isinstance(response, list):
        return [s for s in response if isinstance(s, dict)]
    if isinstance(response, dict):
        items = response.get("sections") or response.get("results") or []
        return [s for s in items if isinstance(s, dict)]
    return []


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


@dataclass
class ApiCall:
    """One LEX request/response, the unit the retrieval audit is derived from."""

    tool: str
    url: str
    payload: Dict[str, Any]
    status: int
    elapsed_ms: int
    response: Any


class LexTools:
    """Executes the legislation tools and records everything it did."""

    def __init__(self, *, timeout: float = 60.0, base_url: str = LEX_API_URL) -> None:
        self.base_url = base_url.rstrip("/")
        # verify=False matches LexChat, which runs behind SSL inspection on the target.
        self._client = httpx.Client(timeout=timeout, verify=False)
        self.api_calls: List[ApiCall] = []
        self.outputs: List[str] = []

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LexTools":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- HTTP ------------------------------------------------------------

    def _post(self, tool: str, path: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                resp = self._client.post(url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= _MAX_RETRIES:
                    raise
                delay = min(_BASE_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
                logger.warning(
                    "[LEX] %s network error (%r); retrying in %.1fs", tool, exc, delay
                )
                time.sleep(delay)
                attempt += 1
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                delay = min(_BASE_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
                logger.warning(
                    "[LEX] %s HTTP %d; retrying in %.1fs", tool, resp.status_code, delay
                )
                time.sleep(delay)
                attempt += 1
                continue

            try:
                resp_json = resp.json()
            except ValueError:
                resp_json = {"text": resp.text}

            self.api_calls.append(
                ApiCall(
                    tool, url, payload, resp.status_code, round(elapsed_ms), resp_json
                )
            )
            resp.raise_for_status()
            return resp_json

    # -- Dispatch --------------------------------------------------------

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Run one tool and return the JSON string LexChat would hand to the model."""
        try:
            result = self._execute(name, args)
        except Exception as exc:
            logger.warning("[LEX] %s failed: %r", name, exc)
            result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        self.outputs.append(result)
        return result

    def _execute(self, name: str, args: Dict[str, Any]) -> str:
        if name == "search_legislation":
            payload = {
                "query": args["query"],
                "year_from": args.get("year_from"),
                "year_to": args.get("year_to"),
                "limit": SEARCH_LIMIT,
                "include_text": False,
            }
            slimmed = slim_search_results(
                self._post(name, "/legislation/search", payload)
            )
            slimmed["results"] = slimmed["results"][:SEARCH_LIMIT]
            slimmed["total"] = len(slimmed["results"])
            return json.dumps(slimmed) + _phase2_nudge(slimmed)

        if name == "search_legislation_sections":
            payload = {
                "query": args["query"],
                "legislation_id": args["legislation_id"],
                "limit": SECTION_LIMIT,
            }
            return json.dumps(self._post(name, "/legislation/section/search", payload))

        if name == "get_legislation_text":
            payload = {"legislation_id": args["legislation_id"]}
            return json.dumps(self._post(name, "/legislation/text", payload))

        return json.dumps({"error": f"Unknown tool {name}. Expected one of {TOOLS}."})

    # -- Derived audit fields --------------------------------------------

    def tool_sequence(self) -> List[str]:
        """Ordered tool names, prefixed as `utils/audit_capture.py` prefixes them."""
        return [f"Worker: {c.tool}" for c in self.api_calls]

    def tools_called(self) -> List[Dict[str, Any]]:
        """The `tools_called` list, shaped as the `responses` table shapes it."""
        return [
            {
                "name": f"Worker: {call.tool}",
                "input_parameters": call.payload,
                "output": output,
            }
            for call, output in zip(self.api_calls, self.outputs)
        ]

    def retrieval_context(self) -> List[str]:
        """The retrieved legal text, shaped as `lex_eval`'s `retrieval_context`.

        Same derivation as `utils/audit_capture.py`, so groundedness metrics can read
        a reference record without a second code path.
        """
        context: List[str] = []
        for call in self.api_calls:
            if call.tool == "get_legislation_text":
                if isinstance(call.response, dict) and call.response.get("full_text"):
                    context.append(call.response["full_text"])
            elif call.tool == "search_legislation_sections":
                for sec in _sections_of(call.response):
                    text = _section_text(sec)
                    if text:
                        title = sec.get("title") or sec.get("section_title") or ""
                        context.append(f"{title}: {text}" if title else text)
        return context

    def sources_retrieved(self) -> List[Dict[str, Any]]:
        """Provisions whose text was actually pulled: what a citation must match."""
        seen: Dict[str, Dict[str, Any]] = {}
        for call in self.api_calls:
            if call.tool == "search_legislation_sections":
                for sec in _sections_of(call.response):
                    uri = sec.get("uri") or sec.get("id") or ""
                    if uri and uri not in seen:
                        seen[uri] = {
                            "uri": uri,
                            "title": sec.get("title", ""),
                            "legislation_id": call.payload.get("legislation_id", ""),
                            "provision_type": sec.get("provision_type", ""),
                            "extent": sec.get("extent", []),
                        }
            elif call.tool == "get_legislation_text":
                lid = call.payload.get("legislation_id", "")
                uri = f"http://www.legislation.gov.uk/id/{lid}"
                if uri not in seen:
                    seen[uri] = {
                        "uri": uri,
                        "title": (
                            call.response.get("title", "")
                            if isinstance(call.response, dict)
                            else ""
                        ),
                        "legislation_id": lid,
                        "provision_type": "full_text",
                        "extent": [],
                    }
        return list(seen.values())

    def sources_discovered(self) -> List[Dict[str, Any]]:
        """Acts and SIs seen in a search result but never read.

        Citing one of these is a weaker claim than citing a provision above: the Act
        exists and was found, but its text was never retrieved. Keeping the two apart
        is what lets a citation check tell "cited what it read" from "cited what it
        merely saw".
        """
        retrieved = {s["legislation_id"] for s in self.sources_retrieved()}
        seen: Dict[str, Dict[str, Any]] = {}
        for call in self.api_calls:
            if call.tool != "search_legislation" or not isinstance(call.response, dict):
                continue
            for item in slim_search_results(call.response).get("results", []):
                lid = item.get("legislation_id")
                if lid and lid not in retrieved and lid not in seen:
                    seen[lid] = {
                        "uri": item.get("url", ""),
                        "title": item.get("title", ""),
                        "legislation_id": lid,
                        "status": item.get("status", ""),
                        "year": item.get("year"),
                        "extent": item.get("extent", []),
                    }
        return list(seen.values())

    def fallback_used(self) -> bool:
        return any(c.tool == "get_legislation_text" for c in self.api_calls)
