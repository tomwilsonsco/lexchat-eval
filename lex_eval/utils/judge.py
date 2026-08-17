"""
Unified judge factory for DeepEval metrics.

Provides an OpenRouter-based judge that implements a `.generate(prompt, schema)`
interface used by custom metrics.

Usage in test files:
    from lex_eval.utils.judge import _judge
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
_DEFAULT_MODEL = "openai/gpt-4o"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096
OPENROUTER_JUDGE_MODEL: str = os.getenv("OPENROUTER_JUDGE_MODEL", _DEFAULT_MODEL)


def _parse_max_tokens(raw: str | None) -> int:
    """Parse the OPENROUTER_JUDGE_MAX_TOKENS env var.

    Returns the default on missing/invalid input, logging a warning.
    """
    if raw is None or raw.strip() == "":
        return _DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "OPENROUTER_JUDGE_MAX_TOKENS=%r is not a valid int; falling back to %s",
            raw,
            _DEFAULT_MAX_TOKENS,
        )
        return _DEFAULT_MAX_TOKENS
    if value <= 0:
        logger.warning(
            "OPENROUTER_JUDGE_MAX_TOKENS=%s must be positive; falling back to %s",
            value,
            _DEFAULT_MAX_TOKENS,
        )
        return _DEFAULT_MAX_TOKENS
    return value


OPENROUTER_JUDGE_MAX_TOKENS: int = _parse_max_tokens(
    os.getenv("OPENROUTER_JUDGE_MAX_TOKENS")
)
OPENROUTER_JUDGE_REASONING_EFFORT: str | None = (
    os.getenv("OPENROUTER_JUDGE_REASONING_EFFORT") or None
)
OPENROUTER_JUDGE_FALLBACK_MODEL: str | None = (
    os.getenv("OPENROUTER_JUDGE_FALLBACK_MODEL") or None
)


def _parse_temperature(raw: str | None) -> float:
    """Parse the OPENROUTER_JUDGE_TEMPERATURE env var.

    Returns 0.0 on missing/invalid input, logging a warning so misconfigurations
    are visible without crashing the eval run.
    """
    if raw is None or raw.strip() == "":
        return _DEFAULT_TEMPERATURE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "OPENROUTER_JUDGE_TEMPERATURE=%r is not a valid float; "
            "falling back to %s",
            raw,
            _DEFAULT_TEMPERATURE,
        )
        return _DEFAULT_TEMPERATURE
    if value < 0.0 or value > 2.0:
        logger.warning(
            "OPENROUTER_JUDGE_TEMPERATURE=%s is outside the typical "
            "0.0–2.0 range; using it anyway but results may be unpredictable",
            value,
        )
    return value


OPENROUTER_JUDGE_TEMPERATURE: float = _parse_temperature(
    os.getenv("OPENROUTER_JUDGE_TEMPERATURE")
)

# Retries for transient errors (rate limits, 5xx), running multiple
# pytest-xdist workers concurrently makes these more likely than in a fully
# serial run. Backoff: 1s, 2s, 4s.
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.0


class OpenRouterJudge:
    """A lightweight judge that calls OpenRouter's chat completions API."""

    def __init__(
        self, model: str | None = None, temperature: float | None = None
    ) -> None:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set; cannot create OpenRouterJudge"
            )
        self._model = model or OPENROUTER_JUDGE_MODEL
        self._temperature = (
            temperature if temperature is not None else OPENROUTER_JUDGE_TEMPERATURE
        )
        self._max_tokens = OPENROUTER_JUDGE_MAX_TOKENS
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        # last_model / total_usage_tokens accumulate across every _call() made
        # since the caller last reset them (generate() itself does NOT reset,
        # since one metric measurement can span multiple generate() calls,
        # e.g. ReferenceAnswerAgreementMetric's retry-on-wrong-label-count
        # loop). Callers that want a per-measurement total must reset both to
        # None immediately before starting that measurement. last_model ends
        # up holding the model that produced the final, successful result;
        # total_usage_tokens sums every attempt's tokens, including
        # empty-content retries that get discarded, since those still cost
        # real tokens.
        self.last_model: str | None = None
        self.total_usage_tokens: int | None = None

    @staticmethod
    def _strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
        """Return a JSON Schema that strict structured-output mode accepts.

        OpenAI models reject a strict schema unless every object in it sets
        ``additionalProperties: false``, which Pydantic does not emit. Other
        providers accept the schema either way, so this is applied always
        rather than per-model.
        """

        def tighten(node: Any) -> Any:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    node["additionalProperties"] = False
                for value in node.values():
                    tighten(value)
            elif isinstance(node, list):
                for value in node:
                    tighten(value)
            return node

        return tighten(schema.model_json_schema())

    def _call(
        self, model: str, prompt: str, max_tokens: int, schema: type[BaseModel] | None
    ) -> Any:
        """Make one judge call (with retry on transient rate-limit/5xx errors).

        Raises ValueError if the model returns empty content, that is a
        reasoning-budget exhaustion, not a transient error, and is handled by
        the retry/fallback ladder in generate().
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": max_tokens,
        }

        if OPENROUTER_JUDGE_REASONING_EFFORT is not None:
            kwargs["extra_body"] = {
                "reasoning": {"effort": OPENROUTER_JUDGE_REASONING_EFFORT}
            }

        if schema is not None:
            # Use structured output (JSON Schema mode) when a schema is provided.
            # This is more reliable than {"type": "json_object"} because it
            # constrains the output to the exact schema shape, preventing models
            # from returning empty or malformed JSON.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": self._strict_schema(schema),
                },
            }

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)

                # Accumulate before the empty-content check below: a call that
                # gets discarded for returning no content still spent tokens.
                usage = getattr(response, "usage", None)
                call_tokens = getattr(usage, "total_tokens", None) if usage else None
                if call_tokens is not None:
                    self.total_usage_tokens = (
                        self.total_usage_tokens or 0
                    ) + call_tokens

                content = response.choices[0].message.content or ""

                if not content.strip():
                    raise ValueError(
                        f"Judge returned empty content for prompt (model={model})"
                    )

                self.last_model = getattr(response, "model", None) or model

                if schema is not None:
                    data = json.loads(content)
                    return schema(**data)

                return content

            except (RateLimitError, APIStatusError) as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = isinstance(exc, RateLimitError) or (
                    status_code is not None and status_code >= 500
                )
                if not retryable or attempt == _MAX_RETRIES:
                    logger.exception("OpenRouter judge call failed")
                    raise
                delay = _RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "OpenRouter judge call failed (%s), retrying in %.0fs "
                    "(attempt %d/%d)",
                    exc,
                    delay,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(delay)
            except Exception:
                logger.exception("OpenRouter judge call failed")
                raise

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Send a prompt to the OpenRouter model and return the response.

        On empty content (the judge exhausted its reasoning budget), retries
        once at double `max_tokens`, then once more against
        OPENROUTER_JUDGE_FALLBACK_MODEL if that is configured and still empty.

        Args:
            prompt: The prompt string to send.
            schema: Optional Pydantic model; the response will be
                    parsed into an instance of this model via JSON mode.

        Returns:
            A parsed Pydantic model instance if *schema* is provided,
            otherwise the plain response text.

        Does not reset last_model/total_usage_tokens itself, see the
        attributes' docstring in __init__ for why; callers that want a clean
        per-measurement total must reset them first.
        """
        try:
            return self._call(self._model, prompt, self._max_tokens, schema)
        except ValueError:
            logger.warning(
                "Judge returned empty content at max_tokens=%d, retrying at %d",
                self._max_tokens,
                self._max_tokens * 2,
            )

        try:
            return self._call(self._model, prompt, self._max_tokens * 2, schema)
        except ValueError:
            if OPENROUTER_JUDGE_FALLBACK_MODEL is None:
                raise
            logger.warning(
                "Judge still returned empty content after retry, falling back to %s",
                OPENROUTER_JUDGE_FALLBACK_MODEL,
            )

        return self._call(
            OPENROUTER_JUDGE_FALLBACK_MODEL, prompt, self._max_tokens * 2, schema
        )


def get_judge() -> OpenRouterJudge | None:
    """Instantiate and return the OpenRouter judge, or None if the API key is missing."""
    if not OPENROUTER_API_KEY:
        return None
    return OpenRouterJudge()


# Module-level singleton used by test files
_judge = get_judge()
