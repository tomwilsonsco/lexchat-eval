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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
_DEFAULT_MODEL = "openai/gpt-4o"
OPENROUTER_JUDGE_MODEL: str = os.getenv("OPENROUTER_JUDGE_MODEL", _DEFAULT_MODEL)


class OpenRouterJudge:
    """A lightweight judge that calls OpenRouter's chat completions API."""

    def __init__(self, model: str | None = None) -> None:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set; cannot create OpenRouterJudge"
            )
        self._model = model or OPENROUTER_JUDGE_MODEL
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Send a prompt to the OpenRouter model and return the response.

        Args:
            prompt: The prompt string to send.
            schema: Optional Pydantic model; the response will be
                    parsed into an instance of this model via JSON mode.

        Returns:
            A parsed Pydantic model instance if *schema* is provided,
            otherwise the plain response text.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""

            if schema is not None:
                data = json.loads(content)
                return schema(**data)

            return content

        except Exception:
            logger.exception("OpenRouter judge call failed")
            raise


def get_judge() -> OpenRouterJudge | None:
    """Instantiate and return the OpenRouter judge, or None if the API key is missing."""
    if not OPENROUTER_API_KEY:
        return None
    return OpenRouterJudge()


# Module-level singleton used by test files
_judge = get_judge()