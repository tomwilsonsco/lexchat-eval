"""
Gemini judge model wrapper for DeepEval metrics.

Mirrors OpenAIJudge so both can be used interchangeably as DeepEval judge
models. Imported by lex_eval/utils/judge.py — do not import directly from
test files.
"""

from __future__ import annotations

import os
from pathlib import Path

from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from google import genai
from google.genai import types

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL: str = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")


class GeminiJudge(DeepEvalBaseLLM):
    """Thin DeepEval wrapper around the Google Gemini generate_content API."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = GEMINI_API_KEY,
    ) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return schema.model_validate_json(response.text)
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return schema.model_validate_json(response.text)
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    def get_model_name(self) -> str:
        return f"gemini/{self.model}"
