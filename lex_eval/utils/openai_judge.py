"""
Shared OpenAI judge model wrapper for DeepEval metrics.

Imported by test_groundedness.py and test_consistency_llm.py so the model
setup lives in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import openai
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL: str = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")


class OpenAIJudge(DeepEvalBaseLLM):
    """Thin DeepEval wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = OPENAI_API_KEY,
    ) -> None:
        self.model = model
        self._client = openai.OpenAI(api_key=api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            response = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
            )
            return response.choices[0].message.parsed
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            response = await self._async_client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
            )
            return response.choices[0].message.parsed
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return f"openai/{self.model}"
