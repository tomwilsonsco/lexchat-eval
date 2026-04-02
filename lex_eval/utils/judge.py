"""
Unified judge factory for DeepEval metrics.

Reads JUDGE_PROVIDER from the environment (default: 'openai') and returns
the appropriate DeepEvalBaseLLM instance, or None if the required API key
for the chosen provider is not set.

Usage in test files:
    from lex_eval.utils.judge import _judge
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

_PROVIDER: str = os.environ.get("JUDGE_PROVIDER", "openai").lower().strip()


def get_judge():
    """Instantiate and return the judge for the configured provider, or None."""
    if _PROVIDER == "gemini":
        from lex_eval.utils.gemini_judge import GeminiJudge, GEMINI_API_KEY

        return GeminiJudge() if GEMINI_API_KEY else None
    else:
        from lex_eval.utils.openai_judge import OpenAIJudge, OPENAI_API_KEY

        return OpenAIJudge() if OPENAI_API_KEY else None


# Module-level singleton used by test files
_judge = get_judge()
