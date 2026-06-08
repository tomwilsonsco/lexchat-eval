import logging
import os
from typing import List, Optional

from .lexchat_client import get_authenticated_client

logger = logging.getLogger(__name__)

# Default allowlist mirrors OPENROUTER_MODEL_LIST in LexChat/server_py/src/config.py
_OPENROUTER_DEFAULT_ALLOWLIST = ",".join(
    [
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-7",
        "google/gemini-2.5-pro",
        "google/gemini-2.0-flash",
        "openai/gpt-4o",
        "mistralai/mistral-large-2411",
        "deepseek/deepseek-r1",
    ]
)


def _load_openrouter_allowlist() -> set:
    """Load the OpenRouter model allowlist from env, with a sensible default."""
    raw = os.getenv("OPENROUTER_EVAL_MODELS", _OPENROUTER_DEFAULT_ALLOWLIST)
    return {m.strip() for m in raw.split(",") if m.strip()}


def get_llms(provider: Optional[str] = None) -> tuple[List[str], List[dict]]:
    """
    Retrieves a list of available LLM names from the LexChat API.

    Args:
        provider: Optional filter — 'ollama' or 'openrouter'.  If given, only
                  models from that provider are returned.  For 'openrouter',
                  results are additionally restricted to the
                  OPENROUTER_EVAL_MODELS env var allowlist.

    Returns:
        Tuple of (llm_names: List[str], all_models: List[dict]) where
        all_models is the raw model list including provider metadata.

    Raises:
        httpx.HTTPError: If the API request fails.
    """
    client = get_authenticated_client()

    try:
        logger.info("Fetching available LLMs...")
        models_response = client.get("/api/models")
        models_response.raise_for_status()
        models = models_response.json()

        if provider:
            models = [
                m for m in models if m.get("provider", "").lower() == provider.lower()
            ]
            logger.info(f"Filtered to provider '{provider}': {len(models)} model(s)")

        # When using OpenRouter, additionally restrict to the env allowlist
        # so we don't end up evaluating 100s of models.
        if provider == "openrouter":
            allowlist = _load_openrouter_allowlist()
            before = len(models)
            models = [m for m in models if m.get("name") in allowlist]
            excluded = before - len(models)
            if excluded:
                logger.info(
                    "OpenRouter allowlist excluded %d model(s) not in OPENROUTER_EVAL_MODELS "
                    "(allowed: %s)",
                    excluded,
                    ", ".join(sorted(allowlist)),
                )

        # Extract just the names from the models
        llm_names = [m.get("name") for m in models if m.get("name")]

        logger.info(f"Found {len(llm_names)} LLMs: {', '.join(llm_names)}")
        return llm_names, models

    finally:
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List LLMs available via LexChat API")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter"],
        help="Filter models by provider",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    provider_label = f" ({args.provider})" if args.provider else ""
    llms, _all = get_llms(provider=args.provider)
    print(f"\nAvailable LLMs{provider_label} ({len(llms)}):")
    for llm in llms:
        print(f"  - {llm}")
