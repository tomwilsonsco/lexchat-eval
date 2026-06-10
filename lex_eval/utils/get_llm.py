import logging
from typing import Optional

from .lexchat_client import get_authenticated_client

logger = logging.getLogger(__name__)


def get_active_model(provider: str) -> Optional[str]:
    """
    Query LexChat's /api/models endpoint and find the active model for a provider.

    The active model is the one configured in LexChat's admin portal — this is
    the model that will actually process requests. The eval does not select or
    override the model; it reads it from the API.

    Args:
        provider: 'ollama' or 'openrouter'.

    Returns:
        The name of the active model, or None if not found.
    """
    client = get_authenticated_client()
    try:
        models_response = client.get("/api/models")
        models_response.raise_for_status()
        models = models_response.json()
        for m in models:
            if (
                m.get("provider", "").lower() == provider.lower()
                and m.get("active") is True
            ):
                return m.get("name")
        return None
    finally:
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Show the active LLM configured in LexChat's admin portal"
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter"],
        default="openrouter",
        help="Provider to query (default: openrouter)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = get_active_model(args.provider)
    if model:
        print(f"\nActive {args.provider} model: {model}")
    else:
        print(f"\nNo active {args.provider} model found. Set one in LexChat's admin portal.")