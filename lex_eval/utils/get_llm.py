import logging
from typing import Optional, Tuple

from .lexchat_client import get_authenticated_client

logger = logging.getLogger(__name__)


def get_active_model() -> Tuple[Optional[str], Optional[str]]:
    """
    Query LexChat's /api/models endpoint and find the active model.

    The active model is the one configured in LexChat's admin portal — this is
    the model that will actually process requests. The eval does not select or
    override the model; it reads it from the API.

    Returns:
        Tuple of (model_name, provider) where both are None if no active model
        was found. Provider is the provider name ("ollama" or "openrouter") for
        informational purposes only.
    """
    client = get_authenticated_client()
    try:
        models_response = client.get("/api/models")
        models_response.raise_for_status()
        models = models_response.json()
        for m in models:
            if m.get("active") is True:
                return m.get("name"), m.get("provider", "").lower()
        return None, None
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, provider = get_active_model()
    if model:
        print(f"\nActive model: {model} (provider: {provider})")
    else:
        print("\nNo active model found. Set one in LexChat's admin portal.")
