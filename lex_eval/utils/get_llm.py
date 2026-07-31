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


def get_summarisation_model() -> Tuple[Optional[str], Optional[str]]:
    """
    Return the summarisation model configured in LexChat's admin portal.

    Calls the admin-only /api/developer/provider-config endpoint to read
    summarisation_model from the active provider's config blob.  Mirrors the
    server-side fallback: if summarisation_model is blank, the main model is
    returned (because that is what LexChat actually uses for summarisation).

    Returns:
        Tuple of (model_name, provider).  Both are None only when the active
        model itself cannot be resolved.
    """
    client = get_authenticated_client()
    try:
        # Resolve the main model and active provider first (needed for fallback).
        models_response = client.get("/api/models")
        models_response.raise_for_status()
        main_model: Optional[str] = None
        active_provider: Optional[str] = None
        for m in models_response.json():
            if m.get("active") is True:
                main_model = m.get("name")
                active_provider = m.get("provider", "").lower()
                break

        if main_model is None:
            return None, None

        # Read the full provider config (admin endpoint).
        cfg_response = client.get("/api/developer/provider-config")
        cfg_response.raise_for_status()
        data = cfg_response.json()

        provider_key = active_provider or data.get("active_provider", "ollama")
        provider_cfgs = {
            p["id"]: p.get("config", {}) for p in data.get("providers", [])
        }
        cfg = provider_cfgs.get(provider_key, {})

        # Match server-side: cfg.get("summarisation_model") or model
        summ_model = cfg.get("summarisation_model") or main_model
        return summ_model, active_provider
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, provider = get_active_model()
    if model:
        print(f"\nActive model (manager/worker): {model}  (provider: {provider})")
    else:
        print("\nNo active model found. Set one in LexChat's admin portal.")

    summ_model, summ_provider = get_summarisation_model()
    if summ_model:
        if summ_model == model:
            print(
                f"Summarisation model:           {summ_model}"
                "  (same as active model — no separate summarisation model configured)"
            )
        else:
            print(
                f"Summarisation model:           {summ_model}  (provider: {summ_provider})"
            )
