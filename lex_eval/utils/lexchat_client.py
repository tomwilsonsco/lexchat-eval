import httpx
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from parent directory's .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


def _get_required_env(key: str) -> str:
    """Get a required environment variable or raise ConfigError."""
    value = os.getenv(key)
    if not value:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


def _validate_config() -> dict[str, str]:
    """Validate and return configuration from environment variables."""
    try:
        base_url = _get_required_env("LEXCHAT_API")
        username = _get_required_env("USERNAME")
        password = _get_required_env("PASSWORD")

        return {"base_url": base_url, "username": username, "password": password}
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        logger.error(f"Please ensure .env file exists at: {env_path}")
        raise


def get_authenticated_client() -> httpx.Client:
    """
    Creates and returns an authenticated HTTP client for the LexChat API.

    The client is configured with the base URL from environment variables
    and authenticated using credentials from the .env file.

    Returns:
        httpx.Client: An authenticated HTTP client with bearer token set.

    Raises:
        ConfigError: If required environment variables are missing.
        httpx.HTTPError: If authentication request fails.

    Note:
        Caller is responsible for closing the client when done.
        Use in a context manager or call client.close() explicitly.
    """
    config = _validate_config()

    # Use longer timeout for LLM operations (5 minutes total)
    # Read timeout is especially important for streaming responses
    timeout = httpx.Timeout(300.0, read=300.0)
    client = httpx.Client(base_url=config["base_url"], timeout=timeout)

    logger.info("Authenticating with LexChat API...")
    try:
        # Perform login
        auth_data = {"username": config["username"], "password": config["password"]}
        resp = client.post("/api/auth/login", json=auth_data)
        resp.raise_for_status()

        token = resp.json().get("token")
        if not token:
            raise ValueError("No token received in authentication response")

        client.headers.update({"Authorization": f"Bearer {token}"})
        logger.info("Successfully authenticated with LexChat API")
        return client

    except httpx.HTTPError as e:
        client.close()
        logger.error(f"HTTP error during authentication: {e}")
        raise
    except (KeyError, ValueError) as e:
        client.close()
        logger.error(f"Invalid authentication response: {e}")
        raise
    except Exception as e:
        client.close()
        logger.error(f"Unexpected error during authentication: {e}")
        raise
