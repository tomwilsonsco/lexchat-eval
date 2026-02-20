import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lexchat_client import get_authenticated_client

logger = logging.getLogger(__name__)


def get_llms() -> List[str]:
    """
    Retrieves a list of available LLM names from the LexChat API.

    Returns:
        List[str]: List of LLM names available in the system.

    Raises:
        httpx.HTTPError: If the API request fails.
    """
    client = get_authenticated_client()

    try:
        logger.info("Fetching available LLMs...")
        models_response = client.get("/api/models")
        models_response.raise_for_status()
        models = models_response.json()

        # Extract just the names from the models
        llm_names = [m.get("name") for m in models if m.get("name")]

        logger.info(f"Found {len(llm_names)} LLMs: {', '.join(llm_names)}")
        return llm_names

    finally:
        client.close()


if __name__ == "__main__":
    # For testing purposes
    logging.basicConfig(level=logging.INFO)
    llms = get_llms()
    print(f"\Available LLMs ({len(llms)}):")
    for llm in llms:
        print(f"  - {llm}")
