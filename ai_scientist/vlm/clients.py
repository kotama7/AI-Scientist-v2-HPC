"""VLM client creation for various providers."""

import os
from typing import Any

import openai


def create_client(model: str) -> tuple[Any, str]:
    """Create a VLM client for the specified model.

    Args:
        model: Model identifier string.

    Returns:
        Tuple of (client, model_name) where client is the API client
        and model_name is the actual model identifier to use with the API.

    Raises:
        ValueError: If the model is not supported.
    """
    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o",
        "o3-mini",
        "gpt-5.2",
    ]:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model.startswith("ollama/"):
        print(f"Using Ollama API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
            base_url="http://localhost:11434/v1"
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")
