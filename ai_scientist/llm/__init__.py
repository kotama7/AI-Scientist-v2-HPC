"""LLM (Large Language Model) module for AI Scientist.

This module provides functionality for interacting with various LLM providers
including OpenAI, Anthropic, Google, and local Ollama models.
"""

from ai_scientist.llm.constants import AVAILABLE_LLMS, MAX_NUM_TOKENS, DEFAULT_MAX_COMPLETION_TOKENS
from ai_scientist.llm.clients import create_client
from ai_scientist.llm.response import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    make_llm_call,
)
from ai_scientist.llm.utils import extract_json_between_markers

__all__ = [
    # Constants
    "AVAILABLE_LLMS",
    "MAX_NUM_TOKENS",
    "DEFAULT_MAX_COMPLETION_TOKENS",
    # Client creation
    "create_client",
    # Response functions
    "get_response_from_llm",
    "get_batch_responses_from_llm",
    "make_llm_call",
    # Utilities
    "extract_json_between_markers",
]
