"""VLM (Vision-Language Model) module for AI Scientist.

This module provides functionality for interacting with vision-language models
for image analysis and understanding.
"""

from ai_scientist.vlm.constants import AVAILABLE_VLMS, MAX_NUM_TOKENS
from ai_scientist.vlm.clients import create_client
from ai_scientist.vlm.response import (
    get_response_from_vlm,
    get_batch_responses_from_vlm,
    make_vlm_call,
)
from ai_scientist.vlm.utils import encode_image_to_base64, extract_json_between_markers

__all__ = [
    # Constants
    "AVAILABLE_VLMS",
    "MAX_NUM_TOKENS",
    # Client creation
    "create_client",
    # Response functions
    "get_response_from_vlm",
    "get_batch_responses_from_vlm",
    "make_vlm_call",
    # Utilities
    "encode_image_to_base64",
    "extract_json_between_markers",
]
