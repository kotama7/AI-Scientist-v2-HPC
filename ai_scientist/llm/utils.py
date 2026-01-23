"""Utility functions for LLM module."""

import json
import os
import re
from datetime import datetime
from typing import Any

from ai_scientist.llm.constants import DEFAULT_MAX_COMPLETION_TOKENS
from ai_scientist.utils.model_params import build_token_params


def default_completion_tokens(model: str) -> int | None:
    """Get the default completion tokens limit for a model.

    Args:
        model: Model identifier string.

    Returns:
        Default token limit or None for server-side default.
    """
    normalized = (model or "").lower()
    if normalized.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3")):
        # Let OpenAI models use their server-side default max.
        return None
    return DEFAULT_MAX_COMPLETION_TOKENS


def token_param(model: str, *, n_tokens: int | None = None) -> dict[str, int]:
    """Return the correct token budget kwarg for the given model.

    Args:
        model: Model identifier string.
        n_tokens: Optional specific token limit.

    Returns:
        Dictionary with appropriate token parameter.
    """
    token_budget = default_completion_tokens(model) if n_tokens is None else n_tokens
    return build_token_params(model, token_budget)


def normalize_openai_content(content: Any) -> str:
    """Normalize OpenAI response content to a string.

    Args:
        content: Content from OpenAI response (various types).

    Returns:
        Normalized string content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            value = content.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (list, dict)):
                nested = normalize_openai_content(value)
                if nested:
                    return nested
        refusal = content.get("refusal")
        if isinstance(refusal, str) and refusal:
            return refusal
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            normalized = normalize_openai_content(part)
            if normalized:
                parts.append(normalized)
        return "".join(parts)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return ""


def extract_openai_message_text(message: Any) -> str:
    """Extract text from an OpenAI message object.

    Args:
        message: Message object from OpenAI response.

    Returns:
        Extracted text content.
    """
    if message is None:
        return ""
    if isinstance(message, dict):
        return normalize_openai_content(message.get("content"))
    return normalize_openai_content(getattr(message, "content", None))


def extract_openai_refusal(message: Any) -> str:
    """Extract refusal message from an OpenAI response.

    Args:
        message: Message object from OpenAI response.

    Returns:
        Refusal text if present, empty string otherwise.
    """
    if isinstance(message, dict):
        refusal = message.get("refusal")
    else:
        refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal
    return ""


def extract_openai_response_text(response: Any) -> str:
    """Extract text content from an OpenAI API response.

    Args:
        response: Response object from OpenAI API.

    Returns:
        Extracted text content.
    """
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or choices[0].get("delta") or {}
            content = extract_openai_message_text(message)
            if content:
                return content
            refusal = extract_openai_refusal(message)
            if refusal:
                return refusal

        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = response.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                text = normalize_openai_content(item)
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts)

    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        content = extract_openai_message_text(message)
        if content:
            return content
        refusal = extract_openai_refusal(message)
        if refusal:
            return refusal

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                item_content = item.get("content")
            else:
                item_content = getattr(item, "content", None)
            text = normalize_openai_content(item_content)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)

    if hasattr(response, "model_dump"):
        try:
            response_dict = response.model_dump()
        except Exception:
            response_dict = None
        if isinstance(response_dict, dict):
            return extract_openai_response_text(response_dict)

    return ""


def dump_empty_llm_response(model: str, response: Any) -> str | None:
    """Dump an empty LLM response to a file for debugging.

    Args:
        model: Model identifier string.
        response: The response object that was empty.

    Returns:
        Path to the dump file if successful, None otherwise.
    """
    if response is None:
        return None
    root_dir = os.environ.get("AI_SCIENTIST_ROOT") or os.getcwd()
    dump_dir = os.path.join(root_dir, "logs", "llm_empty")
    try:
        os.makedirs(dump_dir, exist_ok=True)
    except OSError:
        return None
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model or "unknown")
    dump_path = os.path.join(dump_dir, f"{safe_model}_{timestamp}.json")
    payload = {"model": model, "timestamp": timestamp}
    if hasattr(response, "model_dump"):
        try:
            payload["response"] = response.model_dump()
        except Exception:
            payload["repr"] = repr(response)
    elif isinstance(response, dict):
        payload["response"] = response
    else:
        payload["repr"] = repr(response)
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return dump_path
    except OSError:
        return None


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Extract JSON content from LLM output.

    Looks for JSON content between ```json and ``` markers,
    or attempts to find any JSON-like content if markers are not found.

    Args:
        llm_output: Raw output string from LLM.

    Returns:
        Parsed JSON as a dictionary, or None if no valid JSON found.
    """
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found
