"""Utilities for customizing the persona injected into prompts."""

from __future__ import annotations

from typing import Any

_PERSONA_TOKEN = "{persona}"
_DEFAULT_PERSONA = "AI researcher"
_persona_override: str | None = None


def set_persona_role(role_description: str | None) -> None:
    """Configure the persona role string that replaces {persona} tokens."""
    global _persona_override
    if role_description is None:
        _persona_override = None
        return

    normalized = role_description.strip()
    _persona_override = normalized if normalized else None


def get_persona_role() -> str | None:
    """Return the currently configured persona description, if any."""
    return _persona_override


def apply_persona_override(value: Any) -> Any:
    """Recursively replace {persona} tokens in prompt structures."""
    if isinstance(value, str):
        persona_text = _persona_override or _DEFAULT_PERSONA
        return value.replace(_PERSONA_TOKEN, persona_text)

    if isinstance(value, list):
        return [apply_persona_override(item) for item in value]

    if isinstance(value, dict):
        return {k: apply_persona_override(v) for k, v in value.items()}

    return value
