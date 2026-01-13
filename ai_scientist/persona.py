"""Utilities for customizing the persona injected into prompts."""

from __future__ import annotations

from typing import Any

_PERSONA_PLACEHOLDERS = ("AI researcher", "AI Researcher")
_PERSONA_TOKENS = ("{persona}",)
_DEFAULT_PERSONA = "AI researcher"
_persona_override: str | None = None


def set_persona_role(role_description: str | None) -> None:
    """Configure the persona role string that replaces default placeholders."""
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
    """Recursively replace persona placeholders and tokens in prompt structures."""
    if isinstance(value, str):
        updated = value
        if _persona_override is not None:
            for placeholder in _PERSONA_PLACEHOLDERS:
                updated = updated.replace(placeholder, _persona_override)
        persona_text = _persona_override or _DEFAULT_PERSONA
        for token in _PERSONA_TOKENS:
            updated = updated.replace(token, persona_text)
        return updated

    if isinstance(value, list):
        return [apply_persona_override(item) for item in value]

    if isinstance(value, dict):
        return {k: apply_persona_override(v) for k, v in value.items()}

    return value
