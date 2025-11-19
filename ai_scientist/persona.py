"""Utilities for customizing the persona injected into prompts."""

from __future__ import annotations

from typing import Any

_PERSONA_PLACEHOLDERS = ("AI researcher", "AI Researcher")
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
    """Recursively replace persona placeholders in supported prompt structures."""
    if _persona_override is None:
        return value

    if isinstance(value, str):
        updated = value
        for placeholder in _PERSONA_PLACEHOLDERS:
            updated = updated.replace(placeholder, _persona_override)
        return updated

    if isinstance(value, list):
        return [apply_persona_override(item) for item in value]

    if isinstance(value, dict):
        return {k: apply_persona_override(v) for k, v in value.items()}

    return value
