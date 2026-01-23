"""Parsing utilities for JSON and text processing.

This module provides utility functions for parsing and normalizing
JSON responses and text from LLM outputs.
"""

import ast
import json
from typing import Any


def normalize_language(language: str | None) -> str:
    """Normalize a programming language string.

    Args:
        language: Language string (may be None or various formats).

    Returns:
        Normalized language string (defaults to 'python').
    """
    lang = str(language or "").strip().lower()
    if not lang:
        return "python"
    if lang in {"c++", "cxx"}:
        return "cpp"
    return lang


def strip_json_wrappers(raw_text: str) -> str:
    """Strip markdown code block wrappers from JSON text.

    Args:
        raw_text: Text that may contain JSON wrapped in code blocks.

    Returns:
        Cleaned JSON text.
    """
    cleaned = raw_text.strip()
    if "```" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def parse_json_object(raw_text: str, *, context: str) -> dict[str, Any]:
    """Parse a JSON object from text, with fallback to ast.literal_eval.

    Args:
        raw_text: Text containing JSON.
        context: Description of context for error messages.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If parsing fails or result is not a dict.
    """
    cleaned = strip_json_wrappers(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(cleaned)
        except Exception as exc:
            raise ValueError(f"{context}: failed to parse JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{context}: response must be a JSON object.")
    return parsed


def normalize_phase0_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize a phase0 plan to ensure all required fields exist.

    Args:
        plan: Raw plan dictionary from LLM.

    Returns:
        Normalized plan with all required fields.
    """
    plan_blob = plan.get("plan") if isinstance(plan.get("plan"), dict) else plan
    if not isinstance(plan_blob, dict):
        plan_blob = {}
    plan_blob.setdefault("goal_summary", "")
    plan_blob.setdefault("implementation_strategy", [])
    plan_blob.setdefault("dependencies", {"apt": [], "pip": [], "source": []})
    deps = plan_blob.get("dependencies")
    if not isinstance(deps, dict):
        deps = {"apt": [], "pip": [], "source": []}
    deps.setdefault("apt", [])
    deps.setdefault("pip", [])
    deps.setdefault("source", [])
    plan_blob["dependencies"] = deps
    phase_guidance = plan_blob.get("phase_guidance")
    if not isinstance(phase_guidance, dict):
        phase_guidance = {}
    phase_guidance.setdefault("phase1", {"targets": [], "preferred_commands": [], "done_conditions": []})
    phase_guidance.setdefault("phase2", {"targets": [], "notes": ""})
    phase_guidance.setdefault("phase3", {"compiler_selection_policy": "", "notes": ""})
    phase_guidance.setdefault("phase4", {"output_policy": "", "validation": []})
    plan_blob["phase_guidance"] = phase_guidance
    plan_blob.setdefault("risks_and_mitigations", [])
    return {"plan": plan_blob}


# Backward compatibility aliases (prefixed versions)
_normalize_language = normalize_language
_strip_json_wrappers = strip_json_wrappers
_parse_json_object = parse_json_object
_normalize_phase0_plan = normalize_phase0_plan
