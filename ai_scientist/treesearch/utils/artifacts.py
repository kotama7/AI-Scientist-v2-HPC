"""Artifact management utilities.

This module provides utility functions for managing experiment artifacts,
including copying files, formatting prompts, and saving phase execution results.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ai_scientist.treesearch.backend import compile_prompt_to_md
from ai_scientist.treesearch.utils.config import Config


def resolve_run_root(cfg: Config) -> Path:
    """Resolve the run root directory from config or environment.

    Args:
        cfg: Configuration object.

    Returns:
        Path to the run root directory.
    """
    run_root_env = os.environ.get("AI_SCIENTIST_RUN_ROOT")
    if run_root_env:
        run_root = Path(run_root_env).expanduser().resolve()
    else:
        run_root = Path(cfg.log_dir).parent / "runs"  # 実験ディレクトリ内に配置
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def copy_artifact(src: Path, dest_dir: Path, *, name: str | None = None) -> None:
    """Copy an artifact file to a destination directory.

    Args:
        src: Source file path.
        dest_dir: Destination directory.
        name: Optional name for the destination file.
    """
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (name or src.name)
    try:
        shutil.copy2(src, dest_path)
    except OSError:
        pass


def format_prompt_log_name(label: str, *, session_id: str | None = None, counter: int | None = None) -> str:
    """Format a prompt log filename.

    Args:
        label: Label for the prompt.
        session_id: Optional session identifier.
        counter: Optional numeric counter.

    Returns:
        Formatted filename string.
    """
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "prompt"
    parts: list[str] = []
    if session_id:
        parts.append(session_id)
    if counter is not None:
        parts.append(f"{counter:03d}")
    parts.append(safe_label)
    return "_".join(parts)


def render_prompt_for_log(prompt: Any) -> str:
    """Render a prompt object to a log-friendly string.

    Args:
        prompt: Prompt object (may be dict, list, or other).

    Returns:
        String representation suitable for logging.
    """
    rendered = compile_prompt_to_md(prompt)
    if isinstance(rendered, (list, dict)):
        return json.dumps(rendered, indent=2, default=str)
    return str(rendered)


def write_prompt_log(
    log_dir: Path,
    name: str,
    prompt: Any,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write a prompt to log files (JSON and markdown).

    Args:
        log_dir: Directory to write logs to.
        name: Base name for the log files.
        prompt: Prompt object to log.
        meta: Optional metadata dictionary.
    """
    if prompt is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid circular imports
    from ai_scientist.persona import apply_persona_override

    # Apply persona override to the prompt object
    # so that the JSON log also reflects the substitution
    safe_prompt = apply_persona_override(prompt)

    payload = {
        "meta": meta or {},
        "prompt": safe_prompt,
    }
    (log_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    rendered = render_prompt_for_log(prompt)
    (log_dir / f"{name}.md").write_text(rendered, encoding="utf-8")


def save_phase_execution_artifacts(
    *,
    exp_results_dir: Path,
    phase_log_dir: Path | None,
    run_root: Path | None,
    worker_label: str,
    phase_artifacts: dict | None = None,
    phase_artifacts_raw: str | None = None,
) -> None:
    """Save phase execution artifacts to the experiment results directory.

    Args:
        exp_results_dir: Directory to save artifacts to.
        phase_log_dir: Directory containing phase logs.
        run_root: Run root directory.
        worker_label: Label for the worker.
        phase_artifacts: Parsed phase artifacts dictionary.
        phase_artifacts_raw: Raw phase artifacts text.
    """
    artifacts_dir = exp_results_dir / "phase_artifacts"
    llm_outputs_dir = exp_results_dir / "llm_outputs"
    if phase_log_dir and phase_log_dir.exists():
        for log_name in ("download.log", "coding.log", "compile.log", "run.log"):
            copy_artifact(phase_log_dir / log_name, artifacts_dir, name=log_name)
        prompt_log_dir = phase_log_dir / "prompt_logs"
        if prompt_log_dir.exists():
            for prompt_file in prompt_log_dir.iterdir():
                if prompt_file.is_file():
                    copy_artifact(prompt_file, llm_outputs_dir / "prompt_logs")
    if run_root:
        plans_dir = run_root / "workers" / worker_label / "plans"
        copy_artifact(plans_dir / "phase0_plan.json", llm_outputs_dir)
        copy_artifact(plans_dir / "phase0_history_full.json", llm_outputs_dir)
        copy_artifact(plans_dir / "phase0_llm_output.txt", llm_outputs_dir)
        copy_artifact(run_root / "workers" / worker_label / "phase1_steps.jsonl", llm_outputs_dir)
        copy_artifact(run_root / "workers" / worker_label / "phase1_llm_outputs.jsonl", llm_outputs_dir)
        prompt_root = run_root / "workers" / worker_label / "prompt_logs"
        for prompt_name in ("phase0_prompt.json", "phase0_prompt.md"):
            copy_artifact(prompt_root / prompt_name, llm_outputs_dir / "prompt_logs")
    if phase_artifacts:
        llm_outputs_dir.mkdir(parents=True, exist_ok=True)
        phase_data = phase_artifacts.get("phase_artifacts") if isinstance(phase_artifacts, dict) else None
        if not isinstance(phase_data, dict):
            phase_data = phase_artifacts if isinstance(phase_artifacts, dict) else {}
        try:
            (llm_outputs_dir / "phase2_4_llm_output.json").write_text(
                json.dumps(phase_artifacts, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            if phase_data:
                (llm_outputs_dir / "phase2_llm_output.json").write_text(
                    json.dumps(phase_data.get("coding", {}), indent=2),
                    encoding="utf-8",
                )
                (llm_outputs_dir / "phase3_llm_output.json").write_text(
                    json.dumps(phase_data.get("compile", {}), indent=2),
                    encoding="utf-8",
                )
                (llm_outputs_dir / "phase4_llm_output.json").write_text(
                    json.dumps(phase_data.get("run", {}), indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass
    if phase_artifacts_raw:
        llm_outputs_dir.mkdir(parents=True, exist_ok=True)
        try:
            (llm_outputs_dir / "phase2_4_llm_output_raw.txt").write_text(
                phase_artifacts_raw,
                encoding="utf-8",
            )
        except Exception:
            pass


# Backward compatibility aliases (prefixed versions)
_resolve_run_root = resolve_run_root
_copy_artifact = copy_artifact
_format_prompt_log_name = format_prompt_log_name
_render_prompt_for_log = render_prompt_for_log
_write_prompt_log = write_prompt_log
_save_phase_execution_artifacts = save_phase_execution_artifacts
