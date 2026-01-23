"""File utilities for reading, summarizing, and processing files.

This module provides utility functions for file operations commonly used
in the treesearch module.
"""

import json
from pathlib import Path
from typing import Any

from ai_scientist.treesearch.utils.phase_execution import summarize_text


def read_text(path: Path) -> str:
    """Read text from a file, returning empty string on error.

    Args:
        path: Path to the file.

    Returns:
        File contents as string, or empty string on error.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def summarize_file(path: Path, *, max_lines: int = 40, max_chars: int = 3000) -> str:
    """Summarize a file's contents to a limited size.

    Args:
        path: Path to the file.
        max_lines: Maximum number of lines to include.
        max_chars: Maximum number of characters to include.

    Returns:
        Summarized file contents.
    """
    text = read_text(path)
    return summarize_text(text, max_lines=max_lines, max_chars=max_chars)


def find_previous_run_dir(current_log_dir: Path) -> Path | None:
    """Find the previous run directory based on naming convention.

    Assumes directories are named with a numeric prefix (e.g., '001-run').

    Args:
        current_log_dir: Path to the current log directory.

    Returns:
        Path to the previous run directory, or None if not found.
    """
    parent = current_log_dir.parent
    current_name = current_log_dir.name
    try:
        current_index = int(current_name.split("-", 1)[0])
    except (ValueError, IndexError):
        current_index = None
    candidates: list[tuple[int, Path]] = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name.split("-", 1)[0])
        except (ValueError, IndexError):
            continue
        if current_index is None or idx < current_index:
            candidates.append((idx, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def summarize_phase1_steps(path: Path) -> dict[str, Any]:
    """Summarize phase 1 installation steps from a JSONL log.

    Args:
        path: Path to the phase 1 steps log file.

    Returns:
        Summary dictionary with step counts, failed steps, and dependencies.
    """
    summary: dict[str, Any] = {
        "total_steps": 0,
        "failed_steps": [],
        "dependencies": {"apt": set(), "pip": set(), "source": set()},
        "recent_commands": [],
    }
    text = read_text(path)
    if not text:
        return summary
    entries = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    summary["total_steps"] = len(entries)
    for entry in entries:
        cmd = str(entry.get("command", "")).strip()
        if not cmd:
            continue
        exit_code = entry.get("exit_code")
        if exit_code not in (0, None):
            summary["failed_steps"].append(
                {
                    "step": entry.get("step"),
                    "command": cmd,
                    "exit_code": exit_code,
                    "stderr_summary": entry.get("stderr_summary", ""),
                }
            )
        if "apt-get" in cmd and "install" in cmd:
            segment = cmd.split("install", 1)[-1]
            packages = segment.replace("-y", " ").split()
            summary["dependencies"]["apt"].update([p for p in packages if p and not p.startswith("-")])
        if "pip install" in cmd:
            segment = cmd.split("pip install", 1)[-1]
            packages = [p for p in segment.split() if not p.startswith("-")]
            summary["dependencies"]["pip"].update(packages)
        if "git clone" in cmd:
            summary["dependencies"]["source"].add(cmd)
    summary["recent_commands"] = [
        {"step": e.get("step"), "command": e.get("command"), "exit_code": e.get("exit_code")}
        for e in entries[-5:]
    ]
    summary["dependencies"] = {
        key: sorted(value) for key, value in summary["dependencies"].items()
    }
    return summary


def extract_error_lines(text: str, *, max_lines: int = 12) -> list[str]:
    """Extract error-related lines from text.

    Args:
        text: Text to scan for errors.
        max_lines: Maximum number of error lines to return.

    Returns:
        List of lines containing error keywords.
    """
    if not text:
        return []
    hits = []
    for line in text.splitlines():
        lower = line.lower()
        if "error" in lower or "failed" in lower or "undefined reference" in lower or "not found" in lower:
            hits.append(line.strip())
    return hits[-max_lines:]


def summarize_phase_logs(phase_log_dir: Path) -> dict[str, Any]:
    """Summarize compile and run logs from a phase log directory.

    Args:
        phase_log_dir: Path to the phase log directory.

    Returns:
        Summary dictionary with log summaries and error lines.
    """
    summary: dict[str, Any] = {}
    if not phase_log_dir.exists():
        return summary
    node_dirs = [d for d in phase_log_dir.iterdir() if d.is_dir() and d.name.startswith("node_")]
    if not node_dirs:
        return summary
    node_dirs.sort(key=lambda d: d.stat().st_mtime)
    latest = node_dirs[-1]
    compile_log = latest / "compile.log"
    run_log = latest / "run.log"
    if compile_log.exists():
        compile_text = read_text(compile_log)
        summary["compile_log_summary"] = summarize_file(compile_log)
        summary["compile_errors"] = extract_error_lines(compile_text)
    if run_log.exists():
        run_text = read_text(run_log)
        summary["run_log_summary"] = summarize_file(run_log)
        summary["run_errors"] = extract_error_lines(run_text)
    return summary


def summarize_journal_outputs(prev_log_dir: Path) -> dict[str, Any]:
    """Summarize outputs from a previous run's journal.

    Args:
        prev_log_dir: Path to the previous run's log directory.

    Returns:
        Summary dictionary with workspace tree, file paths, and build plan.
    """
    summary: dict[str, Any] = {}
    journal_files = list(prev_log_dir.rglob("journal.json"))
    if not journal_files:
        return summary
    journal_files.sort(key=lambda p: p.stat().st_mtime)
    latest = journal_files[-1]
    raw = read_text(latest)
    if not raw:
        return summary
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return summary
    nodes = parsed.get("nodes") if isinstance(parsed, dict) else None
    if not isinstance(nodes, list):
        return summary
    phase_node = None
    for node in nodes:
        if isinstance(node, dict) and node.get("phase_artifacts"):
            phase_node = node
    if not phase_node:
        return summary
    artifacts = phase_node.get("phase_artifacts", {})
    if isinstance(artifacts, dict) and artifacts.get("phase_artifacts"):
        artifacts = artifacts.get("phase_artifacts")
    coding = artifacts.get("coding", {}) if isinstance(artifacts, dict) else {}
    workspace = coding.get("workspace", {}) if isinstance(coding, dict) else {}
    files = workspace.get("files", []) if isinstance(workspace, dict) else []
    file_paths = [f.get("path") for f in files if isinstance(f, dict) and f.get("path")]
    tree = workspace.get("tree", []) if isinstance(workspace, dict) else []
    compile_section = artifacts.get("compile", {}) if isinstance(artifacts, dict) else {}
    build_plan = compile_section.get("build_plan", {}) if isinstance(compile_section, dict) else {}
    summary = {
        "workspace_tree": tree[:20],
        "file_paths": file_paths[:20],
        "build_plan": {
            "language": build_plan.get("language"),
            "compiler_selected": build_plan.get("compiler_selected"),
            "workdir": build_plan.get("workdir"),
            "output": build_plan.get("output"),
        },
    }
    return summary


# Backward compatibility aliases (prefixed versions)
_read_text = read_text
_summarize_file = summarize_file
_find_previous_run_dir = find_previous_run_dir
_summarize_phase1_steps = summarize_phase1_steps
_extract_error_lines = extract_error_lines
_summarize_phase_logs = summarize_phase_logs
_summarize_journal_outputs = summarize_journal_outputs
