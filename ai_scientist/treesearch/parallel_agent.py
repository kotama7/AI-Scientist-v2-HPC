"""Parallel agent for experiment execution.

This module provides the ParallelAgent and MinimalAgent classes for orchestrating
parallel experiment execution using tree search.
"""

import ast
import base64
import copy
import filelock
import humanize
import json
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import numpy as np
import os
import pickle
import random
import re
import shutil
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union

from rich import print
from rich.markup import escape

from ai_scientist.memory import DatabaseWriterProcess, MemoryManager
from ai_scientist.prompt_loader import format_prompt, load_prompt, load_prompt_lines

# Import from refactored modules
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .worker import WorkerManager, WorkerResult, WorkerTask
from .gpu import GPUManager, parse_cuda_visible_devices as _parse_cuda_visible_devices
from .ablation import AblationConfig, AblationIdea, HyperparamTuningIdea
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code, extract_memory_updates, remove_memory_update_tags, check_malformed_memory_update, MalformedMemoryUpdateError
from .utils.parsing import (
    normalize_language as _normalize_language,
    strip_json_wrappers as _strip_json_wrappers,
    parse_json_object as _parse_json_object,
    normalize_phase0_plan as _normalize_phase0_plan,
)
from .utils.file_utils import (
    read_text as _read_text,
    summarize_file as _summarize_file,
    find_previous_run_dir as _find_previous_run_dir,
    summarize_phase1_steps as _summarize_phase1_steps,
    extract_error_lines as _extract_error_lines,
    summarize_phase_logs as _summarize_phase_logs,
    summarize_journal_outputs as _summarize_journal_outputs,
    get_experiment_output_filename as _get_experiment_output_filename,
    sanitize_experiment_name as _sanitize_experiment_name,
    get_experiment_output_pattern as _get_experiment_output_pattern,
    DEFAULT_EXPERIMENT_OUTPUT_FILENAME,
)
from .utils.artifacts import (
    resolve_run_root as _resolve_run_root,
    copy_artifact as _copy_artifact,
    format_prompt_log_name as _format_prompt_log_name,
    render_prompt_for_log as _render_prompt_for_log,
    write_prompt_log as _write_prompt_log,
    save_phase_execution_artifacts as _save_phase_execution_artifacts,
)
from .utils.phase_execution import (
    ExecutionEnvironment,
    SingularityWorkerContainer,
    collect_available_compilers,
    collect_available_libs,
    collect_installed_system_packages,
    collect_system_performance_tools,
    run_in_container,
    summarize_command_output,
    summarize_text,
)
from .utils.phase_plan import (
    PhasePlanError,
    apply_workspace_plan,
    combine_sources_for_display,
    extract_phase_artifacts,
    wrap_combined_code,
    MissingMemoryUpdateError,
    MemoryReadOnlyResponse,
)
from .utils.resource import (
    ResourceConfig,
    build_local_binds,
    build_resources_context,
    load_resources,
    resolve_resources_path,
)
from .worker_plan import resolve_worker_plan

logger = logging.getLogger("ai-scientist")


# ============================================================================
# NOTE: The following classes have been moved to separate modules:
# - WorkerTask, WorkerResult, WorkerManager -> ai_scientist.treesearch.worker
# - AblationConfig, AblationIdea, HyperparamTuningIdea -> ai_scientist.treesearch.ablation
# - GPUManager -> ai_scientist.treesearch.gpu
# The utility functions (_normalize_language, _read_text, etc.) have been
# moved to ai_scientist.treesearch.utils submodules.
# ============================================================================


ExecCallbackType = Callable[[str, bool], ExecutionResult]


# Phase name mapping for memory event logging
# Maps internal task_hint values to human-readable phase names
# Phase structure:
#   Phase 0: Setup/Planning (idea loading, resource indexing, planning)
#   Phase 1: Environment Setup (container setup, dependencies installation)
#   Phase 2: Code Implementation (draft, debug, improve)
#   Phase 3: Compile (build the code)
#   Phase 4: Execute/Validate (run experiments, analyze results)
PHASE_NAME_MAP: Dict[str, str] = {
    # Phase 0: Setup and planning
    "phase0": "Phase 0: Setup",
    "phase0_planning": "Phase 0: Planning",
    "ingest_phase0_internal_info": "Phase 0: Setup",
    "idea_md": "Phase 0: Idea Loading",
    "resource_index": "Phase 0: Resource Indexing",
    # Phase 1: Environment setup
    "phase1_iterative": "Phase 1: Environment Setup",
    "phase1-base": "Phase 1: Base Setup",
    "phase1-sandbox": "Phase 1: Sandbox Setup",
    # Phase 2: Code implementation
    "draft": "Phase 2: Draft Implementation",
    "debug": "Phase 2: Debug",
    "improve": "Phase 2: Improve",
    "hyperparam_node": "Phase 2: Hyperparameter Tuning",
    "ablation_node": "Phase 2: Ablation Study",
    # Phase 3: Compile
    "compile": "Phase 3: Compile",
    "compiler_selection": "Phase 3: Compiler Selection",
    # Phase 4: Execute
    "execution_review": "Phase 4: Execution Review",
    # Post-Phase 4 analysis tasks (displayed as-is without phase number)
    # metrics_extraction, parse_metrics, plotting_code, seed_plotting,
    # vlm_analysis, datasets_successfully_tested, stage_completion
    # are not mapped - they will be displayed as-is
}


def get_phase_display_name(task_hint: str) -> str:
    """Convert internal task_hint to human-readable phase name.

    Args:
        task_hint: Internal task hint string (e.g., "draft", "debug")

    Returns:
        Human-readable phase name (e.g., "Phase 2: Draft Implementation")
        or the original task_hint if not in the mapping (for post-Phase 4 tasks)
    """
    # Direct lookup first
    if task_hint in PHASE_NAME_MAP:
        return PHASE_NAME_MAP[task_hint]

    # Handle stage-specific patterns (e.g., "stage_1_..._summary", "1_initial_implementation_...")
    if task_hint.startswith("stage_") and "_summary" in task_hint:
        return "stage_summary"

    # Handle numeric stage prefixes (e.g., "1_initial_implementation_1_preliminary")
    if task_hint and task_hint[0].isdigit() and "_" in task_hint:
        return f"stage_execution_{task_hint.split('_')[0]}"

    # Fallback: return original as-is (for post-Phase 4 tasks like metrics_extraction, plotting_code, etc.)
    return task_hint


PROMPT_BASE = "agent/parallel/"

BASE_SYSTEM_PROMPT = load_prompt("core/system").rstrip("\n")
DOMAIN_NEUTRAL_PROMPT = load_prompt("core/domain_neutral").rstrip("\n")
ENVIRONMENT_INJECTION_TEMPLATE = load_prompt("config/environment/injection").rstrip("\n")
AI_OPTIONAL_PROMPT = load_prompt("core/ai_optional").rstrip("\n")
PHASE1_ITERATIVE_INSTALLER_PROMPT = load_prompt("config/phases/phase1_installer").rstrip("\n")
PHASE1_ITERATIVE_INSTALLER_PROMPT_WITH_MEMORY = load_prompt(
    "config/phases/phase1_installer_with_memory"
).rstrip("\n")
PHASE0_WHOLE_PLANNING_PROMPT = load_prompt("config/phases/phase0_planning").rstrip("\n")
PHASE0_WHOLE_PLANNING_PROMPT_WITH_MEMORY = load_prompt(
    "config/phases/phase0_planning_with_memory"
).rstrip("\n")
ENVIRONMENT_RESOURCES_INJECTION_TEMPLATE = load_prompt("config/environment/resources_injection").rstrip("\n")

IMPLEMENTATION_GUIDELINE_DATASET = tuple(
    load_prompt_lines(PROMPT_BASE + "guidelines/implementation/dataset")
)
DATA_SOURCE_GUIDELINES = {
    "auto": tuple(load_prompt_lines(PROMPT_BASE + "data_source/auto")),
    "huggingface": tuple(load_prompt_lines(PROMPT_BASE + "data_source/huggingface")),
    "local": tuple(load_prompt_lines(PROMPT_BASE + "data_source/local")),
}

RESPONSE_FORMAT_DEFAULT = load_prompt(
    PROMPT_BASE + "response_format/default"
).rstrip("\n")
RESPONSE_FORMAT_SPLIT_PHASE = load_prompt(
    PROMPT_BASE + "response_format/execution_split"
).rstrip("\n")
RESPONSE_FORMAT_SPLIT_PHASE_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "response_format/execution_split_with_memory"
).rstrip("\n")
RESPONSE_FORMAT_METRIC_PARSE = load_prompt(
    PROMPT_BASE + "response_format/metric_parse"
).rstrip("\n")
RESPONSE_FORMAT_DEBUG = load_prompt(
    PROMPT_BASE + "response_format/debug"
).rstrip("\n")
RESPONSE_FORMAT_HPARAM = load_prompt(
    PROMPT_BASE + "response_format/hyperparam"
).rstrip("\n")
RESPONSE_FORMAT_ABLATION = load_prompt(
    PROMPT_BASE + "response_format/ablation"
).rstrip("\n")

DRAFT_INTRO = load_prompt(PROMPT_BASE + "tasks/draft/introduction").rstrip("\n")
DRAFT_INTRO_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "tasks/draft/introduction_with_memory"
).rstrip("\n")
DRAFT_EXP_GUIDELINES = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/draft/experiment_design_sketch_guideline")
)

DEBUG_INTRO = load_prompt(PROMPT_BASE + "tasks/debug/introduction").rstrip("\n")
DEBUG_INTRO_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "tasks/debug/introduction_with_memory"
).rstrip("\n")
DEBUG_BUGFIX_GUIDELINES = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/debug/bugfix_improvement_sketch_guideline")
)

IMPROVE_INTRO = load_prompt(PROMPT_BASE + "tasks/improve/introduction").rstrip("\n")
IMPROVE_INTRO_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "tasks/improve/introduction_with_memory"
).rstrip("\n")

HYPERPARAM_NODE_INTRO_PREFIX = load_prompt(
    PROMPT_BASE + "nodes/hyperparam/introduction"
).rstrip("\n")
HYPERPARAM_NODE_INTRO_PREFIX_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "nodes/hyperparam/introduction_with_memory"
).rstrip("\n")
HYPERPARAM_NODE_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "nodes/hyperparam/instructions")
)

ABLATION_NODE_INTRO_PREFIX = load_prompt(
    PROMPT_BASE + "nodes/ablation/introduction"
).rstrip("\n")
ABLATION_NODE_INTRO_PREFIX_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "nodes/ablation/introduction_with_memory"
).rstrip("\n")
ABLATION_NODE_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "nodes/ablation/instructions")
)

EXECUTION_REVIEW_INTRO = load_prompt(
    PROMPT_BASE + "tasks/execution_review/introduction"
).rstrip("\n")
EXECUTION_REVIEW_INTRO_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "tasks/execution_review/introduction_with_memory"
).rstrip("\n")

PLOTTING_GUIDELINE_BASE = tuple(
    load_prompt_lines(PROMPT_BASE + "guidelines/plotting/base")
)
PLOTTING_GUIDELINE_TAIL = tuple(
    load_prompt_lines(PROMPT_BASE + "guidelines/plotting/tail")
)

DETERMINE_DATASETS_INTRO = load_prompt(
    PROMPT_BASE + "tasks/determine_datasets/introduction"
).rstrip("\n")
DETERMINE_DATASETS_RESPONSE = load_prompt(
    PROMPT_BASE + "tasks/determine_datasets/response_format"
).rstrip("\n")

SELECT_PLOTS_INTRO = load_prompt(
    PROMPT_BASE + "tasks/select_plots/introduction"
).rstrip("\n")

SUMMARY_INTRO = load_prompt(PROMPT_BASE + "tasks/summary/introduction").rstrip("\n")
SUMMARY_INTRO_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "tasks/summary/introduction_with_memory"
).rstrip("\n")

DEFINE_METRICS_INTRO = load_prompt(
    PROMPT_BASE + "tasks/define_metrics/introduction"
).rstrip("\n")
DEFINE_METRICS_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/define_metrics/instructions")
)

PARSE_METRICS_INTRO = load_prompt(
    PROMPT_BASE + "tasks/parse_metrics/introduction"
).rstrip("\n")
PARSE_METRICS_INTRO_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "tasks/parse_metrics/introduction_with_memory"
).rstrip("\n")
PARSE_METRICS_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/parse_metrics/instructions")
)
PARSE_METRICS_EXAMPLE = load_prompt(
    PROMPT_BASE + "tasks/parse_metrics/example"
).rstrip("\n")
MAX_METRIC_PARSE_RETRIES = 3

METRICS_PROMPT_INTRO = load_prompt(
    PROMPT_BASE + "tasks/metrics/introduction"
).rstrip("\n")
VLM_ANALYSIS_PROMPT_TEMPLATE = load_prompt(PROMPT_BASE + "vlm_analysis")
VLM_ANALYSIS_PROMPT_TEMPLATE_WITH_MEMORY = load_prompt(
    PROMPT_BASE + "vlm_analysis_with_memory"
)
SEED_INJECTION_PROMPT = load_prompt(PROMPT_BASE + "seed_injection").rstrip("\n")


def _normalize_language(language: Optional[str]) -> str:
    lang = str(language or "").strip().lower()
    if not lang:
        return "python"
    if lang in {"c++", "cxx"}:
        return "cpp"
    return lang


def _strip_json_wrappers(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if "```" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def _parse_json_object(raw_text: str, *, context: str) -> dict[str, Any]:
    cleaned = _strip_json_wrappers(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            import ast

            parsed = ast.literal_eval(cleaned)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"{context}: failed to parse JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{context}: response must be a JSON object.")
    return parsed


def _normalize_phase0_plan(plan: dict[str, Any]) -> dict[str, Any]:
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


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _summarize_file(path: Path, *, max_lines: int = 40, max_chars: int = 3000) -> str:
    text = _read_text(path)
    return summarize_text(text, max_lines=max_lines, max_chars=max_chars)


def _find_previous_run_dir(current_log_dir: Path) -> Optional[Path]:
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


def _summarize_phase1_steps(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_steps": 0,
        "failed_steps": [],
        "dependencies": {"apt": set(), "pip": set(), "source": set()},
        "recent_commands": [],
    }
    text = _read_text(path)
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


def _extract_error_lines(text: str, *, max_lines: int = 12) -> list[str]:
    if not text:
        return []
    hits = []
    for line in text.splitlines():
        lower = line.lower()
        if "error" in lower or "failed" in lower or "undefined reference" in lower or "not found" in lower:
            hits.append(line.strip())
    return hits[-max_lines:]


def _summarize_phase_logs(phase_log_dir: Path) -> dict[str, Any]:
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
        compile_text = _read_text(compile_log)
        summary["compile_log_summary"] = _summarize_file(compile_log)
        summary["compile_errors"] = _extract_error_lines(compile_text)
    if run_log.exists():
        run_text = _read_text(run_log)
        summary["run_log_summary"] = _summarize_file(run_log)
        summary["run_errors"] = _extract_error_lines(run_text)
    return summary


def _summarize_journal_outputs(prev_log_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    journal_files = list(prev_log_dir.rglob("journal.json"))
    if not journal_files:
        return summary
    journal_files.sort(key=lambda p: p.stat().st_mtime)
    latest = journal_files[-1]
    raw = _read_text(latest)
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


def _has_memory_read_results(results: dict) -> bool:
    """Check if memory operation results contain any read operation data."""
    if not results:
        return False
    read_keys = {"core_get", "archival_search", "recall_search"}
    return bool(read_keys & set(results.keys()))


def _format_memory_results_for_llm(results: dict) -> str:
    """Format memory read results for injection into LLM prompt.

    Args:
        results: Dict containing results from memory read operations.

    Returns:
        Formatted string with <memory_results> block for LLM consumption.
    """
    if not results:
        return ""

    parts = []

    # Core memory GET results
    if "core_get" in results and results["core_get"]:
        parts.append("**Core Memory Values:**")
        for key, value in results["core_get"].items():
            if value is not None:
                parts.append(f"  - {key}: {value}")
            else:
                parts.append(f"  - {key}: (not found)")

    # Archival memory SEARCH results
    if "archival_search" in results and results["archival_search"]:
        parts.append("\n**Archival Search Results:**")
        for i, item in enumerate(results["archival_search"][:10], 1):
            text = item.get("text", "")[:300]
            tags = item.get("tags", [])
            parts.append(f"  [{i}] {text}")
            if tags:
                parts.append(f"      Tags: {', '.join(tags)}")

    # Recall memory SEARCH results
    if "recall_search" in results and results["recall_search"]:
        parts.append("\n**Recall Search Results:**")
        for i, item in enumerate(results["recall_search"][:10], 1):
            kind = item.get("kind", "unknown")
            summary = item.get("summary", "")[:200]
            ts = item.get("ts", "")
            parts.append(f"  [{i}] ({kind}) {summary}")
            if ts:
                parts.append(f"      Time: {ts}")

    if not parts:
        return ""

    return "<memory_results>\n" + "\n".join(parts) + "\n</memory_results>"


def _run_memory_update_phase(
    prompt: dict,
    memory_manager: Any,
    branch_id: str,
    node_id: Optional[str],
    phase_name: str,
    model: str,
    temperature: float,
    max_rounds: int = 2,
    task_description: str = "",
    log_dir: Optional[Path] = None,
) -> None:
    """Run multi-round memory update phase before task execution.

    This function allows the LLM to:
    1. Update memory (core, archival writes)
    2. Read from memory (archival_search, core_get, recall_search)
    3. Re-query with read results if needed

    Args:
        prompt: The base prompt dict (will be modified with Memory Read Results)
        memory_manager: The memory manager instance
        branch_id: Branch ID for memory operations
        node_id: Node ID for memory operations
        phase_name: Name of the phase for logging
        model: LLM model to use
        temperature: Temperature for LLM query
        max_rounds: Maximum number of memory read rounds (default 2)
        task_description: Description of what memory operations to perform
        log_dir: Optional directory for logging prompts and responses
    """
    if not memory_manager or not branch_id:
        return

    memory_prompt = prompt.copy()
    if task_description:
        memory_prompt["Task"] = task_description
    else:
        memory_prompt["Task"] = (
            "Review the context and update your memory with any important findings or insights. "
            "You can also search memory for relevant information. "
            "Respond with ONLY a <memory_update> block containing your memory operations."
        )

    _DB_LOCKED_MAX_RETRIES = 5
    _DB_LOCKED_BASE_DELAY = 1.0

    memory_read_round = 0
    while memory_read_round <= max_rounds:
        try:
            # Log prompt before query
            if log_dir:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_name = f"{phase_name}_memory_round{memory_read_round}"
                prompt_payload = {
                    "meta": {
                        "phase_name": phase_name,
                        "memory_read_round": memory_read_round,
                        "model": model,
                        "branch_id": branch_id,
                        "node_id": node_id,
                    },
                    "prompt": memory_prompt,
                }
                (log_dir / f"{log_name}_prompt.json").write_text(
                    json.dumps(prompt_payload, indent=2, default=str),
                    encoding="utf-8",
                )
                rendered = _render_prompt_for_log(memory_prompt)
                (log_dir / f"{log_name}_prompt.md").write_text(rendered, encoding="utf-8")

            memory_response = query(
                system_message=memory_prompt,
                user_message=None,
                model=model,
                temperature=temperature,
            )

            # Log response after query
            if log_dir and memory_response:
                log_name = f"{phase_name}_memory_round{memory_read_round}"
                response_payload = {
                    "meta": {
                        "phase_name": phase_name,
                        "memory_read_round": memory_read_round,
                        "model": model,
                    },
                    "response": memory_response,
                }
                (log_dir / f"{log_name}_response.json").write_text(
                    json.dumps(response_payload, indent=2, default=str),
                    encoding="utf-8",
                )
                (log_dir / f"{log_name}_response.txt").write_text(
                    memory_response, encoding="utf-8"
                )

            if not memory_response:
                break

            memory_updates = extract_memory_updates(memory_response)
            if not memory_updates:
                break

            memory_results = memory_manager.apply_llm_memory_updates(
                branch_id,
                memory_updates,
                node_id=node_id,
                phase=f"{phase_name}_memory_round{memory_read_round}",
            )

            # Log memory operations details for reproducibility
            if log_dir and memory_results:
                log_name = f"{phase_name}_memory_round{memory_read_round}"
                memory_ops_payload = {
                    "meta": {
                        "phase_name": phase_name,
                        "memory_read_round": memory_read_round,
                        "branch_id": branch_id,
                        "node_id": node_id,
                        "has_read_results": _has_memory_read_results(memory_results),
                    },
                    "input_updates": memory_updates,
                    "operations_log": memory_results.get("operations_log", []),
                    "timing": memory_results.get("timing", {}),
                    "read_results": {
                        "core_get": memory_results.get("core_get", {}),
                        "archival_search": memory_results.get("archival_search", []),
                        "recall_search": memory_results.get("recall_search", []),
                    },
                }
                (log_dir / f"{log_name}_memory_ops.json").write_text(
                    json.dumps(memory_ops_payload, indent=2, default=str),
                    encoding="utf-8",
                )

            # Check if there are read results and we haven't exceeded max rounds
            if _has_memory_read_results(memory_results) and memory_read_round < max_rounds:
                memory_read_round += 1
                # Format results and inject into prompt for re-query
                results_text = _format_memory_results_for_llm(memory_results)
                memory_prompt["Memory Read Results"] = (
                    "Your memory read operations returned the following results:\n\n"
                    f"{results_text}\n\n"
                    "Based on this information, you may:\n"
                    "1. Write additional insights to memory\n"
                    "2. Search for more related information\n"
                    "3. Complete with an empty update if done\n\n"
                    "Respond with ONLY a <memory_update> block."
                )
                continue
            else:
                break

        except Exception as exc:
            if "database is locked" in str(exc):
                # Retry with exponential backoff for transient SQLite lock errors
                retry_ok = False
                for _retry in range(_DB_LOCKED_MAX_RETRIES):
                    delay = _DB_LOCKED_BASE_DELAY * (2 ** _retry) + random.uniform(0, 0.5)
                    logger.info(
                        "%s memory update: database locked, retrying in %.1fs (attempt %d/%d)",
                        phase_name, delay, _retry + 1, _DB_LOCKED_MAX_RETRIES,
                    )
                    time.sleep(delay)
                    try:
                        memory_results = memory_manager.apply_llm_memory_updates(
                            branch_id,
                            memory_updates,
                            node_id=node_id,
                            phase=f"{phase_name}_memory_round{memory_read_round}",
                        )
                        retry_ok = True
                        break
                    except Exception as retry_exc:
                        if "database is locked" not in str(retry_exc):
                            logger.warning(f"{phase_name} memory update failed (round {memory_read_round}): %s", retry_exc)
                            break
                if not retry_ok:
                    logger.warning(f"{phase_name} memory update failed after retries (round {memory_read_round}): %s", exc)
                    break
                # If retry succeeded, continue the normal flow (check read results etc.)
                if _has_memory_read_results(memory_results) and memory_read_round < max_rounds:
                    memory_read_round += 1
                    results_text = _format_memory_results_for_llm(memory_results)
                    memory_prompt["Memory Read Results"] = (
                        "Your memory read operations returned the following results:\n\n"
                        f"{results_text}\n\n"
                        "Based on this information, you may:\n"
                        "1. Write additional insights to memory\n"
                        "2. Search for more related information\n"
                        "3. Complete with an empty update if done\n\n"
                        "Respond with ONLY a <memory_update> block."
                    )
                    continue
                else:
                    break
            else:
                logger.warning(f"{phase_name} memory update failed (round {memory_read_round}): %s", exc)
                break


def _resolve_run_root(cfg: Config) -> Path:
    run_root_env = os.environ.get("AI_SCIENTIST_RUN_ROOT")
    if run_root_env:
        run_root = Path(run_root_env).expanduser().resolve()
    else:
        run_root = Path(cfg.log_dir).parent / "runs"  # 実験ディレクトリ内に配置
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _copy_artifact(src: Path, dest_dir: Path, *, name: Optional[str] = None) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (name or src.name)
    try:
        shutil.copy2(src, dest_path)
    except OSError:
        pass


def _format_prompt_log_name(label: str, *, session_id: Optional[str] = None, counter: Optional[int] = None) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "prompt"
    parts: list[str] = []
    if session_id:
        parts.append(session_id)
    if counter is not None:
        parts.append(f"{counter:03d}")
    parts.append(safe_label)
    return "_".join(parts)


def _render_prompt_for_log(prompt: Any) -> str:
    rendered = compile_prompt_to_md(prompt)
    if isinstance(rendered, (list, dict)):
        return json.dumps(rendered, indent=2, default=str)
    return str(rendered)


from ai_scientist.persona import apply_persona_override


def _write_prompt_log(
    log_dir: Path,
    name: str,
    prompt: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if prompt is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # improved logging: apply persona override to the prompt object
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
    rendered = _render_prompt_for_log(prompt)
    (log_dir / f"{name}.md").write_text(rendered, encoding="utf-8")


def _save_phase_execution_artifacts(
    *,
    exp_results_dir: Path,
    phase_log_dir: Optional[Path],
    run_root: Optional[Path],
    worker_label: str,
    phase_artifacts: Optional[dict] = None,
    phase_artifacts_raw: Optional[str] = None,
) -> None:
    artifacts_dir = exp_results_dir / "phase_artifacts"
    llm_outputs_dir = exp_results_dir / "llm_outputs"
    if phase_log_dir and phase_log_dir.exists():
        for log_name in ("download.log", "coding.log", "compile.log", "run.log"):
            _copy_artifact(phase_log_dir / log_name, artifacts_dir, name=log_name)
        prompt_log_dir = phase_log_dir / "prompt_logs"
        if prompt_log_dir.exists():
            for prompt_file in prompt_log_dir.iterdir():
                if prompt_file.is_file():
                    _copy_artifact(prompt_file, llm_outputs_dir / "prompt_logs")
    if run_root:
        plans_dir = run_root / "workers" / worker_label / "plans"
        _copy_artifact(plans_dir / "phase0_plan.json", llm_outputs_dir)
        _copy_artifact(plans_dir / "phase0_history_full.json", llm_outputs_dir)
        _copy_artifact(plans_dir / "phase0_llm_output.txt", llm_outputs_dir)
        _copy_artifact(run_root / "workers" / worker_label / "phase1_steps.jsonl", llm_outputs_dir)
        _copy_artifact(run_root / "workers" / worker_label / "phase1_llm_outputs.jsonl", llm_outputs_dir)
        # Copy all prompt logs (Phase 0, Phase 1, Phase 2-4, etc.)
        prompt_root = run_root / "workers" / worker_label / "prompt_logs"
        if prompt_root.exists():
            for prompt_file in prompt_root.iterdir():
                if prompt_file.is_file():
                    _copy_artifact(prompt_file, llm_outputs_dir / "prompt_logs")
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


def _run_python_in_container(
    env: ExecutionEnvironment,
    *,
    code: str,
    file_path: Path,
    cwd: Path,
) -> ExecutionResult:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(code, encoding="utf-8")
    exec_start = time.time()
    result = env.run(["bash", "-lc", f"/usr/bin/python3 {file_path.name}"], cwd=cwd)
    exec_time = time.time() - exec_start
    term_out: list[str] = []
    if result.stdout:
        term_out.append(result.stdout)
    if result.stderr:
        term_out.append(result.stderr)
    if result.returncode != 0:
        return ExecutionResult(
            term_out,
            exec_time,
            "RuntimeError",
            {"returncode": result.returncode, "stderr": result.stderr},
            None,
        )
    return ExecutionResult(term_out, exec_time, None, None, None)


def _collect_phase0_history(
    *,
    current_log_dir: Path,
    worker_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {
        "phase_summaries": {},
        "phase1_steps_summary": {},
        "compile_log_summary": "",
        "compile_errors": [],
        "run_log_summary": "",
        "run_errors": [],
        "llm_output_summary": {},
    }
    full: dict[str, Any] = {
        "previous_log_dir": None,
        "previous_run_root": None,
        "phase_summaries": {},
        "phase1_steps": "",
        "compile_log": "",
        "run_log": "",
        "journal_source": "",
    }
    prev_log_dir = _find_previous_run_dir(current_log_dir)
    if not prev_log_dir:
        return summary, full

    full["previous_log_dir"] = str(prev_log_dir)
    prev_run_root = current_log_dir.parent.parent / "runs" / prev_log_dir.name
    full["previous_run_root"] = str(prev_run_root)

    phase_summaries = {}
    for phase in ("phase0", "phase1", "phase2", "phase3", "phase4"):
        matches = []
        matches.extend(prev_log_dir.rglob(f"*{phase}*summary*.txt"))
        matches.extend(prev_log_dir.rglob(f"*{phase}*summary*.json"))
        if not matches:
            continue
        matches.sort(key=lambda p: p.stat().st_mtime)
        latest = matches[-1]
        phase_summaries[phase] = _summarize_file(latest)
        full["phase_summaries"][phase] = _read_text(latest)
    summary["phase_summaries"] = phase_summaries

    steps_path = prev_run_root / "workers" / worker_label / "phase1_steps.jsonl"
    if not steps_path.exists():
        candidates = list(prev_log_dir.rglob("phase1_steps.jsonl"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime)
            steps_path = candidates[-1]
    if steps_path.exists():
        summary["phase1_steps_summary"] = _summarize_phase1_steps(steps_path)
        full["phase1_steps"] = _read_text(steps_path)

    phase_log_summary = _summarize_phase_logs(prev_log_dir / "phase_logs")
    summary.update(phase_log_summary)
    latest_phase_dir = prev_log_dir / "phase_logs"
    if latest_phase_dir.exists():
        node_dirs = [d for d in latest_phase_dir.iterdir() if d.is_dir() and d.name.startswith("node_")]
        if node_dirs:
            node_dirs.sort(key=lambda d: d.stat().st_mtime)
            latest = node_dirs[-1]
            compile_log = latest / "compile.log"
            run_log = latest / "run.log"
            if compile_log.exists():
                full["compile_log"] = _read_text(compile_log)
            if run_log.exists():
                full["run_log"] = _read_text(run_log)

    summary["llm_output_summary"] = _summarize_journal_outputs(prev_log_dir)
    journal_files = list(prev_log_dir.rglob("journal.json"))
    if journal_files:
        journal_files.sort(key=lambda p: p.stat().st_mtime)
        full["journal_source"] = str(journal_files[-1])

    return summary, full

HYPERPARAM_PROMPT_INTRO = load_prompt(
    PROMPT_BASE + "tasks/hyperparam_tuning/introduction"
).rstrip("\n")
HYPERPARAM_PROMPT_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/hyperparam_tuning/instructions")
)
HYPERPARAM_PROMPT_RESPONSE = load_prompt(
    PROMPT_BASE + "tasks/hyperparam_tuning/response_format"
).rstrip("\n")

ABLATION_PROMPT_INTRO = load_prompt(
    PROMPT_BASE + "tasks/ablation_analysis/introduction"
).rstrip("\n")
ABLATION_PROMPT_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/ablation_analysis/instructions")
)
ABLATION_PROMPT_RESPONSE = load_prompt(
    PROMPT_BASE + "tasks/ablation_analysis/response_format"
).rstrip("\n")

SEED_PLOTTING_GUIDELINE_BASE = tuple(
    load_prompt_lines(PROMPT_BASE + "guidelines/seed_plotting/base")
)
SEED_PLOTTING_GUIDELINE_TAIL = tuple(
    load_prompt_lines(PROMPT_BASE + "guidelines/seed_plotting/tail")
)
SEED_PLOTTING_PROMPT_INTRO = load_prompt(
    PROMPT_BASE + "tasks/seed_plotting/introduction"
).rstrip("\n")
SEED_PLOTTING_PROMPT_RESPONSE = load_prompt(
    PROMPT_BASE + "tasks/seed_plotting/response_format"
).rstrip("\n")

def _execute_phase0_planning(
    *,
    phase0_prompt: dict[str, Any],
    cfg: "Config",
    memory_cfg: Any,
    memory_manager: Any,
    branch_id: str,
    plans_dir: Optional[Path],
    prompt_log_root: Optional[Path],
) -> tuple[Optional[dict], Optional[list], Optional[dict]]:
    """Execute Phase 0 planning LLM call and return results.

    This function is called AFTER fork so that Phase 0 is executed
    in the context of each child node, not the parent.

    Args:
        phase0_prompt: The prompt to send to the LLM
        cfg: Configuration object
        memory_cfg: Memory configuration
        memory_manager: Memory manager instance
        branch_id: The child branch ID to record memory on
        plans_dir: Directory to save phase0 plan files
        prompt_log_root: Directory for prompt logs

    Returns:
        Tuple of (phase0_plan, memory_updates, followup_prompt)
    """
    phase0_plan: Optional[dict] = None
    phase0_memory_updates: Optional[list] = None
    phase0_followup_prompt: Optional[dict] = None

    try:
        # Inject memory context into Phase 0 prompt (same as Phase 1-4)
        # This allows LLM to use memory read operations to access previous data
        prompt_with_memory = phase0_prompt.copy()
        if memory_cfg and getattr(memory_cfg, "enabled", False) and memory_manager and branch_id:
            try:
                budget_chars = getattr(memory_cfg, "memory_budget_chars", 4000)
                memory_context = memory_manager.render_for_prompt(
                    branch_id,
                    task_hint="phase0",
                    budget_chars=budget_chars,
                )
                if memory_context:
                    # Add memory context with operation instructions
                    memory_ops_instructions = """

## Memory Operations
The memory operations are already described in the Introduction section.
Use read operations (core_get, archival_search, recall_search) to retrieve
information from previous runs or other nodes before making your plan.
"""
                    prompt_with_memory["Memory"] = memory_context + memory_ops_instructions
                    logger.info("Phase 0: Injected memory context (%d chars) for branch %s", len(memory_context), branch_id)
            except Exception as exc:
                logger.warning("Failed to inject memory context for Phase 0: %s", exc)

        # LLM context (Phase 0 planning): Introduction + Task + History + Environment + Resources + Memory
        phase0_response = query(
            system_message=prompt_with_memory,
            user_message=None,
            model=cfg.agent.code.model,
            temperature=cfg.agent.code.temp,
        )
        phase0_response_raw = phase0_response

        # Save raw LLM output
        if plans_dir:
            try:
                (plans_dir / "phase0_llm_output.txt").write_text(
                    phase0_response_raw,
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Failed to write Phase 0 raw output: %s", exc)

        # Extract and apply memory updates to this node's branch
        if phase0_response_raw:
            if memory_cfg and getattr(memory_cfg, "enabled", False):
                if check_malformed_memory_update(phase0_response_raw):
                    logger.warning("Phase 0 response contains malformed <memory_update>; attempting to strip.")
                phase0_memory_updates = extract_memory_updates(phase0_response_raw)

                # Apply memory updates immediately to this node's branch
                if phase0_memory_updates and memory_manager and branch_id:
                    try:
                        memory_results = memory_manager.apply_llm_memory_updates(
                            branch_id,
                            phase0_memory_updates,
                            node_id=None,
                            phase="phase0",
                        )
                        # Log memory operations for reproducibility
                        if prompt_log_root and memory_results:
                            try:
                                mem_ops_path = prompt_log_root / "phase0_memory_ops.json"
                                mem_ops_payload = {
                                    "timestamp": time.time(),
                                    "phase": "phase0",
                                    "branch_id": branch_id,
                                    "node_id": None,
                                    "input_updates": phase0_memory_updates,
                                    "operations_log": memory_results.get("operations_log", []),
                                    "timing": memory_results.get("timing", {}),
                                    "read_results": {
                                        "core_get": memory_results.get("core_get", {}),
                                        "archival_search": memory_results.get("archival_search", []),
                                        "recall_search": memory_results.get("recall_search", []),
                                    },
                                    "has_read_results": _has_memory_read_results(memory_results),
                                }
                                mem_ops_path.write_text(
                                    json.dumps(mem_ops_payload, indent=2, default=str),
                                    encoding="utf-8",
                                )
                            except Exception as log_exc:
                                logger.warning("Failed to log Phase 0 memory ops: %s", log_exc)
                        # Handle memory read results with follow-up rounds if needed
                        if memory_results and _has_memory_read_results(memory_results):
                            try:
                                results_text = _format_memory_results_for_llm(memory_results)
                                followup_prompt = phase0_prompt.copy()
                                followup_prompt["Memory Read Results"] = (
                                    "Your memory read operations returned the following results:\n\n"
                                    f"{results_text}\n\n"
                                    "Based on this information, you may:\n"
                                    "1. Write additional insights to memory\n"
                                    "2. Search for more related information\n"
                                    "3. Complete with an empty update if done\n\n"
                                    "Respond with ONLY a <memory_update> block."
                                )
                                max_rounds_cfg = getattr(cfg.memory, "max_memory_read_rounds", 2)
                                if max_rounds_cfg > 0:
                                    _run_memory_update_phase(
                                        prompt=followup_prompt,
                                        memory_manager=memory_manager,
                                        branch_id=branch_id,
                                        node_id=None,
                                        phase_name="phase0",
                                        model=cfg.agent.code.model,
                                        temperature=cfg.agent.code.temp,
                                        max_rounds=max(0, max_rounds_cfg - 1),
                                        task_description=(
                                            "You may update memory based on the Phase 0 context and memory read results. "
                                            "Respond with ONLY a <memory_update> block."
                                        ),
                                        log_dir=prompt_log_root,
                                    )
                            except Exception as exc:
                                logger.warning("Failed to run Phase 0 memory read follow-up: %s", exc)
                    except Exception as exc:
                        logger.warning("Failed to apply Phase 0 memory updates to branch %s: %s", branch_id, exc)

                phase0_response = remove_memory_update_tags(phase0_response_raw)
            else:
                phase0_response = phase0_response_raw

        phase0_plan = _normalize_phase0_plan(
            _parse_json_object(phase0_response, context="Phase 0 plan")
        )
    except Exception as exc:
        logger.warning("Phase 0 planning failed: %s", exc)
        phase0_plan = _normalize_phase0_plan({})

    # Save phase0 plan to file
    if plans_dir:
        try:
            (plans_dir / "phase0_plan.json").write_text(
                json.dumps(phase0_plan, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write Phase 0 plan: %s", exc)

    return phase0_plan, phase0_memory_updates, phase0_followup_prompt


def _safe_pickle_test(obj, name="object"):
    """Test if an object can be pickled"""
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        logger.error(f"Cannot pickle {name}: {str(e)}")
        return False


def _extract_npy_schema(file_path: Path, max_depth: int = 2) -> str:
    """
    Extract schema information from a .npy file.

    Returns a string describing the structure of the data, including:
    - Top-level type
    - Keys (if dict)
    - Nested structure up to max_depth

    This helps LLM understand the actual data format for metrics parsing.
    """
    try:
        data = np.load(str(file_path), allow_pickle=True)
        # np.load returns ndarray; if it contains a dict, extract it
        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                # Scalar array containing an object (e.g., dict)
                data = data.item()
            elif data.size == 1:
                data = data.flat[0]

        def describe_structure(obj, depth=0) -> str:
            indent = "  " * depth
            if depth >= max_depth:
                return f"{type(obj).__name__}"

            if isinstance(obj, dict):
                if not obj:
                    return "dict (empty)"
                keys_desc = []
                for k, v in obj.items():
                    v_desc = describe_structure(v, depth + 1)
                    keys_desc.append(f"{indent}  '{k}': {v_desc}")
                return "dict with keys:\n" + "\n".join(keys_desc)
            elif isinstance(obj, (list, tuple)):
                type_name = type(obj).__name__
                if not obj:
                    return f"{type_name} (empty)"
                # Show first element's structure
                elem_desc = describe_structure(obj[0], depth + 1)
                return f"{type_name}[{len(obj)} items], each: {elem_desc}"
            elif isinstance(obj, np.ndarray):
                return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
            else:
                return type(obj).__name__

        return describe_structure(data)
    except Exception as e:
        return f"(failed to read schema: {e})"


def _parse_keyword_prefix_response(
    response: str, keyword_prefix1: str, keyword_prefix2: str
) -> Tuple[Optional[str], Optional[str]]:
    """Parse the response into name and description based on keyword prefix.

    Matching is case-insensitive: e.g. both "REASONING:" and "Reasoning:" are accepted.
    """
    try:
        # Split response into lines and clean up
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        prefix1_lower = keyword_prefix1.lower()
        prefix2_lower = keyword_prefix2.lower()

        # Find the idea and description
        name = None
        description = None

        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith(prefix1_lower):
                name = line[len(keyword_prefix1):].strip()
            elif line_lower.startswith(prefix2_lower):
                description = line[len(keyword_prefix2):].strip()
                # Combine any following lines that don't start with a marker
                desc_lines = []
                for next_line in lines[lines.index(line) + 1 :]:
                    next_lower = next_line.lower()
                    if not next_lower.startswith((prefix1_lower, prefix2_lower)):
                        desc_lines.append(next_line)
                    else:
                        break
                if desc_lines:
                    description = " ".join([description] + desc_lines)

        if name is None or description is None:
            raise ValueError(
                f"Missing required keywords in response: {keyword_prefix1} and/or {keyword_prefix2}"
            )

        return name, description

    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.debug(f"Raw response: {response}")
        return None, None


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.",
            },
        },
        "required": [
            "is_bug",
            "summary",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

vlm_feedback_spec = FunctionSpec(
    name="analyze_experiment_plots",
    json_schema={
        "type": "object",
        "properties": {
            "plot_analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "Detailed analysis of the plot's results and implications",
                        },
                    },
                    "required": ["analysis"],
                },
            },
            "valid_plots_received": {
                "type": "boolean",
                "description": "True if valid plots were received, False otherwise. For example, if the plots are empty or not meaningful, this should be False.",
            },
            "vlm_feedback_summary": {
                "type": "string",
                "description": "Summarize the feedback from the VLM. If the task involves generative modeling, make sure to focus on the generated samples.",
            },
        },
        "required": ["plot_analyses", "valid_plots_received", "vlm_feedback_summary"],
    },
    description="Analyze experimental plots and provide detailed feedback on the results.",
)

metric_parse_spec = FunctionSpec(
    name="parse_metrics",
    json_schema={
        "type": "object",
        "strict": True,
        "properties": {
            "valid_metrics_received": {
                "type": "boolean",
                "description": "True if the metrics were successfully received, False otherwise. For example if the execution output does not contain any metrics, set this to False.",
            },
            "metric_names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "Specify the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.",
                        },
                        "lower_is_better": {
                            "type": "boolean",
                            "description": "Whether lower values are better for this metric",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the metric",
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dataset_name": {
                                        "type": "string",
                                        "description": "The name of the dataset. Never include 'train', 'val', or 'test' in the dataset name.",
                                    },
                                    "value": {
                                        "type": "number",
                                        "description": "The value of the metric for this dataset",
                                    },
                                },
                                "required": [
                                    "dataset_name",
                                    "value",
                                ],
                            },
                        },
                    },
                    "required": [
                        "data",
                        "metric_name",
                        "lower_is_better",
                        "description",
                    ],
                },
                "additionalProperties": False,
            },
        },
        "required": ["valid_metrics_received", "metric_names"],
        "additionalProperties": False,
    },
    description="Parse metrics from execution output",
)


plot_selection_spec = FunctionSpec(
    name="select_plots",
    json_schema={
        "type": "object",
        "properties": {
            "selected_plots": {
                "type": "array",
                "description": "List of selected plot file paths",
                "items": {"type": "string", "description": "Full path to a plot file"},
                "maxItems": 10,
            }
        },
        "required": ["selected_plots"],
    },
    description="Select the 10 most relevant plots for analysis",
)


class AblationConfig:
    """Track state of ablation experiments"""

    def __init__(self, name: str, description: str, code: str, base_node: Node):
        self.name = name
        self.description = description
        self.code = code
        self.base_node = base_node
        self.attempts = 0
        self.max_attempts = 3  # Maximum number of retry attempts
        self.last_error = None
        self.completed = False
        self.current_node = None


class AblationIdea:
    """Ablation idea"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class HyperparamTuningIdea:
    """Hyperparameter tuning idea"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class MinimalAgent:
    """A minimal agent class that only contains what's needed for processing nodes"""

    def __init__(
        self,
        task_desc,
        cfg,
        memory_summary=None,
        evaluation_metrics=None,
        stage=None,
        stage_name=None,
        environment_context: Optional[dict] = None,
        phase0_plan: Optional[dict] = None,
        phase0_history: Optional[dict] = None,
        prompt_log_dir: Optional[Path] = None,
        prompt_session_id: Optional[str] = None,
        memory_manager: Optional[Any] = None,
        branch_id: Optional[str] = None,
    ):
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name
        self.memory_manager = memory_manager
        self.branch_id = branch_id
        self.environment_context = environment_context or {}
        self.phase0_plan = phase0_plan
        self.phase0_history = phase0_history or {}
        self.resources_config: Optional[ResourceConfig] = None
        self.resources_error: Optional[str] = None
        self._resources_prompt_cache: dict[str, dict[str, Any]] = {}
        self.prompt_log_dir = prompt_log_dir
        self.prompt_session_id = prompt_session_id
        self._prompt_log_counter = 0
        # Memory operation tracking for detailed prompt logging
        self._last_memory_injection_log: Optional[dict] = None
        self._last_memory_operation_results: Optional[dict] = None
        resources_path = getattr(self.cfg.exec, "resources", None)
        if resources_path:
            try:
                self.resources_config = load_resources(resolve_resources_path(resources_path))
            except Exception as exc:
                self.resources_error = str(exc)

    @property
    def experiment_name(self) -> str:
        """Get the experiment name from the workspace directory or config."""
        workspace_dir = getattr(self.cfg, "workspace_dir", None)
        if workspace_dir:
            return Path(workspace_dir).name
        return ""

    @property
    def experiment_output_filename(self) -> str:
        """Get the output filename for this experiment."""
        return _get_experiment_output_filename(self.experiment_name)

    @property
    def experiment_output_path(self) -> str:
        """Get the relative output path for this experiment (e.g., 'working/experiment_name_data.npy')."""
        return f"working/{self.experiment_output_filename}"

    def _memory_context(
        self,
        task_hint: str,
        branch_id: Optional[str] = None,
        budget_chars: Optional[int] = None,
        allow_summary_fallback: bool = False,
    ) -> str:
        if not self.memory_manager:
            return (self.memory_summary or "") if allow_summary_fallback else ""
        use_branch = branch_id or self.branch_id
        if not use_branch:
            return self.memory_summary or ""
        budget = budget_chars or getattr(getattr(self.cfg, "memory", None), "memory_budget_chars", 4000)
        return self.memory_manager.render_for_prompt(use_branch, task_hint, budget_chars=budget)

    def _memory_context_with_log(
        self,
        task_hint: str,
        branch_id: Optional[str] = None,
        budget_chars: Optional[int] = None,
        allow_summary_fallback: bool = False,
    ) -> tuple[str, Optional[dict]]:
        """Get memory context with detailed log information.

        Returns:
            Tuple of (memory_text, log_details) where log_details contains
            information about what memory was injected into the prompt.
        """
        if not self.memory_manager:
            fallback = (self.memory_summary or "") if allow_summary_fallback else ""
            return fallback, None
        use_branch = branch_id or self.branch_id
        if not use_branch:
            return self.memory_summary or "", None
        budget = budget_chars or getattr(getattr(self.cfg, "memory", None), "memory_budget_chars", 4000)
        # Check if render_for_prompt_with_log method exists
        if hasattr(self.memory_manager, "render_for_prompt_with_log"):
            text, log_details = self.memory_manager.render_for_prompt_with_log(
                use_branch, task_hint, budget_chars=budget
            )
            return text, log_details
        else:
            # Fallback to regular render if method not available
            text = self.memory_manager.render_for_prompt(use_branch, task_hint, budget_chars=budget)
            return text, None

    def _inject_memory(
        self,
        prompt: dict[str, Any],
        task_hint: str,
        branch_id: Optional[str] = None,
        budget_chars: Optional[int] = None,
        allow_summary_fallback: bool = False,
        allow_empty: bool = False,
        node_id: Optional[str] = None,
    ) -> None:
        # Set current phase for memory event logging (use human-readable name)
        if self.memory_manager and hasattr(self.memory_manager, "set_current_phase"):
            phase_display_name = get_phase_display_name(task_hint)
            self.memory_manager.set_current_phase(phase_display_name)

        # Get memory context with log information for detailed prompt logging
        memory_context, memory_injection_log = self._memory_context_with_log(
            task_hint,
            branch_id=branch_id,
            budget_chars=budget_chars,
            allow_summary_fallback=allow_summary_fallback,
        )
        # Store the memory injection log for later use in prompt logging
        self._last_memory_injection_log = memory_injection_log
        if memory_context or allow_empty:
            # Add memory operation instructions for LLM
            # For split-phase mode with memory enabled, the Response format already
            # contains detailed instructions about the required <memory_update> block.
            # Here we provide a brief reminder and the available operations.
            if self.phase_mode == "split" and self._is_memory_enabled:
                memory_ops_instructions = """

## Memory Operations (REQUIRED)
You MUST include a <memory_update> block at the START of your response before the JSON.
See "Response format" for the exact format. Missing this block will cause a retry.

### Available Operations (use in <memory_update> JSON):
- **core**: {"key": "value"} - Always-visible key-value store (optimal params, configs)
- **archival**: [{"text": "...", "tags": ["TAG"]}] - Searchable long-term store (explanations, lessons)
- **archival_search**: {"query": "...", "k": 5} - Search archival memory
- **recall**: {"kind": "...", "content": "..."} - Recent events
- **consolidate**: true - Trigger memory consolidation

### What to record:
- **core**: Optimal parameters, best configurations, important constraints
- **archival**: Detailed explanations, lessons learned, patterns to avoid/prefer
"""
            else:
                memory_ops_instructions = """

## Memory Operations (Optional)
You can manage your memory by including a <memory_update> block at the end of your response.

### Available Operations:

**1. Core Memory (always-visible key-value store):**
```json
{
  "core": {"key": "value"},
  "core_get": ["key1", "key2"],
  "core_delete": ["obsolete_key"]
}
```

**2. Archival Memory (searchable long-term store):**
```json
{
  "archival": [{"text": "Insight to remember", "tags": ["TAG1"]}],
  "archival_update": [{"id": "record_id", "text": "Updated text"}],
  "archival_search": {"query": "search terms", "k": 5, "tags": ["TAG1"]}
}
```

**3. Recall Memory (recent events):**
```json
{
  "recall": {"kind": "discovery", "content": "What happened"},
  "recall_search": {"query": "search terms", "k": 10},
  "recall_evict": {"oldest": 10, "kind": "debug", "ids": ["id1"]},
  "recall_summarize": true
}
```

**4. Memory Management:**
```json
{
  "consolidate": true
}
```

### Example:
<memory_update>
{
  "core": {"optimal_threads": "8", "best_compiler": "-O3"},
  "archival": [{"text": "Thread count 8 gives 2x speedup on this workload", "tags": ["PERFORMANCE"]}],
  "archival_search": {"query": "compilation errors", "k": 3}
}
</memory_update>

### What to record:
- **core**: Optimal parameters, best configurations, important constraints
- **archival**: Detailed explanations, lessons learned, patterns to avoid/prefer

### Do NOT record:
- Temporary debug information
- Information already logged by the system
- Obvious or trivial observations
"""
            prompt["Memory"] = memory_context + memory_ops_instructions

        # Record memory injection event for tracking what memory was used at each phase
        if self.memory_manager and memory_context:
            use_branch = branch_id or self.branch_id
            use_node_id = node_id or getattr(self, "_current_node_id", None)
            if use_branch:
                try:
                    # Create a summary of injected memory (first 500 chars)
                    memory_summary = memory_context[:500] + ("..." if len(memory_context) > 500 else "")
                    self.memory_manager.mem_recall_append({
                        "ts": time.time(),
                        "run_id": self.memory_manager.run_id,
                        "node_id": use_node_id,
                        "branch_id": use_branch,  # Explicit branch_id to avoid resolution issues
                        "phase": self.stage_name or "unknown",
                        "kind": "memory_injected",
                        "summary": f"Memory injected for {task_hint}\nSize: {len(memory_context)} chars\n---\n{memory_summary}",
                        "refs": [],
                        "task_hint": task_hint,
                        "memory_size": len(memory_context),
                    })
                except Exception as exc:
                    logger.warning("Failed to write memory_injected event: %s", exc)

    @property
    def code_language(self) -> str:
        return _normalize_language(getattr(self.cfg.exec, "language", "python"))

    @property
    def phase_mode(self) -> str:
        return getattr(self.cfg.exec, "phase_mode", "single")

    def _language_requirements(self) -> list[str]:
        lang = self.code_language
        if lang == "python":
            return []
        if lang == "cpp":
            return [
                "Implement the solution in C++ (use ```cpp``` code blocks).",
                "Do not use Python for the main implementation.",
                'Use .cpp sources and set build_plan.language to "cpp" in split-phase.',
            ]
        if lang == "fortran":
            return [
                "Implement the solution in Fortran (use ```fortran``` code blocks).",
                "Do not use Python for the main implementation.",
                'Use .f90 sources and set build_plan.language to "fortran" in split-phase.',
            ]
        return [
            f"Implement the solution in {lang}.",
            "Do not use Python for the main implementation.",
        ]

    def _format_response_format(self, response_format: str) -> str:
        lang = self.code_language
        if lang == "python":
            return response_format
        updated = response_format.replace("```python", f"```{lang}")
        return updated.replace("python", lang)

    @property
    def _prompt_environment(self):
        if self.phase_mode == "split":
            compilers = self.environment_context.get("available_compilers", [])
            libs = self.environment_context.get("available_libs", [])
            performance_tools = self.environment_context.get("system_performance_tools", [])
            installed_packages = self.environment_context.get("installed_system_packages", [])
            # Build system perf tool names list from probe results
            performance_tool_names = ", ".join([t.get("name", "") for t in performance_tools if isinstance(t, dict)]) or "none"
            payload = {
                "available_compilers_json": json.dumps(compilers, indent=2),
                "available_compiler_names": ", ".join([c.get("name", "") for c in compilers if isinstance(c, dict)]) or "none",
                "available_libs": json.dumps(libs, indent=2),
                "system_performance_tools_json": json.dumps(performance_tools, indent=2),
                "system_performance_tool_names": performance_tool_names if performance_tool_names != "none" else "none (no performance tools detected)",
                "installed_system_packages_json": json.dumps(installed_packages, indent=2),
                "installed_system_package_names": ", ".join([p.get("name", "") for p in installed_packages if isinstance(p, dict)]) or "none",
                "container_runtime": self.environment_context.get("container_runtime") or "host",
                "container_digest": self.environment_context.get("container_digest", "NA"),
                "singularity_image": self.environment_context.get("singularity_image") or "none",
                "workspace_mount": self.environment_context.get("workspace_mount", "/workspace"),
                "timeout_seconds": self.environment_context.get("timeout_seconds", self.cfg.exec.timeout),
                "cpu_info": self.environment_context.get("cpu_info", "unknown"),
                "cpu_governor": self.environment_context.get("cpu_governor", "NA"),
                "numa_config": self.environment_context.get("numa_config", "NA"),
                "memory_info": self.environment_context.get("memory_info", "unknown"),
                "assigned_gpu_id": self.environment_context.get("assigned_gpu_id", "unknown"),
                "gpu_info": self.environment_context.get("gpu_info", "unknown"),
                "all_env_vars": self.environment_context.get("all_env_vars", "NA"),
            }
            rendered_message = format_prompt("config/environment/injection", **payload)
            return {"Environment injection": rendered_message}

        if self.cfg.exec.env_packages_template:
            package_template = self.cfg.exec.env_packages_template
        else:
            package_template = "agent/parallel/environment/packages"

        packages = load_prompt_lines(package_template)

        if self.cfg.exec.env_packages_template:
            if "cpp" in package_template:
                env_message_template = "agent/parallel/environment/message_cpp"
            elif "fortran" in package_template:
                env_message_template = "agent/parallel/environment/message_fortran"
            else:
                env_message_template = "agent/parallel/environment/message"
        else:
            env_message_template = "agent/parallel/environment/message"

        message = load_prompt(env_message_template).rstrip("\n")

        pkgs = list(packages)
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        if "{pkg_str}" in message:
            rendered_message = message.replace("{pkg_str}", pkg_str)
        else:
            rendered_message = f"{message}\nAvailable packages: {pkg_str}"

        return {"Installed Packages": rendered_message}

    def _prompt_resources(self, phase: str) -> Optional[Dict[str, Any]]:
        if self.resources_error:
            return {"error": self.resources_error}
        if not self.resources_config or not self.resources_config.has_resources():
            return None
        cached = self._resources_prompt_cache.get(phase)
        if cached:
            return cached
        ctx = build_resources_context(self.resources_config, phase=phase)
        self._resources_prompt_cache[phase] = ctx
        return ctx

    def _inject_resources(self, prompt: dict[str, Any], phase: str) -> None:
        ctx = self._prompt_resources(phase)
        if ctx:
            prompt["Resources"] = ctx

    def _inject_seed_with_llm(self, node: Node, seed: int) -> None:
        prompt: dict[str, Any] = {
            "Introduction": SEED_INJECTION_PROMPT,
            "Language": self.code_language,
            "Default seed": seed,
        }
        raw_phase_artifacts = getattr(node, "phase_artifacts", None)
        phase_data = None
        if isinstance(raw_phase_artifacts, dict):
            phase_data = raw_phase_artifacts.get("phase_artifacts") or raw_phase_artifacts
        elif isinstance(raw_phase_artifacts, list) and raw_phase_artifacts:
            first = raw_phase_artifacts[0]
            if isinstance(first, dict):
                phase_data = first.get("phase_artifacts") or first
        if isinstance(phase_data, list) and phase_data:
            phase_data = phase_data[0]

        files = None
        if self.phase_mode == "split" and isinstance(phase_data, dict):
            coding = phase_data.get("coding", {})
            workspace = coding.get("workspace", {})
            files = workspace.get("files", [])
            if isinstance(files, list) and files:
                prompt["Files"] = [
                    {"path": f.get("path", ""), "content": f.get("content", "")}
                    for f in files
                    if isinstance(f, dict)
                ]
            else:
                files = None

        if files is None:
            prompt["Code"] = wrap_combined_code(node.code, fallback_lang=self.code_language)

        self._log_prompt(
            prompt,
            label="seed_injection",
            meta={"phase": "seed_eval", "seed": seed},
        )
        response_text = query(
            system_message=prompt,
            user_message=None,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.temp,
        )
        try:
            parsed = _parse_json_object(response_text, context="Seed injection")
        except Exception as exc:
            logger.warning("Seed injection failed to parse LLM response: %s", exc)
            return

        updates = parsed.get("files")
        if not isinstance(updates, list) or not updates:
            return

        if files is not None:
            file_map = {
                f.get("path", ""): f for f in files if isinstance(f, dict)
            }
            updated = False
            for update in updates:
                if not isinstance(update, dict):
                    continue
                path = update.get("path")
                content = update.get("content")
                if isinstance(path, str) and path in file_map and isinstance(content, str):
                    file_map[path]["content"] = content
                    updated = True
            if updated:
                node.code = combine_sources_for_display(files)
            return

        target_name = str(getattr(self.cfg.exec, "agent_file_name", "") or "")
        selected_content = None
        for update in updates:
            if not isinstance(update, dict):
                continue
            if update.get("path") == target_name and isinstance(update.get("content"), str):
                selected_content = update["content"]
                break
        if selected_content is None:
            first = updates[0]
            if isinstance(first, dict) and isinstance(first.get("content"), str):
                selected_content = first["content"]
        if selected_content:
            node.code = selected_content

    def _log_prompt(self, prompt: Any, *, label: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.prompt_log_dir:
            return
        self._prompt_log_counter += 1
        name = _format_prompt_log_name(
            label,
            session_id=self.prompt_session_id,
            counter=self._prompt_log_counter,
        )
        import time
        payload: dict[str, Any] = {
            "stage": self.stage_name,
            "timestamp": time.time(),
            "label": label,
            "counter": self._prompt_log_counter,
        }
        if meta:
            payload.update(meta)
        # Include memory injection details if available
        if self._last_memory_injection_log:
            payload["memory_injected"] = self._last_memory_injection_log
        # Include memory operation results if available (from previous LLM response)
        if self._last_memory_operation_results:
            payload["memory_operations"] = self._last_memory_operation_results
            # Clear after use to avoid stale data
            self._last_memory_operation_results = None
        _write_prompt_log(self.prompt_log_dir, name, prompt, meta=payload)

        # Also write to memory_operations.jsonl if memory operations were logged
        if self.prompt_log_dir and payload.get("memory_operations"):
            self._append_memory_ops_log(payload)

        # Log memory injection details to memory_injections.jsonl
        if self.prompt_log_dir and payload.get("memory_injected"):
            self._append_memory_injection_log(payload)

    def _append_memory_ops_log(self, payload: dict[str, Any]) -> None:
        """Append memory operations to a chronological log file."""
        if not self.prompt_log_dir:
            return
        import json
        import time
        mem_log_path = self.prompt_log_dir / "memory_operations.jsonl"
        mem_ops = payload.get("memory_operations", {})
        operations_log = mem_ops.get("operations_log", [])

        entry = {
            "timestamp": time.time(),
            "stage": payload.get("stage"),
            "label": payload.get("label"),
            "counter": payload.get("counter"),
            "timing": mem_ops.get("timing"),
            "operations_count": len(operations_log),
            "operations": operations_log,
            "input_updates_keys": list(mem_ops.get("input_updates", {}).keys()) if mem_ops.get("input_updates") else [],
        }

        try:
            mem_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mem_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Non-critical logging

    def _append_memory_injection_log(self, payload: dict[str, Any]) -> None:
        """Append memory injection details to a chronological log file."""
        if not self.prompt_log_dir:
            return
        import json
        import time
        inj_log_path = self.prompt_log_dir / "memory_injections.jsonl"
        mem_inj = payload.get("memory_injected", {})

        entry = {
            "timestamp": time.time(),
            "stage": payload.get("stage"),
            "label": payload.get("label"),
            "counter": payload.get("counter"),
            "task_hint": mem_inj.get("task_hint"),
            "budget_chars": mem_inj.get("budget_chars"),
            "timing": mem_inj.get("timing"),
            "rendered_chars": mem_inj.get("rendered_chars"),
            "core_count": mem_inj.get("core_count"),
            "recall_count": mem_inj.get("recall_count"),
            "archival_count": mem_inj.get("archival_count"),
            "core_items": mem_inj.get("core_items"),
            "recall_items": mem_inj.get("recall_items"),
            "archival_items": mem_inj.get("archival_items"),
        }

        try:
            inj_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(inj_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Non-critical logging

    def _log_memory_operations_detail(
        self,
        memory_updates: dict[str, Any],
        memory_results: dict[str, Any],
        label: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log detailed memory operations for reproducibility.

        This method writes a comprehensive log of memory operations including:
        - Input updates from LLM (the parsed <memory_update> block)
        - All operations executed with their results
        - Read operation results (archival_search, core_get, recall_search)
        - Timing information

        Args:
            memory_updates: The parsed memory update dict from LLM response.
            memory_results: The results from apply_llm_memory_updates.
            label: Label for the log entry.
            meta: Optional metadata.
        """
        if not self.prompt_log_dir:
            return
        import json
        import time

        # Write to memory_operations.jsonl for chronological tracking
        ops_log_path = self.prompt_log_dir / "memory_operations.jsonl"
        operations_log = memory_results.get("operations_log", [])

        entry = {
            "timestamp": time.time(),
            "stage": self.stage_name,
            "label": label,
            "meta": meta or {},
            "input_updates": memory_updates,
            "operations_count": len(operations_log),
            "operations": operations_log,
            "timing": memory_results.get("timing", {}),
            "read_results": {
                "core_get": memory_results.get("core_get", {}),
                "archival_search": memory_results.get("archival_search", []),
                "recall_search": memory_results.get("recall_search", []),
            },
            "has_read_results": bool(
                memory_results.get("core_get")
                or memory_results.get("archival_search")
                or memory_results.get("recall_search")
            ),
        }

        try:
            ops_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ops_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Non-critical logging

        # Also write a detailed JSON file for this specific operation
        self._prompt_log_counter += 1
        detail_name = _format_prompt_log_name(
            f"{label}_memory_ops",
            session_id=self.prompt_session_id,
            counter=self._prompt_log_counter,
        )
        detail_path = self.prompt_log_dir / f"{detail_name}.json"
        try:
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, default=str)
        except Exception:
            pass  # Non-critical logging

    def _log_response(
        self,
        response: str,
        *,
        label: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log raw LLM response to a file for debugging and analysis.

        Args:
            response: The raw LLM response text.
            label: Label for the response log (e.g., 'phase_artifacts_attempt1').
            meta: Optional metadata to include in the log.
        """
        if not self.prompt_log_dir or not response:
            return
        import time

        # Use the same counter as the corresponding prompt
        name = _format_prompt_log_name(
            f"{label}_response",
            session_id=self.prompt_session_id,
            counter=self._prompt_log_counter,
        )

        payload: dict[str, Any] = {
            "stage": self.stage_name,
            "timestamp": time.time(),
            "label": label,
            "counter": self._prompt_log_counter,
        }
        if meta:
            payload.update(meta)

        log_dir = self.prompt_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Write response with metadata as JSON
        response_payload = {
            "meta": payload,
            "response": response,
        }
        (log_dir / f"{name}.json").write_text(
            json.dumps(response_payload, indent=2, default=str),
            encoding="utf-8",
        )

        # Also write raw response as plain text for easy reading
        (log_dir / f"{name}.txt").write_text(response, encoding="utf-8")

    def _build_impl_guideline(self) -> list[str]:
        """Build implementation guidelines dynamically based on language."""
        impl_guideline: list[str] = []

        # Language-specific requirements
        language_notes = self._language_requirements()
        if language_notes:
            impl_guideline.extend(language_notes)
        else:
            impl_guideline.append(f"Implement the solution in {self.code_language}.")

        # Common structure requirements
        impl_guideline.append("Keep the program self-contained and runnable as-is.")
        impl_guideline.append("Write outputs under ./working (create the directory if needed).")
        impl_guideline.append(
            "Only create or modify files under ./src or ./working; do not create new top-level directories."
        )

        # Python-specific: GPU and model requirements
        if self.code_language == "python":
            impl_guideline.extend([
                "GPU REQUIREMENTS:",
                "  - At the start of your code, add: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
                "  - Move models to device using .to(device)",
                "  - Move input tensors to device using .to(device)",
                "  - Create optimizers AFTER moving model to device",
                "MODEL INPUT GUIDELINES:",
                "  - Always ensure input to the model is properly normalized",
            ])

        # Reproducibility requirements
        impl_guideline.append(
            "REPRODUCIBILITY: If the code uses randomness, define a seed initialization function and call it before any stochastic operations."
        )

        # Data saving requirements
        output_filename = self.experiment_output_filename
        if self.code_language == "python":
            impl_guideline.append(
                f"Save experiment data to working/{output_filename} using np.save()."
            )
        else:
            impl_guideline.append(
                f"Save experiment data to working/{output_filename} using cnpy or a compatible .npy writer."
            )

        # Metrics tracking
        if self.evaluation_metrics:
            impl_guideline.append(
                f"Track and report these additional metrics: {self.evaluation_metrics}"
            )

        # Dataset requirements
        if hasattr(self.cfg.experiment, "num_syn_datasets"):
            num_syn_datasets = self.cfg.experiment.num_syn_datasets
            if num_syn_datasets > 1:
                formatted_dataset_guideline = [
                    line.format(num_syn_datasets=num_syn_datasets)
                    for line in IMPLEMENTATION_GUIDELINE_DATASET
                ]
                impl_guideline.extend(formatted_dataset_guideline)

        # Dataset source guidelines
        dataset_source = getattr(self.cfg.experiment, "dataset_source", None)
        dataset_source_key = (
            str(dataset_source).lower() if dataset_source not in (None, "") else "auto"
        )
        dataset_guidance = DATA_SOURCE_GUIDELINES.get(
            dataset_source_key, DATA_SOURCE_GUIDELINES["auto"]
        )
        impl_guideline.extend(dataset_guidance)

        # Python-specific: code structure and evaluation
        if self.code_language == "python":
            impl_guideline.extend([
                "CODE STRUCTURE:",
                "  - Do NOT put execution code inside 'if __name__ == \"__main__\":' block",
                "  - All code should be at the global scope or in functions called from global scope",
                "  - Start with: import os; working_dir = os.path.join(os.getcwd(), 'working'); os.makedirs(working_dir, exist_ok=True)",
                "EVALUATION:",
                "  - Track and print validation loss at each epoch",
                f"  - Save metrics at the end using np.save(os.path.join(working_dir, '{output_filename}'), experiment_data)",
            ])

        # Timeout warning
        impl_guideline.append(
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}."
        )

        return impl_guideline

    @property
    def _prompt_impl_guideline(self):
        if self.phase_mode == "split":
            domain_guideline = DOMAIN_NEUTRAL_PROMPT.splitlines()
            language_notes = self._language_requirements()
            if language_notes:
                domain_guideline = language_notes + domain_guideline
            return {"Implementation guideline": domain_guideline}

        return {"Implementation guideline": self._build_impl_guideline()}

    @property
    def _is_memory_enabled(self) -> bool:
        """Check if memory management is enabled."""
        return bool(self.memory_manager and getattr(self.cfg, 'memory', None) and getattr(self.cfg.memory, 'enabled', False))

    @property
    def _prompt_resp_fmt(self):
        if self.phase_mode == "split":
            # Use memory-aware format when memory is enabled
            if self._is_memory_enabled:
                return {"Response format": RESPONSE_FORMAT_SPLIT_PHASE_WITH_MEMORY}
            return {"Response format": RESPONSE_FORMAT_SPLIT_PHASE}
        return {"Response format": self._format_response_format(RESPONSE_FORMAT_DEFAULT)}

    @property
    def _prompt_phase_guidance(self):
        if self.phase_mode != "split":
            return {}
        output_filename = self.experiment_output_filename
        return {
            "Phase workflow": [
                "Use the 5-phase plan: Phase 0 whole planning, Phase 1 download/install, Phase 2 coding, Phase 3 compile, Phase 4 run.",
                "Phase 1 may use sudo/apt-get inside Singularity with writable-tmpfs/overlay; if unavailable install under /workspace.",
                "All paths are relative to /workspace; no absolute paths or parent traversal.",
                "In download/install, probe with `command -v ...`/`which ...` before installing (e.g., git, cmake); avoid redundant installs.",
                "compile commands must honor build_plan.compiler_selected chosen from available_compilers (do not invent compilers or switch).",
                f"Run phase must generate /workspace/working/{output_filename} (Python uses numpy save; non-Python must use cnpy).",
                "All generated code/scripts must live under /workspace/src or /workspace/working; do not create new top-level directories.",
                "Do not rely on language adapters, interpreter adapters, or external routers; commands run directly in the worker.",
            ]
        }

    def _phase0_plan_snippet(self, *, include_phase1: bool, include_phase2_4: bool) -> Optional[Dict[str, Any]]:
        if not self.phase0_plan:
            return None
        plan_blob = self.phase0_plan.get("plan") if isinstance(self.phase0_plan, dict) else None
        if not isinstance(plan_blob, dict):
            return None
        snippet: dict[str, Any] = {
            "goal_summary": plan_blob.get("goal_summary", ""),
            "implementation_strategy": plan_blob.get("implementation_strategy", []),
            "dependencies": plan_blob.get("dependencies", {}),
        }
        phase_guidance = plan_blob.get("phase_guidance", {})
        guidance: dict[str, Any] = {}
        if isinstance(phase_guidance, dict):
            if include_phase1:
                guidance["phase1"] = phase_guidance.get("phase1", {})
            if include_phase2_4:
                guidance["phase2"] = phase_guidance.get("phase2", {})
                guidance["phase3"] = phase_guidance.get("phase3", {})
                guidance["phase4"] = phase_guidance.get("phase4", {})
        if guidance:
            snippet["phase_guidance"] = guidance
        risks = plan_blob.get("risks_and_mitigations")
        if risks:
            snippet["risks_and_mitigations"] = risks
        return snippet

    def _apply_split_prompt_layers(
        self, prompt: dict[str, Any], *, task_hint: str = "phase2_coding"
    ) -> dict[str, Any]:
        """Inject common split-phase system guidance, environment context, and memory for Phase 2/3/4."""
        prompt = {"System": BASE_SYSTEM_PROMPT, **prompt}
        prompt["Domain"] = DOMAIN_NEUTRAL_PROMPT
        env_block = self._prompt_environment
        if env_block:
            prompt["Environment"] = env_block.get("Environment injection", env_block)
        self._inject_resources(prompt, phase="phase2")
        # Inject memory for Phase 2/3/4 if not already present
        if "Memory" not in prompt:
            self._inject_memory(
                prompt,
                task_hint,
                allow_summary_fallback=True,
                allow_empty=False,
            )
        phase0_snippet = self._phase0_plan_snippet(include_phase1=False, include_phase2_4=True)
        if phase0_snippet:
            prompt["Phase 0 plan"] = phase0_snippet
        prompt["Instructions"] |= self._prompt_phase_guidance
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_impl_guideline
        return prompt

    def _prompt_metricparse_resp_fmt(self):
        return {"Response format": RESPONSE_FORMAT_METRIC_PARSE}

    @property
    def _prompt_debug_resp_fmt(self):
        return {"Response format": self._format_response_format(RESPONSE_FORMAT_DEBUG)}

    @property
    def _prompt_hyperparam_tuning_resp_fmt(self):
        return {"Response format": self._format_response_format(RESPONSE_FORMAT_HPARAM)}

    @property
    def _prompt_ablation_resp_fmt(self):
        return {"Response format": self._format_response_format(RESPONSE_FORMAT_ABLATION)}

    def _draft(self) -> Node:
        # Select prompt based on memory configuration
        draft_intro = DRAFT_INTRO_WITH_MEMORY if self._is_memory_enabled else DRAFT_INTRO
        prompt: Any = {
            "Introduction": draft_intro,
            "Research idea": self.task_desc,
            "Instructions": {},
        }
        self._inject_memory(
            prompt, "draft", allow_summary_fallback=True, allow_empty=True
        )
        prompt["Instructions"]["Experiment design sketch guideline"] = list(DRAFT_EXP_GUIDELINES)
        prompt["Instructions"]["Evaluation Metric(s)"] = self.evaluation_metrics
        if self.phase_mode == "split":
            prompt = self._apply_split_prompt_layers(prompt, task_hint="phase2_draft")
        else:
            prompt["Instructions"] |= self._prompt_phase_guidance
            prompt["Instructions"] |= self._prompt_resp_fmt
            prompt["Instructions"] |= self._prompt_impl_guideline
            prompt["Instructions"] |= self._prompt_environment
            self._inject_resources(prompt, phase="phase2")

        print("[cyan]--------------------------------[/cyan]")
        print("[cyan]self.task_desc[/cyan]")
        print("[cyan]" + self.task_desc + "[/cyan]")
        print("[cyan]--------------------------------[/cyan]")

        print("MinimalAgent: Getting plan and code")
        if self.phase_mode == "split":
            # LLM context (draft node, split): Introduction/Research idea/Memory/Instructions (guidelines + metrics + response format + impl), optional Data Overview, plus System/Domain, Environment injection, Resources, and Phase 0 plan snippet.
            artifacts = self.generate_phase_artifacts(prompt, log_label="draft")
            files = artifacts["phase_artifacts"]["coding"]["workspace"]["files"]
            code_repr = combine_sources_for_display(files)
            plan = artifacts["phase_artifacts"]["coding"].get("notes", "") or "Split phase plan"
            return Node(
                plan=plan,
                code=code_repr,
                phase_artifacts=artifacts,
                phase_artifacts_raw=getattr(self, "last_phase_artifacts_response", ""),
            )
        # LLM context (draft node, single): Introduction/Research idea/Memory/Instructions (guidelines + metrics + phase guidance + response format + impl + installed packages), optional Data Overview, and Resources.
        plan, code = self.plan_and_code_query(prompt, log_label="draft")
        print("MinimalAgent: Draft complete")
        return Node(plan=plan, code=code)

    def _debug(self, parent_node: Node) -> Node:
        # Select prompt based on memory configuration
        debug_intro = DEBUG_INTRO_WITH_MEMORY if self._is_memory_enabled else DEBUG_INTRO
        prompt: Any = {
            "Introduction": debug_intro,
            "Research idea": self.task_desc,
            "Previous (buggy) implementation": wrap_combined_code(parent_node.code, fallback_lang=self.code_language),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Feedback based on generated plots": parent_node.vlm_feedback_summary,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        self._inject_memory(prompt, "debug")
        prompt["Instructions"] |= self._prompt_debug_resp_fmt
        prompt["Instructions"]["Bugfix improvement sketch guideline"] = list(DEBUG_BUGFIX_GUIDELINES)
        if self.phase_mode == "split":
            prompt = self._apply_split_prompt_layers(prompt, task_hint="phase2_debug")
        else:
            prompt["Instructions"] |= self._prompt_phase_guidance
            prompt["Instructions"] |= self._prompt_impl_guideline
            self._inject_resources(prompt, phase="phase2")

        if self.phase_mode == "split":
            # LLM context (debug node, split): Introduction/Research idea/buggy code + execution output + plot/time feedback, Instructions (debug format + bugfix guidelines + phase guidance + impl), plus System/Domain, Environment injection, Resources, and Phase 0 plan snippet.
            artifacts = self.generate_phase_artifacts(prompt, log_label="debug")
            files = artifacts["phase_artifacts"]["coding"]["workspace"]["files"]
            code_repr = combine_sources_for_display(files)
            plan = artifacts["phase_artifacts"]["coding"].get("notes", "") or "Split phase plan"
            return Node(
                plan=plan,
                code=code_repr,
                parent=parent_node,
                phase_artifacts=artifacts,
                phase_artifacts_raw=getattr(self, "last_phase_artifacts_response", ""),
            )
        # LLM context (debug node, single): Introduction/Research idea/buggy code + execution output + plot/time feedback, Instructions (debug format + bugfix guidelines + phase guidance + impl), optional Data Overview, and Resources.
        plan, code = self.plan_and_code_query(prompt, log_label="debug")
        return Node(plan=plan, code=code, parent=parent_node)

    def _improve(self, parent_node: Node) -> Node:
        # Select prompt based on memory configuration
        improve_intro = IMPROVE_INTRO_WITH_MEMORY if self._is_memory_enabled else IMPROVE_INTRO
        prompt: Any = {
            "Introduction": improve_intro,
            "Research idea": self.task_desc,
            "Feedback based on generated plots": parent_node.vlm_feedback_summary,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        self._inject_memory(
            prompt, "improve", allow_summary_fallback=True, allow_empty=True
        )
        prompt["Previous solution"] = {
            "Code": wrap_combined_code(parent_node.code, fallback_lang=self.code_language),
        }

        if self.phase_mode == "split":
            prompt = self._apply_split_prompt_layers(prompt, task_hint="phase2_improve")
        else:
            prompt["Instructions"] |= self._prompt_phase_guidance
            prompt["Instructions"] |= self._prompt_resp_fmt
            prompt["Instructions"] |= self._prompt_impl_guideline
            self._inject_resources(prompt, phase="phase2")

        if self.phase_mode == "split":
            # LLM context (improve node, split): Introduction/Research idea/Memory + prior code + plot/time feedback, Instructions (response format + phase guidance + impl), plus System/Domain, Environment injection, Resources, and Phase 0 plan snippet.
            artifacts = self.generate_phase_artifacts(prompt, log_label="improve")
            files = artifacts["phase_artifacts"]["coding"]["workspace"]["files"]
            code_repr = combine_sources_for_display(files)
            plan = artifacts["phase_artifacts"]["coding"].get("notes", "") or "Split phase plan"
            return Node(
                plan=plan,
                code=code_repr,
                parent=parent_node,
                phase_artifacts=artifacts,
                phase_artifacts_raw=getattr(self, "last_phase_artifacts_response", ""),
            )
        # LLM context (improve node, single): Introduction/Research idea/Memory + prior code + plot/time feedback, Instructions (response format + phase guidance + impl), and Resources.
        plan, code = self.plan_and_code_query(prompt, log_label="improve")
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _generate_seed_node(self, parent_node: Node):
        return Node(
            plan="Seed node",
            code=parent_node.code,
            parent=parent_node,
            is_seed_node=True,
            phase_artifacts=copy.deepcopy(parent_node.phase_artifacts),
            phase_artifacts_raw=getattr(parent_node, "phase_artifacts_raw", ""),
            worker_sif_path=getattr(parent_node, "worker_sif_path", None),
        )

    def _generate_hyperparam_tuning_node(
        self, parent_node: Node, hyperparam_idea: HyperparamTuningIdea
    ):
        # Select prompt based on memory configuration
        intro_prefix = (
            HYPERPARAM_NODE_INTRO_PREFIX_WITH_MEMORY
            if self._is_memory_enabled
            else HYPERPARAM_NODE_INTRO_PREFIX
        )
        prompt: Any = {
            "Introduction": intro_prefix + hyperparam_idea.name + ". " + hyperparam_idea.description,
            "Base code you are working on": wrap_combined_code(parent_node.code, fallback_lang=self.code_language),
            "Instructions": {},
        }
        self._inject_memory(prompt, "hyperparam_node")
        prompt["Instructions"]["Implementation guideline"] = list(HYPERPARAM_NODE_INSTRUCTIONS)
        if self.phase_mode == "split":
            prompt = self._apply_split_prompt_layers(prompt, task_hint="phase2_hyperparam")
        else:
            prompt["Instructions"] |= self._prompt_phase_guidance
            prompt["Instructions"] |= self._prompt_hyperparam_tuning_resp_fmt
            self._inject_resources(prompt, phase="phase2")
        if self.phase_mode == "split":
            # LLM context (Stage 2 hyperparam, split): Introduction (idea), base code, Instructions (hyperparam guideline + phase guidance + response format + impl), plus System/Domain, Environment injection, Resources, and Phase 0 plan snippet.
            artifacts = self.generate_phase_artifacts(prompt, log_label="hyperparam")
            files = artifacts["phase_artifacts"]["coding"]["workspace"]["files"]
            code_repr = combine_sources_for_display(files)
            plan = artifacts["phase_artifacts"]["coding"].get("notes", "") or "Split phase plan"
            return Node(
                plan="Hyperparam tuning name: " + hyperparam_idea.name + ".\n" + plan,
                code=code_repr,
                parent=parent_node,
                hyperparam_name=hyperparam_idea.name,
                phase_artifacts=artifacts,
                phase_artifacts_raw=getattr(self, "last_phase_artifacts_response", ""),
            )
        # LLM context (Stage 2 hyperparam, single): Introduction (idea), base code, Instructions (hyperparam guideline + phase guidance + response format), and Resources.
        plan, code = self.plan_and_code_query(prompt, log_label="hyperparam")
        return Node(
            plan="Hyperparam tuning name: " + hyperparam_idea.name + ".\n" + plan,
            code=code,
            parent=parent_node,
            hyperparam_name=hyperparam_idea.name,
        )

    def _generate_ablation_node(self, parent_node: Node, ablation_idea: AblationIdea):
        # Select prompt based on memory configuration
        intro_prefix = (
            ABLATION_NODE_INTRO_PREFIX_WITH_MEMORY
            if self._is_memory_enabled
            else ABLATION_NODE_INTRO_PREFIX
        )
        prompt: Any = {
            "Introduction": intro_prefix + ablation_idea.name + ". " + ablation_idea.description,
            "Base code you are working on": wrap_combined_code(parent_node.code, fallback_lang=self.code_language),
            "Instructions": {},
        }
        self._inject_memory(prompt, "ablation_node")
        prompt["Instructions"]["Implementation guideline"] = list(ABLATION_NODE_INSTRUCTIONS)
        if self.phase_mode == "split":
            prompt = self._apply_split_prompt_layers(prompt, task_hint="phase2_ablation")
        else:
            prompt["Instructions"] |= self._prompt_phase_guidance
            prompt["Instructions"] |= self._prompt_ablation_resp_fmt
            self._inject_resources(prompt, phase="phase2")
        if self.phase_mode == "split":
            # LLM context (Stage 4 ablation, split): Introduction (idea), base code, Instructions (ablation guideline + phase guidance + response format + impl), plus System/Domain, Environment injection, Resources, and Phase 0 plan snippet.
            artifacts = self.generate_phase_artifacts(prompt, log_label="ablation")
            files = artifacts["phase_artifacts"]["coding"]["workspace"]["files"]
            code_repr = combine_sources_for_display(files)
            plan = artifacts["phase_artifacts"]["coding"].get("notes", "") or "Split phase plan"
            return Node(
                plan="Ablation name: " + ablation_idea.name + ".\n" + plan,
                code=code_repr,
                parent=parent_node,
                ablation_name=ablation_idea.name,
                phase_artifacts=artifacts,
                phase_artifacts_raw=getattr(self, "last_phase_artifacts_response", ""),
            )
        # LLM context (Stage 4 ablation, single): Introduction (idea), base code, Instructions (ablation guideline + phase guidance + response format), and Resources.
        plan, code = self.plan_and_code_query(prompt, log_label="ablation")
        return Node(
            plan="Ablation name: " + ablation_idea.name + ".\n" + plan,
            code=code,
            parent=parent_node,
            ablation_name=ablation_idea.name,
        )

    def _validate_phase_language(self, artifacts: dict) -> None:
        required = self.code_language
        if required == "python":
            return
        phase_data = artifacts.get("phase_artifacts") or artifacts
        if not isinstance(phase_data, dict):
            raise PhasePlanError("phase_artifacts must be an object.")
        compile_section = phase_data.get("compile", {})
        build_plan = compile_section.get("build_plan", {}) if isinstance(compile_section, dict) else {}
        actual = _normalize_language(build_plan.get("language"))
        if not actual or actual != required:
            raise PhasePlanError(
                f"build_plan.language must be '{required}', got '{actual or 'missing'}'."
            )

    def _fallback_phase_artifacts(self, last_error: str) -> dict:
        output_path = self.experiment_output_path
        lang = self.code_language
        if lang == "cpp":
            file_path = "src/main.cpp"
            content = (
                "// Fallback placeholder due to LLM parse failure\n"
                f"// Error: {last_error}\n"
                "#include <iostream>\n\n"
                "int main() {\n"
                "    std::cout << \"Placeholder execution; no real code generated\" << std::endl;\n"
                "    return 0;\n"
                "}\n"
            )
            compile_plan = {
                "language": "cpp",
                "compiler_selected": "g++",
                "cflags": ["-std=c++17", "-O2"],
                "ldflags": [],
                "workdir": "/workspace",
                "output": "bin/a.out",
            }
            compile_commands = [
                "{compiler_selected} -std=c++17 -O2 src/main.cpp -o bin/a.out"
            ]
            run_commands = ["./bin/a.out"]
            compile_notes = "fallback compile plan"
        elif lang == "fortran":
            file_path = "src/main.f90"
            content = (
                "! Fallback placeholder due to LLM parse failure\n"
                f"! Error: {last_error}\n"
                "program main\n"
                "    implicit none\n"
                "    print *, 'Placeholder execution; no real code generated'\n"
                "end program main\n"
            )
            compile_plan = {
                "language": "fortran",
                "compiler_selected": "gfortran",
                "cflags": ["-O2"],
                "ldflags": [],
                "workdir": "/workspace",
                "output": "bin/a.out",
            }
            compile_commands = [
                "{compiler_selected} -O2 src/main.f90 -o bin/a.out"
            ]
            run_commands = ["./bin/a.out"]
            compile_notes = "fallback compile plan"
        else:
            file_path = "src/main.py"
            content = (
                "# Fallback placeholder due to LLM parse failure\n"
                f"# Error: {last_error}\n"
                "print('Placeholder execution; no real code generated')\n"
            )
            compile_plan = {
                "language": "python",
                "compiler_selected": "",  # Empty for python - no compilation needed
                "cflags": [],
                "ldflags": [],
                "workdir": "/workspace",
                "output": output_path,
            }
            compile_commands = []
            run_commands = ["python3 src/main.py"]
            compile_notes = "no compile needed for python"

        return {
            "phase_artifacts": {
                "download": {"commands": [], "notes": f"fallback after parse error: {last_error}"},
                "coding": {
                    "workspace": {
                        "root": "/workspace",
                        "tree": ["workspace/", "workspace/src/", "workspace/working/"],
                        "files": [
                            {
                                "path": file_path,
                                "mode": "0644",
                                "encoding": "utf-8",
                                "content": content,
                            }
                        ],
                    },
                    "notes": "fallback coding plan",
                },
                "compile": {
                    "build_plan": compile_plan,
                    "commands": compile_commands,
                    "notes": compile_notes,
                },
                "run": {
                    "commands": run_commands,
                    "expected_outputs": [output_path],
                    "notes": "fallback run",
                },
            },
            "constraints": {
                "allow_sudo_in_singularity": True,
                "allow_apt_get_in_singularity": True,
                "write_only_under_workspace": True,
                "no_absolute_paths": True,
                "no_parent_traversal": True,
                "python_output_must_use_numpy": True,
                "non_python_output_must_use_cnpy": True,
            },
        }

    def generate_phase_artifacts(
        self,
        prompt,
        retries: int = 3,
        log_label: str = "phase2",
        max_memory_read_rounds: Optional[int] = None,
    ) -> dict:
        """Query the LLM for split-phase output and validate the JSON structure.

        When memory is enabled and the LLM requests read operations (core_get,
        archival_search, recall_search), the system will execute those operations
        and re-query the LLM with the results, allowing it to make informed decisions.

        Args:
            prompt: The prompt dict to send to the LLM.
            retries: Number of retry attempts for parsing failures.
            log_label: Label for logging purposes.
            max_memory_read_rounds: Maximum number of memory read + re-query cycles
                to prevent infinite loops. If None, uses config value
                (memory.max_memory_read_rounds, default 2).

        Returns:
            The parsed phase artifacts dict.
        """
        last_error = ""
        last_response = ""
        memory_enabled = self._is_memory_enabled
        memory_read_round = 0

        # Get max rounds from config or parameter
        if max_memory_read_rounds is None:
            max_memory_read_rounds = getattr(
                getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2
            )

        # Hard ceiling on total LLM calls to guarantee termination even if
        # the retry-rollback logic has an unforeseen interaction.  The
        # theoretical maximum is retries (parse errors) + max_memory_read_rounds
        # (free re-queries), plus a small margin.
        max_total_calls = retries + max_memory_read_rounds + 1
        total_calls = 0

        attempt = 0
        while attempt < retries:
            attempt += 1
            total_calls += 1
            if total_calls > max_total_calls:
                logger.warning(
                    "generate_phase_artifacts: reached hard call ceiling "
                    "(%d calls). Breaking to prevent infinite loop.",
                    max_total_calls,
                )
                break
            self._log_prompt(
                prompt,
                label=f"{log_label}_attempt{attempt}_round{memory_read_round}",
                meta={
                    "phase": "phase2",
                    "attempt": attempt,
                    "memory_read_round": memory_read_round,
                    "label": log_label,
                    "model": self.cfg.agent.code.model,
                    "memory_enabled": memory_enabled,
                },
            )
            # LLM context (split-phase artifacts): system_message=prompt dict with task-specific sections, Instructions/Response format, optional System/Domain/Environment/Resources/Phase 0 snippet; request JSON for download/coding/compile/run.
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )
            # Log raw response for debugging
            self._log_response(
                completion_text,
                label=f"{log_label}_attempt{attempt}_round{memory_read_round}",
                meta={
                    "phase": "phase2",
                    "attempt": attempt,
                    "memory_read_round": memory_read_round,
                    "model": self.cfg.agent.code.model,
                    "memory_enabled": memory_enabled,
                },
            )
            try:
                self.last_phase_artifacts_response = completion_text
                artifacts = extract_phase_artifacts(
                    completion_text,
                    default_language=self.code_language,
                    require_memory_update=memory_enabled,
                    experiment_name=self.experiment_name,
                )
                self._validate_phase_language(artifacts)

                # Clear any stale parsing feedback after a successful parse
                prompt.pop("Parsing Feedback", None)

                # Apply memory updates if present
                if memory_enabled and self.memory_manager and self.branch_id:
                    memory_updates = artifacts.get("memory_update")
                    if memory_updates:
                        try:
                            memory_results = self.memory_manager.apply_llm_memory_updates(
                                self.branch_id,
                                memory_updates,
                                node_id=getattr(self, "_current_node_id", None),
                                phase=log_label,
                            )
                            # Store memory operation results for prompt logging
                            self._last_memory_operation_results = memory_results

                            # Log memory operations immediately for reproducibility
                            if self.prompt_log_dir and memory_results:
                                self._log_memory_operations_detail(
                                    memory_updates=memory_updates,
                                    memory_results=memory_results,
                                    label=f"{log_label}_attempt{attempt}_round{memory_read_round}",
                                    meta={
                                        "phase": "phase2",
                                        "attempt": attempt,
                                        "memory_read_round": memory_read_round,
                                        "has_read_results": _has_memory_read_results(memory_results),
                                    },
                                )

                            # Check if there are read results and we haven't exceeded max rounds
                            if (
                                _has_memory_read_results(memory_results)
                                and memory_read_round < max_memory_read_rounds
                            ):
                                memory_read_round += 1
                                # Format results and inject into prompt for re-query
                                results_text = _format_memory_results_for_llm(memory_results)
                                prompt["Memory Read Results"] = (
                                    "Your memory read operations returned the following results. "
                                    "Use this information to make your final decision and output "
                                    "the complete phase_artifacts JSON.\n\n"
                                    f"{results_text}\n\n"
                                    "Now provide your final response with:\n"
                                    "1. A <memory_update> block (can include additional writes based on what you learned)\n"
                                    "2. The complete phase_artifacts JSON\n\n"
                                    "Do NOT include read operations in this response."
                                )
                                # Re-query does NOT consume a retry slot
                                attempt -= 1
                                continue

                        except Exception as exc:
                            logger.warning("Failed to apply LLM memory updates: %s", exc)

                # Remove any memory read results section before returning
                prompt.pop("Memory Read Results", None)
                return artifacts

            except MemoryReadOnlyResponse as exc:
                # The LLM returned only a memory-read request with no JSON body.
                # Execute the read, inject results, and re-query WITHOUT
                # consuming a retry slot.
                if (
                    memory_enabled
                    and self.memory_manager
                    and self.branch_id
                    and memory_read_round < max_memory_read_rounds
                ):
                    try:
                        memory_results = self.memory_manager.apply_llm_memory_updates(
                            self.branch_id,
                            exc.memory_updates,
                            node_id=getattr(self, "_current_node_id", None),
                            phase=log_label,
                        )
                        self._last_memory_operation_results = memory_results

                        if self.prompt_log_dir and memory_results:
                            self._log_memory_operations_detail(
                                memory_updates=exc.memory_updates,
                                memory_results=memory_results,
                                label=f"{log_label}_attempt{attempt}_round{memory_read_round}",
                                meta={
                                    "phase": "phase2",
                                    "attempt": attempt,
                                    "memory_read_round": memory_read_round,
                                    "has_read_results": _has_memory_read_results(memory_results),
                                },
                            )

                        if _has_memory_read_results(memory_results):
                            memory_read_round += 1
                            results_text = _format_memory_results_for_llm(memory_results)
                            prompt["Memory Read Results"] = (
                                "Your memory read operations returned the following results. "
                                "Use this information to make your final decision and output "
                                "the complete phase_artifacts JSON.\n\n"
                                f"{results_text}\n\n"
                                "Now provide your final response with:\n"
                                "1. A <memory_update> block (can include additional writes based on what you learned)\n"
                                "2. The complete phase_artifacts JSON\n\n"
                                "Do NOT include read operations in this response."
                            )
                            # Clear any stale parsing feedback — the LLM did
                            # not actually fail, it just needed to read first.
                            prompt.pop("Parsing Feedback", None)
                            # Do NOT consume a retry slot
                            attempt -= 1
                            continue

                    except Exception as inner_exc:
                        logger.warning(
                            "Failed to execute memory read from read-only response: %s",
                            inner_exc,
                        )

                # If we reach here the read couldn't be serviced (memory
                # disabled, max rounds exceeded, or execution failed).  Fall
                # through to the normal PhasePlanError path so the LLM gets
                # feedback asking it to include the JSON body next time.
                last_error = str(exc)
                last_response = completion_text
                feedback_msg = (
                    "Your previous response contained only a memory read request "
                    "without the required phase_artifacts JSON.\n"
                    "You MUST include the complete phase_artifacts JSON in your "
                    "response even when requesting memory reads.\n"
                    "Start with <memory_update>...</memory_update>, then "
                    "immediately output the phase_artifacts JSON (no markdown fences)."
                )
                prompt["Parsing Feedback"] = feedback_msg

            except MissingMemoryUpdateError as exc:
                # Memory update is required but missing - provide specific feedback
                last_error = str(exc)
                last_response = completion_text
                prompt["Parsing Feedback"] = (
                    "CRITICAL: Your response is missing the required <memory_update> block.\n"
                    "When memory is enabled, you MUST start your response with:\n"
                    "<memory_update>\n"
                    '{"core": {...}, "archival": [...]}\n'
                    "</memory_update>\n\n"
                    "Then immediately follow with the JSON for phase_artifacts (no markdown fences).\n"
                    "Even if you have no memory updates, include an empty block: <memory_update>{}</memory_update>\n\n"
                    "Raw response was:\n"
                    "<<<RAW_RESPONSE_START>>>\n"
                    f"{completion_text}\n"
                    "<<<RAW_RESPONSE_END>>>\n"
                )
            except PhasePlanError as exc:
                last_error = str(exc)
                last_response = completion_text
                feedback_msg = (
                    "The previous response failed validation: "
                    f"{last_error}.\n"
                    "Raw response was:\n"
                    "<<<RAW_RESPONSE_START>>>\n"
                    f"{completion_text}\n"
                    "<<<RAW_RESPONSE_END>>>\n"
                )
                if memory_enabled:
                    feedback_msg += (
                        "\nRemember: Start with <memory_update>...</memory_update>, "
                        "then output JSON for phase_artifacts (no markdown fences)."
                    )
                else:
                    feedback_msg += (
                        "\nReturn strict JSON following the Response format with download/coding/compile/run."
                    )
                prompt["Parsing Feedback"] = feedback_msg

        # Fallback: return a minimal placeholder plan to keep execution moving
        if last_response:
            self.last_phase_artifacts_response = last_response
        return self._fallback_phase_artifacts(last_error)

    def plan_and_code_query(
        self,
        prompt,
        retries=3,
        code_language: Optional[str] = None,
        log_label: str = "plan_and_code",
    ) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        target_language = code_language or self.code_language

        memory_read_round = 0
        max_memory_read_rounds = getattr(
            getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2
        )
        max_total_calls = retries + max_memory_read_rounds + 1
        total_calls = 0

        attempt = 0
        while attempt < retries:
            attempt += 1
            total_calls += 1
            if total_calls > max_total_calls:
                logger.warning(
                    "plan_and_code_query: reached hard call ceiling (%d). Breaking.",
                    max_total_calls,
                )
                break

            self._log_prompt(
                prompt,
                label=f"{log_label}_attempt{attempt}",
                meta={
                    "phase": "single",
                    "attempt": attempt,
                    "memory_read_round": memory_read_round,
                    "label": log_label,
                    "model": self.cfg.agent.code.model,
                },
            )
            # LLM context (plan+code): system_message=prompt dict built by caller (task sections + instructions/format/env/resources as applicable).
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            # Process LLM memory updates if present
            memory_read_handled = False
            if completion_text and self.memory_manager and self.branch_id:
                memory_updates = extract_memory_updates(completion_text)
                if memory_updates:
                    try:
                        memory_results = self.memory_manager.apply_llm_memory_updates(
                            self.branch_id,
                            memory_updates,
                            node_id=getattr(self, "_current_node_id", None),
                            phase=log_label,
                        )
                        # Store memory operation results for prompt logging
                        self._last_memory_operation_results = memory_results

                        # Log memory operations immediately for reproducibility
                        if self.prompt_log_dir and memory_results:
                            self._log_memory_operations_detail(
                                memory_updates=memory_updates,
                                memory_results=memory_results,
                                label=f"{log_label}_attempt{attempt}",
                                meta={
                                    "phase": "plan_code",
                                    "attempt": attempt,
                                    "memory_read_round": memory_read_round,
                                    "has_read_results": _has_memory_read_results(memory_results),
                                },
                            )

                        # If read results exist, inject them and re-query
                        if (
                            _has_memory_read_results(memory_results)
                            and memory_read_round < max_memory_read_rounds
                        ):
                            memory_read_round += 1
                            results_text = _format_memory_results_for_llm(memory_results)
                            prompt["Memory Read Results"] = (
                                "Your memory read operations returned the following results. "
                                "Use this information to produce your plan and code.\n\n"
                                f"{results_text}\n\n"
                                "Now provide your final response with:\n"
                                "1. A <memory_update> block (can include additional writes)\n"
                                "2. Your plan and code\n\n"
                                "Do NOT include read operations in this response."
                            )
                            prompt.pop("Parsing Feedback", None)
                            # Do NOT consume a retry slot
                            attempt -= 1
                            memory_read_handled = True
                    except Exception as exc:
                        logger.warning("Failed to apply LLM memory updates: %s", exc)
                    # Remove memory update tags from completion text before code extraction
                    completion_text = remove_memory_update_tags(completion_text)

            if memory_read_handled:
                continue

            code = extract_code(completion_text, language=target_language)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                prompt.pop("Memory Read Results", None)
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                f"The code extraction failed. Make sure to use the format ```{target_language} ... ``` for the code blocks."
            )
        prompt.pop("Memory Read Results", None)
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def parse_exec_result(
        self, node: Node, exec_result: ExecutionResult, workspace: str
    ):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # Select prompt based on memory configuration
        exec_review_intro = (
            EXECUTION_REVIEW_INTRO_WITH_MEMORY
            if self._is_memory_enabled
            else EXECUTION_REVIEW_INTRO
        )
        prompt = {
            "Introduction": exec_review_intro,
            "Research idea": self.task_desc,
            "Implementation": wrap_combined_code(node.code, fallback_lang=self.code_language),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        branch_id = getattr(node, "branch_id", None)
        self._inject_memory(prompt, "execution_review", branch_id=branch_id)

        # Phase 1: Memory update with multi-round support (optional, only if memory is enabled)
        if self._is_memory_enabled and self.memory_manager and branch_id:
            max_rounds = getattr(getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2)
            _run_memory_update_phase(
                prompt=prompt,
                memory_manager=self.memory_manager,
                branch_id=branch_id,
                node_id=node.id,
                phase_name="execution_review",
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_rounds=max_rounds,
                task_description=(
                    "Review the execution results and update your memory with any important findings, "
                    "patterns, or lessons learned. You can also search memory for related past experiences. "
                    "Respond with ONLY a <memory_update> block containing your memory operations."
                ),
                log_dir=self.prompt_log_dir,
            )

        # Phase 2: Task execution with structured response
        response = cast(
            dict,
            # LLM context (execution review): Introduction + Research idea + Implementation + Execution output + Memory.
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )

        is_bug = True
        summary = ""
        if isinstance(response, dict):
            is_bug = bool(response.get("is_bug", False))
            summary = response.get("summary") or ""
        else:
            logger.warning(
                "Execution review returned non-dict response: %s", type(response)
            )

        node.analysis = summary
        node.is_buggy = is_bug or node.exc_type is not None
        print(
            "[red]Checking if response contains metric name and description[/red]",
            flush=True,
        )
        print(response)

    def _generate_plotting_code(
        self, node: Node, working_dir: str, plot_code_from_prev_stage: str = None
    ) -> str:
        """Generate code for plotting experiment results"""
        prompt_guideline = list(PLOTTING_GUIDELINE_BASE)
        working_path = Path(working_dir)
        if working_path.exists():
            npy_files = sorted(working_path.rglob("*.npy"))
            if npy_files:
                rel_paths = []
                for p in npy_files:
                    try:
                        rel_paths.append(str(p.relative_to(working_path)))
                    except ValueError:
                        rel_paths.append(str(p))
                prompt_guideline.append(
                    "Available .npy files (relative to working_dir unless absolute):\n"
                    + "\n".join(rel_paths)
                )
        prompt_guideline.append(
            "Use the following experiment code to infer the data to plot: " + node.code
        )
        prompt_guideline.extend(PLOTTING_GUIDELINE_TAIL)
        # add instruction for format
        plotting_prompt = {
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= {
            "Response format": RESPONSE_FORMAT_DEFAULT
        }
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }
        self._inject_memory(
            plotting_prompt,
            "plotting_code",
            branch_id=getattr(node, "branch_id", None),
            budget_chars=getattr(self.cfg.memory, "plotting_code_budget_chars", 2000),
        )

        # For stage 3, initialize with stage 2's plotting code
        if (
            self.stage_name
            and self.stage_name.startswith("3_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: Use the following base plotting code as a starting point:",
                    "Base plotting code: " + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between different datasets",
                    "3. Add dataset-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from which dataset",
                    "5. Use consistent naming conventions for saved files",
                ]
            )
        # For stage 4, initialize with stage 3's plotting code
        elif (
            self.stage_name
            and self.stage_name.startswith("4_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: This is an ablation study. Use the following base plotting code as a starting point:",
                    "Base plotting code: \n" + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between ablation and baseline results",
                    "3. Add ablation-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from ablation vs baseline",
                    "5. Use consistent naming conventions for saved files",
                ]
            )

        # LLM context (plotting code): Instructions with Response format + Plotting code guideline (incl. experiment code, optional base plotting code).
        plan, code = self.plan_and_code_query(
            plotting_prompt, code_language="python"
        )

        if self.code_language in ("python", "cpp"):
            imports_to_add: List[str] = []
            if "import matplotlib.pyplot as plt" not in code:
                imports_to_add.append("import matplotlib.pyplot as plt")
            if "import numpy as np" not in code:
                imports_to_add.append("import numpy as np")
            if "import os" not in code:
                imports_to_add.append("import os")
            if "from pathlib import Path" not in code:
                imports_to_add.append("from pathlib import Path")

            if imports_to_add:
                code = "\n".join(imports_to_add) + "\n\n" + code
        node.plot_code = code
        node.plot_plan = plan

        return code

    def _determine_datasets_successfully_tested(self, node: Node) -> List[str]:
        """Determine which datasets are successfully tested based on VLM feedback"""
        plot_analyses = ""
        for i, plot_analysis in enumerate(node.plot_analyses):
            plot_analyses += f"plot {i+1}: {plot_analysis['analysis']}\n"

        determine_prompt = {
            "Introduction": DETERMINE_DATASETS_INTRO,
            "Plot analyses": plot_analyses,
            "VLM feedback summary": node.vlm_feedback_summary,
            "Original plotting code": node.plot_code,
            "Response format": DETERMINE_DATASETS_RESPONSE,
        }
        self._inject_memory(
            determine_prompt,
            "datasets_successfully_tested",
            branch_id=getattr(node, "branch_id", None),
            budget_chars=getattr(self.cfg.memory, "datasets_tested_budget_chars", 1500),
        )

        retry_count = 0
        retry_limit = 5
        while retry_count < retry_limit:
            # LLM context (dataset success check): Introduction + Plot analyses + VLM feedback summary + Original plotting code + Response format.
            response = query(
                system_message=determine_prompt,
                user_message=None,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )

            # Process LLM memory updates if present
            if response and self.memory_manager and self.branch_id:
                memory_updates = extract_memory_updates(response)
                if memory_updates:
                    try:
                        memory_results = self.memory_manager.apply_llm_memory_updates(
                            self.branch_id,
                            memory_updates,
                            node_id=getattr(self, "_current_node_id", None),
                            phase="datasets_successfully_tested",
                        )
                        # Store memory operation results for prompt logging
                        self._last_memory_operation_results = memory_results

                        # Log memory operations immediately for reproducibility
                        if self.prompt_log_dir and memory_results:
                            self._log_memory_operations_detail(
                                memory_updates=memory_updates,
                                memory_results=memory_results,
                                label=f"datasets_tested_retry{retry_count}",
                                meta={
                                    "phase": "datasets_successfully_tested",
                                    "retry_count": retry_count,
                                    "has_read_results": _has_memory_read_results(memory_results),
                                },
                            )
                    except Exception as exc:
                        logger.warning("Failed to apply LLM memory updates: %s", exc)
                    # Remove memory update tags before parsing
                    response = remove_memory_update_tags(response)

            (
                reasoning,
                datasets_successfully_tested_str,
            ) = _parse_keyword_prefix_response(
                response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
            )
            print(f"[green]Reasoning:[/green] {reasoning}")
            print(
                f"[green]Datasets successfully tested:[/green] {datasets_successfully_tested_str}"
            )
            if reasoning is not None and datasets_successfully_tested_str is not None:
                if datasets_successfully_tested_str == "":
                    return [""]
                # Split by comma and clean each dataset name
                datasets = [
                    ds.strip() for ds in datasets_successfully_tested_str.split(",")
                ]
                # Filter out empty strings and ensure all elements are strings
                datasets = [ds for ds in datasets if isinstance(ds, str) and ds]
                logger.info(f"Successfully parsed datasets: {datasets}")
                return datasets

            retry_count += 1
            logger.warning(
                f"Failed to parse successfully tested datasets response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse successfully tested datasets response after {retry_limit} retries. Falling back to an empty list."
        )
        return [""]

    def _analyze_plots_with_vlm(self, node: Node) -> None:
        """Analyze experimental plots using VLM"""
        if not node.plot_paths:
            return

        # for debugging
        print(f"[cyan]Plot paths:[/cyan] {node.plot_paths}")

        def encode_image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                try:
                    return base64.b64encode(image_file.read()).decode("utf-8")
                except Exception as e:
                    print(f"[red]Error encoding image {image_path}: {e}[/red]")
                    return None

        if not len(node.plot_paths) > 10:
            selected_plots = node.plot_paths
        else:
            print(
                f"[red]Warning: {len(node.plot_paths)} plots received, this may be too many to analyze effectively. Calling LLM to select the most relevant plots to analyze.[/red]"
            )
            # select 10 plots to analyze
            prompt_select_plots = {
                "Introduction": SELECT_PLOTS_INTRO,
                "Plot paths": node.plot_paths,
            }
            branch_id = getattr(node, "branch_id", None)
            self._inject_memory(
                prompt_select_plots,
                "plot_selection",
                branch_id=branch_id,
                budget_chars=getattr(self.cfg.memory, "plot_selection_budget_chars", 1000),
            )

            # Phase 1: Memory update with multi-round support (optional, only if memory is enabled)
            if self._is_memory_enabled and self.memory_manager and branch_id:
                max_rounds = getattr(getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2)
                _run_memory_update_phase(
                    prompt=prompt_select_plots,
                    memory_manager=self.memory_manager,
                    branch_id=branch_id,
                    node_id=node.id,
                    phase_name="plot_selection",
                    model=self.cfg.agent.feedback.model,
                    temperature=self.cfg.agent.feedback.temp,
                    max_rounds=max_rounds,
                    task_description=(
                        "Review the available plots and update your memory with any observations about "
                        "plot naming patterns, experiment types, or useful categorizations. "
                        "You can also search memory for related past observations. "
                        "Respond with ONLY a <memory_update> block containing your memory operations."
                    ),
                    log_dir=self.prompt_log_dir,
                )

            # Phase 2: Task execution with structured response
            try:
                response_select_plots = cast(
                    dict,
                    # LLM context (plot selection): Introduction + Plot paths.
                    query(
                        system_message=prompt_select_plots,
                        user_message=None,
                        func_spec=plot_selection_spec,
                        model=self.cfg.agent.feedback.model,
                        temperature=self.cfg.agent.feedback.temp,
                    ),
                )

                print(f"[cyan]Plot selection response:[/cyan] {response_select_plots}")
                # Extract the plot paths list
                selected_plots = response_select_plots.get("selected_plots", [])

                # Validate that all paths exist and are image files
                valid_plots = []
                for plot_path in selected_plots:
                    if (
                        isinstance(plot_path, str)
                        and os.path.exists(plot_path)
                        and plot_path.lower().endswith((".png", ".jpg", ".jpeg"))
                    ):
                        valid_plots.append(plot_path)
                    else:
                        logger.warning(f"Invalid plot path received: {plot_path}")

                # Use the validated list
                if valid_plots:
                    print(f"[cyan]Selected valid plots:[/cyan] {valid_plots}")
                    selected_plots = valid_plots
                else:
                    logger.warning(
                        "No valid plot paths found in response, falling back to first 10 plots"
                    )
                    # fallback to first 10 plots
                    # validate node.plot_paths
                    selected_plots = []
                    for plot_path in node.plot_paths[:10]:
                        if os.path.exists(plot_path) and plot_path.lower().endswith(
                            (".png", ".jpg", ".jpeg")
                        ):
                            selected_plots.append(plot_path)
                        else:
                            logger.warning(f"Invalid plot path received: {plot_path}")

            except Exception as e:
                logger.error(
                    f"Error in plot selection: {str(e)}; falling back to first 10 plots"
                )
                # Fallback to using first 10 plots
                selected_plots = node.plot_paths[:10]

        print("[cyan]Before encoding images[/cyan]")
        branch_id = getattr(node, "branch_id", None)

        # Phase 1: Memory update with multi-round support (optional, only if memory is enabled)
        # This allows VLM to update memory with observations before structured analysis
        if self._is_memory_enabled and self.memory_manager and branch_id:
            vlm_memory_prompt = {
                "Introduction": "You are about to analyze experimental plots. Before the visual analysis, "
                               "review any relevant memory and prepare to record your findings.",
                "Research idea": self.task_desc,
                "Plot paths to analyze": selected_plots,
            }
            self._inject_memory(
                vlm_memory_prompt,
                "vlm_analysis",
                branch_id=branch_id,
                budget_chars=getattr(self.cfg.memory, "vlm_analysis_budget_chars", 1000),
            )
            max_rounds = getattr(getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2)
            _run_memory_update_phase(
                prompt=vlm_memory_prompt,
                memory_manager=self.memory_manager,
                branch_id=branch_id,
                node_id=node.id,
                phase_name="vlm_analysis",
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_rounds=max_rounds,
                task_description=(
                    "Review the experiment context and update your memory with any relevant observations. "
                    "You can search memory for related past VLM analyses or plot patterns. "
                    "After the visual analysis, key findings will be recorded separately. "
                    "Respond with ONLY a <memory_update> block containing your memory operations."
                ),
                log_dir=self.prompt_log_dir,
            )

        # Phase 2: Task execution with VLM (structured response via func_spec)
        memory_context = self._memory_context(
            "vlm_analysis",
            branch_id=branch_id,
            budget_chars=getattr(self.cfg.memory, "vlm_analysis_budget_chars", 1000),
        )
        memory_context_block = (
            f"Memory:\n{memory_context}\n\n" if memory_context else ""
        )
        # Select prompt template based on memory configuration
        # Note: Use template WITHOUT memory update instructions since Phase 1 handles memory
        vlm_template = VLM_ANALYSIS_PROMPT_TEMPLATE if self._is_memory_enabled else VLM_ANALYSIS_PROMPT_TEMPLATE
        analysis_text = vlm_template.format(
            memory_context_block=memory_context_block,
            task_desc=self.task_desc,
        )
        user_message = [
            {
                "type": "text",
                "text": analysis_text,
            }
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(plot_path)}"
                },
            }
            for plot_path in selected_plots
        ]

        response = cast(
            dict,
            # LLM context (VLM analysis): user_message with research idea text + selected plot images (base64).
            query(
                system_message=None,
                user_message=user_message,
                func_spec=vlm_feedback_spec,
                model=self.cfg.agent.vlm_feedback.model,
                temperature=self.cfg.agent.vlm_feedback.temp,
            ),
        )

        # Phase 3: Post-VLM memory update - record VLM findings to memory
        if self._is_memory_enabled and self.memory_manager and branch_id:
            try:
                # Write VLM analysis summary to archival memory
                vlm_summary = response.get("vlm_feedback_summary", "")
                if vlm_summary:
                    self.memory_manager.mem_archival_write(
                        text=f"VLM Analysis for node {node.id}: {vlm_summary}",
                        tags=["VLM_ANALYSIS", f"node:{node.id}"],
                        meta={"node_id": node.id, "branch_id": branch_id, "phase": "vlm_analysis"},
                    )
                # Update core memory with analysis status
                self.memory_manager.set_core(
                    branch_id,
                    "last_vlm_analysis",
                    f"node:{node.id}, plots:{len(selected_plots)}, valid:{response.get('valid_plots_received', False)}",
                    importance=3,
                    op_name="vlm_analysis_complete",
                    phase="vlm_analysis",
                    node_id=node.id,
                )
            except Exception as e:
                logger.warning(f"Failed to write VLM analysis to memory: {e}")
        print(
            f"[cyan]VLM response from {self.cfg.agent.vlm_feedback.model}:[/cyan] {response}"
        )
        if response["valid_plots_received"]:
            node.is_buggy_plots = False
        else:
            node.is_buggy_plots = True

        for index, analysis in enumerate(response["plot_analyses"]):
            analysis["plot_path"] = selected_plots[index]

        node.plot_analyses = response["plot_analyses"]
        node.vlm_feedback_summary = response["vlm_feedback_summary"]

        node.datasets_successfully_tested = (
            self._determine_datasets_successfully_tested(node)
        )

    def _generate_node_summary(self, node: Node) -> dict:
        """Generate a summary of the node's experimental findings"""
        # Select prompt based on memory configuration
        summary_intro = SUMMARY_INTRO_WITH_MEMORY if self._is_memory_enabled else SUMMARY_INTRO
        summary_prompt = {
            "Introduction": summary_intro,
            "Research idea": self.task_desc,
            "Implementation": wrap_combined_code(node.code, fallback_lang=self.code_language),
            "Plan": node.plan,
            "Execution output": wrap_code(node.term_out, lang=""),
            "Analysis": node.analysis,
            "Metric": str(node.metric) if node.metric else "Failed",
            "Plot Analyses": (
                node.plot_analyses if hasattr(node, "plot_analyses") else []
            ),
            "VLM Feedback": (
                node.vlm_feedback_summary
                if hasattr(node, "vlm_feedback_summary")
                else ""
            ),
        }
        branch_id = getattr(node, "branch_id", None)
        self._inject_memory(
            summary_prompt,
            "node_summary",
            branch_id=branch_id,
            budget_chars=getattr(self.cfg.memory, "node_summary_budget_chars", 2000),
        )

        # Phase 1: Memory update with multi-round support (optional, only if memory is enabled)
        if self._is_memory_enabled and self.memory_manager and branch_id:
            max_rounds = getattr(getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2)
            _run_memory_update_phase(
                prompt=summary_prompt,
                memory_manager=self.memory_manager,
                branch_id=branch_id,
                node_id=node.id,
                phase_name="node_summary",
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_rounds=max_rounds,
                task_description=(
                    "Review the experiment results and update your memory with key findings, "
                    "lessons learned, successful approaches, and patterns to avoid in future experiments. "
                    "You can also search memory for related past experiments. "
                    "Respond with ONLY a <memory_update> block containing your memory operations."
                ),
                log_dir=self.prompt_log_dir,
            )

        # Phase 2: Task execution with structured response
        # LLM context (node summary): Introduction + Research idea + Implementation + Plan + Execution output + Analysis + Metric + Plot analyses + VLM feedback + Memory.
        result = cast(
            dict,
            query(
                system_message=summary_prompt,
                user_message=None,
                func_spec={
                    "name": "summarize_experiment",
                    "description": "Summarize experimental findings",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "findings": {
                                "type": "string",
                                "description": "Key findings and results",
                            },
                            "significance": {
                                "type": "string",
                                "description": "Why these results matter",
                            },
                            "next_steps": {
                                "type": "string",
                                "description": "Suggested improvements or next experiments",
                            },
                        },
                        "required": ["findings", "significance"],
                    },
                },
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )

        # Write node summary to memory
        if self.memory_manager:
            branch_id = getattr(node, "branch_id", None)
            if branch_id:
                try:
                    findings = result.get("findings", "")
                    significance = result.get("significance", "")
                    next_steps = result.get("next_steps", "")
                    summary_text = f"Node {node.id} summary: {findings[:200]}"
                    self.memory_manager.mem_recall_append({
                        "ts": time.time(),
                        "run_id": getattr(self.memory_manager, "run_id", ""),
                        "node_id": node.id,
                        "branch_id": branch_id,
                        "phase": self.stage_name or "unknown",
                        "kind": "node_summary",
                        "summary": summary_text,
                        "refs": [],
                    })
                    # Write detailed summary to archival for important findings
                    if findings and significance:
                        self.memory_manager.mem_archival_write(
                            f"Node {node.id} Summary\n\n"
                            f"Metric: {node.metric}\n\n"
                            f"Findings: {findings}\n\n"
                            f"Significance: {significance}\n\n"
                            f"Next steps: {next_steps or 'N/A'}",
                            tags=["NODE_SUMMARY", f"node_uid:{node.id}", f"stage:{self.stage_name or 'unknown'}"],
                            meta={
                                "node_id": node.id,
                                "branch_id": branch_id,
                                "run_id": getattr(self.memory_manager, "run_id", ""),
                                "phase": self.stage_name,
                            },
                        )
                except Exception as mem_exc:
                    logger.warning("Failed to write node summary to memory: %s", mem_exc)

        return result


def _parse_cuda_visible_devices(value: Optional[str]) -> list[str]:
    if not value:
        return []
    tokens = [token.strip() for token in value.split(",")]
    return [token for token in tokens if token and token != "-1"]


class GPUManager:
    """Manages GPU allocation across processes"""

    def __init__(self, num_gpus: int, gpu_ids: Optional[List[str]] = None):
        self.num_gpus = num_gpus
        if gpu_ids:
            self.available_gpus = [str(gpu_id) for gpu_id in gpu_ids]
        else:
            self.available_gpus = [str(i) for i in range(num_gpus)]
        self.available_gpu_set: Set[str] = set(self.available_gpus)
        self.gpu_assignments: Dict[str, str] = {}  # process_id -> gpu_id

    def acquire_gpu(self, process_id: str) -> str:
        """Assigns a GPU to a process"""
        if not self.available_gpus:
            raise RuntimeError("No GPUs available")
        print(f"Available GPUs: {self.available_gpus}")
        print(f"Process ID: {process_id}")
        preferred_id = str(process_id).split("_")[-1]
        if preferred_id in self.available_gpu_set:
            gpu_id = preferred_id
        else:
            gpu_id = self.available_gpus[0]
        print(f"Acquiring GPU {gpu_id} for process {process_id}")
        self.available_gpus.remove(gpu_id)
        self.available_gpu_set.remove(gpu_id)
        self.gpu_assignments[process_id] = gpu_id
        print(f"GPU assignments: {self.gpu_assignments}")
        return gpu_id

    def release_gpu(self, process_id: str):
        """Releases GPU assigned to a process"""
        if process_id in self.gpu_assignments:
            gpu_id = self.gpu_assignments[process_id]
            if gpu_id not in self.available_gpu_set:
                self.available_gpus.append(gpu_id)
                self.available_gpu_set.add(gpu_id)
            del self.gpu_assignments[process_id]


class ParallelAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        stage_name=None,
        best_stage3_node=None,
        best_stage2_node=None,
        best_stage1_node=None,
        memory_manager: Optional[Any] = None,
        root_branch_id: Optional[str] = None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.stage_name = stage_name
        self.memory_manager = memory_manager
        self.root_branch_id = root_branch_id
        self.best_stage3_node = (
            best_stage3_node  # to initialize ablation stuides (stage 4)
        )
        self.best_stage1_node = (
            best_stage1_node  # to initialize hyperparam tuning (stage 2)
        )
        self.best_stage2_node = (
            best_stage2_node  # to initialize plotting code (stage 3)
        )
        plan = resolve_worker_plan(cfg)
        self.worker_plan = plan
        self.num_workers = plan.actual_workers
        self.num_gpus = plan.gpu_count
        logger.info(
            "Worker plan requested=%s actual=%s reasons=%s",
            plan.requested_workers,
            plan.actual_workers,
            ",".join(plan.reasons),
        )
        logger.info(
            "GPU detection torch_cuda_device_count=%s cuda_visible_devices=%s detected_gpus=%s",
            plan.torch_device_count,
            plan.cuda_visible_devices,
            plan.gpu_count,
        )
        if self.num_gpus == 0:
            logger.info("No GPUs detected; CPU workers allowed")
        else:
            logger.info("Detected %s GPUs", self.num_gpus)

        visible_gpus = _parse_cuda_visible_devices(plan.cuda_visible_devices)
        self.gpu_manager = (
            GPUManager(self.num_gpus, gpu_ids=visible_gpus) if self.num_gpus > 0 else None
        )

        if getattr(self.cfg.exec, "phase_mode", "single") == "split":
            run_root = _resolve_run_root(self.cfg)
            per_worker_sif = getattr(self.cfg.exec, "per_worker_sif", True)
            overlay_path = getattr(self.cfg.exec, "container_overlay", None)
            workspace_root = Path(self.cfg.workspace_dir)
            for idx in range(self.num_workers):
                worker_label = f"worker-{idx}" if per_worker_sif else "worker-shared"
                container_root = run_root / "workers" / worker_label / "container"
                worker_sif = container_root / f"{worker_label}.sif"
                sandbox_dir = container_root / f"{worker_label}.sandbox"
                logger.info(
                    "Split worker=%s sif=%s sandbox=%s overlay=%s workdir=%s",
                    worker_label,
                    worker_sif,
                    sandbox_dir,
                    overlay_path,
                    workspace_root,
                )
                if not per_worker_sif:
                    break

        self.timeout = self.cfg.exec.timeout
        self.worker_manager = WorkerManager(max_workers=self.num_workers)
        logger.info(
            "WorkerManager initialized max_workers=%s requested=%s",
            self.num_workers,
            self.worker_plan.requested_workers if hasattr(self, "worker_plan") else self.num_workers,
        )
        self._is_shutdown = False

        # Initialize centralized database writer process to avoid "database is locked" errors
        # This serializes all database writes from worker processes through a single writer
        self._db_writer: Optional[DatabaseWriterProcess] = None
        self._writer_queue: Optional[mp.Queue] = None
        memory_cfg = getattr(cfg, "memory", None)
        if memory_cfg and getattr(memory_cfg, "enabled", False):
            db_path = getattr(memory_cfg, "db_path", None) or (
                Path(cfg.workspace_dir) / "memory" / "memory.sqlite"
            )
            try:
                self._db_writer = DatabaseWriterProcess(db_path)
                self._db_writer.start()
                self._writer_queue = self._db_writer.queue
                logger.info("DatabaseWriterProcess started for centralized memory writes")
            except Exception as exc:
                logger.warning("Failed to start DatabaseWriterProcess: %s", exc)
                self._db_writer = None
                self._writer_queue = None
        # Define the metric once at initialization
        self.evaluation_metrics = self._define_global_metrics()
        self._ablation_state = {  # store ablation names
            "completed_ablations": set(),
        }
        self._hyperparam_tuning_state = {  # store hyperparam tuning ideas
            "tried_hyperparams": set(),
        }

    def _memory_context(self, branch_id: Optional[str], task_hint: str) -> str:
        if not self.memory_manager or not branch_id:
            return ""
        budget = getattr(getattr(self.cfg, "memory", None), "memory_budget_chars", 4000)
        return self.memory_manager.render_for_prompt(branch_id, task_hint, budget_chars=budget)

    @property
    def _is_memory_enabled(self) -> bool:
        """Check if memory management is enabled."""
        return bool(self.memory_manager and getattr(self.cfg, 'memory', None) and getattr(self.cfg.memory, 'enabled', False))

    def _run_execution_review_for_timeout(
        self, node: Node, branch_id: Optional[str]
    ) -> tuple[str, bool]:
        """Run the execution review prompt for a timeout node to get LLM analysis."""
        # Select prompt based on memory configuration
        exec_review_intro = (
            EXECUTION_REVIEW_INTRO_WITH_MEMORY
            if self._is_memory_enabled
            else EXECUTION_REVIEW_INTRO
        )
        prompt = {
            "Introduction": exec_review_intro,
            "Research idea": self.task_desc,
            "Implementation": wrap_combined_code(node.code, fallback_lang=self.code_language),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        memory_context = self._memory_context(branch_id, "execution_review")
        if memory_context:
            prompt["Memory"] = memory_context

        try:
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=review_func_spec,
                    model=self.cfg.agent.feedback.model,
                    temperature=self.cfg.agent.feedback.temp,
                ),
            )
        except Exception as exc:
            logger.error("Timeout execution review failed: %s", exc)
            return "", True

        is_bug = True
        summary = ""
        if isinstance(response, dict):
            is_bug = bool(response.get("is_bug", True))
            summary = response.get("summary") or ""
        else:
            logger.warning(
                "Timeout execution review returned non-dict response: %s", type(response)
            )

        print(
            "[red]Checking if response contains metric name and description[/red]",
            flush=True,
        )
        print(response)

        if summary:
            logger.info("Timeout execution review summary: %s", summary)
        return summary, is_bug

    @property
    def code_language(self) -> str:
        return _normalize_language(getattr(self.cfg.exec, "language", "python"))

    def _define_global_metrics(self) -> str:
        """Define eval metric to be used across all experiments"""

        prompt = {
            "Introduction": DEFINE_METRICS_INTRO,
            "Research idea": self.task_desc,
            "Instructions": list(DEFINE_METRICS_INSTRUCTIONS),
        }
        # NOTE: root branch has no memory data, so we skip memory injection for define_metrics.

        # LLM context (global metrics): Introduction + Research idea + metric-definition Instructions.
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.temp,
        )

        print(f"[green]Defined eval metrics:[/green] {response}")
        return response

    def plan_and_code_query(
        self, prompt, retries=3, code_language: Optional[str] = None
    ) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        target_language = code_language or self.code_language

        memory_read_round = 0
        max_memory_read_rounds = getattr(
            getattr(self.cfg, "memory", None), "max_memory_read_rounds", 2
        )
        max_total_calls = retries + max_memory_read_rounds + 1
        total_calls = 0

        attempt = 0
        while attempt < retries:
            attempt += 1
            total_calls += 1
            if total_calls > max_total_calls:
                logger.warning(
                    "plan_and_code_query (Parallel): reached hard call ceiling (%d). Breaking.",
                    max_total_calls,
                )
                break

            # LLM context (plan+code): system_message=prompt dict built by caller (task sections + instructions/format/env/resources as applicable).
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            # Process LLM memory updates if present
            memory_read_handled = False
            if completion_text and hasattr(self, 'memory_manager') and self.memory_manager:
                branch_id = getattr(self, 'branch_id', None)
                if branch_id:
                    memory_updates = extract_memory_updates(completion_text)
                    if memory_updates:
                        try:
                            memory_results = self.memory_manager.apply_llm_memory_updates(
                                branch_id,
                                memory_updates,
                                node_id=getattr(self, "_current_node_id", None),
                                phase="plan_and_code",
                            )
                            # Store memory operation results for prompt logging
                            self._last_memory_operation_results = memory_results

                            # Log memory operations immediately for reproducibility
                            if getattr(self, 'prompt_log_dir', None) and memory_results:
                                self._log_memory_operations_detail(
                                    memory_updates=memory_updates,
                                    memory_results=memory_results,
                                    label="plan_and_code",
                                    meta={
                                        "phase": "plan_and_code",
                                        "memory_read_round": memory_read_round,
                                        "has_read_results": _has_memory_read_results(memory_results),
                                    },
                                )

                            # If read results exist, inject them and re-query
                            if (
                                _has_memory_read_results(memory_results)
                                and memory_read_round < max_memory_read_rounds
                            ):
                                memory_read_round += 1
                                results_text = _format_memory_results_for_llm(memory_results)
                                prompt["Memory Read Results"] = (
                                    "Your memory read operations returned the following results. "
                                    "Use this information to produce your plan and code.\n\n"
                                    f"{results_text}\n\n"
                                    "Now provide your final response with:\n"
                                    "1. A <memory_update> block (can include additional writes)\n"
                                    "2. Your plan and code\n\n"
                                    "Do NOT include read operations in this response."
                                )
                                prompt.pop("Parsing Feedback", None)
                                # Do NOT consume a retry slot
                                attempt -= 1
                                memory_read_handled = True
                        except Exception as exc:
                            logger.warning("Failed to apply LLM memory updates: %s", exc)
                        # Remove memory update tags from completion text before code extraction
                        completion_text = remove_memory_update_tags(completion_text)

            if memory_read_handled:
                continue

            code = extract_code(completion_text, language=target_language)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                prompt.pop("Memory Read Results", None)
                # merge all code blocks into a single string
                return nl_text, code
            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                f"The code extraction failed. Make sure to use the format ```{target_language} ... ``` for the code blocks."
            )
        prompt.pop("Memory Read Results", None)
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text

    def _generate_seed_eval_aggregation_node(
        self, node: Node, agg_plotting_code: str
    ) -> Node:
        """Generate a special aggregation node for seed evaluation results"""
        return Node(
            plan="Aggregate results from multiple seeds",
            code="# plotting aggregation code",
            plot_code=agg_plotting_code,
            parent=node,
            is_seed_node=True,
            is_seed_agg_node=True,
        )

    def _run_multi_seed_evaluation(self, node: Node) -> List[Node]:
        """Run multiple seeds of the same node to get statistical metrics.
        Returns a list of nodes with different random seeds."""

        # IMPORTANT: Reset workers to ensure all previous tasks are complete
        # This prevents issues where workers from previous stage are still running
        logger.info("Resetting workers before multi-seed evaluation to ensure clean state...")
        print("[yellow]Resetting workers before multi-seed evaluation...[/yellow]")
        self.worker_manager.terminate_and_restart()

        # Convert node to dict for parallel processing
        node_data = node.to_dict()

        # Submit parallel jobs for different seeds
        seed_nodes = []
        task_ids = []
        task_id_to_seed: Dict[int, int] = {}  # Map task_id to seed for tracking

        for seed in range(self.cfg.agent.multi_seed_eval.num_seeds):
            gpu_id = None
            if self.gpu_manager is not None:
                try:
                    process_id = f"worker_{seed}"
                    gpu_id = self.gpu_manager.acquire_gpu(process_id)
                    logger.info(f"Assigned GPU {gpu_id} to seed {seed}")
                except RuntimeError as e:
                    logger.warning(
                        f"Could not acquire GPU for seed {seed}: {e}. Running on CPU"
                    )

            new_ablation_idea = None
            new_hyperparam_idea = None
            best_stage1_plot_code = None
            best_stage2_plot_code = None
            best_stage3_plot_code = None
            seed_eval = True
            memory_summary = ""
            print("[yellow]Starting multi-seed eval...[/yellow]")
            task_id = self.worker_manager.submit(
                self._process_node_wrapper,
                node_data,
                self.task_desc,
                self.cfg,
                gpu_id,
                memory_summary,
                self.evaluation_metrics,
                self.stage_name,
                new_ablation_idea,
                new_hyperparam_idea,
                best_stage1_plot_code,
                best_stage2_plot_code,
                best_stage3_plot_code,
                seed_eval,
                seed,
                seed,
                self.root_branch_id,
                self._writer_queue,  # Pass centralized writer queue
            )
            task_ids.append(task_id)
            task_id_to_seed[task_id] = seed

        # Wait for results with timeout
        results = self.worker_manager.wait_for_results(task_ids, timeout=self.timeout)
        completed_seeds: Set[int] = set()

        # Process completed results
        for task_id, worker_result in results.items():
            seed = task_id_to_seed[task_id]
            process_id = f"worker_{seed}"
            try:
                if worker_result.error:
                    raise worker_result.error
                result_data = worker_result.result
                result_node = Node.from_dict(result_data, self.journal)
                print(f"[seed={seed}] Parent node id: {result_node.parent.id}")
                print(f"[seed={seed}] Sanity check: actual parent node id: {node.id}")
                # Add node to journal's list and assign its step number
                self.journal.append(result_node)
                seed_nodes.append(self.journal.get_node_by_id(result_node.id))
                print(f"[seed={seed}] Added result node to journal")
            except Exception as e:
                logger.exception(f"Error in multi-seed evaluation for seed {seed}")
            finally:
                completed_seeds.add(seed)
                if (
                    self.gpu_manager is not None
                    and process_id in self.gpu_manager.gpu_assignments
                ):
                    self.gpu_manager.release_gpu(process_id)

        # Handle timed out tasks
        timed_out_task_ids = set(task_ids) - set(results.keys())
        if timed_out_task_ids:
            logger.error(f"Timeout waiting for multi-seed evaluation (timeout={self.timeout}s)")
            for task_id in timed_out_task_ids:
                seed = task_id_to_seed[task_id]
                process_id = f"worker_{seed}"
                logger.warning(f"Seed {seed} did not complete within timeout")
                if (
                    self.gpu_manager is not None
                    and process_id in self.gpu_manager.gpu_assignments
                ):
                    self.gpu_manager.release_gpu(process_id)

            # Terminate and restart workers to clean up timed out processes
            self.worker_manager.terminate_and_restart()

        return seed_nodes

    def _run_plot_aggregation(self, node: Node, seed_nodes: List[Node]) -> Node:
        """Generate an aggregation node for seed evaluation results"""
        if seed_nodes:
            try:
                from .interpreter import Interpreter

                # Create aggregation plotting code
                agg_plotting_code = self._aggregate_seed_eval_results(seed_nodes, node)

                # Create a special aggregation node
                agg_node = self._generate_seed_eval_aggregation_node(
                    node, agg_plotting_code
                )
                agg_node.parent = node

                # Execute aggregation plotting code
                print("[blue]Creating Interpreter for seed node aggregation[/blue]")
                plot_interpreter = None
                plot_agent_file_name = (
                    f"{Path(self.cfg.exec.agent_file_name).stem}_plot.py"
                )
                plot_interpreter = Interpreter(
                    working_dir=self.cfg.workspace_dir,
                    timeout=self.cfg.exec.timeout,
                    format_tb_ipython=self.cfg.exec.format_tb_ipython,
                    agent_file_name=plot_agent_file_name,
                    env_vars={"AI_SCIENTIST_ROOT": os.getenv("AI_SCIENTIST_ROOT")},
                    language="python",
                )

                try:
                    working_dir = plot_interpreter.working_dir
                    plot_exec_result = plot_interpreter.run(agg_plotting_code, True)
                    agg_node.absorb_plot_exec_result(plot_exec_result)
                    print(plot_exec_result)
                    plot_interpreter.cleanup_session()
                    # Save aggregated plots
                    plots_dir = Path(working_dir) / "working"
                    print("[red]plots_dir[/red]", plots_dir)
                    if plots_dir.exists():
                        base_dir = Path(self.cfg.workspace_dir).parent  # .parent
                        run_name = Path(self.cfg.workspace_dir).name
                        exp_results_dir = (
                            base_dir
                            / "logs"
                            / run_name
                            / "experiment_results"
                            / f"seed_aggregation_{agg_node.id}"
                        )
                        print("[red]exp_results_dir[/red]", exp_results_dir)
                        exp_results_dir.mkdir(parents=True, exist_ok=True)

                        # Save plotting code
                        with open(
                            exp_results_dir
                            / "aggregation_plotting_code.py",
                            "w",
                        ) as f:
                            f.write(agg_plotting_code)

                        # Move generated plots
                        for plot_file in plots_dir.glob("*.png"):
                            final_path = exp_results_dir / plot_file.name
                            print("mv_from:plot_file.resolve(): ", plot_file.resolve())
                            print("mv_to:final_path: ", final_path)
                            try:
                                shutil.move(str(plot_file.resolve()), str(final_path))
                            except Exception as move_exc:
                                logger.warning(f"Failed to move plot {plot_file} to {final_path}: {move_exc}")
                                try:
                                    shutil.copy2(str(plot_file.resolve()), str(final_path))
                                except Exception:
                                    logger.warning(f"Failed to copy plot {plot_file} to {final_path}, skipping")
                                    continue
                            if not final_path.exists():
                                logger.warning(f"Plot file not found after move: {final_path}")
                                continue
                            web_path = f"../../logs/{Path(self.cfg.workspace_dir).name}/experiment_results/seed_aggregation_{agg_node.id}/{plot_file.name}"
                            agg_node.plots.append(web_path)
                            agg_node.plot_paths.append(str(final_path.absolute()))

                    agg_node.is_buggy = False
                    agg_node.exp_results_dir = exp_results_dir
                    agg_node_dict = agg_node.to_dict()
                    agg_node_new = Node.from_dict(
                        agg_node_dict, self.journal
                    )  # to update the parent-child relationship in the journal
                    # Add aggregation node to journal
                    self.journal.append(agg_node_new)
                finally:
                    if plot_interpreter:
                        plot_interpreter.cleanup_session()

            except Exception as e:
                print(f"Error in seed result aggregation: {str(e)}")

    @staticmethod
    def _process_node_wrapper(
        node_data,
        task_desc,
        cfg,
        gpu_id: int = None,
        memory_summary: str = None,
        evaluation_metrics=None,
        stage_name=None,
        new_ablation_idea=None,
        new_hyperparam_idea=None,
        best_stage3_plot_code=None,
        best_stage2_plot_code=None,
        best_stage1_plot_code=None,
        seed_eval=False,
        seed: Optional[int] = None,
        worker_id: Optional[int] = None,
        explicit_root_branch_id: Optional[str] = None,
        writer_queue: Optional[mp.Queue] = None,
    ):
        """Wrapper function that creates a fresh environment for each process"""
        import os
        import multiprocessing

        # Set CUDA_VISIBLE_DEVICES BEFORE any imports that might initialize CUDA
        # This must happen before PyTorch or any CUDA library is imported
        use_gpu = bool(getattr(cfg.exec, "use_gpu", True))
        if use_gpu and gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from .interpreter import Interpreter
        from .journal import Node

        print("Starting _process_node_wrapper")

        # Debug: print received config values for memory
        memory_cfg = getattr(cfg, "memory", None)
        # Prefer explicit parameter over config value (OmegaConf pickle may lose nested values)
        config_root_branch_id = getattr(memory_cfg, "root_branch_id", None) if memory_cfg else None
        effective_root_branch_id = explicit_root_branch_id or config_root_branch_id
        if memory_cfg:
            print(f"[Worker {worker_id}] Received memory config: "
                  f"enabled={getattr(memory_cfg, 'enabled', None)}, "
                  f"db_path={getattr(memory_cfg, 'db_path', None)}, "
                  f"config_root_branch_id={config_root_branch_id}, "
                  f"explicit_root_branch_id={explicit_root_branch_id}")
        memory_manager: Optional[MemoryManager] = None
        # Use effective_root_branch_id which prefers explicit parameter over config value
        root_branch_id = effective_root_branch_id
        if memory_cfg and getattr(memory_cfg, "enabled", False):
            db_path = getattr(memory_cfg, "db_path", None) or (
                Path(cfg.workspace_dir) / "memory" / "memory.sqlite"
            )
            print(f"[Worker {worker_id}] Memory config: db_path={db_path}, root_branch_id={root_branch_id}")
            if not getattr(memory_cfg, "run_id", None):
                memory_cfg.run_id = Path(cfg.workspace_dir).name
            memory_cfg.workspace_root = str(Path(cfg.workspace_dir))
            memory_cfg.ai_scientist_root = os.environ.get("AI_SCIENTIST_ROOT")
            memory_cfg.phase_mode = getattr(cfg.exec, "phase_mode", "single")
            memory_cfg.memory_log_dir = str(Path(cfg.log_dir) / "memory")
            try:
                # Pass writer_queue to serialize all database writes through central process
                memory_manager = MemoryManager(db_path, memory_cfg, writer_queue=writer_queue)
                if writer_queue is not None:
                    print(f"[Worker {worker_id}] Using centralized database writer process")
                if not root_branch_id:
                    # If no root_branch_id was passed from main process, create one
                    # This should not happen if main process initialized correctly
                    print(f"[Worker {worker_id}] WARNING: No root_branch_id from main process, creating new root")
                    root_branch_id = uuid.uuid4().hex
                    memory_manager.mem_node_fork(None, root_branch_id)
                    memory_manager.update_branch_node_uid(root_branch_id, "root")
                    memory_manager.set_root_branch_id(root_branch_id)
                    try:
                        memory_cfg.root_branch_id = root_branch_id
                    except Exception:
                        pass
                else:
                    print(f"[Worker {worker_id}] Using root_branch_id from main process: {root_branch_id}")
                    memory_manager.set_root_branch_id(root_branch_id)
            except Exception as exc:
                print(f"[Worker {worker_id}] ERROR: Failed to initialize MemoryManager: {exc}")
                memory_manager = None

        # Create process-specific workspace
        # Use worker_id for predictable path (allows timeout recovery to find intermediate results)
        if worker_id is not None:
            workspace = os.path.join(cfg.workspace_dir, f"worker_{worker_id}")
        else:
            workspace = os.path.join(cfg.workspace_dir, f"worker_{multiprocessing.current_process().name}")
        os.makedirs(workspace, exist_ok=True)

        # Path for intermediate results (used for timeout recovery)
        intermediate_result_path = Path(workspace) / "intermediate_result.json"

        def _save_intermediate_result(child_node, stage: str = "unknown"):
            """Save intermediate result for timeout recovery."""
            try:
                intermediate_data = {
                    "plan": child_node.plan or "",
                    "code": child_node.code or "",
                    "analysis": child_node.analysis or "",
                    "exc_type": child_node.exc_type,
                    "exc_info": child_node.exc_info,
                    "is_buggy": child_node.is_buggy,
                    "metric": child_node.metric.to_dict() if child_node.metric and hasattr(child_node.metric, "to_dict") else None,
                    "stage": stage,
                    "node_id": child_node.id,
                    "branch_id": getattr(child_node, "branch_id", None),
                    "term_out": child_node._term_out if child_node._term_out else [],
                    "exec_time": child_node.exec_time,
                    "plot_plan": child_node.plot_plan,
                    "plot_code": child_node.plot_code,
                    "ablation_name": child_node.ablation_name,
                    "hyperparam_name": child_node.hyperparam_name,
                }
                intermediate_result_path.write_text(json.dumps(intermediate_data, default=str), encoding="utf-8")
                logger.info(f"Saved intermediate result at stage '{stage}' to {intermediate_result_path}")
            except Exception as exc:
                logger.warning(f"Failed to save intermediate result: {exc}")

        # Copy files from parent workspace if available (for cross-stage file inheritance)
        # Lock both current workspace (for writing) and parent workspace (for reading)
        # to prevent race conditions with other processes
        parent_workspace_path = node_data.get("workspace_path") if node_data else None
        current_lock_path = Path(workspace) / ".workspace.lock"
        with filelock.FileLock(current_lock_path, timeout=300):
            if parent_workspace_path:
                parent_workspace = Path(parent_workspace_path)
                if parent_workspace.exists() and parent_workspace != Path(workspace):
                    print(f"Copying files from parent workspace: {parent_workspace}")
                    # NOTE: No lock on parent workspace - read-only operation.
                    # Locking both current and parent workspaces caused deadlock
                    # when multiple workers tried to copy from each other's workspaces.
                    def ignore_mounted_data(directory, contents):
                        # Skip 'data' only inside 'input' directory
                        if os.path.basename(directory) == "input":
                            return ["data"] if "data" in contents else []
                        # Skip __pycache__ everywhere (auto-regenerated by Python)
                        return ["__pycache__"] if "__pycache__" in contents else []

                    def rmtree_skip_data(directory: Path):
                        """Remove directory tree but skip 'data' inside 'input' directory."""
                        for subitem in directory.iterdir():
                            if directory.name == "input" and subitem.name == "data":
                                continue  # Skip mounted data
                            if subitem.is_dir():
                                shutil.rmtree(subitem)
                            else:
                                subitem.unlink()

                    def copy_ignore_missing(src, dst, *, follow_symlinks=True):
                        """Copy file, silently ignoring if source disappears (race condition)."""
                        try:
                            return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
                        except FileNotFoundError:
                            # Source file was deleted by another process during copy
                            return dst

                    # Skip directories that are race-condition prone or regenerable
                    # .pydeps: pip packages (will be reinstalled by each worker if needed)
                    # __pycache__: bytecode cache (auto-regenerated by Python)
                    skip_top_level = {".workspace.lock", ".pydeps", "__pycache__"}

                    for item in parent_workspace.iterdir():
                        if item.name in skip_top_level:
                            continue
                        src = parent_workspace / item.name
                        dst = Path(workspace) / item.name
                        try:
                            if src.is_dir():
                                if dst.exists():
                                    if item.name == "input":
                                        # Don't rmtree the whole input dir, as data might be mounted
                                        rmtree_skip_data(dst)
                                    else:
                                        shutil.rmtree(dst)
                                shutil.copytree(src, dst, ignore=ignore_mounted_data, dirs_exist_ok=True, copy_function=copy_ignore_missing)
                            else:
                                copy_ignore_missing(src, dst)
                        except FileNotFoundError:
                            # Source was deleted by another process during copy (race condition)
                            pass
                        except OSError as exc:
                            # EBUSY on 'data' inside 'input' is expected (mounted/symlinked)
                            # ENOENT (errno 2) is also expected in race conditions
                            if exc.errno == 16 and item.name == "input" and getattr(exc, 'filename', None) == 'data':
                                pass  # Silently ignore - data is intentionally skipped
                            elif exc.errno == 2:
                                pass  # Silently ignore - file/dir deleted by another process
                            else:
                                logger.warning(f"Failed to copy {src} to {dst}: {exc}")
                        except Exception as exc:
                            logger.warning(f"Failed to copy {src} to {dst}: {exc}")

                    # Verify and re-copy critical directories (src and its subdirectories)
                    # This ensures source code is properly inherited across stages
                    parent_src = parent_workspace / "src"
                    dst_src = Path(workspace) / "src"
                    if parent_src.exists():
                        # Ensure src directory exists
                        dst_src.mkdir(parents=True, exist_ok=True)
                        # Check each subdirectory in parent's src
                        for subdir in parent_src.iterdir():
                            if subdir.is_dir():
                                dst_subdir = dst_src / subdir.name
                                if not dst_subdir.exists():
                                    print(f"Re-copying missing src subdirectory: {subdir.name}")
                                    try:
                                        shutil.copytree(subdir, dst_subdir, dirs_exist_ok=True)
                                    except Exception as subdir_exc:
                                        logger.error(f"Failed to copy src/{subdir.name}: {subdir_exc}")

        worker_name = f"worker_{worker_id}" if worker_id is not None else f"worker_{multiprocessing.current_process().name}"
        print(f"Process {worker_name} using workspace: {workspace}")
        # Create process-specific working directory
        working_dir = os.path.join(workspace, "working")
        os.makedirs(working_dir, exist_ok=True)
        workspace_path = Path(workspace)
        if memory_manager:
            memory_manager.workspace_root = workspace_path
        phase_log_dir: Optional[Path] = None
        prompt_session_id = f"{int(time.time() * 1000)}_{os.getpid()}"
        prompt_log_root: Optional[Path] = None
        prompt_session_dir: Optional[Path] = None

        def resolve_workdir(requested: Optional[str]) -> Path:
            expected_root = getattr(cfg.exec, "workspace_mount", "/workspace")
            if not requested:
                return workspace_path
            requested_path = str(requested)
            if requested_path.startswith("/"):
                if requested_path.startswith(expected_root):
                    rel = requested_path[len(expected_root) :].lstrip("/")
                    candidate = (workspace_path / rel).resolve()
                else:
                    candidate = workspace_path
            else:
                candidate = (workspace_path / requested_path).resolve()
            try:
                candidate.relative_to(workspace_path)
                return candidate
            except ValueError:
                return workspace_path

        def run_commands_with_logging(env, commands, cwd: Path, log_path: Path, phase_name: str, extra_env=None):
            outputs: list[str] = []
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as log:
                if not commands:
                    log.write(f"No commands provided for {phase_name}\n")
                    return True, outputs, None
                for raw_cmd in commands:
                    if isinstance(raw_cmd, list):
                        printable_cmd = " ".join(raw_cmd)
                        cmd_to_run = list(raw_cmd)
                        if isinstance(env, ExecutionEnvironment) and cmd_to_run:
                            if cmd_to_run[0] in {"python3", "python"}:
                                cmd_to_run[0] = "/usr/bin/python3"
                            elif (
                                len(cmd_to_run) >= 3
                                and cmd_to_run[0] == "bash"
                                and cmd_to_run[1] == "-lc"
                            ):
                                cmd_to_run[2] = re.sub(
                                    r"(^|\s)(python3|python)\s",
                                    r"\1/usr/bin/python3 ",
                                    cmd_to_run[2],
                                )
                    else:
                        printable_cmd = str(raw_cmd)
                        cmd_to_run = printable_cmd
                        if isinstance(env, ExecutionEnvironment):
                            cmd_to_run = re.sub(
                                r"(^|\\s)(python3|python)\\s",
                                r"\\1/usr/bin/python3 ",
                                cmd_to_run,
                            )
                    log.write(f"$ {printable_cmd}\n")
                    if env is None:
                        log.write("ERROR: No container execution environment available.\n")
                        return False, outputs, {"message": "Container execution environment is required."}
                    result = env.run(cmd_to_run, cwd=cwd, extra_env=extra_env)
                    log.write(f"exit_code={result.returncode}\n")
                    if result.stdout:
                        log.write(result.stdout)
                    if result.stderr:
                        log.write(result.stderr)
                    summary = summarize_command_output(result.stdout, result.stderr)
                    outputs.append(
                        f"[{phase_name}] $ {printable_cmd}\n"
                        f"exit_code={result.returncode}\n"
                        f"stdout:\n{summary['stdout']}\n"
                        f"stderr:\n{summary['stderr']}\n"
                    )
                    if result.returncode != 0:
                        log.flush()
                        return False, outputs, result
            return True, outputs, None

        # Log GPU assignment (CUDA_VISIBLE_DEVICES was already set at the top of this function)
        if gpu_id is not None:
            logger.info(f"Process {worker_name} assigned to GPU {gpu_id}")
        else:
            cpu_note = "running on CPU"
            if not bool(getattr(cfg.exec, "use_gpu", True)):
                cpu_note += " (GPU disabled)"
            logger.info(f"Process {worker_name} {cpu_note}")

        environment_context: dict[str, Any] = {}
        exec_env: Optional[ExecutionEnvironment] = None
        active_env: Optional[ExecutionEnvironment] = None
        if getattr(cfg.exec, "phase_mode", "single") == "split":
            image_path = getattr(cfg.exec, "singularity_image", None)
            environment_context = {
                "available_compilers": [],
                "available_libs": [],
                "container_runtime": "singularity" if image_path else None,
                "singularity_image": str(Path(image_path).resolve()) if image_path else None,
                "workspace_mount": getattr(cfg.exec, "workspace_mount", "/workspace"),
                "assigned_gpu_id": gpu_id if use_gpu else None,
            }
            if image_path:
                info_env = ExecutionEnvironment(
                    workspace=workspace_path,
                    image=image_path,
                    runtime_preference="singularity",
                    workspace_mount=getattr(cfg.exec, "workspace_mount", "/workspace"),
                    gpu_id=gpu_id if use_gpu else None,
                    enable_gpu=use_gpu,
                    enable_writable_tmpfs=False,
                    overlay_path=None,
                    extra_start_args=None,
                )
                try:
                    environment_context["available_compilers"] = collect_available_compilers(info_env)
                except Exception as exc:
                    logger.warning("Failed to collect compilers: %s", exc)
                try:
                    environment_context["available_libs"] = collect_available_libs(info_env)
                except Exception as exc:
                    logger.warning("Failed to collect libs: %s", exc)
                try:
                    environment_context["system_performance_tools"] = collect_system_performance_tools(info_env)
                except Exception as exc:
                    logger.warning("Failed to collect system perf tools: %s", exc)
                try:
                    environment_context["installed_system_packages"] = collect_installed_system_packages(info_env)
                except Exception as exc:
                    logger.warning("Failed to collect installed system packages: %s", exc)
                try:
                    os_release_res = info_env.run(["bash", "-lc", "cat /etc/os-release"], cwd=workspace_path)
                    environment_context["os_release"] = summarize_text(os_release_res.stdout, max_lines=20, max_chars=1200)
                except Exception as exc:
                    logger.warning("Failed to read OS release: %s", exc)
                try:
                    cpu_res = info_env.run(["bash", "-lc", "lscpu"], cwd=workspace_path)
                    environment_context["cpu_info"] = summarize_text(cpu_res.stdout, max_lines=60, max_chars=20000)
                except Exception as exc:
                    logger.warning("Failed to read CPU info: %s", exc)
                try:
                    mem_res = info_env.run(["bash", "-lc", "free -h && echo '---' && cat /proc/meminfo | head -n 20"], cwd=workspace_path)
                    environment_context["memory_info"] = summarize_text(mem_res.stdout, max_lines=40, max_chars=2000)
                except Exception as exc:
                    logger.warning("Failed to read memory info: %s", exc)
                if use_gpu:
                    try:
                        # Show assigned GPU for this worker (respects CUDA_VISIBLE_DEVICES)
                        # Also show all available GPUs for reference
                        assigned_gpu_str = f"assigned_gpu_id={gpu_id}" if gpu_id is not None else "assigned_gpu_id=all"
                        cuda_visible = f"CUDA_VISIBLE_DEVICES={gpu_id}" if gpu_id is not None else "CUDA_VISIBLE_DEVICES=all"
                        gpu_res = info_env.run(
                            ["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo 'nvidia-smi not available'"],
                            cwd=workspace_path,
                        )
                        all_gpus = summarize_text(gpu_res.stdout, max_lines=20, max_chars=1000)
                        environment_context["gpu_info"] = f"{assigned_gpu_str}\n{cuda_visible}\nAll GPUs on node:\n{all_gpus}"
                    except Exception as exc:
                        logger.warning("Failed to read GPU info: %s", exc)
                else:
                    environment_context["gpu_info"] = "disabled by config"
                try:
                    net_res = info_env.run(
                        ["bash", "-lc", "command -v getent >/dev/null 2>&1 && getent hosts github.com >/dev/null 2>&1 && echo ok || echo fail"],
                        cwd=workspace_path,
                    )
                    environment_context["network_access"] = "available" if "ok" in (net_res.stdout or "") else "blocked"
                except Exception as exc:
                    logger.warning("Failed to probe network access: %s", exc)
                    environment_context["network_access"] = "unknown"
                # CPU governor
                try:
                    governor_res = info_env.run(
                        ["bash", "-lc", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'NA'"],
                        cwd=workspace_path,
                    )
                    environment_context["cpu_governor"] = (governor_res.stdout or "NA").strip()
                except Exception as exc:
                    logger.warning("Failed to read CPU governor: %s", exc)
                    environment_context["cpu_governor"] = "NA"
                # NUMA configuration (NA if numactl not available)
                try:
                    numa_res = info_env.run(
                        ["bash", "-lc", "command -v numactl >/dev/null 2>&1 && numactl --hardware 2>/dev/null || echo 'NA'"],
                        cwd=workspace_path,
                    )
                    environment_context["numa_config"] = summarize_text(numa_res.stdout or "NA", max_lines=30, max_chars=2000)
                except Exception as exc:
                    logger.warning("Failed to read NUMA config: %s", exc)
                    environment_context["numa_config"] = "NA"
                # All environment variables
                try:
                    all_env_res = info_env.run(["bash", "-lc", "env | sort"], cwd=workspace_path)
                    environment_context["all_env_vars"] = summarize_text(all_env_res.stdout or "none", max_lines=200, max_chars=10000)
                except Exception as exc:
                    logger.warning("Failed to read all environment variables: %s", exc)
                    environment_context["all_env_vars"] = "NA"
                # Container digest (SHA256 of SIF image)
                if image_path:
                    try:
                        digest_res = info_env.run(
                            ["bash", "-lc", f"sha256sum {image_path} 2>/dev/null | cut -d' ' -f1 || echo 'NA'"],
                            cwd=workspace_path,
                        )
                        digest_value = (digest_res.stdout or "NA").strip()
                        if digest_value and len(digest_value) == 64:
                            environment_context["container_digest"] = f"sha256:{digest_value}"
                        else:
                            environment_context["container_digest"] = "NA"
                    except Exception as exc:
                        logger.warning("Failed to compute container digest: %s", exc)
                        environment_context["container_digest"] = "NA"
            environment_context["timeout_seconds"] = cfg.exec.timeout
        else:
            environment_context["timeout_seconds"] = cfg.exec.timeout

        run_root: Optional[Path] = None
        phase0_plan: Optional[dict] = None
        phase0_history_summary: Optional[dict] = None
        phase0_artifact_paths: Optional[List[str]] = None
        phase0_command_str: Optional[str] = None
        phase0_prompt: Optional[Dict[str, Any]] = None
        plans_dir: Optional[Path] = None
        worker_label = f"worker-{worker_id if worker_id is not None else 0}"
        if getattr(cfg.exec, "phase_mode", "single") == "split":
            run_root = _resolve_run_root(cfg)
            plans_dir = run_root / "workers" / worker_label / "plans"
            plans_dir.mkdir(parents=True, exist_ok=True)
            phase0_plan_path = plans_dir / "phase0_plan.json"
            phase0_history_path = plans_dir / "phase0_history_full.json"
            if getattr(cfg.exec, "log_prompts", True):
                prompt_log_root = run_root / "workers" / worker_label / "prompt_logs"
                prompt_session_dir = prompt_log_root / prompt_session_id

            history_summary, history_full = _collect_phase0_history(
                current_log_dir=Path(cfg.log_dir),
                worker_label=worker_label,
            )
            history_full["environment_context"] = environment_context
            phase0_history_path.write_text(json.dumps(history_full, indent=2), encoding="utf-8")

            # Build verified system perf tools note for history injection
            performance_tools_for_history = environment_context.get("system_performance_tools", [])
            performance_tools_names_for_history = [t.get("name", "") for t in performance_tools_for_history if isinstance(t, dict) and t.get("name")]
            performance_tools_note_for_history = (
                f"Verified and functional: {', '.join(performance_tools_names_for_history)}. Use directly without assuming availability issues."
            ) if performance_tools_names_for_history else "No system performance tools detected."
            history_injection = format_prompt(
                "config/environment/history_injection",
                phase_summaries=json.dumps(history_summary.get("phase_summaries", {}), indent=2),
                phase1_steps_summary=json.dumps(history_summary.get("phase1_steps_summary", {}), indent=2),
                compile_log_summary=history_summary.get("compile_log_summary", ""),
                compile_errors=json.dumps(history_summary.get("compile_errors", []), indent=2),
                run_log_summary=history_summary.get("run_log_summary", ""),
                run_errors=json.dumps(history_summary.get("run_errors", []), indent=2),
                llm_output_summary=json.dumps(history_summary.get("llm_output_summary", {}), indent=2),
                environment_info=json.dumps(
                    {
                        "os_release": environment_context.get("os_release", ""),
                        "cpu_info": environment_context.get("cpu_info", ""),
                        "cpu_governor": environment_context.get("cpu_governor", "NA"),
                        "numa_config": environment_context.get("numa_config", "NA"),
                        "memory_info": environment_context.get("memory_info", ""),
                        "assigned_gpu_id": environment_context.get("assigned_gpu_id"),
                        "gpu_info": environment_context.get("gpu_info", ""),
                        "all_env_vars": environment_context.get("all_env_vars", "NA"),
                        "available_compilers": environment_context.get("available_compilers", []),
                        "available_libs": environment_context.get("available_libs", []),
                        "system_performance_tools": environment_context.get("system_performance_tools", []),
                        "system_performance_tools_note": performance_tools_note_for_history,
                        "installed_system_packages": environment_context.get("installed_system_packages", []),
                        "network_access": environment_context.get("network_access", "unknown"),
                        "container_runtime": environment_context.get("container_runtime"),
                        "container_digest": environment_context.get("container_digest", "NA"),
                        "singularity_image": environment_context.get("singularity_image"),
                        "workspace_mount": environment_context.get("workspace_mount"),
                        "timeout_seconds": environment_context.get("timeout_seconds", cfg.exec.timeout),
                    },
                    indent=2,
                ),
                history_full_path=str(phase0_history_path),
            )

            # Phase 0 is always generated for each node (no caching)
            # Build verified system perf tools note
            performance_tools_list = environment_context.get("system_performance_tools", [])
            performance_tools_names = [t.get("name", "") for t in performance_tools_list if isinstance(t, dict) and t.get("name")]
            performance_tools_note = (
                f"The following system performance tools have been verified and are confirmed functional inside this container: {', '.join(performance_tools_names)}. "
                "Use them directly without assuming availability issues."
            ) if performance_tools_names else "No system performance tools detected in this environment."
            phase0_intro = PHASE0_WHOLE_PLANNING_PROMPT
            if memory_cfg and getattr(memory_cfg, "enabled", False):
                phase0_intro = PHASE0_WHOLE_PLANNING_PROMPT_WITH_MEMORY
            phase0_prompt: dict[str, Any] = {
                "Introduction": phase0_intro,
                "Task": task_desc,
                "History": history_injection,
                "Environment": {
                    "os_release": environment_context.get("os_release", ""),
                    "cpu_info": environment_context.get("cpu_info", ""),
                    "memory_info": environment_context.get("memory_info", ""),
                    "assigned_gpu_id": environment_context.get("assigned_gpu_id"),
                    "gpu_info": environment_context.get("gpu_info", ""),
                    "cpu_governor": environment_context.get("cpu_governor", "NA"),
                    "numa_config": environment_context.get("numa_config", "NA"),
                    "all_env_vars": environment_context.get("all_env_vars", "NA"),
                    "available_compilers": environment_context.get("available_compilers", []),
                    "available_libs": environment_context.get("available_libs", []),
                    "system_performance_tools": environment_context.get("system_performance_tools", []),
                    "system_performance_tools_note": performance_tools_note,
                    "installed_system_packages": environment_context.get("installed_system_packages", []),
                    "network_access": environment_context.get("network_access", "unknown"),
                    "container_runtime": environment_context.get("container_runtime"),
                    "container_digest": environment_context.get("container_digest", "NA"),
                    "singularity_image": environment_context.get("singularity_image"),
                    "workspace_mount": environment_context.get("workspace_mount", "/workspace"),
                    "timeout_seconds": environment_context.get("timeout_seconds", cfg.exec.timeout),
                },
            }
            # NOTE: Memory injection for Phase 0 is deferred to _execute_phase0_planning(),
            # which is called after fork. This ensures memory context from child branches is available.
            resources_path = getattr(cfg.exec, "resources", None)
            if resources_path:
                try:
                    resources_cfg = load_resources(resolve_resources_path(resources_path))
                    phase0_prompt["Resources"] = build_resources_context(resources_cfg, phase="phase0")
                except Exception as exc:
                    phase0_prompt["Resources"] = {"error": str(exc)}
            if prompt_log_root:
                _write_prompt_log(
                    prompt_log_root,
                    _format_prompt_log_name("phase0_prompt"),
                    phase0_prompt,
                    meta={
                        "phase": "phase0",
                        "model": cfg.agent.code.model,
                    },
                )
            # NOTE: Phase 0 LLM call is deferred until after fork.
            # This ensures Phase 0 is executed in each child node's context,
            # and memory events are recorded on the correct branch.
            # The phase0_prompt is saved and passed to _execute_phase0_planning after fork.
            phase0_history_summary = history_summary

        if prompt_log_root is None and getattr(cfg.exec, "log_prompts", True):
            prompt_log_root = Path(cfg.log_dir) / "prompt_logs" / worker_label
            prompt_session_dir = prompt_log_root / prompt_session_id

        # Create minimal agent for worker process with the global metric definition
        worker_agent = MinimalAgent(
            task_desc=task_desc,
            cfg=cfg,
            memory_summary=memory_summary,
            evaluation_metrics=evaluation_metrics,
            stage_name=stage_name,
            environment_context=environment_context,
            phase0_plan=phase0_plan,
            phase0_history=phase0_history_summary,
            prompt_log_dir=prompt_session_dir,
            prompt_session_id=prompt_session_id,
            memory_manager=memory_manager,
            branch_id=None,
        )

        process_interpreter: Optional[Interpreter] = None
        if cfg.exec.phase_mode == "single":
            print("Creating Interpreter")
            process_interpreter = Interpreter(
                working_dir=workspace,
                timeout=cfg.exec.timeout,
                format_tb_ipython=cfg.exec.format_tb_ipython,
                agent_file_name=cfg.exec.agent_file_name,
                language=cfg.exec.language,
            )
        plot_interpreter: Optional[Interpreter] = None
        plot_agent_file_name = f"{Path(cfg.exec.agent_file_name).stem}_plot.py"
        plot_interpreter = Interpreter(
            working_dir=workspace,
            timeout=cfg.exec.timeout,
            format_tb_ipython=cfg.exec.format_tb_ipython,
            agent_file_name=plot_agent_file_name,
            language="python",
        )
        parse_interpreter: Optional[Interpreter] = None
        parse_agent_file_name = f"{Path(cfg.exec.agent_file_name).stem}_parse_metrics.py"
        parse_interpreter = Interpreter(
            working_dir=workspace,
            timeout=cfg.exec.timeout,
            format_tb_ipython=cfg.exec.format_tb_ipython,
            agent_file_name=parse_agent_file_name,
            language="python",
        )

        try:
            print(f"stage_name: {stage_name}")
            # Recreate node object from node_data, which becomes a parent node.
            if node_data:
                parent_node = Node.from_dict(node_data, journal=None)
                print(f"Recreated parent node: {parent_node.id}")
            else:
                parent_node = None
                print("No parent node to recreate")

            child_branch_id = None
            if memory_manager:
                # Try to get parent branch_id from parent node
                # Fall back to parent node's id if branch_id is not set
                parent_branch_id = (
                    getattr(parent_node, "branch_id", None) if parent_node else None
                )
                if not parent_branch_id and parent_node:
                    # Use parent node's id as branch_id if branch_id is not explicitly set
                    parent_branch_id = parent_node.id
                if not parent_branch_id:
                    parent_branch_id = root_branch_id
                if not parent_branch_id:
                    parent_branch_id = uuid.uuid4().hex
                    memory_manager.mem_node_fork(None, parent_branch_id)
                    memory_manager.update_branch_node_uid(parent_branch_id, "root")
                    memory_manager.set_root_branch_id(parent_branch_id)
                child_branch_id = uuid.uuid4().hex

                # Build ancestor chain from parent node for correct tree structure
                # This ensures that if intermediate nodes are missing in the DB,
                # they are created with the correct parent-child relationships
                ancestor_chain = []
                if parent_node:
                    current = parent_node
                    while current is not None:
                        node_id = getattr(current, "branch_id", None) or current.id
                        ancestor_chain.append(node_id)
                        current = getattr(current, "parent", None)
                    # Reverse to get root-to-parent order
                    ancestor_chain = ancestor_chain[::-1]

                memory_manager.mem_node_fork(parent_branch_id, child_branch_id, ancestor_chain=ancestor_chain)
                worker_agent.branch_id = child_branch_id

            # Execute Phase 0 planning AFTER fork so it runs in each child node's context
            # This ensures Phase 0 memory events are recorded on the correct child branch
            if phase0_prompt and getattr(cfg.exec, "phase_mode", "single") == "split":
                phase0_plan, _, _ = _execute_phase0_planning(
                    phase0_prompt=phase0_prompt,
                    cfg=cfg,
                    memory_cfg=memory_cfg,
                    memory_manager=memory_manager,
                    branch_id=child_branch_id if child_branch_id else root_branch_id,
                    plans_dir=plans_dir,
                    prompt_log_root=prompt_log_root,
                )
                # Update worker agent with the Phase 0 plan
                worker_agent.phase0_plan = phase0_plan

            # Process the node using worker agent
            print("Starting node processing")
            if seed_eval:
                # Use the parent node's code to run the same code again
                child_node = worker_agent._generate_seed_node(parent_node)
                child_node.parent = parent_node
                # Plot code should also be the same as the parent node
                child_node.plot_code = parent_node.plot_code
                if seed is not None:
                    child_node.seed_value = seed
                    worker_agent._inject_seed_with_llm(child_node, seed)
            else:
                if parent_node is None:
                    print("Drafting new node")
                    child_node = worker_agent._draft()
                elif parent_node.is_buggy:
                    print("Debugging node with id: ", parent_node.id)
                    child_node = worker_agent._debug(parent_node)
                    child_node.parent = parent_node
                else:
                    if (
                        new_hyperparam_idea is not None and new_ablation_idea is None
                    ):  # stage 2
                        child_node = worker_agent._generate_hyperparam_tuning_node(
                            parent_node, new_hyperparam_idea
                        )
                        child_node.parent = parent_node
                        logger.info(
                            f"Processing hyperparam tuning: {child_node.hyperparam_name}"
                        )
                        print(
                            f"[cyan]Running hyperparam tuning: {child_node.hyperparam_name}[/cyan]"
                        )
                    elif (
                        new_ablation_idea is not None and new_hyperparam_idea is None
                    ):  # stage 4
                        child_node = worker_agent._generate_ablation_node(
                            parent_node, new_ablation_idea
                        )
                        child_node.parent = parent_node
                        logger.info(f"Processing ablation: {child_node.ablation_name}")
                        print(
                            f"[cyan]Running ablation study: {child_node.ablation_name}[/cyan]"
                        )
                    else:
                        print("Improving node with id: ", parent_node.id)
                        child_node = worker_agent._improve(parent_node)
                        child_node.parent = parent_node

            if child_branch_id:
                child_node.branch_id = child_branch_id
                try:
                    memory_manager.update_branch_node_uid(child_branch_id, child_node.id)
                except Exception:
                    pass

            # Save intermediate result after node creation (for timeout recovery)
            _save_intermediate_result(child_node, stage="after_node_creation")

            # Note: idea.md and Phase 0 info are no longer auto-ingested.
            # LLM manages core memory directly.

            if memory_manager and child_branch_id:
                try:
                    plan_snippet = (child_node.plan or "")[:1200]
                    memory_manager.mem_recall_append(
                        {
                            "ts": time.time(),
                            "run_id": memory_manager.run_id,
                            "node_id": child_node.id,
                            "branch_id": child_branch_id,
                            "phase": stage_name,
                            "kind": "node_created",
                            "summary": f"stage={stage_name} node_id={child_node.id}\n{plan_snippet}",
                            "refs": [],
                        }
                    )
                except Exception as exc:
                    logger.warning("Failed to write node_created event: %s", exc)

            # Execute and parse results
            print("Running code")
            exec_result: ExecutionResult
            if cfg.exec.phase_mode == "split":
                phase_log_dir = Path(cfg.log_dir) / "phase_logs" / f"node_{child_node.id}"
                phase_log_dir.mkdir(parents=True, exist_ok=True)
                if prompt_session_dir and prompt_session_dir.exists():
                    prompt_dest = phase_log_dir / "prompt_logs"
                    for prompt_file in prompt_session_dir.iterdir():
                        if prompt_file.is_file():
                            _copy_artifact(prompt_file, prompt_dest)
                raw_phase_artifacts = getattr(child_node, "phase_artifacts", None)
                phase_data = None
                if isinstance(raw_phase_artifacts, dict):
                    phase_data = raw_phase_artifacts.get("phase_artifacts") or raw_phase_artifacts
                elif isinstance(raw_phase_artifacts, list) and raw_phase_artifacts:
                    first = raw_phase_artifacts[0]
                    if isinstance(first, dict):
                        phase_data = first.get("phase_artifacts") or first
                if isinstance(phase_data, list) and phase_data:
                    phase_data = phase_data[0]
                if not isinstance(phase_data, dict):
                    exec_result = ExecutionResult(
                        ["Missing phase_artifacts for split execution"],
                        0.0,
                        "PhasePlanError",
                        {"message": "phase_artifacts missing"},
                        None,
                    )
                else:
                    exec_start = time.time()
                    term_outputs: list[str] = []
                    exc_type = None
                    exc_info = None
                    if run_root is None:
                        run_root = _resolve_run_root(cfg)
                    
                    # Load resources configuration if specified
                    resources_config: Optional[ResourceConfig] = None
                    resource_binds: list[str] = []
                    resources_path = getattr(cfg.exec, "resources", None)
                    if resources_path:
                        try:
                            resources_config = load_resources(resolve_resources_path(resources_path))
                            resource_binds = build_local_binds(resources_config)
                            term_outputs.append(f"Loaded resources from {resources_path}")
                        except Exception as exc:
                            term_outputs.append(f"Warning: Failed to load resources: {exc}")
                    
                    worker_container: Optional[SingularityWorkerContainer] = None
                    if getattr(cfg.exec, "singularity_image", None):
                        worker_container = SingularityWorkerContainer(
                            base_image=getattr(cfg.exec, "singularity_image", None),
                            run_root=run_root,
                            workspace=workspace_path,
                            workspace_mount=getattr(cfg.exec, "workspace_mount", "/workspace"),
                            worker_id=worker_id if worker_id is not None else 0,
                            per_worker_sif=getattr(cfg.exec, "per_worker_sif", True),
                            keep_sandbox=getattr(cfg.exec, "keep_sandbox", False),
                            use_fakeroot=getattr(cfg.exec, "use_fakeroot", True),
                            writable_mode=getattr(cfg.exec, "writable_mode", "auto"),
                            enable_writable_tmpfs=getattr(cfg.exec, "writable_tmpfs", True),
                            overlay_path=getattr(cfg.exec, "container_overlay", None),
                            resource_binds=resource_binds,
                            enable_gpu=use_gpu,
                        )
                        logger.info(
                            "Worker container ready worker=%s root=%s sif=%s overlay=%s workdir=%s",
                            worker_label,
                            worker_container.container_root,
                            worker_container.worker_sif,
                            worker_container.overlay_path,
                            workspace_path,
                        )
                    active_env: Optional[ExecutionEnvironment] = None
                    def norm_section(val, default):
                        if isinstance(val, list) and val:
                            val = val[0]
                        return val if isinstance(val, dict) else default

                    download_section = norm_section(phase_data.get("download", {}), {"commands": []})
                    coding_section = norm_section(phase_data.get("coding", {}), {"workspace": {}})
                    compile_section = norm_section(phase_data.get("compile", {}), {"build_plan": {}})
                    run_section = norm_section(phase_data.get("run", {}), {"commands": []})

                    build_plan = compile_section.get("build_plan", {})
                    download_commands = download_section.get("commands", [])
                    coding_workspace = coding_section.get("workspace", {})
                    compile_commands = compile_section.get("commands", [])
                    run_commands = run_section.get("commands", [])
                    # Use dynamic output filename based on experiment name
                    default_output_path = f"working/{_get_experiment_output_filename(Path(cfg.workspace_dir).name)}"
                    expected_outputs = run_section.get("expected_outputs") or [default_output_path]
                    expected_output_paths = [
                        (workspace_path / Path(rel_path)) for rel_path in expected_outputs
                    ]
                    for path in expected_output_paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    available_compiler_names = [
                        c.get("name")
                        for c in environment_context.get("available_compilers", [])
                        if isinstance(c, dict) and c.get("name")
                    ]
                    term_outputs.append(f"available_compilers: {available_compiler_names}")
                    selected_compiler = build_plan.get("compiler_selected")
                    phase1_max_steps = int(getattr(cfg.exec, "phase1_max_steps", 12))

                    def parse_phase1_iterative_response(raw_text: str) -> dict[str, Any]:
                        cleaned = raw_text.strip()

                        # Check for malformed memory_update blocks and raise error to trigger regeneration
                        if check_malformed_memory_update(cleaned):
                            logger.error("[Phase1 Parse Error] Malformed <memory_update> block detected, triggering regeneration")
                            logger.error("[Phase1 Parse Error] Raw text (first 1000 chars): %s", repr(cleaned[:1000]))
                            raise MalformedMemoryUpdateError(f"Malformed <memory_update> block detected: {repr(cleaned[:500])}")

                        # Remove <memory_update>...</memory_update> blocks before parsing
                        # This is critical when memory is enabled, as the LLM includes memory updates before JSON
                        cleaned = remove_memory_update_tags(cleaned)

                        # Try to extract JSON object from anywhere in the response
                        # This handles cases where LLM adds text before/after JSON
                        start = cleaned.find("{")
                        end = cleaned.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            json_candidate = cleaned[start : end + 1]
                        else:
                            json_candidate = cleaned

                        try:
                            parsed = json.loads(json_candidate)
                        except json.JSONDecodeError as json_err:
                            # If JSON parsing failed, try the original cleaned text too
                            try:
                                parsed = json.loads(cleaned)
                            except json.JSONDecodeError:
                                try:
                                    import ast
                                    parsed = ast.literal_eval(json_candidate)
                                except Exception as ast_err:
                                    # Log detailed error info
                                    logger.error("[Phase1 Parse Error] JSON error: %s", json_err)
                                    logger.error("[Phase1 Parse Error] AST error: %s", ast_err)
                                    logger.error("[Phase1 Parse Error] Raw text (first 1000 chars): %s", repr(raw_text[:1000]))
                                    logger.error("[Phase1 Parse Error] JSON candidate (first 500 chars): %s", repr(json_candidate[:500]))
                                    raise ValueError(f"Failed to parse Phase 1 response: {ast_err}\nRaw response (first 500 chars): {repr(raw_text[:500])}") from ast_err
                        if not isinstance(parsed, dict):
                            raise ValueError("Phase 1 response must be a JSON object.")
                        return parsed

                    phase1_llm_log_path: Optional[Path] = None

                    def phase1_iterative_driver(history: list[dict[str, Any]], step_idx: int, max_steps: int) -> dict[str, Any]:
                        # Select prompt based on memory configuration
                        phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT
                        if memory_cfg and getattr(memory_cfg, "enabled", False):
                            phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT_WITH_MEMORY
                        prompt: dict[str, Any] = {
                            "Introduction": phase1_intro,
                            "Task": task_desc,
                            "Phase plan": {
                                "download_commands_seed": download_commands,
                                "compile_plan": build_plan,
                                "compile_commands": compile_commands,
                                "run_commands": run_commands,
                            },
                            "Constraints": phase_data.get("constraints", {}),
                            "Progress": {
                                "step": step_idx,
                                "max_steps": max_steps,
                                "history": history,
                            },
                        }
                        phase0_snippet = worker_agent._phase0_plan_snippet(
                            include_phase1=True, include_phase2_4=False
                        )
                        if phase0_snippet:
                            prompt["Phase 0 plan"] = phase0_snippet
                            phase_guidance = phase0_snippet.get("phase_guidance", {}).get("phase1", {})
                            if phase_guidance:
                                prompt["Phase 0 guidance for Phase 1"] = {
                                    "targets": phase_guidance.get("targets", []),
                                    "preferred_commands": phase_guidance.get("preferred_commands", []),
                                    "done_conditions": phase_guidance.get("done_conditions", []),
                                }
                        env_block = worker_agent._prompt_environment
                        if env_block:
                            prompt["Environment"] = env_block.get("Environment injection", env_block)
                        
                        # Inject resources context if available
                        if resources_config and resources_config.has_resources():
                            resources_ctx = build_resources_context(resources_config, phase="phase1")
                            prompt["Resources"] = resources_ctx

                        worker_agent._inject_memory(prompt, "phase1_iterative")

                        if prompt_session_dir:
                            _write_prompt_log(
                                prompt_session_dir,
                                _format_prompt_log_name(f"phase1_step_{step_idx}"),
                                prompt,
                                meta={
                                    "phase": "phase1",
                                    "step": step_idx,
                                    "max_steps": max_steps,
                                    "model": cfg.agent.code.model,
                                },
                            )

                        # LLM context (Phase 1 iterative install): Introduction + Task + Phase plan + Constraints + Progress history + optional Phase 0 guidance + Environment injection + Resources.
                        # Retry loop for malformed memory_update blocks
                        max_retries = 3
                        for retry_attempt in range(max_retries):
                            response_text = query(
                                system_message=prompt,
                                user_message=None,
                                model=cfg.agent.code.model,
                                temperature=cfg.agent.code.temp,
                            )
                            # DEBUG: Log raw LLM response before parsing
                            logger.info("[Phase1 DEBUG] Raw LLM response (first 500 chars): %s", repr(response_text[:500]) if response_text else "EMPTY")
                            if phase1_llm_log_path:
                                try:
                                    phase1_llm_log_path.parent.mkdir(parents=True, exist_ok=True)
                                    with open(phase1_llm_log_path, "a", encoding="utf-8") as fh:
                                        fh.write(
                                            json.dumps(
                                                {
                                                    "step": step_idx,
                                                    "max_steps": max_steps,
                                                    "response": response_text,
                                                    "retry_attempt": retry_attempt,
                                                }
                                            )
                                            + "\n"
                                        )
                                except Exception as exc:
                                    logger.warning("Failed to write Phase 1 LLM output: %s", exc)

                            # Apply memory updates from Phase 1 response if present
                            if response_text and worker_agent.memory_manager and child_branch_id:
                                memory_updates = extract_memory_updates(response_text)
                                if memory_updates:
                                    try:
                                        memory_results = worker_agent.memory_manager.apply_llm_memory_updates(
                                            child_branch_id,
                                            memory_updates,
                                            node_id=child_node.id,
                                            phase="phase1_iterative",
                                        )
                                        # Log memory operations for reproducibility
                                        if worker_agent.prompt_log_dir and memory_results:
                                            try:
                                                mem_ops_path = worker_agent.prompt_log_dir / f"phase1_iterative_step{step_idx}_memory_ops.json"
                                                mem_ops_payload = {
                                                    "timestamp": time.time(),
                                                    "phase": "phase1_iterative",
                                                    "step_idx": step_idx,
                                                    "branch_id": child_branch_id,
                                                    "node_id": child_node.id,
                                                    "input_updates": memory_updates,
                                                    "operations_log": memory_results.get("operations_log", []),
                                                    "timing": memory_results.get("timing", {}),
                                                    "read_results": {
                                                        "core_get": memory_results.get("core_get", {}),
                                                        "archival_search": memory_results.get("archival_search", []),
                                                        "recall_search": memory_results.get("recall_search", []),
                                                    },
                                                    "has_read_results": _has_memory_read_results(memory_results),
                                                }
                                                mem_ops_path.write_text(
                                                    json.dumps(mem_ops_payload, indent=2, default=str),
                                                    encoding="utf-8",
                                                )
                                            except Exception as log_exc:
                                                logger.warning("Failed to log Phase 1 memory ops: %s", log_exc)
                                        # Handle memory read operations with re-query loop
                                        if (
                                            memory_results
                                            and _has_memory_read_results(memory_results)
                                            and memory_cfg
                                            and getattr(memory_cfg, "enabled", False)
                                        ):
                                            max_rounds = getattr(memory_cfg, "max_memory_read_rounds", 2)
                                            if max_rounds > 0:
                                                results_text = _format_memory_results_for_llm(memory_results)
                                                followup_prompt = prompt.copy()
                                                followup_prompt["Memory Read Results"] = (
                                                    "Your memory read operations returned the following results:\n\n"
                                                    f"{results_text}\n\n"
                                                    "Based on this information, you may:\n"
                                                    "1. Write additional insights to memory\n"
                                                    "2. Search for more related information\n"
                                                    "3. Complete with an empty update if done\n\n"
                                                    "Respond with ONLY a <memory_update> block."
                                                )
                                                _run_memory_update_phase(
                                                    prompt=followup_prompt,
                                                    memory_manager=worker_agent.memory_manager,
                                                    branch_id=child_branch_id,
                                                    node_id=child_node.id,
                                                    phase_name="phase1_iterative",
                                                    model=cfg.agent.code.model,
                                                    temperature=cfg.agent.code.temp,
                                                    max_rounds=max(0, max_rounds - 1),
                                                    task_description=(
                                                        "You may update memory based on Phase 1 context and memory read results. "
                                                        "Respond with ONLY a <memory_update> block."
                                                    ),
                                                    log_dir=worker_agent.prompt_log_dir,
                                                )
                                    except Exception as exc:
                                        logger.warning("Failed to apply Phase 1 memory updates: %s", exc)

                            try:
                                return parse_phase1_iterative_response(response_text)
                            except MalformedMemoryUpdateError as e:
                                if retry_attempt < max_retries - 1:
                                    logger.warning("[Phase1] Malformed memory_update detected, retrying (%d/%d): %s", retry_attempt + 1, max_retries, e)
                                else:
                                    logger.error("[Phase1] Malformed memory_update detected, max retries exceeded: %s", e)
                                    raise

                    try:
                        download_log_path = phase_log_dir / "download.log"
                        steps_log_path = None
                        if run_root is not None:
                            steps_log_path = run_root / "workers" / worker_label / "phase1_steps.jsonl"
                            phase1_llm_log_path = run_root / "workers" / worker_label / "phase1_llm_outputs.jsonl"
                            steps_log_path.parent.mkdir(parents=True, exist_ok=True)

                        # For seed_eval, reuse parent node's worker SIF instead of running Phase 1 again
                        parent_sif_path = getattr(child_node, "worker_sif_path", None)
                        if seed_eval and parent_sif_path and Path(parent_sif_path).exists():
                            logger.info("seed_eval: Reusing parent worker SIF: %s", parent_sif_path)
                            term_outputs.append(f"[seed_eval] Reusing parent worker SIF: {parent_sif_path}")
                            try:
                                active_env = ExecutionEnvironment(
                                    workspace=workspace_path,
                                    image=parent_sif_path,
                                    runtime_preference="singularity",
                                    workspace_mount=getattr(cfg.exec, "workspace_mount", "/workspace"),
                                    gpu_id=gpu_id if use_gpu else None,
                                    enable_gpu=use_gpu,
                                    enable_writable_tmpfs=getattr(cfg.exec, "writable_tmpfs", True),
                                    overlay_path=getattr(cfg.exec, "container_overlay", None),
                                    extra_start_args=getattr(cfg.exec, "container_extra_args", None),
                                    resource_binds=resource_binds,
                                )
                                active_env.start()
                            except Exception as exc:
                                exc_type = "EnvironmentError"
                                exc_info = {"message": f"Failed to start execution environment from parent SIF: {exc}"}
                                logger.warning("seed_eval: Failed to reuse parent SIF: %s", exc)
                        elif worker_container:
                            success, outputs, failure = worker_container.prepare_phase1(
                                download_commands,
                                workspace=workspace_path,
                                workspace_mount=getattr(cfg.exec, "workspace_mount", "/workspace"),
                                download_log=download_log_path,
                                steps_log=steps_log_path,
                                extra_env={"CUDA_VISIBLE_DEVICES": str(gpu_id)} if use_gpu and gpu_id is not None else None,
                                iterative_driver=phase1_iterative_driver,
                                max_steps=phase1_max_steps,
                            )
                            term_outputs.extend(outputs)
                            if success:
                                # Save worker SIF path for potential seed_eval reuse
                                child_node.worker_sif_path = str(worker_container.worker_sif)
                                # Memory event: Phase 1 complete
                                if memory_manager and child_branch_id:
                                    try:
                                        phase1_summary = f"Phase 1 download/install complete for node {child_node.id}"
                                        if download_commands:
                                            phase1_summary += f"\nCommands: {download_commands[:3]}{'...' if len(download_commands) > 3 else ''}"
                                        memory_manager.mem_recall_append({
                                            "ts": time.time(),
                                            "run_id": memory_manager.run_id,
                                            "node_id": child_node.id,
                                            "branch_id": child_branch_id,
                                            "phase": stage_name,
                                            "kind": "phase1_complete",
                                            "summary": phase1_summary,
                                            "refs": [],
                                        })
                                    except Exception as exc:
                                        logger.warning("Failed to write phase1_complete event: %s", exc)
                                try:
                                    active_env = worker_container.create_execution_env(
                                        gpu_id=gpu_id if use_gpu else None,
                                        enable_writable_tmpfs=getattr(cfg.exec, "writable_tmpfs", True),
                                        overlay_path=worker_container.overlay_path,
                                        extra_start_args=getattr(cfg.exec, "container_extra_args", None),
                                    )
                                except Exception as exc:
                                    exc_type = "EnvironmentError"
                                    exc_info = {"message": str(exc)}
                            else:
                                exc_type = "DownloadError"
                                exc_info = failure
                                # Memory event: Phase 1 failed
                                if memory_manager and child_branch_id:
                                    try:
                                        failure_msg = str(failure)[:500] if failure else "Unknown failure"
                                        memory_manager.mem_recall_append({
                                            "ts": time.time(),
                                            "run_id": memory_manager.run_id,
                                            "node_id": child_node.id,
                                            "branch_id": child_branch_id,
                                            "phase": stage_name,
                                            "kind": "phase1_failed",
                                            "summary": f"Phase 1 download/install failed for node {child_node.id}: {failure_msg}",
                                            "refs": [],
                                        })
                                    except Exception as exc:
                                        logger.warning("Failed to write phase1_failed event: %s", exc)
                        else:
                            exc_type = "EnvironmentError"
                            exc_info = {"message": "Singularity image is required for split-phase execution."}
                            term_outputs.append(exc_info["message"])
                        if exc_type is None and active_env is None and worker_container:
                            exc_type = "EnvironmentError"
                            exc_info = {"message": "Failed to initialize worker container."}
                        if exc_type is None and active_env is None:
                            active_env = exec_env
                            if active_env is None:
                                active_env = ExecutionEnvironment(
                                    workspace=workspace_path,
                                    image=getattr(cfg.exec, "singularity_image", None),
                                    runtime_preference="singularity",
                                    workspace_mount=getattr(cfg.exec, "workspace_mount", "/workspace"),
                                    gpu_id=gpu_id if use_gpu else None,
                                    enable_gpu=use_gpu,
                                    enable_writable_tmpfs=getattr(cfg.exec, "writable_tmpfs", True),
                                    overlay_path=getattr(cfg.exec, "container_overlay", None),
                                    extra_start_args=getattr(cfg.exec, "container_extra_args", None),
                                    resource_binds=resource_binds,
                                )
                                try:
                                    active_env.start()
                                except Exception as exc:
                                    logger.warning("Failed to start execution environment: %s", exc)
                        if exc_type is None:
                            try:
                                created_files = apply_workspace_plan(
                                    workspace_path,
                                    coding_workspace,
                                    expected_root=getattr(cfg.exec, "workspace_mount", "/workspace"),
                                )
                                with open(phase_log_dir / "coding.log", "a") as log:
                                    log.write("Generated files:\n")
                                    for path in created_files:
                                        log.write(str(path) + "\n")
                                    for line in coding_workspace.get("tree", []):
                                        log.write(str(line) + "\n")
                                term_outputs.append("Coding phase wrote files.")
                                # Memory event: Coding phase complete
                                if memory_manager and child_branch_id:
                                    try:
                                        files_list = [str(p) for p in created_files[:5]]
                                        if len(created_files) > 5:
                                            files_list.append(f"... and {len(created_files) - 5} more")
                                        memory_manager.mem_recall_append({
                                            "ts": time.time(),
                                            "run_id": memory_manager.run_id,
                                            "node_id": child_node.id,
                                            "branch_id": child_branch_id,
                                            "phase": stage_name,
                                            "kind": "coding_complete",
                                            "summary": f"Coding phase complete for node {child_node.id}\nFiles: {files_list}",
                                            "refs": [],
                                        })
                                    except Exception as exc:
                                        logger.warning("Failed to write coding_complete event: %s", exc)
                            except Exception as exc:
                                exc_type = "CodingError"
                                exc_info = {"message": str(exc)}
                                # Memory event: Coding phase failed
                                if memory_manager and child_branch_id:
                                    try:
                                        memory_manager.mem_recall_append({
                                            "ts": time.time(),
                                            "run_id": memory_manager.run_id,
                                            "node_id": child_node.id,
                                            "branch_id": child_branch_id,
                                            "phase": stage_name,
                                            "kind": "coding_failed",
                                            "summary": f"Coding phase failed for node {child_node.id}: {str(exc)[:300]}",
                                            "refs": [],
                                        })
                                    except Exception as mem_exc:
                                        logger.warning("Failed to write coding_failed event: %s", mem_exc)
                            if exc_type is None:
                                # Check if this is a Python experiment (no compilation needed)
                                build_language = (build_plan.get("language") or "").strip().lower()
                                is_python_experiment = build_language == "python"

                                if is_python_experiment:
                                    # Skip compilation phase for Python experiments
                                    term_outputs.append("Python experiment: skipping compile phase.")
                                elif not selected_compiler:
                                    exc_type = "CompilationError"
                                    exc_info = {"message": "build_plan.compiler_selected is required for compiled languages."}
                                    term_outputs.append(exc_info["message"])
                                elif available_compiler_names and selected_compiler not in available_compiler_names:
                                    exc_type = "CompilationError"
                                    exc_info = {
                                        "message": f"compiler_selected '{selected_compiler}' not in available_compilers.",
                                        "available_compilers": available_compiler_names,
                                    }
                                    term_outputs.append(exc_info["message"])

                                # --- Compile phase (compiled languages only) ---
                                if exc_type is None and not is_python_experiment:
                                    term_outputs.append(f"Using compiler_selected: {selected_compiler}")
                                    fmt_ctx = {**build_plan, "compiler_selected": selected_compiler}
                                    formatted_compile_cmds = []
                                    for cmd in compile_commands:
                                        if isinstance(cmd, str):
                                            try:
                                                formatted_compile_cmds.append(cmd.format(**fmt_ctx))
                                            except Exception:
                                                formatted_compile_cmds.append(cmd)
                                        else:
                                            formatted_compile_cmds.append(cmd)
                                    compile_cwd = resolve_workdir(build_plan.get("workdir"))
                                    compile_cwd.mkdir(parents=True, exist_ok=True)
                                    success, outputs, failure = run_commands_with_logging(
                                        active_env,
                                        formatted_compile_cmds,
                                        compile_cwd,
                                        phase_log_dir / "compile.log",
                                        "compile",
                                        extra_env={"COMPILER_SELECTED": selected_compiler},
                                    )
                                    term_outputs.extend(outputs)
                                    if not success:
                                        exc_type = "CompilationError"
                                        exc_info = {
                                            "returncode": getattr(failure, "returncode", None),
                                            "stderr": getattr(failure, "stderr", ""),
                                        }
                                        # Memory event: Compile phase failed
                                        if memory_manager and child_branch_id:
                                            try:
                                                stderr_snippet = getattr(failure, "stderr", "")[:300] if failure else ""
                                                memory_manager.mem_recall_append({
                                                    "ts": time.time(),
                                                    "run_id": memory_manager.run_id,
                                                    "node_id": child_node.id,
                                                    "branch_id": child_branch_id,
                                                    "phase": stage_name,
                                                    "kind": "compile_failed",
                                                    "summary": f"Compile phase failed for node {child_node.id}\nCompiler: {selected_compiler}\nError: {stderr_snippet}",
                                                    "refs": [],
                                                })
                                            except Exception as exc:
                                                logger.warning("Failed to write compile_failed event: %s", exc)
                                    else:
                                        # Memory event: Compile phase complete
                                        if memory_manager and child_branch_id:
                                            try:
                                                memory_manager.mem_recall_append({
                                                    "ts": time.time(),
                                                    "run_id": memory_manager.run_id,
                                                    "node_id": child_node.id,
                                                    "branch_id": child_branch_id,
                                                    "phase": stage_name,
                                                    "kind": "compile_complete",
                                                    "summary": f"Compile phase complete for node {child_node.id}\nCompiler: {selected_compiler}\nCommands: {formatted_compile_cmds[:2]}",
                                                    "refs": [],
                                                })
                                            except Exception as exc:
                                                logger.warning("Failed to write compile_complete event: %s", exc)

                                # --- Run phase (both Python and compiled languages) ---
                                if exc_type is None:
                                    fmt_ctx = fmt_ctx if not is_python_experiment else {**build_plan}
                                    formatted_run_cmds = []
                                    for cmd in run_commands:
                                        if isinstance(cmd, str):
                                            try:
                                                formatted_run_cmds.append(cmd.format(**fmt_ctx))
                                            except Exception:
                                                formatted_run_cmds.append(cmd)
                                        else:
                                            formatted_run_cmds.append(cmd)
                                    run_cwd = resolve_workdir(build_plan.get("workdir"))
                                    run_cwd.mkdir(parents=True, exist_ok=True)
                                    success, outputs, failure = run_commands_with_logging(
                                        active_env,
                                        formatted_run_cmds,
                                        run_cwd,
                                        phase_log_dir / "run.log",
                                        "run",
                                    )
                                    term_outputs.extend(outputs)
                                    if not success:
                                        exc_type = "RuntimeError"
                                        exc_info = {
                                            "returncode": getattr(failure, "returncode", None),
                                            "stderr": getattr(failure, "stderr", ""),
                                        }
                                        # Memory event: Run phase failed
                                        if memory_manager and child_branch_id:
                                            try:
                                                stderr_snippet = getattr(failure, "stderr", "")[:300] if failure else ""
                                                memory_manager.mem_recall_append({
                                                    "ts": time.time(),
                                                    "run_id": memory_manager.run_id,
                                                    "node_id": child_node.id,
                                                    "branch_id": child_branch_id,
                                                    "phase": stage_name,
                                                    "kind": "run_failed",
                                                    "summary": f"Run phase failed for node {child_node.id}\nError: {stderr_snippet}",
                                                    "refs": [],
                                                })
                                            except Exception as exc:
                                                logger.warning("Failed to write run_failed event: %s", exc)
                                    else:
                                        missing_outputs = [
                                            str(path)
                                            for path in expected_output_paths
                                            if not path.exists()
                                        ]
                                        if missing_outputs:
                                            exc_type = "RuntimeError"
                                            exc_info = {
                                                "message": "Expected output file(s) missing after run phase.",
                                                "missing": missing_outputs,
                                                "expected_outputs": expected_outputs,
                                            }
                                            term_outputs.append(exc_info["message"])
                                            # Memory event: Run phase failed (missing outputs)
                                            if memory_manager and child_branch_id:
                                                try:
                                                    memory_manager.mem_recall_append({
                                                        "ts": time.time(),
                                                        "run_id": memory_manager.run_id,
                                                        "node_id": child_node.id,
                                                        "branch_id": child_branch_id,
                                                        "phase": stage_name,
                                                        "kind": "run_failed",
                                                        "summary": f"Run phase failed for node {child_node.id}: missing outputs {missing_outputs[:3]}",
                                                        "refs": [],
                                                    })
                                                except Exception as exc:
                                                    logger.warning("Failed to write run_failed event: %s", exc)
                                        else:
                                            # Memory event: Run phase complete
                                            if memory_manager and child_branch_id:
                                                try:
                                                    memory_manager.mem_recall_append({
                                                        "ts": time.time(),
                                                        "run_id": memory_manager.run_id,
                                                        "node_id": child_node.id,
                                                        "branch_id": child_branch_id,
                                                        "phase": stage_name,
                                                        "kind": "run_complete",
                                                        "summary": f"Run phase complete for node {child_node.id}\nOutputs: {expected_outputs[:3]}",
                                                        "refs": [],
                                                    })
                                                except Exception as exc:
                                                    logger.warning("Failed to write run_complete event: %s", exc)
                        exec_time = time.time() - exec_start
                        exec_result = ExecutionResult(
                            term_outputs,
                            exec_time,
                            exc_type,
                            exc_info,
                            None,
                        )
                    except Exception as exc:
                        exec_time = time.time() - exec_start
                        exec_result = ExecutionResult(
                            term_outputs + [str(exc)],
                            exec_time,
                            "EnvironmentError",
                            {"message": str(exc)},
                            None,
                        )
                    finally:
                        # NOTE: Do NOT stop active_env here - it's needed for metrics parsing and plotting
                        # active_env will be stopped in the outer finally block (around line 5241)
                        pass
            else:
                exec_result = process_interpreter.run(child_node.code, True)
                process_interpreter.cleanup_session()

            print("Parsing execution results")
            worker_agent.parse_exec_result(
                node=child_node, exec_result=exec_result, workspace=str(workspace_path)
            )

            # Save intermediate result after execution (for timeout recovery)
            _save_intermediate_result(child_node, stage="after_execution")

            # Add check for saved data files
            data_dir = Path(working_dir)
            data_files = list(data_dir.rglob("*.npy")) if data_dir.exists() else []
            # Use dynamic output filename based on experiment name
            expected_output_filename = _get_experiment_output_filename(Path(cfg.workspace_dir).name)
            expected_output_file = workspace_path / "working" / expected_output_filename
            if expected_output_file.exists() and expected_output_file not in data_files:
                data_files.append(expected_output_file)
            # Also check for legacy experiment_data.npy for backwards compatibility
            legacy_output_file = workspace_path / "working" / DEFAULT_EXPERIMENT_OUTPUT_FILENAME
            if legacy_output_file.exists() and legacy_output_file not in data_files:
                data_files.append(legacy_output_file)
            if not data_files:
                logger.warning(
                    "No .npy files found in working directory. Data may not have been saved properly."
                )
            else:
                parse_success = False
                parse_attempt = 0
                last_error_message = ""
                previous_error_message: Optional[str] = None
                previous_error_code: Optional[str] = None
                while (
                    parse_attempt < MAX_METRIC_PARSE_RETRIES and not parse_success
                ):
                    parse_attempt += 1
                    if seed_eval:
                        # Use the parent node's parse code to parse the same data files again
                        parse_metrics_code = parent_node.parse_metrics_code
                        parse_metrics_plan = parent_node.parse_metrics_plan
                        print(
                            f"[blue]SEED EVAL: Parse metrics plan:[/blue] {parse_metrics_plan}"
                        )
                        print(
                            f"[blue]SEED EVAL: Parse metrics code:[/blue] {parse_metrics_code}"
                        )
                    else:
                        # Call LLM to parse data files and extract metrics
                        context_blocks = ["Original Code: " + child_node.code]
                        if data_files:
                            file_info_lines = []
                            for p in sorted(data_files):
                                try:
                                    rel_path = str(p.relative_to(workspace_path))
                                except ValueError:
                                    rel_path = str(p)
                                # Extract schema from each npy file
                                schema = _extract_npy_schema(p)
                                file_info_lines.append(f"- {rel_path}\n  Schema: {schema}")
                            context_blocks.append(
                                "Available .npy files with their data schemas:\n"
                                + "\n".join(file_info_lines)
                            )
                        if previous_error_message:
                            context_blocks.append(
                                "Previous parsing attempt failed with the following error. "
                                "Revise your plan and code to specifically address this issue:\n"
                                f"{previous_error_message}"
                            )
                        if previous_error_code:
                            context_blocks.append(
                                "Below is the last parsing code that failed. Use this as reference and fix the issues:\n"
                                f"{previous_error_code}"
                            )
                        # Select prompt based on memory configuration
                        parse_metrics_intro = (
                            PARSE_METRICS_INTRO_WITH_MEMORY
                            if worker_agent._is_memory_enabled
                            else PARSE_METRICS_INTRO
                        )
                        parse_metrics_prompt = {
                            "Introduction": parse_metrics_intro,
                            "Context": context_blocks,
                            "Instructions": list(PARSE_METRICS_INSTRUCTIONS),
                            "Example data loading code": [
                                PARSE_METRICS_EXAMPLE
                            ],
                            "Response format": worker_agent._prompt_metricparse_resp_fmt(),
                        }
                        worker_agent._inject_memory(
                            parse_metrics_prompt,
                            "parse_metrics",
                            branch_id=getattr(child_node, "branch_id", None),
                            budget_chars=getattr(cfg.memory, "parse_metrics_budget_chars", 2000),
                        )

                        # LLM context (parse-metrics plan): Introduction + Context (original code + prior errors/code) + Instructions + Example parser + Response format + Memory.
                        (
                            parse_metrics_plan,
                            parse_metrics_code,
                        ) = worker_agent.plan_and_code_query(parse_metrics_prompt, retries=3, code_language="python")
                        print(f"[blue]Parse metrics plan:[/blue] {parse_metrics_plan}")
                        print(f"[blue]Parse metrics code:[/blue] {parse_metrics_code}")

                    child_node.parse_metrics_plan = parse_metrics_plan
                    child_node.parse_metrics_code = parse_metrics_code

                    try:
                        # Execute the parsing code
                        use_container_python = (
                            cfg.exec.phase_mode == "split"
                            and isinstance(active_env, ExecutionEnvironment)
                        )
                        if use_container_python:
                            metrics_exec_result = _run_python_in_container(
                                active_env,
                                code=parse_metrics_code,
                                file_path=workspace_path / parse_agent_file_name,
                                cwd=workspace_path,
                            )
                        else:
                            metrics_exec_result = parse_interpreter.run(
                                parse_metrics_code, True
                            )
                            parse_interpreter.cleanup_session()
                        child_node.parse_term_out = metrics_exec_result.term_out
                        child_node.parse_exc_type = metrics_exec_result.exc_type
                        child_node.parse_exc_info = metrics_exec_result.exc_info
                        child_node.parse_exc_stack = metrics_exec_result.exc_stack

                        if metrics_exec_result.exc_type is None:
                            # Extract metrics from the execution output
                            metrics_prompt = {
                                "Introduction": METRICS_PROMPT_INTRO,
                                "Execution Output": metrics_exec_result.term_out,
                            }
                            metrics_branch_id = getattr(child_node, "branch_id", None)
                            worker_agent._inject_memory(
                                metrics_prompt,
                                "metrics_extraction",
                                branch_id=metrics_branch_id,
                                budget_chars=getattr(cfg.memory, "metrics_extraction_budget_chars", 1500),
                            )
                            print(
                                f"[blue]Metrics_exec_result.term_out: {metrics_exec_result.term_out}[/blue]"
                            )
                            print(
                                f"[blue]Metrics Parsing Execution Result:\n[/blue] {metrics_exec_result}"
                            )

                            # Phase 1: Memory update (with multi-round support)
                            if worker_agent._is_memory_enabled and memory_manager and metrics_branch_id:
                                _run_memory_update_phase(
                                    prompt=metrics_prompt,
                                    memory_manager=memory_manager,
                                    branch_id=metrics_branch_id,
                                    node_id=child_node.id,
                                    phase_name="metrics_extraction",
                                    model=cfg.agent.feedback.model,
                                    temperature=cfg.agent.feedback.temp,
                                    max_rounds=getattr(cfg.memory, "max_memory_read_rounds", 2),
                                    task_description=(
                                        "Review the metrics output and update your memory with any important observations, "
                                        "metric patterns, or performance insights that would be useful for future analysis. "
                                        "You can also search memory for patterns from previous metric extractions. "
                                        "Respond with ONLY a <memory_update> block containing your memory operations."
                                    ),
                                    log_dir=worker_agent.prompt_log_dir,
                                )

                            # Phase 2: Task execution with structured response
                            metrics_response = cast(
                                dict,
                                # LLM context (metric extraction): Introduction + Execution Output from metrics parser + Memory.
                                query(
                                    system_message=metrics_prompt,
                                    user_message=None,
                                    func_spec=metric_parse_spec,
                                    model=cfg.agent.feedback.model,
                                    temperature=cfg.agent.feedback.temp,
                                ),
                            )
                            print(f"[blue]Metrics:[/blue] {metrics_response}")
                            if metrics_response["valid_metrics_received"]:
                                child_node.metric = MetricValue(
                                    value={
                                        "metric_names": metrics_response["metric_names"]
                                    }
                                )
                                logger.info(
                                    f"Successfully extracted metrics for node {child_node.id}"
                                )
                                parse_success = True
                                # Write metrics to memory for future reference
                                if memory_manager and child_branch_id:
                                    try:
                                        metrics_summary = (
                                            f"Metrics extracted for node {child_node.id}: "
                                            f"{metrics_response['metric_names']}"
                                        )
                                        memory_manager.mem_recall_append({
                                            "ts": time.time(),
                                            "run_id": memory_manager.run_id,
                                            "node_id": child_node.id,
                                            "branch_id": child_branch_id,
                                            "phase": stage_name,
                                            "kind": "metrics_extracted",
                                            "summary": metrics_summary,
                                            "refs": [],
                                        })
                                        # Write detailed metrics to archival for long-term reference
                                        memory_manager.mem_archival_write(
                                            f"Metrics extraction successful\n"
                                            f"Node: {child_node.id}\n"
                                            f"Stage: {stage_name}\n"
                                            f"Metrics: {json.dumps(metrics_response['metric_names'], indent=2)}\n"
                                            f"Parse plan: {parse_metrics_plan[:500] if parse_metrics_plan else 'N/A'}",
                                            tags=["METRICS", f"node_uid:{child_node.id}", f"stage:{stage_name}"],
                                            meta={
                                                "node_id": child_node.id,
                                                "branch_id": child_branch_id,
                                                "run_id": memory_manager.run_id,
                                                "phase": stage_name,
                                            },
                                        )
                                    except Exception as mem_exc:
                                        logger.warning("Failed to write metrics to memory: %s", mem_exc)
                            else:
                                last_error_message = (
                                    "Metrics parser did not return valid metrics."
                                )
                        else:
                            last_error_message = str(metrics_exec_result.exc_info)

                    except Exception as e:
                        last_error_message = str(e)
                        logger.error(
                            f"Error parsing metrics for node {child_node.id}: {last_error_message}"
                        )
                        child_node.parse_exc_type = str(e)
                        child_node.parse_exc_info = None
                        child_node.parse_exc_stack = None
                        child_node.parse_term_out = (
                            "Error parsing metrics. There was an error in the parsing code: "
                            + str(e)
                        )

                    if not parse_success:
                        logger.warning(
                            f"Metrics parsing attempt {parse_attempt} failed for node {child_node.id}: {last_error_message}"
                        )
                        if parse_attempt < MAX_METRIC_PARSE_RETRIES:
                            logger.info("Retrying metrics parsing with a new attempt.")
                        previous_error_message = last_error_message
                        previous_error_code = parse_metrics_code

                if not parse_success:
                    child_node.metric = WorstMetricValue()
                    child_node.is_buggy = True
                    logger.error(
                        f"No valid metrics received for node {child_node.id} after {MAX_METRIC_PARSE_RETRIES} attempts."
                    )
                    # Write metrics parsing failure to memory
                    if memory_manager and child_branch_id:
                        try:
                            memory_manager.mem_recall_append({
                                "ts": time.time(),
                                "run_id": memory_manager.run_id,
                                "node_id": child_node.id,
                                "branch_id": child_branch_id,
                                "phase": stage_name,
                                "kind": "metrics_failed",
                                "summary": f"Metrics parsing failed for node {child_node.id}: {last_error_message[:200] if last_error_message else 'Unknown error'}",
                                "refs": [],
                            })
                        except Exception as mem_exc:
                            logger.warning("Failed to write metrics failure to memory: %s", mem_exc)

            # if experiment was successful, generate and run plotting code
            if not child_node.is_buggy:
                try:
                    plots_workdir = str(workspace_path) if cfg.exec.phase_mode == "split" else working_dir
                    retry_count = 0
                    while True:
                        if seed_eval:
                            # Use the parent node's plotting code instead of generating new one
                            plotting_code = parent_node.plot_code
                        else:
                            if (
                                worker_agent.stage_name
                                and worker_agent.stage_name.startswith("3_")
                                and best_stage2_plot_code
                            ):
                                plot_code_from_prev_stage = best_stage2_plot_code
                            elif (
                                worker_agent.stage_name
                                and worker_agent.stage_name.startswith("4_")
                                and best_stage3_plot_code
                            ):
                                plot_code_from_prev_stage = best_stage3_plot_code
                            else:
                                plot_code_from_prev_stage = None

                            plotting_code = worker_agent._generate_plotting_code(
                                child_node, plots_workdir, plot_code_from_prev_stage
                            )
                        use_container_python = (
                            cfg.exec.phase_mode == "split"
                            and isinstance(active_env, ExecutionEnvironment)
                        )
                        if use_container_python:
                            plot_exec_result = _run_python_in_container(
                                active_env,
                                code=plotting_code,
                                file_path=workspace_path / plot_agent_file_name,
                                cwd=workspace_path,
                            )
                        else:
                            plot_exec_result = plot_interpreter.run(plotting_code, True)
                            plot_interpreter.cleanup_session()
                        child_node.absorb_plot_exec_result(plot_exec_result)
                        child_node.plot_exec_result = plot_exec_result
                        if child_node.plot_exc_type and retry_count < 3:
                            print(
                                f"[red]Plotting code failed with exception: {child_node.plot_exc_type}[/red]"
                            )
                            print(
                                f"[red]Plotting code term out:[/red] {child_node.plot_term_out}"
                            )
                            print(
                                f"[red]Plotting code code:[/red] {child_node.plot_code}"
                            )
                            retry_count += 1
                            continue
                        else:
                            break

                    print("[blue]Plotting result:[/blue] ", plot_exec_result)
                    # Track generated plots
                    plots_root = Path(plots_workdir)
                    candidate_plot_dirs = [plots_root / "working", plots_root]
                    existing_plot_dirs = [d for d in candidate_plot_dirs if d.exists()]
                    if existing_plot_dirs:
                        print("Plots directory exists, saving plots to node")
                        # Save the plotting code first
                        base_dir = Path(cfg.workspace_dir).parent
                        run_name = Path(cfg.workspace_dir).name
                        exp_results_dir = (
                            base_dir
                            / "logs"
                            / run_name
                            / "experiment_results"
                            / f"experiment_{child_node.id}_proc_{os.getpid()}"
                        )
                        child_node.exp_results_dir = exp_results_dir
                        exp_results_dir.mkdir(parents=True, exist_ok=True)
                        if cfg.exec.phase_mode == "split":
                            plot_code_path = exp_results_dir / "plotting_code.txt"
                            exp_code_path = exp_results_dir / "experiment_code.txt"
                        else:
                            code_suffix = Path(cfg.exec.agent_file_name).suffix or ".py"
                            plot_code_path = exp_results_dir / "plotting_code.py"
                            exp_code_path = exp_results_dir / f"experiment_code{code_suffix}"
                        with open(plot_code_path, "w") as f:
                            f.write(plotting_code)
                        logger.info(f"Saved plotting code to {plot_code_path}")
                        # Save experiment code to experiment_results directory
                        with open(exp_code_path, "w") as f:
                            f.write(child_node.code)
                        logger.info(f"Saved experiment code to {exp_code_path}")
                        # Copy experiment data files to experiment_results directory
                        # Use copy instead of move to preserve originals for downstream use
                        for plots_dir in existing_plot_dirs:
                            for exp_data_file in plots_dir.glob("*.npy"):
                                exp_data_path = exp_results_dir / exp_data_file.name
                                shutil.copy2(str(exp_data_file.resolve()), str(exp_data_path))
                                logger.info(f"Saved experiment data to {exp_data_path}")

                        for plots_dir in existing_plot_dirs:
                            for plot_file in plots_dir.glob("*.png"):
                                # Create the final path in experiment_results directory
                                final_path = exp_results_dir / plot_file.name

                                # Use shutil.move instead of rename to handle cross-device moves
                                try:
                                    shutil.move(str(plot_file.resolve()), str(final_path))
                                except Exception as move_exc:
                                    logger.warning(f"Failed to move plot {plot_file} to {final_path}: {move_exc}")
                                    # Try copy as fallback
                                    try:
                                        shutil.copy2(str(plot_file.resolve()), str(final_path))
                                    except Exception:
                                        logger.warning(f"Failed to copy plot {plot_file} to {final_path}, skipping")
                                        continue

                                if not final_path.exists():
                                    logger.warning(f"Plot file not found after move: {final_path}")
                                    continue

                                # Create a web-friendly relative path starting from logs directory
                                web_path = f"../../logs/{Path(cfg.workspace_dir).name}/experiment_results/experiment_{child_node.id}_proc_{os.getpid()}/{plot_file.name}"

                                child_node.plots.append(web_path)  # For visualization
                                child_node.plot_paths.append(
                                    str(final_path.absolute())
                                )  # For programmatic access

                                logger.info(
                                    f"[green]Generated plot: {plot_file.stem}[/green]"
                                )
                                logger.debug(f"Plot absolute path: {final_path.absolute()}")
                                logger.debug(f"Plot web path: {web_path}")
                except Exception as e:
                    logger.error(
                        f"Error generating plots for node {child_node.id}: {str(e)}"
                    )

                if child_node.plots:
                    try:
                        worker_agent._analyze_plots_with_vlm(child_node)
                        logger.info(
                            f"Generated VLM analysis for plots in node {child_node.id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error analyzing plots for node {child_node.id}: {str(e)}"
                        )

            if (
                cfg.exec.phase_mode == "split"
                and not child_node.is_buggy
                and child_node.is_buggy_plots is False
                and child_node.exp_results_dir
            ):
                try:
                    exp_results_dir = Path(child_node.exp_results_dir)
                    _save_phase_execution_artifacts(
                        exp_results_dir=exp_results_dir,
                        phase_log_dir=phase_log_dir,
                        run_root=run_root,
                        worker_label=worker_label,
                        phase_artifacts=child_node.phase_artifacts,
                        phase_artifacts_raw=getattr(child_node, "phase_artifacts_raw", ""),
                    )
                except Exception as exc:
                    logger.warning("Failed to save phase artifacts: %s", exc)

            if memory_manager and child_branch_id:
                try:
                    result_text = (
                        f"node_id={child_node.id} "
                        f"metric={child_node.metric} "
                        f"is_buggy={child_node.is_buggy} "
                        f"exc_type={child_node.exc_type}"
                    )
                    memory_manager.mem_recall_append(
                        {
                            "ts": time.time(),
                            "run_id": memory_manager.run_id,
                            "node_id": child_node.id,
                            "branch_id": child_branch_id,
                            "phase": stage_name,
                            "kind": "node_result",
                            "summary": result_text,
                            "refs": [],
                        }
                    )
                    # Write successful results to archival for long-term memory
                    if not child_node.is_buggy and child_node.metric is not None:
                        method_changes = getattr(child_node, "method_changes", None) or ""
                        archival_summary = (
                            f"Successful node {child_node.id} in {stage_name}\n"
                            f"Metric: {child_node.metric}\n"
                            f"Method changes: {method_changes[:500] if method_changes else 'N/A'}"
                        )
                        memory_manager.mem_archival_write(
                            archival_summary,
                            tags=["SUCCESS", f"node_uid:{child_node.id}", f"stage:{stage_name}"],
                            meta={
                                "node_id": child_node.id,
                                "branch_id": child_branch_id,
                                "run_id": memory_manager.run_id,
                                "phase": stage_name,
                            },
                        )
                    if child_node.is_buggy and child_node.exc_info:
                        memory_manager.mem_archival_write(
                            f"Execution error: {child_node.exc_info}",
                            tags=["ERROR", f"node_uid:{child_node.id}"],
                            meta={
                                "node_id": child_node.id,
                                "branch_id": child_branch_id,
                                "run_id": memory_manager.run_id,
                            },
                        )
                except Exception as exc:
                    logger.warning("Failed to write node_result memory: %s", exc)

            # Copy workspace to persistent log directory for cross-stage file inheritance
            # This prevents race conditions where worker workspace is overwritten by another task
            node_log_dir = Path(cfg.workspace_dir) / "node_logs" / f"node_{child_node.id}"
            try:
                if node_log_dir.exists():
                    shutil.rmtree(node_log_dir)
                node_log_dir.mkdir(parents=True, exist_ok=True)
                # Copy essential directories (src, bin, input, working)
                for item_name in ["src", "bin", "input", "working"]:
                    src_item = workspace_path / item_name
                    if src_item.exists():
                        dst_item = node_log_dir / item_name
                        if src_item.is_dir():
                            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_item, dst_item)
                print(f"Copied workspace to node log directory: {node_log_dir}")
            except Exception as exc:
                logger.warning(f"Failed to copy workspace to log directory: {exc}")
                # Fall back to worker workspace path if copy fails
                node_log_dir = workspace_path

            # Set workspace_path on child node for cross-stage file inheritance
            child_node.workspace_path = str(node_log_dir)

            # Save final intermediate result (for timeout recovery)
            _save_intermediate_result(child_node, stage="final")

            # Convert result node to dict
            print("Converting result to dict")
            result_data = child_node.to_dict()
            print(f"Result data keys: {result_data.keys()}")
            print(f"Result data size: {len(str(result_data))} chars")
            print("Returning result")
            return result_data

        except Exception as e:
            print(f"Worker process error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            if exec_env:
                try:
                    exec_env.stop()
                except Exception as exc:
                    logger.warning("Failed to stop execution environment: %s", exc)
            # Stop active_env (used for metrics parsing and plotting in split mode)
            if isinstance(active_env, ExecutionEnvironment):
                try:
                    active_env.stop()
                except Exception as exc:
                    logger.warning("Failed to stop active execution environment: %s", exc)
            if plot_interpreter:
                plot_interpreter.cleanup_session()
            if process_interpreter:
                process_interpreter.cleanup_session()
            if parse_interpreter:
                parse_interpreter.cleanup_session()
            if memory_manager:
                memory_manager.close()

    def _generate_hyperparam_tuning_idea(self) -> Optional[HyperparamTuningIdea]:
        """Generate the next hyperparam tuning idea based on what's been done.
        This is minaly for Stage 2 (baseline tuning).
        """
        tried = list(self._hyperparam_tuning_state["tried_hyperparams"])

        hyperparam_tuning_prompt = {
            "Introduction": HYPERPARAM_PROMPT_INTRO,
            "Base code you are working on": wrap_combined_code(self.best_stage1_node.code, fallback_lang=self.code_language),
            "Previous Hyperparam Tuning Attempts": {
                "Has been tried": tried if tried else "Nothing has been tried yet.",
            },
            "Instructions": {
                "Requirements": list(HYPERPARAM_PROMPT_INSTRUCTIONS)
            },
            "Response format": HYPERPARAM_PROMPT_RESPONSE,
        }
        # NOTE: root branch has no memory data, so we skip memory injection for hyperparam_idea.

        retry_count = 0
        retry_limit = 5
        while retry_count < retry_limit:
            # LLM context (Stage 2 idea): Introduction + Base code + Previous hyperparam attempts + Requirements + Response format.
            response = query(
                system_message=hyperparam_tuning_prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            # Parse the response
            hyperparam_name, hyperparam_description = _parse_keyword_prefix_response(
                response, "HYPERPARAM NAME:", "DESCRIPTION:"
            )
            if hyperparam_name and hyperparam_description:
                return HyperparamTuningIdea(
                    name=hyperparam_name, description=hyperparam_description
                )

            retry_count += 1
            logger.warning(
                f"Failed to parse hyperparam tuning response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse hyperparam tuning response after {retry_limit} retries. Falling back to default idea of increasing learning rate."
        )
        return HyperparamTuningIdea(
            name="increase learning rate", description="increase learning rate"
        )

    def _generate_ablation_idea(self) -> Optional[AblationIdea]:
        """Generate the next ablation idea based on what's been done"""

        # Prepare context of what's been tried
        completed = list(self._ablation_state["completed_ablations"])

        ablation_prompt = {
            "Introduction": ABLATION_PROMPT_INTRO,
            "Base code you are working on": wrap_combined_code(self.best_stage3_node.code, fallback_lang=self.code_language),
            "Previous Ablations": {
                "Has been tried": (
                    completed if completed else "Nothing has been tried yet."
                ),
            },
            "Instructions": {
                "Requirements": list(ABLATION_PROMPT_INSTRUCTIONS)
            },
            "Response format": ABLATION_PROMPT_RESPONSE,
        }
        # NOTE: root branch has no memory data, so we skip memory injection for ablation_idea.

        retry_count = 0
        retry_limit = 5
        while retry_count < retry_limit:
            # LLM context (Stage 4 idea): Introduction + Base code + Previous ablations + Requirements + Response format.
            response = query(
                system_message=ablation_prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            # Parse the response
            ablation_name, ablation_description = _parse_keyword_prefix_response(
                response, "ABLATION NAME:", "ABLATION DESCRIPTION:"
            )
            if ablation_name and ablation_description:
                return AblationIdea(
                    name=ablation_name, description=ablation_description
                )

            retry_count += 1
            logger.warning(
                f"Failed to parse ablation response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse ablation response after {retry_limit} retries. Falling back to default idea of removing dropout."
        )
        return AblationIdea(name="add one more layer", description="add one more layer")

    def _get_leaves(self, node: Node) -> List[Node]:
        """Get all leaf nodes in the subtree rooted at node."""
        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves

    def _select_parallel_nodes(self) -> List[Optional[Node]]:
        """Select N nodes to process in parallel,
        balancing between tree exploration and exploitation.
        Note:
        - This function runs in the main process.
        Some design considerations:
        - For Stage 2 and 4, we generate nodes in the main process and
        send them to worker processes.
        This is to make sure we don't run duplicate ideas in parallel.
        - For Stage 1 and 3, we generate nodes in worker processes.
        """
        nodes_to_process = []
        processed_trees = set()
        search_cfg = self.cfg.agent.search
        print(f"[cyan]self.num_workers: {self.num_workers}, [/cyan]")

        while len(nodes_to_process) < self.num_workers:
            # Initial drafting phase, creating root nodes
            print(
                f"Checking draft nodes... num of journal.draft_nodes: {len(self.journal.draft_nodes)}, search_cfg.num_drafts: {search_cfg.num_drafts}"
            )
            if len(self.journal.draft_nodes) < search_cfg.num_drafts:
                nodes_to_process.append(None)
                continue

            # Get viable trees
            viable_trees = [
                root
                for root in self.journal.draft_nodes
                if not all(leaf.is_buggy for leaf in self._get_leaves(root))
            ]

            # Debugging phase (with some probability)
            if random.random() < search_cfg.debug_prob:
                print("Checking debuggable nodes")
                # print(f"Buggy nodes: {self.journal.buggy_nodes}")
                try:
                    debuggable_nodes = None
                    print("Checking buggy nodes...")
                    buggy_nodes = self.journal.buggy_nodes
                    print(f"Type of buggy_nodes: {type(buggy_nodes)}")
                    print(f"Length of buggy_nodes: {len(buggy_nodes)}")

                    for i, n in enumerate(buggy_nodes):
                        if not isinstance(n, Node):
                            print(f"Found non-Node object in journal.buggy_nodes: {n}")
                            raise ValueError(
                                "Found non-Node object in journal.buggy_nodes"
                            )
                    debuggable_nodes = [
                        n
                        for n in self.journal.buggy_nodes
                        if (
                            isinstance(n, Node)
                            and n.is_leaf
                            and n.debug_depth <= search_cfg.max_debug_depth
                        )
                    ]
                except Exception as e:
                    print(f"Error getting debuggable nodes: {e}")
                if debuggable_nodes:
                    print("Found debuggable nodes")
                    node = random.choice(debuggable_nodes)
                    tree_root = node
                    while tree_root.parent:
                        tree_root = tree_root.parent

                    tree_id = id(tree_root)
                    if tree_id not in processed_trees or len(processed_trees) >= len(
                        viable_trees
                    ):
                        nodes_to_process.append(node)
                        processed_trees.add(tree_id)
                        continue

            # Special handling for Stage 4 (Ablation Studies)
            print(f"[red]self.stage_name: {self.stage_name}[/red]")
            # print(f"[red]self.best_stage3_node: {self.best_stage3_node}[/red]")
            if self.stage_name and self.stage_name.startswith("4_"):
                nodes_to_process.append(self.best_stage3_node)
                continue
            # Special handling for Stage 2 (Hyperparam tuning for baseline)
            elif self.stage_name and self.stage_name.startswith("2_"):
                nodes_to_process.append(self.best_stage1_node)
                continue
            else:  # Stage 1, 3 (normal best-first search)
                # Improvement phase
                print("Checking good nodes..")
                good_nodes = self.journal.good_nodes
                if not good_nodes:
                    nodes_to_process.append(None)  # Back to drafting
                    continue

                # Get best node from unprocessed tree if possible
                # Note: best_node_selection does not use memory context as root branch has no data
                best_node = self.journal.get_best_node(cfg=self.cfg)
                tree_root = best_node
                while tree_root.parent:
                    tree_root = tree_root.parent

                tree_id = id(tree_root)
                if tree_id not in processed_trees or len(processed_trees) >= len(
                    viable_trees
                ):
                    nodes_to_process.append(best_node)
                    processed_trees.add(tree_id)
                    continue

                # If we can't use best node (tree already processed), try next best nodes
                for node in sorted(good_nodes, key=lambda n: n.metric, reverse=True):
                    tree_root = node
                    while tree_root.parent:
                        tree_root = tree_root.parent
                    tree_id = id(tree_root)
                    if tree_id not in processed_trees or len(processed_trees) >= len(
                        viable_trees
                    ):
                        nodes_to_process.append(node)
                        processed_trees.add(tree_id)
                        break

        return nodes_to_process

    def step(self, exec_callback: ExecCallbackType):
        print("Selecting nodes to process")
        nodes_to_process = self._select_parallel_nodes()
        print(f"Selected nodes: {[n.id if n else None for n in nodes_to_process]}")

        # Convert nodes to dicts
        node_data_list = []
        for node in nodes_to_process:
            if node:
                try:
                    node_data = node.to_dict()
                    _safe_pickle_test(node_data, f"node {node.id} data")
                    node_data_list.append(node_data)
                except Exception as e:
                    logger.error(f"Error preparing node {node.id}: {str(e)}")
                    raise
            else:
                node_data_list.append(None)  # None means new draft

        # NOTE: root branch has no memory data, so we skip memory context for journal_summary.
        if self.cfg.agent.get("summary", None) is not None:
            memory_summary = self.journal.generate_summary(
                include_code=False,
                memory_context=None,
                **{
                    "model": self.cfg.agent.summary.model,
                    "temp": self.cfg.agent.summary.temp,
                },
            )
        else:
            memory_summary = self.journal.generate_summary(
                include_code=False, memory_context=None
            )

        print("Submitting tasks to workers")
        task_ids = []
        task_id_to_index: Dict[int, int] = {}  # Map task_id to original index
        for node_data in node_data_list:
            gpu_id = None
            worker_idx = len(task_ids)
            if self.gpu_manager is not None:
                try:
                    # Get current process ID for GPU assignment
                    process_id = f"worker_{worker_idx}"
                    gpu_id = self.gpu_manager.acquire_gpu(process_id)
                    logger.info(f"Assigned GPU {gpu_id} to process {process_id}")
                except RuntimeError as e:
                    logger.warning(f"Could not acquire GPU: {e}. Running on CPU")

            if (
                self.stage_name
                and self.stage_name.startswith("2_")
                and node_data["is_buggy"] is False
            ):
                new_hyperparam_idea = self._generate_hyperparam_tuning_idea()
                self._hyperparam_tuning_state["tried_hyperparams"].add(
                    new_hyperparam_idea.name
                )
                new_ablation_idea = None
            elif (
                self.stage_name
                and self.stage_name.startswith("4_")
                and node_data["is_buggy"] is False
            ):
                new_ablation_idea = self._generate_ablation_idea()
                self._ablation_state["completed_ablations"].add(new_ablation_idea.name)
                new_hyperparam_idea = None
            else:
                new_ablation_idea = None
                new_hyperparam_idea = None

            best_stage1_plot_code = (
                self.best_stage1_node.plot_code if self.best_stage1_node else None
            )
            best_stage2_plot_code = (
                self.best_stage2_node.plot_code if self.best_stage2_node else None
            )
            best_stage3_plot_code = (
                self.best_stage3_node.plot_code if self.best_stage3_node else None
            )
            seed_eval = False
            task_id = self.worker_manager.submit(
                self._process_node_wrapper,
                node_data,
                self.task_desc,
                self.cfg,
                gpu_id,
                memory_summary,
                self.evaluation_metrics,
                self.stage_name,
                new_ablation_idea,
                new_hyperparam_idea,
                best_stage1_plot_code,
                best_stage2_plot_code,
                best_stage3_plot_code,
                seed_eval,
                None,
                worker_idx,
                self.root_branch_id,
                self._writer_queue,  # Pass centralized writer queue
            )
            task_ids.append(task_id)
            task_id_to_index[task_id] = worker_idx

        # Add results to journal
        print("Waiting for results")

        # Wait for results with timeout
        results = self.worker_manager.wait_for_results(task_ids, timeout=self.timeout)
        completed_indices = set()

        # Process completed results
        for task_id, worker_result in results.items():
            i = task_id_to_index[task_id]
            completed_indices.add(i)
            try:
                print(f"About to process result from worker {i}")
                if worker_result.error:
                    raise worker_result.error
                result_data = worker_result.result
                if "metric" in result_data:
                    print(f"metric type: {type(result_data['metric'])}")
                    print(f"metric contents: {result_data['metric']}")

                # Create node and restore relationships using journal.
                # Journal acts as a database to look up a parent node,
                # and add the result node as a child.
                result_node = Node.from_dict(result_data, self.journal)
                print("[red]Investigating if result node has metric[/red]", flush=True)
                print(result_node.metric)
                # Update hyperparam tuning state if in Stage 2
                self._update_hyperparam_tuning_state(result_node)
                # Update ablation state if in Stage 4
                self._update_ablation_state(result_node)

                # Add node to journal's list and assign its step number
                self.journal.append(result_node)
                print("Added result node to journal")

            except Exception as e:
                print(f"Error processing node from worker {i}: {escape(str(e))}")
                logger.error(f"Error processing node from worker {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # Release GPU for this process if it was using one
                process_id = f"worker_{i}"
                if (
                    self.gpu_manager is not None
                    and process_id in self.gpu_manager.gpu_assignments
                ):
                    self.gpu_manager.release_gpu(process_id)
                    logger.info(f"Released GPU for process {process_id}")

        # Check if any tasks timed out
        timed_out_indices = set(range(len(task_ids))) - completed_indices
        if timed_out_indices:
            print(f"Overall timeout of {self.timeout} seconds reached")
            logger.error(f"Overall timeout of {self.timeout} seconds reached, processing incomplete tasks")

            # Terminate and restart workers to clean up timed out processes
            self.worker_manager.terminate_and_restart()

        # Process any tasks that timed out
        for i in range(len(task_ids)):
            if i in completed_indices:
                continue

            print(f"Worker process {i} timed out, couldn't get the result")
            logger.error(f"Worker process {i} timed out after {self.timeout} seconds")

            # Create error node and add to journal
            parent_node = nodes_to_process[i] if i < len(nodes_to_process) else None
            node_data = node_data_list[i] if i < len(node_data_list) else None

            # Try to load intermediate result from worker's workspace
            intermediate_result_path = Path(self.cfg.workspace_dir) / f"worker_{i}" / "intermediate_result.json"
            intermediate_data = None
            if intermediate_result_path.exists():
                try:
                    intermediate_data = json.loads(intermediate_result_path.read_text(encoding="utf-8"))
                    logger.info(f"Loaded intermediate result from {intermediate_result_path} (stage: {intermediate_data.get('stage', 'unknown')})")
                except Exception as exc:
                    logger.warning(f"Failed to load intermediate result from {intermediate_result_path}: {exc}")

            def _normalize_text(value, fallback):
                if value is None:
                    return fallback
                text = str(value).strip()
                return text if text else fallback

            # Prioritize intermediate result data over parent node data
            if intermediate_data:
                plan_source = intermediate_data.get("plan")
                code_source = intermediate_data.get("code")
                analysis_source = intermediate_data.get("analysis")
                term_out_source = intermediate_data.get("term_out", [])
                exec_time_source = intermediate_data.get("exec_time")
                branch_hint = intermediate_data.get("branch_id") or (
                    parent_node.branch_id
                    if parent_node and getattr(parent_node, "branch_id", None)
                    else self.root_branch_id
                )
                # Try to restore metric from intermediate result
                metric_data = intermediate_data.get("metric")
            else:
                plan_source = (
                    parent_node.plan
                    if parent_node
                    else node_data.get("plan") if isinstance(node_data, dict) else None
                )
                code_source = (
                    parent_node.code
                    if parent_node
                    else node_data.get("code") if isinstance(node_data, dict) else None
                )
                analysis_source = None
                term_out_source = []
                exec_time_source = None
                branch_hint = (
                    parent_node.branch_id
                    if parent_node and getattr(parent_node, "branch_id", None)
                    else self.root_branch_id
                )
                metric_data = None

            plan_fallback = (
                "Plan unavailable because execution timed out before the plan completed."
            )
            code_fallback = (
                "Code unavailable because execution timed out before the implementation completed."
            )
            plan_text = _normalize_text(plan_source, plan_fallback)
            code_text = _normalize_text(code_source, code_fallback)

            def _snippet(value):
                if len(value) <= 400:
                    return value
                return value[:400].rstrip() + "..."

            plan_snippet = _snippet(plan_text)
            code_snippet = _snippet(code_text)
            memory_context = self._memory_context(branch_hint, "timeout_summary")
            memory_snippet = (
                memory_context.strip()
                if memory_context and memory_context.strip()
                else "No memory history recorded before this timeout."
            )
            if len(memory_snippet) > 400:
                memory_snippet = memory_snippet[:400].rstrip() + "..."
            stage_label = self.stage_name or "unknown stage"
            intermediate_stage = intermediate_data.get("stage", "unknown") if intermediate_data else "not_started"
            timeout_summary = (
                f"Timeout after {humanize.naturaldelta(self.timeout)} while running {stage_label} "
                f"(worker stopped at stage: {intermediate_stage}). "
                f"Plan snapshot: {plan_snippet}. "
                f"Code snapshot: {code_snippet}. "
                f"Memory context: {memory_snippet}."
            )

            # Combine term_out from intermediate result with timeout message
            combined_term_out = list(term_out_source) if term_out_source else []
            combined_term_out.append(f"\n[TIMEOUT] {timeout_summary}")

            exec_result = ExecutionResult(
                term_out=combined_term_out,
                exec_time=exec_time_source if exec_time_source else self.timeout,
                exc_type="TimeoutError",
                exc_info={
                    "message": f"Worker process timed out after {self.timeout} seconds.",
                    "memory_context": memory_snippet,
                    "intermediate_stage": intermediate_stage,
                },
                exc_stack=[],
            )

            # Create a new branch in memory DB for the timeout error node
            # branch_hint is the parent branch (from intermediate data, parent node, or root)
            parent_branch_id = branch_hint
            error_branch_id = uuid.uuid4().hex
            if self.memory_manager:
                try:
                    # Build ancestor chain for correct tree structure
                    ancestor_chain = []
                    if parent_node:
                        current = parent_node
                        while current is not None:
                            node_id = getattr(current, "branch_id", None) or current.id
                            ancestor_chain.append(node_id)
                            current = getattr(current, "parent", None)
                        ancestor_chain = ancestor_chain[::-1]  # root-to-parent order

                    self.memory_manager.mem_node_fork(
                        parent_branch_id, error_branch_id, ancestor_chain=ancestor_chain
                    )
                    logger.info(f"Created memory branch {error_branch_id} for timeout error node (parent: {parent_branch_id})")
                except Exception as exc:
                    logger.warning(f"Failed to create memory branch for timeout error node: {exc}")
                    # Fall back to using parent's branch_id if branch creation fails
                    error_branch_id = parent_branch_id

            error_node = Node(
                plan=plan_text,
                code=code_text,
                exc_type="TimeoutError",
                exc_info={
                    "message": f"Worker process timed out after {self.timeout} seconds.",
                    "memory_context": memory_snippet,
                    "intermediate_stage": intermediate_stage,
                },
                is_buggy=True,
                parent=parent_node,
                metric=WorstMetricValue(),
                analysis=analysis_source if analysis_source else timeout_summary,
                exec_time_feedback=(
                    f"Execution stopped at {humanize.naturaldelta(self.timeout)} because of the timeout."
                ),
                branch_id=error_branch_id,
            )
            # Copy additional fields from intermediate result if available
            if intermediate_data:
                if intermediate_data.get("plot_plan"):
                    error_node.plot_plan = intermediate_data["plot_plan"]
                if intermediate_data.get("plot_code"):
                    error_node.plot_code = intermediate_data["plot_code"]
                if intermediate_data.get("ablation_name"):
                    error_node.ablation_name = intermediate_data["ablation_name"]
                if intermediate_data.get("hyperparam_name"):
                    error_node.hyperparam_name = intermediate_data["hyperparam_name"]

            error_node.absorb_exec_result(exec_result)
            review_summary, review_is_bug = self._run_execution_review_for_timeout(
                error_node, error_branch_id
            )
            final_analysis = error_node.analysis or timeout_summary
            if review_summary:
                final_analysis += f"\n\nLLM execution review:\n{review_summary}"
                error_node._term_out.append(f"\nLLM review:\n{review_summary}")
            error_node.analysis = final_analysis
            error_node.is_buggy = error_node.is_buggy or review_is_bug
            if parent_node:
                parent_node.children.add(error_node)
            self.journal.append(error_node)
            logger.error(f"Added timeout error node {error_node.id} to journal (parent: {parent_node.id if parent_node else None}, intermediate_stage: {intermediate_stage})")

            # Update the branch's node_uid in memory DB to link it with the error node
            if self.memory_manager and error_branch_id != parent_branch_id:
                try:
                    self.memory_manager.update_branch_node_uid(error_branch_id, error_node.id)
                    logger.info(f"Updated memory branch {error_branch_id} with node_uid {error_node.id}")
                except Exception as exc:
                    logger.warning(f"Failed to update branch node_uid for timeout error node: {exc}")

            # Release GPU for this process if it was using one
            process_id = f"worker_{i}"
            if (
                self.gpu_manager is not None
                and process_id in self.gpu_manager.gpu_assignments
            ):
                self.gpu_manager.release_gpu(process_id)
                logger.info(f"Released GPU for process {process_id}")

    def _update_hyperparam_tuning_state(self, result_node: Node):
        """Update hyperparam tuning tracking state based on execution results."""
        if not self.stage_name or not self.stage_name.startswith("2_"):
            return

        hyperparam_name = result_node.hyperparam_name
        if hyperparam_name is None:
            print(
                f"[red]hyperparam_name is None for result_node: {result_node.id}[/red]"
            )
            return

        if not result_node.is_buggy:
            self._hyperparam_tuning_state["tried_hyperparams"].add(hyperparam_name)
            logger.info(f"Hyperparam tuning {hyperparam_name} ran successfully")
        else:
            logger.warning(f"Hyperparam tuning {hyperparam_name} failed")

    def _update_ablation_state(self, result_node: Node):
        """Update ablation tracking state based on execution results.

        Args:
            result_node: Node containing ablation execution results
        """
        if not self.stage_name or not self.stage_name.startswith("4_"):
            return

        ablation_name = result_node.ablation_name
        if ablation_name is None:
            print(f"[red]ablation_name is None for result_node: {result_node.id}[/red]")
            return

        if not result_node.is_buggy:
            self._ablation_state["completed_ablations"].add(ablation_name)
            logger.info(f"Ablation {ablation_name} completed successfully")

    def _aggregate_seed_eval_results(
        self, seed_nodes: List[Node], parent_node: Node
    ) -> str:
        """Generate aggregated plots from multi-seed evaluation results.

        Args:
            seed_nodes: List of nodes from seed evaluation
            parent_node: The original node that was evaluated

        Returns:
            str: The plotting code for aggregated results
        """
        prompt_guideline = list(SEED_PLOTTING_GUIDELINE_BASE)
        prompt_guideline.extend(SEED_PLOTTING_GUIDELINE_TAIL)
        # add instruction for format
        plotting_prompt = {
            "Introduction": SEED_PLOTTING_PROMPT_INTRO,
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= {
            "Response format": SEED_PLOTTING_PROMPT_RESPONSE
        }
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }
        plotting_prompt["Instructions"] |= {
            "Plotting code reference": (
                "plotting code 1:\n" + seed_nodes[0].plot_code + "\n\n"
                "plotting code 2:\n" + seed_nodes[1].plot_code + "\n\n"
                "plotting code 3:\n" + seed_nodes[2].plot_code + "\n\n"
            ),
        }
        memory_context = self._memory_context(
            getattr(parent_node, "branch_id", None) or self.root_branch_id,
            "seed_plotting",
        )
        if memory_context:
            plotting_prompt["Memory"] = memory_context
        seed_data_paths = []
        for seed_node in seed_nodes:
            exp_dir = Path(seed_node.exp_results_dir) if seed_node.exp_results_dir else None
            if not exp_dir:
                logger.warning(f"Seed node {seed_node.id} has no exp_results_dir")
                continue
            # Resolve to absolute path to avoid cwd-dependent resolution
            if not exp_dir.is_absolute():
                exp_dir = Path(os.getcwd()) / exp_dir
            if exp_dir.exists():
                npy_files = sorted(exp_dir.rglob("*.npy"))
                if npy_files:
                    # Only include files that actually exist
                    valid_paths = [str(p.resolve()) for p in npy_files if p.is_file()]
                    seed_data_paths.extend(valid_paths)
                else:
                    logger.warning(f"No .npy files found in {exp_dir} for seed node {seed_node.id}")
            else:
                logger.warning(f"exp_results_dir does not exist: {exp_dir} for seed node {seed_node.id}")
        if seed_data_paths:
            plotting_prompt["Instructions"] |= {
                "Experiment Data Path": "\n".join(seed_data_paths)
            }
        # LLM context (seed aggregation plotting): Introduction + Instructions (Response format + plotting guidelines + prior plotting code + data paths).
        plan, code = self.plan_and_code_query(
            plotting_prompt, code_language="python"
        )

        print("[green]Plan:[/green]\n", plan)
        print(f"[green]Generated aggregated plotting code (before path injection):[/green]\n{code}")

        # Inject actual data paths into the generated code to ensure LLM-generated
        # code uses the correct paths, even if LLM didn't properly embed them
        if seed_data_paths:
            import re
            paths_str = ",\n        ".join(f'"{p}"' for p in seed_data_paths)
            paths_replacement = f"experiment_data_path_list = [\n        {paths_str}\n    ]"

            # Match various patterns of empty or placeholder experiment_data_path_list
            patterns = [
                r'experiment_data_path_list\s*=\s*\(\s*\[\s*#[^\]]*\]\s*\)',  # Tuple with comment
                r'experiment_data_path_list\s*=\s*\[\s*#[^\]]*\]',  # With comment
                r'experiment_data_path_list\s*=\s*\(\s*\[\s*\]\s*\)',  # Tuple wrapped empty
                r'experiment_data_path_list\s*=\s*\[\s*\]',  # Simple empty list
            ]
            injected = False
            for pattern in patterns:
                if re.search(pattern, code):
                    code = re.sub(pattern, paths_replacement, code, count=1)
                    injected = True
                    print(f"[yellow]Injected {len(seed_data_paths)} data paths into plotting code[/yellow]")
                    break

            if not injected:
                print("[yellow]Warning: Could not find experiment_data_path_list pattern to inject paths[/yellow]")
                print(f"[yellow]Available paths: {seed_data_paths}[/yellow]")

        print(f"[green]Final aggregated plotting code:[/green]\n{code}")

        # Validate syntax of generated code
        code = self._validate_and_fix_python_syntax(code)

        return code

    def _validate_and_fix_python_syntax(self, code: str, max_retries: int = 3) -> str:
        """Validate Python syntax and attempt to fix errors using LLM.

        Args:
            code: The generated Python code to validate
            max_retries: Maximum number of LLM fix attempts

        Returns:
            The validated (and possibly fixed) code

        Raises:
            SyntaxError: If the code cannot be fixed after max_retries
        """
        for attempt in range(max_retries):
            try:
                ast.parse(code)
                if attempt > 0:
                    print(f"[green]Syntax validated after {attempt} LLM fix(es)[/green]")
                return code
            except SyntaxError as e:
                print(f"[yellow]Syntax error detected (attempt {attempt + 1}/{max_retries}): {e}[/yellow]")

                # Show problematic code context
                lines = code.split('\n')
                start = max(0, e.lineno - 5) if e.lineno else 0
                end = min(len(lines), e.lineno + 3) if e.lineno else min(len(lines), 8)
                context_lines = []
                for i in range(start, end):
                    marker = ">>> " if e.lineno and i == e.lineno - 1 else "    "
                    context_lines.append(f"{marker}{i + 1}: {lines[i]}")
                error_context = '\n'.join(context_lines)
                print(f"[yellow]Problematic code context:[/yellow]\n{error_context}")

                # Attempt to fix using LLM
                fixed_code = self._llm_syntax_fix(code, e)
                if fixed_code is None or fixed_code == code:
                    print(f"[red]LLM could not fix syntax error: {e}[/red]")
                    raise
                code = fixed_code

        # Final validation
        ast.parse(code)
        return code

    def _llm_syntax_fix(self, code: str, error: SyntaxError) -> Optional[str]:
        """Use LLM to fix syntax errors in generated code.

        Args:
            code: The code with syntax error
            error: The SyntaxError exception

        Returns:
            Fixed code, or None if LLM could not fix it
        """
        # Build error context
        lines = code.split('\n')
        error_line = error.lineno - 1 if error.lineno else 0
        start = max(0, error_line - 10)
        end = min(len(lines), error_line + 5)

        numbered_lines = []
        for i in range(start, end):
            marker = ">>>" if i == error_line else "   "
            numbered_lines.append(f"{marker} {i + 1}: {lines[i]}")
        error_context = '\n'.join(numbered_lines)

        fix_prompt = {
            "Task": "Fix the Python syntax error in the code below.",
            "Error": {
                "type": "SyntaxError",
                "message": str(error.msg),
                "line": error.lineno,
                "offset": error.offset,
            },
            "Code context around error": error_context,
            "Full code": code,
            "Instructions": [
                "Analyze the syntax error and fix it.",
                "Return ONLY the complete fixed Python code.",
                "Do NOT include any explanation, just the code.",
                "Make minimal changes - only fix the syntax error.",
                "Common issues include: unmatched brackets in comments, missing colons, unclosed strings.",
                "Wrap the fixed code in ```python ... ``` markers.",
            ],
        }

        print(f"[yellow]Requesting LLM to fix syntax error...[/yellow]")

        try:
            response = query(
                system_message=fix_prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=0.0,  # Use low temperature for deterministic fix
            )

            fixed_code = extract_code(response, language="python")
            if fixed_code and fixed_code.strip():
                print(f"[green]LLM returned fixed code ({len(fixed_code)} chars)[/green]")
                return fixed_code
            else:
                print(f"[red]LLM response did not contain valid code[/red]")
                return None
        except Exception as e:
            print(f"[red]LLM syntax fix query failed: {e}[/red]")
            return None

    def __enter__(self):
        return self

    def cleanup(self):
        """Cleanup parallel workers and resources"""
        if not self._is_shutdown:
            print("Shutting down WorkerManager...")
            try:
                # Release all GPUs
                if self.gpu_manager is not None:
                    for process_id in list(self.gpu_manager.gpu_assignments.keys()):
                        self.gpu_manager.release_gpu(process_id)

                # Shutdown WorkerManager (terminates all worker processes)
                self.worker_manager.shutdown(wait=False)

                print("WorkerManager shutdown complete")

                # Shutdown centralized database writer process
                if self._db_writer is not None:
                    print("Shutting down DatabaseWriterProcess...")
                    try:
                        self._db_writer.shutdown(timeout=30.0)
                        print("DatabaseWriterProcess shutdown complete")
                    except Exception as db_exc:
                        print(f"Error shutting down DatabaseWriterProcess: {db_exc}")
                    finally:
                        self._db_writer = None
                        self._writer_queue = None

            except Exception as e:
                print(f"Error during WorkerManager shutdown: {e}")
            finally:
                self._is_shutdown = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
