from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Optional, Set, Any, Callable, cast, Dict, Tuple
import json
import random
import os
import shutil
import re
import uuid
import copy
from queue import Queue
import logging
import humanize
import time
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
from .utils.phase_execution import (
    ExecutionEnvironment,
    SingularityWorkerContainer,
    collect_available_compilers,
    collect_available_libs,
    collect_system_performance_tools,
    collect_installed_system_packages,
    summarize_text,
    summarize_command_output,
    run_in_container,
)
from .utils.phase_plan import (
    PhasePlanError,
    apply_workspace_plan,
    combine_sources_for_display,
    extract_phase_artifacts,
    wrap_combined_code,
)
from .utils.resource import (
    ResourceConfig,
    load_resources,
    build_local_binds,
    build_resources_context,
    resolve_resources_path,
)
from .worker_plan import resolve_worker_plan
import pickle
import ast
import numpy as np
from dataclasses import asdict
from ai_scientist.prompt_loader import load_prompt, load_prompt_lines, format_prompt
from ai_scientist.memory import MemoryManager
from ai_scientist.memory.resource_memory import (
    build_resource_snapshot,
    update_resource_snapshot_if_changed,
)

from rich import print
from rich.markup import escape
from pathlib import Path
import base64
import sys
import filelock


logger = logging.getLogger("ai-scientist")

ExecCallbackType = Callable[[str, bool], ExecutionResult]


PROMPT_BASE = "agent/parallel/"

BASE_SYSTEM_PROMPT = load_prompt("core/system").rstrip("\n")
DOMAIN_NEUTRAL_PROMPT = load_prompt("core/domain_neutral").rstrip("\n")
ENVIRONMENT_INJECTION_TEMPLATE = load_prompt("config/environment/injection").rstrip("\n")
AI_OPTIONAL_PROMPT = load_prompt("core/ai_optional").rstrip("\n")
PHASE1_ITERATIVE_INSTALLER_PROMPT = load_prompt("config/phases/phase1_installer").rstrip("\n")
PHASE0_WHOLE_PLANNING_PROMPT = load_prompt("config/phases/phase0_planning").rstrip("\n")
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
DRAFT_EXP_GUIDELINES = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/draft/experiment_design_sketch_guideline")
)

DEBUG_INTRO = load_prompt(PROMPT_BASE + "tasks/debug/introduction").rstrip("\n")
DEBUG_BUGFIX_GUIDELINES = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/debug/bugfix_improvement_sketch_guideline")
)

IMPROVE_INTRO = load_prompt(PROMPT_BASE + "tasks/improve/introduction").rstrip("\n")

HYPERPARAM_NODE_INTRO_PREFIX = load_prompt(
    PROMPT_BASE + "nodes/hyperparam/introduction"
).rstrip("\n")
HYPERPARAM_NODE_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "nodes/hyperparam/instructions")
)

ABLATION_NODE_INTRO_PREFIX = load_prompt(
    PROMPT_BASE + "nodes/ablation/introduction"
).rstrip("\n")
ABLATION_NODE_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "nodes/ablation/instructions")
)

EXECUTION_REVIEW_INTRO = load_prompt(
    PROMPT_BASE + "tasks/execution_review/introduction"
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

DEFINE_METRICS_INTRO = load_prompt(
    PROMPT_BASE + "tasks/define_metrics/introduction"
).rstrip("\n")
DEFINE_METRICS_INSTRUCTIONS = tuple(
    load_prompt_lines(PROMPT_BASE + "tasks/define_metrics/instructions")
)

PARSE_METRICS_INTRO = load_prompt(
    PROMPT_BASE + "tasks/parse_metrics/introduction"
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
SEED_INJECTION_PROMPT = load_prompt(PROMPT_BASE + "seed_injection").rstrip("\n")


def _normalize_language(language: str | None) -> str:
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


def _find_previous_run_dir(current_log_dir: Path) -> Path | None:
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


def _resolve_run_root(cfg: Config) -> Path:
    run_root_env = os.environ.get("AI_SCIENTIST_RUN_ROOT")
    if run_root_env:
        run_root = Path(run_root_env).expanduser().resolve()
    else:
        run_root = Path(cfg.log_dir).parent.parent / "runs" / cfg.exp_name
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _copy_artifact(src: Path, dest_dir: Path, *, name: str | None = None) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (name or src.name)
    try:
        shutil.copy2(src, dest_path)
    except OSError:
        pass


def _format_prompt_log_name(label: str, *, session_id: str | None = None, counter: int | None = None) -> str:
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
    meta: dict[str, Any] | None = None,
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
    phase_log_dir: Path | None,
    run_root: Path | None,
    worker_label: str,
    phase_artifacts: dict | None = None,
    phase_artifacts_raw: str | None = None,
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
        prompt_root = run_root / "workers" / worker_label / "prompt_logs"
        for prompt_name in ("phase0_prompt.json", "phase0_prompt.md"):
            _copy_artifact(prompt_root / prompt_name, llm_outputs_dir / "prompt_logs")
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
    """Parse the response into name and description based on keyword prefix"""
    try:
        # Split response into lines and clean up
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        # Find the idea and description
        name = None
        description = None

        for line in lines:
            if line.startswith(keyword_prefix1):
                name = line.replace(keyword_prefix1, "").strip()
            elif line.startswith(keyword_prefix2):
                description = line.replace(keyword_prefix2, "").strip()
                # Combine any following lines that don't start with a marker
                desc_lines = []
                for next_line in lines[lines.index(line) + 1 :]:
                    if not next_line.startswith((keyword_prefix1, keyword_prefix2)):
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
        environment_context: dict | None = None,
        phase0_plan: dict | None = None,
        phase0_history: dict | None = None,
        prompt_log_dir: Path | None = None,
        prompt_session_id: str | None = None,
        memory_manager: Any | None = None,
        branch_id: str | None = None,
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
        self.resources_config: ResourceConfig | None = None
        self.resources_error: str | None = None
        self._resources_prompt_cache: dict[str, dict[str, Any]] = {}
        self.prompt_log_dir = prompt_log_dir
        self.prompt_session_id = prompt_session_id
        self._prompt_log_counter = 0
        resources_path = getattr(self.cfg.exec, "resources", None)
        if resources_path:
            try:
                self.resources_config = load_resources(resolve_resources_path(resources_path))
            except Exception as exc:
                self.resources_error = str(exc)

    def _memory_context(
        self,
        task_hint: str,
        branch_id: str | None = None,
        budget_chars: int | None = None,
        allow_summary_fallback: bool = False,
    ) -> str:
        if not self.memory_manager:
            return (self.memory_summary or "") if allow_summary_fallback else ""
        use_branch = branch_id or self.branch_id
        if not use_branch:
            return self.memory_summary or ""
        budget = budget_chars or getattr(getattr(self.cfg, "memory", None), "memory_budget_chars", 4000)
        return self.memory_manager.render_for_prompt(use_branch, task_hint, budget_chars=budget)

    def _inject_memory(
        self,
        prompt: dict[str, Any],
        task_hint: str,
        branch_id: str | None = None,
        budget_chars: int | None = None,
        allow_summary_fallback: bool = False,
        allow_empty: bool = False,
    ) -> None:
        memory_context = self._memory_context(
            task_hint,
            branch_id=branch_id,
            budget_chars=budget_chars,
            allow_summary_fallback=allow_summary_fallback,
        )
        if memory_context or allow_empty:
            prompt["Memory"] = memory_context

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
                "singularity_image": self.environment_context.get("singularity_image") or "none",
                "workspace_mount": self.environment_context.get("workspace_mount", "/workspace"),
                "cpu_info": self.environment_context.get("cpu_info", "unknown"),
                "memory_info": self.environment_context.get("memory_info", "unknown"),
                "gpu_info": self.environment_context.get("gpu_info", "unknown"),
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

    def _prompt_resources(self, phase: str) -> dict[str, Any] | None:
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

    def _log_prompt(self, prompt: Any, *, label: str, meta: dict[str, Any] | None = None) -> None:
        if not self.prompt_log_dir:
            return
        self._prompt_log_counter += 1
        name = _format_prompt_log_name(
            label,
            session_id=self.prompt_session_id,
            counter=self._prompt_log_counter,
        )
        payload: dict[str, Any] = {"stage": self.stage_name}
        if meta:
            payload.update(meta)
        _write_prompt_log(self.prompt_log_dir, name, prompt, meta=payload)

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
        if self.code_language == "python":
            impl_guideline.append(
                "Save experiment data to working/experiment_data.npy using np.save()."
            )
        else:
            impl_guideline.append(
                "Save experiment data to working/experiment_data.npy using cnpy or a compatible .npy writer."
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
                "  - Save metrics at the end using np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
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
    def _prompt_resp_fmt(self):
        if self.phase_mode == "split":
            return {"Response format": RESPONSE_FORMAT_SPLIT_PHASE}
        return {"Response format": self._format_response_format(RESPONSE_FORMAT_DEFAULT)}

    @property
    def _prompt_phase_guidance(self):
        if self.phase_mode != "split":
            return {}
        return {
            "Phase workflow": [
                "Use the 5-phase plan: Phase 0 whole planning, Phase 1 download/install, Phase 2 coding, Phase 3 compile, Phase 4 run.",
                "Phase 1 may use sudo/apt-get inside Singularity with writable-tmpfs/overlay; if unavailable install under /workspace.",
                "All paths are relative to /workspace; no absolute paths or parent traversal.",
                "In download/install, probe with `command -v ...`/`which ...` before installing (e.g., git, cmake); avoid redundant installs.",
                "compile commands must honor build_plan.compiler_selected chosen from available_compilers (do not invent compilers or switch).",
                "Run phase must generate /workspace/working/experiment_data.npy (Python uses numpy save; non-Python must use cnpy).",
                "Do not rely on language adapters, interpreter adapters, or external routers; commands run directly in the worker.",
            ]
        }

    def _phase0_plan_snippet(self, *, include_phase1: bool, include_phase2_4: bool) -> dict[str, Any] | None:
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
        prompt: Any = {
            "Introduction": DRAFT_INTRO,
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
        prompt: Any = {
            "Introduction": DEBUG_INTRO,
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
        prompt: Any = {
            "Introduction": IMPROVE_INTRO,
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
        intro_prefix = HYPERPARAM_NODE_INTRO_PREFIX
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
        intro_prefix = ABLATION_NODE_INTRO_PREFIX
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
                "compiler_selected": "python",
                "cflags": [],
                "ldflags": [],
                "workdir": "/workspace",
                "output": "working/experiment_data.npy",
            }
            compile_commands = []
            run_commands = ["python src/main.py"]
            compile_notes = "no compile needed for placeholder"

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
                    "expected_outputs": ["working/experiment_data.npy"],
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

    def generate_phase_artifacts(self, prompt, retries: int = 3, log_label: str = "phase2") -> dict:
        """Query the LLM for split-phase output and validate the JSON structure."""
        last_error = ""
        last_response = ""
        for attempt in range(1, retries + 1):
            self._log_prompt(
                prompt,
                label=f"{log_label}_attempt{attempt}",
                meta={
                    "phase": "phase2",
                    "attempt": attempt,
                    "label": log_label,
                    "model": self.cfg.agent.code.model,
                },
            )
            # LLM context (split-phase artifacts): system_message=prompt dict with task-specific sections, Instructions/Response format, optional System/Domain/Environment/Resources/Phase 0 snippet; request JSON for download/coding/compile/run.
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )
            try:
                self.last_phase_artifacts_response = completion_text
                artifacts = extract_phase_artifacts(
                    completion_text,
                    default_language=self.code_language,
                )
                self._validate_phase_language(artifacts)
                return artifacts
            except PhasePlanError as exc:
                last_error = str(exc)
                last_response = completion_text
                prompt["Parsing Feedback"] = (
                    "The previous response failed validation: "
                    f"{last_error}.\n"
                    "Raw response was:\n"
                    "<<<RAW_RESPONSE_START>>>\n"
                    f"{completion_text}\n"
                    "<<<RAW_RESPONSE_END>>>\n"
                    "Return strict JSON following the Response format with download/coding/compile/run."
                )
        # Fallback: return a minimal placeholder plan to keep execution moving
        if last_response:
            self.last_phase_artifacts_response = last_response
        return self._fallback_phase_artifacts(last_error)

    def plan_and_code_query(
        self,
        prompt,
        retries=3,
        code_language: str | None = None,
        log_label: str = "plan_and_code",
    ) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        target_language = code_language or self.code_language
        for attempt in range(1, retries + 1):
            self._log_prompt(
                prompt,
                label=f"{log_label}_attempt{attempt}",
                meta={
                    "phase": "single",
                    "attempt": attempt,
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

            code = extract_code(completion_text, language=target_language)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                f"The code extraction failed. Make sure to use the format ```{target_language} ... ``` for the code blocks."
            )
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def parse_exec_result(
        self, node: Node, exec_result: ExecutionResult, workspace: str
    ):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": EXECUTION_REVIEW_INTRO,
            "Research idea": self.task_desc,
            "Implementation": wrap_combined_code(node.code, fallback_lang=self.code_language),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        self._inject_memory(prompt, "execution_review", branch_id=getattr(node, "branch_id", None))

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
            self._inject_memory(
                prompt_select_plots,
                "plot_selection",
                branch_id=getattr(node, "branch_id", None),
                budget_chars=getattr(self.cfg.memory, "plot_selection_budget_chars", 1000),
            )

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
        memory_context = self._memory_context(
            "vlm_analysis",
            branch_id=getattr(node, "branch_id", None),
            budget_chars=getattr(self.cfg.memory, "vlm_analysis_budget_chars", 1000),
        )
        memory_context_block = (
            f"Memory:\n{memory_context}\n\n" if memory_context else ""
        )
        analysis_text = VLM_ANALYSIS_PROMPT_TEMPLATE.format(
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
        summary_prompt = {
            "Introduction": SUMMARY_INTRO,
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
        self._inject_memory(
            summary_prompt,
            "node_summary",
            branch_id=getattr(node, "branch_id", None),
            budget_chars=getattr(self.cfg.memory, "node_summary_budget_chars", 2000),
        )

        return cast(
            dict,
            # LLM context (node summary): Introduction + Research idea + Implementation + Plan + Execution output + Analysis + Metric + Plot analyses + VLM feedback + Memory.
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


class GPUManager:
    """Manages GPU allocation across processes"""

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus: Set[int] = set(range(num_gpus))
        self.gpu_assignments: Dict[str, int] = {}  # process_id -> gpu_id

    def acquire_gpu(self, process_id: str) -> int:
        """Assigns a GPU to a process"""
        if not self.available_gpus:
            raise RuntimeError("No GPUs available")
        print(f"Available GPUs: {self.available_gpus}")
        print(f"Process ID: {process_id}")
        preferred_id = None
        try:
            preferred_id = int(str(process_id).split("_")[-1])
        except Exception:
            preferred_id = None
        if preferred_id is not None and preferred_id in self.available_gpus:
            gpu_id = preferred_id
        else:
            gpu_id = min(self.available_gpus)
        print(f"Acquiring GPU {gpu_id} for process {process_id}")
        self.available_gpus.remove(gpu_id)
        self.gpu_assignments[process_id] = gpu_id
        print(f"GPU assignments: {self.gpu_assignments}")
        return gpu_id

    def release_gpu(self, process_id: str):
        """Releases GPU assigned to a process"""
        if process_id in self.gpu_assignments:
            gpu_id = self.gpu_assignments[process_id]
            self.available_gpus.add(gpu_id)
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
        memory_manager: Any | None = None,
        root_branch_id: str | None = None,
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

        self.gpu_manager = GPUManager(self.num_gpus) if self.num_gpus > 0 else None

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
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        logger.info(
            "Process pool initialized max_workers=%s requested=%s",
            getattr(self.executor, "_max_workers", self.num_workers),
            self.worker_plan.requested_workers if hasattr(self, "worker_plan") else self.num_workers,
        )
        self._is_shutdown = False
        # Define the metric once at initialization
        self.evaluation_metrics = self._define_global_metrics()
        self._ablation_state = {  # store ablation names
            "completed_ablations": set(),
        }
        self._hyperparam_tuning_state = {  # store hyperparam tuning ideas
            "tried_hyperparams": set(),
        }

    def _memory_context(self, branch_id: str | None, task_hint: str) -> str:
        if not self.memory_manager or not branch_id:
            return ""
        budget = getattr(getattr(self.cfg, "memory", None), "memory_budget_chars", 4000)
        return self.memory_manager.render_for_prompt(branch_id, task_hint, budget_chars=budget)

    def _run_execution_review_for_timeout(
        self, node: Node, branch_id: str | None
    ) -> tuple[str, bool]:
        """Run the execution review prompt for a timeout node to get LLM analysis."""
        prompt = {
            "Introduction": EXECUTION_REVIEW_INTRO,
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
        memory_context = self._memory_context(self.root_branch_id, "define_metrics")
        if memory_context:
            prompt["Memory"] = memory_context

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
        self, prompt, retries=3, code_language: str | None = None
    ) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        target_language = code_language or self.code_language
        for _ in range(retries):
            # LLM context (plan+code): system_message=prompt dict built by caller (task sections + instructions/format/env/resources as applicable).
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            code = extract_code(completion_text, language=target_language)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code
            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                f"The code extraction failed. Make sure to use the format ```{target_language} ... ``` for the code blocks."
            )
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

        # Convert node to dict for parallel processing
        node_data = node.to_dict()

        # Submit parallel jobs for different seeds
        seed_nodes = []
        futures = []
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
            futures.append(
                self.executor.submit(
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
                )
            )

        for seed, future in enumerate(futures):
            try:
                result_data = future.result(timeout=self.timeout)
                result_node = Node.from_dict(result_data, self.journal)
                print(f"Parent node id: {result_node.parent.id}")
                print(f"Sanity check: actual parent node id: {node.id}")
                # Add node to journal's list and assign its step number
                self.journal.append(result_node)
                seed_nodes.append(self.journal.get_node_by_id(result_node.id))
                print("Added result node to journal")
            except Exception as e:
                logger.error(f"Error in multi-seed evaluation: {str(e)}")
            finally:
                process_id = f"worker_{seed}"
                if (
                    self.gpu_manager is not None
                    and process_id in self.gpu_manager.gpu_assignments
                ):
                    self.gpu_manager.release_gpu(process_id)

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
                            plot_file.resolve().rename(final_path)
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
        seed: int | None = None,
        worker_id: int | None = None,
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

        memory_cfg = getattr(cfg, "memory", None)
        memory_manager: MemoryManager | None = None
        root_branch_id = getattr(memory_cfg, "root_branch_id", None) if memory_cfg else None
        if memory_cfg and getattr(memory_cfg, "enabled", False):
            db_path = getattr(memory_cfg, "db_path", None) or (
                Path(cfg.workspace_dir) / "memory" / "memory.sqlite"
            )
            if not getattr(memory_cfg, "run_id", None):
                memory_cfg.run_id = Path(cfg.workspace_dir).name
            memory_cfg.workspace_root = str(Path(cfg.workspace_dir))
            memory_cfg.ai_scientist_root = os.environ.get("AI_SCIENTIST_ROOT")
            memory_cfg.phase_mode = getattr(cfg.exec, "phase_mode", "single")
            memory_cfg.memory_log_dir = str(Path(cfg.log_dir) / "memory")
            memory_manager = MemoryManager(db_path, memory_cfg)
            if not root_branch_id:
                root_branch_id = uuid.uuid4().hex
                memory_manager.mem_node_fork(None, root_branch_id)
                memory_manager.update_branch_node_uid(root_branch_id, "root")
                memory_manager.set_root_branch_id(root_branch_id)
                try:
                    memory_cfg.root_branch_id = root_branch_id
                except Exception:
                    pass
            else:
                memory_manager.set_root_branch_id(root_branch_id)

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
                    parent_lock_path = parent_workspace / ".workspace.lock"
                    with filelock.FileLock(parent_lock_path, timeout=300):
                        def ignore_mounted_data(directory, contents):
                            # Skip 'data' only inside 'input' directory
                            if os.path.basename(directory) == "input":
                                return ["data"] if "data" in contents else []
                            return []

                        def rmtree_skip_data(directory: Path):
                            """Remove directory tree but skip 'data' inside 'input' directory."""
                            for subitem in directory.iterdir():
                                if directory.name == "input" and subitem.name == "data":
                                    continue  # Skip mounted data
                                if subitem.is_dir():
                                    shutil.rmtree(subitem)
                                else:
                                    subitem.unlink()

                        for item in parent_workspace.iterdir():
                            if item.name == ".workspace.lock":
                                continue  # Skip the lock file itself
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
                                    shutil.copytree(src, dst, ignore=ignore_mounted_data, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src, dst)
                            except Exception as exc:
                                logger.warning(f"Failed to copy {src} to {dst}: {exc}")

        worker_name = f"worker_{worker_id}" if worker_id is not None else f"worker_{multiprocessing.current_process().name}"
        print(f"Process {worker_name} using workspace: {workspace}")
        # Create process-specific working directory
        working_dir = os.path.join(workspace, "working")
        os.makedirs(working_dir, exist_ok=True)
        workspace_path = Path(workspace)
        if memory_manager:
            memory_manager.workspace_root = workspace_path
        resources_path = getattr(cfg.exec, "resources", None)
        resource_snapshot = None  # Will be saved after child_branch_id is created
        if memory_manager and resources_path:
            try:
                resource_snapshot = build_resource_snapshot(
                    resources_path,
                    workspace_root=workspace_path,
                    ai_scientist_root=os.environ.get("AI_SCIENTIST_ROOT"),
                    phase_mode=getattr(cfg.exec, "phase_mode", "single"),
                    log=logger,
                )
            except Exception as exc:
                logger.warning("Failed to build resource snapshot: %s", exc)
        phase_log_dir: Path | None = None
        prompt_session_id = f"{int(time.time() * 1000)}_{os.getpid()}"
        prompt_log_root: Path | None = None
        prompt_session_dir: Path | None = None

        def resolve_workdir(requested: str | None) -> Path:
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
        exec_env: ExecutionEnvironment | None = None
        if getattr(cfg.exec, "phase_mode", "single") == "split":
            image_path = getattr(cfg.exec, "singularity_image", None)
            environment_context = {
                "available_compilers": [],
                "available_libs": [],
                "container_runtime": "singularity" if image_path else None,
                "singularity_image": str(Path(image_path).resolve()) if image_path else None,
                "workspace_mount": getattr(cfg.exec, "workspace_mount", "/workspace"),
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
                        gpu_res = info_env.run(
                            ["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo 'nvidia-smi not available'"],
                            cwd=workspace_path,
                        )
                        environment_context["gpu_info"] = summarize_text(gpu_res.stdout, max_lines=20, max_chars=1200)
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

        run_root: Path | None = None
        phase0_plan: dict | None = None
        phase0_history_summary: dict | None = None
        phase0_artifact_paths: list[str] | None = None
        phase0_command_str: str | None = None
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
                        "memory_info": environment_context.get("memory_info", ""),
                        "gpu_info": environment_context.get("gpu_info", ""),
                        "available_compilers": environment_context.get("available_compilers", []),
                        "available_libs": environment_context.get("available_libs", []),
                        "system_performance_tools": environment_context.get("system_performance_tools", []),
                        "system_performance_tools_note": performance_tools_note_for_history,
                        "installed_system_packages": environment_context.get("installed_system_packages", []),
                        "network_access": environment_context.get("network_access", "unknown"),
                        "container_runtime": environment_context.get("container_runtime"),
                        "singularity_image": environment_context.get("singularity_image"),
                        "workspace_mount": environment_context.get("workspace_mount"),
                    },
                    indent=2,
                ),
                history_full_path=str(phase0_history_path),
            )

            if phase0_plan_path.exists():
                try:
                    phase0_plan = json.loads(phase0_plan_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Failed to load existing Phase 0 plan: %s", exc)
                    phase0_plan = None
            if phase0_plan is None:
                # Build verified system perf tools note
                performance_tools_list = environment_context.get("system_performance_tools", [])
                performance_tools_names = [t.get("name", "") for t in performance_tools_list if isinstance(t, dict) and t.get("name")]
                performance_tools_note = (
                    f"The following system performance tools have been verified and are confirmed functional inside this container: {', '.join(performance_tools_names)}. "
                    "Use them directly without assuming availability issues."
                ) if performance_tools_names else "No system performance tools detected in this environment."
                phase0_prompt: dict[str, Any] = {
                    "Introduction": PHASE0_WHOLE_PLANNING_PROMPT,
                    "Task": task_desc,
                    "History": history_injection,
                    "Environment": {
                        "os_release": environment_context.get("os_release", ""),
                        "available_compilers": environment_context.get("available_compilers", []),
                        "available_libs": environment_context.get("available_libs", []),
                        "system_performance_tools": environment_context.get("system_performance_tools", []),
                        "system_performance_tools_note": performance_tools_note,
                        "installed_system_packages": environment_context.get("installed_system_packages", []),
                        "network_access": environment_context.get("network_access", "unknown"),
                        "container_runtime": environment_context.get("container_runtime"),
                        "singularity_image": environment_context.get("singularity_image"),
                        "workspace_mount": environment_context.get("workspace_mount", "/workspace"),
                        "timeout_seconds": cfg.exec.timeout,
                    },
                }
                if memory_manager and root_branch_id:
                    budget = getattr(
                        getattr(cfg, "memory", None), "memory_budget_chars", 4000
                    )
                    memory_context = memory_manager.render_for_prompt(
                        root_branch_id, "phase0_planning", budget_chars=budget
                    )
                    if memory_context:
                        phase0_prompt["Memory"] = memory_context
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
                try:
                    # LLM context (Phase 0 planning): Introduction + Task + History (phase summaries/compile-run logs/errors/LLM outputs) + Environment snapshot + optional Resources.
                    phase0_response = query(
                        system_message=phase0_prompt,
                        user_message=None,
                        model=cfg.agent.code.model,
                        temperature=cfg.agent.code.temp,
                    )
                    try:
                        (plans_dir / "phase0_llm_output.txt").write_text(
                            phase0_response,
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        logger.warning("Failed to write Phase 0 raw output: %s", exc)
                    phase0_plan = _normalize_phase0_plan(
                        _parse_json_object(phase0_response, context="Phase 0 plan")
                    )
                except Exception as exc:
                    logger.warning("Phase 0 planning failed: %s", exc)
                    phase0_plan = _normalize_phase0_plan({})
                try:
                    phase0_plan_path.write_text(
                        json.dumps(phase0_plan, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.warning("Failed to write Phase 0 plan: %s", exc)
            if getattr(memory_cfg, "persist_phase0_internal", True):
                # Prepare phase0 artifacts for later ingestion (will be ingested per child node)
                phase0_artifact_paths = [str(phase0_plan_path), str(phase0_history_path)]
                llm_out = plans_dir / "phase0_llm_output.txt"
                if llm_out.exists():
                    phase0_artifact_paths.append(str(llm_out))

                def _extract_phase0_commands(plan: Any) -> str | None:
                    if not isinstance(plan, dict):
                        return None
                    phase_artifacts = plan.get("phase_artifacts") or plan.get("plan") or {}
                    if isinstance(phase_artifacts, dict):
                        commands: list[str] = []
                        for section in ("download", "compile", "run"):
                            section_data = phase_artifacts.get(section, {})
                            if isinstance(section_data, dict):
                                section_cmds = section_data.get("commands") or []
                                for cmd in section_cmds:
                                    if isinstance(cmd, list):
                                        commands.append(" ".join([str(c) for c in cmd]))
                                    else:
                                        commands.append(str(cmd))
                        return " | ".join(commands) if commands else None
                    return None

                phase0_command_str = _extract_phase0_commands(phase0_plan)
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

                # Save resource snapshot to child branch (not root branch)
                if resource_snapshot is not None:
                    try:
                        update_resource_snapshot_if_changed(
                            resource_snapshot, memory_manager, branch_id=child_branch_id
                        )
                    except Exception as exc:
                        logger.warning("Failed to persist resource snapshot to child branch: %s", exc)

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

            if memory_manager and child_branch_id and getattr(memory_cfg, "persist_idea_md", True):
                candidates = []
                desc_file = getattr(cfg, "desc_file", None)
                for candidate in (
                    desc_file,
                    Path(cfg.workspace_dir) / "idea.md",
                    workspace_path / "idea.md",
                    workspace_path / "working" / "idea.md",
                ):
                    if not candidate:
                        continue
                    path = Path(candidate)
                    if path.exists():
                        candidates.append(path)
                if candidates:
                    latest_path = max(candidates, key=lambda p: p.stat().st_mtime)
                    try:
                        memory_manager.ingest_idea_md(
                            child_branch_id, node_uid=child_node.id, idea_path=latest_path
                        )
                    except Exception as exc:
                        logger.warning("Failed to ingest idea.md: %s", exc)
            # Ingest Phase 0 internal info for this child node (not root branch)
            if memory_manager and child_branch_id and phase0_plan and phase0_artifact_paths:
                try:
                    memory_manager.ingest_phase0_internal_info(
                        child_branch_id,
                        node_uid=child_node.id,
                        phase0_payload=phase0_plan,
                        artifact_paths=phase0_artifact_paths,
                        command_str=phase0_command_str,
                    )
                except Exception as exc:
                    logger.warning("Failed to ingest Phase 0 internal info: %s", exc)
            if memory_manager and child_branch_id:
                try:
                    plan_snippet = (child_node.plan or "")[:1200]
                    memory_manager.mem_recall_append(
                        {
                            "ts": time.time(),
                            "run_id": memory_manager.run_id,
                            "node_id": child_node.id,
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
                    resources_config: ResourceConfig | None = None
                    resource_binds: list[str] = []
                    resources_path = getattr(cfg.exec, "resources", None)
                    if resources_path:
                        try:
                            resources_config = load_resources(resolve_resources_path(resources_path))
                            resource_binds = build_local_binds(resources_config)
                            term_outputs.append(f"Loaded resources from {resources_path}")
                        except Exception as exc:
                            term_outputs.append(f"Warning: Failed to load resources: {exc}")
                    
                    worker_container: SingularityWorkerContainer | None = None
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
                    active_env: ExecutionEnvironment | None = None
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
                    expected_outputs = run_section.get("expected_outputs") or ["working/experiment_data.npy"]
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
                        if "```" in cleaned:
                            start = cleaned.find("{")
                            end = cleaned.rfind("}")
                            if start != -1 and end != -1 and end > start:
                                cleaned = cleaned[start : end + 1]
                        try:
                            parsed = json.loads(cleaned)
                        except json.JSONDecodeError:
                            try:
                                import ast

                                parsed = ast.literal_eval(cleaned)
                            except Exception as exc:  # pragma: no cover
                                raise ValueError(f"Failed to parse Phase 1 response: {exc}") from exc
                        if not isinstance(parsed, dict):
                            raise ValueError("Phase 1 response must be a JSON object.")
                        return parsed

                    phase1_llm_log_path: Path | None = None

                    def phase1_iterative_driver(history: list[dict[str, Any]], step_idx: int, max_steps: int) -> dict[str, Any]:
                        prompt: dict[str, Any] = {
                            "Introduction": PHASE1_ITERATIVE_INSTALLER_PROMPT,
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
                        response_text = query(
                            system_message=prompt,
                            user_message=None,
                            model=cfg.agent.code.model,
                            temperature=cfg.agent.code.temp,
                        )
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
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception as exc:
                                logger.warning("Failed to write Phase 1 LLM output: %s", exc)
                        return parse_phase1_iterative_response(response_text)

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
                                if memory_manager and resources_path:
                                    try:
                                        memory_manager.mem_resources_resolve_and_refresh(
                                            memory_manager.run_id or ""
                                        )
                                    except Exception as exc:
                                        logger.warning("Failed to update resource snapshot after Phase 1: %s", exc)
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
                            except Exception as exc:
                                exc_type = "CodingError"
                                exc_info = {"message": str(exc)}
                            if exc_type is None:
                                if not selected_compiler:
                                    exc_type = "CompilationError"
                                    exc_info = {"message": "build_plan.compiler_selected is required."}
                                    term_outputs.append(exc_info["message"])
                                elif available_compiler_names and selected_compiler not in available_compiler_names:
                                    exc_type = "CompilationError"
                                    exc_info = {
                                        "message": f"compiler_selected '{selected_compiler}' not in available_compilers.",
                                        "available_compilers": available_compiler_names,
                                    }
                                    term_outputs.append(exc_info["message"])
                                else:
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
                                    else:
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
                        if isinstance(active_env, ExecutionEnvironment):
                            active_env.stop()
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
            expected_output_file = workspace_path / "working" / "experiment_data.npy"
            if expected_output_file.exists() and expected_output_file not in data_files:
                data_files.append(expected_output_file)
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
                        parse_metrics_prompt = {
                            "Introduction": PARSE_METRICS_INTRO,
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
                            worker_agent._inject_memory(
                                metrics_prompt,
                                "metrics_extraction",
                                branch_id=getattr(child_node, "branch_id", None),
                                budget_chars=getattr(cfg.memory, "metrics_extraction_budget_chars", 1500),
                            )
                            print(
                                f"[blue]Metrics_exec_result.term_out: {metrics_exec_result.term_out}[/blue]"
                            )
                            print(
                                f"[blue]Metrics Parsing Execution Result:\n[/blue] {metrics_exec_result}"
                            )

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
                        # Move experiment data files to experiment_results directory
                        for plots_dir in existing_plot_dirs:
                            for exp_data_file in plots_dir.glob("*.npy"):
                                exp_data_path = exp_results_dir / exp_data_file.name
                                exp_data_file.resolve().rename(exp_data_path)
                                logger.info(f"Saved experiment data to {exp_data_path}")

                        for plots_dir in existing_plot_dirs:
                            for plot_file in plots_dir.glob("*.png"):
                                # Get the base directory (parent of workspaces/logs)
                                base_dir = Path(cfg.workspace_dir).parent.parent
                                run_name = Path(cfg.workspace_dir).name

                                # Create the final path in logs directory
                                final_path = exp_results_dir / plot_file.name
                                plot_file.resolve().rename(final_path)

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
                            "phase": stage_name,
                            "kind": "node_result",
                            "summary": result_text,
                            "refs": [],
                        }
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

            # Set workspace_path on child node for cross-stage file inheritance
            child_node.workspace_path = str(workspace_path)

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
        memory_context = self._memory_context(self.root_branch_id, "hyperparam_idea")
        if memory_context:
            hyperparam_tuning_prompt["Memory"] = memory_context

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
        memory_context = self._memory_context(self.root_branch_id, "ablation_idea")
        if memory_context:
            ablation_prompt["Memory"] = memory_context

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
                memory_context = self._memory_context(self.root_branch_id, "best_node_selection")
                best_node = self.journal.get_best_node(
                    cfg=self.cfg, memory_context=memory_context
                )
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

        memory_context = self._memory_context(self.root_branch_id, "journal_summary")
        if self.cfg.agent.get("summary", None) is not None:
            memory_summary = self.journal.generate_summary(
                include_code=False,
                memory_context=memory_context,
                **{
                    "model": self.cfg.agent.summary.model,
                    "temp": self.cfg.agent.summary.temp,
                },
            )
        else:
            memory_summary = self.journal.generate_summary(
                include_code=False, memory_context=memory_context
            )

        print("Submitting tasks to process pool")
        futures = []
        for node_data in node_data_list:
            gpu_id = None
            worker_idx = len(futures)
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
            futures.append(
                self.executor.submit(
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
                )
            )

        # Add results to journal
        print("Waiting for results")

        # Create mapping from future to index for proper error handling
        future_to_index = {future: i for i, future in enumerate(futures)}
        completed_indices = set()

        # Use as_completed with overall timeout to process results as they complete
        try:
            for future in as_completed(futures, timeout=self.timeout):
                i = future_to_index[future]
                completed_indices.add(i)
                try:
                    print(f"About to get result from future (worker {i})")
                    result_data = future.result()  # Already completed, no timeout needed
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

        except FuturesTimeoutError:
            # Handle futures that didn't complete within the timeout
            print(f"Overall timeout of {self.timeout} seconds reached")
            logger.error(f"Overall timeout of {self.timeout} seconds reached, processing incomplete futures")

        # Process any futures that timed out
        for i, future in enumerate(futures):
            if i in completed_indices:
                continue

            print(f"Worker process {i} timed out, couldn't get the result")
            logger.error(f"Worker process {i} timed out after {self.timeout} seconds")

            # Cancel the future (may not stop already-running process in ProcessPoolExecutor)
            future.cancel()

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
                branch_id=branch_hint,
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
                error_node, branch_hint
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
            if exp_dir and exp_dir.exists():
                npy_files = sorted(exp_dir.rglob("*.npy"))
                if npy_files:
                    seed_data_paths.extend(str(p) for p in npy_files)
                # Skip if no .npy files found - don't add non-existent paths
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

    def _llm_syntax_fix(self, code: str, error: SyntaxError) -> str | None:
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
            print("Shutting down parallel executor...")
            try:
                # Release all GPUs
                if self.gpu_manager is not None:
                    for process_id in list(self.gpu_manager.gpu_assignments.keys()):
                        self.gpu_manager.release_gpu(process_id)

                # Shutdown executor first
                self.executor.shutdown(wait=False, cancel_futures=True)

                # Force terminate all worker processes
                if self.executor._processes:
                    ## Get copy of processes
                    processes = list(self.executor._processes.values())

                    # Then terminate processes if they're still alive
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=1)

                print("Executor shutdown complete")

            except Exception as e:
                print(f"Error during executor shutdown: {e}")
            finally:
                self._is_shutdown = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
