from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence

from .resource_memory import (
    RESOURCE_DIGEST_KEY,
    RESOURCE_INDEX_JSON_KEY,
    RESOURCE_INDEX_KEY,
    RESOURCE_ITEM_TAG,
    RESOURCE_USED_KEY,
    RESOURCE_USED_TAG,
    build_resource_snapshot,
    track_resource_usage,
    update_resource_snapshot_if_changed,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _now_ts() -> float:
    return time.time()


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = [tags]
    elif isinstance(tags, (list, tuple, set)):
        raw = list(tags)
    else:
        raw = [str(tags)]
    cleaned = []
    for tag in raw:
        tag_str = str(tag).strip()
        if tag_str:
            cleaned.append(tag_str)
    return cleaned


_SECRET_PATTERNS = [
    re.compile(r"(?i)(api_key|secret|token|password)\s*[:=]\s*([^\s]+)"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
]


def _redact(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}=***" if m.groups() else "***", redacted)
    return redacted


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# LLM compression cache: {(text_hash, max_chars, context_hint): compressed_text}
_compression_cache: dict[tuple[str, int, str], str] = {}

# Default character budgets for each memory section
DEFAULT_SECTION_BUDGETS: dict[str, int] = {
    "idea_summary": 800,
    "idea_section_limit": 400,
    "phase0_summary": 800,
    "archival_snippet": 400,
    "results": 600,
}

# Fixed compression prompt file path
COMPRESSION_PROMPT_FILE = "prompt/config/memory/compression.txt"


def _load_compression_prompt(prompt_file: str | Path | None) -> str | None:
    """Load compression prompt template from file."""
    if not prompt_file:
        return None
    path = Path(prompt_file)
    if not path.is_absolute():
        # Try relative to ai_scientist root
        for candidate in [
            Path(__file__).parent.parent.parent / prompt_file,
            Path.cwd() / prompt_file,
        ]:
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        logger.warning("Compression prompt file not found: %s", prompt_file)
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read compression prompt: %s", exc)
        return None


def _compress_with_llm(
    text: str,
    max_chars: int,
    context_hint: str,
    *,
    client: Any = None,
    model: str | None = None,
    prompt_template: str | None = None,
    use_cache: bool = True,
    max_iterations: int = 1,
    memory_context: str = "",
) -> str:
    """
    Compress text using LLM to fit within max_chars while preserving key information.
    Falls back to simple truncation on errors or if LLM is not available.

    Args:
        text: Original text to compress
        max_chars: Target maximum character count
        context_hint: Description of what the text represents (e.g., "idea summary", "phase0 config")
        client: LLM client (optional, will be created if not provided)
        model: LLM model name (optional, defaults to gpt-4o-mini)
        prompt_template: Prompt template string with {text}, {max_chars}, {current_chars}, {context_hint}, {memory_context} placeholders
        use_cache: Whether to cache compression results
        max_iterations: Maximum number of compression attempts
        memory_context: Optional experiment context from memory (research goals, config)

    Returns:
        Compressed text fitting within max_chars
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    # Check cache (include memory_context in cache key for context-aware caching)
    cache_key = (sha256(text.encode()).hexdigest()[:16], max_chars, context_hint, memory_context[:100] if memory_context else "")
    if use_cache and cache_key in _compression_cache:
        cached = _compression_cache[cache_key]
        if len(cached) <= max_chars:
            return cached

    # Fallback if no client/model provided
    if client is None or model is None:
        return _truncate(text, max_chars)

    # Load prompt template if not provided
    if prompt_template is None:
        prompt_template = _load_compression_prompt(COMPRESSION_PROMPT_FILE)
        if prompt_template is None:
            prompt_template = (
                "Compress the following {context_hint} text to fit within {max_chars} characters. "
                "Preserve key information, facts, and metrics. Output only the compressed text.\n\n"
                "{memory_context_section}"
                "Text:\n{text}"
            )

    try:
        # Import here to avoid circular dependency
        from ai_scientist.llm import get_response_from_llm

        current_text = text
        for i in range(max_iterations):
            # If it already fits, break
            if len(current_text) <= max_chars:
                break

            # Build memory context section for prompt
            memory_context_section = ""
            if memory_context:
                memory_context_section = f"Experiment Context:\n{memory_context}\n\n"

            prompt = prompt_template.format(
                text=current_text,
                max_chars=max_chars,
                current_chars=len(current_text),
                context_hint=context_hint,
                memory_context=memory_context,
                memory_context_section=memory_context_section,
            )
            
            system_message = "You are a text compression assistant. Output only the compressed text, no explanations."
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=model,
                system_message=system_message,
                temperature=0.3,
            )
            
            if not response:
                logger.warning(f"LLM returned empty response for compression (iter {i+1}).")
                break

            response = response.strip()
            # Safety check: if LLM returns longer text, stop to avoid infinite expansion
            if len(response) >= len(current_text) and len(response) > max_chars:
                logger.warning(f"LLM compression resulted in longer/equal text size (iter {i+1}). Stopping.")
                break

            current_text = response
            
        # Final check
        if len(current_text) > max_chars:
            logger.warning(
                f"LLM compression failed to meet target {max_chars} chars after {max_iterations} attempts. "
                "Falling back to truncation."
            )
            result = _truncate(current_text, max_chars)
        else:
            result = current_text
            
        # Update cache
        if use_cache:
            _compression_cache[cache_key] = result
        
        return result

    except Exception as exc:
        logger.warning("LLM compression failed, falling back to truncation: %s", exc)
        return _truncate(text, max_chars)


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=True)
        return value
    except Exception:
        return str(value)


def _core_lines(entries: dict[str, str]) -> list[str]:
    return [f"{key}: {value}" for key, value in sorted(entries.items(), key=lambda kv: kv[0])]


def _core_size(entries: dict[str, str]) -> int:
    return len("\n".join(_core_lines(entries)))


def _coerce_importance(value: Any) -> int:
    try:
        importance = int(value)
    except Exception:
        importance = 3
    return max(1, min(5, importance))


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("\n", 1)
    if len(parts) < 2:
        return text
    rest = parts[1]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return text
    return rest[end_idx + len("\n---") :].lstrip()


def _extract_resource_ids(tags_json: str | None) -> list[str]:
    try:
        tags = json.loads(tags_json) if tags_json else []
    except json.JSONDecodeError:
        tags = []
    ids: list[str] = []
    for tag in tags:
        tag_str = str(tag)
        if tag_str.startswith("resource_id:"):
            ids.append(tag_str.split("resource_id:", 1)[1])
    if not ids:
        for tag in tags:
            tag_str = str(tag)
            if tag_str.startswith("resource:"):
                ids.append(tag_str)
    return ids


def _parse_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("### ") and current is None:
            current = line[4:].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _summarize_idea(
    text: str,
    max_chars: int = 800,
    compress_fn: Any = None,
    section_limit: int = 400,
) -> str:
    """
    Summarize idea markdown content into bullet points.

    Args:
        text: Raw idea markdown text
        max_chars: Maximum characters for final summary
        compress_fn: Optional compression function (text, max_chars, context) -> str
                     If provided, uses LLM compression; otherwise uses truncation
        section_limit: Maximum characters per section bullet (default 400)
    """
    sections = _parse_markdown_sections(text)
    purpose = sections.get("Abstract") or sections.get("Task goal") or ""
    hypothesis = sections.get("Short Hypothesis") or sections.get("Hypothesis") or ""
    method = sections.get("Experiments") or sections.get("Code") or ""
    evaluation = sections.get("Task evaluation") or sections.get("Evaluation") or ""
    risks = sections.get("Risk Factors And Limitations") or sections.get(
        "Risk Factors and Limitations"
    ) or ""

    def pick(src: str, limit: int = section_limit, context: str = "section") -> str:
        cleaned = " ".join(src.split())
        if compress_fn is not None and len(cleaned) > limit:
            return compress_fn(cleaned, limit, f"idea {context}")
        return _truncate(cleaned, limit)

    bullets = [
        f"- Purpose: {pick(purpose, context='purpose')}" if purpose else "- Purpose: (not provided)",
        f"- Hypothesis: {pick(hypothesis, context='hypothesis')}" if hypothesis else "- Hypothesis: (not provided)",
        f"- Method/Variables: {pick(method, context='method')}" if method else "- Method/Variables: (not provided)",
        f"- Evaluation: {pick(evaluation, context='evaluation')}" if evaluation else "- Evaluation: (not provided)",
        f"- Known failures/mitigations: {pick(risks, context='risks')}" if risks else "- Known failures/mitigations: (not provided)",
    ]
    summary = "\n".join(bullets)
    if compress_fn is not None and len(summary) > max_chars:
        return compress_fn(summary, max_chars, "idea summary")
    return _truncate(summary, max_chars)


def _extract_cpu_info_from_lscpu(cpu_lines: list[str]) -> tuple[list[str], bool]:
    """Extract CPU info from lscpu format."""
    items = []
    extracted = False
    for line in cpu_lines:
        line_lower = line.lower()
        if "model name" in line_lower:
            items.append(line.strip().replace("Model name:", "CPU:").strip())
            extracted = True
        elif line_lower.startswith("cpu(s):"):
            items.append(line.strip())
            extracted = True
        elif "socket(s)" in line_lower:
            items.append(line.strip())
            extracted = True
        elif "numa node(s)" in line_lower:
            items.append(line.strip())
            extracted = True
    return items, extracted


def _extract_cpu_info_from_condensed(cpu_info: str) -> list[str]:
    """Extract CPU info from condensed semicolon format."""
    cpu_summary_parts = []
    # Extract CPU model
    for segment in cpu_info.split(";"):
        segment = segment.strip()
        segment_lower = segment.lower()
        if any(x in segment_lower for x in ["processor", "epyc", "xeon", "core", "intel", "amd"]):
            if "online" not in segment_lower:
                cpu_summary_parts.append(f"CPU:{segment}")
                break
    # Extract socket/core info
    for segment in cpu_info.split(";"):
        segment = segment.strip()
        segment_lower = segment.lower()
        if "socket" in segment_lower or "core" in segment_lower:
            if "processor" not in segment_lower:
                cpu_summary_parts.append(segment)
                break
    # Extract NUMA info
    for segment in cpu_info.split(";"):
        segment = segment.strip()
        if "numa" in segment.lower():
            if len(segment) > 50:
                segment = segment[:50] + "..."
            cpu_summary_parts.append(segment)
            break
    return cpu_summary_parts


def _extract_cpu_info(cpu_info: str | dict) -> list[str]:
    """Extract CPU information from various formats."""
    if not cpu_info:
        return []

    # Handle dict format
    if isinstance(cpu_info, dict):
        items = []
        # Common keys for CPU info dictionaries
        key_mapping = {
            "model_name": "CPU",
            "model": "CPU",
            "cpu_model": "CPU",
            "cores": "cores",
            "cpu_cores": "cores",
            "threads": "threads",
            "sockets": "sockets",
            "architecture": "arch",
        }
        for key, label in key_mapping.items():
            if key in cpu_info and cpu_info[key]:
                items.append(f"{label}={cpu_info[key]}")
        # If no mapped keys found, extract all primitive values
        if not items:
            for key, value in cpu_info.items():
                if isinstance(value, (str, int, float)) and value:
                    items.append(f"{key}={value}")
        return items

    # Handle string format
    cpu_lines = cpu_info.split("\\n") if "\\n" in cpu_info else cpu_info.split("\n")
    items, lscpu_extracted = _extract_cpu_info_from_lscpu(cpu_lines)

    if not lscpu_extracted and cpu_info:
        items = _extract_cpu_info_from_condensed(cpu_info)

    return items


def _extract_os_info(os_release: str | dict) -> str | None:
    """Extract OS information from os_release string or dict."""
    if not os_release:
        return None

    # Handle dict format
    if isinstance(os_release, dict):
        # Try common keys for OS name
        for key in ["PRETTY_NAME", "pretty_name", "NAME", "name", "os_name"]:
            if key in os_release and os_release[key]:
                return f"OS={os_release[key]}"
        # Fallback: use first string value found
        for value in os_release.values():
            if isinstance(value, str) and value:
                return f"OS={value}"
        return None

    # Handle string format
    # Try PRETTY_NAME format first
    for line in os_release.split("\\n") if "\\n" in os_release else os_release.split("\n"):
        if "PRETTY_NAME" in line:
            os_name = line.replace("PRETTY_NAME=", "").replace('"', '').strip()
            return f"OS={os_name}"

    # If not PRETTY_NAME format, use the string directly
    if "PRETTY_NAME" not in os_release:
        return f"OS={os_release.strip()}"

    return None


def _extract_compiler_info(compilers: list) -> str | None:
    """Extract compiler information."""
    if not compilers:
        return None

    compiler_strs = []
    for c in compilers[:3]:  # Limit to first 3
        if isinstance(c, dict):
            name = c.get("name", "")
            version = c.get("version", "").split()[0] if c.get("version") else ""
            if name:
                compiler_strs.append(f"{name}:{version}" if version else name)

    if compiler_strs:
        return f"compilers=[{', '.join(compiler_strs)}]"
    return None


def _extract_environment_context(payload: dict, build_plan: dict | None) -> dict | None:
    """Extract environment_context from payload, build_plan, or artifacts."""
    env_ctx = payload.get("environment_context")
    if env_ctx:
        return env_ctx

    if isinstance(build_plan, dict):
        env_ctx = build_plan.get("environment_context")
        if env_ctx:
            return env_ctx

    # Search in artifacts array
    artifacts = payload.get("artifacts", [])
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        content_str = artifact.get("content", "")
        if not content_str or not isinstance(content_str, str):
            continue
        try:
            content_obj = json.loads(content_str)
            if isinstance(content_obj, dict):
                env_ctx = content_obj.get("environment_context")
                if env_ctx:
                    return env_ctx
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _summarize_phase0(
    payload: Any,
    command_str: str | None,
    max_chars: int = 800,
    compress_fn: Any = None,
) -> str:
    """
    Summarize Phase 0 configuration payload.

    Args:
        payload: Phase 0 configuration dictionary
        command_str: Command string if available
        max_chars: Maximum characters for summary
        compress_fn: Optional compression function (text, max_chars, context) -> str
    """
    items: list[str] = []

    # Keys to skip (typically large nested structures or redundant data)
    skip_keys = {"environment_context", "env_context", "plan", "build_plan", "cpu_info", "os_release", "available_compilers"}

    if isinstance(payload, dict):
        # Extract all primitive-type key-value pairs from top level
        for key, value in payload.items():
            if key in skip_keys:
                continue
            # Include primitive types and simple collections
            if isinstance(value, (str, int, float, bool)):
                items.append(f"{key}={value}")
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, (str, int, float, bool)) for v in value):
                items.append(f"{key}={value}")

        # Extract nested build_plan if present
        build_plan = payload.get("plan") or payload.get("build_plan")
        if isinstance(build_plan, dict):
            for key, value in build_plan.items():
                if isinstance(value, (str, int, float, bool)):
                    items.append(f"{key}={value}")

        # Extract environment context
        env_ctx = _extract_environment_context(payload, build_plan)

        if isinstance(env_ctx, dict):
            # Extract CPU info
            cpu_info_items = _extract_cpu_info(env_ctx.get("cpu_info", ""))
            items.extend(cpu_info_items)

            # Extract CPU governor
            cpu_governor = env_ctx.get("cpu_governor", "")
            if cpu_governor and cpu_governor != "NA":
                items.append(f"cpu_governor={cpu_governor}")

            # Extract NUMA config (summarized)
            numa_config = env_ctx.get("numa_config", "")
            if numa_config and numa_config != "NA" and isinstance(numa_config, str):
                # Just extract key NUMA info
                numa_lines = numa_config.split("\n")
                for line in numa_lines[:5]:
                    if "node" in line.lower() or "cpu" in line.lower():
                        items.append(f"numa={line.strip()}")
                        break

            # Extract OS info
            os_info = _extract_os_info(env_ctx.get("os_release", ""))
            if os_info:
                items.append(os_info)

            # Extract compiler info
            compiler_info = _extract_compiler_info(env_ctx.get("available_compilers", []))
            if compiler_info:
                items.append(compiler_info)

            # Container runtime
            container = env_ctx.get("container_runtime", "")
            if container:
                items.append(f"container={container}")

            # Container digest
            container_digest = env_ctx.get("container_digest", "")
            if container_digest and container_digest != "NA":
                items.append(f"container_digest={container_digest[:20]}...")

            # Parallel env vars (summarized)
            parallel_env_vars = env_ctx.get("parallel_env_vars", "")
            if parallel_env_vars and parallel_env_vars != "NA" and parallel_env_vars != "none":
                # Extract key variables
                for line in parallel_env_vars.split("\n"):
                    line = line.strip()
                    if line.startswith("OMP_NUM_THREADS="):
                        items.append(line)
                    elif line.startswith("OMP_PROC_BIND="):
                        items.append(line)
                    elif line.startswith("MKL_NUM_THREADS="):
                        items.append(line)
                    elif line.startswith("OPENBLAS_NUM_THREADS="):
                        items.append(line)
    if command_str:
        items.append(f"command={command_str}")
    if not items:
        items.append("No structured Phase 0 info captured.")
    result = " | ".join(items)
    if compress_fn is not None and len(result) > max_chars:
        return compress_fn(result, max_chars, "phase0 configuration")
    return _truncate(result, max_chars)


class MemoryManager:
    def __init__(self, db_path: str | Path, config: Any | None = None):
        self.config = config or {}
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = _cfg_get(self.config, "run_id", None)
        self.run_id = str(run_id).strip() if run_id else None
        workspace_root = _cfg_get(self.config, "workspace_root", None)
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        ai_scientist_root = _cfg_get(self.config, "ai_scientist_root", None)
        self.ai_scientist_root = (
            Path(ai_scientist_root).resolve() if ai_scientist_root else None
        )
        self.root_branch_id = _cfg_get(self.config, "root_branch_id", None)
        self.core_max_chars = int(_cfg_get(self.config, "core_max_chars", 2000))
        self.archival_max_chars = int(_cfg_get(self.config, "archival_max_chars", 8000))  # Max chars per archival entry
        self.recall_max_events = int(_cfg_get(self.config, "recall_max_events", 20))
        self.retrieval_k = int(_cfg_get(self.config, "retrieval_k", 8))
        self.use_fts_setting = str(_cfg_get(self.config, "use_fts", "auto")).lower()
        self.max_compression_iterations = int(
            _cfg_get(self.config, "max_compression_iterations", 3)
        )
        self.always_inject_phase0_summary = bool(
            _cfg_get(self.config, "always_inject_phase0_summary", True)
        )
        self.always_inject_idea_summary = bool(
            _cfg_get(self.config, "always_inject_idea_summary", True)
        )
        self.redact_secrets = bool(_cfg_get(self.config, "redact_secrets", True))
        self.memory_log_enabled = bool(_cfg_get(self.config, "memory_log_enabled", True))
        self.memory_log_max_chars = int(_cfg_get(self.config, "memory_log_max_chars", 400))
        self.memory_log_dir: Path | None = None
        self.memory_log_path: Path | None = None
        log_dir = _cfg_get(self.config, "memory_log_dir", None)
        if not log_dir:
            base_log_dir = _cfg_get(self.config, "log_dir", None)
            if base_log_dir:
                log_dir = str(Path(base_log_dir) / "memory")
        if log_dir:
            try:
                self.memory_log_dir = Path(log_dir)
                self.memory_log_dir.mkdir(parents=True, exist_ok=True)
                self.memory_log_path = self.memory_log_dir / "memory_calls.jsonl"
            except Exception:
                self.memory_log_dir = None
                self.memory_log_path = None
        
        # LLM compression settings
        self.use_llm_compression = bool(
            _cfg_get(self.config, "use_llm_compression", False)
        )
        self.compression_model = str(
            _cfg_get(self.config, "compression_model", "gpt-4o-mini")
        )
        self._compression_prompt_template = _load_compression_prompt(
            COMPRESSION_PROMPT_FILE
        )
        
        # Section-specific character budgets (merge config with defaults)
        section_budgets_cfg = _cfg_get(self.config, "section_budgets", {}) or {}
        self.section_budgets = {
            key: int(_cfg_get(section_budgets_cfg, key, default))
            for key, default in DEFAULT_SECTION_BUDGETS.items()
        }

        # Writeup memory limits (for final_writeup_memory.json)
        self.writeup_recall_limit = int(_cfg_get(self.config, "writeup_recall_limit", 10))
        self.writeup_archival_limit = int(_cfg_get(self.config, "writeup_archival_limit", 10))
        self.writeup_core_value_max_chars = int(_cfg_get(self.config, "writeup_core_value_max_chars", 500))
        self.writeup_recall_text_max_chars = int(_cfg_get(self.config, "writeup_recall_text_max_chars", 300))
        self.writeup_archival_text_max_chars = int(_cfg_get(self.config, "writeup_archival_text_max_chars", 400))

        # Memory pressure management settings
        pressure_thresholds_cfg = _cfg_get(self.config, "pressure_thresholds", {}) or {}
        self.pressure_thresholds = {
            "medium": float(_cfg_get(pressure_thresholds_cfg, "medium", 0.7)),
            "high": float(_cfg_get(pressure_thresholds_cfg, "high", 0.85)),
            "critical": float(_cfg_get(pressure_thresholds_cfg, "critical", 0.95)),
        }
        self.auto_consolidate = bool(_cfg_get(self.config, "auto_consolidate", True))
        self.consolidation_trigger = str(_cfg_get(self.config, "consolidation_trigger", "high"))
        self.recall_consolidation_threshold = float(
            _cfg_get(self.config, "recall_consolidation_threshold", 1.5)
        )  # Consolidate when recall_count > recall_max_events * threshold

        self._compression_client: Any = None
        self._compression_model_name: str | None = None
        
        # Initialize compression client lazily when first needed
        if self.use_llm_compression:
            try:
                from ai_scientist.llm import create_client
                self._compression_client, self._compression_model_name = create_client(
                    self.compression_model
                )
            except Exception as exc:
                logger.warning("Failed to create compression LLM client: %s", exc)
                self.use_llm_compression = False
        
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=30, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._fts_enabled = self._init_fts()

    def _build_memory_context(self, branch_id: str | None, max_context_chars: int = 800) -> str:
        """
        Build memory context from core memory for enhanced compression.

        Fetches idea_md_summary and phase0_summary from core memory to provide
        experiment context during compression, helping the LLM preserve
        information relevant to the research goals.

        Args:
            branch_id: Branch ID to fetch memory from
            max_context_chars: Maximum characters for the context string

        Returns:
            Memory context string (may be empty if no relevant memory found)
        """
        if not branch_id:
            return ""

        context_parts = []

        # Fetch idea summary (research goals)
        try:
            idea_summary = self.get_core(branch_id, "idea_md_summary", log_event=False)
            if idea_summary:
                # Truncate to reasonable size for context
                truncated_idea = _truncate(idea_summary, max_context_chars // 2)
                context_parts.append(f"Research Goal: {truncated_idea}")
        except Exception as exc:
            logger.debug("Failed to fetch idea_md_summary for compression context: %s", exc)

        # Fetch phase0 summary (experiment configuration)
        try:
            phase0_summary = self.get_core(branch_id, "phase0_summary", log_event=False)
            if phase0_summary:
                truncated_phase0 = _truncate(phase0_summary, max_context_chars // 2)
                context_parts.append(f"Experiment Config: {truncated_phase0}")
        except Exception as exc:
            logger.debug("Failed to fetch phase0_summary for compression context: %s", exc)

        return "\n".join(context_parts)

    def _compress(
        self,
        text: str,
        max_chars: int,
        context_hint: str,
        *,
        branch_id: str | None = None,
    ) -> str:
        """
        Compress text using LLM if enabled, otherwise fall back to simple truncation.

        When branch_id is provided and LLM compression is enabled, fetches memory
        context (idea_md_summary, phase0_summary) to help the LLM preserve
        information relevant to the research goals.

        Args:
            text: Original text to compress
            max_chars: Target maximum character count
            context_hint: Description of what the text represents
            branch_id: Optional branch ID to fetch memory context from

        Returns:
            Compressed or truncated text fitting within max_chars
        """
        if not self.use_llm_compression or self._compression_client is None:
            return _truncate(text, max_chars)

        # Build enhanced context with memory information
        memory_context = ""
        if branch_id:
            memory_context = self._build_memory_context(branch_id)

        return _compress_with_llm(
            text=text,
            max_chars=max_chars,
            context_hint=context_hint,
            client=self._compression_client,
            model=self._compression_model_name,
            prompt_template=self._compression_prompt_template,
            use_cache=True,
            max_iterations=self.max_compression_iterations,
            memory_context=memory_context,
        )

    def _sanitize_detail_value(self, value: Any) -> Any:
        if isinstance(value, str):
            redacted = _redact(value, self.redact_secrets)
            return _truncate(redacted, self.memory_log_max_chars)
        if isinstance(value, dict):
            return {k: self._sanitize_detail_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_detail_value(v) for v in value]
        return _safe_json_value(value)

    def _log_memory_event(
        self,
        op: str,
        memory_type: str,
        *,
        branch_id: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.memory_log_enabled or not self.memory_log_path:
            return
        payload: dict[str, Any] = {
            "ts": _now_ts(),
            "op": op,
            "memory_type": memory_type,
            "branch_id": branch_id,
            "node_id": node_id,
            "phase": phase,
            "run_id": self.run_id,
        }
        if details:
            payload["details"] = self._sanitize_detail_value(details)
        try:
            self.memory_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.memory_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning("Failed to write memory log event: %s", exc)

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                parent_id TEXT NULL,
                node_uid TEXT NULL,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS core_kv (
                branch_id TEXT,
                key TEXT,
                value TEXT,
                updated_at REAL,
                PRIMARY KEY (branch_id, key)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS core_meta (
                branch_id TEXT,
                key TEXT,
                importance INTEGER,
                ttl TEXT,
                updated_at REAL,
                PRIMARY KEY (branch_id, key)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_id TEXT,
                kind TEXT,
                text TEXT,
                tags TEXT,
                created_at REAL,
                task_hint TEXT,
                memory_size INTEGER
            )
            """
        )
        # Migration: add task_hint and memory_size columns if they don't exist (for existing DBs)
        try:
            cur.execute("ALTER TABLE events ADD COLUMN task_hint TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cur.execute("ALTER TABLE events ADD COLUMN memory_size INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS archival (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_id TEXT,
                text TEXT,
                tags TEXT,
                created_at REAL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_archival_branch ON archival(branch_id)"
        )
        self._conn.commit()

    def _init_fts(self) -> bool:
        if self.use_fts_setting == "off":
            return False
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS archival_fts USING fts5(text, tags, branch_id)"
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError as exc:
            if self.use_fts_setting == "on":
                raise RuntimeError("FTS5 requested but unavailable") from exc
            logger.warning("FTS5 unavailable, falling back to keyword search.")
            return False

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def create_branch(
        self,
        parent_branch_id: str | None,
        node_uid: str | None,
        branch_id: str | None = None,
    ) -> str:
        branch_id = branch_id or uuid.uuid4().hex
        existing = self._conn.execute(
            "SELECT id FROM branches WHERE id=?", (branch_id,)
        ).fetchone()
        if existing:
            return branch_id
        self._conn.execute(
            "INSERT INTO branches (id, parent_id, node_uid, created_at) VALUES (?, ?, ?, ?)",
            (branch_id, parent_branch_id, node_uid, _now_ts()),
        )
        self._conn.commit()
        if parent_branch_id is None and not self.root_branch_id:
            self.set_root_branch_id(branch_id)
        return branch_id

    def update_branch_node_uid(self, branch_id: str, node_uid: str) -> None:
        self._conn.execute(
            "UPDATE branches SET node_uid=? WHERE id=?",
            (node_uid, branch_id),
        )
        self._conn.commit()

    def set_root_branch_id(self, branch_id: str) -> None:
        self.root_branch_id = branch_id
        try:
            if hasattr(self.config, "root_branch_id"):
                setattr(self.config, "root_branch_id", branch_id)
        except Exception:
            pass

    def _default_branch_id(self) -> str | None:
        root = self.root_branch_id
        if root:
            return root
        return _cfg_get(self.config, "root_branch_id", None)

    def _branch_exists(self, branch_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM branches WHERE id=?", (branch_id,)
        ).fetchone()
        return bool(row)

    def _resolve_branch_id(self, node_id: str | None) -> str | None:
        if not node_id:
            return None
        if self._branch_exists(node_id):
            return node_id
        row = self._conn.execute(
            "SELECT id FROM branches WHERE node_uid=?", (node_id,)
        ).fetchone()
        return row["id"] if row else None

    def _branch_chain(self, branch_id: str) -> list[str]:
        chain = []
        seen = set()
        current = branch_id
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            row = self._conn.execute(
                "SELECT parent_id FROM branches WHERE id=?", (current,)
            ).fetchone()
            current = row["parent_id"] if row else None
        return chain

    def _set_core_value(self, branch_id: str, key: str, value: str) -> None:
        text = _redact(str(value), self.redact_secrets)
        self._conn.execute(
            "INSERT OR REPLACE INTO core_kv (branch_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
            (branch_id, key, text, _now_ts()),
        )

    def _set_core_meta(self, branch_id: str, key: str, *, importance: int, ttl: str | None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO core_meta (branch_id, key, importance, ttl, updated_at) VALUES (?, ?, ?, ?, ?)",
            (branch_id, key, int(importance), ttl, _now_ts()),
        )

    def _delete_core_key(
        self,
        branch_id: str,
        key: str,
        *,
        reason: str | None = None,
        op_name: str | None = None,
        phase: str | None = None,
        node_id: str | None = None,
        log_event: bool = True,
    ) -> None:
        self._conn.execute(
            "DELETE FROM core_kv WHERE branch_id=? AND key=?",
            (branch_id, key),
        )
        self._conn.execute(
            "DELETE FROM core_meta WHERE branch_id=? AND key=?",
            (branch_id, key),
        )
        if log_event:
            self._log_memory_event(
                op_name or "core_delete",
                "core",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={"key": key, "reason": reason or ""},
            )

    def _enforce_core_budget(self, branch_id: str) -> None:
        if self.core_max_chars <= 0:
            return
        rows = self._conn.execute(
            "SELECT key, value FROM core_kv WHERE branch_id=?",
            (branch_id,),
        ).fetchall()
        entries = {row["key"]: row["value"] for row in rows}
        if not entries:
            return
        current_size = _core_size(entries)
        if current_size <= self.core_max_chars:
            return

        meta_rows = self._conn.execute(
            "SELECT key, importance FROM core_meta WHERE branch_id=?",
            (branch_id,),
        ).fetchall()
        importance_map = {row["key"]: _coerce_importance(row["importance"]) for row in meta_rows}
        protected = {
            RESOURCE_INDEX_KEY,
            RESOURCE_DIGEST_KEY,
            "idea_md_summary",
            "phase0_summary",
        }
        removable_keys = [
            (importance_map.get(key, 3), key)
            for key in entries
            if key not in protected
        ]
        removable_keys.sort(key=lambda pair: (pair[0], pair[1]))
        for _, key in removable_keys:
            value = entries.pop(key, "")
            if value:
                try:
                    self._insert_archival(
                        branch_id,
                        f"Core evicted: {key}\n{value}",
                        tags=[f"CORE_EVICT", f"core_key:{key}"],
                    )
                except Exception:
                    pass
            self._delete_core_key(
                branch_id,
                key,
                reason="evict",
                op_name="core_evict",
            )
            current_size = _core_size(entries)
            if current_size <= self.core_max_chars:
                self._conn.commit()
                return

        # Still too large; summarize remaining core into a digest.
        snapshot_text = "\n".join(_core_lines(entries))
        record_id = None
        if snapshot_text:
            try:
                record_id = self._insert_archival(
                    branch_id,
                    snapshot_text,
                    tags=["CORE_SNAPSHOT", f"branch_id:{branch_id}"],
                )
            except Exception:
                record_id = None
        resource_index = entries.get(RESOURCE_INDEX_KEY, "")
        for key in list(entries.keys()):
            if key == RESOURCE_INDEX_KEY:
                continue
            self._delete_core_key(
                branch_id,
                key,
                reason="digest",
                op_name="core_digest_compact",
            )
        available = self.core_max_chars - len(resource_index)
        if resource_index:
            available -= len("CoreDigest: ")
        digest_body = self._compress(snapshot_text, max(0, available), "core memory digest")
        if digest_body:
            digest_value = digest_body
            if record_id:
                digest_value = f"{digest_body}\n(ref: {record_id})"
            self._set_core_value(branch_id, "CoreDigest", digest_value)
            self._set_core_meta(branch_id, "CoreDigest", importance=5, ttl=None)
        self._conn.commit()

    # ==========================================================================
    # Memory Pressure Management (MemGPT-style)
    # ==========================================================================

    def _count_events(self, branch_id: str, include_ancestors: bool = True) -> int:
        """Count the number of recall events for a branch."""
        if include_ancestors:
            branch_ids = self._branch_chain(branch_id)
        else:
            branch_ids = [branch_id]
        if not branch_ids:
            return 0
        placeholders = ",".join(["?"] * len(branch_ids))
        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM events WHERE branch_id IN ({placeholders})",
            branch_ids,
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def _count_archival(self, branch_id: str, include_ancestors: bool = True) -> int:
        """Count the number of archival entries for a branch."""
        if include_ancestors:
            branch_ids = self._branch_chain(branch_id)
        else:
            branch_ids = [branch_id]
        if not branch_ids:
            return 0
        placeholders = ",".join(["?"] * len(branch_ids))
        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM archival WHERE branch_id IN ({placeholders})",
            branch_ids,
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def _estimate_archival_chars(self, branch_id: str, include_ancestors: bool = True) -> int:
        """Estimate total characters in archival memory for a branch."""
        if include_ancestors:
            branch_ids = self._branch_chain(branch_id)
        else:
            branch_ids = [branch_id]
        if not branch_ids:
            return 0
        placeholders = ",".join(["?"] * len(branch_ids))
        row = self._conn.execute(
            f"SELECT SUM(LENGTH(text)) as total FROM archival WHERE branch_id IN ({placeholders})",
            branch_ids,
        ).fetchone()
        return int(row["total"]) if row and row["total"] else 0

    def _get_core_usage(self, branch_id: str) -> dict:
        """Get core memory usage statistics."""
        rows = self._conn.execute(
            "SELECT key, value FROM core_kv WHERE branch_id=?",
            (branch_id,),
        ).fetchall()
        entries = {row["key"]: row["value"] for row in rows}
        used = _core_size(entries)
        max_chars = self.core_max_chars
        ratio = used / max_chars if max_chars > 0 else 0.0
        return {
            "used": used,
            "max": max_chars,
            "ratio": min(ratio, 1.0),
            "entry_count": len(entries),
        }

    def check_memory_pressure(self, branch_id: str) -> dict:
        """
        Check memory pressure and return status with recommendations.

        Memory pressure is calculated based on:
        - Core memory usage vs core_max_chars
        - Recall event count vs recall_max_events
        - Archival memory size (informational)

        Returns:
            {
                "pressure_level": "low" | "medium" | "high" | "critical",
                "core_usage": {"used": int, "max": int, "ratio": float, "entry_count": int},
                "recall_usage": {"count": int, "max": int, "ratio": float},
                "archival_usage": {"count": int, "estimated_chars": int},
                "recommendations": ["consolidate_recall", "evict_core", ...],
                "overall_ratio": float,
            }
        """
        if not branch_id:
            return {
                "pressure_level": "low",
                "core_usage": {"used": 0, "max": self.core_max_chars, "ratio": 0.0, "entry_count": 0},
                "recall_usage": {"count": 0, "max": self.recall_max_events, "ratio": 0.0},
                "archival_usage": {"count": 0, "estimated_chars": 0},
                "recommendations": [],
                "overall_ratio": 0.0,
            }

        # Core memory usage
        core_usage = self._get_core_usage(branch_id)

        # Recall memory usage
        recall_count = self._count_events(branch_id)
        recall_max = self.recall_max_events
        recall_ratio = recall_count / recall_max if recall_max > 0 else 0.0
        recall_usage = {
            "count": recall_count,
            "max": recall_max,
            "ratio": min(recall_ratio, 2.0),  # Can exceed 1.0
        }

        # Archival memory usage (informational)
        archival_count = self._count_archival(branch_id)
        archival_chars = self._estimate_archival_chars(branch_id)
        archival_usage = {
            "count": archival_count,
            "estimated_chars": archival_chars,
        }

        # Calculate overall pressure (weighted average)
        # Core memory is more critical, so weight it higher
        core_weight = 0.6
        recall_weight = 0.4
        overall_ratio = (
            core_usage["ratio"] * core_weight +
            min(recall_usage["ratio"], 1.0) * recall_weight
        )

        # Determine pressure level
        if overall_ratio >= self.pressure_thresholds["critical"]:
            pressure_level = "critical"
        elif overall_ratio >= self.pressure_thresholds["high"]:
            pressure_level = "high"
        elif overall_ratio >= self.pressure_thresholds["medium"]:
            pressure_level = "medium"
        else:
            pressure_level = "low"

        # Generate recommendations
        recommendations = []
        if recall_usage["ratio"] > 1.0:
            recommendations.append("consolidate_recall")
        if core_usage["ratio"] > self.pressure_thresholds["high"]:
            recommendations.append("evict_core")
        if core_usage["ratio"] > self.pressure_thresholds["medium"]:
            recommendations.append("compress_core")
        if recall_usage["ratio"] > self.pressure_thresholds["medium"]:
            recommendations.append("summarize_recall")
        if archival_count > 100:
            recommendations.append("prune_archival")

        result = {
            "pressure_level": pressure_level,
            "core_usage": core_usage,
            "recall_usage": recall_usage,
            "archival_usage": archival_usage,
            "recommendations": recommendations,
            "overall_ratio": round(overall_ratio, 3),
        }

        # Log the pressure check
        self._log_memory_event(
            "check_memory_pressure",
            "pressure",
            branch_id=branch_id,
            details={
                "pressure_level": pressure_level,
                "overall_ratio": result["overall_ratio"],
                "core_ratio": core_usage["ratio"],
                "recall_ratio": recall_usage["ratio"],
                "recommendations": recommendations,
            },
        )

        return result

    def _load_importance_prompt(self) -> str | None:
        """Load importance evaluation prompt template from file."""
        prompt_file = Path(__file__).parent.parent.parent / "prompt/config/memory/importance_evaluation.txt"
        if not prompt_file.exists():
            logger.warning("Importance evaluation prompt file not found: %s", prompt_file)
            return None
        try:
            return prompt_file.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read importance evaluation prompt: %s", exc)
            return None

    def evaluate_importance_with_llm(
        self,
        items: list[dict],
        context: str,
        task_hint: str,
    ) -> list[tuple[str, int, str]]:
        """
        Use LLM to evaluate importance of memory items for the current context.

        Args:
            items: List of memory items to evaluate, each with {"key": str, "value": str, "type": str}
            context: Current task context description
            task_hint: Hint about the current task/phase

        Returns:
            List of (item_key, importance_score, reason) tuples
        """
        if not items:
            return []

        # If LLM compression is not enabled, return default importance
        if not self.use_llm_compression or self._compression_client is None:
            return [(item.get("key", ""), 3, "default") for item in items]

        prompt_template = self._load_importance_prompt()
        if not prompt_template:
            return [(item.get("key", ""), 3, "no_template") for item in items]

        # Prepare items for evaluation (truncate values to reduce token usage)
        eval_items = []
        for item in items:
            eval_items.append({
                "key": item.get("key", ""),
                "type": item.get("type", "unknown"),
                "value_preview": _truncate(str(item.get("value", "")), 200),
            })

        items_json = json.dumps(eval_items, ensure_ascii=False, indent=2)
        prompt = prompt_template.format(
            context=context,
            task_hint=task_hint,
            items_json=items_json,
        )

        try:
            from ai_scientist.llm import get_response_from_llm

            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message="You are a memory management assistant. Output only valid JSON.",
                temperature=0.2,
            )

            if not response:
                logger.warning("LLM returned empty response for importance evaluation")
                return [(item.get("key", ""), 3, "empty_response") for item in items]

            # Parse JSON response
            response = response.strip()
            # Handle markdown code blocks
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            evaluations = json.loads(response)
            results = []
            key_to_eval = {e.get("key"): e for e in evaluations}

            for item in items:
                key = item.get("key", "")
                eval_data = key_to_eval.get(key, {})
                score = _coerce_importance(eval_data.get("score", 3))
                reason = str(eval_data.get("reason", ""))[:100]
                results.append((key, score, reason))

            self._log_memory_event(
                "evaluate_importance_with_llm",
                "importance",
                details={
                    "item_count": len(items),
                    "context_preview": _truncate(context, 100),
                    "task_hint": task_hint,
                    "results_count": len(results),
                },
            )

            return results

        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM importance response: %s", exc)
            return [(item.get("key", ""), 3, "parse_error") for item in items]
        except Exception as exc:
            logger.warning("LLM importance evaluation failed: %s", exc)
            return [(item.get("key", ""), 3, "error") for item in items]

    def _load_consolidation_prompt(self) -> str | None:
        """Load consolidation prompt template from file."""
        prompt_file = Path(__file__).parent.parent.parent / "prompt/config/memory/consolidation.txt"
        if not prompt_file.exists():
            logger.warning("Consolidation prompt file not found: %s", prompt_file)
            return None
        try:
            return prompt_file.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read consolidation prompt: %s", exc)
            return None

    def _consolidate_with_llm(
        self,
        entries: list[dict],
        context: str,
        max_chars: int,
    ) -> str:
        """
        Use LLM to consolidate multiple memory entries into a summary.

        Args:
            entries: List of memory entries to consolidate
            context: Current task context
            max_chars: Maximum characters for the consolidated summary

        Returns:
            Consolidated summary text
        """
        if not entries:
            return ""

        # Fallback to simple concatenation and truncation if LLM not available
        if not self.use_llm_compression or self._compression_client is None:
            combined = "\n".join(str(e.get("text", e.get("value", ""))) for e in entries)
            return _truncate(combined, max_chars)

        prompt_template = self._load_consolidation_prompt()
        if not prompt_template:
            combined = "\n".join(str(e.get("text", e.get("value", ""))) for e in entries)
            return _truncate(combined, max_chars)

        entries_json = json.dumps(entries, ensure_ascii=False, indent=2)
        prompt = prompt_template.format(
            context=context,
            max_chars=max_chars,
            entries_json=entries_json,
        )

        try:
            from ai_scientist.llm import get_response_from_llm

            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message="You are a memory consolidation assistant. Output only the consolidated text.",
                temperature=0.3,
            )

            if not response:
                combined = "\n".join(str(e.get("text", e.get("value", ""))) for e in entries)
                return _truncate(combined, max_chars)

            return _truncate(response.strip(), max_chars)

        except Exception as exc:
            logger.warning("LLM consolidation failed: %s", exc)
            combined = "\n".join(str(e.get("text", e.get("value", ""))) for e in entries)
            return _truncate(combined, max_chars)

    def consolidate_recall_events(
        self,
        branch_id: str,
        max_age_hours: float | None = None,
    ) -> int:
        """
        Consolidate older recall events by summarizing and moving to archival.

        This method:
        1. Finds recall events exceeding the recall_max_events limit
        2. Groups related events by kind/phase
        3. Creates consolidated summaries using LLM
        4. Writes summaries to archival memory
        5. Deletes the original events

        Args:
            branch_id: Branch ID to consolidate events for
            max_age_hours: Optional age threshold (hours) for events to consolidate

        Returns:
            Number of events consolidated
        """
        if not branch_id:
            return 0

        branch_ids = self._branch_chain(branch_id)
        if not branch_ids:
            return 0

        # Get all events for the branch chain
        placeholders = ",".join(["?"] * len(branch_ids))
        all_events = self._conn.execute(
            f"""
            SELECT id, branch_id, kind, text, tags, created_at
            FROM events
            WHERE branch_id IN ({placeholders})
            ORDER BY created_at DESC
            """,
            branch_ids,
        ).fetchall()

        if not all_events:
            return 0

        # Keep the most recent events, consolidate the rest
        threshold = int(self.recall_max_events * self.recall_consolidation_threshold)
        if len(all_events) <= threshold:
            return 0

        # Events to keep (most recent)
        events_to_keep = all_events[:self.recall_max_events]
        # Events to consolidate (older ones)
        events_to_consolidate = all_events[self.recall_max_events:]

        if not events_to_consolidate:
            return 0

        # Group events by kind for better consolidation
        event_groups: dict[str, list[dict]] = {}
        for event in events_to_consolidate:
            kind = event["kind"] or "general"
            if kind not in event_groups:
                event_groups[kind] = []
            event_groups[kind].append({
                "id": event["id"],
                "kind": kind,
                "text": event["text"],
                "created_at": event["created_at"],
            })

        consolidated_count = 0
        event_ids_to_delete = []

        for kind, group_events in event_groups.items():
            if not group_events:
                continue

            # Prepare entries for consolidation
            entries = [
                {
                    "kind": e["kind"],
                    "text": e["text"],
                    "timestamp": e["created_at"],
                }
                for e in group_events
            ]

            # Create consolidated summary
            context = f"recall events of type '{kind}'"
            summary = self._consolidate_with_llm(
                entries,
                context,
                max_chars=self.archival_max_chars,
            )

            if summary:
                # Write consolidated summary to archival
                time_range = ""
                if group_events:
                    oldest = min(e["created_at"] for e in group_events)
                    newest = max(e["created_at"] for e in group_events)
                    time_range = f" (from {oldest:.0f} to {newest:.0f})"

                archival_text = (
                    f"Consolidated recall events [{kind}]{time_range}:\n"
                    f"Event count: {len(group_events)}\n\n"
                    f"{summary}"
                )

                self._insert_archival(
                    branch_id,
                    archival_text,
                    tags=[
                        "RECALL_CONSOLIDATED",
                        "AUTO_CONSOLIDATE",
                        f"kind:{kind}",
                        f"event_count:{len(group_events)}",
                    ],
                )

            # Mark events for deletion
            event_ids_to_delete.extend(e["id"] for e in group_events)
            consolidated_count += len(group_events)

        # Delete consolidated events
        if event_ids_to_delete:
            id_placeholders = ",".join(["?"] * len(event_ids_to_delete))
            self._conn.execute(
                f"DELETE FROM events WHERE id IN ({id_placeholders})",
                event_ids_to_delete,
            )
            self._conn.commit()

        self._log_memory_event(
            "consolidate_recall_events",
            "consolidation",
            branch_id=branch_id,
            details={
                "total_events": len(all_events),
                "events_kept": len(events_to_keep),
                "events_consolidated": consolidated_count,
                "groups_processed": len(event_groups),
            },
        )

        return consolidated_count

    def auto_consolidate_memory(
        self,
        branch_id: str,
        pressure_level: str,
        task_hint: str | None = None,
    ) -> dict:
        """
        Automatically consolidate memory based on pressure level.

        Actions based on pressure_level:
        - medium: Evaluate core importance, compress low-priority items
        - high: Consolidate recall events, evict low-importance core items
        - critical: Aggressive consolidation, create digest summaries

        Args:
            branch_id: Branch ID to consolidate memory for
            pressure_level: Current pressure level ("medium", "high", "critical")
            task_hint: Optional hint about current task for context

        Returns:
            {
                "actions_taken": list of actions performed,
                "freed_chars": estimated characters freed,
                "core_items_evicted": number of core items evicted,
                "recall_events_consolidated": number of recall events consolidated,
                "new_pressure_level": pressure level after consolidation,
            }
        """
        if not branch_id:
            return {
                "actions_taken": [],
                "freed_chars": 0,
                "core_items_evicted": 0,
                "recall_events_consolidated": 0,
                "new_pressure_level": "low",
            }

        actions_taken = []
        freed_chars = 0
        core_items_evicted = 0
        recall_events_consolidated = 0

        context = task_hint or "research experiment"

        # Step 1: Evaluate core memory importance (medium+ pressure)
        if pressure_level in ("medium", "high", "critical"):
            core_rows = self._conn.execute(
                "SELECT key, value FROM core_kv WHERE branch_id=?",
                (branch_id,),
            ).fetchall()

            if core_rows:
                # Prepare items for importance evaluation
                protected_keys = {
                    RESOURCE_INDEX_KEY,
                    RESOURCE_DIGEST_KEY,
                    RESOURCE_INDEX_JSON_KEY,
                    RESOURCE_USED_KEY,
                    "idea_md_summary",
                    "phase0_summary",
                    "CoreDigest",
                }
                eval_items = [
                    {"key": row["key"], "value": row["value"], "type": "core"}
                    for row in core_rows
                    if row["key"] not in protected_keys
                ]

                if eval_items:
                    evaluations = self.evaluate_importance_with_llm(
                        eval_items, context, task_hint or "memory_consolidation"
                    )

                    # Update importance values based on LLM evaluation
                    for key, score, reason in evaluations:
                        self._set_core_meta(branch_id, key, importance=score, ttl=None)
                        if score <= 2 and pressure_level in ("high", "critical"):
                            # Evict low-importance items
                            value = next(
                                (row["value"] for row in core_rows if row["key"] == key), ""
                            )
                            if value:
                                freed_chars += len(value)
                                self._insert_archival(
                                    branch_id,
                                    f"Core evicted (auto): {key}\n{value}\nReason: {reason}",
                                    tags=["CORE_EVICT", "AUTO_CONSOLIDATE", f"core_key:{key}"],
                                )
                            self._delete_core_key(
                                branch_id, key, reason="auto_consolidate", op_name="auto_evict"
                            )
                            core_items_evicted += 1

                    actions_taken.append(f"evaluated_core_importance:{len(eval_items)}")
                    if core_items_evicted > 0:
                        actions_taken.append(f"evicted_core_items:{core_items_evicted}")

        # Step 2: Consolidate recall events (high+ pressure)
        if pressure_level in ("high", "critical"):
            recall_consolidated = self.consolidate_recall_events(branch_id)
            if recall_consolidated > 0:
                recall_events_consolidated = recall_consolidated
                actions_taken.append(f"consolidated_recall:{recall_consolidated}")

        # Step 3: Aggressive consolidation for critical pressure
        if pressure_level == "critical":
            # Force core budget enforcement
            self._enforce_core_budget(branch_id)
            actions_taken.append("enforced_core_budget")

        self._conn.commit()

        # Check new pressure level
        new_pressure = self.check_memory_pressure(branch_id)

        result = {
            "actions_taken": actions_taken,
            "freed_chars": freed_chars,
            "core_items_evicted": core_items_evicted,
            "recall_events_consolidated": recall_events_consolidated,
            "new_pressure_level": new_pressure["pressure_level"],
        }

        self._log_memory_event(
            "auto_consolidate_memory",
            "consolidation",
            branch_id=branch_id,
            phase=task_hint,
            details=result,
        )

        return result

    def set_core(
        self,
        branch_id: str,
        key: str,
        value: str,
        *,
        ttl: str | None = None,
        importance: int | None = None,
        op_name: str | None = None,
        phase: str | None = None,
        node_id: str | None = None,
        log_event: bool = True,
    ) -> None:
        if value is None:
            return
        self._set_core_value(branch_id, key, value)
        importance_val = _coerce_importance(importance if importance is not None else 3)
        self._set_core_meta(branch_id, key, importance=importance_val, ttl=ttl)
        if log_event:
            value_str = str(value)
            self._log_memory_event(
                op_name or "set_core",
                "core",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "key": key,
                    "value_preview": _truncate(
                        _redact(value_str, self.redact_secrets),
                        self.memory_log_max_chars,
                    ),
                    "value_chars": len(value_str),
                    "importance": importance_val,
                    "ttl": ttl,
                },
            )
        self._enforce_core_budget(branch_id)
        self._conn.commit()

    def get_core(
        self,
        branch_id: str,
        key: str,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        phase: str | None = None,
        node_id: str | None = None,
    ) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM core_kv WHERE branch_id=? AND key=?",
            (branch_id, key),
        ).fetchone()
        value = row["value"] if row else None
        if log_event:
            self._log_memory_event(
                op_name or "get_core",
                "core",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={"key": key, "found": value is not None},
            )
        return value

    def mem_core_get(self, keys: list[str] | None) -> dict[str, str]:
        branch_id = self._default_branch_id()
        if not branch_id:
            return {}
        if not keys:
            rows = self._conn.execute(
                "SELECT key, value FROM core_kv WHERE branch_id=?",
                (branch_id,),
            ).fetchall()
            result = {row["key"]: row["value"] for row in rows}
            self._log_memory_event(
                "mem_core_get",
                "core",
                branch_id=branch_id,
                details={"keys": None, "count": len(result)},
            )
            return result
        result = {key: self.get_core(branch_id, key, log_event=False) or "" for key in keys}
        self._log_memory_event(
            "mem_core_get",
            "core",
            branch_id=branch_id,
            details={"keys": keys, "count": len(result)},
        )
        return result

    def mem_core_set(
        self,
        key: str,
        value: str,
        *,
        ttl: str | None = None,
        importance: int = 3,
        branch_id: str | None = None,
    ) -> None:
        use_branch = branch_id or self._default_branch_id()
        if not use_branch:
            raise ValueError("mem_core_set requires a branch id")
        self.set_core(
            use_branch,
            key,
            value,
            ttl=ttl,
            importance=importance,
            op_name="mem_core_set",
        )

    def mem_core_del(self, key: str) -> None:
        branch_id = self._default_branch_id()
        if not branch_id:
            return
        self._delete_core_key(
            branch_id,
            key,
            reason="explicit",
            op_name="mem_core_del",
        )
        self._conn.commit()

    def write_event(
        self,
        branch_id: str,
        kind: str,
        text: str,
        tags: Any = None,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
        skip_duplicate: bool = True,
        task_hint: str | None = None,
        memory_size: int | None = None,
    ) -> None:
        payload = _redact(str(text), self.redact_secrets)
        tag_list = _normalize_tags(tags)
        tags_json = json.dumps(tag_list)

        # Check for duplicate: same branch_id, kind, text, and tags
        if skip_duplicate:
            existing = self._conn.execute(
                "SELECT id FROM events WHERE branch_id = ? AND kind = ? AND text = ? AND tags = ? LIMIT 1",
                (branch_id, kind, payload, tags_json),
            ).fetchone()
            if existing:
                # Duplicate found, skip insertion
                return

        self._conn.execute(
            "INSERT INTO events (branch_id, kind, text, tags, created_at, task_hint, memory_size) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (branch_id, kind, payload, tags_json, _now_ts(), task_hint, memory_size),
        )
        self._conn.commit()
        if log_event:
            if not node_id or not phase:
                for tag in tag_list:
                    if not node_id and str(tag).startswith("node_id:"):
                        node_id = str(tag).split("node_id:", 1)[1]
                    if not phase and str(tag).startswith("phase:"):
                        phase = str(tag).split("phase:", 1)[1]
            self._log_memory_event(
                op_name or "write_event",
                "recall",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "kind": kind,
                    "summary_preview": _truncate(payload, self.memory_log_max_chars),
                    "tags": tag_list,
                },
            )

        # Auto-consolidate recall events if overflow detected
        if self.auto_consolidate:
            recall_count = self._count_events(branch_id, include_ancestors=True)
            overflow_threshold = int(self.recall_max_events * self.recall_consolidation_threshold)
            if recall_count > overflow_threshold:
                try:
                    self.consolidate_recall_events(branch_id)
                except Exception as exc:
                    logger.warning("Auto-consolidation of recall events failed: %s", exc)

    def _insert_archival(
        self,
        branch_id: str,
        text: str,
        tags: Any = None,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
        skip_duplicate: bool = True,
    ) -> int | None:
        payload = _redact(str(text), self.redact_secrets)
        original_len = len(payload)

        # Apply LLM compression if text exceeds archival_max_chars limit
        if self.archival_max_chars > 0 and len(payload) > self.archival_max_chars:
            compressed = self._compress(payload, self.archival_max_chars, "archival entry")
            if compressed:
                payload = compressed

        tag_list = _normalize_tags(tags)
        tags_json = json.dumps(tag_list)

        # Check for duplicate: same branch_id, text, and tags
        if skip_duplicate:
            existing = self._conn.execute(
                "SELECT id FROM archival WHERE branch_id = ? AND text = ? AND tags = ? LIMIT 1",
                (branch_id, payload, tags_json),
            ).fetchone()
            if existing:
                # Duplicate found, skip insertion
                return existing[0]

        cur = self._conn.execute(
            "INSERT INTO archival (branch_id, text, tags, created_at) VALUES (?, ?, ?, ?)",
            (branch_id, payload, tags_json, _now_ts()),
        )
        row_id = cur.lastrowid
        if self._fts_enabled and row_id:
            self._conn.execute(
                "INSERT INTO archival_fts (rowid, text, tags, branch_id) VALUES (?, ?, ?, ?)",
                (row_id, payload, tags_json, branch_id),
            )
        self._conn.commit()
        if log_event:
            self._log_memory_event(
                op_name or "write_archival",
                "archival",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "record_id": row_id,
                    "tags": tag_list,
                    "text_preview": _truncate(payload, self.memory_log_max_chars),
                    "text_chars": len(str(text)),
                },
            )
        return row_id

    def write_archival(self, branch_id: str, text: str, tags: Any = None) -> None:
        self._insert_archival(branch_id, text, tags=tags, op_name="write_archival")

    def _fetch_events(self, branch_ids: Sequence[str], limit: int) -> list[sqlite3.Row]:
        if not branch_ids:
            return []
        placeholders = ",".join(["?"] * len(branch_ids))
        query = (
            f"SELECT kind, text, tags, created_at FROM events "
            f"WHERE branch_id IN ({placeholders}) "
            "ORDER BY created_at DESC LIMIT ?"
        )
        rows = self._conn.execute(query, [*branch_ids, limit]).fetchall()
        return rows

    def retrieve_archival(
        self,
        branch_id: str,
        query: str | None,
        k: int | None = None,
        include_ancestors: bool = True,
        tags_filter: Any = None,
        *,
        log_event: bool = True,
        op_name: str | None = None,
        node_id: str | None = None,
        phase: str | None = None,
    ) -> list[dict[str, Any]]:
        if not branch_id:
            return []
        k = k or self.retrieval_k
        branch_ids = self._branch_chain(branch_id) if include_ancestors else [branch_id]
        if not branch_ids:
            return []
        tag_list = _normalize_tags(tags_filter)
        placeholders = ",".join(["?"] * len(branch_ids))
        cleaned_query = (query or "").strip()
        rows: list[sqlite3.Row] = []
        if self._fts_enabled and cleaned_query:
            sql = (
                "SELECT a.id, a.branch_id, a.text, a.tags, a.created_at "
                "FROM archival_fts f JOIN archival a ON a.id = f.rowid "
                f"WHERE a.branch_id IN ({placeholders}) AND f.text MATCH ? "
                "ORDER BY a.created_at DESC LIMIT ?"
            )
            rows = self._conn.execute(sql, [*branch_ids, cleaned_query, k]).fetchall()
        else:
            sql = (
                "SELECT id, branch_id, text, tags, created_at FROM archival "
                f"WHERE branch_id IN ({placeholders}) "
                "ORDER BY created_at DESC"
            )
            rows = self._conn.execute(sql, branch_ids).fetchall()

        def tag_ok(tags_json: str | None) -> bool:
            if not tag_list:
                return True
            try:
                tags = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                tags = []
            return all(tag in tags for tag in tag_list)

        if not cleaned_query:
            filtered = [row for row in rows if tag_ok(row["tags"])]
            results = [dict(row) for row in filtered[:k]]
            if log_event:
                self._log_memory_event(
                    op_name or "retrieve_archival",
                    "archival",
                    branch_id=branch_id,
                    node_id=node_id,
                    phase=phase,
                    details={
                        "query": "",
                        "k": k,
                        "results": len(results),
                        "include_ancestors": include_ancestors,
                        "tags_filter": tag_list,
                    },
                )
            return results

        tokens = [t.lower() for t in re.split(r"\W+", cleaned_query) if t]
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if not tag_ok(row["tags"]):
                continue
            text = row["text"] or ""
            text_l = text.lower()
            score = 0.0
            for token in tokens:
                if token and token in text_l:
                    score += 1.0
            score += (row["created_at"] or 0) * 1e-12
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [dict(row) for _, row in scored[:k]]
        if log_event:
            self._log_memory_event(
                op_name or "retrieve_archival",
                "archival",
                branch_id=branch_id,
                node_id=node_id,
                phase=phase,
                details={
                    "query": cleaned_query,
                    "k": k,
                    "results": len(results),
                    "include_ancestors": include_ancestors,
                    "tags_filter": tag_list,
                },
            )
        return results

    def mem_recall_append(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        payload = dict(event)
        payload.setdefault("ts", _now_ts())
        if not payload.get("run_id") and self.run_id:
            payload["run_id"] = self.run_id
        node_id = payload.get("node_id")
        branch_id = self._resolve_branch_id(node_id) or self._default_branch_id()
        if not branch_id:
            return
        kind = str(payload.get("kind") or "event")
        summary = str(payload.get("summary") or "")
        refs = payload.get("refs") or []
        phase = str(payload.get("phase") or "")
        task_hint = payload.get("task_hint")
        memory_size = payload.get("memory_size")
        tags = [
            "RECALL_EVENT",
            f"run_id:{payload.get('run_id')}" if payload.get("run_id") else None,
            f"node_id:{node_id}" if node_id else None,
            f"phase:{phase}" if phase else None,
            f"kind:{kind}" if kind else None,
        ]
        for ref in refs if isinstance(refs, Iterable) and not isinstance(refs, str) else []:
            tags.append(f"ref:{ref}")
        tag_list = [tag for tag in tags if tag]
        self.write_event(
            branch_id,
            kind,
            summary,
            tags=tag_list,
            log_event=False,
            task_hint=task_hint,
            memory_size=memory_size,
        )
        self._log_memory_event(
            "mem_recall_append",
            "recall",
            branch_id=branch_id,
            node_id=node_id,
            phase=phase,
            details={
                "kind": kind,
                "summary_preview": _truncate(
                    _redact(summary, self.redact_secrets),
                    self.memory_log_max_chars,
                ),
                "tags": tag_list,
            },
        )

    def mem_recall_search(self, query: str, *, k: int = 20) -> list[dict]:
        branch_id = self._default_branch_id()
        if not branch_id:
            return []
        branch_ids = self._branch_chain(branch_id)
        rows = self._fetch_events(branch_ids, limit=max(k * 5, k))
        tokens = [t.lower() for t in re.split(r"\W+", query or "") if t]
        results: list[dict] = []
        for row in rows:
            text = (row["text"] or "")
            text_l = text.lower()
            if tokens and not any(token in text_l for token in tokens):
                continue
            tags_json = row["tags"]
            try:
                tag_list = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                tag_list = []
            meta = {"tags": tag_list}
            for tag in tag_list:
                if tag.startswith("run_id:"):
                    meta["run_id"] = tag.split("run_id:", 1)[1]
                if tag.startswith("node_id:"):
                    meta["node_id"] = tag.split("node_id:", 1)[1]
                if tag.startswith("phase:"):
                    meta["phase"] = tag.split("phase:", 1)[1]
            refs = [tag.split("ref:", 1)[1] for tag in tag_list if tag.startswith("ref:")]
            if refs:
                meta["refs"] = refs
            results.append(
                {
                    "ts": row["created_at"],
                    "kind": row["kind"],
                    "summary": text,
                    **meta,
                }
            )
            if len(results) >= k:
                break
        self._log_memory_event(
            "mem_recall_search",
            "recall",
            branch_id=branch_id,
            details={"query": query, "k": k, "results": len(results)},
        )
        return results

    def mem_archival_write(
        self,
        text: str,
        *,
        tags: list[str],
        meta: dict | None = None,
    ) -> str:
        meta_in = meta
        meta = dict(meta or {})
        if self.run_id and not meta.get("run_id"):
            meta["run_id"] = self.run_id
        branch_id = None
        if meta.get("branch_id"):
            branch_id = self._resolve_branch_id(meta["branch_id"]) or meta["branch_id"]
        node_id = meta.get("node_id")
        if not branch_id and node_id:
            branch_id = self._resolve_branch_id(node_id)
        if not branch_id:
            branch_id = self._default_branch_id()
        if not branch_id:
            raise ValueError("mem_archival_write requires a branch id")
        tag_list = list(tags or [])
        if meta.get("run_id"):
            tag_list.append(f"run_id:{meta['run_id']}")
        if node_id:
            tag_list.append(f"node_id:{node_id}")
        record_id = self._insert_archival(
            branch_id,
            text,
            tags=tag_list,
            log_event=False,
        )
        text_str = str(text)
        self._log_memory_event(
            "mem_archival_write",
            "archival",
            branch_id=branch_id,
            node_id=node_id,
            phase=meta.get("phase"),
            details={
                "record_id": record_id,
                "tags": tag_list,
                "text_preview": _truncate(
                    _redact(text_str, self.redact_secrets),
                    self.memory_log_max_chars,
                ),
                "text_chars": len(text_str),
            },
        )
        return str(record_id) if record_id is not None else ""

    def mem_archival_update(
        self,
        record_id: str,
        *,
        text: str | None = None,
        tags: list[str] | None = None,
        meta: dict | None = None,
    ) -> None:
        if not record_id:
            return
        meta = dict(meta or {})
        updates: list[str] = []
        params: list[Any] = []
        if text is not None:
            payload = _redact(str(text), self.redact_secrets)
            updates.append("text=?")
            params.append(payload)
        if tags is not None or meta_in is not None:
            tag_list = list(tags or [])
            if self.run_id and not meta.get("run_id"):
                meta["run_id"] = self.run_id
            if meta.get("run_id"):
                tag_list.append(f"run_id:{meta['run_id']}")
            if meta.get("node_id"):
                tag_list.append(f"node_id:{meta['node_id']}")
            updates.append("tags=?")
            params.append(json.dumps(_normalize_tags(tag_list)))
        if not updates:
            return
        params.append(record_id)
        self._conn.execute(
            f"UPDATE archival SET {', '.join(updates)} WHERE id=?",
            params,
        )
        if self._fts_enabled and text is not None:
            self._conn.execute(
                "UPDATE archival_fts SET text=? WHERE rowid=?",
                (payload, record_id),
            )
        self._conn.commit()
        details = {"record_id": record_id}
        if tags is not None:
            details["tags"] = _normalize_tags(tags)
        if text is not None:
            text_str = str(text)
            details["text_preview"] = _truncate(
                _redact(text_str, self.redact_secrets),
                self.memory_log_max_chars,
            )
            details["text_chars"] = len(text_str)
        self._log_memory_event(
            "mem_archival_update",
            "archival",
            branch_id=meta.get("branch_id") if isinstance(meta, dict) else None,
            node_id=meta.get("node_id") if isinstance(meta, dict) else None,
            phase=meta.get("phase") if isinstance(meta, dict) else None,
            details=details,
        )

    def mem_archival_search(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        k: int = 10,
    ) -> list[dict]:
        branch_id = self._default_branch_id()
        if not branch_id:
            return []
        rows = self.retrieve_archival(
            branch_id=branch_id,
            query=query,
            k=k,
            include_ancestors=True,
            tags_filter=tags,
            log_event=False,
        )
        self._log_memory_event(
            "mem_archival_search",
            "archival",
            branch_id=branch_id,
            details={"query": query, "k": k, "results": len(rows), "tags": tags or []},
        )
        return rows

    def mem_archival_get(self, record_id: str) -> dict:
        row = self._conn.execute(
            "SELECT id, branch_id, text, tags, created_at FROM archival WHERE id=?",
            (record_id,),
        ).fetchone()
        result = dict(row) if row else {}
        self._log_memory_event(
            "mem_archival_get",
            "archival",
            branch_id=result.get("branch_id") if result else None,
            details={
                "record_id": record_id,
                "found": bool(result),
            },
        )
        return result

    def mem_node_fork(
        self,
        parent_node_id: str | None,
        child_node_id: str,
        ancestor_chain: list[str] | None = None,
    ) -> None:
        """Fork a new branch from parent.

        Args:
            parent_node_id: The parent node's ID (or None for root-level nodes)
            child_node_id: The child node's ID
            ancestor_chain: Optional list of ancestor node IDs from root to parent
                           (e.g., [grandparent_id, parent_id]). If provided, missing
                           branches will be created in the correct order to preserve
                           the full tree structure.
        """
        parent_branch = self._resolve_branch_id(parent_node_id) if parent_node_id else None

        # If parent_node_id was provided but couldn't be resolved, we need to create
        # the missing branches. Use ancestor_chain if available for correct tree structure.
        if parent_node_id and not parent_branch:
            if ancestor_chain:
                # Create missing branches in order from root to parent
                # ancestor_chain should be ordered: [grandparent, ..., parent]
                prev_branch = self.root_branch_id
                for ancestor_id in ancestor_chain:
                    existing = self._resolve_branch_id(ancestor_id)
                    if not existing:
                        self.create_branch(prev_branch, node_uid=ancestor_id, branch_id=ancestor_id)
                        logger.debug(f"Auto-created missing ancestor branch: {ancestor_id} (parent: {prev_branch})")
                        prev_branch = ancestor_id
                    else:
                        prev_branch = existing
                parent_branch = prev_branch
            else:
                # Fallback: create parent branch with root as its parent
                # This may result in incorrect tree structure but maintains backward compatibility
                root_branch = self.root_branch_id
                self.create_branch(root_branch, node_uid=parent_node_id, branch_id=parent_node_id)
                parent_branch = parent_node_id
                logger.debug(f"Auto-created missing parent branch (no ancestor_chain): {parent_node_id}")

        branch_id = child_node_id or uuid.uuid4().hex
        self.create_branch(parent_branch, node_uid=child_node_id, branch_id=branch_id)
        self._log_memory_event(
            "mem_node_fork",
            "node",
            branch_id=branch_id,
            node_id=child_node_id,
            details={
                "parent_node_id": parent_node_id,
                "parent_branch_id": parent_branch,
                "child_branch_id": branch_id,
                "parent_auto_created": parent_node_id and parent_node_id == parent_branch,
            },
        )

    def mem_node_read(self, node_id: str, scope: str = "all") -> dict:
        branch_id = self._resolve_branch_id(node_id)
        if not branch_id:
            return {}
        data: dict[str, Any] = {}
        branch_chain = self._branch_chain(branch_id)
        if scope in ("all", "core"):
            core_rows = self._conn.execute(
                f"SELECT branch_id, key, value, updated_at FROM core_kv WHERE branch_id IN ({','.join(['?']*len(branch_chain))})",
                branch_chain,
            ).fetchall()
            core_latest: dict[str, tuple[str, float]] = {}
            for row in core_rows:
                key = row["key"]
                updated_at = row["updated_at"] or 0
                if key not in core_latest or updated_at > core_latest[key][1]:
                    core_latest[key] = (row["value"], updated_at)
            data["core"] = {key: value for key, (value, _) in core_latest.items()}
        if scope in ("all", "recall"):
            recall_rows = self._fetch_events(branch_chain, self.recall_max_events)
            recall: list[dict[str, Any]] = []
            for row in recall_rows:
                recall.append(
                    {
                        "ts": row["created_at"],
                        "kind": row["kind"],
                        "summary": row["text"],
                        "tags": row["tags"],
                    }
                )
            data["recall"] = recall
        if scope in ("all", "archival"):
            data["archival"] = self.retrieve_archival(
                branch_id=branch_id,
                query="",
                k=self.retrieval_k,
                include_ancestors=True,
                log_event=False,
            )
        self._log_memory_event(
            "mem_node_read",
            "node",
            branch_id=branch_id,
            node_id=node_id,
            details={
                "scope": scope,
                "core_count": len(data.get("core", {}) or {}),
                "recall_count": len(data.get("recall", []) or []),
                "archival_count": len(data.get("archival", []) or []),
            },
        )
        return data

    def mem_node_write(
        self,
        node_id: str,
        *,
        core_updates: dict | None = None,
        recall_event: dict | None = None,
        archival_records: list[dict] | None = None,
    ) -> None:
        branch_id = self._resolve_branch_id(node_id) or node_id
        self._log_memory_event(
            "mem_node_write",
            "node",
            branch_id=branch_id,
            node_id=node_id,
            phase=recall_event.get("phase") if isinstance(recall_event, dict) else None,
            details={
                "core_updates": list(core_updates.keys()) if core_updates else [],
                "recall_event": bool(recall_event),
                "archival_records": len(archival_records or []),
            },
        )
        if core_updates:
            for key, value in core_updates.items():
                ttl = None
                importance = 3
                actual_value = value
                if isinstance(value, dict) and "value" in value:
                    actual_value = value.get("value")
                    ttl = value.get("ttl")
                    importance = _coerce_importance(value.get("importance", 3))
                self.set_core(branch_id, key, str(actual_value), ttl=ttl, importance=importance)
        if recall_event:
            event = dict(recall_event)
            if "node_id" not in event:
                event["node_id"] = node_id
            self.mem_recall_append(event)
        if archival_records:
            for record in archival_records:
                if not isinstance(record, dict):
                    continue
                text = record.get("text", "")
                tags = record.get("tags", [])
                meta = record.get("meta", {})
                if "node_id" not in meta:
                    meta = {**meta, "node_id": node_id}
                self.mem_archival_write(str(text), tags=list(tags or []), meta=meta)

    def mem_resources_index_update(self, run_id: str, index_text: str, *, branch_id: str | None = None) -> None:
        use_branch = branch_id or self._default_branch_id()
        if not use_branch:
            return
        if run_id and not self.run_id:
            self.run_id = str(run_id)
        self.set_core(
            use_branch,
            RESOURCE_INDEX_KEY,
            index_text,
            importance=5,
            log_event=False,
        )
        self._log_memory_event(
            "mem_resources_index_update",
            "resources",
            branch_id=use_branch,
            details={
                "run_id": run_id,
                "index_chars": len(index_text or ""),
            },
        )

    def mem_resources_snapshot_upsert(self, run_id: str, item_name: str, snapshot: dict) -> None:
        tags = ["RESOURCE_ITEM", f"resource_name:{item_name}"]
        if run_id:
            tags.append(f"run_id:{run_id}")
        payload = json.dumps(snapshot, ensure_ascii=True, indent=2)
        self.mem_archival_write(payload, tags=tags, meta={"run_id": run_id})
        self._log_memory_event(
            "mem_resources_snapshot_upsert",
            "resources",
            branch_id=self._default_branch_id(),
            details={
                "run_id": run_id,
                "item_name": item_name,
                "payload_chars": len(payload),
            },
        )

    def mem_resources_resolve_and_refresh(self, run_id: str) -> None:
        branch_id = self._default_branch_id()
        if not branch_id:
            return
        raw_index = self.get_core(branch_id, RESOURCE_INDEX_JSON_KEY, log_event=False)
        resource_file = None
        if raw_index:
            try:
                parsed = json.loads(raw_index)
                resource_file = parsed.get("resource_file")
            except json.JSONDecodeError:
                resource_file = None
        if not resource_file:
            self._log_memory_event(
                "mem_resources_resolve_and_refresh",
                "resources",
                branch_id=branch_id,
                details={"run_id": run_id, "resource_file": "", "status": "skipped"},
            )
            return
        if run_id and not self.run_id:
            self.run_id = str(run_id)
        workspace_root = self.workspace_root or self.db_path.parent.parent
        self._log_memory_event(
            "mem_resources_resolve_and_refresh",
            "resources",
            branch_id=branch_id,
            details={"run_id": run_id, "resource_file": resource_file, "status": "refresh"},
        )
        snapshot = build_resource_snapshot(
            resource_file,
            workspace_root=workspace_root,
            ai_scientist_root=str(self.ai_scientist_root) if self.ai_scientist_root else None,
            phase_mode=_cfg_get(self.config, "phase_mode", "unknown"),
            log=logger,
        )
        update_resource_snapshot_if_changed(snapshot, self)

    def _render_resource_items(self, branch_id: str, task_hint: str | None) -> str:
        query = (task_hint or "").strip()
        if not query:
            query = "resource"
        rows = self.retrieve_archival(
            branch_id=branch_id,
            query=query,
            k=self.retrieval_k,
            include_ancestors=True,
            tags_filter=[RESOURCE_ITEM_TAG],
            log_event=False,
        )
        if not rows:
            return ""
        resource_ids: set[str] = set()
        chunks: list[str] = []
        for row in rows:
            for rid in _extract_resource_ids(row.get("tags")):
                if rid:
                    resource_ids.add(rid)
            chunks.append(self._compress(row.get("text", ""), 1200, "resource item"))
        if resource_ids:
            note = f"prompt:{query}"
            for rid in sorted(resource_ids):
                track_resource_usage(rid, {"ltm": self, "branch_id": branch_id, "note": note})
        return "\n\n---\n\n".join(chunk for chunk in chunks if chunk).strip()

    def _render_core_memory(
        self, branch_ids: list[str], branch_id: str, no_limit: bool
    ) -> tuple[str, str]:
        """Render core memory section and extract resource index."""
        core_rows = self._conn.execute(
            f"SELECT branch_id, key, value, updated_at FROM core_kv WHERE branch_id IN ({','.join(['?']*len(branch_ids))})",
            branch_ids,
        ).fetchall()

        core_latest: dict[str, tuple[str, float]] = {}
        for row in core_rows:
            key = row["key"]
            updated_at = row["updated_at"] or 0
            if key not in core_latest or updated_at > core_latest[key][1]:
                core_latest[key] = (row["value"], updated_at)

        # Extract special keys
        core_lines: list[str] = []
        core_latest.pop("idea_md_hash", None)
        resource_index = core_latest.pop(RESOURCE_INDEX_KEY, (None, 0))[0]
        core_latest.pop(RESOURCE_INDEX_JSON_KEY, None)
        core_latest.pop(RESOURCE_DIGEST_KEY, None)
        core_latest.pop(RESOURCE_USED_KEY, None)
        idea_summary = core_latest.pop("idea_md_summary", (None, 0))[0]

        # Handle idea summary
        if no_limit and self.always_inject_idea_summary:
            idea_archival = self.retrieve_archival(
                branch_id, query="idea", k=1, include_ancestors=True, tags_filter=["IDEA_MD"], log_event=False
            )
            if idea_archival:
                core_lines.append(f"- Idea (full text): {idea_archival[0].get('text', '')}")
            elif idea_summary:
                core_lines.append(f"- Idea summary: {idea_summary}")
        elif self.always_inject_idea_summary:
            core_lines.append(f"- Idea summary: {idea_summary or '(not available)'}")

        # Handle phase0 summary
        phase0_summary = core_latest.pop("phase0_summary", (None, 0))[0]
        if self.always_inject_phase0_summary:
            core_lines.append(f"- Phase 0 internal summary: {phase0_summary or '(not available)'}")

        # Add remaining core items
        for key, (value, _) in sorted(core_latest.items(), key=lambda kv: kv[0]):
            core_lines.append(f"- {key}: {value}")

        core_text = "\n".join(core_lines).strip()
        return core_text, resource_index or ""

    def _render_recall_memory(self, branch_ids: list[str]) -> tuple[str, list]:
        """Render recall memory section."""
        recall_rows = self._fetch_events(branch_ids, self.recall_max_events)
        recall_lines = []
        for row in recall_rows:
            tags = ""
            try:
                tag_list = json.loads(row["tags"]) if row["tags"] else []
                if tag_list:
                    tags = f" (tags: {', '.join(tag_list)})"
            except json.JSONDecodeError:
                tags = ""
            recall_lines.append(f"- [{row['kind']}] {row['text']}{tags}")
        recall_text = "\n".join(recall_lines).strip()
        return recall_text, recall_rows

    def _render_archival_memory(
        self, branch_id: str, task_hint: str | None, no_limit: bool
    ) -> tuple[str, list]:
        """Render archival memory section."""
        archival_rows = self.retrieve_archival(
            branch_id=branch_id,
            query=task_hint or "",
            k=self.retrieval_k,
            include_ancestors=True,
            log_event=False,
        )
        archival_lines = []
        for row in archival_rows:
            tags = ""
            try:
                tag_list = json.loads(row.get("tags") or "[]")
                if tag_list:
                    tags = f" (tags: {', '.join(tag_list)})"
            except json.JSONDecodeError:
                tags = ""
            if no_limit:
                snippet = row.get("text", "")
            else:
                snippet = self._compress(
                    row.get("text", ""),
                    self.section_budgets["archival_snippet"],
                    "archival memory entry"
                )
            archival_lines.append(f"- {snippet}{tags}")
        archival_text = "\n".join(archival_lines).strip()
        return archival_text, archival_rows

    def _combine_memory_sections(
        self, sections: list[tuple[str, str]], budget_chars: int, no_limit: bool
    ) -> str:
        """Combine memory sections with budget management."""
        if no_limit:
            rendered_parts = [f"{title}:\n{body}\n" for title, body in sections]
        else:
            remaining = max(int(budget_chars), 0)
            rendered_parts = []
            for title, body in sections:
                block = f"{title}:\n{body}\n"
                if remaining <= 0:
                    break
                if len(block) <= remaining:
                    rendered_parts.append(block)
                    remaining -= len(block)
                else:
                    if remaining > len(title) + 10:
                        truncated = self._compress(block, remaining, f"{title} section")
                        rendered_parts.append(truncated)
                    break
        return "\n".join(part.strip() for part in rendered_parts if part.strip()).strip()

    def render_for_prompt(
        self,
        branch_id: str,
        task_hint: str | None,
        budget_chars: int,
        no_limit: bool = False,
    ) -> str:
        """Render memory for prompt injection.

        Args:
            branch_id: Branch ID to render memory for
            task_hint: Optional task hint for retrieval
            budget_chars: Character budget (ignored if no_limit=True)
            no_limit: If True, skip all truncation/compression

        Returns:
            Rendered memory string
        """
        if not branch_id:
            return ""
        branch_ids = self._branch_chain(branch_id)
        if not branch_ids:
            return ""

        # Render each memory section
        core_text, resource_index = self._render_core_memory(branch_ids, branch_id, no_limit)
        resource_items_text = self._render_resource_items(branch_id, task_hint)
        recall_text, recall_rows = self._render_recall_memory(branch_ids)
        archival_text, archival_rows = self._render_archival_memory(branch_id, task_hint, no_limit)

        # Build sections
        sections = []
        if resource_index:
            sections.append(("Resource Index", resource_index))
        if core_text:
            sections.append(("Core Memory", core_text))
        if recall_text:
            sections.append(("Recall Memory", recall_text))
        if archival_text:
            sections.append(("Archival Memory", archival_text))
        if resource_items_text:
            sections.append(("Resource Memory", resource_items_text))

        if not sections:
            self._log_memory_event(
                "render_for_prompt",
                "prompt",
                branch_id=branch_id,
                phase=task_hint,
                details={
                    "budget_chars": budget_chars,
                    "core_count": len(core_text.splitlines()) if core_text else 0,
                    "recall_count": len(recall_rows),
                    "archival_count": len(archival_rows),
                    "resource_items": 0,
                },
            )
            return ""

        # Combine sections with budget management
        rendered = self._combine_memory_sections(sections, budget_chars, no_limit)

        self._log_memory_event(
            "render_for_prompt",
            "prompt",
            branch_id=branch_id,
            phase=task_hint,
            details={
                "budget_chars": budget_chars,
                "core_count": len(core_text.splitlines()) if core_text else 0,
                "recall_count": len(recall_rows),
                "archival_count": len(archival_rows),
                "resource_items": 1 if resource_items_text else 0,
            },
        )
        return rendered

    def ingest_phase0_internal_info(
        self,
        branch_id: str,
        node_uid: str,
        phase0_payload: Any,
        artifact_paths: Sequence[str] | None,
        command_str: str | None,
    ) -> None:
        payload_obj = phase0_payload
        if is_dataclass(phase0_payload):
            payload_obj = asdict(phase0_payload)
        artifacts = []
        for path in artifact_paths or []:
            p = Path(path)
            if not p.exists():
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                content = ""
            artifacts.append(
                {
                    "path": str(p),
                    "content": self._compress(content, 5000, "phase0 artifact content"),
                }
            )

        payload = {
            "branch_id": branch_id,
            "node_uid": node_uid,
            "command": command_str,
            "phase0_payload": payload_obj,
            "artifacts": artifacts,
            "created_at": _now_ts(),
        }
        md_lines = [
            "# Phase 0 Internal Info",
            "",
            f"- node_uid: {node_uid}",
            f"- branch_id: {branch_id}",
            f"- command: {command_str or ''}",
            "",
            "## Phase 0 Payload",
            "```json",
            json.dumps(payload_obj, indent=2),
            "```",
        ]
        if artifacts:
            md_lines.append("")
            md_lines.append("## Artifacts")
            for item in artifacts:
                md_lines.append(f"### {item['path']}")
                md_lines.append("```")
                md_lines.append(item["content"])
                md_lines.append("```")
        md_text = "\n".join(md_lines)

        # Prepare payload for summarization (merge artifacts so they can be parsed)
        summary_payload = payload_obj.copy()
        summary_payload["artifacts"] = artifacts
        
        summary = _summarize_phase0(
            summary_payload,
            command_str,
            max_chars=self.section_budgets["phase0_summary"],
            compress_fn=self._compress,
        )
        self.set_core(
            branch_id,
            "phase0_summary",
            summary,
            importance=5,
            op_name="ingest_phase0_internal_info",
            phase="phase0",
            node_id=node_uid,
        )
        archival_text = f"{summary}\n\n{md_text}"
        self.mem_archival_write(
            archival_text,
            tags=["PHASE0_INTERNAL", f"node_uid:{node_uid}", f"branch_id:{branch_id}"],
            meta={
                "node_id": node_uid,
                "run_id": self.run_id,
                "branch_id": branch_id,
                "phase": "phase0",
            },
        )

    def ingest_idea_md(
        self,
        branch_id: str,
        node_uid: str,
        idea_path: str | Path,
        is_root: bool = False,
    ) -> None:
        path = Path(idea_path)
        if not path.exists():
            return
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read idea.md: %s", exc)
            return
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        previous_hash = self.get_core(branch_id, "idea_md_hash", log_event=False)
        if previous_hash == content_hash:
            return

        summary = _summarize_idea(
            content,
            max_chars=self.section_budgets["idea_summary"],
            compress_fn=self._compress,
            section_limit=self.section_budgets["idea_section_limit"],
        )
        self.set_core(
            branch_id,
            "idea_md_summary",
            summary,
            importance=5,
            op_name="ingest_idea_md",
            phase="idea_md",
            node_id=node_uid,
        )
        self.set_core(
            branch_id,
            "idea_md_hash",
            content_hash,
            op_name="ingest_idea_md",
            phase="idea_md",
            node_id=node_uid,
        )
        tags = ["IDEA_MD", f"node_uid:{node_uid}", f"branch_id:{branch_id}"]
        if is_root:
            tags.append("ROOT_IDEA")
        archival_text = f"{summary}\n\n{content}"
        self.mem_archival_write(
            archival_text,
            tags=tags,
            meta={
                "node_id": node_uid,
                "run_id": self.run_id,
                "branch_id": branch_id,
                "phase": "idea_md",
            },
        )

    def _collect_experimental_results(self, best_branch: str, no_budget_limit: bool) -> str:
        """Collect experimental results from events and core memory."""
        result_events = self._fetch_events(self._branch_chain(best_branch), self.recall_max_events)
        result_texts = []
        for row in result_events:
            kind = str(row["kind"]).lower()
            if kind in {"node_result", "experiment_result", "run_complete", "phase_complete"}:
                if row["text"]:
                    result_texts.append(row["text"])

        # Fall back to core memory if no result events
        if not result_texts:
            results_from_core = self.get_core(best_branch, "experiment_results") or self.get_core(best_branch, "results") or ""
            if results_from_core:
                result_texts.append(results_from_core)

        # Combine results
        if result_texts:
            core_snapshot = "\n\n".join(result_texts)
        else:
            core_snapshot = "No experimental results recorded."

        if not no_budget_limit:
            core_snapshot = self._compress(core_snapshot, self.section_budgets["results"], "experimental results", branch_id=best_branch)

        return core_snapshot

    def _get_idea_text_and_summary(
        self, run_dir: Path, best_branch: str, no_budget_limit: bool
    ) -> tuple[str, str]:
        """Get idea text and summary."""
        idea_text = ""
        if no_budget_limit:
            idea_path = run_dir / "idea.md"
            if idea_path.exists():
                try:
                    idea_text = idea_path.read_text(encoding="utf-8")
                except Exception:
                    pass

        if not idea_text:
            idea_archival = self.retrieve_archival(
                best_branch, query="idea", k=1, include_ancestors=True, tags_filter=["IDEA_MD"]
            )
            idea_text = idea_archival[0]["text"] if idea_archival else ""

        # Get idea summary
        idea_summary = self.get_core(best_branch, "idea_md_summary") or ""
        if not idea_summary and idea_text:
            if no_budget_limit:
                idea_summary = idea_text
            else:
                idea_summary = _summarize_idea(
                    idea_text,
                    max_chars=self.section_budgets["idea_summary"],
                    compress_fn=self._compress,
                    section_limit=self.section_budgets["idea_section_limit"],
                )

        return idea_text, idea_summary

    def _get_phase0_summary(
        self, memory_dir: Path, best_branch: str, no_budget_limit: bool
    ) -> str:
        """Get phase0 summary."""
        phase0_summary = self.get_core(best_branch, "phase0_summary") or ""
        if no_budget_limit:
            phase0_path = memory_dir / "phase0_internal_info.json"
            if phase0_path.exists():
                try:
                    phase0_data = json.loads(phase0_path.read_text(encoding="utf-8"))
                    phase0_parts = []
                    if phase0_data.get("node_uid"):
                        phase0_parts.append(f"node_uid: {phase0_data['node_uid']}")
                    if phase0_data.get("branch_id"):
                        phase0_parts.append(f"branch_id: {phase0_data['branch_id']}")
                    if phase0_data.get("command"):
                        phase0_parts.append(f"command: {phase0_data['command']}")

                    if phase0_data.get("phase0_payload"):
                        payload = phase0_data["phase0_payload"]
                        if isinstance(payload, dict):
                            if payload.get("plan", {}).get("goal_summary"):
                                phase0_parts.append(f"goal: {payload['plan']['goal_summary']}")
                            env_ctx = payload.get("plan", {}).get("environment_context") or {}
                            if env_ctx:
                                if env_ctx.get("cpu_info"):
                                    phase0_parts.append(f"CPU: {env_ctx['cpu_info'][:200]}")
                                if env_ctx.get("os_release"):
                                    phase0_parts.append(f"OS: {env_ctx['os_release']}")

                    if phase0_data.get("artifacts"):
                        artifact_paths = [a.get("path", "") for a in phase0_data["artifacts"] if isinstance(a, dict)]
                        if artifact_paths:
                            phase0_parts.append(f"artifacts: {', '.join(artifact_paths[:5])}")

                    if phase0_parts:
                        phase0_summary = "\n".join(phase0_parts)
                except Exception:
                    pass

        return phase0_summary

    def _collect_failure_notes(self, best_branch: str) -> list[str]:
        """Collect failure notes from events."""
        failure_events = self._fetch_events(self._branch_chain(best_branch), self.recall_max_events)
        failure_notes = [
            row["text"]
            for row in failure_events
            if str(row["kind"]).lower() in {"error", "exception", "failure"}
        ]
        return failure_notes

    def _load_resource_snapshot_and_index(
        self, memory_dir: Path, root_branch_id: str, best_branch: str
    ) -> tuple[dict, dict, dict]:
        """Load resource snapshot and build index."""
        resource_snapshot_path = memory_dir / "resource_snapshot.json"
        resource_snapshot = {}
        if resource_snapshot_path.exists():
            try:
                resource_snapshot = json.loads(resource_snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                resource_snapshot = {}

        resource_index = {}
        raw_index = self.get_core(root_branch_id, RESOURCE_INDEX_JSON_KEY)
        if not raw_index and best_branch:
            raw_index = self.get_core(best_branch, RESOURCE_INDEX_JSON_KEY)
        if raw_index:
            try:
                resource_index = json.loads(raw_index)
            except json.JSONDecodeError:
                resource_index = {}

        # Merge items from snapshot if missing in KV index
        if resource_snapshot.get("items"):
            if "items" not in resource_index:
                resource_index["items"] = []
            existing_ids = {i.get("id") for i in resource_index["items"] if isinstance(i, dict)}
            for item in resource_snapshot["items"]:
                if isinstance(item, dict) and item.get("id") and item.get("id") not in existing_ids:
                    resource_index["items"].append(item)

        item_index = {
            item.get("id"): item
            for item in resource_index.get("items", [])
            if isinstance(item, dict) and item.get("id")
        }

        return resource_snapshot, resource_index, item_index

    def _collect_resource_usage(
        self, root_branch_id: str, best_branch: str, item_index: dict, no_budget_limit: bool
    ) -> list[dict[str, Any]]:
        """Collect information about used resources."""
        used_ids: list[str] = []
        raw_used = self.get_core(root_branch_id, RESOURCE_USED_KEY)
        if not raw_used and best_branch:
            raw_used = self.get_core(best_branch, RESOURCE_USED_KEY)
        if raw_used:
            try:
                parsed = json.loads(raw_used)
                if isinstance(parsed, list):
                    used_ids = [str(item) for item in parsed]
            except json.JSONDecodeError:
                used_ids = []

        usage_notes: dict[str, list[str]] = {}
        event_rows = self._fetch_events(self._branch_chain(best_branch), 200)
        for row in event_rows:
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except json.JSONDecodeError:
                tags = []
            if RESOURCE_USED_TAG not in tags and str(row["kind"]).lower() != "resource_used":
                continue
            text = row["text"] or ""
            parts = [part.strip() for part in text.split("|", 1)]
            if not parts:
                continue
            rid = parts[0]
            note = parts[1] if len(parts) > 1 else ""
            if rid:
                usage_notes.setdefault(rid, [])
                if note and note not in usage_notes[rid]:
                    usage_notes[rid].append(note)
                if rid not in used_ids:
                    used_ids.append(rid)

        resources_used: list[dict[str, Any]] = []
        for rid in used_ids:
            item = item_index.get(rid, {})
            if not item:
                resources_used.append({"id": rid, "note": "No index entry available."})
                continue

            # Retrieve detailed resource information from archival memory
            summary = ""
            resource_notes = item.get("notes", "")
            if item.get("class") in {"template", "document"}:
                # First try to get JSON entry with RESOURCE_ITEM tag (has notes field)
                rows = self.retrieve_archival(
                    branch_id=root_branch_id or best_branch,
                    query=f'"{rid}"',
                    k=2,  # Get both JSON and YAML entries
                    include_ancestors=True,
                    tags_filter=[RESOURCE_ITEM_TAG],
                )
                if rows:
                    # Look for JSON entry
                    for row in rows:
                        archival_text = row.get("text", "")
                        try:
                            # Parse JSON from archival text to get notes
                            archival_data = json.loads(archival_text)
                            if not resource_notes:
                                resource_notes = archival_data.get("notes", "")
                            if no_budget_limit:
                                summary = _strip_frontmatter(archival_text)
                            else:
                                summary = _truncate(_strip_frontmatter(archival_text), 300)
                            break  # Found JSON entry
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, try YAML entry for summary
                            if not summary:
                                if no_budget_limit:
                                    summary = _strip_frontmatter(archival_text)
                                else:
                                    summary = _truncate(_strip_frontmatter(archival_text), 300)

            resources_used.append(
                {
                    "id": rid,
                    "class": item.get("class", ""),
                    "name": item.get("name", ""),
                    "source": item.get("source", ""),
                    "resource": item.get("resource", ""),
                    "digest": item.get("digest", ""),
                    "staged_path": item.get("staged_path", ""),
                    "dest_path": item.get("dest_path", item.get("container_path", "")),
                    "availability": item.get("availability", ""),
                    "notes": resource_notes,  # Resource description/purpose
                    "summary": summary,
                    "usage_notes": usage_notes.get(rid, []),
                }
            )

        return resources_used

    def _build_paper_sections(
        self,
        idea_text: str,
        idea_summary: str,
        phase0_summary: str,
        core_snapshot: str,
        failure_notes: list[str],
        resources_used: list[dict[str, Any]],
        no_budget_limit: bool,
        best_branch: str | None = None,
    ) -> dict[str, Any]:
        """
        Build paper sections dynamically using LLM to analyze memory and determine
        the appropriate structure and content for the research paper.
        """
        # Collect all available memory data for LLM analysis
        memory_context = {
            "idea_text": idea_text,
            "idea_summary": idea_summary,
            "phase0_summary": phase0_summary,
            "experimental_results": core_snapshot,
            "failure_notes": failure_notes,
            "resources_used": resources_used,
            "best_branch": best_branch,
        }

        # Generate sections using LLM
        sections = self._generate_sections_with_llm(memory_context)

        return sections

    def _generate_sections_with_llm(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """
        Use LLM to analyze memory and generate appropriate paper sections dynamically.

        Args:
            memory_context: Dictionary containing all available memory data

        Returns:
            Dictionary with dynamically generated paper sections
        """
        # If LLM compression is not enabled, fall back to a simpler structure
        if not self.use_llm_compression or self._compression_client is None:
            logger.warning("LLM compression not enabled. Using fallback section structure.")
            return self._build_fallback_sections(memory_context)

        try:
            from ai_scientist.llm import get_response_from_llm

            # Prepare memory summary for LLM
            memory_summary = self._prepare_memory_summary(memory_context)

            # Prompt for LLM to analyze memory and generate sections
            prompt = f"""Analyze the following research memory data and generate appropriate sections for a research paper.

Memory Data:
{memory_summary}

Based on this memory data, determine what sections are needed for a comprehensive research paper and generate the content for each section.

You should return a JSON object where each key is a section name (use descriptive, snake_case names like "title_candidates", "abstract", "methodology", etc.) and each value is the content for that section extracted and synthesized from the memory data.

Important guidelines:
- Include standard research paper sections (title, abstract, problem statement, methodology, experimental results, etc.)
- Extract and synthesize relevant information from the memory data for each section
- Do NOT limit the length of any section - include all relevant information
- If certain information is not available in the memory, you can note that or omit those details
- Be comprehensive and thorough - this is for generating a complete research paper

Return ONLY a valid JSON object, no additional text or formatting."""

            system_message = "You are a research paper writing assistant. Analyze memory data and generate comprehensive paper sections in JSON format."

            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message,
                temperature=0.3,
            )

            if not response:
                logger.warning("LLM returned empty response for section generation. Using fallback.")
                return self._build_fallback_sections(memory_context)

            # Parse JSON response
            try:
                # Try to extract JSON from response if it's wrapped in markdown code blocks
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()

                sections = json.loads(response)

                # Validate that we got a dictionary
                if not isinstance(sections, dict):
                    logger.warning("LLM returned non-dictionary response. Using fallback.")
                    return self._build_fallback_sections(memory_context)

                # Add resources_used back to the sections
                sections["resources_used"] = memory_context["resources_used"]

                logger.info(f"Successfully generated {len(sections)} paper sections using LLM.")
                return sections

            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse LLM response as JSON: {exc}. Using fallback.")
                return self._build_fallback_sections(memory_context)

        except Exception as exc:
            logger.warning(f"Section generation with LLM failed: {exc}. Using fallback.")
            return self._build_fallback_sections(memory_context)

    def _prepare_memory_summary(self, memory_context: dict[str, Any]) -> str:
        """Prepare a formatted summary of memory context for LLM analysis."""
        parts = []

        if memory_context.get("idea_text"):
            parts.append(f"## Research Idea\n{memory_context['idea_text'][:2000]}")

        if memory_context.get("idea_summary"):
            parts.append(f"## Idea Summary\n{memory_context['idea_summary'][:1000]}")

        # Retrieve detailed phase0 information from archival memory
        best_branch = memory_context.get("best_branch")
        if best_branch:
            try:
                # Get phase0 internal info from archival memory
                phase0_rows = self._conn.execute(
                    """SELECT text FROM archival
                    WHERE branch_id = ?
                    AND tags LIKE '%PHASE0_INTERNAL%'
                    LIMIT 1""",
                    (best_branch,)
                ).fetchall()

                if phase0_rows:
                    phase0_full_text = phase0_rows[0][0]
                    # Extract hardware info from the JSON
                    # Look for phase0_history_full.json section which contains environment_context
                    try:
                        # Find the JSON block containing environment_context
                        start_marker = "phase0_history_full.json"
                        json_start_idx = phase0_full_text.find(start_marker)
                        if json_start_idx >= 0:
                            # Find the start of the JSON object after the marker
                            json_start_idx = phase0_full_text.find("{", json_start_idx)
                            if json_start_idx >= 0:
                                # Find the end of the JSON object (balanced braces)
                                brace_count = 0
                                json_end_idx = json_start_idx
                                for i in range(json_start_idx, len(phase0_full_text)):
                                    if phase0_full_text[i] == '{':
                                        brace_count += 1
                                    elif phase0_full_text[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end_idx = i + 1
                                            break

                                if json_end_idx > json_start_idx:
                                    json_text = phase0_full_text[json_start_idx:json_end_idx]
                                    phase0_data = json.loads(json_text)
                                    env_context = phase0_data.get("environment_context", {})

                                    # Format hardware information
                                    hw_info = []
                                    if "cpu_info" in env_context:
                                        cpu = env_context["cpu_info"]
                                        hw_info.append(f"CPU: {cpu.get('model', 'Unknown')}")
                                        hw_info.append(f"Sockets: {cpu.get('sockets', 'N/A')}, Cores per socket: {cpu.get('cores_per_socket', 'N/A')}, Threads per core: {cpu.get('threads_per_core', 'N/A')}")
                                        hw_info.append(f"Total CPUs: {cpu.get('cpus', 'N/A')}, NUMA nodes: {cpu.get('numa_nodes', 'N/A')}")

                                    if "memory_info" in env_context:
                                        mem = env_context["memory_info"]
                                        hw_info.append(f"Memory: Total={mem.get('total', 'N/A')}, Available={mem.get('available', 'N/A')}")

                                    if "gpu_info" in env_context and env_context["gpu_info"]:
                                        hw_info.append(f"GPU: {', '.join(env_context['gpu_info'])}")

                                    if "available_compilers" in env_context:
                                        compilers = env_context["available_compilers"]
                                        compiler_list = [f"{c.get('name', 'unknown')} {c.get('version', '')}" for c in compilers if isinstance(c, dict)]
                                        if compiler_list:
                                            hw_info.append(f"Compilers: {', '.join(compiler_list[:3])}")

                                    if hw_info:
                                        parts.append(f"## Hardware Configuration\n" + "\n".join(hw_info))
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        logger.debug(f"Could not parse phase0 JSON for hardware info: {e}")
            except Exception as e:
                logger.debug(f"Could not retrieve phase0 hardware info: {e}")

        if memory_context.get("phase0_summary"):
            parts.append(f"## Experimental Setup (Phase 0)\n{memory_context['phase0_summary'][:1500]}")

        if memory_context.get("experimental_results"):
            parts.append(f"## Experimental Results\n{memory_context['experimental_results'][:2000]}")

        if memory_context.get("failure_notes"):
            failure_text = "\n".join(memory_context["failure_notes"])
            if failure_text:
                parts.append(f"## Failure Notes\n{failure_text[:1000]}")

        if memory_context.get("resources_used"):
            resources_summary = f"Number of resources used: {len(memory_context['resources_used'])}"
            parts.append(f"## Resources\n{resources_summary}")

        return "\n\n".join(parts)

    def _build_fallback_sections(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """Build a simple fallback section structure when LLM is not available."""
        parsed_idea = _parse_markdown_sections(memory_context.get("idea_text", ""))
        idea_summary = memory_context.get("idea_summary", "")

        def extract_bullet(summary_text: str, bullet_name: str) -> str:
            """Helper to extract a specific bullet point from summary."""
            for line in summary_text.splitlines():
                if line.lstrip().startswith(f"- {bullet_name}:"):
                    return line.split(":", 1)[1].strip()
            return ""

        # Extract basic information
        title_text = parsed_idea.get("Title") or parsed_idea.get("Name") or extract_bullet(idea_summary, "Title") or idea_summary
        abstract_text = parsed_idea.get("Abstract") or parsed_idea.get("Task goal") or extract_bullet(idea_summary, "Purpose") or idea_summary
        hypothesis_text = parsed_idea.get("Short Hypothesis") or parsed_idea.get("Hypothesis") or extract_bullet(idea_summary, "Hypothesis") or idea_summary
        problem_text = parsed_idea.get("Problem Statement") or parsed_idea.get("Motivation") or abstract_text
        method_text = parsed_idea.get("Experiments") or parsed_idea.get("Code") or parsed_idea.get("Method") or extract_bullet(idea_summary, "Method/Variables") or idea_summary
        narrative_text = parsed_idea.get("Related Work") or parsed_idea.get("Narrative") or "Related work positioning, key trade-offs, implications."

        failure_text = "\n".join(memory_context.get("failure_notes", [])) if memory_context.get("failure_notes") else "No failures recorded."

        sections = {
            "title_candidates": title_text,
            "abstract_material": abstract_text,
            "problem_statement": problem_text,
            "hypothesis": hypothesis_text,
            "method": method_text,
            "experimental_setup": memory_context.get("phase0_summary", ""),
            "phase0_internal_info_summary": memory_context.get("phase0_summary", ""),
            "results": memory_context.get("experimental_results", ""),
            "failure_modes_timeline": failure_text,
            "narrative_bullets": narrative_text,
            "resources_used": memory_context.get("resources_used", []),
        }

        return sections

    def generate_final_memory_for_paper(
        self,
        run_dir: str | Path,
        root_branch_id: str,
        best_branch_id: str | None,
        artifacts_index: dict[str, Any] | None = None,
        no_budget_limit: bool = True,
    ) -> dict[str, Any]:
        """Generate final memory for paper writeup.

        Args:
            run_dir: Path to run directory
            root_branch_id: Root branch ID
            best_branch_id: Best branch ID (or None to use root)
            artifacts_index: Optional artifacts index dict
            no_budget_limit: If True (default), do not truncate/compress content

        Returns:
            Dictionary with all paper sections
        """
        run_dir = Path(run_dir)
        memory_dir = run_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        best_branch = best_branch_id or root_branch_id

        # Collect all necessary data using helper methods
        core_snapshot = self._collect_experimental_results(best_branch, no_budget_limit)
        idea_text, idea_summary = self._get_idea_text_and_summary(run_dir, best_branch, no_budget_limit)
        phase0_summary = self._get_phase0_summary(memory_dir, best_branch, no_budget_limit)
        failure_notes = self._collect_failure_notes(best_branch)
        resource_snapshot, resource_index, item_index = self._load_resource_snapshot_and_index(
            memory_dir, root_branch_id, best_branch
        )
        resources_used = self._collect_resource_usage(root_branch_id, best_branch, item_index, no_budget_limit)

        # Build paper sections
        sections = self._build_paper_sections(
            idea_text,
            idea_summary,
            phase0_summary,
            core_snapshot,
            failure_notes,
            resources_used,
            no_budget_limit,
            best_branch=best_branch,
        )

        if artifacts_index:
            sections["artifacts_index"] = artifacts_index

        # Build final writeup memory payload
        phase0_path = memory_dir / "phase0_internal_info.json"
        phase0_payload = {}
        if phase0_path.exists():
            try:
                phase0_payload = json.loads(phase0_path.read_text(encoding="utf-8"))
            except Exception:
                phase0_payload = {}


        # Build final writeup memory payload (condensed but reproducible)
        phase0_path = memory_dir / "phase0_internal_info.json"
        phase0_payload = {}
        if phase0_path.exists():
            try:
                phase0_payload = json.loads(phase0_path.read_text(encoding="utf-8"))
            except Exception:
                phase0_payload = {}
        idea_path = run_dir / "idea.md"
        if not idea_path.exists():
            idea_path = Path(_cfg_get(self.config, "desc_file", "")) if _cfg_get(self.config, "desc_file", "") else idea_path

        resources_section = {
            "resource_file": {
                "path": resource_snapshot.get("resource_file") or resource_index.get("resource_file", ""),
                "sha": resource_snapshot.get("resource_file_sha") or resource_index.get("resource_file_sha", ""),
                "digest": resource_index.get("resource_digest", ""),
            },
            "normalized": resource_snapshot.get("normalized") or resource_index.get("normalized", {}),
            "items": resource_snapshot.get("items") or resource_index.get("items", []),
            "used": resources_used,
        }
        branch_chain = self._branch_chain(best_branch)
        provenance = []
        for bid in branch_chain:
            row = self._conn.execute(
                "SELECT node_uid FROM branches WHERE id=?", (bid,)
            ).fetchone()
            if row and row["node_uid"]:
                provenance.append(row["node_uid"])
            else:
                provenance.append(bid)

        # Fetch events to extract method changes and results notes
        event_rows = self._fetch_events(branch_chain, 200)
        method_changes = []
        results_notes = []
        for row in event_rows:
            kind = str(row["kind"]).lower()
            if kind == "node_created":
                # Full content if no_budget_limit
                if no_budget_limit:
                    method_changes.append(row["text"] or "")
                else:
                    method_changes.append(self._compress(row["text"] or "", 800, "method change note", branch_id=best_branch))
            if kind == "node_result":
                if no_budget_limit:
                    results_notes.append(row["text"] or "")
                else:
                    results_notes.append(self._compress(row["text"] or "", 800, "results note", branch_id=best_branch))

        # === Collect 3-tier memory from best branch (with limits and compression) ===
        # Core Memory (short-term / always-injected context) - all keys, compressed values
        core_memory_data: dict[str, Any] = {}
        for bid in branch_chain:
            core_rows = self._conn.execute(
                "SELECT key, value, updated_at FROM core_kv WHERE branch_id=?",
                (bid,),
            ).fetchall()
            meta_rows = self._conn.execute(
                "SELECT key, importance FROM core_meta WHERE branch_id=?",
                (bid,),
            ).fetchall()
            importance_map = {r["key"]: r["importance"] for r in meta_rows}
            for row in core_rows:
                key = row["key"]
                if key not in core_memory_data:
                    value = row["value"] or ""
                    # Compress value if over limit (unless no_budget_limit)
                    if not no_budget_limit and len(value) > self.writeup_core_value_max_chars:
                        value = self._compress(value, self.writeup_core_value_max_chars, f"core memory: {key}", branch_id=best_branch)
                    core_memory_data[key] = {
                        "value": value,
                        "importance": importance_map.get(key, 3),
                    }

        # Recall Memory (event timeline) - limited entries, compressed text
        recall_memory_data: list[dict[str, Any]] = []
        recall_subset = event_rows[:self.writeup_recall_limit] if not no_budget_limit else event_rows
        for row in recall_subset:
            text = row["text"] or ""
            if not no_budget_limit and len(text) > self.writeup_recall_text_max_chars:
                text = self._compress(text, self.writeup_recall_text_max_chars, "recall event", branch_id=best_branch)
            recall_memory_data.append({
                "ts": row["created_at"],
                "kind": row["kind"],
                "text": text,
            })

        # Archival Memory (long-term) - limited entries, compressed text
        archival_k = 100 if no_budget_limit else self.writeup_archival_limit
        archival_memory_data = self.retrieve_archival(
            branch_id=best_branch,
            query="",
            k=archival_k,
            include_ancestors=True,
            log_event=False,
        )
        archival_memory_list: list[dict[str, Any]] = []
        for row in archival_memory_data:
            text = row.get("text", "") or ""
            if not no_budget_limit and len(text) > self.writeup_archival_text_max_chars:
                text = self._compress(text, self.writeup_archival_text_max_chars, "archival entry", branch_id=best_branch)
            archival_memory_list.append({
                "id": row.get("id"),
                "text": text,
                "tags": row.get("tags"),
            })

        writeup_memory = {
            "run_id": self.run_id or run_dir.name,
            "idea": {
                "summary": idea_summary,
                "path": str(idea_path) if idea_path and idea_path.exists() else "",
                "content": idea_text if no_budget_limit else self._compress(idea_text or "", 4000, "idea content", branch_id=best_branch),
            },
            "phase0_env": {
                "summary": phase0_summary,
                "path": str(phase0_path) if phase0_path.exists() else "",
                "raw": phase0_payload,
            },
            "resources": resources_section,
            "method_changes": method_changes,
            "experiments": {
                "phase0_plan": phase0_payload.get("phase0_payload", {}),
                "phase0_command": phase0_payload.get("command", ""),
                "workspace_dir": str(run_dir),
            },
            "results": {
                "core_snapshot": core_snapshot,
                "notes": results_notes,
                "artifacts": artifacts_index or {},
            },
            "negative_results": failure_notes,
            "provenance": provenance,
            # 3-tier memory from best branch
            "core_memory": core_memory_data,
            "recall_memory": recall_memory_data,
            "archival_memory": archival_memory_list,
        }
        writeup_path = memory_dir / "final_writeup_memory.json"
        writeup_path.write_text(json.dumps(writeup_memory, indent=2), encoding="utf-8")

        # Generate markdown dynamically from sections
        md_sections = ["# Final Memory For Paper", ""]

        # Convert section keys to readable headers and add content
        for key, value in sections.items():
            if key == "resources_used":
                # Handle resources separately below
                continue

            # Convert snake_case to Title Case for headers
            header = " / ".join(word.capitalize() for word in key.replace("_", " ").split())

            md_sections.append(f"## {header}")

            # Handle different value types
            if isinstance(value, str):
                md_sections.append(value)
            elif isinstance(value, (list, dict)):
                # For complex types, convert to readable format
                md_sections.append(json.dumps(value, indent=2))
            else:
                md_sections.append(str(value))

            md_sections.append("")

        # Add resources section with improved formatting
        md_sections.append("## Resources Used")
        md_sections.append("")
        if resources_used:
            for i, entry in enumerate(resources_used, 1):
                resource_name = entry.get("name", "unknown")
                resource_class = entry.get("class", "unknown")

                # Create a header for each resource
                md_sections.append(f"### {i}. {resource_name} ({resource_class})")
                md_sections.append("")

                # Add description/purpose if available
                resource_notes = entry.get("notes", "")
                if resource_notes:
                    md_sections.append(f"**Description**: {resource_notes}")
                    md_sections.append("")

                # Add structured information
                if entry.get("source"):
                    md_sections.append(f"- **Source**: {entry['source']}")

                resource_name_field = entry.get("resource", "")
                if resource_name_field:
                    md_sections.append(f"- **Resource**: {resource_name_field}")

                original_path = entry.get("staged_path") or entry.get("dest_path", "")
                if original_path:
                    md_sections.append(f"- **Path**: {original_path}")

                # Add digest (shortened)
                digest = entry.get("digest", "")
                if digest:
                    digest_short = digest[:16] + "..." if len(digest) > 16 else digest
                    md_sections.append(f"- **Digest**: `{digest_short}`")

                # Add usage context if available
                usage_notes = entry.get("usage_notes") or []
                if usage_notes:
                    md_sections.append(f"- **Usage Context**: {', '.join(usage_notes)}")

                md_sections.append("")
        else:
            md_sections.append("No explicit resource usage recorded.")
            md_sections.append("")
        md_path = memory_dir / str(_cfg_get(self.config, "final_memory_filename_md", "final_memory_for_paper.md"))
        json_path = memory_dir / str(_cfg_get(self.config, "final_memory_filename_json", "final_memory_for_paper.json"))
        md_path.write_text("\n".join(md_sections), encoding="utf-8")
        json_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")
        return sections
