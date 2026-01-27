from __future__ import annotations

import ast
import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Generator, Iterable, Sequence

from ai_scientist.prompt_loader import PromptNotFoundError, load_prompt

from .resource_memory import (
    RESOURCE_DIGEST_KEY,
    RESOURCE_INDEX_JSON_KEY,
    RESOURCE_INDEX_KEY,
    RESOURCE_ITEM_TAG,
    RESOURCE_USED_KEY,
    RESOURCE_USED_TAG,
    track_resource_usage,
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


def _row_get(row: sqlite3.Row | None, key: str, default: Any = None) -> Any:
    """Safely get a value from a sqlite3.Row object, similar to dict.get()."""
    if row is None:
        return default
    try:
        value = row[key]
        return value if value is not None else default
    except (KeyError, IndexError):
        return default


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

# Phase name mapping for memory event logging
# Maps internal task_hint values to human-readable phase names
# Phase structure:
#   Phase 0: Setup/Planning (idea loading, resource indexing, planning)
#   Phase 1: Environment Setup (container setup, dependencies installation)
#   Phase 2: Code Implementation (draft, debug, improve)
#   Phase 3: Compile (build the code)
#   Phase 4: Execute/Validate (run experiments, analyze results)
PHASE_NAME_MAP: dict[str, str] = {
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
    "datasets_successfully_tested": "Phase 4: Datasets Tested",
    # Post-Phase 4 analysis tasks (displayed as-is without phase number)
    # metrics_extraction, parse_metrics, plotting_code, seed_plotting,
    # vlm_analysis, stage_completion
    # are not mapped - they will be displayed as-is
}


def _get_phase_display_name(task_hint: str | None) -> str | None:
    """Convert internal task_hint to human-readable phase name.

    Args:
        task_hint: Internal task hint string (e.g., "draft", "debug")

    Returns:
        Human-readable phase name (e.g., "Phase 2: Draft Implementation")
        or the original task_hint if not in the mapping (for post-Phase 4 tasks)
        or None if task_hint is None
    """
    if task_hint is None:
        return None

    # If already a "Phase X:" format, return as-is
    if task_hint.startswith("Phase "):
        return task_hint

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


# Prompt names for load_prompt (relative to prompt/ directory, without .txt extension)
COMPRESSION_PROMPT_NAME = "config/memory/compression"
COMPRESSION_SYSTEM_MESSAGE_NAME = "config/memory/compression_system_message"
IMPORTANCE_EVALUATION_PROMPT_NAME = "config/memory/importance_evaluation"
IMPORTANCE_EVALUATION_SYSTEM_MESSAGE_NAME = "config/memory/importance_evaluation_system_message"
CONSOLIDATION_PROMPT_NAME = "config/memory/consolidation"
CONSOLIDATION_SYSTEM_MESSAGE_NAME = "config/memory/consolidation_system_message"
PAPER_SECTION_GENERATION_NAME = "config/memory/paper_section_generation"
PAPER_SECTION_GENERATION_SYSTEM_MESSAGE_NAME = "config/memory/paper_section_generation_system_message"
PAPER_SECTION_OUTLINE_NAME = "config/memory/paper_section_outline"
PAPER_SECTION_OUTLINE_SYSTEM_MESSAGE_NAME = "config/memory/paper_section_outline_system_message"
PAPER_SECTION_FILL_NAME = "config/memory/paper_section_fill"
PAPER_SECTION_FILL_SYSTEM_MESSAGE_NAME = "config/memory/paper_section_fill_system_message"
KEYWORD_EXTRACTION_NAME = "config/memory/keyword_extraction"
KEYWORD_EXTRACTION_SYSTEM_MESSAGE_NAME = "config/memory/keyword_extraction_system_message"


def _try_load_prompt(name: str, description: str = "prompt") -> str | None:
    """Try to load a prompt template using load_prompt, returning None if not found.

    Args:
        name: Prompt name (relative to prompt/ directory, without .txt extension)
        description: Description of the prompt for logging purposes

    Returns:
        The prompt text if loaded successfully, None otherwise
    """
    if not name:
        return None
    try:
        return load_prompt(name)
    except PromptNotFoundError:
        logger.warning("%s file not found: %s", description.capitalize(), name)
        return None
    except Exception as exc:
        logger.warning("Failed to read %s: %s", description, exc)
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
        prompt_template = _try_load_prompt(COMPRESSION_PROMPT_NAME, "compression prompt")
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
            
            system_message = _try_load_prompt(COMPRESSION_SYSTEM_MESSAGE_NAME, "compression system message")
            if system_message is None:
                system_message = "You are a text compression assistant. Output only the compressed text, no explanations."
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=model,
                system_message=system_message.strip(),
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


def _to_md_str(value: Any) -> str:
    """Convert value to markdown-safe string for joining."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
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

        # Current phase tracking for memory event logging
        self._current_phase: str | None = None

        # LLM compression settings
        self.use_llm_compression = bool(
            _cfg_get(self.config, "use_llm_compression", False)
        )
        self.compression_model = str(
            _cfg_get(self.config, "compression_model", "gpt-4o-mini")
        )
        self._compression_prompt_template = _try_load_prompt(
            COMPRESSION_PROMPT_NAME, "compression prompt"
        )
        
        # Memory retrieval/display budgets
        self.archival_snippet_budget_chars = int(
            _cfg_get(self.config, "archival_snippet_budget_chars", 6000)
        )
        self.results_budget_chars = int(
            _cfg_get(self.config, "results_budget_chars", 4000)
        )

        # Writeup memory limits (for final_writeup_memory.json)
        self.writeup_recall_limit = int(_cfg_get(self.config, "writeup_recall_limit", 10))
        self.writeup_archival_limit = int(_cfg_get(self.config, "writeup_archival_limit", 10))
        self.writeup_core_value_max_chars = int(_cfg_get(self.config, "writeup_core_value_max_chars", 500))
        self.writeup_recall_text_max_chars = int(_cfg_get(self.config, "writeup_recall_text_max_chars", 300))
        self.writeup_archival_text_max_chars = int(_cfg_get(self.config, "writeup_archival_text_max_chars", 400))
        self.paper_section_mode = str(_cfg_get(self.config, "paper_section_mode", "memory_summary")).strip().lower()
        self.paper_section_count = int(_cfg_get(self.config, "paper_section_count", 12))

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

    def set_current_phase(self, phase: str | None) -> None:
        """Set the current phase for memory event logging.

        All subsequent memory operations will be tagged with this phase
        unless they explicitly specify a different phase.

        Args:
            phase: The phase name (e.g., "draft", "execution_review", "phase1_iterative")
                   or None to clear the current phase.
        """
        self._current_phase = phase

    @contextmanager
    def phase_context(self, phase: str) -> Generator[None, None, None]:
        """Context manager for scoped phase tracking.

        All memory operations within this context will be tagged with the
        specified phase unless they explicitly specify a different phase.

        Example:
            with mem.phase_context("draft"):
                mem.set_core(branch_id, "key", "value")  # Tagged with phase="draft"

        Args:
            phase: The phase name for this context.
        """
        previous_phase = self._current_phase
        self._current_phase = phase
        try:
            yield
        finally:
            self._current_phase = previous_phase

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
        # Use _current_phase as fallback if phase is not explicitly provided
        raw_phase = phase if phase is not None else self._current_phase
        # Convert to human-readable phase name
        effective_phase = _get_phase_display_name(raw_phase)
        payload: dict[str, Any] = {
            "ts": _now_ts(),
            "op": op,
            "memory_type": memory_type,
            "branch_id": branch_id,
            "node_id": node_id,
            "phase": effective_phase,
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

        # Inherited memory management tables (Copy-on-Write for branch inheritance)
        # Stores event IDs that are excluded from inherited view for a branch
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inherited_exclusions (
                branch_id TEXT,
                excluded_event_id INTEGER,
                excluded_at REAL,
                PRIMARY KEY (branch_id, excluded_event_id)
            )
            """
        )
        # Stores summaries of consolidated inherited events for a branch
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inherited_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_id TEXT,
                summary_text TEXT,
                summarized_event_ids TEXT,
                kind TEXT,
                created_at REAL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_inherited_exclusions_branch ON inherited_exclusions(branch_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_inherited_summaries_branch ON inherited_summaries(branch_id)"
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
        # Force WAL checkpoint so new branch is immediately visible to other processes
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass  # Ignore checkpoint errors as they are non-critical
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

    def _resolve_branch_id(self, node_id: str | None, *, retry: bool = True) -> str | None:
        """Resolve node_id to branch_id.

        In multi-process WAL mode, a branch created by another process may not be
        immediately visible. This method retries with WAL refresh if not found.

        Args:
            node_id: The node ID to resolve
            retry: If True, retry with WAL refresh when not found (default True)

        Returns:
            The branch_id if found, None otherwise
        """
        if not node_id:
            return None
        if self._branch_exists(node_id):
            return node_id
        row = self._conn.execute(
            "SELECT id FROM branches WHERE node_uid=?", (node_id,)
        ).fetchone()
        if row:
            return row["id"]

        # Branch not found - in WAL mode, it may exist but not be visible yet
        if retry:
            import time
            for attempt in range(3):
                # Force WAL checkpoint to make recent writes visible
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                time.sleep(0.1 * (attempt + 1))  # 0.1s, 0.2s, 0.3s delays
                # Retry the lookup
                if self._branch_exists(node_id):
                    return node_id
                row = self._conn.execute(
                    "SELECT id FROM branches WHERE node_uid=?", (node_id,)
                ).fetchone()
                if row:
                    return row["id"]
            logger.debug(f"Branch not found after retries: node_id={node_id}")

        return None

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
        removable_keys = [
            (importance_map.get(key, 3), key)
            for key in entries
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
        for key in list(entries.keys()):
            self._delete_core_key(
                branch_id,
                key,
                reason="digest",
                op_name="core_digest_compact",
            )
        available = self.core_max_chars
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

    def _count_events(
        self,
        branch_id: str,
        include_ancestors: bool = True,
        respect_exclusions: bool = False,
    ) -> int:
        """Count the number of recall events for a branch.

        Args:
            branch_id: Branch ID to count events for
            include_ancestors: If True, include events from ancestor branches
            respect_exclusions: If True, exclude events in inherited_exclusions table

        Returns:
            Count of recall events
        """
        if include_ancestors:
            branch_ids = self._branch_chain(branch_id)
        else:
            branch_ids = [branch_id]
        if not branch_ids:
            return 0

        placeholders = ",".join(["?"] * len(branch_ids))
        params: list = list(branch_ids)

        # Build exclusion clause if needed
        exclusion_clause = ""
        if respect_exclusions:
            excluded_event_ids = self._get_inherited_exclusions(branch_id)
            if excluded_event_ids:
                exclusion_placeholders = ",".join(["?"] * len(excluded_event_ids))
                exclusion_clause = f" AND id NOT IN ({exclusion_placeholders})"
                params.extend(excluded_event_ids)

        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM events WHERE branch_id IN ({placeholders}){exclusion_clause}",
            params,
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

    def check_memory_pressure(self, branch_id: str, *, phase: str | None = None) -> dict:
        """
        Check memory pressure and return status with recommendations.

        Memory pressure is calculated based on:
        - Core memory usage vs core_max_chars
        - Recall event count vs recall_max_events
        - Archival memory size (informational)

        Args:
            branch_id: The branch to check memory pressure for.
            phase: Optional phase name for logging (e.g., "draft", "phase1_iterative").

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
            phase=phase,
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
        return _try_load_prompt(IMPORTANCE_EVALUATION_PROMPT_NAME, "importance evaluation prompt")

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

            system_message = _try_load_prompt(IMPORTANCE_EVALUATION_SYSTEM_MESSAGE_NAME, "importance evaluation system message")
            if system_message is None:
                system_message = "You are a memory management assistant. Output only valid JSON."
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message.strip(),
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
        return _try_load_prompt(CONSOLIDATION_PROMPT_NAME, "consolidation prompt")

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

            system_message = _try_load_prompt(CONSOLIDATION_SYSTEM_MESSAGE_NAME, "consolidation system message")
            if system_message is None:
                system_message = "You are a memory consolidation assistant. Output only the consolidated text."
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message.strip(),
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

        # Only consolidate events for this specific branch, not parent branches.
        # Using _branch_chain would include parent nodes' events, which would
        # incorrectly delete parent nodes' events when consolidating a child node.
        all_events = self._conn.execute(
            """
            SELECT id, branch_id, kind, text, tags, created_at
            FROM events
            WHERE branch_id = ?
            ORDER BY created_at DESC
            """,
            (branch_id,),
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

    def consolidate_inherited_memory(
        self,
        branch_id: str,
        max_events: int | None = None,
    ) -> int:
        """
        Consolidate inherited recall events using Copy-on-Write semantics.

        This method handles memory inheritance consolidation without modifying
        ancestor branches. When the total inherited events exceed the threshold:
        1. Older inherited events are summarized using LLM
        2. Summaries are stored in inherited_summaries (local to this branch)
        3. Excluded event IDs are stored in inherited_exclusions
        4. Ancestor data remains unchanged (Copy-on-Write)

        Args:
            branch_id: Branch ID to consolidate inherited memory for
            max_events: Maximum events to keep (default: recall_max_events)

        Returns:
            Number of inherited events consolidated (excluded from view)
        """
        if not branch_id:
            return 0

        max_events = max_events or self.recall_max_events
        threshold = int(max_events * self.recall_consolidation_threshold)

        # Get the full branch chain (ancestors)
        branch_chain = self._branch_chain(branch_id)
        if len(branch_chain) <= 1:
            # No ancestors to inherit from
            return 0

        # Get IDs of ancestor branches (excluding self)
        ancestor_branch_ids = branch_chain[1:]  # Skip self (first element)
        if not ancestor_branch_ids:
            return 0

        # Get already excluded event IDs for this branch
        existing_exclusions = set()
        rows = self._conn.execute(
            "SELECT excluded_event_id FROM inherited_exclusions WHERE branch_id = ?",
            (branch_id,),
        ).fetchall()
        for row in rows:
            existing_exclusions.add(row[0])

        # Fetch all inherited events (from ancestors only, respecting existing exclusions)
        placeholders = ",".join(["?"] * len(ancestor_branch_ids))
        exclusion_clause = ""
        params: list = list(ancestor_branch_ids)
        if existing_exclusions:
            exclusion_placeholders = ",".join(["?"] * len(existing_exclusions))
            exclusion_clause = f" AND id NOT IN ({exclusion_placeholders})"
            params.extend(existing_exclusions)

        inherited_events = self._conn.execute(
            f"""
            SELECT id, branch_id, kind, text, tags, created_at
            FROM events
            WHERE branch_id IN ({placeholders}){exclusion_clause}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()

        if len(inherited_events) <= threshold:
            return 0

        # Events to keep (most recent)
        events_to_keep = inherited_events[:max_events]
        # Events to consolidate (older ones) - these are from ancestors
        events_to_consolidate = inherited_events[max_events:]

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
                "branch_id": event["branch_id"],
                "kind": kind,
                "text": event["text"],
                "created_at": event["created_at"],
            })

        consolidated_count = 0
        event_ids_to_exclude: list[int] = []
        now = time.time()

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

            # Create consolidated summary using LLM
            context = f"inherited recall events of type '{kind}'"
            summary = self._consolidate_with_llm(
                entries,
                context,
                max_chars=self.archival_max_chars,
            )

            if summary:
                # Write summary to inherited_summaries table (local to this branch)
                time_range = ""
                if group_events:
                    oldest = min(e["created_at"] for e in group_events)
                    newest = max(e["created_at"] for e in group_events)
                    time_range = f" (from {oldest:.0f} to {newest:.0f})"

                summary_text = (
                    f"[Inherited Summary - {kind}]{time_range}\n"
                    f"Consolidated {len(group_events)} events from ancestors:\n\n"
                    f"{summary}"
                )

                event_ids_json = json.dumps([e["id"] for e in group_events])
                self._conn.execute(
                    """
                    INSERT INTO inherited_summaries
                    (branch_id, summary_text, summarized_event_ids, kind, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (branch_id, summary_text, event_ids_json, kind, now),
                )

            # Mark events for exclusion (Copy-on-Write: don't delete, just exclude)
            event_ids_to_exclude.extend(e["id"] for e in group_events)
            consolidated_count += len(group_events)

        # Insert exclusions (these events will be filtered out for this branch)
        if event_ids_to_exclude:
            for event_id in event_ids_to_exclude:
                try:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO inherited_exclusions
                        (branch_id, excluded_event_id, excluded_at)
                        VALUES (?, ?, ?)
                        """,
                        (branch_id, event_id, now),
                    )
                except Exception:
                    pass  # Ignore duplicate key errors
            self._conn.commit()

        self._log_memory_event(
            "consolidate_inherited_memory",
            "consolidation",
            branch_id=branch_id,
            details={
                "total_inherited_events": len(inherited_events),
                "events_kept": len(events_to_keep),
                "events_consolidated": consolidated_count,
                "groups_processed": len(event_groups),
                "ancestor_branches": len(ancestor_branch_ids),
            },
        )

        return consolidated_count

    def _get_inherited_exclusions(self, branch_id: str) -> set[int]:
        """Get the set of event IDs excluded from inheritance for a branch."""
        if not branch_id:
            return set()
        rows = self._conn.execute(
            "SELECT excluded_event_id FROM inherited_exclusions WHERE branch_id = ?",
            (branch_id,),
        ).fetchall()
        return {row[0] for row in rows}

    def _get_inherited_summaries(self, branch_id: str) -> list[dict]:
        """Get inherited summaries for a branch."""
        if not branch_id:
            return []
        rows = self._conn.execute(
            """
            SELECT id, summary_text, summarized_event_ids, kind, created_at
            FROM inherited_summaries
            WHERE branch_id = ?
            ORDER BY created_at DESC
            """,
            (branch_id,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "summary_text": row["summary_text"],
                "summarized_event_ids": json.loads(row["summarized_event_ids"] or "[]"),
                "kind": row["kind"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

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
        new_pressure = self.check_memory_pressure(branch_id, phase=task_hint)

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
    ) -> bool:
        """Write an event to the database.

        Returns:
            True if the event was written, False if skipped due to duplicate.
        """
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
                return False

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
                # First, try to consolidate inherited memory (Copy-on-Write)
                # This preserves ancestor data while reducing the effective recall count
                try:
                    self.consolidate_inherited_memory(branch_id)
                except Exception as exc:
                    logger.warning("Auto-consolidation of inherited memory failed: %s", exc)

                # Then consolidate own events if still over threshold
                own_count = self._count_events(branch_id, include_ancestors=False)
                own_threshold = int(self.recall_max_events * self.recall_consolidation_threshold)
                if own_count > own_threshold:
                    try:
                        self.consolidate_recall_events(branch_id)
                    except Exception as exc:
                        logger.warning("Auto-consolidation of recall events failed: %s", exc)

        return True

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

    def _fetch_events(
        self,
        branch_ids: Sequence[str],
        limit: int,
        current_branch_id: str | None = None,
    ) -> list[sqlite3.Row]:
        """Fetch recall events from branches with inherited exclusion support.

        Args:
            branch_ids: List of branch IDs to fetch events from (typically branch chain)
            limit: Maximum number of events to return
            current_branch_id: If provided, exclude events that have been consolidated
                              for this branch via inherited_exclusions table

        Returns:
            List of event rows ordered by created_at DESC
        """
        if not branch_ids:
            return []

        # Get inherited exclusions if current_branch_id is provided
        excluded_event_ids: set[int] = set()
        if current_branch_id:
            excluded_event_ids = self._get_inherited_exclusions(current_branch_id)

        placeholders = ",".join(["?"] * len(branch_ids))
        params: list = list(branch_ids)

        # Build exclusion clause if needed
        exclusion_clause = ""
        if excluded_event_ids:
            exclusion_placeholders = ",".join(["?"] * len(excluded_event_ids))
            exclusion_clause = f" AND id NOT IN ({exclusion_placeholders})"
            params.extend(excluded_event_ids)

        params.append(limit)

        query = (
            f"SELECT id, kind, text, tags, created_at FROM events "
            f"WHERE branch_id IN ({placeholders}){exclusion_clause} "
            "ORDER BY created_at DESC LIMIT ?"
        )
        rows = self._conn.execute(query, params).fetchall()
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
        # Use explicit branch_id if provided and valid, otherwise resolve from node_id
        # This is important because node_uid may not be set on the branch yet during processing
        explicit_branch_id = payload.get("branch_id")
        if explicit_branch_id and self._branch_exists(explicit_branch_id):
            branch_id = explicit_branch_id
        else:
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
        written = self.write_event(
            branch_id,
            kind,
            summary,
            tags=tag_list,
            log_event=False,
            task_hint=task_hint,
            memory_size=memory_size,
        )
        # Only log if the event was actually written (not skipped due to duplicate)
        if written:
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
        if tags is not None or meta is not None:
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
        phase: str | None = None,
    ) -> None:
        """Fork a new branch from parent.

        Args:
            parent_node_id: The parent node's ID (or None for root-level nodes)
            child_node_id: The child node's ID
            ancestor_chain: Optional list of ancestor node IDs from root to parent
                           (e.g., [grandparent_id, parent_id]). If provided, missing
                           branches will be created in the correct order to preserve
                           the full tree structure.
            phase: Optional phase name for the node fork operation. If not provided,
                   defaults to "tree_structure" to indicate tree structure operations.
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
        # Use provided phase, or fall back to _current_phase, or default to "tree_structure"
        effective_phase = phase if phase is not None else (self._current_phase or "tree_structure")
        self._log_memory_event(
            "mem_node_fork",
            "node",
            branch_id=branch_id,
            node_id=child_node_id,
            phase=effective_phase,
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
            if "branch_id" not in event:
                event["branch_id"] = branch_id
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
                if "branch_id" not in meta:
                    meta = {**meta, "branch_id": branch_id}
                self.mem_archival_write(str(text), tags=list(tags or []), meta=meta)

    def apply_llm_memory_updates(
        self,
        branch_id: str,
        updates: dict,
        node_id: str | None = None,
        phase: str | None = None,
    ) -> dict:
        """Apply memory updates from LLM response.

        This method processes memory update instructions embedded in LLM responses
        and applies them to the appropriate memory layers.

        Args:
            branch_id: The branch ID to apply updates to.
            updates: A dict containing memory updates with optional keys:
                - "core": dict of key-value pairs to set in core memory
                - "core_delete": list of keys to delete from core memory
                - "core_get": list of keys to retrieve from core memory
                - "archival": list of dicts with "text" and optional "tags"
                - "archival_update": list of dicts with "id", "text", and optional "tags"
                - "archival_search": dict with "query" and optional "k", "tags"
                - "recall": dict with event data to append to recall
                - "recall_search": dict with "query" and optional "k"
                - "consolidate": bool to trigger memory consolidation
            node_id: Optional node ID for logging.
            phase: Optional phase name for logging.

        Returns:
            A dict containing results from read operations (get, search).
        """
        if not updates or not isinstance(updates, dict):
            return {}

        use_phase = phase or self._current_phase
        results: dict = {}

        # Core memory SET operations
        if "core" in updates and isinstance(updates["core"], dict):
            for key, value in updates["core"].items():
                importance = 4  # LLM insights are moderately important by default
                ttl = None
                actual_value = value

                # Support extended format: {"value": "...", "importance": 5, "ttl": "..."}
                if isinstance(value, dict) and "value" in value:
                    actual_value = value.get("value")
                    importance = _coerce_importance(value.get("importance", 4))
                    ttl = value.get("ttl")
                else:
                    actual_value = str(value)

                self.set_core(
                    branch_id,
                    key,
                    actual_value,
                    importance=importance,
                    ttl=ttl,
                    op_name="llm_memory_update",
                    phase=use_phase,
                    node_id=node_id,
                )

        # Core memory GET operations
        if "core_get" in updates:
            keys = updates["core_get"]
            if isinstance(keys, list):
                results["core_get"] = {}
                for key in keys:
                    value = self.get_core(branch_id, key, log_event=False)
                    results["core_get"][key] = value

        # Core memory DELETE/EVICT operations
        if "core_delete" in updates:
            keys = updates["core_delete"]
            if isinstance(keys, list):
                for key in keys:
                    self._delete_core_key(
                        branch_id,
                        key,
                        reason="llm_requested",
                        log_event=True,
                        node_id=node_id,
                    )
            elif isinstance(keys, str):
                self._delete_core_key(
                    branch_id,
                    keys,
                    reason="llm_requested",
                    log_event=True,
                    node_id=node_id,
                )

        # Archival memory WRITE operations
        if "archival" in updates and isinstance(updates["archival"], list):
            for item in updates["archival"]:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "")
                if not text:
                    continue
                tags = item.get("tags", [])
                if not isinstance(tags, list):
                    tags = [str(tags)] if tags else []
                # Add LLM_INSIGHT tag to identify LLM-generated archival entries
                if "LLM_INSIGHT" not in tags:
                    tags = ["LLM_INSIGHT"] + tags
                meta = item.get("meta", {})
                if not isinstance(meta, dict):
                    meta = {}
                meta.update({
                    "node_id": node_id,
                    "branch_id": branch_id,
                    "phase": use_phase,
                    "source": "llm",
                })
                self.mem_archival_write(str(text), tags=tags, meta=meta)

        # Archival memory UPDATE operations
        if "archival_update" in updates and isinstance(updates["archival_update"], list):
            for item in updates["archival_update"]:
                if not isinstance(item, dict):
                    continue
                record_id = item.get("id")
                if not record_id:
                    continue
                text = item.get("text")
                tags = item.get("tags")
                self.mem_archival_update(record_id, text=text, tags=tags)

        # Archival memory SEARCH operations
        if "archival_search" in updates and isinstance(updates["archival_search"], dict):
            search_params = updates["archival_search"]
            query = search_params.get("query", "")
            k = search_params.get("k", 10)
            tags_filter = search_params.get("tags")
            if query:
                search_results = self.retrieve_archival(
                    branch_id=branch_id,
                    query=query,
                    k=k,
                    include_ancestors=True,
                    tags_filter=tags_filter,
                    log_event=True,
                )
                results["archival_search"] = [
                    {"id": r.get("id"), "text": r.get("text", "")[:500], "tags": r.get("tags", [])}
                    for r in search_results
                ]

        # Recall memory APPEND operations
        if "recall" in updates and isinstance(updates["recall"], dict):
            event = dict(updates["recall"])
            event["node_id"] = node_id
            event["branch_id"] = branch_id
            event["source"] = "llm"
            if "phase" not in event and use_phase:
                event["phase"] = use_phase
            if "kind" not in event:
                event["kind"] = "llm_insight"
            self.mem_recall_append(event)

        # Recall memory SEARCH operations
        if "recall_search" in updates and isinstance(updates["recall_search"], dict):
            search_params = updates["recall_search"]
            query = search_params.get("query", "")
            k = search_params.get("k", 20)
            if query:
                search_results = self.mem_recall_search(query, k=k)
                results["recall_search"] = [
                    {"kind": r.get("kind"), "summary": str(r.get("summary", ""))[:300], "ts": r.get("ts")}
                    for r in search_results
                ]

        # Recall memory EVICT operations (move to archival before deleting)
        if "recall_evict" in updates:
            evict_params = updates["recall_evict"]
            evicted_count = 0
            archived_count = 0
            if isinstance(evict_params, dict):
                ids_to_archive: list[str] = []

                # Collect IDs to evict by explicit IDs
                ids_to_evict = evict_params.get("ids", [])
                if ids_to_evict and isinstance(ids_to_evict, list):
                    ids_to_archive.extend(ids_to_evict)

                # Collect IDs to evict by kind
                kind_to_evict = evict_params.get("kind")
                if kind_to_evict:
                    kind_rows = self._conn.execute(
                        "SELECT id FROM events WHERE kind = ? AND branch_id = ?",
                        (kind_to_evict, branch_id),
                    ).fetchall()
                    ids_to_archive.extend([r["id"] for r in kind_rows])

                # Collect IDs to evict by oldest N
                evict_oldest = evict_params.get("oldest", 0)
                if evict_oldest and isinstance(evict_oldest, int) and evict_oldest > 0:
                    oldest_rows = self._conn.execute(
                        "SELECT id FROM events WHERE branch_id = ? ORDER BY created_at ASC LIMIT ?",
                        (branch_id, evict_oldest),
                    ).fetchall()
                    ids_to_archive.extend([r["id"] for r in oldest_rows])

                # Deduplicate IDs
                ids_to_archive = list(dict.fromkeys(ids_to_archive))

                # Fetch full event data and archive to archival memory
                if ids_to_archive:
                    id_placeholders = ",".join(["?"] * len(ids_to_archive))
                    events_to_archive = self._conn.execute(
                        f"SELECT * FROM events WHERE id IN ({id_placeholders})",
                        ids_to_archive,
                    ).fetchall()

                    for evt in events_to_archive:
                        evt_dict = dict(evt)
                        # Build archival text from event data
                        summary = evt_dict.get("summary", "")
                        kind = evt_dict.get("kind", "unknown")
                        ts = evt_dict.get("ts") or evt_dict.get("created_at", "")
                        archival_text = f"[Evicted Recall Event]\nKind: {kind}\nTime: {ts}\n\n{summary}"

                        self.mem_archival_write(
                            archival_text,
                            tags=["EVICTED_RECALL", f"RECALL_{kind.upper()}" if kind else "RECALL_UNKNOWN"],
                            meta={
                                "source": "recall_evict",
                                "original_event_id": evt_dict.get("id"),
                                "original_kind": kind,
                                "original_ts": ts,
                                "node_id": node_id,
                                "branch_id": branch_id,
                                "phase": phase,
                            },
                        )
                        archived_count += 1

                    # Delete from recall after archiving
                    self._conn.execute(
                        f"DELETE FROM events WHERE id IN ({id_placeholders})",
                        ids_to_archive,
                    )
                    evicted_count = len(ids_to_archive)
                    self._conn.commit()

            results["recall_evict"] = {"evicted_count": evicted_count, "archived_count": archived_count}

        # Recall memory SUMMARIZE operations
        if "recall_summarize" in updates:
            summarize_params = updates["recall_summarize"]
            if isinstance(summarize_params, dict) or summarize_params is True:
                try:
                    consolidated = self.consolidate_recall_events(branch_id)
                    results["recall_summarize"] = {"status": "success", "events_consolidated": consolidated}
                except Exception as exc:
                    logger.warning("LLM-requested recall summarize failed: %s", exc)
                    results["recall_summarize"] = {"status": "failed", "error": str(exc)}

        # Memory CONSOLIDATION operations
        if updates.get("consolidate"):
            try:
                # Use "high" pressure level to trigger consolidation
                consolidate_result = self.auto_consolidate_memory(branch_id, pressure_level="high")
                results["consolidate"] = {"status": "success", "details": consolidate_result}
            except Exception as exc:
                logger.warning("LLM-requested consolidation failed: %s", exc)
                results["consolidate"] = {"status": "failed", "error": str(exc)}

        self._log_memory_event(
            "apply_llm_memory_updates",
            "llm_update",
            branch_id=branch_id,
            node_id=node_id,
            phase=use_phase,
            details={
                "core_keys": list(updates.get("core", {}).keys()) if isinstance(updates.get("core"), dict) else [],
                "core_delete_keys": updates.get("core_delete", []) if isinstance(updates.get("core_delete"), list) else [],
                "core_get_keys": updates.get("core_get", []) if isinstance(updates.get("core_get"), list) else [],
                "archival_count": len(updates.get("archival", [])) if isinstance(updates.get("archival"), list) else 0,
                "archival_update_count": len(updates.get("archival_update", [])) if isinstance(updates.get("archival_update"), list) else 0,
                "has_archival_search": "archival_search" in updates,
                "has_recall": "recall" in updates,
                "has_recall_search": "recall_search" in updates,
                "has_recall_evict": "recall_evict" in updates,
                "has_recall_summarize": "recall_summarize" in updates,
                "has_consolidate": updates.get("consolidate", False),
            },
        )

        return results

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
            for rid in _extract_resource_ids(_row_get(row, "tags")):
                if rid:
                    resource_ids.add(rid)
            chunks.append(self._compress(_row_get(row, "text", ""), 1200, "resource item"))
        if resource_ids:
            note = f"prompt:{query}"
            for rid in sorted(resource_ids):
                track_resource_usage(rid, {"ltm": self, "branch_id": branch_id, "note": note})
        return "\n\n---\n\n".join(chunk for chunk in chunks if chunk).strip()

    def _render_core_memory(
        self, branch_ids: list[str], branch_id: str, no_limit: bool
    ) -> tuple[str, str, list[dict]]:
        """Render core memory section and extract resource index.

        Returns:
            Tuple of (core_text, resource_index, core_items) where core_items
            is a list of dicts with key/value for logging purposes.
        """
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

        # Build core items for logging (truncated values)
        core_items: list[dict] = []
        # Add all core items (LLM manages what to store)
        # Note: idea_md_summary and phase0_summary are no longer auto-injected
        for key, (value, _) in sorted(core_latest.items(), key=lambda kv: kv[0]):
            core_lines.append(f"- {key}: {value}")
            # Truncate value for logging (max 300 chars)
            truncated_value = value[:300] + "..." if len(value) > 300 else value
            core_items.append({"key": key, "value": truncated_value})

        core_text = "\n".join(core_lines).strip()
        return core_text, resource_index or "", core_items

    def _render_recall_memory(self, branch_ids: list[str]) -> tuple[str, list]:
        """Render recall memory section with inherited memory consolidation support.

        This method:
        1. Fetches recall events, excluding those consolidated via inherited_exclusions
        2. Includes inherited summaries from consolidated ancestor events
        3. Returns combined recall text and row data
        """
        # Current branch is the first in the chain
        current_branch_id = branch_ids[0] if branch_ids else None

        # Fetch events with inherited exclusion support
        recall_rows = self._fetch_events(
            branch_ids, self.recall_max_events, current_branch_id=current_branch_id
        )

        recall_lines = []

        # First, add inherited summaries (consolidated ancestor events)
        if current_branch_id:
            inherited_summaries = self._get_inherited_summaries(current_branch_id)
            for summary in inherited_summaries:
                recall_lines.append(f"- [inherited_summary] {summary['summary_text']}")

        # Then add regular recall events
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
                tag_list = json.loads(_row_get(row, "tags") or "[]")
                if tag_list:
                    tags = f" (tags: {', '.join(tag_list)})"
            except json.JSONDecodeError:
                tags = ""
            if no_limit:
                snippet = _row_get(row, "text", "")
            else:
                snippet = self._compress(
                    _row_get(row, "text", ""),
                    self.archival_snippet_budget_chars,
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
        core_text, resource_index, core_items = self._render_core_memory(branch_ids, branch_id, no_limit)
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

        # Convert task_hint to human-readable phase name for logging
        phase_display_name = _get_phase_display_name(task_hint)

        # Build detailed items for logging (truncated for size)
        recall_items = []
        for row in recall_rows[:20]:  # Limit to 20 items
            text = str(_row_get(row, "text", ""))[:200]
            recall_items.append({
                "kind": _row_get(row, "kind"),
                "text": text + "..." if len(str(_row_get(row, "text", ""))) > 200 else text,
            })

        archival_items = []
        for row in archival_rows[:10]:  # Limit to 10 items
            text = str(_row_get(row, "text", ""))[:200]
            tags = []
            try:
                tags = json.loads(_row_get(row, "tags") or "[]")
            except (json.JSONDecodeError, TypeError):
                pass
            archival_items.append({
                "text": text + "..." if len(str(_row_get(row, "text", ""))) > 200 else text,
                "tags": tags,
            })

        # Build log details with injection context
        log_details = {
            "budget_chars": budget_chars,
            "task_hint": task_hint,
            "core_count": len(core_items),
            "recall_count": len(recall_rows),
            "archival_count": len(archival_rows),
            "resource_items": 1 if resource_items_text else 0,
            # Detailed items for visualization
            "core_items": core_items,
            "recall_items": recall_items,
            "archival_items": archival_items,
            "archival_query": task_hint,
            "archival_k": self.retrieval_k,
        }

        if not sections:
            self._log_memory_event(
                "render_for_prompt",
                "prompt",
                branch_id=branch_id,
                phase=phase_display_name,
                details=log_details,
            )
            return ""

        # Combine sections with budget management
        rendered = self._combine_memory_sections(sections, budget_chars, no_limit)

        self._log_memory_event(
            "render_for_prompt",
            "prompt",
            branch_id=branch_id,
            phase=phase_display_name,
            details=log_details,
        )
        return rendered

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
            core_snapshot = self._compress(core_snapshot, self.results_budget_chars, "experimental results", branch_id=best_branch)

        return core_snapshot

    def _get_idea_text_and_summary(
        self, run_dir: Path, best_branch: str, no_budget_limit: bool
    ) -> tuple[str, str]:
        """Get idea text and summary."""
        idea_text = ""
        if no_budget_limit:
            idea_paths = [run_dir / "idea.md", run_dir.parent / "idea.md"]
            for idea_path in idea_paths:
                if idea_path.exists():
                    try:
                        idea_text = idea_path.read_text(encoding="utf-8")
                        break
                    except Exception:
                        pass

        if not idea_text:
            idea_archival = self.retrieve_archival(
                best_branch, query="idea", k=1, include_ancestors=True, tags_filter=["IDEA_MD"]
            )
            idea_text = idea_archival[0]["text"] if idea_archival else ""

        # Get idea summary from core memory (LLM manages this now)
        idea_summary = self.get_core(best_branch, "idea_md_summary") or ""
        # If no summary in core memory, use the text directly (no auto-summarization)
        if not idea_summary and idea_text:
            idea_summary = idea_text if no_budget_limit else self._compress(idea_text, 4000, "idea summary")

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
                        archival_text = _row_get(row, "text", "")
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
        run_dir: Path,
        best_branch: str,
        resources_used: list[dict[str, Any]],
        no_budget_limit: bool,
    ) -> dict[str, Any]:
        """
        Build paper sections by letting LLM autonomously retrieve relevant
        information from all 3 memory tiers (Core, Recall, Archive).

        The LLM receives the full memory context and decides what information
        is relevant for each paper section.
        """
        # Collect all 3-tier memory for the best branch
        branch_chain = self._branch_chain(best_branch)

        # Core Memory - all key-value pairs
        core_memory: dict[str, Any] = {}
        for bid in branch_chain:
            core_rows = self._conn.execute(
                "SELECT key, value FROM core_kv WHERE branch_id=?",
                (bid,),
            ).fetchall()
            for row in core_rows:
                key = row["key"]
                if key not in core_memory:
                    core_memory[key] = row["value"] or ""

        # Recall Memory - all events
        recall_memory: list[dict[str, Any]] = []
        event_rows = self._fetch_events(branch_chain, 500 if no_budget_limit else 100)
        for row in event_rows:
            recall_memory.append({
                "ts": row["created_at"],
                "kind": row["kind"],
                "text": row["text"] or "",
            })

        # Archival Memory - all entries
        archival_k = 200 if no_budget_limit else 50
        archival_rows = self.retrieve_archival(
            branch_id=best_branch,
            query="",
            k=archival_k,
            include_ancestors=True,
            log_event=False,
        )
        archival_memory: list[dict[str, Any]] = []
        for row in archival_rows:
            archival_memory.append({
                "id": _row_get(row, "id"),
                "text": _row_get(row, "text", "") or "",
                "tags": _row_get(row, "tags", []),
            })

        idea_text, idea_summary = self._get_idea_text_and_summary(run_dir, best_branch, no_budget_limit)

        # Build memory context for LLM
        memory_context = {
            "core_memory": core_memory,
            "recall_memory": recall_memory,
            "archival_memory": archival_memory,
            "resources_used": resources_used,
            "best_branch": best_branch,
            "idea_text": idea_text,
            "idea_summary": idea_summary,
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

        if self.paper_section_mode == "idea_then_memory":
            return self._generate_sections_from_idea(memory_context)

        return self._generate_sections_from_memory_summary(memory_context)

    def _generate_sections_from_memory_summary(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """Generate sections by letting the LLM decide from a memory summary (current default)."""
        try:
            from ai_scientist.llm import get_response_from_llm

            # Prepare memory summary for LLM
            memory_summary = self._prepare_memory_summary(memory_context)

            # Load prompt template from file
            prompt_template = _try_load_prompt(PAPER_SECTION_GENERATION_NAME, "paper section generation prompt")
            if prompt_template is None:
                # Fallback to hardcoded prompt
                prompt_template = """Analyze the following research memory data and generate appropriate sections for a research paper.

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

            prompt = prompt_template.format(memory_summary=memory_summary)

            system_message = _try_load_prompt(PAPER_SECTION_GENERATION_SYSTEM_MESSAGE_NAME, "paper section generation system message")
            if system_message is None:
                system_message = "You are a research paper writing assistant. Analyze memory data and generate comprehensive paper sections in JSON format."

            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message.strip(),
                temperature=0.3,
            )

            if not response:
                logger.warning("LLM returned empty response for section generation. Using fallback.")
                return self._build_fallback_sections(memory_context)

            # Parse JSON response
            try:
                response = self._extract_json_payload(response)
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

    def _generate_sections_from_idea(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """Generate a fixed number of sections from the idea, then fill each via memory search."""
        idea_text = (memory_context.get("idea_text") or memory_context.get("idea_summary") or "").strip()
        if not idea_text:
            logger.warning("Idea text not found; falling back to memory-summary section generation.")
            return self._generate_sections_from_memory_summary(memory_context)

        section_count = max(3, int(self.paper_section_count))

        try:
            from ai_scientist.llm import get_response_from_llm

            outline_prompt = _try_load_prompt(PAPER_SECTION_OUTLINE_NAME, "paper section outline prompt")
            if outline_prompt is None:
                outline_prompt = """You are given a research idea. Propose exactly {section_count} paper sections that would be necessary to write the paper.

Idea:
{idea_text}

Return a JSON array of objects. Each object must have:
- key: snake_case identifier
- title: human-readable section title
- focus: 1-2 sentence description of what the section should cover
- keywords: array of search keywords/phrases to retrieve memory

Requirements:
- Exactly {section_count} sections
- Include at least 'experimental_setup' and 'results'
- Avoid duplicates
- Return ONLY valid JSON (no markdown fences)
"""

            outline_prompt = outline_prompt.format(
                idea_text=idea_text,
                section_count=section_count,
            )

            system_message = _try_load_prompt(PAPER_SECTION_OUTLINE_SYSTEM_MESSAGE_NAME, "paper section outline system message")
            if system_message is None:
                system_message = "You are a research paper planning assistant. Produce section outlines in JSON."

            outline_response, _ = get_response_from_llm(
                prompt=outline_prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message.strip(),
                temperature=0.2,
            )

            if not outline_response:
                logger.warning("LLM returned empty outline response. Falling back to memory-summary mode.")
                return self._generate_sections_from_memory_summary(memory_context)

            outline_payload = self._extract_json_payload(outline_response)
            try:
                parsed = json.loads(outline_payload)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse outline JSON: {exc}. Falling back to memory-summary mode.")
                return self._generate_sections_from_memory_summary(memory_context)

            sections_list = parsed.get("sections", []) if isinstance(parsed, dict) else parsed
            if not isinstance(sections_list, list):
                logger.warning("Outline response is not a list. Falling back to memory-summary mode.")
                return self._generate_sections_from_memory_summary(memory_context)

            normalized_sections = self._normalize_section_outline(sections_list, section_count, memory_context)
            if not normalized_sections:
                logger.warning("Outline normalization failed. Falling back to memory-summary mode.")
                return self._generate_sections_from_memory_summary(memory_context)

            filled_sections: dict[str, Any] = {}
            for section in normalized_sections:
                content = self._fill_section_from_memory(section, memory_context)
                filled_sections[section["key"]] = content

            filled_sections["resources_used"] = memory_context.get("resources_used", [])
            logger.info(
                "Generated %d sections from idea outline with memory fill.",
                len(filled_sections),
            )
            return filled_sections

        except Exception as exc:
            logger.warning(f"Idea-based section generation failed: {exc}. Using fallback.")
            return self._build_fallback_sections(memory_context)

    def _fill_section_from_memory(self, section: dict[str, Any], memory_context: dict[str, Any]) -> str:
        """Fill a single section using memory search and LLM synthesis."""
        memory_slice = self._filter_memory_for_section(section, memory_context)
        try:
            from ai_scientist.llm import get_response_from_llm

            prompt_template = _try_load_prompt(PAPER_SECTION_FILL_NAME, "paper section fill prompt")
            if prompt_template is None:
                prompt_template = """Write the content for the following paper section using ONLY the provided memory slice.

Section Spec (JSON):
{section_spec}

Memory Slice (JSON):
{memory_slice}

Guidelines:
- Use only the memory slice; do not invent details.
- If relevant info is missing, say \"Not found in memory.\" succinctly.
- Return plain markdown text (no JSON).
"""

            prompt = prompt_template.format(
                section_spec=json.dumps(section, ensure_ascii=True, indent=2),
                memory_slice=json.dumps(memory_slice, ensure_ascii=True, indent=2),
            )

            system_message = _try_load_prompt(PAPER_SECTION_FILL_SYSTEM_MESSAGE_NAME, "paper section fill system message")
            if system_message is None:
                system_message = "You are a research paper writing assistant."

            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message.strip(),
                temperature=0.3,
            )
            content = response.strip() if response else "Not found in memory."
            if str(section.get("key", "")) == "experimental_setup":
                content = self._append_experimental_setup_facts(content, memory_slice)
            return content
        except Exception as exc:
            logger.warning("Failed to fill section %s: %s", section.get("key"), exc)
            return "Not found in memory."

    def _append_experimental_setup_facts(self, content: str, memory_slice: dict[str, Any]) -> str:
        """Append deterministic environment facts if present in memory."""
        core = memory_slice.get("core_memory", {}) or {}
        facts: list[str] = []

        compiler = core.get("selected_compiler")
        if not compiler:
            compiler = self._get_core_any_branch("selected_compiler")
        if compiler:
            facts.append(f"- Compiler (core): {compiler}")

        omp_smoke = core.get("omp_smoke_build")
        if not omp_smoke:
            omp_smoke = self._get_core_any_branch("omp_smoke_build")
        parsed = None
        if omp_smoke:
            if isinstance(omp_smoke, dict):
                parsed = omp_smoke
            elif isinstance(omp_smoke, str):
                try:
                    parsed = ast.literal_eval(omp_smoke)
                except Exception:
                    parsed = None
        if isinstance(parsed, dict):
            flags = parsed.get("flags")
            if flags:
                facts.append(f"- Compiler flags (omp_smoke_build): `{flags}`")
            compiler_used = parsed.get("compiler")
            if compiler_used and not compiler:
                facts.append(f"- Compiler (omp_smoke_build): {compiler_used}")

        env_verified = core.get("ablation_studies_1_first_attempt_environment_verified")
        if not env_verified:
            env_verified = self._get_core_any_branch("ablation_studies_1_first_attempt_environment_verified")
        if env_verified:
            facts.append(f"- Environment verified: {env_verified}")

        if not facts:
            return content

        return f"{content}\n\n### Extracted environment facts\n" + "\n".join(facts)

    def _get_core_any_branch(self, key: str) -> Any:
        """Fetch a core_kv value across all branches (latest by updated_at)."""
        try:
            row = self._conn.execute(
                "SELECT value FROM core_kv WHERE key=? ORDER BY updated_at DESC LIMIT 1",
                (key,),
            ).fetchone()
            if row:
                return row["value"]
        except Exception:
            return None
        return None

    def _sample_memory_for_keyword_generation(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """
        Sample memory data for LLM keyword generation.
        Returns a representative subset to avoid overwhelming the LLM.
        """
        core_memory = memory_context.get("core_memory", {}) or {}
        recall_memory = memory_context.get("recall_memory", []) or []
        archival_memory = memory_context.get("archival_memory", []) or []

        # Sample core memory keys (up to 10 entries)
        core_sample = {}
        for i, (key, value) in enumerate(core_memory.items()):
            if i >= 10:
                break
            value_str = str(value)
            # Truncate long values
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            core_sample[key] = value_str

        # Sample recall events (up to 15 entries, distributed across timeline)
        recall_sample = []
        if len(recall_memory) > 15:
            # Take samples from beginning, middle, and end
            indices = [0, 1, 2, len(recall_memory)//3, len(recall_memory)//2,
                      2*len(recall_memory)//3, -3, -2, -1]
            for idx in indices[:15]:
                if 0 <= idx < len(recall_memory) or idx < 0:
                    event = recall_memory[idx]
                    text = str(event.get("text", ""))
                    if len(text) > 150:
                        text = text[:150] + "..."
                    recall_sample.append({
                        "kind": event.get("kind", ""),
                        "text": text,
                    })
        else:
            for event in recall_memory[:15]:
                text = str(event.get("text", ""))
                if len(text) > 150:
                    text = text[:150] + "..."
                recall_sample.append({
                    "kind": event.get("kind", ""),
                    "text": text,
                })

        # Sample archival memory (up to 10 entries)
        archival_sample = []
        for i, entry in enumerate(archival_memory[:10]):
            text = str(entry.get("text", ""))
            if len(text) > 150:
                text = text[:150] + "..."
            archival_sample.append({
                "tags": entry.get("tags", []),
                "text": text,
            })

        return {
            "core_memory_sample": core_sample,
            "recall_memory_sample": recall_sample,
            "archival_memory_sample": archival_sample,
        }

    def _generate_section_keywords_with_llm(
        self,
        section: dict[str, Any],
        memory_context: dict[str, Any]
    ) -> list[str]:
        """
        Use LLM to dynamically generate optimal keywords for a section based on memory content.

        Args:
            section: Section definition (key, title, focus)
            memory_context: Full memory data (core, recall, archival)

        Returns:
            List of generated keywords (5-15 items)
        """
        # Check if LLM compression is available
        if not self.use_llm_compression or self._compression_client is None:
            logger.warning("LLM not available for keyword generation. Using title/focus-based keywords.")
            return self._section_keywords(section)

        try:
            from ai_scientist.llm import get_response_from_llm

            # Sample memory to avoid overwhelming the LLM
            memory_sample = self._sample_memory_for_keyword_generation(memory_context)

            # Load prompt template from file
            prompt_template = _try_load_prompt(KEYWORD_EXTRACTION_NAME, "keyword extraction prompt")
            if prompt_template is None:
                # Fallback to hardcoded prompt
                prompt_template = """Analyze the following memory data and generate optimal search keywords to extract information for the paper section "{section_title}".

Section description: {section_focus}

Memory Sample:
{memory_sample}

Requirements:
- Generate 5-15 keywords that are likely to appear in the memory
- Include both general terms and specific technical terms
- Include synonyms and abbreviations (e.g., "OpenMP" and "omp")
- Focus on terms that actually exist in the memory sample
- Prioritize keywords that match the section's focus

Return a JSON array: ["keyword1", "keyword2", ...]"""

            prompt = prompt_template.format(
                section_title=section.get('title', ''),
                section_focus=section.get('focus', ''),
                memory_sample=json.dumps(memory_sample, indent=2, ensure_ascii=False),
            )

            system_message = _try_load_prompt(KEYWORD_EXTRACTION_SYSTEM_MESSAGE_NAME, "keyword extraction system message")
            if system_message is None:
                system_message = "You are a keyword extraction assistant. Generate search keywords in JSON array format."

            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self._compression_client,
                model=self._compression_model_name,
                system_message=system_message.strip(),
                temperature=0.3,
            )

            if not response:
                logger.warning(f"LLM returned empty response for keyword generation (section: {section.get('key')}). Using fallback.")
                return self._section_keywords(section)

            # Parse JSON response
            try:
                response_clean = self._extract_json_payload(response)
                keywords = json.loads(response_clean)

                if not isinstance(keywords, list):
                    logger.warning(f"LLM returned non-list response for keywords (section: {section.get('key')}). Using fallback.")
                    return self._section_keywords(section)

                # Filter and validate keywords
                valid_keywords = []
                for kw in keywords:
                    if isinstance(kw, str) and kw.strip():
                        valid_keywords.append(kw.strip())

                if not valid_keywords:
                    logger.warning(f"No valid keywords generated (section: {section.get('key')}). Using fallback.")
                    return self._section_keywords(section)

                logger.info(f"Generated {len(valid_keywords)} keywords for section '{section.get('key')}' using LLM.")
                return valid_keywords[:15]  # Limit to 15 keywords

            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse LLM keyword response as JSON (section: {section.get('key')}): {exc}. Using fallback.")
                return self._section_keywords(section)

        except Exception as exc:
            logger.warning(f"Keyword generation with LLM failed (section: {section.get('key')}): {exc}. Using fallback.")
            return self._section_keywords(section)

    def _filter_memory_for_section(self, section: dict[str, Any], memory_context: dict[str, Any]) -> dict[str, Any]:
        """Search memory for section-relevant content using simple keyword matching."""
        keywords = self._section_keywords(section)
        keywords_lower = [k.lower() for k in keywords if k]

        def matches(text: str) -> bool:
            text_lower = text.lower()
            return any(k in text_lower for k in keywords_lower)

        core_memory = memory_context.get("core_memory", {}) or {}
        recall_memory = memory_context.get("recall_memory", []) or []
        archival_memory = memory_context.get("archival_memory", []) or []

        filtered_core: dict[str, Any] = {}
        for key, value in core_memory.items():
            value_str = str(value)
            if not keywords_lower or matches(key) or matches(value_str):
                filtered_core[key] = _truncate(value_str, 2000)
            if len(filtered_core) >= 20:
                break

        filtered_recall = []
        for event in recall_memory:
            text = str(event.get("text", ""))
            kind = str(event.get("kind", ""))
            if not keywords_lower or matches(text) or matches(kind):
                filtered_recall.append({
                    "ts": event.get("ts", ""),
                    "kind": kind,
                    "text": _truncate(text, 1500),
                })
            if len(filtered_recall) >= 30:
                break

        filtered_archival = []
        for entry in archival_memory:
            text = str(entry.get("text", ""))
            tags = entry.get("tags", [])
            tags_text = " ".join(tags) if isinstance(tags, list) else str(tags)
            if not keywords_lower or matches(text) or matches(tags_text):
                filtered_archival.append({
                    "id": entry.get("id", ""),
                    "tags": tags,
                    "text": _truncate(text, 1500),
                })
            if len(filtered_archival) >= 20:
                break

        section_key = str(section.get("key", "") or "")
        if section_key == "experimental_setup":
            for core_key in (
                "selected_compiler",
                "omp_smoke_build",
                "phase1_complete_reason",
                "ablation_studies_1_first_attempt_environment_verified",
            ):
                if core_key in core_memory and core_key not in filtered_core:
                    filtered_core[core_key] = _truncate(str(core_memory.get(core_key, "")), 2000)

        return {
            "idea_summary": memory_context.get("idea_summary", ""),
            "section_key": section.get("key"),
            "core_memory": filtered_core,
            "recall_memory": filtered_recall,
            "archival_memory": filtered_archival,
        }

    def _section_keywords(self, section: dict[str, Any]) -> list[str]:
        keywords = []
        for item in section.get("keywords", []) or []:
            if isinstance(item, str) and item.strip():
                keywords.append(item.strip())
        if not keywords:
            title = str(section.get("title", ""))
            focus = str(section.get("focus", ""))
            keywords.extend(self._tokenize_keywords(title))
            keywords.extend(self._tokenize_keywords(focus))
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for kw in keywords:
            kw_norm = kw.strip()
            if not kw_norm or kw_norm.lower() in seen:
                continue
            seen.add(kw_norm.lower())
            deduped.append(kw_norm)
        return deduped[:12]

    def _tokenize_keywords(self, text: str) -> list[str]:
        tokens = re.split(r"[^A-Za-z0-9]+", text)
        return [t for t in tokens if len(t) >= 3]

    def _normalize_section_outline(
        self,
        sections: list[dict[str, Any]],
        section_count: int,
        memory_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for section in sections:
            if not isinstance(section, dict):
                continue
            key = section.get("key") or section.get("name") or ""
            title = section.get("title") or section.get("name") or key
            focus = section.get("focus") or ""
            key = self._to_snake_case(str(key or title))
            if not key or key in seen_keys:
                continue
            normalized.append({
                "key": key,
                "title": str(title),
                "focus": str(focus),
                "keywords": section.get("keywords", []) if isinstance(section.get("keywords", []), list) else [],
            })
            seen_keys.add(key)

        # Required sections with minimal static keywords
        # Keywords will be dynamically generated by LLM based on memory content
        required = {
            "experimental_setup": {
                "key": "experimental_setup",
                "title": "Experimental Setup",
                "focus": "Hardware/software environment, datasets, and evaluation protocol.",
                "keywords": [],  # Will be generated by LLM
            },
            "results": {
                "key": "results",
                "title": "Results",
                "focus": "Key results, metrics, and comparative findings.",
                "keywords": [],  # Will be generated by LLM
            },
        }
        indexed = {sec["key"]: sec for sec in normalized}
        for key, spec in required.items():
            if key in indexed:
                existing = indexed[key]
                if not existing.get("title"):
                    existing["title"] = spec.get("title", existing.get("title", ""))
                if not existing.get("focus"):
                    existing["focus"] = spec.get("focus", existing.get("focus", ""))
                merged_keywords = []
                for kw in existing.get("keywords", []) or []:
                    if isinstance(kw, str) and kw.strip():
                        merged_keywords.append(kw.strip())
                for kw in spec.get("keywords", []) or []:
                    if isinstance(kw, str) and kw.strip() and kw.strip() not in merged_keywords:
                        merged_keywords.append(kw.strip())
                existing["keywords"] = merged_keywords
            else:
                normalized.append(spec)
                seen_keys.add(key)
                indexed[key] = spec

        fallback_order = [
            {
                "key": "title_candidates",
                "title": "Title Candidates",
                "focus": "Possible paper titles.",
                "keywords": ["title", "name"],
            },
            {
                "key": "abstract_material",
                "title": "Abstract Material",
                "focus": "Summary of the idea, approach, and results.",
                "keywords": ["abstract", "summary", "goal"],
            },
            {
                "key": "problem_statement",
                "title": "Problem Statement",
                "focus": "What problem is being solved and why it matters.",
                "keywords": ["problem", "motivation", "challenge"],
            },
            {
                "key": "hypothesis",
                "title": "Hypothesis",
                "focus": "Key hypothesis or expected outcome.",
                "keywords": ["hypothesis", "expectation"],
            },
            {
                "key": "method",
                "title": "Method",
                "focus": "Method description and implementation details.",
                "keywords": ["method", "implementation", "algorithm"],
            },
            {
                "key": "ablations_negative",
                "title": "Ablations / Negative Results",
                "focus": "Ablations, failure cases, or negative results.",
                "keywords": ["ablation", "negative", "failure"],
            },
            {
                "key": "threats_to_validity",
                "title": "Threats To Validity",
                "focus": "Limitations and threats to validity.",
                "keywords": ["limitations", "threats", "validity"],
            },
            {
                "key": "reproducibility_checklist",
                "title": "Reproducibility Checklist",
                "focus": "Reproducibility details and checklist items.",
                "keywords": ["reproducibility", "checklist"],
            },
            {
                "key": "narrative_bullets",
                "title": "Narrative Bullets",
                "focus": "Narrative framing or positioning.",
                "keywords": ["narrative", "positioning", "related work"],
            },
            {
                "key": "failure_modes_timeline",
                "title": "Failure Modes Timeline",
                "focus": "Timeline of failures or debugging notes.",
                "keywords": ["failure", "timeline", "debug"],
            },
        ]

        if len(normalized) < section_count:
            for spec in fallback_order:
                if len(normalized) >= section_count:
                    break
                if spec["key"] not in seen_keys:
                    normalized.append(spec)
                    seen_keys.add(spec["key"])

        if len(normalized) > section_count:
            required_keys = ["experimental_setup", "results"]
            trimmed: list[dict[str, Any]] = []
            for key in required_keys:
                for sec in normalized:
                    if sec["key"] == key and sec not in trimmed:
                        trimmed.append(sec)
                        break
            for sec in normalized:
                if len(trimmed) >= section_count:
                    break
                if sec["key"] in required_keys:
                    continue
                trimmed.append(sec)
            normalized = trimmed[:section_count]

        # Generate keywords dynamically using LLM if memory_context is provided
        logger.info(f"[Keyword Gen] _normalize_section_outline called with {len(normalized)} sections, memory_context={'provided' if memory_context is not None else 'None'}")
        if memory_context is not None:
            logger.info(f"[Keyword Gen] Processing {len(normalized)} sections for keyword generation/validation...")
            for section in normalized:
                # Only generate keywords if section has no keywords or empty keywords
                existing_keywords = section.get("keywords", [])
                section_key = section.get("key", "unknown")
                logger.info(f"[Keyword Gen] Checking section '{section_key}': has {len(existing_keywords) if existing_keywords else 0} existing keywords")

                if not existing_keywords or (isinstance(existing_keywords, list) and len(existing_keywords) == 0):
                    logger.info(f"[Keyword Gen] Generating keywords for section '{section_key}' using LLM (empty keywords detected)...")
                    generated_keywords = self._generate_section_keywords_with_llm(section, memory_context)
                    section["keywords"] = generated_keywords
                    logger.info(f"[Keyword Gen] Successfully generated {len(generated_keywords)} keywords for section '{section_key}': {generated_keywords[:5] if len(generated_keywords) > 0 else '[]'}...")
                else:
                    preview = existing_keywords[:3] if len(existing_keywords) > 3 else existing_keywords
                    logger.info(f"[Keyword Gen] Section '{section_key}' already has {len(existing_keywords)} keywords (from LLM outline): {preview}...")
        else:
            logger.info("[Keyword Gen] memory_context is None, skipping keyword generation")

        return normalized

    def _to_snake_case(self, text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return text.lower()

    def _extract_json_payload(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        return response.strip()

    def _prepare_memory_summary(self, memory_context: dict[str, Any]) -> str:
        """Prepare a formatted summary of 3-tier memory for LLM analysis.

        The LLM will autonomously search through this memory to extract
        relevant information for paper sections.
        """
        parts = []

        idea_text = memory_context.get("idea_text", "") or ""
        idea_summary = memory_context.get("idea_summary", "") or ""
        if idea_text or idea_summary:
            idea_lines = ["## Idea"]
            if idea_summary:
                idea_lines.append(f"Summary:\n{idea_summary[:4000]}")
            if idea_text and idea_text != idea_summary:
                idea_lines.append(f"Text:\n{idea_text[:4000]}")
            parts.append("\n\n".join(idea_lines))

        # Core Memory (always-injected key-value context)
        core_memory = memory_context.get("core_memory", {})
        if core_memory:
            core_lines = ["## Core Memory (Key-Value Context)"]
            for key, value in core_memory.items():
                # Truncate very long values for the summary
                value_str = str(value)[:2000] if len(str(value)) > 2000 else str(value)
                core_lines.append(f"### {key}\n{value_str}")
            parts.append("\n\n".join(core_lines))

        # Recall Memory (event timeline)
        recall_memory = memory_context.get("recall_memory", [])
        if recall_memory:
            recall_lines = ["## Recall Memory (Event Timeline)"]
            for event in recall_memory[:50]:  # Limit for LLM context
                ts = event.get("ts", "")
                kind = event.get("kind", "")
                text = event.get("text", "")[:1000]  # Truncate long events
                recall_lines.append(f"- [{ts}] {kind}: {text}")
            if len(recall_memory) > 50:
                recall_lines.append(f"... and {len(recall_memory) - 50} more events")
            parts.append("\n".join(recall_lines))

        # Archival Memory (long-term searchable memory)
        archival_memory = memory_context.get("archival_memory", [])
        if archival_memory:
            archival_lines = ["## Archival Memory (Long-term Storage)"]
            for entry in archival_memory[:30]:  # Limit for LLM context
                entry_id = entry.get("id", "")
                tags = entry.get("tags", [])
                text = entry.get("text", "")[:1500]  # Truncate long entries
                tags_str = ", ".join(tags) if tags else "no tags"
                archival_lines.append(f"### Entry {entry_id} [{tags_str}]\n{text}")
            if len(archival_memory) > 30:
                archival_lines.append(f"... and {len(archival_memory) - 30} more archival entries")
            parts.append("\n\n".join(archival_lines))

        # Resources used
        resources_used = memory_context.get("resources_used", [])
        if resources_used:
            resource_lines = ["## Resources Used"]
            for res in resources_used:
                name = res.get("name", "unknown")
                res_class = res.get("class", "unknown")
                resource_lines.append(f"- {name} ({res_class})")
            parts.append("\n".join(resource_lines))

        return "\n\n---\n\n".join(parts)

    def _build_fallback_sections(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """Build a simple fallback section structure when LLM is not available.

        Extracts information from 3-tier memory (Core, Recall, Archive).
        """
        core_memory = memory_context.get("core_memory", {})
        recall_memory = memory_context.get("recall_memory", [])
        archival_memory = memory_context.get("archival_memory", [])

        # Helper to find archival entries by tag
        def find_archival_by_tag(tag: str) -> str:
            for entry in archival_memory:
                tags = entry.get("tags", [])
                if tag in tags:
                    return entry.get("text", "")
            return ""

        # Helper to collect recall events by kind
        def collect_recall_by_kind(kind: str) -> list[str]:
            return [e.get("text", "") for e in recall_memory if e.get("kind") == kind]

        # Extract idea from archival (IDEA_MD tag)
        idea_text = find_archival_by_tag("IDEA_MD") or find_archival_by_tag("ROOT_IDEA")
        parsed_idea = _parse_markdown_sections(idea_text) if idea_text else {}

        # Extract from core memory
        phase0_summary = core_memory.get("phase0_summary", "")
        experimental_results = core_memory.get("experimental_results", "") or core_memory.get("results", "")

        # Extract failure notes from recall events
        failure_notes = collect_recall_by_kind("error_encountered")
        failure_text = "\n".join(failure_notes) if failure_notes else "No failures recorded."

        # Build sections from extracted data
        title_text = parsed_idea.get("Title") or parsed_idea.get("Name") or "Title not found in memory"
        abstract_text = parsed_idea.get("Abstract") or parsed_idea.get("Task goal") or "Abstract not found in memory"
        hypothesis_text = parsed_idea.get("Short Hypothesis") or parsed_idea.get("Hypothesis") or "Hypothesis not found in memory"
        problem_text = parsed_idea.get("Problem Statement") or parsed_idea.get("Motivation") or abstract_text
        method_text = parsed_idea.get("Experiments") or parsed_idea.get("Code") or parsed_idea.get("Method") or "Method not found in memory"
        narrative_text = parsed_idea.get("Related Work") or parsed_idea.get("Narrative") or "Related work positioning, key trade-offs, implications."

        sections = {
            "title_candidates": title_text,
            "abstract_material": abstract_text,
            "problem_statement": problem_text,
            "hypothesis": hypothesis_text,
            "method": method_text,
            "experimental_setup": phase0_summary,
            "phase0_internal_info_summary": phase0_summary,
            "results": experimental_results,
            "ablations_negative": "Not available (LLM analysis required)",
            "failure_modes_timeline": failure_text,
            "threats_to_validity": "Not available (LLM analysis required)",
            "reproducibility_checklist": "Not available (LLM analysis required)",
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

        # Collect resource data for the sections
        resource_snapshot, resource_index, item_index = self._load_resource_snapshot_and_index(
            memory_dir, root_branch_id, best_branch
        )
        resources_used = self._collect_resource_usage(root_branch_id, best_branch, item_index, no_budget_limit)

        # Build paper sections using LLM self-retrieval from 3-tier memory
        sections = self._build_paper_sections(
            run_dir=run_dir,
            best_branch=best_branch,
            resources_used=resources_used,
            no_budget_limit=no_budget_limit,
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

        # Extract data from 3-tier memory for writeup_memory
        branch_chain = self._branch_chain(best_branch)

        # Get idea text from archival memory
        idea_text = ""
        idea_summary = ""
        for bid in branch_chain:
            idea_rows = self._conn.execute(
                "SELECT text FROM archival WHERE branch_id=? AND tags LIKE '%IDEA_MD%' LIMIT 1",
                (bid,),
            ).fetchall()
            if idea_rows:
                idea_text = idea_rows[0]["text"] or ""
                idea_summary = idea_text[:500] if idea_text else ""
                break

        # Get phase0 summary from core memory
        phase0_summary = ""
        for bid in branch_chain:
            core_row = self._conn.execute(
                "SELECT value FROM core_kv WHERE branch_id=? AND key='phase0_summary' LIMIT 1",
                (bid,),
            ).fetchone()
            if core_row:
                phase0_summary = core_row["value"] or ""
                break

        # Get core snapshot from core memory (results)
        core_snapshot = ""
        for bid in branch_chain:
            for key_name in ("results", "experimental_results", "core_snapshot"):
                core_row = self._conn.execute(
                    "SELECT value FROM core_kv WHERE branch_id=? AND key=? LIMIT 1",
                    (bid, key_name),
                ).fetchone()
                if core_row and core_row["value"]:
                    core_snapshot = core_row["value"]
                    break
            if core_snapshot:
                break

        # Get failure notes from recall memory
        failure_notes: list[str] = []
        event_rows_for_failures = self._fetch_events(branch_chain, 100)
        for row in event_rows_for_failures:
            if str(row["kind"]).lower() == "error_encountered":
                failure_notes.append(row["text"] or "")

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
            text = _row_get(row, "text", "") or ""
            if not no_budget_limit and len(text) > self.writeup_archival_text_max_chars:
                text = self._compress(text, self.writeup_archival_text_max_chars, "archival entry", branch_id=best_branch)
            archival_memory_list.append({
                "id": _row_get(row, "id"),
                "text": text,
                "tags": _row_get(row, "tags"),
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

        # Extract best node data from artifacts_index
        best_node_data = artifacts_index.get("best_node_data") if artifacts_index else None
        top_nodes_data = artifacts_index.get("top_nodes_data", []) if artifacts_index else []

        # Generate comprehensive markdown for paper writeup
        md_sections = ["# Final Memory For Paper", ""]

        # ===== Section 1: Executive Summary =====
        md_sections.append("## Executive Summary")
        md_sections.append("")
        if best_node_data:
            metric_info = best_node_data.get("metric", {})
            md_sections.append(f"**Best Result Metric**: {metric_info.get('value', 'N/A')} ({metric_info.get('name', 'metric')})")
            md_sections.append("")
            if best_node_data.get("analysis"):
                md_sections.append("**Analysis Summary**:")
                md_sections.append(_to_md_str(best_node_data["analysis"]))
                md_sections.append("")
        else:
            md_sections.append("No best node data available.")
            md_sections.append("")

        # ===== Section 2: Best Node Details =====
        md_sections.append("## Best Node Details")
        md_sections.append("")
        if best_node_data:
            md_sections.append(f"**Node ID**: `{best_node_data.get('id', 'N/A')}`")
            md_sections.append(f"**Branch ID**: `{best_node_data.get('branch_id', 'N/A')}`")
            if best_node_data.get("workspace_path"):
                md_sections.append(f"**Workspace**: `{best_node_data['workspace_path']}`")
            if best_node_data.get("exp_results_dir"):
                md_sections.append(f"**Results Directory**: `{best_node_data['exp_results_dir']}`")
            md_sections.append("")

            # Overall Plan
            if best_node_data.get("overall_plan"):
                md_sections.append("### Overall Plan")
                md_sections.append("```")
                md_sections.append(_to_md_str(best_node_data["overall_plan"]))
                md_sections.append("```")
                md_sections.append("")

            # Implementation Plan
            if best_node_data.get("plan"):
                md_sections.append("### Implementation Plan")
                md_sections.append("```")
                md_sections.append(_to_md_str(best_node_data["plan"]))
                md_sections.append("```")
                md_sections.append("")

            # Code Implementation
            if best_node_data.get("code"):
                md_sections.append("### Code Implementation")
                md_sections.append("```python")
                md_sections.append(_to_md_str(best_node_data["code"]))
                md_sections.append("```")
                md_sections.append("")

            # Phase Artifacts
            phase_artifacts = best_node_data.get("phase_artifacts", {})
            if phase_artifacts:
                md_sections.append("### Phase Artifacts")
                md_sections.append("")
                for phase_name, artifact in phase_artifacts.items():
                    md_sections.append(f"#### {phase_name}")
                    if isinstance(artifact, dict):
                        for key, value in artifact.items():
                            if isinstance(value, str) and len(value) > 500:
                                md_sections.append(f"**{key}**:")
                                md_sections.append("```")
                                md_sections.append(value[:2000] + "..." if len(value) > 2000 else value)
                                md_sections.append("```")
                            else:
                                md_sections.append(f"- **{key}**: {value}")
                    else:
                        md_sections.append(str(artifact))
                    md_sections.append("")
        else:
            md_sections.append("No best node data available.")
            md_sections.append("")

        # ===== Section 3: VLM Analysis & Visual Feedback =====
        md_sections.append("## VLM Analysis & Visual Feedback")
        md_sections.append("")
        if best_node_data:
            # Plot analyses
            plot_analyses = best_node_data.get("plot_analyses", [])
            if plot_analyses:
                md_sections.append("### Plot Analyses")
                for i, analysis in enumerate(plot_analyses, 1):
                    md_sections.append(f"#### Analysis {i}")
                    md_sections.append(_to_md_str(analysis))
                    md_sections.append("")

            # VLM feedback summary
            vlm_feedback = best_node_data.get("vlm_feedback_summary", [])
            if vlm_feedback:
                md_sections.append("### VLM Feedback Summary")
                for i, feedback in enumerate(vlm_feedback, 1):
                    md_sections.append(f"{i}. {feedback}")
                md_sections.append("")

            # Datasets tested
            datasets_tested = best_node_data.get("datasets_successfully_tested", [])
            if datasets_tested:
                md_sections.append("### Datasets Successfully Tested")
                for dataset in datasets_tested:
                    md_sections.append(f"- {dataset}")
                md_sections.append("")

            # Plot paths
            plot_paths = best_node_data.get("plot_paths", [])
            if plot_paths:
                md_sections.append("### Generated Plots")
                for path in plot_paths:
                    md_sections.append(f"- `{path}`")
                md_sections.append("")

            if not (plot_analyses or vlm_feedback or datasets_tested or plot_paths):
                md_sections.append("No VLM analysis data available.")
                md_sections.append("")
        else:
            md_sections.append("No VLM analysis data available.")
            md_sections.append("")

        # ===== Section 4: Top Nodes Comparison =====
        md_sections.append("## Top Nodes Comparison")
        md_sections.append("")
        if top_nodes_data:
            md_sections.append("| Rank | Node ID | Metric | Key Findings |")
            md_sections.append("|------|---------|--------|--------------|")
            for i, node in enumerate(top_nodes_data, 1):
                node_id = node.get("id", "N/A")[:8] + "..."
                metric_val = node.get("metric", {}).get("value", "N/A")
                analysis = node.get("analysis", "")[:100] + "..." if len(node.get("analysis", "")) > 100 else node.get("analysis", "N/A")
                analysis = analysis.replace("\n", " ").replace("|", "/")
                md_sections.append(f"| {i} | `{node_id}` | {metric_val} | {analysis} |")
            md_sections.append("")

            # Detailed breakdown for each top node
            md_sections.append("### Top Nodes Details")
            md_sections.append("")
            for i, node in enumerate(top_nodes_data, 1):
                md_sections.append(f"#### Rank {i}: `{node.get('id', 'N/A')}`")
                md_sections.append(f"- **Metric**: {node.get('metric', {}).get('value', 'N/A')}")
                if node.get("plan"):
                    plan_preview = node["plan"][:500] + "..." if len(node["plan"]) > 500 else node["plan"]
                    md_sections.append(f"- **Plan**: {plan_preview}")
                if node.get("analysis"):
                    md_sections.append(f"- **Analysis**: {node['analysis']}")
                vlm_summary = node.get("vlm_feedback_summary", [])
                if vlm_summary:
                    md_sections.append(f"- **VLM Feedback**: {'; '.join(vlm_summary[:3])}")
                md_sections.append("")
        else:
            md_sections.append("No top nodes data available.")
            md_sections.append("")

        # ===== Section 5: LLM-Generated Sections (from 3-tier memory) =====
        md_sections.append("## Memory-Based Analysis")
        md_sections.append("")
        # Convert section keys to readable headers and add content
        for key, value in sections.items():
            if key in ("resources_used", "artifacts_index"):
                # Handle separately
                continue

            # Convert snake_case to Title Case for headers
            header = " ".join(word.capitalize() for word in key.replace("_", " ").split())

            md_sections.append(f"### {header}")

            # Handle different value types
            if isinstance(value, str):
                md_sections.append(value)
            elif isinstance(value, (list, dict)):
                # For complex types, convert to readable format
                md_sections.append("```json")
                md_sections.append(json.dumps(value, indent=2))
                md_sections.append("```")
            else:
                md_sections.append(str(value))

            md_sections.append("")

        # ===== Section 6: Resources Used =====
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

        # ===== Section 7: Execution Feedback =====
        md_sections.append("## Execution Feedback")
        md_sections.append("")
        if best_node_data and best_node_data.get("exec_time_feedback"):
            md_sections.append(_to_md_str(best_node_data["exec_time_feedback"]))
        else:
            md_sections.append("No execution time feedback recorded.")
        md_sections.append("")

        # ===== Section 8: Negative Results & Failures =====
        md_sections.append("## Negative Results & Lessons Learned")
        md_sections.append("")
        if failure_notes:
            for i, note in enumerate(failure_notes, 1):
                md_sections.append(f"### Failure {i}")
                md_sections.append(_to_md_str(note))
                md_sections.append("")
        else:
            md_sections.append("No significant failures recorded.")
            md_sections.append("")

        # ===== Section 9: Provenance =====
        md_sections.append("## Provenance Chain")
        md_sections.append("")
        if provenance:
            md_sections.append("```")
            md_sections.append(" -> ".join(provenance))
            md_sections.append("```")
        else:
            md_sections.append("No provenance information available.")
        md_sections.append("")

        # Write outputs
        md_path = memory_dir / str(_cfg_get(self.config, "final_memory_filename_md", "final_memory_for_paper.md"))
        json_path = memory_dir / str(_cfg_get(self.config, "final_memory_filename_json", "final_memory_for_paper.json"))

        # Add node data to sections for JSON output
        sections["best_node_data"] = best_node_data
        sections["top_nodes_data"] = top_nodes_data

        md_path.write_text("\n".join(md_sections), encoding="utf-8")
        json_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")
        return sections
