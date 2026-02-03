# MemGPT Implementation (Implementation Details)

This document provides technical details about the MemGPT-style memory
implementation in `ai_scientist/memory/`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MemoryManager                                  │
│  (ai_scientist/memory/memgpt_store.py)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Core Layer   │  │ Recall Layer │  │ Archival     │  │ Resource    │ │
│  │              │  │              │  │ Layer        │  │ Tracker     │ │
│  │ - Key/Value  │  │ - Timeline   │  │ - FTS5       │  │ - Snapshot  │ │
│  │ - Importance │  │ - Window     │  │ - Tags       │  │ - Digest    │ │
│  │ - TTL        │  │ - Branch     │  │ - Search     │  │ - Usage     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│         │                 │                 │                 │         │
│         └─────────────────┴─────────────────┴─────────────────┘         │
│                                    │                                     │
│                                    v                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      SQLite Database                              │  │
│  │  - core_kv, core_meta, events, archival, branches tables         │  │
│  │  - archival_fts (FTS5 virtual table)                             │  │
│  │  - inherited_exclusions, inherited_summaries (CoW consolidation) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Classes

### MemoryManager

Main entry point for memory operations.

**File**: `ai_scientist/memory/memgpt_store.py`

```python
class MemoryManager:
    """Hierarchical memory manager with MemGPT-style layers."""

    def __init__(
        self,
        db_path: str | Path,
        config: dict | None = None,
        llm_client: Any = None,
        llm_model: str | None = None,
    ):
        """
        Initialize memory manager.

        Args:
            db_path: Path to SQLite database file
            config: Memory configuration dict
            llm_client: LLM client for compression (optional)
            llm_model: Model name for compression (optional)
        """
```

**Key Methods**:

```python
# Core memory operations
def mem_core_get(self, keys: list[str] | None) -> dict[str, str]
def mem_core_set(
    self,
    key: str,
    value: str,
    *,
    ttl: str | None = None,
    importance: int = 3,
    branch_id: str | None = None,
) -> None
def mem_core_del(self, key: str) -> None

# Recall memory operations
def mem_recall_append(self, event: dict) -> None
def mem_recall_search(self, query: str, *, k: int = 20) -> list[dict]

# Archival memory operations
def mem_archival_write(
    self,
    text: str,
    *,
    tags: list[str],
    meta: dict | None = None,
) -> str
def mem_archival_update(
    self,
    record_id: str,
    *,
    text: str | None = None,
    tags: list[str] | None = None,
    meta: dict | None = None,
) -> None
def mem_archival_search(
    self,
    query: str,
    *,
    tags: list[str] | None = None,
    k: int = 10,
) -> list[dict]
def mem_archival_get(self, record_id: str) -> dict

# Branch operations
def mem_node_fork(
    self,
    parent_node_id: str | None,
    child_node_id: str,
    ancestor_chain: list[str] | None = None,
    phase: str | None = None,
) -> None
def mem_node_read(self, node_id: str, scope: str = "all") -> dict
def mem_node_write(
    self,
    node_id: str,
    *,
    core_updates: dict | None = None,
    recall_event: dict | None = None,
    archival_records: list[dict] | None = None,
) -> None

# LLM memory operations
def apply_llm_memory_updates(
    self,
    branch_id: str,
    updates: dict,
    node_id: str | None = None,
    phase: str | None = None,
) -> dict

# Context injection
def render_for_prompt(
    self,
    branch_id: str,
    task_hint: str | None,
    budget_chars: int,
    no_limit: bool = False,
) -> str
def render_for_prompt_with_log(
    self,
    branch_id: str,
    task_hint: str | None,
    budget_chars: int,
    no_limit: bool = False,
) -> tuple[str, dict]
```

## Database Schema

### SQL Table Definitions (Actual Implementation)

```sql
-- Branch hierarchy table
CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    parent_id TEXT NULL,
    node_uid TEXT NULL,
    created_at REAL
);

-- Core memory key-value pairs
CREATE TABLE IF NOT EXISTS core_kv (
    branch_id TEXT,
    key TEXT,
    value TEXT,
    updated_at REAL,
    PRIMARY KEY (branch_id, key)
);

-- Core memory metadata (importance, TTL)
CREATE TABLE IF NOT EXISTS core_meta (
    branch_id TEXT,
    key TEXT,
    importance INTEGER,
    ttl TEXT,  -- Note: TEXT type for flexible TTL representation
    updated_at REAL,
    PRIMARY KEY (branch_id, key)
);

-- Recall memory (events timeline)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT,
    kind TEXT,
    text TEXT,
    tags TEXT,
    created_at REAL,
    task_hint TEXT,
    memory_size INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id);

-- Archival memory (long-term storage)
CREATE TABLE IF NOT EXISTS archival (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT,
    text TEXT,
    tags TEXT,
    created_at REAL
);

-- FTS5 full-text search virtual table (if available)
CREATE VIRTUAL TABLE IF NOT EXISTS archival_fts USING fts5(
    text, tags, branch_id
);

-- Inherited memory consolidation (Copy-on-Write)
CREATE TABLE IF NOT EXISTS inherited_exclusions (
    branch_id TEXT,
    excluded_event_id INTEGER,
    excluded_at REAL,
    PRIMARY KEY (branch_id, excluded_event_id)
);

CREATE TABLE IF NOT EXISTS inherited_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT,
    summary_text TEXT,
    summarized_event_ids TEXT,
    kind TEXT,
    created_at REAL
);
```

## LLM Compression Implementation

### Compression Function

```python
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
) -> str:
    """
    Compress text using LLM to fit within max_chars.

    Args:
        text: Original text to compress
        max_chars: Target character limit
        context_hint: Description of text context (e.g., "idea summary")
        client: LLM client
        model: Model name
        prompt_template: Custom prompt template
        use_cache: Enable caching
        max_iterations: Max compression attempts

    Returns:
        Compressed text fitting within max_chars
    """
```

### Compression Flow

```python
def compress_content(self, text: str, max_chars: int, context: str) -> str:
    """Compress content with fallback."""
    # 1. Check if already fits
    if len(text) <= max_chars:
        return text

    # 2. Check cache
    cache_key = (sha256(text.encode()).hexdigest()[:16], max_chars, context)
    if cache_key in self._compression_cache:
        return self._compression_cache[cache_key]

    # 3. Try LLM compression
    if self.llm_client and self.llm_model:
        try:
            compressed = _compress_with_llm(
                text, max_chars, context,
                client=self.llm_client,
                model=self.llm_model,
                max_iterations=self.config.get("max_compression_iterations", 5)
            )
            self._compression_cache[cache_key] = compressed
            return compressed
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}")

    # 4. Fallback to truncation
    return _truncate(text, max_chars)
```

## Branch Management

### Fork Operation

```python
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

    Returns:
        None (does not return a branch ID)
    """
    parent_branch = self._resolve_branch_id(parent_node_id) if parent_node_id else None

    # If parent_node_id was provided but couldn't be resolved, we need to create
    # the missing branches. Use ancestor_chain if available for correct tree structure.
    if parent_node_id and not parent_branch:
        if ancestor_chain:
            # Create missing branches in order from root to parent
            prev_branch = self.root_branch_id
            for ancestor_id in ancestor_chain:
                existing = self._resolve_branch_id(ancestor_id)
                if not existing:
                    self.create_branch(prev_branch, node_uid=ancestor_id, branch_id=ancestor_id)
                    prev_branch = ancestor_id
                else:
                    prev_branch = existing
            parent_branch = prev_branch

    branch_id = child_node_id or uuid.uuid4().hex
    self.create_branch(parent_branch, node_uid=child_node_id, branch_id=branch_id)
```

### Visibility Chain

```python
def _branch_chain(self, branch_id: str) -> list[str]:
    """Get branch ID chain from current to root (current first)."""
    chain = []
    current = branch_id

    while current:
        chain.append(current)
        row = self._conn.execute(
            "SELECT parent_id FROM branches WHERE id = ?",
            (current,)
        ).fetchone()
        current = row["parent_id"] if row else None

    return chain  # Current first, root last
```

## LLM Memory Operations

### apply_llm_memory_updates

This method processes memory update instructions from LLM responses.

```python
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
        updates: A dict containing memory updates with optional keys (preferred
            MemGPT-style keys; legacy `core`/`archival` keys are normalized):
            - "mem_core_set": dict of key-value pairs to set in core memory
            - "mem_core_get": list of keys to retrieve from core memory
            - "mem_core_del": list of keys to delete from core memory
            - "mem_archival_write": list of dicts with "text" and "tags"
            - "mem_archival_update": list of dicts with "id" and optional "text"/"tags"
            - "mem_archival_search": dict with "query", optional "k" and "tags"
            - "mem_recall_append": dict with event data to append
            - "mem_recall_search": dict with "query" and optional "k"
            - "mem_recall_evict": dict with eviction parameters
            - "consolidate": bool to trigger memory consolidation
        node_id: Optional node ID for tracking.
        phase: Optional phase name for logging.

    Returns:
        dict containing results of read operations (mem_core_get, mem_archival_search, etc.)
    """
```

### Prompt Injection

```python
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
        Formatted memory string for prompt injection containing:
        - Resource Index section (if available)
        - Core Memory section (LLM-set key-value pairs)
        - Recall Memory section (recent events)
        - Archival Memory section (semantic search results)
    """
```

## Resource Memory Integration

### Resource Snapshot

**File**: `ai_scientist/memory/resource_memory.py`

```python
def build_resource_snapshot(
    resource_file: str | Path,
    config: dict,
    resolve_paths: bool = True
) -> dict:
    """
    Build resource snapshot for memory storage.

    Returns:
        {
            "digest": "sha256:...",
            "resource_file_sha": "...",
            "config_normalized": {...},
            "items": [
                {
                    "id": "...",
                    "class": "local_data|github|huggingface",
                    "fetch_status": "pending|available|failed",
                    "path": "...",
                    "tree_summary": "...",
                    "content_excerpt": "..."
                }
            ]
        }
    """
```

### Usage Tracking

```python
def track_resource_usage(
    memory_manager: MemoryManager,
    resource_id: str,
    usage_type: str,
    usage_context: str = None
) -> None:
    """
    Track resource usage for final memory export.

    Args:
        memory_manager: MemoryManager instance
        resource_id: Resource item ID
        usage_type: Type of usage (read, reference, include)
        usage_context: Context description
    """
    usage_record = {
        "resource_id": resource_id,
        "usage_type": usage_type,
        "context": usage_context,
        "timestamp": time.time()
    }
    memory_manager.mem_archival_write(
        text=json.dumps(usage_record),
        tags=[RESOURCE_USED_TAG, f"resource_id:{resource_id}"]
    )
```

## Final Memory Export

### Export Function

```python
def export_final_memory(self, output_dir: Path) -> tuple[Path, Path]:
    """
    Export final memory for paper generation.

    Args:
        output_dir: Output directory path

    Returns:
        Tuple of (markdown_path, json_path)
    """
    # Collect all memory
    final = {
        "core": self.mem_core_get(None),
        "recall": self.mem_recall_search("*", k=None),
        "archival_summary": self._summarize_archival(),
        "resources_used": self._get_used_resources(),
        "experiment_timeline": self._build_timeline()
    }

    # Write JSON
    json_path = output_dir / self.config.get(
        "final_memory_filename_json", "final_memory_for_paper.json"
    )
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2)

    # Write Markdown
    md_path = output_dir / self.config.get(
        "final_memory_filename_md", "final_memory_for_paper.md"
    )
    with open(md_path, "w") as f:
        f.write(self._format_final_markdown(final))

    return md_path, json_path
```

## Integration Points

### Parallel Agent Integration

```python
# In parallel_agent.py
class ParallelAgent:
    def __init__(self, config, memory_manager=None):
        self.memory_manager = memory_manager

    def _inject_memory(
        self,
        prompt: dict[str, Any],
        task_hint: str,
        branch_id: str | None = None,
        budget_chars: int | None = None,
        allow_summary_fallback: bool = False,
        allow_empty: bool = False,
        node_id: str | None = None,
    ) -> None:
        """Inject memory context into prompt."""
        if not self.memory_manager:
            return

        memory_context = self.memory_manager.render_for_prompt(
            branch_id or self.branch_id,
            task_hint,
            budget_chars or self.cfg.memory.memory_budget_chars,
        )

        if memory_context or allow_empty:
            # Add memory operation instructions
            prompt["Memory"] = memory_context + self._get_memory_ops_instructions()

    def _record_event(self, event: dict):
        """Record event to recall memory."""
        if self.memory_manager:
            self.memory_manager.mem_recall_append(event)
```

### Agent Manager Integration

```python
# In agent_manager.py
class AgentManager:
    def _write_memory_event(
        self,
        branch_id: str,
        phase: str,
        kind: str,
        summary: str,
        refs: list[str] | None = None,
        write_archival: bool = False,
        archival_tags: list[str] | None = None,
    ):
        """Handle memory event recording."""
        if not self.memory_manager:
            return

        # Write to recall memory
        self.memory_manager.mem_recall_append({
            "ts": time.time(),
            "run_id": getattr(self.memory_manager, "run_id", ""),
            "node_id": branch_id,
            "branch_id": branch_id,
            "phase": phase,
            "kind": kind,
            "summary": summary[:2000],
            "refs": refs or [],
        })

        # Optionally write to archival
        if write_archival:
            tags = archival_tags or []
            tags.extend([f"phase:{phase}", f"kind:{kind}"])
            self.memory_manager.mem_archival_write(
                text=summary,
                tags=tags,
                meta={"phase": phase, "branch_id": branch_id},
            )
```

## Testing

Unit tests in `tests/`:

```python
# test_memory.py
def test_core_memory_operations():
    """Test core memory CRUD operations."""

def test_recall_window():
    """Test recall memory windowing."""

def test_archival_search():
    """Test archival FTS5 search."""

def test_branch_isolation():
    """Test branch visibility isolation."""

def test_llm_compression():
    """Test LLM-based compression."""

def test_resource_tracking():
    """Test resource snapshot and usage tracking."""

def test_apply_llm_memory_updates():
    """Test LLM memory update application."""

def test_inherited_memory_consolidation():
    """Test Copy-on-Write inherited memory consolidation."""
```

Run tests:
```bash
pytest tests/test_memory*.py -v
```
