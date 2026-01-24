# MemGPT Implementation (MemGPT機能の実装)

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
│  │  - core_memory, recall_memory, archival_memory, branches tables  │  │
│  │  - fts_archival (FTS5 virtual table)                             │  │
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
def mem_core_get(self, keys: list[str] | None = None) -> dict[str, str]
def mem_core_set(self, key: str, value: str, importance: int = 3, ttl: int | None = None)
def mem_core_del(self, key: str)

# Recall memory operations
def mem_recall_append(self, event_type: str, content: str)
def mem_recall_search(self, query: str, limit: int = 10) -> list[dict]

# Archival memory operations
def mem_archival_write(self, content: str, tags: list[str] = None) -> str
def mem_archival_update(self, record_id: str, content: str, tags: list[str] = None)
def mem_archival_search(self, query: str, limit: int = None) -> list[dict]
def mem_archival_get(self, record_id: str) -> dict | None

# Branch operations
def mem_node_fork(self, parent_node_id: str, child_node_name: str) -> str
def mem_node_read(self, node_id: str, scope: str = "all") -> dict
def mem_node_write(self, node_id: str, core: dict = None, recall: str = None, archival: list = None)

# Context injection
def get_memory_context(self, task_hint: str = None) -> dict
def format_memory_for_prompt(self) -> str
```

## Database Schema

### SQL Table Definitions

```sql
-- Core memory table
CREATE TABLE IF NOT EXISTS core_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    importance INTEGER DEFAULT 3,
    ttl INTEGER,  -- Unix timestamp or NULL
    created_at REAL DEFAULT (unixepoch('subsec')),
    updated_at REAL DEFAULT (unixepoch('subsec')),
    UNIQUE(branch_id, key)
);

-- Recall memory table
CREATE TABLE IF NOT EXISTS recall_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL DEFAULT (unixepoch('subsec'))
);
CREATE INDEX IF NOT EXISTS idx_recall_branch ON recall_memory(branch_id);

-- Archival memory table
CREATE TABLE IF NOT EXISTS archival_memory (
    id TEXT PRIMARY KEY,
    branch_id TEXT NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT,  -- JSON array of tags
    created_at REAL DEFAULT (unixepoch('subsec'))
);
CREATE INDEX IF NOT EXISTS idx_archival_branch ON archival_memory(branch_id);
CREATE INDEX IF NOT EXISTS idx_archival_created ON archival_memory(created_at);

-- FTS5 full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS fts_archival USING fts5(
    content,
    content='archival_memory',
    content_rowid='rowid'
);

-- Branch hierarchy table
CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    node_name TEXT,
    created_at REAL DEFAULT (unixepoch('subsec')),
    FOREIGN KEY (parent_id) REFERENCES branches(id)
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
                max_iterations=self.config.get("max_compression_iterations", 3)
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
def mem_node_fork(self, parent_node_id: str, child_node_name: str) -> str:
    """
    Create a child branch inheriting from parent.

    Args:
        parent_node_id: Parent branch ID
        child_node_name: Name for new child node

    Returns:
        New child branch ID
    """
    child_id = str(uuid.uuid4())

    with self._get_connection() as conn:
        # Insert new branch
        conn.execute(
            "INSERT INTO branches (id, parent_id, node_name) VALUES (?, ?, ?)",
            (child_id, parent_node_id, child_node_name)
        )

        # Copy parent's core memory to child
        conn.execute("""
            INSERT INTO core_memory (branch_id, key, value, importance, ttl)
            SELECT ?, key, value, importance, ttl
            FROM core_memory WHERE branch_id = ?
        """, (child_id, parent_node_id))

    return child_id
```

### Visibility Chain

```python
def _get_ancestor_chain(self, branch_id: str) -> list[str]:
    """Get branch ID chain from root to current."""
    chain = []
    current = branch_id

    with self._get_connection() as conn:
        while current:
            chain.append(current)
            row = conn.execute(
                "SELECT parent_id FROM branches WHERE id = ?",
                (current,)
            ).fetchone()
            current = row[0] if row else None

    return list(reversed(chain))  # Root first
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
        json.dumps(usage_record),
        tags=[RESOURCE_USED_TAG, f"resource_id:{resource_id}"]
    )
```

## Context Injection

### Memory Context Builder

```python
def get_memory_context(self, task_hint: str = None) -> dict:
    """
    Build memory context for prompt injection.

    Args:
        task_hint: Task context for relevance scoring

    Returns:
        {
            "core": {"key": "value", ...},
            "recall": [{"type": "...", "content": "...", "timestamp": ...}, ...],
            "archival": [{"id": "...", "content": "...", "tags": [...], ...}, ...]
        }
    """
    context = {}

    # Core: always inject
    context["core"] = self.mem_core_get()

    # Recall: recent events within window
    context["recall"] = self.mem_recall_search(
        query="*",
        limit=self.config.get("recall_max_events", 20)
    )

    # Archival: relevant entries based on task_hint
    if task_hint:
        archival_results = self.mem_archival_search(
            query=task_hint,
            limit=self.config.get("retrieval_k", 8)
        )
    else:
        archival_results = []
    context["archival"] = archival_results

    return context
```

### Prompt Formatting

```python
def format_memory_for_prompt(self) -> str:
    """
    Format memory context as prompt string.

    Returns:
        Formatted memory section for prompt injection
    """
    context = self.get_memory_context()
    sections = []

    # Core section
    if context.get("core"):
        sections.append("## Core Memory")
        for key, value in context["core"].items():
            sections.append(f"**{key}**: {value}")

    # Recall section
    if context.get("recall"):
        sections.append("\n## Recent Events")
        for event in context["recall"][-5:]:  # Last 5 events
            sections.append(f"- [{event['type']}] {event['content'][:200]}...")

    # Archival section
    if context.get("archival"):
        sections.append("\n## Retrieved Context")
        for record in context["archival"]:
            snippet = self.compress_content(
                record["content"],
                self.config.get("section_budgets", {}).get("archival_snippet", 3000),
                "archival snippet"
            )
            sections.append(f"- {snippet}")

    return "\n".join(sections)
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
        "core": self.mem_core_get(),
        "recall": self.mem_recall_search("*", limit=None),
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
        self.memory = memory_manager

    def _inject_memory_context(self, prompt_parts: list, task_hint: str):
        """Inject memory context into prompt."""
        if self.memory and self.memory.is_enabled():
            memory_section = self.memory.format_memory_for_prompt()
            prompt_parts.insert(0, f"## Memory Context\n{memory_section}\n")

    def _record_event(self, event_type: str, content: str):
        """Record event to recall memory."""
        if self.memory and self.memory.is_enabled():
            self.memory.mem_recall_append(event_type, content)
```

### Agent Manager Integration

```python
# In agent_manager.py
class AgentManager:
    def _on_node_complete(self, node_id: str, result: dict):
        """Handle node completion."""
        if self.memory:
            # Archive results
            self.memory.mem_archival_write(
                json.dumps(result),
                tags=[f"results:{node_id}", result.get("status", "unknown")]
            )

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
```

Run tests:
```bash
pytest tests/test_memory*.py -v
```
