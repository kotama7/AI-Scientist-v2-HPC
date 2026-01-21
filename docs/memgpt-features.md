# MemGPT Features (MemGPT機能の有無)

This document describes the MemGPT-style memory features available in the
HPC-AutoResearch system.

## Feature Overview

| Feature | Status | Description |
|---------|--------|-------------|
| Hierarchical Memory | Implemented | Core/Recall/Archival three-tier memory |
| Branch Isolation | Implemented | Child nodes inherit but don't pollute siblings |
| LLM Compression | Implemented | Intelligent compression via LLM |
| FTS5 Search | Implemented | Full-text search for archival memory |
| Resource Tracking | Implemented | Resource snapshots in long-term memory |
| Persistence | Implemented | SQLite-based persistent storage |
| Final Memory Export | Implemented | End-of-run memory summary for papers |

## Memory Layers

### Core Memory

**Purpose**: Always-injected essential context that persists across all prompts.

**Characteristics**:
- Bounded by `memory.core_max_chars` (default 16000)
- Contains pinned entries with importance levels (1-5)
- Automatic eviction when budget exceeded (lowest importance first)
- Entries can have TTL (time-to-live)

**Default Pinned Entries**:
| Key | Content | Importance |
|-----|---------|------------|
| `IDEA_SUMMARY` | Compressed research idea bullets | 5 |
| `PHASE0_SUMMARY` | Phase 0 planning summary | 5 |
| `RESOURCE_INDEX` | Resource file digest | 4 |
| `CURRENT_STAGE` | Current tree search stage | 3 |

### Recall Memory

**Purpose**: Recent event timeline for the current branch.

**Characteristics**:
- Windowed by `memory.recall_max_events` (default 20)
- Events include timestamp, type, and content
- Branch-specific (child inherits parent's timeline)
- FIFO eviction when window exceeded

**Event Types**:
- `phase_complete`: Phase transition events
- `code_generated`: Code generation events
- `execution_result`: Run output events
- `error_encountered`: Error events
- `metric_extracted`: Metric extraction events
- `plot_generated`: Plot generation events

### Archival Memory

**Purpose**: Long-term searchable memory for detailed content.

**Characteristics**:
- Unlimited storage (SQLite-backed)
- FTS5 full-text search (with keyword fallback)
- Tagged entries for efficient retrieval
- Retrieved via `memory.retrieval_k` limit (default 8)

**Common Tags**:
| Tag | Content |
|-----|---------|
| `PHASE0_INTERNAL` | Full Phase 0 planning output |
| `IDEA_MD` | Full idea markdown |
| `ROOT_IDEA` | Original idea (immutable) |
| `resource:<class>:<name>` | Resource item details |
| `resource_path:<path>` | Resource by path |
| `code:<node_id>` | Code snapshots |
| `results:<node_id>` | Execution results |

## Branch Behavior

### Inheritance Model

```
           ROOT (node_0)
          /            \
     node_1           node_2
    /      \              \
 node_3   node_4        node_5
```

**Inheritance Rules**:
1. Child nodes inherit Core memory from parent
2. Child nodes inherit Recall timeline (events up to fork point)
3. Child nodes can search parent's Archival memory
4. Writes are isolated to current branch

**Example**:
- `node_3` can see Core/Recall/Archival from `node_1` and `ROOT`
- `node_3` cannot see memory from `node_2`, `node_4`, or `node_5`
- Writes from `node_3` are invisible to `node_4` (sibling)

## LLM Compression

### Purpose

Compress long text to fit within character budgets while preserving key
information.

### Configuration

```yaml
memory:
  use_llm_compression: true
  compression_model: gpt-5.2
  memory_budget_chars: 24000
  section_budgets:
    idea_summary: 9600
    idea_section_limit: 4800
    phase0_summary: 5000
    archival_snippet: 3000
    results: 4000
```

### Compression Flow

```
Original Text (10000 chars)
         │
         v
    ┌─────────────────┐
    │ Check Cache     │
    │ (text hash key) │
    └────────┬────────┘
             │ miss
             v
    ┌─────────────────┐
    │ LLM Compression │
    │ (iterative)     │
    └────────┬────────┘
             │
             v
    ┌─────────────────┐
    │ Validate Length │
    │ (<= max_chars)  │
    └────────┬────────┘
             │ fail
             v
    ┌─────────────────┐
    │ Fallback        │
    │ Truncation      │
    └────────┬────────┘
             │
             v
    Compressed Text (3000 chars)
```

### Compression Prompt

Located at `prompt/config/memory/compression.txt`:

```
Compress the following {context_hint} text to fit within {max_chars} characters.
Preserve:
- Key facts, metrics, and numerical values
- Critical decisions and conclusions
- Essential technical details
- Important relationships and dependencies
```

### Caching

Compression results are cached by:
- Text content hash (sha256)
- Target character count
- Context hint

Cache is in-memory (not persisted).

## Resource Tracking

### Resource Index

When memory is enabled, resource files are tracked:

```python
RESOURCE_INDEX = {
    "digest": "sha256:abc123...",  # Content hash
    "resource_file_sha": "...",    # File hash
    "config_normalized": {...},    # Essential config fields
    "items": [
        {
            "id": "dataset-1",
            "class": "local_data",
            "fetch_status": "available",
            "path": "/workspace/data/...",
            "tree_summary": "...",
            "content_excerpt": "..."
        }
    ]
}
```

### Fetch Status Tracking

| Status | Meaning |
|--------|---------|
| `pending` | Resource declared but not fetched |
| `available` | Resource fetched and accessible |
| `failed` | Fetch attempt failed |

### Resource Memory Operations

```python
# Update resource index
mem_resources_index_update(index_data)

# Upsert resource item snapshot
mem_resources_snapshot_upsert(item_snapshot)

# Refresh after Phase 1 fetch
mem_resources_resolve_and_refresh()

# Track resource usage
track_resource_usage(resource_id, usage_type)
```

## Persistence

### Database Schema

SQLite database at `experiments/<run>/memory/memory.sqlite`:

**Tables**:
- `core_memory`: Core entries (key, value, importance, ttl, branch_id)
- `recall_memory`: Recall events (timestamp, type, content, branch_id)
- `archival_memory`: Archival records (id, content, tags_json, created_at)
- `branches`: Branch hierarchy (id, parent_id, node_name)
- `fts_archival`: FTS5 virtual table for full-text search

### Output Files

End-of-run memory exports:
- `experiments/<run>/memory/final_memory_for_paper.md`: Human-readable summary
- `experiments/<run>/memory/final_memory_for_paper.json`: Structured JSON
- `experiments/<run>/memory/resource_snapshot.json`: Resource tracking data

## Configuration Reference

Full memory configuration in `bfts_config.yaml`:

```yaml
memory:
  # Core settings
  enabled: true
  db_path: null                    # Auto: experiments/<run>/memory/memory.sqlite
  core_max_chars: 16000
  recall_max_events: 20
  retrieval_k: 8
  use_fts: auto                    # auto/true/false

  # Persistence toggles
  persist_phase0_internal: true
  always_inject_phase0_summary: true
  persist_idea_md: true
  always_inject_idea_summary: true
  final_memory_enabled: true
  final_memory_filename_md: final_memory_for_paper.md
  final_memory_filename_json: final_memory_for_paper.json
  redact_secrets: true

  # LLM compression
  use_llm_compression: true
  compression_model: gpt-5.2
  memory_budget_chars: 24000

  # Section budgets
  datasets_tested_budget_chars: 4000
  metrics_extraction_budget_chars: 4000
  plotting_code_budget_chars: 4000
  plot_selection_budget_chars: 4000
  vlm_analysis_budget_chars: 4000
  node_summary_budget_chars: 4000
  parse_metrics_budget_chars: 4000

  section_budgets:
    idea_summary: 9600
    idea_section_limit: 4800
    phase0_summary: 5000
    archival_snippet: 3000
    results: 4000

  # Logging
  memory_log_enabled: true
  memory_log_dir: null              # Auto: experiments/<run>/memory/
  memory_log_max_chars: 1600
```

## CLI Flags

```bash
# Enable memory
--enable_memgpt

# Custom database path
--memory_db /path/to/memory.sqlite

# Tune injection limits
--memory_core_max_chars 16000
--memory_recall_max_events 20
--memory_retrieval_k 8

# Compression settings
--memory_max_compression_iterations 3
```
