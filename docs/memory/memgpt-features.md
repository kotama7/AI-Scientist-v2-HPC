# MemGPT Features (Enabled vs Disabled)

This document describes the MemGPT-style memory features available in the
HPC-AutoResearch system.

## Feature Overview

| Feature | Status | Description |
|---------|--------|-------------|
| Hierarchical Memory | Implemented | Core/Recall/Archival three-tier memory |
| Branch Isolation | Implemented | Child nodes inherit but don't pollute siblings |
| LLM Compression | Implemented | Intelligent compression via LLM |
| FTS5 Search | Implemented | Full-text search for archival memory |
| Resource Tracking | Implemented (manual) | Resource snapshots in long-term memory (only if built) |
| Persistence | Implemented | SQLite-based persistent storage |
| Final Memory Export | Implemented | End-of-run memory summary for papers |

## Memory Layers

### Core Memory

**Purpose**: Key-value store for essential context that persists across prompts.

**Characteristics**:
- Bounded by `memory.core_max_chars` (default 2000)
- Contains entries with importance levels (1-5)
- Automatic eviction when budget exceeded (lowest importance first)
- Entries can have TTL (time-to-live)

**Database Keys** (stored in `core_kv` table):
| Key | Content | Save Method | Prompt Injection |
|-----|---------|-------------|------------------|
| `RESOURCE_INDEX` | Resource digest and paths | Optional (if snapshot/index built) | Separate "Resource Index" section |
| (LLM-defined keys) | e.g., `optimal_threads`, `best_flags` | LLM-managed | "Core Memory" section |

**Prompt Injection Structure** (in the system message):
```
## Memory

Resource Index:                    ← RESOURCE_INDEX (if present, separate section)
{digest: "sha256...", items: [...]}

Core Memory:                       ← Only keys saved by the LLM
- optimal_threads: 8               ← Arbitrary keys saved by the LLM
- best_compiler: -O3               ← Arbitrary keys saved by the LLM

Recall Memory:
...
```

**Note**:
- `RESOURCE_INDEX` is injected only if a snapshot/index was created. It appears
  as a separate "Resource Index" section, not inside "Core Memory".
- All Core Memory keys are saved by the LLM via `<memory_update>` (no reserved keys).
- If the LLM does not save anything, the Core Memory section is empty.

### Recall Memory

**Purpose**: Recent event timeline for the current branch.

**Characteristics**:
- Windowed by `memory.recall_max_events` (default 5)
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
- Retrieved via `memory.retrieval_k` limit (default 4)

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
  paper_section_mode: idea_then_memory  # memory_summary | idea_then_memory
  paper_section_count: 12
  max_compression_iterations: 5
  # Per-section character budgets (flat keys under memory)
  datasets_tested_budget_chars: 8000
  metrics_extraction_budget_chars: 12000
  plotting_code_budget_chars: 8000
  plot_selection_budget_chars: 8000
  vlm_analysis_budget_chars: 12000
  node_summary_budget_chars: 8000
  parse_metrics_budget_chars: 12000
  archival_snippet_budget_chars: 12000
  results_budget_chars: 12000
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
# Track resource usage
track_resource_usage(resource_id, usage_type)
```

## Persistence

### Database Schema

SQLite database at `experiments/<run>/memory/memory.sqlite`:

**Tables**:
- `core_kv`: Core entries (key, value, branch_id, updated_at)
- `core_meta`: Core metadata (importance, ttl)
- `recall_events`: Recall events (ts, kind, text, tags, branch_id)
- `archival`: Archival records (id, text, tags, created_at)
- `branches`: Branch hierarchy (branch_id, parent_branch_id, node_uid, created_at)
- `archival_fts`: FTS5 virtual table for full-text search (if enabled)

### Output Files

End-of-run memory exports:
- `experiments/<run>/memory/final_memory_for_paper.md`: Human-readable summary
- `experiments/<run>/memory/final_memory_for_paper.json`: Structured JSON
- `experiments/<run>/memory/resource_snapshot.json`: Resource tracking data (only if a snapshot was created)

## Configuration Reference

Full memory configuration in `bfts_config.yaml`:

```yaml
memory:
  # Core settings
  enabled: true                     # Memory is ENABLED by default
  db_path: null                     # Auto: experiments/<run>/memory/memory.sqlite
  core_max_chars: 2000
  recall_max_events: 5
  retrieval_k: 4
  use_fts: auto                     # auto/true/false

  # Persistence toggles
  final_memory_enabled: true
  final_memory_filename_md: final_memory_for_paper.md
  final_memory_filename_json: final_memory_for_paper.json
  redact_secrets: true

  # LLM compression
  use_llm_compression: true
  compression_model: gpt-5.2
  memory_budget_chars: 24000
  paper_section_mode: idea_then_memory  # memory_summary | idea_then_memory
  paper_section_count: 12
  max_compression_iterations: 5

  # Section budgets (flat keys under memory)
  datasets_tested_budget_chars: 8000
  metrics_extraction_budget_chars: 12000
  plotting_code_budget_chars: 8000
  plot_selection_budget_chars: 8000
  vlm_analysis_budget_chars: 12000
  node_summary_budget_chars: 8000
  parse_metrics_budget_chars: 12000
  archival_snippet_budget_chars: 12000
  results_budget_chars: 12000

  # Writeup memory limits
  writeup_recall_limit: 10
  writeup_archival_limit: 10
  writeup_core_value_max_chars: 5000
  writeup_recall_text_max_chars: 5000
  writeup_archival_text_max_chars: 5000

  # Logging
  memory_log_enabled: true
  memory_log_dir: null              # Auto: experiments/<run>/memory/
  memory_log_max_chars: 1000

  # Memory Pressure Management
  auto_consolidate: true
  consolidation_trigger: high
  recall_consolidation_threshold: 1.5
  max_memory_read_rounds: 5
  pressure_thresholds:
    medium: 0.7
    high: 0.85
    critical: 0.95
```

## CLI Flags

```bash
# Enable memory
--enable_memgpt

# Custom database path
--memory_db /path/to/memory.sqlite

# Tune injection limits
--memory_core_max_chars 2000
--memory_recall_max_events 5
--memory_retrieval_k 4

# Compression settings
--memory_max_compression_iterations 5
```

## Disabled Mode (MemGPT Off Behavior)

When MemGPT is disabled (`memory.enabled=false` or `--enable_memgpt` not passed),
the following features are **completely unavailable**:

### Features lost without MemGPT

| Feature | Impact |
|---------|--------|
| Hierarchical Memory | No Core/Recall/Archival layers |
| Branch Isolation | No memory inheritance between nodes |
| LLM Compression | No intelligent content compression |
| FTS5 Search | No full-text search of archived content |
| Resource Tracking | No resource usage persistence |
| Final Memory Export | No `final_memory_for_paper.*` files |
| SQLite Persistence | No `memory/memory.sqlite` |

### Context management without MemGPT

**Critical**: Without MemGPT, there is **no context budget management**.

- Idea descriptions, task descriptions, and phase summaries are injected
  **as full text** into every prompt.
- No truncation or compression is applied to the main context.
- If the total prompt exceeds the LLM's context window, the API may:
  - Return a `context_length_exceeded` error
  - Silently truncate input (losing important information)

### Truncation that still exists (independent of MemGPT)

Some fixed truncation mechanisms operate regardless of MemGPT status:

```python
# treesearch/utils/response.py - Terminal output truncation
def trim_long_string(string, threshold=50000, k=5000):
    # Truncates to first k + last k chars when exceeding threshold

# treesearch/utils/phase_execution.py - Execution log truncation
# Limits log lines (default 500) and characters (default 100000)
```

These are **safety limits for execution results only**, not general context
management. The main prompt content (idea, experiments, history) has no such
protection without MemGPT.

### When to enable MemGPT

| Scenario | MemGPT Recommended? |
|----------|---------------------|
| Quick testing / debugging | Optional |
| Short, simple experiments | Optional |
| Complex multi-stage experiments | **Yes** |
| Long-running research | **Yes** |
| Paper generation | **Strongly recommended** |
| LLM with limited context (e.g., <32K tokens) | **Required** |

### Diagnostic: Checking context size

Without MemGPT, you can monitor prompt sizes in the logs. If you see prompts
approaching your LLM's context limit, enable MemGPT to prevent failures:

```bash
python launch_scientist_bfts.py \
  --enable_memgpt \
  --memory_core_max_chars 2000 \
  --memory_retrieval_k 4 \
  ...
```
