# Phase 0: Setup and Planning - Memory Flow

This document describes the memory flow during Phase 0 (Setup and Planning).

## Phase 0 Overview

Phase 0 is executed **once per node** (same as Phase 1-4).
Each node generates its own Phase 0 plan with history from previous runs if available.

Key points:
- **Per-node execution**: Phase 0 is generated for every node (no caching)
- **History injection**: Previous run results are collected and injected into the prompt
- **Memory context**: If memory enabled, core/archival memory is injected
- **Output**: Plan is saved to `phase0_plan.json` for logging purposes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 0 FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. IDEA LOADING                                                      │    │
│  │    File: idea.md                                                     │    │
│  │    Function: _ingest_idea_md()                                       │    │
│  │                                                                       │    │
│  │    Memory writes:                                                     │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ CORE:                                                        │   │    │
│  │    │   idea_md_summary: "Compressed summary of research goals"   │   │    │
│  │    │                                                              │   │    │
│  │    │ ARCHIVAL:                                                    │   │    │
│  │    │   tags: [IDEA_MD, ROOT_IDEA]                                 │   │    │
│  │    │   content: Full idea.md text                                 │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. RESOURCE INDEXING                                                 │    │
│  │    File: resources.yaml                                              │    │
│  │    Function: _snapshot_resources_to_memory()                         │    │
│  │                                                                       │    │
│  │    Memory writes:                                                     │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ CORE:                                                        │   │    │
│  │    │   RESOURCE_INDEX: {                                          │   │    │
│  │    │     digest: "sha256...",                                     │   │    │
│  │    │     resource_sha: "...",                                     │   │    │
│  │    │     items: [{name, class, path, status, summary}]            │   │    │
│  │    │   }                                                          │   │    │
│  │    │                                                              │   │    │
│  │    │ ARCHIVAL:                                                    │   │    │
│  │    │   tags: [resource:<class>:<name>, resource_path:<path>]      │   │    │
│  │    │   content: Resource file summaries                           │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. WHOLE PLANNING (Phase 0 Planning)                                 │    │
│  │    Function: _whole_plan_query()                                     │    │
│  │                                                                       │    │
│  │    Prompt injection:                                                  │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ _inject_memory(prompt, "phase0_planning")                    │   │    │
│  │    │                                                              │   │    │
│  │    │ Injected context:                                            │   │    │
│  │    │   - Core: idea_md_summary, RESOURCE_INDEX                   │   │    │
│  │    │   - Archival: Top-K relevant resources                       │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                       │    │
│  │    LLM generates: phase0_plan.json                                   │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ {                                                            │   │    │
│  │    │   "plan": {                                                  │   │    │
│  │    │     "goal_summary": "...",                                   │   │    │
│  │    │     "phase1": {commands, notes},                             │   │    │
│  │    │     "phase2-4": {implementation details}                     │   │    │
│  │    │   },                                                         │   │    │
│  │    │   "internal": {                                              │   │    │
│  │    │     environment_analysis, assumptions, constraints...        │   │    │
│  │    │   }                                                          │   │    │
│  │    │ }                                                            │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. INGEST PHASE 0 INTERNAL INFO                                      │    │
│  │    Function: _ingest_phase0_internal_info()                          │    │
│  │                                                                       │    │
│  │    Memory writes:                                                     │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ CORE:                                                        │   │    │
│  │    │   phase0_summary: "Compressed summary of Phase 0 plan"      │   │    │
│  │    │                                                              │   │    │
│  │    │ ARCHIVAL:                                                    │   │    │
│  │    │   tags: [PHASE0_INTERNAL]                                    │   │    │
│  │    │   content: Full internal info (environment analysis, etc.)  │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Function Flow

### 1. Idea Loading (`_ingest_idea_md`)

**Location**: `parallel_agent.py`

**Trigger**: Called during `ParallelAgent.__init__()` when memory is enabled

**Input**: `idea.md` file content

**Memory Operations**:
```python
# Write to archival (full content)
memory_manager.mem_archival_write(
    text=idea_md_content,
    tags=["IDEA_MD", "ROOT_IDEA"],
    meta={"source": "idea_md", "node_id": "root"}
)

# Write compressed summary to core
summary = compress_text(idea_md_content, max_chars=idea_summary_budget)
memory_manager.set_core(
    branch_id=root_branch_id,
    key="idea_md_summary",
    value=summary,
    importance=5,  # Always keep
    op_name="ingest_idea_md"
)
```

### 2. Resource Indexing (`_snapshot_resources_to_memory`)

**Location**: `memgpt_store.py`

**Trigger**: Called during initialization when resources.yaml exists

**Input**: Parsed resources.yaml configuration

**Memory Operations**:
```python
# Write RESOURCE_INDEX to core (pinned, high importance)
memory_manager.set_core(
    branch_id=root_branch_id,
    key="RESOURCE_INDEX",
    value=json.dumps(resource_index),
    importance=5,  # Pinned
    op_name="resource_snapshot"
)

# Write each resource item to archival
for item in resource_items:
    memory_manager.mem_archival_write(
        text=item.summary,
        tags=[
            f"resource:{item.class_name}:{item.name}",
            f"resource_path:{item.path}"
        ],
        meta={...}
    )
```

### 3. Whole Planning (`_whole_plan_query`)

**Location**: `parallel_agent.py`

**Trigger**: Called from `run_with_phase0_planning()`

**Prompt Structure**:
```python
prompt = {
    "Introduction": PHASE0_WHOLE_PLANNING_PROMPT,
    "Research idea": task_desc,
    "Instructions": {...},
    "Response format": PHASE0_RESPONSE_FORMAT,
}

# Memory injection
_inject_memory(prompt, "phase0_planning")

# Adds to prompt:
# prompt["Memory"] = {
#   "Core Memory": idea_md_summary, RESOURCE_INDEX
#   "Archival Memory": Top-K resource items
#   "Memory Operations": Instructions for <memory_update>
# }
```

**LLM Output**: `phase0_plan.json` containing:
- `plan.goal_summary` - High-level goals
- `plan.phase1` - Download/install commands
- `plan.phase2-4` - Implementation approach
- `internal` - Environment analysis, assumptions, constraints

### 4. Ingest Phase 0 Internal Info (`_ingest_phase0_internal_info`)

**Location**: `parallel_agent.py`

**Trigger**: Called after Phase 0 planning completes successfully

**Input**: `internal` section from phase0_plan.json

**Memory Operations**:
```python
# Write full internal info to archival
memory_manager.mem_archival_write(
    text=json.dumps(internal_info),
    tags=["PHASE0_INTERNAL"],
    meta={"source": "phase0_planning", "node_id": "root"}
)

# Compress and write to core
summary = compress_text(internal_text, max_chars=phase0_summary_budget)
memory_manager.set_core(
    branch_id=root_branch_id,
    key="phase0_summary",
    value=summary,
    importance=5,  # Always keep
    op_name="ingest_phase0_internal"
)
```

## Memory State After Phase 0

After Phase 0 completes, the memory contains:

### Core Memory (Always Visible)
| Key | Content | Importance |
|-----|---------|------------|
| `idea_md_summary` | Compressed research goals | 5 (pinned) |
| `phase0_summary` | Compressed Phase 0 plan | 5 (pinned) |
| `RESOURCE_INDEX` | Resource digest and paths | 5 (pinned) |

### Archival Memory (Searchable)
| Tags | Content |
|------|---------|
| `IDEA_MD`, `ROOT_IDEA` | Full idea.md text |
| `PHASE0_INTERNAL` | Full Phase 0 internal info |
| `resource:*` | Resource file summaries |

### Recall Memory
Empty at this point (no events yet)

## Files Generated

Phase 0 also writes files to disk:

```
experiments/<run>/
├── plans/
│   ├── phase0_plan.json      # Full Phase 0 plan
│   ├── phase0_history_full.json  # Full history
│   └── phase0_llm_output.txt # Raw LLM output
├── prompts/
│   ├── phase0_prompt.json    # Prompt sent to LLM
│   └── phase0_prompt.md      # Human-readable prompt
└── memory/
    ├── memory.sqlite         # Memory database
    └── memory_calls.jsonl    # Operation log
```

## See Also

- [memory_flow.md](memory_flow.md) - Overview
- [memory_flow_phases.md](memory_flow_phases.md) - Phase 1-4 flow
