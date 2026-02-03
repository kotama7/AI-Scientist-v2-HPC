# Phase 0: Setup and Planning - Memory Flow

This document describes the memory flow during Phase 0 (Setup and Planning).

## Phase 0 Overview

Phase 0 is executed **once per node** (same as Phase 1-4).
Each node generates its own Phase 0 plan with history from previous runs if available.

Key points:
- **Per-node execution**: Phase 0 is generated for every node (no caching)
- **History injection**: Previous run results are collected and injected into the prompt
- **Memory injection**: If memory is enabled, memory context is auto-injected into the prompt (same as Phase 1-4)
- **Memory operations**: Phase 0 responses must start with a `<memory_update>` block; read operations can access previous data
- **LLM-managed memory**: `idea_md_summary`, `phase0_summary`, and `PHASE0_INTERNAL` are managed by the LLM via `<memory_update>` blocks
- **Output**: Plan is saved to `phase0_plan.json` for logging purposes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 0 FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 1. IDEA LOADING (Prompt Injection)                                 │    │
│  │    File: idea.md                                                   │    │
│  │                                                                    │    │
│  │    The idea.md content is directly injected into the LLM prompt.  │    │
│  │    LLM can optionally save summaries to memory via <memory_update>│    │
│  │                                                                    │    │
│  │    LLM-managed memory (optional):                                  │    │
│  │    ┌──────────────────────────────────────────────────────────┐   │    │
│  │    │ CORE (if LLM chooses to save):                           │   │    │
│  │    │   idea_md_summary: "Compressed summary of research goals"│   │    │
│  │    │                                                          │   │    │
│  │    │ ARCHIVAL (if LLM chooses to save):                       │   │    │
│  │    │   tags: [IDEA_MD, ROOT_IDEA]                             │   │    │
│  │    │   content: Key information from idea.md                  │   │    │
│  │    └──────────────────────────────────────────────────────────┘   │    │
│  └───────────────────────────────────┬────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 2. OPTIONAL RESOURCE SNAPSHOT                                      │    │
│  │    File: resources.yaml/json (if provided)                         │    │
│  │    Function: resource_memory.build_resource_snapshot() (manual)    │    │
│  │                                                                    │    │
│  │    Memory writes (only if you persist the snapshot):               │    │
│  │    ┌──────────────────────────────────────────────────────────┐   │    │
│  │    │ CORE:                                                    │   │    │
│  │    │   RESOURCE_INDEX: {                                      │   │    │
│  │    │     digest: "sha256...",                                 │   │    │
│  │    │     resource_sha: "...",                                 │   │    │
│  │    │     items: [{name, class, path, status, summary}]        │   │    │
│  │    │   }                                                      │   │    │
│  │    │                                                          │   │    │
│  │    │ ARCHIVAL:                                                │   │    │
│  │    │   tags: [resource:<class>:<name>, resource_path:<path>]  │   │    │
│  │    │   content: Resource file summaries                       │   │    │
│  │    └──────────────────────────────────────────────────────────┘   │    │
│  └───────────────────────────────────┬────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 3. WHOLE PLANNING (Phase 0 Planning)                               │    │
│  │    Function: _execute_phase0_planning()                            │    │
│  │                                                                    │    │
│  │    Prompt:                                                        │    │
│  │    - memory enabled: `prompt/config/phases/phase0_planning_with_memory.txt` │
│  │    - memory disabled: `prompt/config/phases/phase0_planning.txt`   │    │
│  │    + Memory context injected via render_for_prompt() (same as Phase 1-4) │
│  │                                                                    │    │
│  │    LLM response format (memory enabled):                          │    │
│  │    ┌──────────────────────────────────────────────────────────┐   │    │
│  │    │ <memory_update>{...}</memory_update>                     │   │    │
│  │    │ { "plan": {...} }                                         │   │    │
│  │    └──────────────────────────────────────────────────────────┘   │    │
│  │                                                                    │    │
│  │    If memory disabled: response is JSON only.                      │    │
│  │    Memory updates (if present) are extracted and applied before parsing. │    │
│  └───────────────────────────────────┬────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 4. POST-PHASE 0 MEMORY (LLM-Managed)                               │    │
│  │                                                                    │    │
│  │    LLM-managed memory via <memory_update> blocks:                  │    │
│  │    ┌──────────────────────────────────────────────────────────┐   │    │
│  │    │ CORE (if LLM chooses to save):                           │   │    │
│  │    │   phase0_summary: "Compressed summary of Phase 0 plan"   │   │    │
│  │    │                                                          │   │    │
│  │    │ ARCHIVAL (if LLM chooses to save):                       │   │    │
│  │    │   tags: [PHASE0_INTERNAL]                                │   │    │
│  │    │   content: Environment analysis, assumptions, constraints│   │    │
│  │    └──────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Function Flow

### 1. Idea Loading (Prompt Injection)

**Note**: There is no `_ingest_idea_md()` function. The idea.md content is directly injected into the LLM prompt.

**Behavior**:
- The `idea.md` file is read and included in the prompt context
- LLM can optionally use `<memory_update>` to save summaries to memory
- `idea_md_summary` and archival entries with `IDEA_MD` tag are **LLM-managed** (not auto-saved)

**LLM Memory Operations (optional)**:
```json
<memory_update>
{
  "mem_core_set": {
    "idea_md_summary": "Brief summary of the research goals..."
  },
  "mem_archival_write": [
    {
      "text": "Key information from idea.md...",
      "tags": ["IDEA_MD", "ROOT_IDEA"]
    }
  ]
}
</memory_update>
```

### 2. Optional Resource Snapshot (manual)

**Location**: `ai_scientist/memory/resource_memory.py`

**Trigger**: Not automatic. Call `build_resource_snapshot()` yourself and persist
the snapshot to memory if you want a `RESOURCE_INDEX` section or archival
resource items to appear in prompts.

### 3. Whole Planning (`_execute_phase0_planning`)

**Location**: `parallel_agent.py`

**Trigger**: Called from `run_with_phase0_planning()` after fork

**Memory Injection** (same as Phase 1-4):
```python
# Memory context is injected before the LLM query
if memory_cfg and memory_cfg.enabled and memory_manager and branch_id:
    memory_context = memory_manager.render_for_prompt(
        branch_id,
        task_hint="phase0",
        budget_chars=memory_cfg.memory_budget_chars,
    )
    if memory_context:
        prompt["Memory"] = memory_context + memory_ops_instructions
```

**Prompt Structure**:
```python
prompt = {
    "Introduction": PHASE0_WHOLE_PLANNING_PROMPT,  # or phase0_planning_with_memory when enabled
    "Task": task_desc,
    "History": history_injection,
    "Environment": environment_context,
    "Resources": resources_context (optional),
    "Memory": memory_context + instructions,  # Injected if memory enabled
}

# Memory Operations are specified inside the phase0 planning prompt when enabled:
# - <memory_update> is REQUIRED before the JSON plan
# - Read operations (mem_core_get / mem_archival_search / mem_recall_search) are supported
# - LLM can see existing memory data and use read operations to access it
```

**LLM Output**:
- If memory enabled: the raw response begins with `<memory_update>...</memory_update>` followed by the JSON plan.
- If memory disabled: the response is JSON only.
- Memory updates are extracted and applied; the JSON plan (without the memory block) is saved to `phase0_plan.json`.

### 4. Post-Phase 0 Memory (LLM-Managed)

**Note**: There is no `_ingest_phase0_internal_info()` function. Memory management is delegated to the LLM.

**Behavior**:
- `<memory_update>` is required when memory is enabled (an empty `{}` block is acceptable if there is nothing to save)
- LLM decides what information is important enough to persist
- `phase0_summary` and `PHASE0_INTERNAL` archival entries are **LLM-managed**
- If read operations are used, the system returns `<memory_results>` and allows follow-up updates (same read flow as later phases)

**LLM Memory Operations (optional)**:
```json
<memory_update>
{
  "mem_core_set": {
    "phase0_summary": "Environment: Linux + CUDA 12.1, Strategy: OpenMP parallelization..."
  },
  "mem_archival_write": [
    {
      "text": "Detailed environment analysis and constraints...",
      "tags": ["PHASE0_INTERNAL"]
    }
  ]
}
</memory_update>
```

## Memory State After Phase 0

After Phase 0 completes, the memory may contain (depending on LLM behavior):

### Core Memory (Always Visible)
| Key | Content | Source |
|-----|---------|--------|
| `RESOURCE_INDEX` | Resource digest and paths | Optional (only if snapshot/index is built) |
| `idea_md_summary` | Compressed research goals | LLM-managed (optional) |
| `phase0_summary` | Compressed Phase 0 plan | LLM-managed (optional) |

### Archival Memory (Searchable)
| Tags | Content | Source |
|------|---------|--------|
| `resource:*` | Resource file summaries | Optional (only if snapshot/index is built) |
| `IDEA_MD`, `ROOT_IDEA` | Key idea information | LLM-managed (optional) |
| `PHASE0_INTERNAL` | Phase 0 internal info | LLM-managed (optional) |

### Recall Memory
May contain events from previous runs or parent branches (accessible via memory injection)

## Files Generated

Phase 0 also writes files to disk:

```
experiments/<run>/
├── runs/
│   └── workers/worker-*/
│       ├── plans/
│       │   ├── phase0_plan.json         # Full Phase 0 plan
│       │   ├── phase0_history_full.json # Full history
│       │   └── phase0_llm_output.txt    # Raw LLM output
│       └── prompt_logs/
│           ├── phase0_prompt.json       # Prompt sent to LLM (if log_prompts)
│           └── phase0_prompt.md         # Human-readable prompt
├── memory/
│   └── memory.sqlite                    # Memory database
└── logs/
    └── memory/
        └── memory_calls.jsonl           # Operation log (plus memory_*.jsonl)
```

## See Also

- [memory-flow.md](memory-flow.md) - Overview
- [memory-flow-phases.md](memory-flow-phases.md) - Phase 1-4 flow
