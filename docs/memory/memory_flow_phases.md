# Phase 1-4: Execution Phases - Memory Flow

This document describes the memory flow during Phase 1-4 execution
(Download/Install, Coding, Compile, Run) in split-phase mode.

## Phase 1-4 Overview

These phases are executed **for each node** during tree search.
In split-phase mode, they are combined into a single LLM call that generates
a structured JSON response (`phase_artifacts`).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 1-4 EXECUTION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. BRANCH CREATION                                                   │    │
│  │    Function: mem_node_fork()                                         │    │
│  │                                                                       │    │
│  │    Creates child branch inheriting parent's memory                   │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ parent_branch ──fork──> child_branch                         │   │    │
│  │    │                                                              │   │    │
│  │    │ Child inherits:                                              │   │    │
│  │    │   - Core Memory (visible)                                    │   │    │
│  │    │   - Recall Memory (visible)                                  │   │    │
│  │    │   - Archival Memory (searchable)                             │   │    │
│  │    │                                                              │   │    │
│  │    │ Child writes are isolated (siblings don't see each other)    │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. PROMPT BUILDING                                                   │    │
│  │    Functions: _generate_draft_prompt() / _generate_debug_prompt()   │    │
│  │               _generate_improve_prompt() / etc.                      │    │
│  │                                                                       │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ prompt = {                                                   │   │    │
│  │    │   "Introduction": task description,                          │   │    │
│  │    │   "Research idea": idea summary,                             │   │    │
│  │    │   "Instructions": guidelines + response format,              │   │    │
│  │    │   "System": system instructions,                             │   │    │
│  │    │   "Domain": HPC domain knowledge,                            │   │    │
│  │    │   "Environment": container/compiler info,                    │   │    │
│  │    │   "Resources": available resources,                          │   │    │
│  │    │   "Phase 0 plan": phase0_plan snippet,                       │   │    │
│  │    │ }                                                            │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. MEMORY INJECTION                                                  │    │
│  │    Function: _inject_memory(prompt, task_hint, branch_id)           │    │
│  │                                                                       │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ prompt["Memory"] = render_for_prompt(branch_id, task_hint)  │   │    │
│  │    │                                                              │   │    │
│  │    │ + Memory Operations Instructions (for split-phase mode):    │   │    │
│  │    │   "You MUST include a <memory_update> block at START..."    │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                       │    │
│  │    Recall event logged: memory_context_injected                      │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. LLM QUERY (with multi-turn memory read)                          │    │
│  │    Function: generate_phase_artifacts()                              │    │
│  │                                                                       │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ ROUND 0:                                                     │   │    │
│  │    │   LLM receives prompt                                        │   │    │
│  │    │   LLM outputs:                                               │   │    │
│  │    │     <memory_update>                                          │   │    │
│  │    │     {...} ← may include read operations                      │   │    │
│  │    │     </memory_update>                                         │   │    │
│  │    │     {"phase_artifacts": {...}}                               │   │    │
│  │    │                                                              │   │    │
│  │    │ MEMORY OPERATIONS:                                           │   │    │
│  │    │   extract_phase_artifacts() → parse <memory_update>          │   │    │
│  │    │   apply_llm_memory_updates() → execute operations            │   │    │
│  │    │   _has_memory_read_results() → check for reads               │   │    │
│  │    │                                                              │   │    │
│  │    │ IF read results AND round < max_memory_read_rounds:          │   │    │
│  │    │   ROUND 1+:                                                  │   │    │
│  │    │   prompt["Memory Read Results"] = formatted results          │   │    │
│  │    │   Re-query LLM                                               │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 5. PHASE EXECUTION                                                   │    │
│  │                                                                       │    │
│  │    phase_artifacts = {                                               │    │
│  │      "download": {commands, notes},                                  │    │
│  │      "coding": {workspace: {root, tree, files}},                    │    │
│  │      "compile": {build_plan, commands},                              │    │
│  │      "run": {commands, expected_outputs}                             │    │
│  │    }                                                                 │    │
│  │                                                                       │    │
│  │    PHASE 1 (Download):  run_commands_with_logging()                 │    │
│  │      ↓ success/fail → Recall: phase1_complete / phase1_failed       │    │
│  │                                                                       │    │
│  │    PHASE 2 (Coding):    apply_workspace_plan()                       │    │
│  │      ↓ success/fail → Recall: coding_complete / coding_failed       │    │
│  │                                                                       │    │
│  │    PHASE 3 (Compile):   run_commands_with_logging()                 │    │
│  │      ↓ success/fail → Recall: compile_complete / compile_failed     │    │
│  │      ↓ if failed → Archival: [ERROR] compilation details            │    │
│  │                                                                       │    │
│  │    PHASE 4 (Run):       run_commands_with_logging()                 │    │
│  │      ↓ success/fail → Recall: run_complete / run_failed             │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 6. NODE RESULT                                                       │    │
│  │    Function: mem_recall_append()                                     │    │
│  │                                                                       │    │
│  │    Recall: node_result {                                             │    │
│  │      node_id, branch_id, is_buggy, metric, exec_time                │    │
│  │    }                                                                 │    │
│  │                                                                       │    │
│  │    If success with metric:                                           │    │
│  │      Archival: [SUCCESS, stage:X] details                           │    │
│  │    If failure:                                                       │    │
│  │      Archival: [ERROR] exception details                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Flow by Node Type

### Draft Node (`_generate_draft_prompt`)

**Task Hint**: `draft`

**Prompt Structure**:
```python
prompt = {
    "Introduction": introduction text,
    "Research idea": task_desc,
    "Instructions": {
        "Response format": RESPONSE_FORMAT_SPLIT_PHASE_WITH_MEMORY,
        "Implementation": guidelines,
    },
}
_inject_split_phase_context(prompt)  # System, Domain, Environment, Resources
_inject_memory(prompt, "draft", branch_id=node.branch_id)
```

**Memory Events**:
- Recall: `node_created` (at start)
- Recall: `phase1_complete/failed`, `coding_complete/failed`, etc.
- Recall: `node_result` (at end)
- Archival: `[SUCCESS]` or `[ERROR]` details

### Debug Node (`_generate_debug_prompt`)

**Task Hint**: `debug`

**Prompt Structure**:
```python
prompt = {
    "Introduction": introduction text,
    "Research idea": task_desc,
    "Code": buggy_code,
    "Execution output": term_out,
    "Plot feedback": plot_feedback,  # if available
    "Instructions": {
        "Response format": RESPONSE_FORMAT_SPLIT_PHASE_WITH_MEMORY,
        "Debug": debug guidelines,
    },
}
_inject_split_phase_context(prompt)
_inject_memory(prompt, "debug", branch_id=node.branch_id)
```

### Improve Node (`_generate_improve_prompt`)

**Task Hint**: `improve`

**Prompt Structure**:
```python
prompt = {
    "Introduction": introduction text,
    "Research idea": task_desc,
    "Code": working_code,
    "Plot feedback": plot_feedback,
    "Time feedback": exec_time_feedback,
    "Instructions": {
        "Response format": RESPONSE_FORMAT_SPLIT_PHASE_WITH_MEMORY,
        "Improve": improvement guidelines,
    },
}
_inject_split_phase_context(prompt)
_inject_memory(prompt, "improve", branch_id=node.branch_id)
```

### Hyperparameter Node (`_generate_hyperparam_prompt`)

**Task Hint**: `hyperparam_node`

**Stage**: Stage 2 (hyperparameter tuning)

**Prompt Structure**:
```python
prompt = {
    "Introduction": introduction text,
    "Research idea": task_desc,
    "Base code": best_stage1_code,
    "Hyperparam idea": new_hyperparam_idea,
    "Instructions": {...},
}
_inject_split_phase_context(prompt)
_inject_memory(prompt, "hyperparam_node", branch_id=node.branch_id)
```

### Ablation Node (`_generate_ablation_prompt`)

**Task Hint**: `ablation_node`

**Stage**: Stage 4 (ablation study)

**Prompt Structure**:
```python
prompt = {
    "Introduction": introduction text,
    "Research idea": task_desc,
    "Base code": best_stage3_code,
    "Ablation idea": new_ablation_idea,
    "Instructions": {...},
}
_inject_split_phase_context(prompt)
_inject_memory(prompt, "ablation_node", branch_id=node.branch_id)
```

## Phase Execution Details

### Phase 1: Download/Install

**Function**: `run_commands_with_logging()`

**Commands from**: `phase_artifacts["download"]["commands"]`

**Memory Events**:
```python
# On completion
mem_recall_append({
    "kind": "phase1_complete" or "phase1_failed",
    "node_id": node.id,
    "branch_id": node.branch_id,
    "commands_run": len(commands),
    "error": error_details if failed
})
```

### Phase 2: Coding

**Function**: `apply_workspace_plan()`

**Files from**: `phase_artifacts["coding"]["workspace"]["files"]`

**Memory Events**:
```python
# On completion
mem_recall_append({
    "kind": "coding_complete" or "coding_failed",
    "node_id": node.id,
    "branch_id": node.branch_id,
    "files_created": len(files)
})
```

### Phase 3: Compile

**Function**: `run_commands_with_logging()`

**Commands from**: `phase_artifacts["compile"]["commands"]`

**Memory Events**:
```python
# On completion
mem_recall_append({
    "kind": "compile_complete" or "compile_failed",
    "node_id": node.id,
    "branch_id": node.branch_id,
    "error": error_lines if failed
})

# On failure - detailed archival entry
if failed:
    mem_archival_write(
        text=f"Compilation failed: {error_details}",
        tags=["ERROR", "COMPILE", f"node:{node.id}"]
    )
```

### Phase 4: Run

**Function**: `run_commands_with_logging()`

**Commands from**: `phase_artifacts["run"]["commands"]`

**Memory Events**:
```python
# On completion
mem_recall_append({
    "kind": "run_complete" or "run_failed",
    "node_id": node.id,
    "branch_id": node.branch_id,
    "outputs_found": output_files,
    "error": error_details if failed
})
```

## Response Format with Memory

In split-phase mode with memory enabled, LLM must output:

```
<memory_update>
{
  "core": {"key": "value"},
  "archival": [{"text": "insight", "tags": ["TAG"]}],
  "archival_search": {"query": "...", "k": 5}  // Optional - triggers re-query
}
</memory_update>
{"phase_artifacts": {...}, "constraints": {...}}
```

**Validation**:
- `extract_phase_artifacts()` parses `<memory_update>` block
- `MissingMemoryUpdateError` raised if block missing (when `require_memory_update=True`)
- System retries with feedback if missing

## Memory Branch Inheritance

```
root_branch (Phase 0 memory)
    │
    ├── node_1_branch (fork from root)
    │   ├── Inherits: Core, Recall, Archival from root
    │   └── Writes: isolated to this branch
    │
    ├── node_2_branch (fork from root, sibling of node_1)
    │   ├── Inherits: Core, Recall, Archival from root
    │   └── Writes: isolated (doesn't see node_1's writes)
    │
    └── node_3_branch (fork from node_1, child of node_1)
        ├── Inherits: Core, Recall, Archival from root + node_1
        └── Writes: isolated
```

## Files Generated Per Node

```
experiments/<run>/
├── phase_logs/
│   └── node_<id>/
│       ├── download.log    # Phase 1 output
│       ├── compile.log     # Phase 3 output
│       ├── run.log         # Phase 4 output
│       └── artifacts.json  # Full phase_artifacts
├── prompts/
│   └── <session>/
│       ├── draft_attempt1_round0.json
│       ├── draft_attempt1_round0.md
│       └── ...
└── memory/
    └── memory_calls.jsonl  # All memory operations
```

## See Also

- [memory_flow.md](memory_flow.md) - Overview
- [memory_flow_phase0.md](memory_flow_phase0.md) - Phase 0 flow
- [memory_flow_post_execution.md](memory_flow_post_execution.md) - Post-execution flow
