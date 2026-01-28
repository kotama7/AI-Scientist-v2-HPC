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

## Phase 2/3/4 Prompt Selection

When `memory.enabled=true`, the system automatically selects memory-enabled prompts
that include `<memory_update>` instructions and examples.

### Prompt Switching Table

| Task | `memory.enabled=false` | `memory.enabled=true` |
|------|------------------------|----------------------|
| **Draft** | `prompt/agent/parallel/tasks/draft/introduction.txt` | `prompt/agent/parallel/tasks/draft/introduction_with_memory.txt` |
| **Debug** | `prompt/agent/parallel/tasks/debug/introduction.txt` | `prompt/agent/parallel/tasks/debug/introduction_with_memory.txt` |
| **Improve** | `prompt/agent/parallel/tasks/improve/introduction.txt` | `prompt/agent/parallel/tasks/improve/introduction_with_memory.txt` |
| **Hyperparam** | `prompt/agent/parallel/nodes/hyperparam/introduction.txt` | `prompt/agent/parallel/nodes/hyperparam/introduction_with_memory.txt` |
| **Ablation** | `prompt/agent/parallel/nodes/ablation/introduction.txt` | `prompt/agent/parallel/nodes/ablation/introduction_with_memory.txt` |
| **Execution Review** | `prompt/agent/parallel/tasks/execution_review/introduction.txt` | `prompt/agent/parallel/tasks/execution_review/introduction_with_memory.txt` |
| **Summary** | `prompt/agent/parallel/tasks/summary/introduction.txt` | `prompt/agent/parallel/tasks/summary/introduction_with_memory.txt` |
| **Parse Metrics** | `prompt/agent/parallel/tasks/parse_metrics/introduction.txt` | `prompt/agent/parallel/tasks/parse_metrics/introduction_with_memory.txt` |
| **VLM Analysis** | `prompt/agent/parallel/vlm_analysis.txt` | `prompt/agent/parallel/vlm_analysis_with_memory.txt` |

### Code Implementation

```python
# Example from _draft() in parallel_agent.py
def _draft(self) -> Node:
    # Select prompt based on memory configuration
    draft_intro = DRAFT_INTRO_WITH_MEMORY if self._is_memory_enabled else DRAFT_INTRO
    prompt: Any = {
        "Introduction": draft_intro,
        "Research idea": self.task_desc,
        ...
    }
```

### Memory-Enabled Prompt Features

Memory-enabled prompts include:

1. **Memory Operations Section**: Instructions for `<memory_update>` blocks
2. **Write Examples**: How to write to `core` and `archival` memory
3. **Read Examples**: How to use `core_get` and `archival_search`
4. **Task-Specific Guidelines**: What information to record for each task type

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

**Stage**: Stage 2 (baseline_tuning)

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

**Stage**: Stage 4 (ablation_studies)

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

Phase 1 has two modes: **iterative** (default) and **batch**.

#### Iterative Mode (Default)

In iterative mode, Phase 1 runs as a multi-step loop with LLM-driven decision making.
Each iteration queries the LLM for the next command to execute.

**Function**: `phase1_iterative_driver()` in `parallel_agent.py`

**Prompt Selection**:
```python
phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT
if memory_cfg and getattr(memory_cfg, "enabled", False):
    phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT_WITH_MEMORY
```

| `memory.enabled` | Prompt File |
|------------------|-------------|
| `true` (default) | `prompt/config/phases/phase1_installer_with_memory.txt` |
| `false` | `prompt/config/phases/phase1_installer.txt` |

**Iteration Flow**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 1 ITERATIVE LOOP                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  for step_idx in range(phase1_max_steps):                                    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Build Prompt:                                                        │    │
│  │   - Introduction (memory-enabled or standard)                       │    │
│  │   - Task description                                                 │    │
│  │   - Phase 0 guidance (targets, preferred_commands, done_conditions) │    │
│  │   - Progress history (previous step results)                        │    │
│  │   - Memory context (via _inject_memory())                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LLM Query → Response:                                                │    │
│  │   <memory_update>                                                    │    │
│  │   {"core": {"pkg_version": "1.0"}, "archival": [...]}               │    │
│  │   </memory_update>                                                   │    │
│  │   {"command": "pip install numpy", "done": false, "notes": "..."}   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Memory Processing (if memory.enabled):                               │    │
│  │   1. extract_memory_updates() - parse <memory_update> block         │    │
│  │   2. apply_llm_memory_updates() - execute write/read operations     │    │
│  │   3. If read results exist:                                          │    │
│  │      - _run_memory_update_phase() - re-query loop                   │    │
│  │      - up to max_memory_read_rounds iterations                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ parse_phase1_iterative_response() → {command, done, notes}          │    │
│  │                                                                      │    │
│  │ if done == true:                                                     │    │
│  │     break  # Exit loop                                               │    │
│  │ else:                                                                │    │
│  │     Execute command in container                                     │    │
│  │     Record result to history                                         │    │
│  │     continue  # Next iteration                                       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**LLM-Driven Memory Operations (per iteration)**:

| Operation | Example | Purpose |
|-----------|---------|---------|
| Write Core | `"core": {"numpy_version": "1.24.0"}` | Track installed versions |
| Write Archival | `"archival": [{"text": "...", "tags": ["PHASE1_INSTALL"]}]` | Log installation details |
| Read Core | `"core_get": ["python_deps_path"]` | Check previous state |
| Search Archival | `"archival_search": {"query": "PHASE1_ERROR", "k": 3}` | Find error recovery strategies |

#### Batch Mode (Legacy)

**Function**: `run_commands_with_logging()`

**Commands from**: `phase_artifacts["download"]["commands"]`

**Memory Events**:
```python
# On completion - uses dict format for mem_recall_append
mem_recall_append({
    "ts": time.time(),
    "run_id": memory_manager.run_id,
    "node_id": node.id,
    "branch_id": node.branch_id,
    "phase": stage_name,
    "kind": "phase1_complete",  # or "phase1_failed"
    "summary": "Phase 1 download/install complete for node ...",
    "refs": [],
})
```

### Phase 2: Coding

**Function**: `apply_workspace_plan()`

**Files from**: `phase_artifacts["coding"]["workspace"]["files"]`

**Memory Events**:
```python
# On completion
mem_recall_append({
    "ts": time.time(),
    "run_id": memory_manager.run_id,
    "node_id": node.id,
    "branch_id": node.branch_id,
    "phase": stage_name,
    "kind": "coding_complete",  # or "coding_failed"
    "summary": f"Coding phase complete for node {node.id}\nFiles: {files_list}",
    "refs": [],
})
```

### Phase 3: Compile

**Function**: `run_commands_with_logging()`

**Commands from**: `phase_artifacts["compile"]["commands"]`

**Memory Events**:
```python
# On completion
mem_recall_append({
    "ts": time.time(),
    "run_id": memory_manager.run_id,
    "node_id": node.id,
    "branch_id": node.branch_id,
    "phase": stage_name,
    "kind": "compile_complete",  # or "compile_failed"
    "summary": f"Compile phase complete for node {node.id}\nCompiler: {selected_compiler}",
    "refs": [],
})

# On failure - detailed archival entry is also written
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
    "ts": time.time(),
    "run_id": memory_manager.run_id,
    "node_id": node.id,
    "branch_id": node.branch_id,
    "phase": stage_name,
    "kind": "run_complete",  # or "run_failed"
    "summary": f"Run phase complete for node {node.id}\nOutputs: {expected_outputs[:3]}",
    "refs": [],
})
```

## Available Memory Functions

The memory system is organized in three layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     System-Level Functions                               │
│  apply_llm_memory_updates(), render_for_prompt()                        │
│  Log: memory_system.jsonl                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                     Public API (mem_* functions)                         │
│  mem_core_set(), mem_recall_append(), mem_archival_write(), etc.        │
│  Log: memory_public_api.jsonl                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                     Primitive Functions (DB operations)                  │
│  set_core(), write_event(), _insert_archival(), retrieve_archival()     │
│  Log: memory_primitive.jsonl                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Note**: All operations are also logged to `memory_calls.jsonl` with a `layer` field.
Memory operations during phase execution are additionally logged to phase_log files:
- `prompt_logs/<session>/memory_operations.jsonl` - operations with timestamps
- `prompt_logs/<session>/memory_injections.jsonl` - injection details

---

### Layer 1: Primitive Functions (Low-Level DB Operations)

These functions directly interact with the SQLite database. They are called by Public API functions.

| Function | Signature | Table | Log Op |
|----------|-----------|-------|--------|
| `set_core` | `(branch_id, key, value, *, ttl, importance, op_name) -> None` | `core_kv`, `core_meta` | `llm_memory_update` |
| `get_core` | `(branch_id, key, *, log_event, op_name, phase, node_id) -> str \| None` | `core_kv` | (none) |
| `write_event` | `(branch_id, kind, text, tags, *, log_event, ...) -> bool` | `events` | (none) |
| `_fetch_events` | `(branch_ids, limit, current_branch_id) -> list[Row]` | `events` | (none) |
| `_insert_archival` | `(branch_id, text, tags, *, log_event, op_name, node_id) -> int` | `archival` | (none) |
| `retrieve_archival` | `(branch_id, query, k, include_ancestors, tags_filter, ...) -> list[dict]` | `archival`, `archival_fts` | `retrieve_archival` |
| `create_branch` | `(parent_branch_id, node_uid, branch_id) -> str` | `branches` | (none) |

**Example Log Entry (Primitive)**:
```json
{"ts": 1769570057.97, "op": "llm_memory_update", "memory_type": "core", "branch_id": "...", "details": {"key": "phase0_summary", "value_preview": "...", "importance": 4}}
```

---

### Layer 2: Public API Functions (mem_* functions)

These are the main interface for memory operations. They call Primitive functions internally.

#### Core Memory

| Function | Signature | Calls | Log Op |
|----------|-----------|-------|--------|
| `mem_core_get` | `(keys: list[str] \| None) -> dict[str, str]` | `get_core()` | (none) |
| `mem_core_set` | `(key, value, *, ttl, importance, branch_id) -> None` | `set_core()` | (via set_core) |
| `mem_core_del` | `(key: str) -> None` | DELETE SQL | (none) |

#### Recall Memory

| Function | Signature | Calls | Log Op |
|----------|-----------|-------|--------|
| `mem_recall_append` | `(event: dict) -> None` | `write_event()` | `mem_recall_append` |
| `mem_recall_search` | `(query, *, k=20) -> list[dict]` | `_fetch_events()` | (none) |

**Event dict format for `mem_recall_append`**:
```python
{
    "ts": float,           # Timestamp
    "run_id": str,         # Run identifier
    "node_id": str,        # Node identifier
    "branch_id": str,      # Branch identifier
    "phase": str,          # Phase name (e.g., "draft", "debug")
    "kind": str,           # Event kind (e.g., "phase1_complete", "coding_failed")
    "summary": str,        # Event summary text
    "refs": list[str],     # Optional references
}
```

**Example Log Entry**:
```json
{"ts": 1769570062.03, "op": "mem_recall_append", "memory_type": "recall", "branch_id": "...", "phase": "stage_execution_1", "details": {"kind": "memory_injected", "summary_preview": "..."}}
```

#### Archival Memory

| Function | Signature | Calls | Log Op |
|----------|-----------|-------|--------|
| `mem_archival_write` | `(text, *, tags, meta) -> str` | `_insert_archival()` | `mem_archival_write` |
| `mem_archival_update` | `(record_id, *, text, tags, meta) -> None` | UPDATE SQL | (none) |
| `mem_archival_search` | `(query, *, tags, k=10) -> list[dict]` | `retrieve_archival()` | (via retrieve_archival) |
| `mem_archival_get` | `(record_id: str) -> dict` | SELECT SQL | (none) |

**Example Log Entry**:
```json
{"ts": 1769570057.97, "op": "mem_archival_write", "memory_type": "archival", "branch_id": "...", "details": {"record_id": 1, "tags": ["LLM_INSIGHT"], "text_preview": "..."}}
```

#### Branch/Node Operations

| Function | Signature | Calls | Log Op |
|----------|-----------|-------|--------|
| `mem_node_fork` | `(parent_node_id, child_node_id, ancestor_chain, phase) -> None` | `create_branch()` | `mem_node_fork` |
| `mem_node_read` | `(node_id, scope="all") -> dict` | multiple get functions | (none) |
| `mem_node_write` | `(node_id, *, core_updates, recall_event, archival_records) -> None` | multiple set functions | (none) |

**Example Log Entry**:
```json
{"ts": 1769569767.18, "op": "mem_node_fork", "memory_type": "node", "branch_id": "...", "details": {"parent_node_id": null, "child_branch_id": "..."}}
```

---

### Layer 3: System-Level Functions (High-Level Integration)

These functions orchestrate multiple Public API calls for complex operations.

| Function | Signature | Purpose | Log Op |
|----------|-----------|---------|--------|
| `apply_llm_memory_updates` | `(branch_id, updates, node_id, phase) -> dict` | Process LLM memory update block | `apply_llm_memory_updates` |
| `render_for_prompt` | `(branch_id, task_hint, budget_chars, no_limit) -> str` | Render memory for prompt injection | `render_for_prompt` |
| `render_for_prompt_with_log` | `(branch_id, task_hint, budget_chars, no_limit) -> tuple[str, dict]` | Render with detailed log | `render_for_prompt` |

**Example Log Entry (apply_llm_memory_updates)**:
```json
{
  "ts": 1769570057.97,
  "op": "apply_llm_memory_updates",
  "memory_type": "llm_update",
  "branch_id": "...",
  "details": {
    "core_keys": ["phase0_summary"],
    "archival_count": 1,
    "has_archival_search": false,
    "has_recall_search": false,
    "operations_log": [
      {"type": "core_set", "key": "phase0_summary", "value": "..."},
      {"type": "archival_write", "text": "...", "tags": ["LLM_INSIGHT"]}
    ]
  }
}
```

---

### Automatic/Internal Operations

These operations are triggered automatically by the system:

| Log Op | Trigger | Description |
|--------|---------|-------------|
| `check_memory_pressure` | Automatic | Monitor memory usage |
| `consolidate_inherited_memory` | Automatic | Consolidate inherited events (CoW) |
| `auto_consolidate_memory` | Automatic | Full memory consolidation |
| `evaluate_importance_with_llm` | Automatic | LLM-based importance scoring |
| `vlm_analysis_complete` | After VLM | VLM analysis completion marker |

---

### LLM Memory Update Keys

The `updates` dict for `apply_llm_memory_updates` supports:

| Key | Type | Internal Call | Description |
|-----|------|---------------|-------------|
| `core` | `dict[str, str]` | `mem_core_set()` | Key-value pairs to set |
| `core_get` | `list[str]` | `mem_core_get()` | Keys to retrieve (returns in result) |
| `core_delete` | `list[str]` | `mem_core_del()` | Keys to delete |
| `archival` | `list[dict]` | `mem_archival_write()` | Records to write |
| `archival_update` | `list[dict]` | `mem_archival_update()` | Records to update |
| `archival_search` | `dict` | `mem_archival_search()` | Search query (returns in result) |
| `recall` | `dict` | `mem_recall_append()` | Event to append |
| `recall_search` | `dict` | `mem_recall_search()` | Search query (returns in result) |
| `recall_evict` | `dict` | internal | Eviction params |
| `recall_summarize` | `bool` | internal | Trigger consolidation |
| `consolidate` | `bool` | internal | Trigger full consolidation |

---

### Actual Usage Statistics (from memory_calls.jsonl)

| Log Op | Count | Layer |
|--------|-------|-------|
| `llm_memory_update` | 510 | Primitive (via set_core) |
| `mem_archival_write` | 331 | Public API |
| `mem_recall_append` | 314 | Public API |
| `apply_llm_memory_updates` | 281 | System-Level |
| `render_for_prompt` | 26 | System-Level |
| `mem_node_fork` | 21 | Public API |
| `retrieve_archival` | 12 | Primitive |
| `vlm_analysis_complete` | 10 | Internal |
| `check_memory_pressure` | 8 | Internal |
| `consolidate_inherited_memory` | 6 | Internal |

**Note**: `mem_archival_search`, `mem_recall_search`, `mem_core_get` are called internally by `apply_llm_memory_updates` and `render_for_prompt`, but do not generate separate log entries

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

## Branch Fork Implementation

**Function**: `mem_node_fork()` in `memgpt_store.py`

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
        phase: Optional phase name for the node fork operation
    """
```

**Notes**:
- Returns `None` (does not return a branch ID)
- Automatically creates missing ancestor branches if `ancestor_chain` is provided
- Logs the fork operation to memory event log

## Files Generated Per Node

```
experiments/<run>/
├── logs/<index>-<exp_name>/
│   └── phase_logs/
│       └── node_<id>/
│           ├── download.log    # Phase 1 output
│           ├── compile.log     # Phase 3 output
│           ├── run.log         # Phase 4 output
│           └── artifacts.json  # Full phase_artifacts
├── logs/<index>-<exp_name>/
│   └── prompt_logs/
│       └── <session>/
│           ├── draft_attempt1_round0.json     # Full prompt with memory injection
│           ├── draft_attempt1_round0.md       # Rendered prompt
│           ├── memory_operations.jsonl        # Chronological memory operations
│           └── memory_injections.jsonl        # Chronological memory injections
└── memory/
    ├── memory_calls.jsonl      # All memory operations (main log)
    ├── memory_primitive.jsonl  # Layer 1: DB operations (set_core, get_core, etc.)
    ├── memory_public_api.jsonl # Layer 2: Public API (mem_* functions)
    ├── memory_system.jsonl     # Layer 3: System-level (apply_llm_memory_updates, render_for_prompt)
    └── memory_internal.jsonl   # Internal operations (check_memory_pressure, etc.)
```

### Log File Contents

**memory_calls.jsonl** (main log):
```json
{
  "ts": 1706123456.789,
  "op": "mem_recall_append",
  "layer": "public_api",
  "branch_id": "branch_abc123",
  "node_id": "node_0",
  "phase": "Draft/Code Generation",
  "details": {...}
}
```

**memory_operations.jsonl** (phase_log - operations with timestamps):
```json
{
  "timestamp": 1706123456.789,
  "stage": "1_creative_research_1_first_attempt",
  "label": "draft_attempt1",
  "counter": 1,
  "timing": {
    "start_timestamp": 1706123456.0,
    "end_timestamp": 1706123456.789,
    "duration_seconds": 0.789
  },
  "operations_count": 3,
  "operations": [
    {"type": "core_set", "timestamp": 1706123456.1, "key": "best_params", "value": "..."},
    {"type": "archival_write", "timestamp": 1706123456.3, "text": "...", "tags": ["..."]}
  ]
}
```

**memory_injections.jsonl** (phase_log - injection details):
```json
{
  "timestamp": 1706123456.789,
  "stage": "1_creative_research_1_first_attempt",
  "label": "draft_attempt1",
  "task_hint": "draft",
  "budget_chars": 24000,
  "rendered_chars": 15432,
  "core_count": 5,
  "recall_count": 12,
  "archival_count": 8,
  "timing": {
    "start_timestamp": 1706123456.0,
    "end_timestamp": 1706123456.1,
    "duration_seconds": 0.1
  }
}
```

## Database Schema (Actual Implementation)

The memory system uses SQLite with the following tables:

```sql
-- Branch hierarchy
CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    parent_id TEXT NULL,
    node_uid TEXT NULL,
    created_at REAL
);

-- Core memory (key-value pairs)
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
    ttl TEXT,  -- Note: TEXT type, not INTEGER
    updated_at REAL,
    PRIMARY KEY (branch_id, key)
);

-- Recall memory (events)
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

-- Archival memory
CREATE TABLE IF NOT EXISTS archival (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT,
    text TEXT,
    tags TEXT,
    created_at REAL
);

-- FTS5 full-text search (if available)
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

## See Also

- [memory-flow.md](memory-flow.md) - Overview
- [memory-flow-phase0.md](memory-flow-phase0.md) - Phase 0 flow
- [memory-flow-post-execution.md](memory-flow-post-execution.md) - Post-execution flow
