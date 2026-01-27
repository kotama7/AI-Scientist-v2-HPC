# Standard Execution Flow (Memory Disabled)

This document describes the execution flow when `memory.enabled=false`.
Note: Memory is **enabled by default** (`memory.enabled=true`). For the
memory-enabled flow (default), see [../memory/memory-flow.md](../memory/memory-flow.md).

## Overview

Without memory, the execution flow is simpler:
- No memory injection into prompts
- No `<memory_update>` blocks required in LLM responses
- No multi-turn memory read cycles
- Standard JSON-only response format

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD EXECUTION FLOW (No Memory)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Phase 0   │───▶│   Phase 1   │───▶│   Phase 2   │───▶│   Phase 3   │  │
│  │  Planning   │    │   Install   │    │   Coding    │    │   Compile   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                    │         │
│                                               ┌────────────────────┘         │
│                                               ▼                              │
│                                        ┌─────────────┐                       │
│                                        │   Phase 4   │                       │
│                                        │   Execute   │                       │
│                                        └─────────────┘                       │
│                                               │                              │
│                        ┌──────────────────────┼──────────────────────┐       │
│                        ▼                      ▼                      ▼       │
│                 ┌───────────┐          ┌───────────┐          ┌───────────┐ │
│                 │  Metrics  │          │  Plotting │          │    VLM    │ │
│                 │Extraction │          │   Code    │          │ Analysis  │ │
│                 └───────────┘          └───────────┘          └───────────┘ │
│                        │                      │                      │       │
│                        └──────────────────────┼──────────────────────┘       │
│                                               ▼                              │
│                                        ┌─────────────┐                       │
│                                        │    Node     │                       │
│                                        │   Summary   │                       │
│                                        └─────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 0: Planning

**Function**: `_process_node_wrapper()` (within each worker process)

**Location**: `parallel_agent.py`

**Execution**: Once per node (same as Phase 1-4)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 0 FLOW (Per Node)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Node execution starts (_process_node_wrapper)                               │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. COLLECT HISTORY (_collect_phase0_history)                         │    │
│  │                                                                       │    │
│  │    From previous runs:                                                │    │
│  │    ├── Phase summaries (phase0-4 summaries)                           │    │
│  │    ├── Compile/run logs                                               │    │
│  │    ├── Error extracts                                                 │    │
│  │    └── LLM output summaries                                           │    │
│  │                                                                       │    │
│  │    → Saved to phase0_history_full.json                                │    │
│  │    → Formatted via history_injection template                         │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. BUILD PROMPT                                                      │    │
│  │                                                                       │    │
│  │    phase0_prompt = {                                                  │    │
│  │        "Introduction": PHASE0_WHOLE_PLANNING_PROMPT,                  │    │
│  │        "Task": task_desc,                                             │    │
│  │        "History": history_injection,  # from step 1                   │    │
│  │        "Environment": {                                               │    │
│  │            os_release, cpu_info, memory_info,                         │    │
│  │            gpu_info, available_compilers, available_libs,             │    │
│  │            system_performance_tools, network_access, ...              │    │
│  │        },                                                             │    │
│  │        "Resources": resources_context,  # optional                    │    │
│  │    }                                                                  │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. QUERY LLM                                                         │    │
│  │                                                                       │    │
│  │    phase0_response = query(phase0_prompt)                             │    │
│  │    phase0_plan = _normalize_phase0_plan(parse_json(response))         │    │
│  │                                                                       │    │
│  │    → Saved to phase0_plan.json (for logging)                          │    │
│  │    → Saved to phase0_llm_output.txt (raw response)                    │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. PREPARE FOR NODE EXECUTION                                        │    │
│  │                                                                       │    │
│  │    phase0_plan is passed to MinimalAgent                              │    │
│  │    ├── Used in _phase0_plan_snippet() for subsequent prompts          │    │
│  │    └── Provides phase_guidance for Phase 1-4                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prompt Structure (Phase 0)

```python
phase0_prompt = {
    "Introduction": PHASE0_WHOLE_PLANNING_PROMPT,
    "Task": task_desc,
    "History": history_injection,  # Previous run results
    "Environment": {
        "os_release": "...",
        "cpu_info": "...",
        "memory_info": "...",
        "gpu_info": "...",
        "available_compilers": [...],
        "available_libs": [...],
        "system_performance_tools": [...],
        "network_access": "yes/no/unknown",
        "timeout_seconds": 3600,
    },
    "Resources": resources_context,  # Optional
}
```

## Phase 1-4: Split-Phase Execution

**Function**: `_run_node()` → `generate_phase_artifacts()` → `_execute_split_phase()`

**Location**: `parallel_agent.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SPLIT-PHASE EXECUTION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Node Types: draft, debug, improve, hyperparam, ablation                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. BUILD PROMPT                                                      │    │
│  │                                                                       │    │
│  │    prompt = {                                                         │    │
│  │        "System": BASE_SYSTEM_PROMPT,                                  │    │
│  │        "Domain": DOMAIN_NEUTRAL_PROMPT,                               │    │
│  │        "Environment": env_info,                                       │    │
│  │        "Resources": resources_context,                                │    │
│  │        "Previous code": parent_node.code (if exists),                 │    │
│  │        "Previous error": parent_node.term_out (if debug),             │    │
│  │        "Journal summary": journal.generate_summary(),                 │    │
│  │        "Instructions": {                                              │    │
│  │            "Task": node_specific_task,                                │    │
│  │            "Response format": RESPONSE_FORMAT_SPLIT_PHASE,            │    │
│  │        },                                                             │    │
│  │    }                                                                  │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. QUERY LLM (generate_phase_artifacts)                              │    │
│  │                                                                       │    │
│  │    completion = query(prompt, model, temperature)                     │    │
│  │                        │                                              │    │
│  │                        ▼                                              │    │
│  │    artifacts = extract_phase_artifacts(completion)                    │    │
│  │    # Returns: {phase_artifacts, constraints}                          │    │
│  │    # NO memory_update processing                                      │    │
│  │                                                                       │    │
│  │    If parse error: retry up to 3 times                                │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. EXECUTE PHASES (_execute_split_phase)                             │    │
│  │                                                                       │    │
│  │    Phase 1: Download/Install                                          │    │
│  │    ├── Run download commands in container                             │    │
│  │    └── Install dependencies (pip, apt-get, source builds)            │    │
│  │                                                                       │    │
│  │    Phase 2: Write Code                                                │    │
│  │    ├── Apply workspace plan (create files)                            │    │
│  │    └── Write source files to workspace                                │    │
│  │                                                                       │    │
│  │    Phase 3: Compile                                                   │    │
│  │    ├── Run build commands                                             │    │
│  │    └── Check for compilation errors                                   │    │
│  │                                                                       │    │
│  │    Phase 4: Run                                                       │    │
│  │    ├── Execute program                                                │    │
│  │    ├── Capture stdout/stderr                                          │    │
│  │    └── Collect output files (.npy)                                    │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. RESULT                                                            │    │
│  │                                                                       │    │
│  │    node.code = generated_code                                         │    │
│  │    node.term_out = execution_output                                   │    │
│  │    node.stage = phase_reached                                         │    │
│  │    node.success = (phase_reached == "run" and no errors)              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Response Format (Standard)

When memory is disabled, the LLM uses `execution_split.txt` format:

```json
{
  "phase_artifacts": {
    "download": {"commands": [...], "notes": "..."},
    "coding": {
      "workspace": {
        "root": "/workspace",
        "tree": ["workspace/", "workspace/src/", ...],
        "files": [
          {"path": "src/main.c", "mode": "0644", "content": "..."}
        ]
      },
      "notes": "..."
    },
    "compile": {
      "build_plan": {
        "language": "c",
        "compiler_selected": "gcc",
        "cflags": ["-O3"],
        "ldflags": [],
        "output": "bin/a.out"
      },
      "commands": ["gcc ..."],
      "notes": "..."
    },
    "run": {
      "commands": ["./bin/a.out"],
      "expected_outputs": ["working/experiment_data.npy"],
      "notes": "..."
    }
  },
  "constraints": {
    "allow_sudo_in_singularity": true,
    "write_only_under_workspace": true,
    ...
  }
}
```

## Post-Execution Processing

**Location**: `parallel_agent.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       POST-EXECUTION FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  After Phase 4 completes (success or failure):                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. METRICS EXTRACTION (_extract_metrics)                             │    │
│  │                                                                       │    │
│  │    Input: term_out (execution output)                                 │    │
│  │    Prompt: Parse output for speedup, accuracy, etc.                   │    │
│  │    Output: node.metric = extracted_metrics                            │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. PLOTTING CODE GENERATION (_generate_plotting_code)                │    │
│  │                                                                       │    │
│  │    Input: experiment code, available .npy files                       │    │
│  │    Prompt: Generate visualization code                                │    │
│  │    Output: plotting.py → executed → .png files                        │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. VLM ANALYSIS (_analyze_plots_with_vlm)                            │    │
│  │                                                                       │    │
│  │    Condition: Only if .png files exist                                │    │
│  │    Input: Generated plot images (base64 encoded)                      │    │
│  │    Prompt: Analyze visualizations                                     │    │
│  │    Output: node.vlm_analysis = analysis_text                          │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. NODE SUMMARY (_generate_node_summary)                             │    │
│  │                                                                       │    │
│  │    Input: plan, code, term_out, metrics, vlm_analysis                 │    │
│  │    Prompt: Summarize experiment results                               │    │
│  │    Output: node.summary = summary_text                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tree Search Loop

**Function**: `step()` in `ParallelAgent`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TREE SEARCH LOOP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌──────────────┐                                    │
│                          │    Start     │                                    │
│                          └──────┬───────┘                                    │
│                                 │                                            │
│                                 ▼                                            │
│                    ┌────────────────────────┐                                │
│                    │  Select nodes to run   │                                │
│                    │  (_select_parallel_    │                                │
│                    │   nodes)               │                                │
│                    └───────────┬────────────┘                                │
│                                │                                             │
│           ┌────────────────────┼────────────────────┐                        │
│           │                    │                    │  (parallel workers)    │
│           ▼                    ▼                    ▼                        │
│     ┌──────────┐        ┌──────────┐        ┌──────────┐                    │
│     │ Worker 0 │        │ Worker 1 │        │ Worker 2 │                    │
│     │          │        │          │        │          │                    │
│     │ Phase 0  │        │ Phase 0  │        │ Phase 0  │  (per node)        │
│     │    ↓     │        │    ↓     │        │    ↓     │                    │
│     │ Phase1-4 │        │ Phase1-4 │        │ Phase1-4 │  (per node)        │
│     └────┬─────┘        └────┬─────┘        └────┬─────┘                    │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              │                                               │
│                              ▼                                               │
│                    ┌────────────────────┐                                    │
│                    │  Select Best Node  │                                    │
│                    │  (metric-based)    │                                    │
│                    └─────────┬──────────┘                                    │
│                              │                                               │
│                              ▼                                               │
│                    ┌────────────────────┐  No                                │
│                    │   More budget?     │──────▶ End                         │
│                    └─────────┬──────────┘                                    │
│                              │ Yes                                           │
│                              ▼                                               │
│                    ┌────────────────────┐                                    │
│                    │  Generate children │                                    │
│                    │  (expand tree)     │                                    │
│                    └─────────┬──────────┘                                    │
│                              │                                               │
│                              └────────────────▶ (loop)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Workspace Inheritance

**Location**: `parallel_agent.py`, `agent_manager.py`

Each node's workspace is persisted to a log directory to enable safe cross-stage
file inheritance without race conditions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WORKSPACE INHERITANCE FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. NODE PROCESSING COMPLETE                                                 │
│     │                                                                        │
│     ├── Copy workspace to: node_logs/node_<id>/                              │
│     │   (src/, bin/, input/, working/ directories)                           │
│     │                                                                        │
│     └── Set node.workspace_path = "node_logs/node_<id>/"                     │
│                                                                              │
│  2. CHILD NODE STARTS                                                        │
│     │                                                                        │
│     ├── Read parent_workspace_path from parent node                          │
│     │   (points to node_logs/node_<parent_id>/)                              │
│     │                                                                        │
│     └── Copy files from node_logs/ to worker workspace                       │
│         (Safe: node_logs/ is persistent, not overwritten)                    │
│                                                                              │
│  3. STAGE COMPLETE                                                           │
│     │                                                                        │
│     ├── Copy best node's workspace to: stage_best/<stage_name>/              │
│     │                                                                        │
│     ├── Update best_node.workspace_path = "stage_best/<stage_name>/"         │
│     │                                                                        │
│     └── Delete all node_logs/node_* directories (cleanup)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
workspace_dir/
├── worker_0/              # Temporary worker workspace (reused per task)
├── worker_1/
├── worker_2/
├── worker_3/
├── node_logs/             # Persistent node workspaces (during stage)
│   ├── node_<id_1>/       # ← node.workspace_path points here
│   │   ├── src/
│   │   ├── bin/
│   │   ├── input/
│   │   └── working/
│   ├── node_<id_2>/
│   └── ...                # Cleaned up after stage completion
└── stage_best/            # Best node workspace per stage (permanent)
    ├── 1_creative_research_1_first_attempt/
    ├── 2_hyperparam_tuning/
    ├── 3_plotting/
    └── 4_ablation/
```

### Why This Design?

Previously, `workspace_path` pointed to worker directories (e.g., `worker_3/`).
This caused race conditions when:

1. Worker 3 processes Node A → `workspace_path = "worker_3/"`
2. Node A's child (Node B) is assigned to Worker 1
3. Worker 1 tries to copy from `worker_3/`
4. But Worker 3 is now processing Node C, overwriting its workspace
5. Result: `[Errno 2] No such file or directory`

The new design uses persistent `node_logs/` directories that are not overwritten
by other workers, eliminating race conditions.

## Key Differences: Memory Disabled vs Enabled

| Aspect | Memory Disabled | Memory Enabled |
|--------|-----------------|----------------|
| Response format | `execution_split.txt` | `execution_split_with_memory.txt` |
| LLM response | JSON only | `<memory_update>` + JSON |
| Memory context | Not injected | Injected via `_inject_memory()` |
| Read operations | N/A | Multi-turn re-query |
| Branch isolation | N/A | Per-branch memory stores |
| Persistence | None | SQLite database |

## Configuration

When memory is disabled, these settings are ignored:

```yaml
memory:
  enabled: false  # Note: default is true (memory enabled)
  # Below settings have no effect when disabled:
  # db_path, budget_chars, max_memory_read_rounds, etc.
```

## See Also

- [execution-modes.md](../configuration/execution-modes.md) - Split vs single execution
- [llm-context.md](llm-context.md) - Prompt assembly details
- [../memory/memory-flow.md](../memory/memory-flow.md) - Memory-enabled flow
