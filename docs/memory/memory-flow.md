# Memory Flow in Node Execution

This document describes the complete memory flow during experiment execution
in the AI-Scientist system with MemGPT-style memory enabled.

For detailed phase-by-phase documentation, see:
- [Phase 0 Flow](memory-flow-phase0.md) - Setup and Planning
- [Phase 1-4 Flow](memory-flow-phases.md) - Execution phases (Download/Coding/Compile/Run)
- [Post-Execution Flow](memory-flow-post-execution.md) - Metrics, Plotting, VLM Analysis, Summary

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT EXECUTION OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    PRE-FORK SETUP (on root branch)                       │    │
│  │  • Resource indexing (resources.yaml → archival + RESOURCE_INDEX core)  │    │
│  │  • Phase 0 prompt preparation (NOT executed yet)                        │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │                                              │
│                                   ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    TREE SEARCH LOOP (Stages 1-4)                         │    │
│  │  For each node:                                                          │    │
│  │    ┌──────────────────────────────────────────────────────────────────┐ │    │
│  │    │ 1. Fork branch from parent → child_branch_id                     │ │    │
│  │    │ 2. PHASE 0: Setup (executed per-node on child branch)            │ │    │
│  │    │    • Idea loading (idea.md → prompt injection)                   │ │    │
│  │    │    • LLM planning (generates phase0_plan.json)                   │ │    │
│  │    │    • Memory writes: idea_md_summary, phase0_summary → Core       │ │    │
│  │    │                     PHASE0_INTERNAL, IDEA_MD → Archival          │ │    │
│  │    │ 3. PHASE 1-4 execution (split-phase mode)                        │ │    │
│  │    │ 4. Metrics extraction                                            │ │    │
│  │    │ 5. Plotting code generation                                      │ │    │
│  │    │ 6. VLM analysis (if plots exist)                                 │ │    │
│  │    │ 7. Node summary generation                                       │ │    │
│  │    └──────────────────────────────────────────────────────────────────┘ │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │                                              │
│                                   ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      STAGE COMPLETION                                    │    │
│  │  • Multi-seed evaluation                                                │    │
│  │  • Plot aggregation                                                     │    │
│  │  • Stage summary → Memory                                               │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │                                              │
│                                   ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        FINAL MEMORY EXPORT                               │    │
│  │  • final_memory_for_paper.md                                            │    │
│  │  • final_memory_for_paper.json                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Memory Injection Points Summary

**Note**: Phase 0 is executed **per-node after fork**. Each child node runs its own Phase 0 planning, and memory events are recorded on the child branch (not root).

| Phase | Task Hint | Function | Memory Budget | Notes |
|-------|-----------|----------|---------------|-------|
| **Phase 0** | `resource_index` | - | - | Writes to root branch (pre-fork) |
| **Phase 0** | `phase0_planning` | `render_for_prompt()` | `memory_budget_chars` | Per-node after fork, memory injected |
| **Phase 1** | `phase1_iterative` | `_inject_memory()` | `memory_budget_chars` | Iterative download/install |
| **Phase 2** | `draft` | `_inject_memory()` | `memory_budget_chars` | Draft implementation |
| **Phase 2** | `debug` | `_inject_memory()` | `memory_budget_chars` | Debug buggy code |
| **Phase 2** | `improve` | `_inject_memory()` | `memory_budget_chars` | Improve working code |
| **Phase 2** | `hyperparam_node` | `_inject_memory()` | `memory_budget_chars` | Stage 2 hyperparameter tuning |
| **Phase 2** | `ablation_node` | `_inject_memory()` | `memory_budget_chars` | Stage 4 ablation study |
| **Phase 4** | `execution_review` | `_inject_memory()` | `memory_budget_chars` | Review execution results **(Two-Phase)** |
| **Post** | `metrics_extraction` | `_inject_memory()` | `metrics_extraction_budget_chars` | Extract metrics from output **(Two-Phase)** |
| **Post** | `plotting_code` | `_inject_memory()` | `plotting_code_budget_chars` | Generate plotting code |
| **Post** | `plot_selection` | `_inject_memory()` | `plot_selection_budget_chars` | Select plots for analysis **(Two-Phase)** |
| **Post** | `vlm_analysis` | `_inject_memory()` | `vlm_analysis_budget_chars` | VLM analyzes plots **(Two-Phase)** |
| **Post** | `node_summary` | `_inject_memory()` | `node_summary_budget_chars` | Generate node summary **(Two-Phase)** |
| **Agent** | `substage_goals` | `_inject_memory_into_text()` | `memory_budget_chars` | AgentManager goals |
| **Agent** | `stage_completion` | `_inject_memory_into_text()` | `memory_budget_chars` | Check completion |
| **Agent** | `define_metrics` | `_memory_context()` | `memory_budget_chars` | Define global metrics |

**(Two-Phase)**: Uses `_run_memory_update_phase()` for multi-round memory operations before `func_spec` task execution.

## Memory Context Structure

When `_inject_memory()` is called, it adds a "Memory" section to the prompt containing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Context Structure                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ## Core Memory (always visible)                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ idea_md_summary: "Research on optimizing matrix mult..."  │  │
│  │ phase0_summary: "Using OpenMP with 8 threads..."          │  │
│  │ RESOURCE_INDEX: {resource digest and paths}               │  │
│  │ [LLM-set keys]: optimal_threads: "8", best_flags: "-O3"   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ## Recall Memory (recent events)                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ [node_created] Node 5 started with plan...                │  │
│  │ [compile_failed] Error: undefined reference...            │  │
│  │ [compile_complete] Build succeeded with -fopenmp          │  │
│  │ [metrics_extracted] Speedup: 2.3x                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ## Archival Memory (semantic search results)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Query: {task_hint} + recent context                        │  │
│  │ Top-K results:                                             │  │
│  │   - [PERFORMANCE] 8 threads optimal for this workload     │  │
│  │   - [ERROR] Missing -fopenmp flag causes link errors      │  │
│  │   - [LLM_INSIGHT] Dynamic scheduling better for unbalanced│  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ## Memory Operations (Instructions for LLM)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ (split-phase mode with memory enabled)                     │  │
│  │ "You MUST include a <memory_update> block..."             │  │
│  │                                                            │  │
│  │ (other modes)                                              │  │
│  │ "You can manage your memory by including..."              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1 Iterative Memory Flow

Phase 1 (download/install) is unique because it runs as a multi-iteration loop with
command execution between LLM calls. Each iteration can have its own memory operations.

**Prompt Selection:**

```python
# parallel_agent.py (Phase 1 prompt selection)
phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT
if memory_cfg and getattr(memory_cfg, "enabled", False):
    phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT_WITH_MEMORY
```

| `memory.enabled` | Prompt File |
|------------------|-------------|
| `true` (default) | `prompt/config/phases/phase1_installer_with_memory.txt` |
| `false` | `prompt/config/phases/phase1_installer.txt` |

**Iteration Flow with Memory:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 ITERATIVE FLOW (memory.enabled=true)                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Iteration 1                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Build prompt: Introduction + Task + Phase0 guidance + History + Memory  │ │
│  │ 2. LLM Query → <memory_update> + JSON {command, done, notes}               │ │
│  │ 3. Apply memory updates (writes immediately, reads trigger re-query)       │ │
│  │ 4. Parse JSON → Execute command in container                               │ │
│  │ 5. Record result to history                                                │ │
│  │ 6. Check done flag → false, continue                                       │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                   │
│                              ▼                                                   │
│  Iteration 2                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ History now includes: step 1 command, exit_code, stdout/stderr summary     │ │
│  │ LLM sees previous results and decides next action                          │ │
│  │ May use "archival_search" to recall error recovery from earlier runs       │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                   │
│                              ▼                                                   │
│  ... (up to phase1_max_steps iterations)                                         │
│                              │                                                   │
│                              ▼                                                   │
│  Final Iteration (done=true)                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ LLM returns: {"command": "", "done": true, "notes": "All deps installed"}  │ │
│  │ Memory update: {"core": {"phase1_status": "completed"}, ...}               │ │
│  │ → Exit Phase 1 loop                                                        │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Memory Operations in Phase 1:**

| Operation | Purpose | Example |
|-----------|---------|---------|
| **Write to Core** | Track installation state | `"core": {"numpy_version": "1.24.0"}` |
| **Write to Archival** | Log details for future reference | `"archival": [{"text": "...", "tags": ["PHASE1_INSTALL"]}]` |
| **Read from Core** | Check previous installation state | `"core_get": ["python_deps_path"]` |
| **Search Archival** | Find error recovery strategies | `"archival_search": {"query": "PHASE1_ERROR", "k": 3}` |

**Read Operation Re-Query (within each iteration):**

```
LLM Response: <memory_update>{"archival_search": {"query": "...", "k": 3}}</memory_update>
                          {"command": "pip install numpy", "done": false}
        │
        ▼
apply_llm_memory_updates() → memory_results with read data
        │
        ▼
_has_memory_read_results(memory_results) == True?
        │
   ┌────┴────┐
  Yes       No
   │         │
   ▼         ▼
Re-query   Parse JSON
  Loop     & Execute
   │
   ▼
_run_memory_update_phase() with max_rounds from cfg.memory.max_memory_read_rounds
   │
   ▼
After read loop completes → Continue to JSON parse & command execution
```

**Code Location (parallel_agent.py):**

```python
# Apply memory updates from Phase 1 response if present
if response_text and worker_agent.memory_manager and child_branch_id:
    memory_updates = extract_memory_updates(response_text)
    if memory_updates:
        memory_results = worker_agent.memory_manager.apply_llm_memory_updates(...)

        # Handle memory read operations with re-query loop
        if _has_memory_read_results(memory_results):
            max_rounds = getattr(memory_cfg, "max_memory_read_rounds", 5)
            if max_rounds > 0:
                _run_memory_update_phase(...)  # Re-query for read results
```

## Multi-Turn Memory Read Flow

When memory is enabled and LLM uses read operations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Turn Memory Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Round 0 (Initial Query):                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Prompt:                                                    │  │
│  │   - Task description                                       │  │
│  │   - Memory context (Core + Recall + Archival)             │  │
│  │   - Memory Operations instructions                         │  │
│  │                                                            │  │
│  │ LLM Response:                                              │  │
│  │   <memory_update>                                          │  │
│  │   {                                                        │  │
│  │     "archival_search": {"query": "optimal config"}  ← READ │  │
│  │     "core": {"new_key": "value"}                    ← WRITE│  │
│  │   }                                                        │  │
│  │   </memory_update>                                         │  │
│  │   {"phase_artifacts": {...}}                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ System executes:                                           │  │
│  │   1. apply_llm_memory_updates()                            │  │
│  │   2. Write operations persisted                            │  │
│  │   3. Read operations return results                        │  │
│  │   4. _has_memory_read_results() == True                    │  │
│  │   5. Round (0) < max_memory_read_rounds (5)                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  Round 1 (Re-Query with Results):                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Prompt += "Memory Read Results":                           │  │
│  │   <memory_results>                                         │  │
│  │   **Archival Search Results:**                             │  │
│  │     [1] Thread count 8 optimal for matrix workload         │  │
│  │         Tags: PERFORMANCE                                   │  │
│  │   </memory_results>                                        │  │
│  │                                                            │  │
│  │ LLM Response (final):                                      │  │
│  │   <memory_update>                                          │  │
│  │   {"core": {"optimal_threads": "8"}}  ← Uses search result │  │
│  │   </memory_update>                                         │  │
│  │   {"phase_artifacts": {...}}           ← Final output      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Two-Phase Pattern for Structured Responses

Some phases use `func_spec` (OpenAI function calling) to get structured JSON responses.
Since `func_spec` responses cannot include `<memory_update>` blocks, these phases use
a **two-phase pattern**:

```
┌─────────────────────────────────────────────────────────────────┐
│              Two-Phase Pattern for func_spec Methods            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: Memory Update (Free Text Response)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Function: _run_memory_update_phase()                      │  │
│  │                                                            │  │
│  │ Round 0:                                                   │  │
│  │   Prompt with task description + Memory context           │  │
│  │   LLM returns: <memory_update>{...}</memory_update>       │  │
│  │   → apply_llm_memory_updates()                            │  │
│  │   → Check for read results                                │  │
│  │                                                            │  │
│  │ Round 1+ (if read results and round < max_rounds):        │  │
│  │   Inject read results into prompt                         │  │
│  │   Re-query LLM for additional updates                     │  │
│  │   → Repeat until no more reads or max rounds reached      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  PHASE 2: Task Execution (Structured Response via func_spec)    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ LLM query with func_spec                                  │  │
│  │ Returns: Structured JSON (dict)                           │  │
│  │ No memory_update blocks in response                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Methods using Two-Phase Pattern**:

| Method | Phase 1 Task | Phase 2 Output |
|--------|--------------|----------------|
| `execution_review` | Review output, record insights | `{"should_retry": bool, "reason": str}` |
| `plot_selection` | Analyze available plots | `{"selected_plots": [...]}` |
| `vlm_analysis` | Prepare for visual analysis, search related memories | `{"plot_analyses": [...], "vlm_feedback_summary": str}` |
| `node_summary` | Summarize node findings | `{"summary": str, "key_findings": [...]}` |
| `metrics_extraction` | Record metric patterns | `{"valid_metrics_received": bool, "metric_names": {...}}` |

**Helper Function**: `_run_memory_update_phase()`

```python
def _run_memory_update_phase(
    prompt: dict,
    memory_manager: Any,
    branch_id: str,
    node_id: str | None,
    phase_name: str,
    model: str,
    temperature: float,
    max_rounds: int = 2,           # From cfg.memory.max_memory_read_rounds
    task_description: str = "",
) -> None:
    """Run multi-round memory update phase before task execution."""
```

## Configuration

```yaml
memory:
  enabled: true

  # Memory budgets
  memory_budget_chars: 4000           # Default for most phases
  metrics_extraction_budget_chars: 1500
  plotting_code_budget_chars: 2000
  vlm_analysis_budget_chars: 1000
  node_summary_budget_chars: 2000

  # Multi-turn flow
  max_memory_read_rounds: 5           # Max re-query cycles for read operations

  # Memory limits
  core_max_chars: 2000
  recall_max_events: 5
  retrieval_k: 4                      # Top-K archival hits
```

## Files Involved

| File | Role |
|------|------|
| `parallel_agent.py` | Main execution flow, `_inject_memory()`, `generate_phase_artifacts()` |
| `agent_manager.py` | Stage management, `_inject_memory_into_text()` |
| `memgpt_store.py` | Memory operations, `render_for_prompt()`, `apply_llm_memory_updates()` |
| `phase_plan.py` | `<memory_update>` extraction, `MissingMemoryUpdateError` |
| `config.py` | `MemoryConfig` dataclass |

## Logging

All memory operations are logged to:
- `experiments/<run>/memory/memory_calls.jsonl` - Detailed operation log
- `experiments/<run>/memory/memory.sqlite` - Persistent memory database

## See Also

- [memory.md](memory.md) - General memory system documentation
- [memory-flow-phase0.md](memory-flow-phase0.md) - Phase 0 detailed flow
- [memory-flow-phases.md](memory-flow-phases.md) - Phase 1-4 detailed flow
- [memory-flow-post-execution.md](memory-flow-post-execution.md) - Post-execution flow
