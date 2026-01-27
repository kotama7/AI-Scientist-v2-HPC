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
│  │                         PHASE 0: SETUP                                   │    │
│  │  • Idea loading (idea.md → archival + core summary)                     │    │
│  │  • Resource indexing (resources.yaml → archival + RESOURCE_INDEX core)  │    │
│  │  • Whole planning (LLM generates phase 0 plan)                          │    │
│  │                                                                          │    │
│  │  Memory writes: idea_md_summary, phase0_summary → Core                  │    │
│  │                 PHASE0_INTERNAL, IDEA_MD → Archival                     │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │                                              │
│                                   ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    TREE SEARCH LOOP (Stages 1-4)                         │    │
│  │  For each node:                                                          │    │
│  │    ┌──────────────────────────────────────────────────────────────────┐ │    │
│  │    │ 1. Fork branch from parent                                       │ │    │
│  │    │ 2. PHASE 1-4 execution (split-phase mode)                        │ │    │
│  │    │ 3. Metrics extraction                                            │ │    │
│  │    │ 4. Plotting code generation                                      │ │    │
│  │    │ 5. VLM analysis (if plots exist)                                 │ │    │
│  │    │ 6. Node summary generation                                       │ │    │
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
│  │  • final_memory-for-paper.md                                            │    │
│  │  • final_memory_for_paper.json                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Memory Injection Points Summary

| Phase | Task Hint | Function | Memory Budget | Notes |
|-------|-----------|----------|---------------|-------|
| **Phase 0** | `idea_md` | - | - | Writes to memory only |
| **Phase 0** | `resource_index` | - | - | Writes to memory only |
| **Phase 0** | `phase0_planning` | `_inject_memory()` | `memory_budget_chars` | Planning prompt |
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
| **Post** | `vlm_analysis` | `_inject_memory()` | `vlm_analysis_budget_chars` | VLM analyzes plots |
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
│  │   5. Round (0) < max_memory_read_rounds (2)                │  │
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
  max_memory_read_rounds: 2           # Max re-query cycles for read operations

  # Memory limits
  core_max_chars: 2000
  recall_max_events: 20
  retrieval_k: 8                      # Top-K archival hits
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
