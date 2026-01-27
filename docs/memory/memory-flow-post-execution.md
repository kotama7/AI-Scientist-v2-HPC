# Post-Execution Flow - Memory Flow

This document describes the memory flow during post-execution phases:
Execution Review, Metrics Extraction, Plotting, Plot Selection, VLM Analysis, and Node Summary.

## Post-Execution Overview

After Phase 1-4 complete (whether success or failure), additional processing
steps extract insights and generate summaries that are stored in memory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     POST-EXECUTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1-4 Complete                                                         │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 0. EXECUTION REVIEW (on failure, before retry)                      │    │
│  │    Task hint: "execution_review"                                    │    │
│  │    Uses: Two-Phase Pattern (func_spec)                              │    │
│  │                                                                       │    │
│  │    Input: Terminal output, error messages                           │    │
│  │    Output: {"should_retry": bool, "reason": str}                    │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. METRICS EXTRACTION                                                │    │
│  │    Task hint: "metrics_extraction"                                   │    │
│  │    Uses: Two-Phase Pattern (func_spec)                               │    │
│  │    Budget: metrics_extraction_budget_chars                           │    │
│  │                                                                       │    │
│  │    Input: Execution output (stdout/stderr)                           │    │
│  │    Output: Structured metrics (speedup, accuracy, etc.)              │    │
│  │                                                                       │    │
│  │    Memory:                                                            │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ _inject_memory(prompt, "metrics_extraction", branch_id)      │   │    │
│  │    │                                                              │   │    │
│  │    │ On success:                                                  │   │    │
│  │    │   Recall: metrics_extracted {metrics_dict}                   │   │    │
│  │    │   Archival: [METRICS] detailed metrics with tags             │   │    │
│  │    │                                                              │   │    │
│  │    │ On failure:                                                  │   │    │
│  │    │   Recall: metrics_failed {error}                             │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. PLOTTING CODE GENERATION                                          │    │
│  │    Task hint: "plotting_code"                                        │    │
│  │    Budget: plotting_code_budget_chars                                │    │
│  │                                                                       │    │
│  │    Input: experiment code, available .npy files                      │    │
│  │    Output: Python code for visualization                             │    │
│  │                                                                       │    │
│  │    Memory:                                                            │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ _inject_memory(prompt, "plotting_code", branch_id)           │   │    │
│  │    │                                                              │   │    │
│  │    │ Note: Uses plan_and_code_query() not generate_phase_artifacts│   │    │
│  │    │ Memory update via extract_memory_updates() if present        │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                       │    │
│  │    Then: Execute plotting code → Generate plots                      │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2.5 PLOT SELECTION (if multiple plots exist)                        │    │
│  │    Task hint: "plot_selection"                                      │    │
│  │    Uses: Two-Phase Pattern (func_spec)                              │    │
│  │    Budget: plot_selection_budget_chars                              │    │
│  │                                                                       │    │
│  │    Input: List of available plot images                             │    │
│  │    Output: {"selected_plots": [...]}                                │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. VLM ANALYSIS (if plots exist)                                     │    │
│  │    Task hint: "vlm_analysis"                                         │    │
│  │    Budget: vlm_analysis_budget_chars                                 │    │
│  │                                                                       │    │
│  │    Input: Generated plot images                                      │    │
│  │    Output: Analysis text (insights, observations)                    │    │
│  │                                                                       │    │
│  │    Memory:                                                            │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ _inject_memory(prompt, "vlm_analysis", branch_id)            │   │    │
│  │    │                                                              │   │    │
│  │    │ Result stored:                                               │   │    │
│  │    │   node.vlm_analysis = analysis_text                          │   │    │
│  │    │   (Not directly written to memory, but available for summary)│   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. NODE SUMMARY GENERATION                                           │    │
│  │    Task hint: "node_summary"                                         │    │
│  │    Uses: Two-Phase Pattern (func_spec)                               │    │
│  │    Budget: node_summary_budget_chars                                 │    │
│  │                                                                       │    │
│  │    Input: Node's execution results, metrics, VLM analysis            │    │
│  │    Output: Comprehensive summary                                     │    │
│  │                                                                       │    │
│  │    Memory:                                                            │    │
│  │    ┌─────────────────────────────────────────────────────────────┐   │    │
│  │    │ _inject_memory(prompt, "node_summary", branch_id)            │   │    │
│  │    │                                                              │   │    │
│  │    │ On completion:                                               │   │    │
│  │    │   Recall: node_summary {summary_dict}                        │   │    │
│  │    │   Archival: [NODE_SUMMARY, node:<id>] full summary           │   │    │
│  │    └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Two-Phase Pattern

Several post-execution methods use `func_spec` (OpenAI function calling) for structured
responses. Since `func_spec` cannot include `<memory_update>` blocks, these methods use
the **two-phase pattern** with `_run_memory_update_phase()` helper:

**Phase 1**: Memory update phase (free text, multi-round capable)
**Phase 2**: Task execution with `func_spec` (structured JSON response)

See [memory-flow.md](memory-flow.md#two-phase-pattern-for-structured-responses) for details.

## Detailed Function Flows

### 0. Execution Review (`_query_execution_review`)

**Location**: `parallel_agent.py`

**Trigger**: After Phase 4 fails (before retry)

**Uses Two-Phase Pattern**: Yes (via `_run_memory_update_phase()`)

**Purpose**: Analyze execution failure and decide whether to retry

**Prompt Structure**:
```python
prompt = {
    "Introduction": EXECUTION_REVIEW_INTRO,
    "Terminal output": term_out[:term_out_limit],
    "Error details": exc_info,
}
_inject_memory(
    prompt,
    "execution_review",
    branch_id=node.branch_id,
    budget_chars=cfg.memory.execution_review_budget_chars
)
```

**Phase 1: Memory Update**:
```python
if self._is_memory_enabled and memory_manager and branch_id:
    _run_memory_update_phase(
        prompt=review_prompt,
        memory_manager=memory_manager,
        branch_id=branch_id,
        node_id=node.id,
        phase_name="execution_review",
        model=cfg.agent.feedback.model,
        temperature=cfg.agent.feedback.temp,
        max_rounds=getattr(cfg.memory, "max_memory_read_rounds", 2),
        task_description=(
            "Review the execution failure and update your memory with any important "
            "observations, error patterns, or debugging insights."
        ),
    )
```

**Phase 2: Task Execution (func_spec)**:
```python
review_response = query(
    system_message=review_prompt,
    user_message=None,
    func_spec=execution_review_spec,  # Structured response
    model=cfg.agent.feedback.model,
    temperature=cfg.agent.feedback.temp,
)
# Returns: {"should_retry": bool, "reason": str}
```

### 0.5 Plot Selection (`_query_plot_selection`)

**Location**: `parallel_agent.py`

**Trigger**: After plots are generated (if multiple plots exist)

**Uses Two-Phase Pattern**: Yes (via `_run_memory_update_phase()`)

**Purpose**: Select the most relevant plots for VLM analysis and paper

**Prompt Structure**:
```python
prompt = {
    "Introduction": PLOT_SELECTION_INTRO,
    "Available plots": plot_list,
    "Experiment context": node.plan,
}
_inject_memory(
    prompt,
    "plot_selection",
    branch_id=node.branch_id,
    budget_chars=cfg.memory.plot_selection_budget_chars
)
```

**Phase 1: Memory Update**:
```python
if self._is_memory_enabled and memory_manager and branch_id:
    _run_memory_update_phase(
        prompt=selection_prompt,
        memory_manager=memory_manager,
        branch_id=branch_id,
        node_id=node.id,
        phase_name="plot_selection",
        model=cfg.agent.feedback.model,
        temperature=cfg.agent.feedback.temp,
        max_rounds=getattr(cfg.memory, "max_memory_read_rounds", 2),
        task_description=(
            "Review the available plots and update your memory with observations about "
            "which visualizations best represent the experiment results."
        ),
    )
```

**Phase 2: Task Execution (func_spec)**:
```python
selection_response = query(
    system_message=selection_prompt,
    user_message=None,
    func_spec=plot_selection_spec,  # Structured response
    model=cfg.agent.feedback.model,
    temperature=cfg.agent.feedback.temp,
)
# Returns: {"selected_plots": [...]}
```

### 1. Metrics Extraction (`_extract_metrics`)

**Location**: `parallel_agent.py`

**Trigger**: After successful execution (Phase 4 complete)

**Uses Two-Phase Pattern**: Yes (via `_run_memory_update_phase()`)

**Prompt Structure**:
```python
prompt = {
    "Introduction": metrics_extraction_intro,
    "Execution output": term_out[:max_chars],
    "Evaluation metrics": self.evaluation_metrics,
    "Instructions": parse_metrics_instructions,
}
_inject_memory(
    prompt,
    "metrics_extraction",
    branch_id=node.branch_id,
    budget_chars=cfg.memory.metrics_extraction_budget_chars
)
```

**Phase 1: Memory Update**:
```python
if worker_agent._is_memory_enabled and memory_manager and metrics_branch_id:
    _run_memory_update_phase(
        prompt=metrics_prompt,
        memory_manager=memory_manager,
        branch_id=metrics_branch_id,
        node_id=child_node.id,
        phase_name="metrics_extraction",
        model=cfg.agent.feedback.model,
        temperature=cfg.agent.feedback.temp,
        max_rounds=cfg.memory.max_memory_read_rounds,  # Default: 2
        task_description=(
            "Review the metrics output and update your memory with any important "
            "observations, metric patterns, or performance insights."
        ),
    )
```

**Phase 2: Task Execution (func_spec)**:
```python
metrics_response = query(
    system_message=metrics_prompt,
    user_message=None,
    func_spec=metric_parse_spec,  # Structured response
    model=cfg.agent.feedback.model,
    temperature=cfg.agent.feedback.temp,
)
# Returns: {"valid_metrics_received": bool, "metric_names": {...}}
```

**Memory Operations**:
```python
# On successful extraction
mem_recall_append({
    "kind": "metrics_extracted",
    "node_id": node.id,
    "branch_id": node.branch_id,
    "metrics": extracted_metrics
})

mem_archival_write(
    text=f"Metrics for node {node.id}: {json.dumps(extracted_metrics)}",
    tags=["METRICS", f"node:{node.id}", f"stage:{stage_name}"]
)

# On failure
mem_recall_append({
    "kind": "metrics_failed",
    "node_id": node.id,
    "error": str(error)
})
```

### 2. Plotting Code Generation (`_generate_plotting_code`)

**Location**: `parallel_agent.py`

**Trigger**: After successful execution

**Prompt Structure**:
```python
plotting_prompt = {
    "Instructions": {
        "Response format": RESPONSE_FORMAT_DEFAULT,
        "Plotting code guideline": [
            guidelines,
            "Available .npy files: " + npy_files,
            "Use the following experiment code: " + node.code,
        ],
    },
}
_inject_memory(
    plotting_prompt,
    "plotting_code",
    branch_id=node.branch_id,
    budget_chars=cfg.memory.plotting_code_budget_chars
)
```

**LLM Query**: Uses `plan_and_code_query()` (not `generate_phase_artifacts()`)

**Memory Handling**:
```python
# If LLM includes <memory_update> in response
memory_updates = extract_memory_updates(completion_text)
if memory_updates:
    memory_manager.apply_llm_memory_updates(
        branch_id=node.branch_id,
        memory_updates,
        node_id=node.id,
        phase="plotting_code"
    )
    completion_text = remove_memory_update_tags(completion_text)
```

### 3. VLM Analysis (`_analyze_plots_with_vlm`)

**Location**: `parallel_agent.py`

**Trigger**: After plots are generated (if any .png files exist)

**Prompt Structure**:
```python
prompt = {
    "Introduction": "Analyze the following plots...",
    "Experiment context": node.plan,
    "Images": [encoded_plot_images],  # Base64 encoded
    "Instructions": vlm_analysis_instructions,
}
_inject_memory(
    prompt,
    "vlm_analysis",
    branch_id=node.branch_id,
    budget_chars=cfg.memory.vlm_analysis_budget_chars
)
```

**Model**: Uses VLM-capable model (e.g., GPT-4V, Claude 3)

**Output**: Stored in `node.vlm_analysis` for use in summary

### 4. Node Summary (`_generate_node_summary`)

**Location**: `parallel_agent.py`

**Trigger**: After all post-processing complete

**Uses Two-Phase Pattern**: Yes (via `_run_memory_update_phase()`)

**Prompt Structure**:
```python
prompt = {
    "Introduction": summary_intro,
    "Node information": {
        "plan": node.plan,
        "code": node.code[:code_limit],
        "execution_result": node.term_out[:term_limit],
        "metrics": node.metric,
        "vlm_analysis": node.vlm_analysis,
    },
    "Instructions": summary_instructions,
}
_inject_memory(
    prompt,
    "node_summary",
    branch_id=node.branch_id,
    budget_chars=cfg.memory.node_summary_budget_chars
)
```

**Phase 1: Memory Update**:
```python
if self._is_memory_enabled and memory_manager and branch_id:
    _run_memory_update_phase(
        prompt=summary_prompt,
        memory_manager=memory_manager,
        branch_id=branch_id,
        node_id=node.id,
        phase_name="node_summary",
        model=cfg.agent.feedback.model,
        temperature=cfg.agent.feedback.temp,
        max_rounds=getattr(cfg.memory, "max_memory_read_rounds", 2),
        task_description=(
            "Review the node's execution results and update your memory with key findings, "
            "successful approaches, errors encountered, and insights for future nodes."
        ),
    )
```

**Phase 2: Task Execution (func_spec)**:
```python
summary_response = query(
    system_message=summary_prompt,
    user_message=None,
    func_spec=node_summary_spec,  # Structured response
    model=cfg.agent.feedback.model,
    temperature=cfg.agent.feedback.temp,
)
# Returns: {"summary": str, "key_findings": [...], ...}
```

**Memory Operations**:
```python
# On completion
summary_dict = {
    "node_id": node.id,
    "approach": summary.approach,
    "key_findings": summary.key_findings,
    "metrics_summary": summary.metrics_summary,
    "next_steps": summary.next_steps,
}

mem_recall_append({
    "kind": "node_summary",
    "node_id": node.id,
    "branch_id": node.branch_id,
    "summary": summary_dict
})

mem_archival_write(
    text=f"Node {node.id} summary: {json.dumps(summary_dict)}",
    tags=["NODE_SUMMARY", f"node:{node.id}", f"stage:{stage_name}"]
)
```

## Stage-Level Memory Events

At stage completion (from `agent_manager.py`):

### Substage Completion

```python
mem_recall_append({
    "kind": "substage_complete",
    "stage_name": stage.name,
    "best_node_id": best_node.id,
    "metric": best_node.metric
})

mem_archival_write(
    text=f"Substage {stage.name} complete: {feedback}",
    tags=["STAGE_COMPLETE", f"stage:{stage.name}"]
)
```

### Stage Progression

```python
mem_recall_append({
    "kind": "stage_progression",
    "from_stage": current_stage.name,
    "to_stage": next_stage.name,
    "reason": progression_reason
})
```

### Stage 2 Completion (Datasets)

```python
mem_archival_write(
    text=f"Stage 2 complete: {datasets_tested}",
    tags=["DATASETS_TESTED", "STAGE2_COMPLETE"]
)
```

### Stage 4 Completion (Ablations)

```python
mem_archival_write(
    text=f"Stage 4 ablations: {ablation_results}",
    tags=["ABLATIONS", "STAGE4_COMPLETE"]
)
```

## Journal Summary Generation

**Task hint**: `journal_summary`

**Function**: `journal.generate_summary()`

**Trigger**: Before each tree search step (in `step()`)

```python
memory_context = _memory_context(root_branch_id, "journal_summary")
memory_summary = journal.generate_summary(
    include_code=False,
    memory_context=memory_context,
    model=cfg.agent.summary.model,
    temp=cfg.agent.summary.temp,
)
```

**Note**: Journal summary is used in-memory but NOT written back to memory
(to keep root branch clean).

## Global Metrics Definition

**Task hint**: `define_metrics`

**Function**: `_define_global_metrics()`

**Trigger**: Once during experiment initialization

```python
prompt = {
    "Introduction": DEFINE_METRICS_INTRO,
    "Research idea": task_desc,
    "Instructions": DEFINE_METRICS_INSTRUCTIONS,
}
memory_context = _memory_context(root_branch_id, "define_metrics")
if memory_context:
    prompt["Memory"] = memory_context

# LLM defines evaluation metrics
# Result stored in self.evaluation_metrics
```

## Memory Budget Configuration

```yaml
memory:
  # Multi-round memory read settings
  max_memory_read_rounds: 2           # Max re-query cycles for read operations

  # Post-execution budgets (Two-Phase methods)
  execution_review_budget_chars: 2000
  metrics_extraction_budget_chars: 1500
  plot_selection_budget_chars: 1000
  node_summary_budget_chars: 2000

  # Other post-execution budgets
  plotting_code_budget_chars: 2000
  vlm_analysis_budget_chars: 1000
  parse_metrics_budget_chars: 2000
  datasets_tested_budget_chars: 1500
```

**Note**: Methods marked with "Two-Phase Pattern" in the overview diagram use
`_run_memory_update_phase()` which respects `max_memory_read_rounds` for
iterative memory operations before task execution.

## Final Memory Export

At experiment completion, comprehensive paper-ready documentation is generated
by extracting information from the best performing nodes:

```python
# Generate final memory for paper writeup
memory_manager.generate_final_memory_for_paper(
    run_dir=Path(cfg.workspace_dir),
    root_branch_id=root_branch_id,
    best_branch_id=best_node.branch_id,
    artifacts_index={
        "best_node_data": {...},    # Best node's full attributes
        "top_nodes_data": [...],    # Top N nodes for comparison
    },
)
```

**Output Files**:
- `final_memory-for-paper.md` - Comprehensive markdown for paper writeup
- `final_memory_for_paper.json` - Structured data with node information
- `final_writeup_memory.json` - Complete writeup payload

**Contents**:
- Executive summary with best metric
- Best node details (plan, code, phase artifacts)
- VLM analysis and visual feedback
- Top nodes comparison table
- Memory-based analysis (Core/Recall/Archival)
- Resources used
- Negative results and lessons learned
- Provenance chain

For detailed documentation, see [memory-for-paper.md](memory-for-paper.md).

## See Also

- [memory-flow.md](memory-flow.md) - Overview
- [memory-flow-phase0.md](memory-flow-phase0.md) - Phase 0 flow
- [memory-flow-phases.md](memory-flow-phases.md) - Phase 1-4 flow
- [memory-for-paper.md](memory-for-paper.md) - Final memory for paper generation
