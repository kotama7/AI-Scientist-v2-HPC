# Memory For Paper Generation

This document describes the `generate_final_memory_for_paper` functionality, which extracts comprehensive research information from the best performing nodes and generates paper-ready documentation.

## Overview

After the tree search completes, the system generates a final memory document that consolidates:
- Best node's implementation details (code, plan, phase artifacts)
- VLM (Vision Language Model) analysis and visual feedback
- Top N nodes comparison for ablation/comparative analysis
- 3-tier memory content (Core, Recall, Archival)
- Resource usage information
- Negative results and lessons learned

## Output Files

The function generates the following files in `<workspace>/memory/`:

| File | Description |
|------|-------------|
| `final_memory_for_paper.md` | Human-readable markdown for paper writeup |
| `final_memory_for_paper.json` | Structured JSON for programmatic access |
| `final_writeup_memory.json` | Complete writeup memory payload |

## Markdown Structure

The generated `final_memory_for_paper.md` contains the following sections:

### 1. Executive Summary
- Best result metric value and name
- Analysis summary from the best node

### 2. Best Node Details
- Node ID and Branch ID
- Workspace and results directory paths
- Overall plan and implementation plan
- Code implementation
- Phase artifacts (Phase 0-4 outputs)

### 3. VLM Analysis & Visual Feedback
- Plot analyses from Vision Language Model
- VLM feedback summary
- Successfully tested datasets
- Generated plot paths

### 4. Top Nodes Comparison
- Table comparing top 5 nodes by metric
- Detailed breakdown of each top node's:
  - Metric value
  - Plan summary
  - Analysis
  - VLM feedback

### 5. Memory-Based Analysis
- LLM-generated sections from 3-tier memory
- Core memory insights
- Recall memory timeline analysis
- Archival memory content

### 6. Resources Used
- Data sources and datasets
- File paths and digests
- Usage context

### 7. Execution Feedback
- Execution time analysis
- Performance notes

### 8. Negative Results & Lessons Learned
- Recorded failures and errors
- Insights from unsuccessful attempts

### 9. Provenance Chain
- Branch lineage from root to best node

## Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                      Tree Search Completion                         │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Collect Best Nodes                            │
│                                                                     │
│  1. Find best_node by metric                                       │
│  2. Collect top 5 nodes sorted by metric                           │
│  3. Extract node attributes:                                        │
│     - id, branch_id, plan, overall_plan, code                      │
│     - phase_artifacts, analysis, metric                            │
│     - plot_analyses, vlm_feedback_summary                          │
│     - datasets_successfully_tested, plot_paths                     │
│     - exec_time_feedback, workspace_path                           │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                 generate_final_memory_for_paper()                   │
│                                                                     │
│  Inputs:                                                            │
│  - run_dir: Workspace directory path                               │
│  - root_branch_id: Root branch ID                                  │
│  - best_branch_id: Best node's branch ID                           │
│  - artifacts_index: Contains best_node_data and top_nodes_data     │
└────────────────────────────────────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  3-Tier Memory │   │   Node Data   │   │   Resources   │
    │   Extraction   │   │  Integration  │   │   Collection  │
    │                │   │               │   │               │
    │ - Core Memory  │   │ - Best node   │   │ - Data files  │
    │ - Recall Memory│   │ - Top 5 nodes │   │ - Datasets    │
    │ - Archival     │   │ - VLM data    │   │ - Artifacts   │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Generate Markdown & JSON                         │
│                                                                     │
│  - final_memory_for_paper.md (human-readable)                      │
│  - final_memory_for_paper.json (structured data)                   │
│  - final_writeup_memory.json (complete payload)                    │
└────────────────────────────────────────────────────────────────────┘
```

## Node Data Extraction

The following attributes are extracted from the best node:

| Attribute | Description |
|-----------|-------------|
| `id` | Unique node identifier |
| `branch_id` | Memory branch ID for this node |
| `plan` | Implementation plan text |
| `overall_plan` | High-level research plan |
| `code` | Python code implementation |
| `phase_artifacts` | Output from each phase (0-4) |
| `analysis` | Post-execution analysis |
| `metric` | Performance metric (value and name) |
| `exp_results_dir` | Directory containing experiment results |
| `plot_analyses` | VLM analysis of generated plots |
| `vlm_feedback_summary` | Summary of VLM feedback |
| `datasets_successfully_tested` | List of successful dataset tests |
| `plot_paths` | Paths to generated visualization plots |
| `exec_time_feedback` | Execution time analysis |
| `workspace_path` | Node's workspace directory |

## Top Nodes Data

For comparative analysis, the top 5 nodes (by metric) are collected with:
- `id`, `branch_id`
- `plan` (truncated for summary)
- `metric` (value and name)
- `analysis`
- `vlm_feedback_summary`

## Configuration

The following configuration options affect output:

```yaml
memory:
  final_memory_enabled: true           # Enable/disable final memory generation
  final_memory_filename_md: "final_memory_for_paper.md"
  final_memory_filename_json: "final_memory_for_paper.json"
```

## API Reference

### `generate_final_memory_for_paper()`

```python
def generate_final_memory_for_paper(
    self,
    run_dir: str | Path,
    root_branch_id: str,
    best_branch_id: str | None,
    artifacts_index: dict[str, Any] | None = None,
    no_budget_limit: bool = True,
) -> dict[str, Any]:
    """Generate final memory for paper writeup.

    Args:
        run_dir: Path to run directory
        root_branch_id: Root branch ID
        best_branch_id: Best branch ID (or None to use root)
        artifacts_index: Dictionary containing:
            - log_dir: Log directory path
            - workspace_dir: Workspace directory path
            - best_node_id: ID of the best node
            - best_node_data: Comprehensive best node attributes
            - top_nodes_data: List of top N nodes for comparison
        no_budget_limit: If True (default), do not truncate/compress content

    Returns:
        Dictionary with all paper sections including node data
    """
```

### artifacts_index Structure

```python
artifacts_index = {
    "log_dir": str,                    # Path to log directory
    "workspace_dir": str,              # Path to workspace
    "best_node_id": str,               # Best node's ID
    "best_node_data": {                # Best node's full data
        "id": str,
        "branch_id": str,
        "plan": str,
        "overall_plan": str,
        "code": str,
        "phase_artifacts": dict,
        "analysis": str,
        "metric": {"value": float, "name": str},
        "exp_results_dir": str,
        "plot_analyses": list[str],
        "vlm_feedback_summary": list[str],
        "datasets_successfully_tested": list[str],
        "plot_paths": list[str],
        "exec_time_feedback": str,
        "workspace_path": str,
    },
    "top_nodes_data": [                # Top N nodes for comparison
        {
            "id": str,
            "branch_id": str,
            "plan": str,
            "metric": {"value": float, "name": str},
            "analysis": str,
            "vlm_feedback_summary": list[str],
        },
        ...
    ],
}
```

## Usage Example

The function is automatically called at the end of tree search:

```python
# In perform_experiments_bfts_with_agentmanager.py
if memory_manager and getattr(memory_cfg, "final_memory_enabled", True):
    # Collect best node and top nodes
    best_node = find_best_node(manager.journals)
    top_nodes = collect_top_nodes(manager.journals, n=5)

    # Build comprehensive artifacts_index
    artifacts_index = {
        "log_dir": str(cfg.log_dir),
        "workspace_dir": str(cfg.workspace_dir),
        "best_node_id": best_node.id,
        "best_node_data": extract_node_data(best_node),
        "top_nodes_data": [extract_node_data(n) for n in top_nodes],
    }

    # Generate final memory
    memory_manager.generate_final_memory_for_paper(
        run_dir=Path(cfg.workspace_dir),
        root_branch_id=root_branch_id,
        best_branch_id=best_node.branch_id,
        artifacts_index=artifacts_index,
    )
```

## Related Documentation

- [memory.md](memory.md) - Memory system overview
- [memory_flow.md](memory_flow.md) - Memory architecture
- [memory_flow_phases.md](memory_flow_phases.md) - Phase execution with memory
- [memgpt-implementation.md](memgpt-implementation.md) - Implementation details
