# Visualization Tools

This document explains the HTML tools used to visualize experiment progress and
results.

## Overview

AI-Scientist-v2 generates the following HTML files for visualization:

| File | Purpose |
|------|---------|
| `unified_tree_viz.html` | Unified tree search viewer (code, plan, metrics, plots) across all stages |
| `tree_plot.html` | Per-stage tree viewer |
| `memory_database.html` | Detailed memory operations viewer (own/inherited data) |

These files are written under `experiments/<run>/logs/`.

## Output locations

```
experiments/<run>/logs/
├── unified_tree_viz.html           # Unified viewer across all stages
├── memory_database.html            # Memory database viewer
├── memory/                         # Memory operation logs (jsonl)
├── phase_logs/                     # Phase execution logs
├── stage_1_initial_implementation_*/
│   ├── tree_data.json              # Stage 1 tree data
│   ├── tree_plot.html              # Stage 1 tree viewer
│   └── ...
├── stage_2_baseline_tuning_*/
│   ├── tree_data.json
│   ├── tree_plot.html
│   └── ...
├── stage_3_creative_research_*/
│   └── ...
└── stage_4_ablation_studies_*/
    └── ...
```

## How to use

Open the HTML files in a local browser or a lightweight dev server:

```bash
# Open directly in a browser
firefox experiments/<run>/logs/unified_tree_viz.html

# Or use a dev server (auto-reload)
# e.g., VSCode Live Server
```

---

## unified_tree_viz.html

### Overview

An interactive tree search visualizer built with p5.js. It shows node
relationships for each stage.

### Layout

```
+------------------+------------------------+
|                  |                        |
|   Tree Canvas    |     Detail Panel       |
|   (Left 40%)     |     (Right 60%)        |
|                  |                        |
|   Click a node   |   - Plan               |
|   to view        |   - Exception Info     |
|   details        |   - Execution Time     |
|                  |   - Metrics            |
|                  |   - Memory Operations  |
|                  |   - Plot Plan          |
|                  |   - Plots              |
|                  |   - VLM Feedback       |
|                  |   - Code               |
|                  |   - Plot Code          |
+------------------+------------------------+
```

### Stage tabs (Stage 1–4 buttons)

The four fixed tabs at the top switch between stages.

| Button | Stage | Description | `selectStage()` behavior |
|--------|-------|-------------|--------------------------|
| **Stage 1** | Preliminary Investigation | Initial implementation and validation (`stage_1_initial_implementation_*`) | Loads `stageData['Stage_1']` and redraws the tree |
| **Stage 2** | Baseline Tuning | Baseline tuning (`stage_2_baseline_tuning_*`) | Loads `stageData['Stage_2']` and redraws the tree |
| **Stage 3** | Research Agenda Execution | Creative research (`stage_3_creative_research_*`) | Loads `stageData['Stage_3']` and redraws the tree |
| **Stage 4** | Ablation Studies | Ablation (`stage_4_ablation_studies_*`) | Loads `stageData['Stage_4']` and redraws the tree |

**Implementation details** (`template.js`):
- `selectStage(stageId)`: toggles tab CSS and calls `startSketch(stageId)`
- `loadAllStageData(baseTreeData)`: loads each stage's `tree_data.json` via `fetch()`; only stages in `completed_stages` are loaded
- `updateTabVisibility()`: disables tabs for stages that were not loaded

### Node interactions

- **Click** (`mousePressed`): selects a node and updates the detail panel via `setNodeInfo()`
- **Hover**: cursor changes to `HAND` (checked by `isMouseOver()`)
- **Selection**: node color changes to the accent color (`#1a439e`) and a checkmark is drawn
- **Animation**: node appearance uses scale + pop effects (`appearProgress`, `popEffect`)

### Detail panel sections

When a node is selected, `setNodeInfo()` renders data from `tree_data.json`.

#### Plan (`#plan`)
Syntax-highlighted plan text via `highlight.js`.

#### Exception Info (`#exc_info`)
Shows error details from `treeData.exc_type`, `exc_info`, `exc_stack`. If no
error, shows "No exception info available".

#### Execution Time (`#exec_time`, `#exec_time_feedback`)
- `treeData.exec_time[nodeIndex]`: execution time in seconds
- `treeData.exec_time_feedback[nodeIndex]`: feedback on runtime

#### Metrics (`#metrics`)
Displays metrics in a table:
- **metric_name**
- **description**
- **lower_is_better**
- **data** per dataset (dataset name + value)

Iterates `metrics.metric_names` and renders per-metric tables.

#### Memory Operations (`#memory-panel`)
A dedicated panel that groups memory ops by phase, using
`treeData.memory_events[nodeIndex]`.

##### Phase navigation buttons

| Button | Function | Behavior |
|--------|----------|----------|
| **◀ Prev** | `shiftMemoryPhase(-1)` | Cycle to previous phase |
| **Next ▶** | `shiftMemoryPhase(1)` | Cycle to next phase |

The current phase label is shown in `#memory-phase-label`.

**Phase grouping** (`groupMemoryEvents` + `inferPhaseFromOp`):
- If `phase` field exists, use it
- Else infer from op name:
  - `node_fork`, `branch` → `node_setup`
  - `resources` → `resource_init`
  - `core_set`, `set_core`, `core_get`, `get_core` → `initialization`
  - `archival` → `archival_ops`
  - other → `system`

**Phase order** (`sortMemoryPhases`):
`node_setup` → `resource_init` → `initialization` → `phase0` → `phase1` →
`phase2` → `phase3` → `phase4` → `define_metrics` → `journal_summary` →
`archival_ops` → `system`

Each phase shows a summary table (`renderMemorySummary`) followed by individual
operation events.

##### Filter buttons

Eight filter buttons filter operation types:

| Button | `data-filter` | Included ops (`MEMORY_OP_CATEGORIES`) | Color |
|--------|--------------|---------------------------------------|-------|
| **All** | `all` | All ops | - |
| **📖 Reads** | `reads` | `get_core`, `mem_core_get`, `render_for_prompt`, `mem_node_read`, `mem_archival_search`, `mem_archival_get`, `mem_recall_search`, `retrieve_archival` | `#4dabf7` |
| **💾 Writes** | `writes` | `set_core`, `mem_core_set`, `write_archival`, `mem_archival_write`, `mem_archival_update`, `mem_node_write`, `write_event` | `#69db7c` |
| **🗑️ Deletes** | `deletes` | `core_evict`, `core_delete`, `core_digest_compact` | `#ff6b6b` |
| **🌿 Forks** | `forks` | `mem_node_fork` | `#da77f2` |
| **🔄 Recalls** | `recalls` | `mem_recall_append` | `#ffd43b` |
| **📦 Resources** | `resources` | Present in template, not mapped in JS | - |
| **🔧 Maintenance** | `maintenance` | `consolidate_recall_events`, `check_memory_pressure`, `auto_consolidate_memory`, `evaluate_importance_with_llm` | `#adb5bd` |

`setMemoryFilter(filter)` updates CSS, rerenders the phase, and uses
`filterEventsByCategory()` to filter events.

##### Event card rendering (`formatMemoryEvent`)

Each event card shows:
- **Badge**: category icon and label (e.g., `📖 Reads`, `💾 Writes`)
- **Op name** (`op`): e.g., `mem_core_set`, `render_for_prompt`
- **Memory type** (`memory_type`): `core`, `archival`, `recall`, etc.
- **Key info** (when relevant): `key`, `value_chars`, `record_id`
- **Metadata**: timestamp, `node_id`, `branch_id`
- **Details**: `details` JSON, expandable in `<pre>`

#### Plot Plan (`#plot_plan`)
Displays `treeData.plot_plan[nodeIndex]`.

#### Plots (`#plots`)
Renders plot images from `treeData.plots[nodeIndex]` via `<img>` tags.

#### VLM Feedback (`#vlm_feedback`)
Displays VLM feedback in three subsections:

1. **Plot Analysis** (`treeData.plot_analyses[nodeIndex]`)
   - `analysis.plot_path`
   - `analysis.analysis`
   - `analysis.key_findings`
2. **VLM Feedback Summary** (`treeData.vlm_feedback_summary[nodeIndex]`)
3. **Datasets Successfully Tested** (`treeData.datasets_successfully_tested[nodeIndex]`)

#### Code (`#code`)
Shows `treeData.code[nodeIndex]` with Python syntax highlighting.

#### Plot Code (`#plot_code`)
Shows `treeData.plot_code[nodeIndex]` with Python syntax highlighting.

---

## memory_database.html

### Overview

A detailed viewer for the memory database. Uses a resizable p5.js panel layout
and a modular template system (v2) that composes `memory_database.js`,
`tree_canvas.js`, `resizable.js`, `common.css`, and `memory_database.css`.

### Layout

```
+------------------+|+------------------------+
|                  ||                        |
|   Tree Canvas    ||     Detail Panel       |
|   (tree view)    ||     (tabbed)           |
|                  ||                        |
+------------------+|+------------------------+
                    ^
                    Resizer (drag to adjust)
```

- **Left panel**: p5.js tree canvas; clicking nodes updates the right panel
- **Resizer**: draggable splitter (`ResizablePanel`), ratio saved to `localStorage`
- **Right panel**: six tabs for memory views

### View tabs (6 buttons)

Buttons call `switchView(view)`, update `currentView`, and rerender via
`renderNodeContent()`.

| Tab | `data-view` | Render function | Description |
|-----|-------------|-----------------|-------------|
| **Summary** | `summary` | `renderSummaryView()` | Summary stats of memory ops |
| **Effective Memory** | `effective` | `renderEffectiveMemoryView()` | Memory actually visible to the LLM (own + inherited) |
| **Memory Flow** | `memory-flow` | `renderMemoryFlowView()` | Operation + injection sequence |
| **By Phase** | `by-phase` | `renderByPhaseView()` | Grouped by phase |
| **Timeline** | `timeline` | `renderTimelineView()` | Chronological view |
| **All Data** | `all` | `renderAllDataView()` | Detailed own/inherited view |

Below are the tab details and relevant functions.

### Summary tab (`renderSummaryView`)

- **Inheritance Chain**: `renderAncestorChain()` shows `nodeData.ancestors`; each
  ancestor is clickable via `selectNodeByIndex(index)`.
- **This Node's Memory Operations**: `renderOperationsSummary()` shows a summary
  by category.

### Effective Memory tab (`renderEffectiveMemoryView`)

Shows the memory state the LLM sees (own + inherited), including:
- Core memory
- Recall memory
- Archival search results

### Memory Flow tab (`renderMemoryFlowView`)

Shows the sequence of memory operations and injections, useful for debugging
context assembly.

### By Phase tab (`renderByPhaseView`)

Groups operations by phase and shows counts and details per phase.

### Timeline tab (`renderTimelineView`)

Chronological list of all memory operations.

### All Data tab (`renderAllDataView`)

The most detailed view, showing:
- Own vs inherited records
- Full operation detail payloads
- Raw data as stored in logs
