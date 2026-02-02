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

The generated `final_memory_for_paper.md` contains two types of sections:

### Part 1: Fixed Sections (from artifacts_index)

These sections are generated from node data provided in `artifacts_index`:

1. **Executive Summary**
   - Best result metric value and name
   - Analysis summary from the best node

2. **Best Node Details**
   - Node ID and Branch ID
   - Workspace and results directory paths
   - Overall plan and implementation plan
   - Code implementation
   - Phase artifacts (Phase 0-4 outputs)

3. **VLM Analysis & Visual Feedback**
   - Plot analyses from Vision Language Model
   - VLM feedback summary
   - Successfully tested datasets
   - Generated plot paths

4. **Top Nodes Comparison**
   - Table comparing top 5 nodes by metric
   - Detailed breakdown of each top node

### Part 2: LLM-Generated Sections (from 3-tier memory)

These sections are **dynamically generated** by the LLM based on available memory content.
The structure varies depending on `paper_section_mode`:

**Mode: `memory_summary` (default)**
- LLM autonomously decides what sections to include
- Common sections generated:
  - Title Candidates
  - Abstract Material
  - Problem Statement
  - Hypothesis
  - Method
  - **Experimental Setup** (⚠️ may be incomplete if PHASE0_INTERNAL is missing)
  - Results
  - Ablations/Negative Results
  - Failure Modes Timeline
  - Threats to Validity
  - Reproducibility Checklist
  - Narrative Bullets
  - Resources Used

**Mode: `idea_then_memory`**
- LLM generates section outline from idea.md first
- Then fills each section using memory search
- Number of sections controlled by `paper_section_count` config

**Note**: Section names use snake_case internally (e.g., `experimental_setup`) but are displayed as Title Case in markdown (e.g., "Experimental Setup").

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
│              LLM-Based Section Generation                           │
│                                                                     │
│  _build_paper_sections():                                          │
│  1. Collect 3-tier memory (Core, Recall, Archival)                │
│  2. Prepare memory summary for LLM                                 │
│  3. LLM generates paper sections dynamically                       │
│  4. Mode: memory_summary or idea_then_memory                       │
└────────────────────────────────────────────────────────────────────┘
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

## Implementation Details

### Memory Retrieval Process

The system retrieves memory from the best branch's chain using `_build_paper_sections()`:

**Location**: `ai_scientist/memory/memgpt_store.py:4114`

**Process**:
```python
# 1. Collect all core memory key-value pairs
core_memory = {}
for branch_id in branch_chain:
    # Fetch all core_kv entries for this branch
    # Earlier branches override later ones (child -> parent -> root)

# 2. Collect recent recall events
recall_memory = []
event_rows = self._fetch_events(branch_chain, k=500 if no_budget_limit else 100)
# Returns events ordered by created_at

# 3. Collect archival memory
archival_k = 200 if no_budget_limit else 50
archival_rows = self.retrieve_archival(
    branch_id=best_branch,
    query="",  # Empty query = retrieve all
    k=archival_k,
    include_ancestors=True,
)
# Returns entries ordered by created_at DESC (newest first)
```

**IMPORTANT - Two-Stage Limitation**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: retrieve_archival() - memgpt_store.py:4155-4161       │
├─────────────────────────────────────────────────────────────────┤
│ Fetch 200 entries from database, ORDER BY created_at DESC      │
│                                                                 │
│ Position  | Entry                | Created At    | Visible?    │
│ ---------------------------------------------------------------│
│    1      | [CORE_EVICT] ...     | 1769789792   | ✓ YES      │
│    2      | [SUMMARY] ...        | 1769789791   | ✓ YES      │
│   ...     | ...                  | ...          | ✓ YES      │
│   29      | [LLM_INSIGHT] ...    | 1769759800   | ✓ YES      │
│   30      | [RECALL_CONSOL] ...  | 1769759756   | ✓ YES      │
│   31      | [PHASE0_INTERNAL]... | 1769757910   | ✗ NO       │ ← LLM cannot see
│   32      | [IDEA_MD] ...        | 1769757900   | ✗ NO       │ ← LLM cannot see
│   ...     | ...                  | ...          | ✗ NO       │
│  200      | ...                  | ...          | ✗ NO       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: _prepare_memory_summary() - memgpt_store.py:4952      │
├─────────────────────────────────────────────────────────────────┤
│ Take only archival_memory[:30] for LLM context                 │
│                                                                 │
│ LLM sees: Entries 1-30 only                                    │
│ LLM CANNOT see: Entries 31-200 (including PHASE0_INTERNAL)     │
│                                                                 │
│ → Result: "Not found in memory" for environment information    │
└─────────────────────────────────────────────────────────────────┘
```

**Stage 1: Memory Retrieval** (`memgpt_store.py:4153-4161`)
- Retrieves up to **200 archival entries** (or 50 if budget limited)
- Ordered by `created_at DESC` (newest first)
- All 200 entries are stored in `archival_memory` list

**Stage 2: LLM Context Preparation** (`memgpt_store.py:4952`)
- Only the **first 30 entries** from the 200 are passed to LLM via `archival_memory[:30]`
- This hard limit is in `_prepare_memory_summary()`

**Impact on PHASE0_INTERNAL**:
- PHASE0_INTERNAL entries are created early (during Phase 0)
- If more than 30 newer entries exist, PHASE0_INTERNAL is in positions 31-200
- LLM never sees entries beyond position 30
- Result: Environment information is missing from final output ("Not found in memory")

### LLM-Based Section Generation

The system uses two modes for generating paper sections:

**Mode 1: `memory_summary` (default)**
- Location: `_generate_sections_from_memory_summary()`
- Prepares a comprehensive memory summary (Core + Recall + Archival)
- LLM autonomously decides what sections to generate
- Returns dynamic section structure based on available memory

**Mode 2: `idea_then_memory`**
- Location: `_generate_sections_from_idea()`
- First generates section outline from idea.md
- Then fills each section using memory search
- Returns fixed number of sections (configurable)

**Memory Summary Structure** (passed to LLM):
```markdown
## Idea
Summary: [idea_md_summary from core]
Text: [idea.md content]

## Core Memory (Key-Value Context)
### key1
value1
### key2
value2
...

## Recall Memory (Event Timeline)
- [timestamp] kind: text
- [timestamp] kind: text
...
(limited to first 50 events)

## Archival Memory (Long-term Storage)
### Entry 1 [tags]
text
### Entry 2 [tags]
text
...
### Entry 30 [tags]
text
... and N more archival entries
(⚠️ CRITICAL LIMITATION: Only first 30 of 200 retrieved entries are shown here)
(PHASE0_INTERNAL is often in positions 31-200, so LLM cannot see it)

## Resources Used
- resource1 (class)
- resource2 (class)
```

### Section Filtering

For specific sections like `experimental_setup`, the system:
1. Uses keyword matching to filter relevant memory
2. Adds specific core keys: `selected_compiler`, `omp_smoke_build`, etc.
3. Appends environment facts from `_append_experimental_setup_facts()`

**Location**: `memgpt_store.py:4628-4693`

## Configuration

The following configuration options affect output:

```yaml
memory:
  final_memory_enabled: true           # Enable/disable final memory generation
  final_memory_filename_md: "final_memory_for_paper.md"
  final_memory_filename_json: "final_memory_for_paper.json"

  # Section generation mode
  paper_section_mode: "memory_summary"  # or "idea_then_memory"
  paper_section_count: 12               # Number of sections for idea_then_memory mode

  # LLM compression settings
  use_llm_compression: true
  compression_model: "gpt-4"
  max_compression_iterations: 5
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

## Internal Implementation Flow

### Step-by-Step Execution

When `generate_final_memory_for_paper()` is called, the following sequence occurs:

**1. Resource Collection** (`memgpt_store.py:5060-5063`)
```python
resource_snapshot, resource_index, item_index = self._load_resource_snapshot_and_index(
    memory_dir, root_branch_id, best_branch
)
resources_used = self._collect_resource_usage(root_branch_id, best_branch, item_index, no_budget_limit)
```

**2. Build Paper Sections** (`memgpt_store.py:5066-5071`)
```python
sections = self._build_paper_sections(
    run_dir=run_dir,
    best_branch=best_branch,
    resources_used=resources_used,
    no_budget_limit=no_budget_limit,
)
```

This calls `_build_paper_sections()` which:

a. **Collects 3-tier memory** (`memgpt_store.py:4129-4168`):
   - Core memory: All key-value pairs from branch chain
   - Recall memory: Up to 500 events (or 100 if budget limited)
   - Archival memory: Up to **200 entries** (or 50 if budget limited), ordered by `created_at DESC`

b. **Prepares memory context** (`memgpt_store.py:4172-4181`):
   ```python
   memory_context = {
       "core_memory": core_memory,
       "recall_memory": recall_memory,
       "archival_memory": archival_memory,  # Ordered by created_at DESC
       "resources_used": resources_used,
       "best_branch": best_branch,
       "idea_text": idea_text,
       "idea_summary": idea_summary,
   }
   ```

c. **Generates sections with LLM** (`memgpt_store.py:4184`):
   - Calls `_generate_sections_with_llm(memory_context)`
   - Which calls either:
     - `_generate_sections_from_memory_summary()` (default), or
     - `_generate_sections_from_idea()` (if `paper_section_mode: idea_then_memory`)
   - **⚠️ Before passing to LLM**: `_prepare_memory_summary()` limits archival to **first 30 entries** (`memgpt_store.py:4952`)

**3. Extract Additional Data from Memory** (`memgpt_store.py:5099-5145`)
- Get idea text from archival (tag: IDEA_MD)
- Get phase0_summary from core memory
- Get results from core memory
- Get failure notes from recall events

**4. Build Writeup Memory Payload** (`memgpt_store.py:5146-5183`)
```python
writeup_memory = {
    "run_id": run_id,
    "idea": idea_text,
    "phase0_env": phase0_summary,
    "resources": resources_section,
    "method_changes": method_changes,
    "experiments": core_snapshot,
    "results": results_notes,
    "negative_results": failure_notes,
    "provenance": provenance,  # Branch chain
}
```

**5. Write Output Files** (`memgpt_store.py:5186-5215`)
- `final_memory_for_paper.json`: Sections + artifacts_index
- `final_memory_for_paper.md`: Human-readable markdown
- `final_writeup_memory.json`: Complete writeup payload

### Critical Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `generate_final_memory_for_paper()` | `memgpt_store.py:5034` | Main entry point |
| `_build_paper_sections()` | `memgpt_store.py:4114` | Extract and organize memory |
| `_generate_sections_with_llm()` | `memgpt_store.py:4188` | LLM-based section generation |
| `_prepare_memory_summary()` | `memgpt_store.py:4907` | Format memory for LLM (⚠️ limits archival to 30) |
| `_filter_memory_for_section()` | `memgpt_store.py:4628` | Filter memory by keywords |
| `_append_experimental_setup_facts()` | `memgpt_store.py:4412` | Add compiler/environment facts |
| `retrieve_archival()` | `memgpt_store.py:2522` | Query archival memory |

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

## Known Issues and Troubleshooting

### Issue: Environment Information Missing ("Not found in memory")

**Symptom**: The `experimental_setup` section in `final_memory_for_paper.md` shows:
```markdown
- **CPU model / sockets / cores / NUMA topology / memory capacity:** Not found in memory.
- **Operating system:** Not found in memory.
- **Compiler and flags:** Not found in memory.
```

**Root Cause** (Two-Stage Process):

**Stage 1 - Memory Retrieval** (`memgpt_store.py:4155-4161`):
- `retrieve_archival()` fetches up to **200 archival entries**
- Entries are ordered by `created_at DESC` (newest first)
- All 200 entries are available in `memory_context["archival_memory"]`

**Stage 2 - LLM Context Preparation** (`memgpt_store.py:4952`):
- `_prepare_memory_summary()` takes only `archival_memory[:30]`
- Only the **first 30 entries** (newest 30) are formatted for LLM
- Entries 31-200 are **never shown to the LLM**

**Why PHASE0_INTERNAL is missing**:
1. PHASE0_INTERNAL is created during Phase 0 (early in the experiment)
2. As the experiment progresses, newer archival entries are added (consolidations, evictions, insights)
3. If more than 30 newer entries exist, PHASE0_INTERNAL falls into positions 31-200
4. LLM only sees positions 1-30, so it cannot access PHASE0_INTERNAL
5. LLM outputs "Not found in memory" for environment information

**Verification**:
```bash
# 1. Check if PHASE0_INTERNAL exists in the database
sqlite3 <run_dir>/memory/memory.sqlite \
  "SELECT COUNT(*) FROM archival WHERE tags LIKE '%PHASE0_INTERNAL%';"

# 2. Check the creation timestamp and position
sqlite3 <run_dir>/memory/memory.sqlite \
  "SELECT id, created_at FROM archival WHERE tags LIKE '%PHASE0_INTERNAL%' ORDER BY created_at;"

# 3. Count ALL newer entries (determines if PHASE0_INTERNAL is beyond position 30)
sqlite3 <run_dir>/memory/memory.sqlite \
  "SELECT COUNT(*) FROM archival WHERE created_at > <timestamp_from_above>;"
# If this count is > 30, PHASE0_INTERNAL is NOT visible to LLM

# 4. Verify position in the ordered list (1-30 = visible, 31+ = invisible)
sqlite3 <run_dir>/memory/memory.sqlite \
  "WITH ordered AS (
    SELECT id, created_at, tags, ROW_NUMBER() OVER (ORDER BY created_at DESC) as pos
    FROM archival
  )
  SELECT pos, id, tags FROM ordered WHERE tags LIKE '%PHASE0_INTERNAL%';"
# If pos > 30, LLM cannot see it

# 5. See what the LLM actually sees (first 30 entries)
sqlite3 <run_dir>/memory/memory.sqlite \
  "SELECT id, tags FROM archival ORDER BY created_at DESC LIMIT 30;"
```

**Current Workarounds**:
1. Check the archival database directly for `PHASE0_INTERNAL` entries
2. Read `phase0_history_full.json` for environment_context
3. Manually extract environment information from early archival entries

**Proposed Fixes**:

**Option 1: Prioritize Important Tags** (Recommended)
- Modify `_prepare_memory_summary()` to always include entries with critical tags
- Ensure PHASE0_INTERNAL, IDEA_MD are in the first 30 regardless of timestamp
- Example: Sort by importance first, then by timestamp

**Option 2: Increase the 30-Entry Limit**
- Change `archival_memory[:30]` to `archival_memory[:100]` in `memgpt_store.py:4952`
- Trade-off: Larger LLM context, higher cost
- Since 200 entries are already retrieved, this only affects LLM context size

**Option 3: Tag-Based Pre-Filtering**
- Before limiting to 30, filter archival_memory to include all critical tags
- Add critical entries to guaranteed list, then fill remaining slots with newest entries
- Example: `critical_entries + newest_entries[:remaining_slots]`

**Option 4: Use Core Memory for Environment Info**
- Store environment summary in core memory (always visible)
- Key: `environment_info` with JSON value containing CPU, OS, compiler, etc.
- Downside: Requires changing how Phase 0 stores environment data

**Related Code Locations**:
- Stage 1 - Memory retrieval (200 entries): `memgpt_store.py:4153-4168`
- Stage 2 - LLM context preparation (30 entry limit): `memgpt_store.py:4907-4972`, specifically line 4952
- Section filtering: `memgpt_store.py:4628-4693`
- Archival query function: `memgpt_store.py:2522` (retrieve_archival)

### Issue: Sections Generated Vary by Run

**Symptom**: Different runs generate different section structures

**Cause**: Using `memory_summary` mode allows LLM to autonomously decide sections based on available memory

**Solution**: Switch to `idea_then_memory` mode for consistent section structure:
```yaml
memory:
  paper_section_mode: "idea_then_memory"
  paper_section_count: 12
```

### Issue: Missing Specific Information in Sections

**Symptom**: Expected information is not included in generated sections

**Debug Steps**:
1. Check if information exists in memory database:
   ```bash
   sqlite3 <run_dir>/memory/memory.sqlite "SELECT * FROM archival WHERE text LIKE '%keyword%';"
   ```

2. Verify archival entry timestamp and ordering:
   ```bash
   sqlite3 <run_dir>/memory/memory.sqlite "SELECT id, created_at, tags FROM archival ORDER BY created_at DESC LIMIT 50;"
   ```

3. Check if entry is in top 30 (memory summary limit):
   - If entry is older than position 30, it won't be seen by LLM

**Solutions**:
- Increase archival limit in code (modify `archival_memory[:30]` to higher value)
- Ensure critical information is tagged appropriately
- Use core memory for critical information (always visible)

## Example Output

### File Locations

After generation, you can find the output files at:

```
<run_dir>/memory/
├── final_memory_for_paper.md         # Human-readable markdown
├── final_memory_for_paper.json       # Structured sections
├── final_writeup_memory.json         # Complete writeup payload
├── memory.sqlite                     # Memory database (source data)
└── phase0_internal_info.json         # Phase 0 environment data (if exists)
```

### Sample Section Output

**When PHASE0_INTERNAL is available:**
```markdown
### Experimental Setup

#### Hardware and software environment
- **CPU model**: AMD EPYC 9554 64-Core Processor
- **Sockets**: 2
- **Cores**: 128 (64 per socket)
- **NUMA topology**: 2 NUMA nodes (node0: CPUs 0-63, node1: CPUs 64-127)
- **Memory capacity**: 1.5TB total (773GB per NUMA node)
- **Operating system**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Compiler**: gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
- **Compiler flags**: -O3 -march=native -fopenmp
```

**When PHASE0_INTERNAL is missing (the issue):**
```markdown
### Experimental Setup

#### Hardware and software environment
- **CPU model / sockets / cores / NUMA topology / memory capacity:** Not found in memory.
- **Operating system:** Not found in memory.
- **Compiler and flags:** The Himeno baseline is compiled with `-O3 -march=native`...
```

### Checking Generated Sections

You can inspect what sections were generated:

```bash
# List all section keys in the JSON output
jq 'keys' <run_dir>/memory/final_memory_for_paper.json

# View a specific section
jq '.experimental_setup' <run_dir>/memory/final_memory_for_paper.json

# Check if artifacts_index is included
jq '.artifacts_index.best_node_data.id' <run_dir>/memory/final_memory_for_paper.json
```

## Related Documentation

- [memory.md](memory.md) - Memory system overview
- [memory-flow.md](memory-flow.md) - Memory architecture
- [memory-flow-phase0.md](memory-flow-phase0.md) - Phase 0 memory operations
- [memory-flow-phases.md](memory-flow-phases.md) - Phase execution with memory
- [memgpt-implementation.md](memgpt-implementation.md) - Implementation details
- [memgpt-features.md](memgpt-features.md) - Available memory features
