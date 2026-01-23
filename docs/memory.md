# MemGPT-Style Memory

The system supports hierarchical memory per branch when `memory.enabled=true`.
It persists essential context across tree search branches without flooding every
prompt.

## Memory overview

![Memory flow](images/memory_flow.png)

## Enabling memory

- Set `memory.enabled=true` in `bfts_config.yaml`, or
- Pass `--enable_memgpt` on the CLI.

The default database path is `experiments/<run>/memory/memory.sqlite` and can be
overridden with `--memory_db`.

## Behavior when MemGPT is disabled (memGPT無効時の挙動)

When `memory.enabled=false` (or `--enable_memgpt` is not passed), the system
operates **without any context management**. This has significant implications:

### What happens without MemGPT

| Aspect | With MemGPT | Without MemGPT |
|--------|-------------|----------------|
| **Idea/task description** | Compressed summary injected | Full text injected as-is |
| **Context budget** | Managed by `memory_budget_chars` etc. | **No management** (depends on LLM API limits) |
| **Execution output** | LLM compression + truncation | Fixed truncation only |
| **Branch inheritance** | Child nodes inherit parent memory | None (each node is independent) |
| **Long-term memory** | SQLite persistence | None |
| **Final memory export** | `final_memory_for_paper.*` generated | Not generated |

### Context overflow risk

Without MemGPT, the system does **not** perform intelligent truncation or
compression on the main context (idea, task description, phase summaries).
These are injected as full text into every prompt.

**Important**: If the combined prompt exceeds the LLM's context window limit,
the API may return an error or silently truncate the input. This is especially
problematic for:

- Complex ideas with long abstracts and experiment descriptions
- Multi-stage experiments with accumulated history
- Large resource file summaries

### Existing truncation (independent of MemGPT)

Some truncation mechanisms exist regardless of MemGPT status, but they apply
only to specific outputs:

- **Terminal output**: `trim_long_string()` in `treesearch/utils/response.py`
  truncates execution output to a fixed character limit.
- **Execution logs**: `treesearch/utils/phase_execution.py` limits log lines
  and characters for feedback prompts.

These are hardcoded safety limits for execution results, **not** general
context management.

### Recommendations

| Use Case | Recommendation |
|----------|----------------|
| Short experiments / testing | MemGPT optional |
| Long-running complex experiments | **Enable MemGPT** |
| Paper generation planned | **Strongly recommend MemGPT** |
| Limited LLM context window | **Require MemGPT** |

To enable MemGPT:

```bash
python launch_scientist_bfts.py --enable_memgpt ...
```

Or in `bfts_config.yaml`:

```yaml
memory:
  enabled: true
```

## Memory layers

- Core: always-injected essentials (idea summary + Phase 0 internal summary,
  plus keys you set).
- Recall: recent events for this branch (small window).
- Archival: long-term store searched at injection time (FTS5 if available;
  keyword fallback otherwise).

## Branch behavior

- Child nodes inherit Core/Recall/Archival visibility from ancestors.
- Writes are isolated to the current branch (siblings do not see each other).

## Persistence hooks

- Phase 0 internal info is captured into archival memory (tag
  `PHASE0_INTERNAL`) and summarized into Core.
- `idea.md` is archived at run start (tags `IDEA_MD`, `ROOT_IDEA`) and on updates
  per node; a short summary is always injected.
- Run end generates `experiments/<run>/memory/final_memory_for_paper.md|json`.

## Phase 1-4 Memory Operations

Memory is updated throughout all experiment phases, not just Phase 0:

### Node-level events (parallel_agent.py)

| Event | Memory Layer | Tags | When |
|-------|--------------|------|------|
| `node_created` | Recall | - | Node initialized with plan |
| `node_result` | Recall | - | Node execution completed |
| Success details | Archival | `SUCCESS`, `stage:X` | Node succeeded with metric |
| Error details | Archival | `ERROR` | Node failed with exception |

### Split-Phase Execution Events (phase_mode=split)

When `exec.phase_mode` is set to `"split"`, additional memory events track
the multi-step execution process between `node_created` and `node_result`:

```
node_created
    │
    ├─ phase1_complete / phase1_failed    (Download/Install)
    │
    ├─ coding_complete / coding_failed    (File generation)
    │
    ├─ compile_complete / compile_failed  (Compilation)
    │
    └─ run_complete / run_failed          (Execution)
    │
node_result
```

| Event | Memory Layer | When |
|-------|--------------|------|
| `phase1_complete` | Recall | Phase 1 download/install succeeded |
| `phase1_failed` | Recall | Phase 1 download/install failed |
| `coding_complete` | Recall | Coding phase wrote files to workspace |
| `coding_failed` | Recall | Coding phase failed to write files |
| `compile_complete` | Recall | Compilation succeeded |
| `compile_failed` | Recall | Compilation failed (with error details) |
| `run_complete` | Recall | Run phase succeeded with expected outputs |
| `run_failed` | Recall | Run phase failed or outputs missing |

These split-phase events enable:
- Debugging multi-step HPC workflow failures
- Tracking which phase caused issues
- Learning from compilation/runtime errors
- Optimizing build configurations across nodes

### Metrics & Summary events

| Event | Memory Layer | Tags | When |
|-------|--------------|------|------|
| `metrics_extracted` | Recall + Archival | `METRICS` | Metrics successfully parsed |
| `metrics_failed` | Recall | - | Metrics parsing failed |
| `node_summary` | Recall + Archival | `NODE_SUMMARY` | Node summary generated |
| `journal_summary` | Recall | - | Journal-level summary generated |
| `stage_summary` | Recall + Archival | `SUMMARY`, `FINDINGS` | Stage progress summary |

These metrics and summary events enable:
- Tracking experimental metrics across nodes
- Storing key findings for future reference
- Providing context for analysis and report generation
- Learning from measurement patterns

### Stage-level events (agent_manager.py)

| Event | Memory Layer | Tags | When |
|-------|--------------|------|------|
| `substage_complete` | Recall + Archival | `STAGE_COMPLETE` | Substage goals achieved |
| `substage_incomplete` | Recall | - | Substage criteria not met |
| `stage2_complete` | Recall + Archival | `DATASETS_TESTED` | Stage 2 finished |
| `stage4_complete` | Recall + Archival | `ABLATIONS` | Ablations completed |
| `goal_generated` | Recall + Archival | `GOAL` | New substage goals created |
| `stage_progression` | Recall (+Archival) | `STAGE_PROGRESSION` | Stage ready for next |

These events enable:
- Tracking experiment progress across stages
- Learning from successful approaches
- Avoiding repeated failures
- Providing context for future decisions

## Resources in long-term memory

When memory is enabled, resource files are snapshotted into MemGPT long-term
memory:

- A pinned `RESOURCE_INDEX` core entry stores the resource digest, resource file
  sha, normalized config (essential fields), and an item table with retrieval
  tags.
- Items record `fetch_status` (`pending`/`available`/`failed`), resolved paths,
  staging info, and bounded summaries (`tree_summary` / `content_excerpt`) per
  `include_*` + `max_*`.
- GitHub/HF items are recorded as pending until their `dest` exists; after
  Phase 1 fetch, the snapshot and digest are updated.
- Prompts always include `RESOURCE_INDEX` and the top-K most relevant resource
  item memories (tags like `resource:<class>:<name>` and
  `resource_path:<path>`).
- Final paper memory includes a "Resources used" section with ids/digests/staged
  paths for referenced resources.

## LLM-based compression

When `memory.use_llm_compression=true` (default), the memory system uses an LLM
to intelligently compress content instead of simple truncation. This preserves
key information while fitting within size limits.

- Compression uses the model specified in `memory.compression_model`.
- The prompt template lives at `prompt/config/memory/compression.txt`.
- Compression is cached per text hash to avoid redundant LLM calls.
- Falls back to simple truncation on errors or when LLM is unavailable.
- Use `--memory_max_compression_iterations` (default 3) to control iterative
  compression attempts when content exceeds budget.

### Compression prompt template

The compression prompt (`prompt/config/memory/compression.txt`) instructs the LLM to:

1. Preserve key facts, metrics, and numerical values.
2. Keep critical decisions and conclusions.
3. Retain essential technical details and relationships.
4. Preserve information relevant to the experiment context (if provided).
5. Remove redundant information and verbose explanations.
6. Produce coherent, readable summaries (not keyword fragments).

The template accepts the following placeholders:

| Placeholder | Description |
| --- | --- |
| `{max_chars}` | Target maximum character count |
| `{current_chars}` | Current character count of the text |
| `{context_hint}` | Description of what the text represents |
| `{text}` | The original text to compress |
| `{memory_context}` | Raw memory context string (research goals, config) |
| `{memory_context_section}` | Formatted section with memory context (or empty if none) |

### Memory-context-aware compression

When a `branch_id` is provided during compression, the system automatically fetches
relevant memory context to help the LLM preserve information important to the research:

- **idea_md_summary**: Research goals and objectives from the idea document.
- **phase0_summary**: Experiment configuration and setup details.

This context is injected into the compression prompt as the `{memory_context_section}`
placeholder, helping the LLM make better decisions about what information to preserve.
For example, when compressing experimental results, the LLM will prioritize metrics
that align with the stated research goals.

### Section budgets

Section budgets control per-section character limits. Configure via
`memory.section_budgets.<key>` in `bfts_config.yaml`:

| Section | Default chars | Purpose |
| --- | --- | --- |
| `idea_summary` | 9600 | Compressed research idea |
| `idea_section_limit` | 4800 | Per-section limit for idea summary bullets |
| `phase0_summary` | 5000 | Phase 0 configuration summary |
| `archival_snippet` | 3000 | Archival memory excerpts |
| `results` | 4000 | Result summaries |

Additional per-section budget keys (configure directly under `memory`):

| Key | Default chars | Purpose |
| --- | --- | --- |
| `datasets_tested_budget_chars` | 4000 | Tested datasets summary |
| `metrics_extraction_budget_chars` | 4000 | Metrics extraction |
| `plotting_code_budget_chars` | 4000 | Plotting code summary |
| `plot_selection_budget_chars` | 4000 | Plot selection summary |
| `vlm_analysis_budget_chars` | 4000 | VLM analysis summary |
| `node_summary_budget_chars` | 4000 | Node summaries |
| `parse_metrics_budget_chars` | 4000 | Parsed metrics |

The overall budget is controlled by `memory.memory_budget_chars` (default 24000).

## Tuning memory size

- `memory.core_max_chars`: caps injected Core content.
- `memory.recall_max_events`: caps Recall events per branch.
- `memory.retrieval_k`: limits Archival hits injected per prompt.

## Inspecting memory artifacts

- `experiments/<run>/memory/memory.sqlite` stores the persistent memory tables.
- `experiments/<run>/memory/resource_snapshot.json` captures resolved resources.
- `experiments/<run>/memory/final_memory_for_paper.*` summarizes end-state memory.

If you need a clean slate, delete the run directory or set a new
`--memory_db` path for the next run.

## Performance notes

- If your SQLite build lacks FTS5, archival search falls back to keyword search.
- Large resource snapshots can increase memory size and injection time; tighten
  `max_chars` or `max_total_chars` in resource files if prompts grow too large.

## Memory Pressure Management (MemGPT-style)

The system implements MemGPT-style memory pressure management that automatically
detects memory overflow conditions and consolidates memory to stay within limits.

### How it works

1. **Pressure Detection**: Before each prompt injection, the system checks memory
   usage across Core and Recall layers. Pressure levels are:
   - `low`: < 70% usage (no action needed)
   - `medium`: 70-85% usage (evaluate importance, compress if needed)
   - `high`: 85-95% usage (consolidate recall, evict low-importance core)
   - `critical`: > 95% usage (aggressive consolidation, create digest)

2. **LLM-based Importance Evaluation**: When pressure is detected, an LLM
   evaluates the importance of memory items for the current task context,
   assigning scores 1-5. Items with low scores may be evicted or archived.

3. **Auto-consolidation**: Based on pressure level, the system:
   - Groups and summarizes related recall events
   - Moves low-importance core items to archival
   - Creates digest summaries for critical pressure

4. **Recall Event Consolidation**: When recall events exceed
   `recall_max_events * recall_consolidation_threshold`, older events are:
   - Grouped by kind/type
   - Summarized using LLM
   - Written to archival memory
   - Removed from recall

### Configuration

```yaml
memory:
  # Enable auto-consolidation (default: true)
  auto_consolidate: true

  # Trigger level for consolidation: "medium", "high", or "critical"
  consolidation_trigger: high

  # Recall overflow threshold multiplier (default: 1.5)
  recall_consolidation_threshold: 1.5

  # Pressure thresholds (0.0 - 1.0)
  pressure_thresholds:
    medium: 0.7    # 70% usage
    high: 0.85     # 85% usage
    critical: 0.95 # 95% usage
```

### Disabling memory pressure management

To disable auto-consolidation:

```yaml
memory:
  auto_consolidate: false
```

Or set `consolidation_trigger: critical` to only consolidate in extreme cases.

### Prompt templates

Memory pressure management uses these prompt templates:

- `prompt/config/memory/importance_evaluation.txt`: LLM importance scoring
- `prompt/config/memory/consolidation.txt`: LLM memory consolidation

Customize these files to adjust how the LLM evaluates and consolidates memory.

## MemGPT memory API (internal)

Memory operations are routed through a MemGPT-style API adapter (core/recall/
archival/node/resources). Core is bounded by `memory.core_max_chars` with
eviction + digesting; detailed content is stored in archival memory.

Available adapter methods (internal; arguments omitted):

- `mem_core_get`: fetch core entries for the current branch (or selected keys).
- `mem_core_set`: set/update a core key (supports `ttl` and `importance`).
- `mem_core_del`: delete a core key.
- `mem_recall_append`: append a recall event to the branch timeline.
- `mem_recall_search`: search recent recall events across the branch chain.
- `mem_archival_write`: write an archival record (long-term memory).
- `mem_archival_update`: update an archival record by id.
- `mem_archival_search`: search archival memory (FTS5 when available).
- `mem_archival_get`: fetch a single archival record by id.
- `mem_node_fork`: create a child branch for a new node.
- `mem_node_read`: read core/recall/archival for a node
  (`scope=all|core|recall|archival`).
- `mem_node_write`: batch write core updates, recall event, and/or archival
  records for a node.
- `mem_resources_index_update`: update the resource index entry for the run.
- `mem_resources_snapshot_upsert`: upsert a resource item snapshot into
  archival memory.
- `mem_resources_resolve_and_refresh`: refresh resource snapshots once
  resources are available.
