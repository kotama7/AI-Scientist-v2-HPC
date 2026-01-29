# MemGPT-Style Memory

The system supports hierarchical memory per branch when `memory.enabled=true`.
It persists essential context across tree search branches without flooding every
prompt.

## Memory overview

![Memory flow](../images/memory_flow.png)

For a detailed flow diagram of memory operations during node execution, see
[memory-flow.md](memory-flow.md).

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

### Phase 1 Iterative Memory Operations

Phase 1 (download/install) runs as an iterative loop with LLM-driven memory operations.
When `memory.enabled=true`, the system uses `phase1_installer_with_memory.txt` prompt
which supports full `<memory_update>` blocks including read operations.

**Execution Flow per Iteration:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 ITERATION N                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Prompt Assembly                                                          │
│     ├─ Introduction (phase1_installer_with_memory.txt)                       │
│     ├─ Task description                                                      │
│     ├─ Phase 0 guidance (targets, preferred_commands, done_conditions)       │
│     ├─ Progress history (previous steps + results)                           │
│     └─ Memory injection via _inject_memory()                                 │
│                                                                              │
│  2. LLM Query                                                                │
│     └─ Returns: <memory_update> block + JSON command                         │
│                                                                              │
│  3. Memory Update Processing                                                 │
│     ├─ extract_memory_updates() parses <memory_update>                       │
│     └─ apply_llm_memory_updates() processes operations:                      │
│                                                                              │
│         Write Operations (immediate):                                        │
│         ├─ "core": {"key": "value"} → Core memory                            │
│         ├─ "archival": [{...}] → Archival with tags                          │
│         └─ "recall": {...} → Recall timeline                                 │
│                                                                              │
│         Read Operations (trigger re-query):                                  │
│         ├─ "core_get": ["key1", "key2"]                                      │
│         ├─ "archival_search": {"query": "...", "k": N}                       │
│         └─ "recall_search": {"query": "...", "k": N}                         │
│                                                                              │
│  4. Read Operation Re-Query Loop (if read results exist)                     │
│     ├─ _has_memory_read_results() checks for read results                    │
│     ├─ _format_memory_results_for_llm() formats results                      │
│     ├─ Inject "Memory Read Results" section into prompt                      │
│     └─ _run_memory_update_phase() re-queries up to max_memory_read_rounds    │
│                                                                              │
│  5. JSON Parse & Command Execution                                           │
│     ├─ parse_phase1_iterative_response() extracts {command, done, notes}     │
│     └─ Execute command in Singularity container                              │
│                                                                              │
│  6. Loop Control                                                             │
│     ├─ If done=true → Exit Phase 1                                           │
│     └─ If done=false → Proceed to Iteration N+1                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Prompt Selection Logic (parallel_agent.py):**

```python
phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT
if memory_cfg and getattr(memory_cfg, "enabled", False):
    phase1_intro = PHASE1_ITERATIVE_INSTALLER_PROMPT_WITH_MEMORY
```

| `memory.enabled` | Prompt File |
|------------------|-------------|
| `true` (default) | `prompt/config/phases/phase1_installer_with_memory.txt` |
| `false` | `prompt/config/phases/phase1_installer.txt` |

**Common Phase 1 Memory Operations:**

| Use Case | Memory Operation | Tags |
|----------|------------------|------|
| Record package version | `"core": {"numpy_version": "1.24.0"}` | - |
| Record git commit SHA | `"core": {"cnpy_commit": "abc1234"}` | - |
| Log installation details | `"archival": [{"text": "...", "tags": ["PHASE1_INSTALL"]}]` | `PHASE1_INSTALL` |
| Log error recovery | `"archival": [{"text": "...", "tags": ["PHASE1_ERROR"]}]` | `PHASE1_ERROR` |
| Record build config | `"archival": [{"text": "...", "tags": ["BUILD_CONFIG"]}]` | `BUILD_CONFIG` |
| Retrieve previous error | `"archival_search": {"query": "PHASE1_ERROR", "k": 3}` | - |
| Check installed packages | `"core_get": ["numpy_version", "python_deps_path"]` | - |

**Read Operation Flow:**

```
LLM Response with read operation
         │
         v
apply_llm_memory_updates()
         │
         ├─ Execute writes (core, archival, recall)
         │
         └─ Execute reads (core_get, archival_search, recall_search)
                  │
                  v
         Return memory_results
                  │
                  v
_has_memory_read_results(memory_results)?
         │
    ┌────┴────┐
   Yes       No
    │         │
    v         v
Re-query   Continue to
  Loop     JSON parse
    │
    v
Format results → Inject "Memory Read Results" → LLM Query
         │
         └─ Repeat up to max_memory_read_rounds times
```

### Phase 2/3/4 Memory-Enabled Prompts

Similar to Phase 1, all Phase 2/3/4 tasks use memory-enabled prompts when `memory.enabled=true`.
These prompts include `<memory_update>` instructions specific to each task type.

**Prompt Selection Table:**

| Task | `memory.enabled=false` | `memory.enabled=true` |
|------|------------------------|----------------------|
| **Draft** | `tasks/draft/introduction.txt` | `tasks/draft/introduction_with_memory.txt` |
| **Debug** | `tasks/debug/introduction.txt` | `tasks/debug/introduction_with_memory.txt` |
| **Improve** | `tasks/improve/introduction.txt` | `tasks/improve/introduction_with_memory.txt` |
| **Hyperparam** | `nodes/hyperparam/introduction.txt` | `nodes/hyperparam/introduction_with_memory.txt` |
| **Ablation** | `nodes/ablation/introduction.txt` | `nodes/ablation/introduction_with_memory.txt` |
| **Execution Review** | `tasks/execution_review/introduction.txt` | `tasks/execution_review/introduction_with_memory.txt` |
| **Summary** | `tasks/summary/introduction.txt` | `tasks/summary/introduction_with_memory.txt` |
| **Parse Metrics** | `tasks/parse_metrics/introduction.txt` | `tasks/parse_metrics/introduction_with_memory.txt` |
| **VLM Analysis** | `vlm_analysis.txt` | `vlm_analysis_with_memory.txt` |

**Code Pattern (parallel_agent.py):**

```python
def _draft(self) -> Node:
    # Select prompt based on memory configuration
    draft_intro = DRAFT_INTRO_WITH_MEMORY if self._is_memory_enabled else DRAFT_INTRO
    prompt: Any = {
        "Introduction": draft_intro,
        ...
    }
```

**Task-Specific Memory Guidelines:**

Each memory-enabled prompt includes task-specific guidelines:

| Task | Core Keys | Archival Tags |
|------|-----------|---------------|
| Draft | `algorithm_approach`, `baseline_method` | `PHASE2_DRAFT`, `BASELINE` |
| Debug | `last_bug_type`, `common_pitfalls` | `PHASE2_DEBUG`, `BUG_FIX` |
| Improve | `improvement_count`, `best_metric_achieved` | `PHASE2_IMPROVE`, `OPTIMIZATION` |
| Hyperparam | `tuning_method`, `best_params` | `HYPERPARAM`, `TUNING_RESULTS` |
| Ablation | `critical_components`, `removable_components` | `ABLATION`, `COMPONENT_ANALYSIS` |
| Summary | `iteration_number`, `key_metric` | `SUMMARY`, `ITERATION_N` |
| VLM Analysis | `plots_analyzed`, `main_finding` | `VLM_ANALYSIS`, `PLOT_INSIGHTS` |
| Metrics | `datasets_parsed`, `metric_names` | `METRICS_PARSE`, `RESULTS` |

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

Section budgets control per-section character limits. Configure directly under
`memory` in `bfts_config.yaml` (flat structure):

| Key | Default chars | Purpose |
| --- | --- | --- |
| `datasets_tested_budget_chars` | 8000 | Tested datasets summary |
| `metrics_extraction_budget_chars` | 12000 | Metrics extraction |
| `plotting_code_budget_chars` | 8000 | Plotting code summary |
| `plot_selection_budget_chars` | 8000 | Plot selection summary |
| `vlm_analysis_budget_chars` | 12000 | VLM analysis summary |
| `node_summary_budget_chars` | 8000 | Node summaries |
| `parse_metrics_budget_chars` | 12000 | Parsed metrics |
| `archival_snippet_budget_chars` | 12000 | Archival memory excerpts |
| `results_budget_chars` | 12000 | Result summaries |

Additional settings:
| Key | Default | Purpose |
| --- | --- | --- |
| `paper_section_mode` | `idea_then_memory` | Section generation mode (`memory_summary` or `idea_then_memory`) |
| `paper_section_count` | 12 | Number of sections to generate for `idea_then_memory` mode |
| `max_compression_iterations` | 5 | Maximum iterative compression attempts |

The overall budget is controlled by `memory.memory_budget_chars` (default 24000).

## Tuning memory size

- `memory.core_max_chars`: caps injected Core content.
- `memory.recall_max_events`: caps Recall events per branch.
- `memory.retrieval_k`: limits Archival hits injected per prompt.

## Inspecting memory artifacts

- `experiments/<run>/memory/memory.sqlite` stores the persistent memory tables.
- `experiments/<run>/memory/resource_snapshot.json` captures resolved resources.
- `experiments/<run>/memory/final_memory_for_paper.*` summarizes end-state memory.
- `experiments/<run>/logs/<run>/memory_database.html` interactive visualization.

If you need a clean slate, delete the run directory or set a new
`--memory_db` path for the next run.

For details on the memory visualization tool, see
[visualization.md](../visualization/visualization.md#memory_databasehtml).

## Performance notes

- If your SQLite build lacks FTS5, archival search falls back to keyword search.
- Large resource snapshots can increase memory size and injection time; tighten
  `max_chars` or `max_total_chars` in resource files if prompts grow too large.

## Centralized Database Writer (Parallel Execution)

When running with multiple parallel workers (`num_workers > 1`), the system uses
a centralized database writer process to avoid SQLite "database is locked" errors.

### Architecture

```
Worker Process 1 ─┐
Worker Process 2 ─┼─> Write Queue ─> DatabaseWriterProcess ─> SQLite DB
Worker Process N ─┘
```

All database write operations from worker processes are serialized through a
single dedicated writer process, eliminating concurrent write conflicts while
maintaining high throughput through batching.

### How it works

1. **Main process** starts `DatabaseWriterProcess` before spawning workers
2. **Writer process** owns the sole write connection to SQLite
3. **Worker processes** receive a `writer_queue` (multiprocessing.Queue)
4. **MemoryManager** routes writes through the queue instead of direct connection
5. **Reads** remain local to each worker (SQLite WAL handles concurrent reads)

### Features

| Feature | Description |
|---------|-------------|
| **Serialized writes** | Eliminates "database is locked" errors |
| **Batched commits** | Groups multiple writes for better I/O performance |
| **Automatic retry** | Exponential backoff on transient errors |
| **Graceful shutdown** | Drains queue and commits pending writes |
| **Health check** | `ping()` method for monitoring writer status |

### Configuration

The database writer is automatically enabled when:
- `memory.enabled=true`
- `num_workers > 1` (parallel execution)

No additional configuration is required. The writer uses the same `db_path`
specified in `memory.db_path` or defaults to
`experiments/<run>/memory/memory.sqlite`.

### Troubleshooting

If you still see "database is locked" errors:

1. Check that `DatabaseWriterProcess` started successfully (look for
   "DatabaseWriterProcess started" in logs)
2. Verify workers are receiving the `writer_queue` (look for
   "Using centralized database writer process" in worker logs)
3. Increase `busy_timeout` in memgpt_store.py if needed (default: 60s)

See [troubleshooting.md](../development/troubleshooting.md#database-is-locked-errors)
for more details.

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

5. **Inherited Memory Consolidation (Copy-on-Write)**: When the total inherited
   events from ancestors exceed the threshold, the system uses Copy-on-Write
   semantics to consolidate without modifying ancestor data:
   - Older inherited events are summarized using LLM
   - Summaries are stored in `inherited_summaries` table (local to this branch)
   - Excluded event IDs are stored in `inherited_exclusions` table
   - Ancestor branch data remains unchanged
   - Child branches see consolidated view while siblings see original data

### Inherited Memory Consolidation Details

The inherited memory consolidation feature ensures that each branch can manage
its inherited memory independently without affecting other branches. This is
achieved through Copy-on-Write (CoW) semantics:

#### How it works

```
Parent Branch (SQLite):      50 events (unchanged)
Child Branch A (SQLite):     20 events (own)
Child Branch B (SQLite):     15 events (own)

Without CoW consolidation:
  Child A sees: 50 (parent) + 20 (own) = 70 events
  Child B sees: 50 (parent) + 15 (own) = 65 events

With CoW consolidation (threshold=30):
  Child A triggers consolidation:
    - Summarizes 40 oldest parent events → stored in inherited_summaries
    - Records excluded IDs in inherited_exclusions
    - Effective view: summary + 10 recent parent + 20 own = ~30 events
  Child B (unaffected):
    - Still sees all 50 parent + 15 own = 65 events
  Parent (unaffected):
    - Still has all 50 events in SQLite
```

#### Database tables

| Table | Purpose |
|-------|---------|
| `inherited_exclusions` | Event IDs excluded from inherited view for each branch |
| `inherited_summaries` | LLM-generated summaries of consolidated inherited events |

#### When consolidation triggers

Inherited memory consolidation is triggered automatically when:
1. A recall event is appended (`mem_recall_append`)
2. The total inherited event count exceeds `recall_max_events * recall_consolidation_threshold`
3. `auto_consolidate` is enabled

The consolidation:
1. First consolidates inherited memory (Copy-on-Write)
2. Then consolidates own events if still over threshold (traditional deletion)

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

## LLM Memory Operations

The LLM can directly manage memory by including a `<memory_update>` block in its
responses. This enables the agent to record insights, search past experiences,
and manage its memory during experiment execution.

### Enabling LLM memory operations

LLM memory operations are automatically enabled when MemGPT is active
(`memory.enabled=true`). The memory operation instructions are injected into
prompts via `_inject_memory()` in `parallel_agent.py`.

### Available operations (LLM-callable)

The following operations can be invoked by the LLM via `<memory_update>` blocks:

**Core Memory:**
- `core`: set/update key-value pairs in core memory (always-visible).
- `core_get`: retrieve values for specified keys.
- `core_delete`: delete (evict) keys from core memory.

**Archival Memory:**
- `archival`: write new archival records with text and tags.
- `archival_update`: update existing archival records by ID.
- `archival_search`: search archival memory (semantic/keyword search).

**Recall Memory:**
- `recall`: append an event to the recall timeline.
- `recall_search`: search recent recall events.
- `recall_evict`: move recall events to archival, then delete (by IDs, kind, or oldest N).
- `recall_summarize`: consolidate and summarize recall events.

**Memory Management:**
- `consolidate`: trigger memory consolidation and compression.

### Operation format reference

The LLM can include a `<memory_update>` block at the end of its response with
any combination of the following operations:

#### Core Memory (always-visible key-value store)

| Operation | Key | Format | Description |
|-----------|-----|--------|-------------|
| SET | `core` | `{"key": "value"}` | Set key-value pairs in core memory |
| GET | `core_get` | `["key1", "key2"]` | Retrieve values for specified keys |
| DELETE | `core_delete` | `["key1"]` or `"key"` | Delete keys from core memory (evict) |

#### Archival Memory (searchable long-term store)

| Operation | Key | Format | Description |
|-----------|-----|--------|-------------|
| WRITE | `archival` | `[{"text": "...", "tags": ["TAG"]}]` | Write new archival records |
| UPDATE | `archival_update` | `[{"id": "...", "text": "..."}]` | Update existing records by ID |
| SEARCH | `archival_search` | `{"query": "...", "k": 5, "tags": ["TAG"]}` | Search archival memory |

#### Recall Memory (recent events timeline)

| Operation | Key | Format | Description |
|-----------|-----|--------|-------------|
| APPEND | `recall` | `{"kind": "...", "content": "..."}` | Append event to recall |
| SEARCH | `recall_search` | `{"query": "...", "k": 10}` | Search recall events |
| EVICT | `recall_evict` | `{"oldest": N}` or `{"kind": "..."}` or `{"ids": [...]}` | Move recall events to archival, then delete |
| SUMMARIZE | `recall_summarize` | `true` | Consolidate recall events into summaries |

#### Memory Management

| Operation | Key | Format | Description |
|-----------|-----|--------|-------------|
| CONSOLIDATE | `consolidate` | `true` | Trigger memory consolidation/compression |

### Example usage

```xml
<memory_update>
{
  "core": {
    "optimal_threads": "8",
    "best_compiler_flags": "-O3 -march=native"
  },
  "core_get": ["previous_best_time"],
  "archival": [
    {
      "text": "Thread count 8 gives 2x speedup on matrix multiplication workload",
      "tags": ["PERFORMANCE", "THREADING"]
    }
  ],
  "archival_search": {
    "query": "compilation errors",
    "k": 3
  },
  "recall": {
    "kind": "discovery",
    "content": "Found optimal configuration after testing 5 variants"
  }
}
</memory_update>
```

### What to record

**Core memory** (always visible in prompts):
- Optimal parameters discovered (thread counts, batch sizes)
- Best configurations found (compiler flags, environment variables)
- Important constraints or limits identified
- Key metrics or thresholds

**Archival memory** (searchable, long-term):
- Detailed explanations of why something works or fails
- Lessons learned from debugging sessions
- Patterns to avoid or prefer
- Experimental results with context

### What NOT to record

- Temporary debug information
- Information already logged by the system (errors, metrics)
- Obvious or trivial observations
- Large data dumps (use summaries instead)

### Return values and multi-turn flow

Read operations (`core_get`, `archival_search`, `recall_search`) trigger a
**multi-turn flow** when used in split-phase execution.

For a complete flow diagram, see [memory-flow.md](memory-flow.md).

#### How it works

1. LLM outputs `<memory_update>` with read operations + initial JSON
2. System executes ALL operations (writes are persisted, reads return results)
3. If read results exist:
   - Results are formatted as `<memory_results>` block
   - Injected into prompt as "Memory Read Results" section
   - LLM is re-queried to produce final response using the retrieved information
4. Process repeats until no read operations or max rounds reached

#### Configuration

```yaml
memory:
  max_memory_read_rounds: 5  # Maximum re-query cycles (default: 5)
```

#### What happens when max rounds exceeded

If the LLM continues to include read operations after `max_memory_read_rounds`:

- Read operations are still **executed** (results logged)
- Results are **NOT** returned to LLM (no re-query)
- Current artifacts are returned as-is
- This prevents infinite loops while still persisting any write operations

#### Benefits

This allows the LLM to:
- Search for past learnings before making decisions
- Retrieve optimal parameters discovered in previous experiments
- Check archival memory for relevant patterns or errors
- Make informed decisions based on historical data

#### Other return values

Eviction operations (`recall_evict`) return the count of evicted events and the
count of events archived to archival memory (events are moved to archival with
tag `EVICTED_RECALL` before deletion).

Summarization operations (`recall_summarize`) return status and the count of
consolidated events.

### Implementation details

Memory operations are processed by `apply_llm_memory_updates()` in
`memgpt_store.py`. The function:

1. Parses `<memory_update>` blocks from LLM responses
2. Validates the JSON structure
3. Applies operations in order: core → archival → recall → recall_evict → recall_summarize → consolidate
4. Returns results dict containing read operation outputs
5. Logs all operations to `memory_calls.jsonl`

The multi-turn flow is implemented in `generate_phase_artifacts()` in
`parallel_agent.py`:

1. Calls `apply_llm_memory_updates()` and captures results
2. Checks if read results exist via `_has_memory_read_results()`
3. If within round limit, formats results via `_format_memory_results_for_llm()`
4. Injects results as "Memory Read Results" section and re-queries LLM
5. Continues until no read results or max rounds reached

LLM-generated archival entries are automatically tagged with `LLM_INSIGHT` for
identification.
