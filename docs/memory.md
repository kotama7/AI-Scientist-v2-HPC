# MemGPT-Style Memory

The system supports hierarchical memory per branch when `memory.enabled=true`.
It persists essential context across tree search branches without flooding every
prompt.

## Memory overview

![Memory flow](images/memory_flow.png)
<!-- IMAGE_PROMPT:
Create a diagram (16:9) showing three stacked memory layers labeled "Core", "Recall", "Archival" feeding into a "Prompt" box. On the right, draw a small tree of nodes to show branch inheritance, with arrows from parent to child. Add a bold arrow labeled "selected-best promotion" from a child back to a parent node. Use flat vector style, blue/teal/gray palette, white background, sans-serif labels. Title: "Memory flow". -->

## Enabling memory

- Set `memory.enabled=true` in `bfts_config.yaml`, or
- Pass `--enable_memgpt` on the CLI.

The default database path is `experiments/<run>/memory/memory.sqlite` and can be
overridden with `--memory_db`.

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
4. Remove redundant information and verbose explanations.
5. Produce coherent, readable summaries (not keyword fragments).

The template accepts `{max_chars}`, `{current_chars}`, `{context_hint}`, and
`{text}` placeholders.

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
