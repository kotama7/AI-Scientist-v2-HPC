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
- When a node is selected as best, memory can be promoted to parents.

## Persistence hooks

- Phase 0 internal info is captured into
  `experiments/<run>/memory/phase0_internal_info.json|md`, tagged
  `PHASE0_INTERNAL`, and summarized into Core.
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
- `mem_node_promote`: promote child memory to a parent (`selected_best`,
  `resources_update`, `writeup_ready`).
- `mem_resources_index_update`: update the resource index entry for the run.
- `mem_resources_snapshot_upsert`: upsert a resource item snapshot into
  archival memory.
- `mem_resources_resolve_and_refresh`: refresh resource snapshots once
  resources are available.
