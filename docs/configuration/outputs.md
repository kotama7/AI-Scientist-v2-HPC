# Outputs

Each run creates a directory under `experiments/` that captures inputs, logs,
artifacts, and optional memory snapshots.

## Core run artifacts

- `experiments/<timestamp>_<idea>_attempt_<id>/idea.md`
- `experiments/<timestamp>_<idea>_attempt_<id>/idea.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/bfts_config.yaml`

## Logs and workspace

- `experiments/<timestamp>_<idea>_attempt_<id>/logs/<index>-<exp_name>/`
  (stage journals, configs, tree plots, `manager.pkl`)
- `experiments/<timestamp>_<idea>_attempt_<id>/logs/<index>-<exp_name>/unified_tree_viz.html`
  (multi-stage tree viewer; loads per-stage `tree_data.json`)
- `experiments/<timestamp>_<idea>_attempt_<id>/logs/<index>-<exp_name>/stage_*/tree_data.json`
  (tree data with `stage_dir_map` and `log_dir_path` metadata for the viewer)
- `experiments/<timestamp>_<idea>_attempt_<id>/logs/<index>-<exp_name>/phase_logs/node_<id>/prompt_logs/`
  (system prompt logs per node)
- `experiments/<timestamp>_<idea>_attempt_<id>/<index>-<exp_name>/`
  (workspace with `input/` and `working/`)

## Node logs and stage best (workspace inheritance)

Node workspaces are persisted for cross-stage file inheritance:

- `experiments/<timestamp>_<idea>_attempt_<id>/<index>-<exp_name>/node_logs/node_<id>/`
  (temporary: node workspace copied after processing, deleted after stage completion)
- `experiments/<timestamp>_<idea>_attempt_<id>/<index>-<exp_name>/stage_best/<stage_name>/`
  (permanent: best node's workspace preserved for next stage inheritance)

Structure within each node log or stage best directory:
```
node_logs/node_<id>/
├── src/           # Source code
├── bin/           # Compiled binaries
├── input/         # Input data (excluding mounted data)
└── working/       # Working directory with experiment outputs
```

These directories enable safe workspace inheritance without race conditions
between parallel workers. See [../architecture/execution-flow.md](../architecture/execution-flow.md#workspace-inheritance) for details.

## Plotting and writeups

- `experiments/<timestamp>_<idea>_attempt_<id>/figures/` (plot aggregation output)
- `experiments/<timestamp>_<idea>_attempt_<id>/auto_plot_aggregator.py`
- `experiments/<timestamp>_<idea>_attempt_<id>/<experiment_dir_basename>.pdf`
  and reflection PDFs (if writeup enabled)
- `experiments/<timestamp>_<idea>_attempt_<id>/review_text.txt` and
  `review_img_cap_ref.json` (if review enabled; the launcher only runs review if
  writeup ran)

## Token tracking

- `experiments/<timestamp>_<idea>_attempt_<id>/token_tracker.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/token_tracker_interactions.json`

## Memory (when enabled)

- `experiments/<timestamp>_<idea>_attempt_<id>/memory/memory.sqlite`
- `experiments/<timestamp>_<idea>_attempt_<id>/memory/resource_snapshot.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/memory/final_memory-for-paper.md`
- `experiments/<timestamp>_<idea>_attempt_<id>/memory/final_memory_for_paper.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/memory/final_writeup_memory.json`

## Temporary experiment results

During execution, `experiment_results/` is copied out of
`logs/<index>-<exp_name>/` for plot aggregation and then removed by the launcher
unless you skip plotting. Prompt logs are also copied into
`experiment_results/.../llm_outputs/prompt_logs/`.

## Where to look first

- Failures during install/compile/run: check `logs/<index>-<exp_name>/` and
  `phase_logs/node_<id>/`.
- Plotting issues: inspect `auto_plot_aggregator.py` and the `figures/` output.
- Memory or resource injection: inspect `memory/resource_snapshot.json`.

## Example tree (trimmed)

```text
experiments/2024-09-12_himeno_attempt_0/
  idea.md
  idea.json
  bfts_config.yaml
  logs/0-run/
    unified_tree_viz.html
    phase_logs/
  0-run/
    worker_0/              # Temporary worker workspace
    worker_1/
    node_logs/             # Temporary node workspaces (cleaned after stage)
      node_abc123/
        src/
        bin/
        working/
    stage_best/            # Permanent best node workspaces
      1_creative_research_1_first_attempt/
      2_hyperparam_tuning/
  figures/
  memory/
  2024-09-12_himeno_attempt_0.pdf
```
