# Outputs

Each run creates a directory under `experiments/` that captures inputs, logs,
artifacts, and optional memory snapshots. The run directory is also the
workspace (`cfg.workspace_dir`), while `cfg.log_dir` points to a `logs/`
subdirectory inside the run.

## Core run artifacts

- `experiments/<timestamp>_<idea>_attempt_<id>/idea.md`
- `experiments/<timestamp>_<idea>_attempt_<id>/idea.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/bfts_config.yaml`

## Log directory (cfg.log_dir)

Top-level files in `experiments/<run>/logs/`:

- `unified_tree_viz.html` (multi-stage tree viewer)
- `memory_database.html` (memory viewer; generated only if memory is enabled and
  `memory.sqlite` exists)
- `manager.pkl` (serialized AgentManager / journals)
- `draft_summary.json`, `baseline_summary.json`, `research_summary.json`,
  `ablation_summary.json` (when `generate_report=true`)

Per-stage logs live under stage directories:

- `experiments/<run>/logs/stage_<stage_name>/`
  - `journal.json`
  - `config.yaml`
  - `tree_plot.html`
  - `tree_data.json`
  - `best_solution_<node_id>.<ext>` + `best_node_id.txt`
  - `notes/` (stage summaries, transitions)

Per-node phase logs:

- `experiments/<run>/logs/phase_logs/node_<id>/`
  - `download.log`, `coding.log`, `compile.log`, `run.log`
  - `prompt_logs/` (copied prompt/response artifacts when available)

Prompt logs (when `exec.log_prompts=true`):

- `experiments/<run>/logs/prompt_logs/<worker_label>/<session_id>/`
  - `*_prompt.json` / `*_prompt.md`
  - `*_response.json` / `*_response.txt`
  - `memory_operations.jsonl`, `memory_injections.jsonl` (memory-enabled runs)

Memory logging (when `memory.memory_log_enabled=true`):

- `experiments/<run>/logs/memory/`
  - memory operation and injection logs (`*.jsonl`)

## Workspace directory (cfg.workspace_dir)

The run directory itself is the workspace:

- `experiments/<run>/data/` (copied or linked input data)
- `experiments/<run>/worker_<id>/` (active worker workspace)
- `experiments/<run>/node_logs/node_<id>/` (temporary node snapshots; cleaned
  after stage completion)
- `experiments/<run>/stage_best/<stage_name>/` (best node workspace for
  inheritance)
- `experiments/<run>/resources/<class>/<name>/` (staged templates/docs when
  using resource files)

Structure within a node workspace (active or saved):

```text
node_logs/node_<id>/
├── src/           # Source code
├── bin/           # Compiled binaries
├── input/         # Input data (excluding mounted data)
└── working/       # Working directory with experiment outputs
    └── {experiment_name}_data.npy  # Experiment results (dynamic filename)
```

## Experiment results (plots/data snapshots)

Per-node results are stored outside the run directory in a shared log root:

- `experiments/logs/<run>/experiment_results/experiment_<node_id>_proc_<pid>/`
  - `experiment_code.*`, `plotting_code.*`, `.npy` outputs, generated plots
- `experiments/logs/<run>/experiment_results/seed_aggregation_<node_id>/`
  - aggregated plots for multi-seed evaluation

These paths are referenced by the HTML visualizers using relative URLs.

## Plotting and writeups

- `experiments/<run>/figures/` (plot aggregation output)
- `experiments/<run>/auto_plot_aggregator.py`
- `experiments/<run>/<run>.pdf` and reflection PDFs (if writeup enabled)
- `experiments/<run>/review_text.txt` and `review_img_cap_ref.json` (if review enabled)

## Token tracking

- `experiments/<run>/token_tracker.json`
- `experiments/<run>/token_tracker_interactions.json`

## Memory (when enabled)

- `experiments/<run>/memory/memory.sqlite`
- `experiments/<run>/memory/resource_snapshot.json` (only if a snapshot was created)
- `experiments/<run>/memory/final_memory_for_paper.md` (used for paper generation)

## Split-mode runtime assets (runs/)

When `exec.phase_mode=split`, per-worker runtime assets live under:

- `experiments/<run>/runs/workers/worker-*/container/` (SIF/sandbox files)
- `experiments/<run>/runs/workers/worker-*/plans/` (phase0 plan/history)
- `experiments/<run>/runs/workers/worker-*/prompt_logs/` (phase0 + memory ops)
- `experiments/<run>/runs/workers/worker-*/phase1_steps.jsonl`
- `experiments/<run>/runs/workers/worker-*/phase1_llm_outputs.jsonl`

You can override this root by setting `AI_SCIENTIST_RUN_ROOT`.

## Where to look first

- Failures during install/compile/run: check `logs/phase_logs/node_<id>/`.
- Plotting issues: inspect `auto_plot_aggregator.py` and `figures/`.
- Memory or resource injection: inspect `memory/` and `logs/memory/`.

## Example tree (trimmed)

```text
experiments/2026-01-30_himeno_attempt_0/
  idea.md
  idea.json
  bfts_config.yaml
  logs/
    unified_tree_viz.html
    memory_database.html
    manager.pkl
    stage_1_initial_implementation_1_preliminary/
      journal.json
      tree_plot.html
      tree_data.json
      best_node_id.txt
      notes/
    phase_logs/
      node_abc123/
        download.log
        prompt_logs/
  node_logs/
    node_abc123/
      src/
      working/
  stage_best/
    1_initial_implementation_1_preliminary/
  figures/
  memory/
  runs/
  2026-01-30_himeno_attempt_0.pdf
```
