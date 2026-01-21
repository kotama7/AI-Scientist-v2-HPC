# Configuration

The default configuration lives in `bfts_config.yaml`. The launcher copies this
file into each experiment directory and overrides per-run fields such as
`desc_file`, `data_dir`, `workspace_dir`, and `log_dir`.

## Annotated sample (excerpt)

```yaml
# Top-level paths
data_dir: "data"                    # Task data directory
log_dir: logs                       # Log output directory
workspace_dir: workspaces           # Experiment workspace
copy_data: True                     # Copy vs symlink data (copy prevents accidental modification)
exp_name: run                       # Experiment name prefix

exec:
  # phase_mode is typically set via --phase_mode CLI flag (default: split)
  singularity_image: /path/to/base.sif
  use_gpu: true                     # enable --nv for Singularity
  workspace_mount: /workspace       # Container mount point
  phase1_max_steps: 12              # Max Phase 1 installer iterations
  log_prompts: true                 # Log prompts as JSON/Markdown

agent:
  type: parallel                    # Agent type (parallel recommended)
  num_workers: 4                    # Parallel workers
  role_description: "HPC Researcher"  # Persona for prompts
  stages:
    stage1_max_iters: 40            # Per-stage iteration limits
    stage2_max_iters: 24
    stage3_max_iters: 24
    stage4_max_iters: 36
  multi_seed_eval:
    num_seeds: 3                    # Seeds for multi-seed evaluation
  search:
    max_debug_depth: 3              # Debug exploration depth
    debug_prob: 0.5                 # Debug branch probability
    num_drafts: 3                   # Number of draft solutions
  code:
    model: gpt-5.2
    temp: 1.0
  feedback:
    model: gpt-5.2
    temp: 0.5
  vlm_feedback:
    model: gpt-5.2
    temp: 0.5
  summary:
    model: gpt-5.2
    temp: 0.3
  select_node:
    model: gpt-5.2
    temp: 0.3

experiment:
  num_syn_datasets: 1               # Number of synthetic datasets
  dataset_source: auto              # local, huggingface, or auto

memory:
  enabled: true
  core_max_chars: 16000
  recall_max_events: 20
  retrieval_k: 8
  use_llm_compression: true
  compression_model: gpt-5.2
  memory_budget_chars: 24000
  section_budgets:
    idea_summary: 9600
    idea_section_limit: 4800
    phase0_summary: 5000
    archival_snippet: 3000
    results: 4000

report:
  model: gpt-5.2
  temp: 1.0
```

## How configuration is applied

1. `launch_scientist_bfts.py` loads `bfts_config.yaml`.
2. It writes a copy into the experiment directory so each run is self-contained.
3. CLI flags override specific fields for that run.

The copy inside the experiment directory is the source of truth for the run.

## Override precedence

1. `bfts_config.yaml` defaults in the repo.
2. CLI flags for the current run (launcher-level overrides).
3. The per-run config copy in `experiments/<run>/bfts_config.yaml`.

## Key sections

### Top-level paths

- `data_dir`: directory containing task data (default `data`).
- `log_dir`: directory for log output (default `logs`).
- `workspace_dir`: experiment workspace directory (default `workspaces`).
- `copy_data`: if true, copy data to workspace; if false, use symlinks (default true for safety).
- `exp_name`: experiment name prefix (default `run`).
- `generate_report`: enable report generation (default true).

### `exec`

- `singularity_image`: path to the base SIF (absolute in the checked-in config;
  override for your system).
- `use_gpu`: enable GPU access in Singularity (`--nv`); set false for CPU-only containers.
- `workspace_mount`: container mount point (default `/workspace`).
- `writable_tmpfs`, `container_overlay`, `writable_mode`: Phase 1 write access.
- `container_extra_args`: extra Singularity args for instance start.
- `per_worker_sif`, `keep_sandbox`, `use_fakeroot`: per-worker SIF behavior.
- `phase1_max_steps`: max iterative installer steps.
- `log_prompts`: write prompt logs (JSON + Markdown) for split/single runs.
- `resources`: optional path to a JSON/YAML resource file.

### `agent`

- `type`: agent type (default `parallel` for parallel execution).
- `num_workers`: parallel workers mapped to GPUs.
- `role_description`: persona description injected into prompts (e.g., "HPC Researcher").
  Replaces placeholders like `{persona}` and "AI researcher" in prompt templates.
- `steps`: fallback iteration count when stage-specific limits are not set.
- `stages.*`: per-stage max iterations:
  - `stage1_max_iters`: Stage 1 (initial exploration) iterations.
  - `stage2_max_iters`: Stage 2 iterations.
  - `stage3_max_iters`: Stage 3 iterations.
  - `stage4_max_iters`: Stage 4 (final refinement) iterations.
- `multi_seed_eval`: multi-seed evaluation settings:
  - `num_seeds`: number of seeds for evaluation (default 3 when num_workers >= 3).
- `search`: tree search parameters:
  - `max_debug_depth`: maximum debug exploration depth.
  - `debug_prob`: probability of debug branch exploration.
  - `num_drafts`: number of draft solutions to generate.
- `code`: LLM settings for code generation (`model`, `temp`).
- `feedback`: LLM settings for program output/traceback evaluation.
- `vlm_feedback`: VLM settings for visual feedback analysis.
- `summary`: LLM settings for result summary generation.
- `select_node`: LLM settings for best node selection.

### `experiment`

- `num_syn_datasets`: number of synthetic datasets to generate (default 1).
- `dataset_source`: where to source datasets (`local`, `huggingface`, or `auto` for LLM decision).

### `report`

- `model`, `temp`: LLM settings for final report generation.

### `memory`

Core settings:

- `enabled`: toggle memory (default false).
- `db_path`: SQLite path (default under `experiments/<run>/memory/`).
- `core_max_chars`, `recall_max_events`, `retrieval_k`: injection limits.
- `use_fts`: full-text search mode (`auto`, `true`, or `false`).

Persistence toggles:

- `persist_phase0_internal`: persist Phase 0 internal state.
- `always_inject_phase0_summary`: always inject Phase 0 summary into prompts.
- `persist_idea_md`: persist idea markdown.
- `always_inject_idea_summary`: always inject idea summary into prompts.
- `final_memory_enabled`: write `final_memory_for_paper.*` at run end.
- `final_memory_filename_md`, `final_memory_filename_json`: output file names.
- `redact_secrets`: mask sensitive information in memory outputs.

LLM compression (for intelligent memory truncation):

- `use_llm_compression`: enable LLM-based compression (default true).
- `compression_model`: model for compression (default `gpt-5.2`).
- `memory_budget_chars`: overall memory budget in characters.
- Per-section character budgets for compression:
  - `datasets_tested_budget_chars`: budget for tested datasets summary.
  - `metrics_extraction_budget_chars`: budget for metrics extraction.
  - `plotting_code_budget_chars`: budget for plotting code summary.
  - `plot_selection_budget_chars`: budget for plot selection summary.
  - `vlm_analysis_budget_chars`: budget for VLM analysis summary.
  - `node_summary_budget_chars`: budget for node summaries.
  - `parse_metrics_budget_chars`: budget for parsed metrics.
- `section_budgets.*`: named section limits for compression:
  - `idea_summary`: compressed research idea.
  - `idea_section_limit`: per-section limit for idea summary bullets.
  - `phase0_summary`: Phase 0 configuration summary.
  - `archival_snippet`: archival memory excerpts.
  - `results`: result summaries.

Memory logging (for debugging):

- `memory_log_enabled`: write memory event logs (default true).
- `memory_log_dir`: directory for memory logs (default under `experiments/<run>/memory/`).
- `memory_log_max_chars`: max chars per log entry.

Writeup memory limits (for `final_writeup_memory.json` generation):

- `writeup_recall_limit`: maximum number of recall memory entries (default 10).
- `writeup_archival_limit`: maximum number of archival memory entries (default 10).
- `writeup_core_value_max_chars`: max characters per core memory value (default 5000).
- `writeup_recall_text_max_chars`: max characters per recall memory text (default 3000).
- `writeup_archival_text_max_chars`: max characters per archival memory text (default 4000).


## Paths and run layout

The launcher rewrites the following per run:

- `desc_file`: the idea markdown for the run.
- `data_dir`: output directory for experiment data.
- `workspace_dir`: per-run working directory.
- `log_dir`: where logs and tree artifacts are stored.

Check the run-local `bfts_config.yaml` in `experiments/<run>/` to confirm the
final values.

## Common CLI overrides

Execution and container settings:

- `--singularity_image`: override `exec.singularity_image`.
- `--use_gpu`: set `exec.use_gpu` (use `--use_gpu false` to disable).
- `--num_workers`: override `agent.num_workers`.
- `--writable_mode`: override `exec.writable_mode` (`auto`, `tmpfs`, `overlay`, `none`).
- `--phase1_max_steps`: override `exec.phase1_max_steps`.
- `--phase_mode`: execution mode (`split` or `single`).
- `--container_overlay`: writable overlay image path for apt-get inside Singularity.
- `--disable_writable_tmpfs`: disable `--writable-tmpfs` for container instances.
- `--per_worker_sif`: whether to build per-worker SIFs (default true).
- `--keep_sandbox`: keep sandbox directories after building worker SIFs.
- `--use_fakeroot`: use `--fakeroot` for Singularity build/exec.
- `--resources`: set `exec.resources` to a JSON/YAML resource file.

Idea and experiment settings:

- `--load_ideas`: path to JSON file with pregenerated ideas.
- `--idea_idx`: index of idea to run from the ideas file.
- `--additional-information`: path to text file with supplementary information to append to idea.
- `--attempt_id`: attempt ID for distinguishing parallel runs of the same idea.

Memory settings:

- `--enable_memgpt`: enable `memory.enabled`.
- `--memory_db`: override `memory.db_path`.
- `--memory_core_max_chars`, `--memory_recall_max_events`,
  `--memory_retrieval_k`: tune memory injection limits.
- `--memory_max_compression_iterations`: max iterative compression attempts (default 3).

Writeup and review settings:

- `--writeup-type`: writeup format (`normal` or `auto`).
- `--writeup-page-limit`: page limit for normal writeups (0 to disable).
- `--writeup-retries`: number of writeup attempts (default 3).
- `--writeup-reflections`: number of reflection steps during writeup (default 3).
- `--skip_writeup`, `--skip_review`, `--skip_plot`: skip downstream stages.

Model selection:

- `--model_agg_plots`: model for plot aggregation.
- `--model_agg_plots_ref`: number of reflections for plot aggregation.
- `--model_writeup`: model for paper writeup.
- `--model_writeup_small`: smaller model for writeup tasks.
- `--model_citation`: model for citation gathering.
- `--num_cite_rounds`: number of citation rounds.
- `--model_review`: model for paper review.

Run `python launch_scientist_bfts.py --help` for the full list of flags.
