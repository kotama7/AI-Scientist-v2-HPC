# Configuration

The default configuration lives in `bfts_config.yaml`. The launcher copies this
file into each experiment directory and overrides per-run fields such as
`desc_file`, `data_dir`, `workspace_dir`, and `log_dir`.

## Annotated sample (excerpt)

```yaml
exec:
  phase_mode: split          # split or single
  singularity_image: /path/to/base.sif
  use_gpu: true              # enable --nv for Singularity
  language: cpp              # affects codegen constraints
  writable_mode: auto        # auto/tmpfs/overlay/none

agent:
  num_workers: 4
  code:
    model: gpt-5.2
    temp: 1.0

memory:
  enabled: true
  core_max_chars: 10000
  retrieval_k: 8
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

### `exec`

- `phase_mode`: `split` (default) or `single`.
- `singularity_image`: path to the base SIF (absolute in the checked-in config;
  override for your system).
- `use_gpu`: enable GPU access in Singularity (`--nv`); set false for CPU-only containers.
- `language`: default is `cpp` (affects code generation constraints).
- `workspace_mount`: container mount point (default `/workspace`).
- `writable_tmpfs`, `container_overlay`, `writable_mode`: Phase 1 write access.
- `container_extra_args`: extra Singularity args for instance start.
- `per_worker_sif`, `keep_sandbox`, `use_fakeroot`: per-worker SIF behavior.
- `phase1_max_steps`: max iterative installer steps.
- `log_prompts`: write prompt logs (JSON + Markdown) for split/single runs.
- `resources`: optional path to a JSON/YAML resource file.

### `agent`

- `num_workers`: parallel workers mapped to GPUs.
- `stages.*`: per-stage max iterations.
- `code`, `feedback`, `summary`, `select_node`: LLM model choices.
- `seed`, `temperature`, and other model controls live alongside model names.

### `report`

- `model`, `temp`: summary report generation.

### `memory`

- `enabled`: toggle memory (default false).
- `db_path`: SQLite path (default under `experiments/<run>/memory/`).
- `core_max_chars`, `recall_max_events`, `retrieval_k`: injection limits.
- `persist_phase0_internal`, `persist_idea_md`: persistence toggles.
- `final_memory_enabled`: write `final_memory_for_paper.*` at run end.

## Paths and run layout

The launcher rewrites the following per run:

- `desc_file`: the idea markdown for the run.
- `data_dir`: output directory for experiment data.
- `workspace_dir`: per-run working directory.
- `log_dir`: where logs and tree artifacts are stored.

Check the run-local `bfts_config.yaml` in `experiments/<run>/` to confirm the
final values.

## Common CLI overrides

- `--singularity_image`: override `exec.singularity_image`.
- `--use_gpu`: set `exec.use_gpu` (use `--use_gpu false` to disable).
- `--num_workers`: override `agent.num_workers`.
- `--writable_mode`: override `exec.writable_mode`.
- `--phase1_max_steps`: override `exec.phase1_max_steps`.
- `--resources`: set `exec.resources` to a JSON/YAML resource file.
- `--enable_memgpt`: enable `memory.enabled`.
- `--memory_db`: override `memory.db_path`.
- `--memory_core_max_chars`, `--memory_recall_max_events`,
  `--memory_retrieval_k`: tune memory injection limits.

Run `python launch_scientist_bfts.py --help` for the full list of flags.
