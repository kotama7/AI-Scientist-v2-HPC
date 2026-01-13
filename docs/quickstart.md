# Quickstart

Note: the defaults in `perform_ideation_temp_free.py` and `launch_scientist_bfts.py`
point to files that are not present in this fork. Pass explicit paths as shown.

## 0. Prerequisites

- Install host dependencies (see `docs/installation.md`).
- Ensure `singularity` is available if you plan to use split mode.
- Export the API keys you need (see `docs/credentials.md`).

## 1. Prepare a base SIF image

Split-phase runs require a base Singularity image. The default config reads
`exec.singularity_image` from `bfts_config.yaml`, but you can override it with
`--singularity_image` on the CLI.

See `template/README.md` for a minimal workflow that pulls an image to
`template/base.sif`.

## 2. Generate ideas (optional)

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/himeno_benchmark_challenge.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 3 \
  --num-reflections 5
```

This writes a JSON file next to the workshop file (same basename, `.json`).

If you already have an ideas JSON file, you can skip this step and move to the
experiment launch.

## 3. Run a full experiment

```bash
python launch_scientist_bfts.py \
  --writeup-type icbinb \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 4
```

The launcher creates a run directory under `experiments/` and writes `idea.md`,
`idea.json`, and a run-specific `bfts_config.yaml` before starting the tree
search.

Useful flags:

- `--additional-information <file>`: append extra text to the idea prompt.
- `--skip_plot` / `--skip_writeup` / `--skip_review`: skip later stages.
- `--attempt_id <n>`: disambiguate parallel runs of the same idea.
- `--writable_mode {auto,tmpfs,overlay,none}`: control Phase 1 writable behavior.
- `--resources <file>`: pass a resources JSON/YAML file (see
  `docs/resource-files.md` and `data_resources.json` for an example).
- `--enable_memgpt`: enable hierarchical memory.
- `--memory_db <path>`: override SQLite path (defaults to
  `experiments/<run>/memory/memory.sqlite`).
- `--memory_core_max_chars`, `--memory_recall_max_events`,
  `--memory_retrieval_k`: tune memory injection limits.

Expected runtime varies by model, number of workers, and writeup settings.
For a first validation run, consider reducing `--num_workers` and skipping
plot/writeup steps.

## 4. Minimal run (no plots/writeup)

If you only want to validate split execution and skip writeup dependencies:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 2 \
  --skip_plot --skip_writeup --skip_review
```

## 5. Generate plots/writeup for an existing experiment

```bash
python generate_paper.py \
  --experiment-dir experiments/<timestamp>_<idea>_attempt_<id> \
  --writeup-type icbinb \
  --model-agg-plots o3-mini-2025-01-31 \
  --model-writeup o1-preview-2024-09-12
```

Use this when you already have a run directory and want to regenerate plots or
writeups without re-running the experiment.

## 6. Verify outputs

After a successful run, check:

- `experiments/<run>/logs/` for stage logs and tree visualization.
- `experiments/<run>/figures/` for aggregated plots (if enabled).
- `experiments/<run>/<run>.pdf` for the generated paper (if enabled).

See `docs/outputs.md` for the full directory layout.

## What "success" looks like

- Logs show successful Phase 1 install (split mode) and Phase 2-4 execution.
- `experiments/<run>/logs/.../unified_tree_viz.html` renders the search tree.
- `experiments/<run>/figures/` includes plots (if enabled).
- `experiments/<run>/<run>.pdf` exists when writeup is enabled.
