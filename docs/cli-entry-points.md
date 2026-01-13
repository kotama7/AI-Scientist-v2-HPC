# CLI Entry Points

This fork exposes a small set of scripts that cover ideation, execution, and
writeup. The launcher is the most common entry point.

## Entry point decision guide

![Entry point decision tree](images/cli_entry_point_decision_tree.png)

## `launch_scientist_bfts.py`

End-to-end pipeline: load an idea from JSON, write `idea.md`/`idea.json`, copy
`bfts_config.yaml` into the run folder, run the BFTS experiment, then optionally
aggregate plots, write a paper, and run the review pass.

Inputs:

- `--load_ideas` path to an idea JSON list.
- `--idea_idx` to choose a specific idea.
- `--singularity_image` for split mode.

Outputs:

- A run directory under `experiments/` with logs, artifacts, and optional
  writeups.

Common flags:

- `--skip_plot`, `--skip_writeup`, `--skip_review` to disable downstream stages.
- `--writable_mode` and `--phase1_max_steps` to control split-mode installs.
- `--use_gpu` to toggle GPU access inside Singularity.
- `--resources` to add mounted data and prompt context.
- `--enable_memgpt` to turn on hierarchical memory.

Example:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 4
```

## `generate_paper.py`

Plot aggregation + writeup + review for an existing experiment directory. Use
this when you already have a run directory and only want plots or a paper.

Key inputs:

- `--experiment-dir` pointing to an existing run.
- `--model-agg-plots` and `--model-writeup` for LLM selection.

Example:

```bash
python generate_paper.py \
  --experiment-dir experiments/<run> \
  --writeup-type icbinb
```

## `ai_scientist/perform_ideation_temp_free.py`

Generate idea JSON from a workshop description Markdown file (optional Semantic
Scholar search). It writes a JSON file next to the workshop file with the same
basename.

Key inputs:

- `--workshop-file` with the description.
- `--model` to control the generator.
- `--max-num-generations`, `--num-reflections` for sampling control.

Example:

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/himeno_benchmark_challenge.md \
  --model gpt-4o-2024-05-13
```

## `ai_scientist/perform_plotting.py`

Plot aggregation only. Writes and runs `auto_plot_aggregator.py` inside the
experiment folder.

Example:

```bash
python ai_scientist/perform_plotting.py \
  --experiment_dir experiments/<run>
```

## `ai_scientist/perform_writeup.py`

Eight-page writeup pipeline. Use for full-length writeups.

## `ai_scientist/perform_icbinb_writeup.py`

Four-page writeup pipeline (shorter format).

## `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py`

Core BFTS run given a config file. This is normally called by
`launch_scientist_bfts.py` rather than invoked directly.

## Other utilities (usually called by the launcher)

- `ai_scientist/perform_llm_review.py`: text-only paper review.
- `ai_scientist/perform_vlm_review.py`: image-caption review (VLM).
- `ai_scientist/perform_plotting.py`: plot aggregation for existing runs.

## Related files

- `bfts_config.yaml` for defaults used by the launcher.
- `prompt/` for system messages and response formats.
- `docs/quickstart.md` for runnable examples.
