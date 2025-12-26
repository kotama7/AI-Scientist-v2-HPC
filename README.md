<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2 (HPC Fork)</b><br>
    <b>Split-Phase Execution with Singularity</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p>

This fork targets HPC environments with a Singularity-based, split-phase execution path. It orchestrates: idea loading/generation → BFTS tree search experiments → plot aggregation → LaTeX writeup → optional PDF review.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Credentials](#credentials)
4. [CLI Entry Points](#cli-entry-points)
5. [Quickstart](#quickstart)
6. [Configuration](#configuration)
7. [Execution Modes](#execution-modes)
8. [Resource Files](#resource-files)
9. [Outputs](#outputs)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)
12. [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)

## Requirements

- Linux host (the launcher uses `psutil` for cleanup).
- Python 3.10+ for the control plane.
- Singularity CLI (the code invokes `singularity`; Apptainer must be aliased or symlinked).
- Torch on the host (imported by the launcher for GPU detection).
- GPU + CUDA recommended (default config uses GPU workers and maps workers to GPU IDs).
- LaTeX toolchain for writeups: `pdflatex`, `bibtex`, `chktex`.
- `pdftotext` (from poppler) for PDF checks and reviews.

## Installation

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Host-side requirements (control plane)
pip install -r requirements.txt
pip install psutil

# Torch is imported by the launcher for GPU detection
# (use a CUDA-enabled build for GPU clusters; adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Optional (only if you use YAML resource files):

```bash
pip install pyyaml
```

### Singularity Image

The split-phase path requires a base SIF image. The default config points to
`exec.singularity_image` in `bfts_config.yaml` (override with `--singularity_image`).
See `template/README.md` for a minimal way to pull a base image into `template/base.sif`.

The image should include:
- Python 3.10+
- CUDA toolkit
- Build tools (gcc, make, cmake)
- Git
- Any extra libs you want available during Phase 1 installs

If you plan to use Hugging Face resources, ensure `huggingface_hub` is installed inside the worker image.

## Credentials

Set only the variables needed for the models you use:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."        # Claude models
export GEMINI_API_KEY="..."           # Gemini models (OpenAI-compatible endpoint)
export OPENROUTER_API_KEY="..."       # Llama 3.1 via OpenRouter
export DEEPSEEK_API_KEY="..."         # deepseek-coder-v2-0724
export HUGGINGFACE_API_KEY="..."      # deepcoder-14b via Hugging Face API
export OLLAMA_API_KEY="..."           # Optional; local Ollama endpoint
export S2_API_KEY="..."               # Optional; Semantic Scholar
```

## CLI Entry Points

- `launch_scientist_bfts.py`: end-to-end pipeline. Loads a single idea from a JSON list, writes `idea.md`/`idea.json`, copies `bfts_config.yaml` into the run folder, launches the BFTS experiment, then optionally aggregates plots, writes a paper, and runs the review pass.
- `generate_paper.py`: plots + writeup + review for an existing experiment directory.
- `ai_scientist/perform_ideation_temp_free.py`: generate idea JSON from a workshop description Markdown file (with optional Semantic Scholar search).
- `ai_scientist/perform_plotting.py`: plot aggregation only (writes and runs `auto_plot_aggregator.py` in the experiment folder).
- `ai_scientist/perform_writeup.py`: 8-page writeup pipeline.
- `ai_scientist/perform_icbinb_writeup.py`: 4-page writeup pipeline.
- `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py`: core BFTS run given a config (normally called by `launch_scientist_bfts.py`).

## Quickstart

Note: the defaults in `perform_ideation_temp_free.py` and `launch_scientist_bfts.py` point to files that are not present in this fork. Pass explicit paths as shown below.

### 1. Generate ideas (optional)

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/himeno_benchmark_challenge.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 3 \
  --num-reflections 5
```

This writes a JSON file next to the workshop file (same basename, `.json`).

### 2. Run a full experiment

```bash
python launch_scientist_bfts.py \
  --writeup-type icbinb \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 4
```

Useful flags:
- `--additional-information <file>`: append extra text to the idea prompt.
- `--skip_plot` / `--skip_writeup` / `--skip_review`: skip later stages.
- `--attempt_id <n>`: disambiguate parallel runs of the same idea.
- `--writable_mode {auto,tmpfs,overlay,none}`: control Phase 1 writable behavior.
- `--resources <file>`: pass a resources JSON/YAML file (see below).

### 3. Generate plots/writeup for an existing experiment

```bash
python generate_paper.py \
  --experiment-dir experiments/<timestamp>_<idea>_attempt_<id> \
  --writeup-type icbinb \
  --model-agg-plots o3-mini-2025-01-31 \
  --model-writeup o1-preview-2024-09-12
```

## Configuration

The default configuration lives in `bfts_config.yaml`. The launcher copies it into each run directory and overrides fields such as `desc_file`, `data_dir`, `workspace_dir`, and `log_dir`.

Key sections (defaults from `bfts_config.yaml`):

- `exec`
  - `phase_mode`: `split` (default) or `single`.
  - `singularity_image`: path to the base SIF (absolute in the checked-in config; override for your system).
  - `language`: default is `cpp` (affects code generation constraints).
  - `workspace_mount`: container mount point (default `/workspace`).
  - `writable_tmpfs`, `container_overlay`, `writable_mode`: control Phase 1 write access.
  - `container_extra_args`: extra Singularity args for instance start.
  - `per_worker_sif`, `keep_sandbox`, `use_fakeroot`: per-worker SIF behavior.
  - `phase1_max_steps`: max iterative installer steps.
  - `resources`: optional path to a JSON/YAML resource file.
- `agent`
  - `num_workers`: parallel workers mapped to GPUs.
  - `stages.*`: per-stage max iterations.
  - `code`, `feedback`, `summary`, `select_node`: LLM model choices.
- `report`
  - `model`, `temp`: summary report generation.

## Execution Modes

### Split Mode (`exec.phase_mode=split`)

Runs the experiment as four explicit phases inside Singularity:

0. Planning (Phase 0, prompt-only; produces the phase plan)
1. Download & install (Phase 1)
2. Coding (Phase 2)
3. Compile (Phase 3)
4. Run (Phase 4)

The LLM outputs a structured JSON payload with per-phase artifacts. The run phase must produce `working/experiment_data.npy` inside the container by default (the expected outputs can be overridden by the plan). Per-worker SIFs are built when `per_worker_sif=true` under `experiments/<...>/workers/worker-*/container/`.

Relevant flags in `launch_scientist_bfts.py`:
- `--per_worker_sif`, `--keep_sandbox`, `--use_fakeroot`
- `--writable_mode`, `--phase1_max_steps`
- `--container_overlay`, `--disable_writable_tmpfs`
- `--singularity_image` (required for split mode)

### Single Mode (`exec.phase_mode=single`)

Uses the legacy flow without split phases. Code executes on the host environment (no container), and package guidance comes from the prompt templates. Resource binds from `--resources` are only applied in split mode.

## Resource Files

You can supply a JSON/YAML resource file with `--resources`. The file supports:

```json
{
  "local": [
    {
      "name": "input_data",
      "host_path": "/shared/datasets/my_data",
      "mount_path": "/workspace/input/data",
      "read_only": true
    }
  ],
  "github": [
    {
      "name": "cnpy",
      "repo": "https://github.com/rogersce/cnpy.git",
      "ref": "v1.0.0",
      "dest": "/workspace/third_party/cnpy",
      "as": "library"
    }
  ],
  "huggingface": [
    {
      "name": "my_model",
      "type": "model",
      "repo_id": "org/model-name",
      "revision": "abc123def456...",
      "dest": "/workspace/input/my_model"
    }
  ]
}
```

Notes:
- `mount_path`/`dest` must be under `/workspace`.
- `host_path` must exist for local resources.
- Local resources are bind-mounted into containers.
- GitHub/Hugging Face resources are not auto-fetched; Phase 1 is instructed to `git clone` / `huggingface_hub` download to the `dest` paths.
- YAML resource files require `pyyaml` on the host.

## Outputs

Each run creates a directory under `experiments/`:

- `experiments/<timestamp>_<idea>_attempt_<id>/idea.md`
- `experiments/<timestamp>_<idea>_attempt_<id>/idea.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/bfts_config.yaml`
- `experiments/<timestamp>_<idea>_attempt_<id>/logs/<index>-<exp_name>/` (stage journals, configs, tree plots, `manager.pkl`)
- `experiments/<timestamp>_<idea>_attempt_<id>/<index>-<exp_name>/` (workspace with `input/` and `working/`)
- `experiments/<timestamp>_<idea>_attempt_<id>/figures/` (plot aggregation output)
- `experiments/<timestamp>_<idea>_attempt_<id>/auto_plot_aggregator.py`
- `experiments/<timestamp>_<idea>_attempt_<id>/<experiment_dir_basename>.pdf` and reflection PDFs (if writeup enabled)
- `experiments/<timestamp>_<idea>_attempt_<id>/review_text.txt` and `review_img_cap_ref.json` (if review enabled; `launch_scientist_bfts.py` only runs review if writeup ran)
- `experiments/<timestamp>_<idea>_attempt_<id>/token_tracker.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/token_tracker_interactions.json`

During execution, `experiment_results/` is copied out of `logs/<index>-<exp_name>/` for plot aggregation and then removed by the launcher (unless you skip plotting).

## Testing

```bash
python -m unittest tests/test_smoke_split.py
python -m unittest tests/test_resource.py
```

## Troubleshooting

- Split mode fails with "Singularity image is required": pass `--singularity_image` or update `exec.singularity_image` in `bfts_config.yaml`.
- Phase 1 cannot write inside the container: try `--writable_mode overlay` with `--container_overlay /path/to/overlay.img`, or disable tmpfs via `--disable_writable_tmpfs`.
- `singularity build` fails due to permissions: try `--use_fakeroot false`.
- Resource validation errors: ensure `mount_path`/`dest` are under `/workspace` and local `host_path` exists.
- Hugging Face downloads fail: install `huggingface_hub` inside the worker image and provide `HUGGINGFACE_API_KEY` if needed.

## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Acknowledgement

The tree search component is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. This HPC fork extends the original [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) with split-phase execution and Singularity container support.
