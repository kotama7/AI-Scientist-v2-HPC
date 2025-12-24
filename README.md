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

This fork targets HPC environments with a Singularity-based, split-phase execution path. It orchestrates: idea generation/loading → BFTS tree search experiments → plot aggregation → LaTeX writeup → optional PDF review.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Credentials](#credentials)
4. [Main Entry Points](#main-entry-points)
5. [Quickstart](#quickstart)
6. [Configuration](#configuration)
7. [Execution Modes](#execution-modes)
8. [Resource Files](#resource-files)
9. [Outputs](#outputs)
10. [Testing](#testing)
11. [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)

## Requirements

- Linux host (the launch script uses `psutil` for cleanup).
- Python 3.11 for the control plane.
- Singularity CLI (the code invokes `singularity`; Apptainer must be aliased or symlinked).
- GPU + CUDA for the default config (`bfts_config.yaml` uses GPU workers).
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
# (use a CUDA-enabled build for GPU clusters)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Singularity Image

The split-phase path requires a base SIF image. The default config expects
`docker/ai-scientist-worker-nv.sif` (edit `bfts_config.yaml` or pass `--singularity_image`).

The image should include:
- Python 3.11+
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
export HUGGINGFACE_API_KEY="..."      # DeepCoder via Hugging Face API
export OLLAMA_API_KEY="..."           # Optional; local Ollama endpoint
export S2_API_KEY="..."               # Optional; Semantic Scholar
```

## Main Entry Points

- `ai_scientist/perform_ideation_temp_free.py`: generate idea JSON from a workshop description.
- `launch_scientist_bfts.py`: run full pipeline (experiment, plots, writeup, review).
- `ai_scientist/perform_plotting.py`: aggregate plots from an existing run.
- `ai_scientist/perform_writeup.py` / `ai_scientist/perform_icbinb_writeup.py`: generate writeups from a run directory.

## Quickstart

### 1. Generate ideas (optional)

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/i_cant_believe_its_not_better.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 3 \
  --num-reflections 5
```

This writes a JSON file next to the workshop file (same basename, `.json`).

### 2. Run a full experiment

```bash
python launch_scientist_bfts.py \
  --writeup-type icbinb \
  --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json \
  --idea_idx 0 \
  --singularity_image /path/to/ai-scientist-worker-nv.sif \
  --phase_mode split \
  --num_workers 4 \
  --model_writeup o1-preview-2024-09-12 \
  --model_citation gpt-4o-2024-11-20 \
  --model_review gpt-4o-2024-11-20 \
  --model_agg_plots o3-mini-2025-01-31
```

Useful flags:
- `--additional-information <file>`: append extra text to the idea prompt.
- `--skip_writeup` / `--skip_review`: skip later stages.
- `--attempt_id <n>`: disambiguate parallel runs of the same idea.

## Configuration

The default configuration lives in `bfts_config.yaml`. The launcher copies it into each run directory and overrides fields such as `desc_file`, `workspace_dir`, and `log_dir`.

Key sections:

- `exec`
  - `phase_mode`: `split` (default) or `single`.
  - `singularity_image`: path to the base SIF.
  - `language`: default is `cpp` (affects code generation constraints).
  - `writable_mode`: `auto`, `tmpfs`, `overlay`, or `none`.
  - `per_worker_sif`: build a worker SIF per GPU.
  - `phase1_max_steps`: max iterative installer steps.
- `agent`
  - `num_workers`: parallel workers mapped to GPUs.
  - `stages.*`: per-stage max iterations.
  - `code`, `feedback`, `summary`: LLM model choices.
- `report`
  - `model`, `temp`: summary report generation.

## Execution Modes

### Split Mode (`exec.phase_mode=split`)

Runs the experiment as four explicit phases inside Singularity:

1. Download & install (Phase 1)
2. Coding (Phase 2)
3. Compile (Phase 3)
4. Run (Phase 4)

The LLM outputs a structured JSON payload with per-phase artifacts. The run phase must produce `working/experiment_data.npy` inside the container. Per-worker SIFs are built when `per_worker_sif=true`.

Relevant flags in `launch_scientist_bfts.py`:
- `--per_worker_sif`, `--keep_sandbox`, `--use_fakeroot`
- `--writable_mode`, `--phase1_max_steps`
- `--container_overlay`, `--disable_writable_tmpfs`

### Single Mode (`exec.phase_mode=single`)

Uses the legacy flow without split phases. Code executes on the host environment (no container), and package guidance comes from the prompt templates.

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
- Local resources are bind-mounted into containers.
- GitHub and Hugging Face resources are fetched during Phase 1.

## Outputs

Each run creates a directory under `experiments/`:

- `experiments/<timestamp>_<idea>_attempt_<id>/idea.md`
- `experiments/<timestamp>_<idea>_attempt_<id>/idea.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/bfts_config.yaml`
- `experiments/<timestamp>_<idea>_attempt_<id>/logs/`
- `experiments/<timestamp>_<idea>_attempt_<id>/figures/` (plot aggregation output)
- `experiments/<timestamp>_<idea>_attempt_<id>/<run>.pdf` and reflection PDFs (if writeup enabled)
- `experiments/<timestamp>_<idea>_attempt_<id>/review_text.txt` and `review_img_cap_ref.json` (if review enabled)
- `experiments/<timestamp>_<idea>_attempt_<id>/token_tracker.json`
- `experiments/<timestamp>_<idea>_attempt_<id>/token_tracker_interactions.json`

During execution, `experiment_results/` is copied out of logs for plot aggregation and then removed by the launcher.

## Testing

```bash
python -m unittest tests/test_smoke_split.py
python -m unittest tests/test_resource.py
```

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
