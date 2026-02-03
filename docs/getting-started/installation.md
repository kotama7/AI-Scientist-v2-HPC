# Installation

This page covers the host-side control plane setup. The worker runtime is
provided by the Singularity image (see the final section).

## Repository layout and environment

- The launcher sets `AI_SCIENTIST_ROOT` automatically to the repo root when you
  run `launch_scientist_bfts.py` or `generate_paper.py`.
- If you call lower-level scripts directly, export it yourself:

```bash
export AI_SCIENTIST_ROOT="$(pwd)"
```

## Choose an environment manager

Conda is recommended for HPC environments, but a virtualenv also works.

### Conda example

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
```

### Virtualenv example

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Python dependencies

```bash
# Host-side requirements (control plane)
pip install -r requirements.txt
pip install psutil
```

The requirements file includes LLM API clients, plotting libraries, and
tree-search utilities. Keep the host environment separate from any container
environment so you can iterate on control-plane code without rebuilding images.

## Install Torch (for GPU detection)

Use a CUDA-enabled build on GPU clusters and adjust for your CUDA version:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c 
```

## Optional: YAML resource files

Install `pyyaml` only if you plan to use YAML resource files:

```bash
pip install pyyaml
```

## PDF

```bash
# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex
```

## Prepare the Singularity image

Split-phase execution needs a base SIF image. The default config points at
`exec.singularity_image` in `bfts_config.yaml`, but you can override it with
`--singularity_image` on the CLI.

A minimal path to build/pull an image lives in `template/README.md`, which
documents pulling a base image into `template/base.sif`.

If your cluster requires explicit module loads, do that before building or
pulling the image (CUDA, compiler toolchains, and Singularity).

## Sanity checks

- `singularity --version`
- `python -c "import torch; print(torch.cuda.is_available())"`
- `python -c "import psutil; print(psutil.__version__)"`
