# Singularity base image template

The split-phase worker containers start from a shared base image. This file
documents where to place it and what it should contain so Phase 1 installs and
Phase 2-4 execution work reliably.

## Where the image lives

Place the shared base image at `template/base.sif`. The launcher references it
via `exec.singularity_image` in `bfts_config.yaml` or `--singularity_image` on
the CLI.

## Suggested contents

Include the minimum tools needed for Phase 1 installs and Phase 2-4 execution:

- Python 3.10+
- Git, curl, and build tools (`gcc`, `make`, `cmake`)
- CUDA runtime (if you plan to use GPUs)
- `pip` with common scientific libraries if you want faster startup

## Quick pull (minimal base)

A quick way to pull a minimal CUDA-capable Ubuntu base (adjust as needed) using
**Singularity**:

```bash
cd "$(git rev-parse --show-toplevel)"/template
singularity pull --force base.sif docker://ubuntu:22.04
```

If you need CUDA, choose an appropriate upstream (for example `docker://nvidia/cuda:12.4.1-runtime-ubuntu22.04`). The runner will point to this image via `--singularity_image template/base.sif` (or the corresponding config field).

## Validate the image

Run a quick check inside the container before launching experiments:

```bash
singularity exec template/base.sif python -c "import sys; print(sys.version)"
singularity exec template/base.sif bash -lc "gcc --version && git --version"
```

## Image placeholder (diagram)

![Singularity base image layers](../docs/images/singularity_base_image_layers.png)
<!-- IMAGE_PROMPT:
Create a technical layered diagram (4:3) showing a base SIF image as stacked layers labeled: OS (Ubuntu), Build tools (gcc/make/cmake), Python 3.10+, Optional CUDA runtime. Next to it, show a separate translucent "Per-worker overlay/sandbox" layer labeled "Phase 1 installs". Use simple 3D-ish flat layers, thin outlines, blue/gray palette on white background. Add a caption "Singularity base image layers". Sans-serif text, no logos, no gradients. -->
