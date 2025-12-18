# Singularity base image template

Place the shared base image at `template/base.sif`. A quick way to pull a minimal CUDA-capable Ubuntu base (adjust as needed) using **Singularity**:

```bash
cd "$(git rev-parse --show-toplevel)"/template
singularity pull --force base.sif docker://ubuntu:22.04
```

If you need CUDA, choose an appropriate upstream (for example `docker://nvidia/cuda:12.4.1-runtime-ubuntu22.04`). The runner will point to this image via `--singularity_image template/base.sif` (or the corresponding config field).
