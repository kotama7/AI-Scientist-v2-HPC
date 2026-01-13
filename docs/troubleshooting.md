# Troubleshooting

## Collect debug info first

- `experiments/<run>/logs/` for phase logs and errors.
- `experiments/<run>/logs/.../unified_tree_viz.html` for node history.
- `experiments/<run>/logs/.../phase_logs/node_<id>/prompt_logs/` for prompt inputs.
- `experiments/<run>/memory/` if memory is enabled.

## "Singularity image is required"

Split mode needs a base image. Pass `--singularity_image` or update
`exec.singularity_image` in `bfts_config.yaml`.

## `singularity` command not found

Ensure Apptainer is aliased or symlinked to `singularity`, or update your PATH
so the launcher can invoke it.

## Phase 1 cannot write inside the container

Try one of the following:

- `--writable_mode overlay` with `--container_overlay /path/to/overlay.img`.
- Disable tmpfs via `--disable_writable_tmpfs`.

## `singularity build` fails due to permissions

Try `--use_fakeroot false`.

## `torch` import fails on the host

Install Torch in the control-plane environment. The launcher imports Torch to
map workers to GPUs even in split mode.

## CUDA is not detected

- Confirm `CUDA_VISIBLE_DEVICES` is set correctly in your job environment.
- Verify `python -c "import torch; print(torch.cuda.is_available())"` on the
  host.

## Resource validation errors

Double-check:

- `mount_path`/`dest` are under `/workspace`.
- Local `host_path` exists.
- `items.path` is relative and does not include `..`.

## Hugging Face downloads fail

Install `huggingface_hub` inside the worker image and provide
`HUGGINGFACE_API_KEY` if needed.

## Phase 1 repeatedly fails to install deps

- Increase `--phase1_max_steps` so the iterative installer can recover.
- Switch to `--writable_mode overlay` with a larger overlay image.
- Bake the dependency into the base SIF if it is always needed.

## Tree visualization is blank

- Ensure `unified_tree_viz.html` and the per-stage `tree_data.json` files exist.
- Open the HTML file from the `logs/` directory so relative paths resolve.
