# Execution Modes

The system supports two execution modes: split (default) and single. Split mode
runs inside Singularity and divides work into explicit phases. Single mode runs
on the host and is useful for local development.

## Split Mode overview

![Split overview](../images/phasing_flow.png)

## Split vs single comparison

| Aspect | Split mode | Single mode |
| --- | --- | --- |
| Runtime | Inside Singularity | Host environment |
| Phases | 0-4 explicit | Legacy combined flow |
| Dependencies | In container | On host |
| Best for | HPC and reproducibility | Local iteration/debug |
| Requires | `--singularity_image` | No container needed |

## Split mode (`exec.phase_mode=split`)

Split mode runs the experiment as explicit phases inside Singularity:

0. Planning (Phase 0, prompt-only; produces the phase plan)
1. Download & install (Phase 1)
2. Coding (Phase 2)
3. Compile (Phase 3)
4. Run (Phase 4)

The LLM outputs a structured JSON payload with per-phase artifacts. The run
phase must produce `working/{experiment_name}_data.npy` inside the container
(e.g., `stability_autotuning_data.npy`). The expected outputs can be
overridden by the plan.

Per-worker SIFs are built when `per_worker_sif=true` under
`experiments/<...>/workers/worker-*/container/`.

Relevant flags in `launch_scientist_bfts.py`:

- `--singularity_image` (required for split mode)
- `--use_gpu` (set false to disable `--nv`)
- `--per_worker_sif`, `--keep_sandbox`, `--use_fakeroot`
- `--writable_mode`, `--phase1_max_steps`
- `--container_overlay`, `--disable_writable_tmpfs`

### Writable modes (Phase 1)

- `auto`: prefer tmpfs, fall back to overlay when needed.
- `tmpfs`: fastest but memory-backed (limited size).
- `overlay`: slower but more persistent; requires an overlay image.
- `none`: read-only; use only if you pre-installed dependencies.

## Single mode (`exec.phase_mode=single`)

Single mode uses the legacy flow without split phases. Code executes on the host
environment (no container), and package guidance comes from the prompt
templates. Bind-mounts are only applied in split mode, but resource context and
staged templates/docs are still available to the LLM/workspace.

Use single mode when:

- You are iterating locally and want to skip container setup.
- You need direct access to host tooling that is not in the image.
- You are debugging code generation without the Phase 1 install loop.

## Parallel workers

Worker parallelism follows `agent.num_workers` / `--num_workers` and is not
capped by GPU count. GPUs are assigned when available; extra workers run on CPU.
Startup logs include requested vs. actual workers, GPU detection
(`CUDA_VISIBLE_DEVICES` and torch), and split-mode per-worker container paths
(SIF/overlay/workdir).
Set `exec.use_gpu=false` or `--use_gpu false` to force CPU-only containers.

## Phase outputs and logs

- Phase 0: planning artifacts and the phase plan, stored in the run logs.
- Phase 1: install logs and environment notes inside the worker container.
- Phase 2-4: code, compile, and execution logs under
  `experiments/<run>/logs/`.

See `docs/outputs.md` for the full log layout.
