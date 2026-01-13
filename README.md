<div align="center">
  <h1>
    <b>Title Undecided</b><br>
  </h1>
</div>

This fork targets HPC environments with a Singularity-based, split-phase execution path. It orchestrates: idea loading/generation → BFTS tree search experiments → plot aggregation → LaTeX writeup → optional PDF review.

## At a glance

- Split-phase execution with explicit install/coding/compile/run steps inside Singularity.
- Per-run isolation: every experiment gets its own config, logs, workspace, and artifacts.
- Tree-search agent manager with parallel workers (GPU-aware, CPU fallback).
- Optional MemGPT-style memory across branches for longer context.
- Resource files to mount datasets and inject templates/docs into prompts.

## Repository layout

- `launch_scientist_bfts.py`: main launcher (ideas -> experiments -> plots/writeup/review).
- `generate_paper.py`: plots/writeup/review for an existing run directory.
- `ai_scientist/`: core agent logic, tree search, prompts, and memory.
- `prompt/`: split-phase and stage prompts, response schemas, and writeup templates.
- `template/`: base Singularity image instructions (`template/README.md`).
- `docs/`: expanded guides for requirements, configuration, resources, outputs, and troubleshooting.
- `experiments/`: run outputs (generated at runtime).

## Typical workflows

1) **Run end-to-end**: generate ideas (optional) -> run `launch_scientist_bfts.py` -> review `experiments/<run>/`.
2) **Reuse a run**: skip the experiment and run `generate_paper.py` for plots/writeups.
3) **Iterate locally**: use `--phase_mode single` for quick iteration without Singularity.

## Where to start

- If you are new: start with `docs/requirements.md`, `docs/installation.md`, and `docs/quickstart.md`.
- If you are operating on HPC: read `docs/execution-modes.md`, `docs/configuration.md`, and `docs/outputs.md`.
- If you are extending prompts/resources: read `docs/llm-context.md` and `docs/resource-files.md`.

## Documentation

Detailed guides live in `docs/`. Start with
[docs/README.md](docs/README.md) for the full index.

- Requirements: [docs/requirements.md](docs/requirements.md) (host + container
  dependencies, optional tools); related: [requirements.txt](requirements.txt),
  [bfts_config.yaml](bfts_config.yaml), [template/README.md](template/README.md).
- Installation: [docs/installation.md](docs/installation.md) (conda/pip/torch
  setup, image prep); related: [requirements.txt](requirements.txt),
  [template/README.md](template/README.md).
- Credentials: [docs/credentials.md](docs/credentials.md) (model provider API
  keys and scope); related: [bfts_config.yaml](bfts_config.yaml),
  [ai_scientist/llm.py](ai_scientist/llm.py).
- CLI entry points: [docs/cli-entry-points.md](docs/cli-entry-points.md) (what
  each script does); related: [launch_scientist_bfts.py](launch_scientist_bfts.py),
  [generate_paper.py](generate_paper.py),
  [ai_scientist/perform_ideation_temp_free.py](ai_scientist/perform_ideation_temp_free.py).
- Quickstart: [docs/quickstart.md](docs/quickstart.md) (minimal end-to-end run);
  related: [template/README.md](template/README.md),
  [data_resources.json](data_resources.json).
- Configuration: [docs/configuration.md](docs/configuration.md) (how
  `bfts_config.yaml` is applied); related: [bfts_config.yaml](bfts_config.yaml),
  [launch_scientist_bfts.py](launch_scientist_bfts.py).
- Execution modes: [docs/execution-modes.md](docs/execution-modes.md) (split vs
  single, worker behavior); related:
  [prompt/execution_split_schema.txt](prompt/execution_split_schema.txt),
  [ai_scientist/treesearch/parallel_agent.py](ai_scientist/treesearch/parallel_agent.py).
- LLM context: [docs/llm-context.md](docs/llm-context.md) (prompt assembly and
  stage inputs); related: [prompt/](prompt/), [prompt/base_system.txt](prompt/base_system.txt).
- MemGPT-style memory: [docs/memory.md](docs/memory.md) (hierarchical memory +
  persistence); related: [ai_scientist/memory/memgpt_store.py](ai_scientist/memory/memgpt_store.py),
  [ai_scientist/memory/resource_memory.py](ai_scientist/memory/resource_memory.py).
- Resource files: [docs/resource-files.md](docs/resource-files.md) (JSON/YAML
  schema and staging rules); related: [data_resources.json](data_resources.json),
  [tests/test_resource.py](tests/test_resource.py).
- Outputs: [docs/outputs.md](docs/outputs.md) (run directories, logs, artifacts);
  related: [ai_scientist/treesearch/utils/viz_templates/template.html](ai_scientist/treesearch/utils/viz_templates/template.html).
- Testing: [docs/testing.md](docs/testing.md) (unit tests and scope); related:
  [tests/](tests/).
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md) (common
  failures and fixes); related: [bfts_config.yaml](bfts_config.yaml),
  [template/README.md](template/README.md).
- Citation: [docs/citation.md](docs/citation.md) (bibtex and paper link);
  related: [README.md](README.md).

## Acknowledgement

The tree search component is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. This HPC fork extends the original [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) with split-phase execution and Singularity container support.
