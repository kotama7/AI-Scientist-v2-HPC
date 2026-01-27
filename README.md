<div align="center">
  <h1>
    <b>HPC-AutoResearch</b><br>
  </h1>
</div>

HPC-AutoResearch targets HPC environments with a Singularity-based, split-phase execution path. It orchestrates: idea loading/generation → BFTS tree search experiments → plot aggregation → LaTeX writeup → optional PDF review.

## At a glance

- Split-phase execution with explicit install/coding/compile/run steps inside Singularity.
- Per-run isolation: every experiment gets its own config, logs, workspace, and artifacts.
- Tree-search agent manager with parallel workers (GPU-aware, CPU fallback).
- Optional MemGPT-style memory with LLM-based compression across branches for longer context.
- Resource files to mount datasets and inject templates/docs into prompts.
- Configurable persona system for role-specific prompt customization.
- Token tracking for monitoring LLM usage and costs.
- Multi-seed evaluation for robust result validation.

## Repository layout

- `launch_scientist_bfts.py`: main launcher (ideas -> experiments -> plots/writeup/review).
- `generate_paper.py`: plots/writeup/review for an existing run directory.
- `ai_scientist/`: core agent logic, tree search, prompts, memory, and utilities.
  - `memory/`: MemGPT-style hierarchical memory implementation.
  - `treesearch/`: BFTS agent manager and parallel workers.
  - `utils/`: token tracking and model parameter utilities.
  - `persona.py`: configurable persona system for prompt customization.
- `prompt/`: split-phase and stage prompts, response schemas, and writeup templates.
  - `common/`: base system prompts and domain-neutral instructions.
  - `phases/`: Phase 0 planning and Phase 1 installer prompts.
  - `memory/`: memory compression prompt templates.
  - `schemas/`: structured response schemas for split execution.
- `template/`: base Singularity image instructions (`template/README.md`).
- `docs/`: expanded guides for requirements, configuration, resources, outputs, and troubleshooting.
- `tests/`: unit tests for memory, resources, compression, and parallelism.
- `experiments/`: run outputs (generated at runtime).

## Typical workflows

1) **Run end-to-end**: generate ideas (optional) -> run `launch_scientist_bfts.py` -> review `experiments/<run>/`.
2) **Reuse a run**: skip the experiment and run `generate_paper.py` for plots/writeups.
3) **Iterate locally**: use `--phase_mode single` for quick iteration without Singularity.
4) **Custom persona**: set `agent.role_description` in config to customize the agent's role (e.g., "HPC Researcher").
5) **Enable memory**: use `--enable_memgpt` or set `memory.enabled=true` for hierarchical context management.
   - **Note**: Without MemGPT, there is **no context budget management**. Idea/task descriptions are injected as full text, which may exceed LLM context limits for complex experiments. See [docs/memory/memory.md](docs/memory/memory.md) for details.
6) **Parallel experiments**: adjust `--num_workers` to scale across available GPUs.

## Where to start

- If you are new: start with `docs/getting-started/requirements.md`, `docs/getting-started/installation.md`, and `docs/getting-started/quickstart.md`.
- If you are operating on HPC: read `docs/configuration/execution-modes.md`, `docs/configuration/configuration.md`, and `docs/configuration/outputs.md`.
- If you are extending prompts/resources: read `docs/architecture/llm-context.md` and `docs/architecture/resource-files.md`.

## Documentation

Detailed guides live in `docs/`. Start with
[docs/README.md](docs/README.md) for the full index.

- Requirements: [docs/getting-started/requirements.md](docs/getting-started/requirements.md) (host + container
  dependencies, optional tools); related: [requirements.txt](requirements.txt),
  [bfts_config.yaml](bfts_config.yaml), [template/README.md](template/README.md).
- Installation: [docs/getting-started/installation.md](docs/getting-started/installation.md) (conda/pip/torch
  setup, image prep); related: [requirements.txt](requirements.txt),
  [template/README.md](template/README.md).
- Credentials: [docs/getting-started/credentials.md](docs/getting-started/credentials.md) (model provider API
  keys and scope); related: [bfts_config.yaml](bfts_config.yaml),
  [ai_scientist/llm/](ai_scientist/llm/).
- CLI entry points: [docs/configuration/cli-entry-points.md](docs/configuration/cli-entry-points.md) (what
  each script does); related: [launch_scientist_bfts.py](launch_scientist_bfts.py),
  [generate_paper.py](generate_paper.py),
  [ai_scientist/perform_ideation_temp_free.py](ai_scientist/perform_ideation_temp_free.py).
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) (minimal end-to-end run);
  related: [template/README.md](template/README.md),
  [data_resources.json](data_resources.json).
- Configuration: [docs/configuration/configuration.md](docs/configuration/configuration.md) (how
  `bfts_config.yaml` is applied); related: [bfts_config.yaml](bfts_config.yaml),
  [launch_scientist_bfts.py](launch_scientist_bfts.py).
- Execution modes: [docs/configuration/execution-modes.md](docs/configuration/execution-modes.md) (split vs
  single, worker behavior); related:
  [prompt/execution_split_schema.txt](prompt/execution_split_schema.txt),
  [ai_scientist/treesearch/parallel_agent.py](ai_scientist/treesearch/parallel_agent.py).
- LLM context: [docs/architecture/llm-context.md](docs/architecture/llm-context.md) (prompt assembly and
  stage inputs); related: [prompt/](prompt/), [prompt/base_system.txt](prompt/base_system.txt).
- MemGPT-style memory: [docs/memory/memory.md](docs/memory/memory.md) (hierarchical memory +
  persistence); related: [ai_scientist/memory/memgpt_store.py](ai_scientist/memory/memgpt_store.py),
  [ai_scientist/memory/resource_memory.py](ai_scientist/memory/resource_memory.py).
- Resource files: [docs/architecture/resource-files.md](docs/architecture/resource-files.md) (JSON/YAML
  schema and staging rules); related: [data_resources.json](data_resources.json),
  [tests/test_resource.py](tests/test_resource.py).
- Outputs: [docs/configuration/outputs.md](docs/configuration/outputs.md) (run directories, logs, artifacts);
  related: [ai_scientist/treesearch/utils/viz_templates/template.html](ai_scientist/treesearch/utils/viz_templates/template.html).
- Testing: [docs/development/testing.md](docs/development/testing.md) (unit tests and scope); related:
  [tests/](tests/).
- Troubleshooting: [docs/development/troubleshooting.md](docs/development/troubleshooting.md) (common
  failures and fixes); related: [bfts_config.yaml](bfts_config.yaml),
  [template/README.md](template/README.md).
- Citation: [docs/citation.md](docs/citation.md) (bibtex and paper link);
  related: [README.md](README.md).

## Acknowledgement

The tree search component is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. This project extends the original [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) with split-phase execution, MemGPT-style context-engineering and Singularity container support.
