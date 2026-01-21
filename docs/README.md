# Documentation Index

This folder expands the root README into section-level guides. Each guide is
grounded in the repository layout, scripts, and config files, so you can jump
between docs and code without losing context.

## How to use this index

- Open the guide that matches your task, then follow the "Related files" links
  to see concrete defaults and scripts.
- Use the "Suggested paths" section below if you are new or operating in HPC.
- All guides include image placeholders with generation prompts so you can
  replace them with custom diagrams later.

## Suggested paths

- **New user**: requirements -> installation -> quickstart -> outputs.
- **HPC operator**: requirements -> execution modes -> configuration -> outputs
  -> troubleshooting.
- **Prompt/LLM tuning**: llm context -> prompt structure -> configuration -> resource files.
- **Understanding codebase**: file roles -> prompt structure -> llm context details.
- **Persona customization**: llm context (Persona system) -> configuration (`agent.role_description`).
- **Memory optimization**: memory -> memgpt features -> memgpt implementation -> configuration.
- **MemGPT deep dive**: memgpt features -> memgpt implementation -> memory.
- **Testing/QA**: testing -> outputs -> troubleshooting.

## Guides

The list below includes a short description and related files worth opening
alongside the corresponding guide.

- Requirements: [requirements.md](requirements.md) (host + container
  dependencies, optional tools); related: [requirements.txt](../requirements.txt),
  [bfts_config.yaml](../bfts_config.yaml), [template/README.md](../template/README.md).
- Installation: [installation.md](installation.md) (conda/pip/torch setup,
  image prep); related: [requirements.txt](../requirements.txt),
  [template/README.md](../template/README.md).
- Credentials: [credentials.md](credentials.md) (model provider keys and scope);
  related: [bfts_config.yaml](../bfts_config.yaml),
  [ai_scientist/llm.py](../ai_scientist/llm.py).
- CLI entry points: [cli-entry-points.md](cli-entry-points.md) (what each script
  does); related: [launch_scientist_bfts.py](../launch_scientist_bfts.py),
  [generate_paper.py](../generate_paper.py),
  [ai_scientist/perform_ideation_temp_free.py](../ai_scientist/perform_ideation_temp_free.py).
- Quickstart: [quickstart.md](quickstart.md) (minimal end-to-end run); related:
  [template/README.md](../template/README.md),
  [data_resources.json](../data_resources.json).
- Configuration: [configuration.md](configuration.md) (how `bfts_config.yaml` is
  applied); related: [bfts_config.yaml](../bfts_config.yaml),
  [launch_scientist_bfts.py](../launch_scientist_bfts.py).
- Execution modes: [execution-modes.md](execution-modes.md) (split vs single,
  worker behavior); related: [prompt/agent/parallel/response_format/execution_split.txt](../prompt/agent/parallel/response_format/execution_split.txt),
  [ai_scientist/treesearch/parallel_agent.py](../ai_scientist/treesearch/parallel_agent.py).
- LLM context: [llm-context.md](llm-context.md) (prompt assembly and stage
  inputs); related: [prompt/](../prompt/), [prompt/core/system.txt](../prompt/core/system.txt).
- LLM context details: [llm-context-details.md](llm-context-details.md) (detailed
  context components per phase); related: [ai_scientist/treesearch/parallel_agent.py](../ai_scientist/treesearch/parallel_agent.py).
- File roles: [file-roles.md](file-roles.md) (role and responsibility of each
  file); related: [ai_scientist/](../ai_scientist/), [prompt/](../prompt/).
- Prompt structure: [prompt-structure.md](prompt-structure.md) (prompt directory
  organization and contents); related: [prompt/](../prompt/),
  [ai_scientist/prompt_loader.py](../ai_scientist/prompt_loader.py).
- MemGPT-style memory: [memory.md](memory.md) (hierarchical memory +
  persistence); related: [ai_scientist/memory/](../ai_scientist/memory/).
- MemGPT features: [memgpt-features.md](memgpt-features.md) (available memory
  features and configuration); related: [bfts_config.yaml](../bfts_config.yaml).
- MemGPT implementation: [memgpt-implementation.md](memgpt-implementation.md)
  (technical implementation details); related:
  [ai_scientist/memory/memgpt_store.py](../ai_scientist/memory/memgpt_store.py).
- Resource files: [resource-files.md](resource-files.md) (JSON/YAML schema and
  staging rules); related: [data_resources.json](../data_resources.json),
  [tests/test_resource.py](../tests/test_resource.py).
- Outputs: [outputs.md](outputs.md) (run directories, logs, artifacts); related:
  [ai_scientist/treesearch/utils/viz_templates/template.html](../ai_scientist/treesearch/utils/viz_templates/template.html).
- Testing: [testing.md](testing.md) (unit tests and scope); related:
  [tests/](../tests/).
- Troubleshooting: [troubleshooting.md](troubleshooting.md) (common failures and
  fixes); related: [bfts_config.yaml](../bfts_config.yaml),
  [template/README.md](../template/README.md).
- Citation: [citation.md](citation.md) (bibtex and paper link); related:
  [README.md](../README.md).

## Additional references

- Templates and image assets: [template/](../template/), [docs/logo_v1.png](logo_v1.png).
- Prompt templates: [prompt/](../prompt/) (system messages, stage instructions,
  response formats).
  - Core prompts: [prompt/core/](../prompt/core/) (base system, domain-neutral).
  - Phase prompts: [prompt/config/phases/](../prompt/config/phases/) (planning, installer).
  - Memory prompts: [prompt/config/memory/](../prompt/config/memory/) (compression template).
  - Response formats: [prompt/agent/parallel/response_format/](../prompt/agent/parallel/response_format/) (split execution format).
- Example ideas: [ai_scientist/ideas/](../ai_scientist/ideas/).
- Memory implementation: [ai_scientist/memory/](../ai_scientist/memory/).
- Persona system: [ai_scientist/persona.py](../ai_scientist/persona.py).
- Token tracking: [ai_scientist/utils/token_tracker.py](../ai_scientist/utils/token_tracker.py).
