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
- **Prompt/LLM tuning**: llm context -> configuration -> resource files -> memory.
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
  worker behavior); related: [prompt/execution_split_schema.txt](../prompt/execution_split_schema.txt),
  [ai_scientist/treesearch/parallel_agent.py](../ai_scientist/treesearch/parallel_agent.py).
- LLM context: [llm-context.md](llm-context.md) (prompt assembly and stage
  inputs); related: [prompt/](../prompt/), [prompt/base_system.txt](../prompt/base_system.txt).
- MemGPT-style memory: [memory.md](memory.md) (hierarchical memory +
  persistence); related: [ai_scientist/memory/](../ai_scientist/memory/).
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
- Example ideas: [ai_scientist/ideas/](../ai_scientist/ideas/).
