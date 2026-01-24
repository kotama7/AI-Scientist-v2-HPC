# Documentation Index

This folder contains documentation for AI-Scientist-v2-HPC, organized by topic.

## Directory Structure

```
docs/
├── getting-started/     # Installation, setup, and first run
├── configuration/       # Configuration files and CLI
├── architecture/        # System architecture and internals
├── memory/              # MemGPT-style memory system
├── development/         # Testing and troubleshooting
├── images/              # Diagrams and figures
├── README.md            # This file
└── citation.md          # Citation information
```

## Quick Navigation

### Getting Started

New users should follow this path:

1. [requirements.md](getting-started/requirements.md) - Host and container dependencies
2. [installation.md](getting-started/installation.md) - Conda/pip/torch setup
3. [credentials.md](getting-started/credentials.md) - API keys and model providers
4. [quickstart.md](getting-started/quickstart.md) - Minimal end-to-end run

### Configuration & Usage

- [configuration.md](configuration/configuration.md) - `bfts_config.yaml` reference
- [cli-entry-points.md](configuration/cli-entry-points.md) - Script entry points
- [execution-modes.md](configuration/execution-modes.md) - Split vs single execution
- [outputs.md](configuration/outputs.md) - Run directories and artifacts

### Architecture & Internals

- [execution_flow_standard.md](architecture/execution_flow_standard.md) - Standard execution flow (no memory)
- [llm-context.md](architecture/llm-context.md) - Prompt assembly overview
- [llm-context-details.md](architecture/llm-context-details.md) - Detailed context per phase
- [prompt-structure.md](architecture/prompt-structure.md) - Prompt directory layout
- [file-roles.md](architecture/file-roles.md) - Code file responsibilities
- [resource-files.md](architecture/resource-files.md) - JSON/YAML resource schemas

### Memory System

The memory system documentation is organized by flow:

- [memory.md](memory/memory.md) - **Start here** - Memory system overview and API
- [memory_flow.md](memory/memory_flow.md) - High-level architecture and injection points
- [memory_flow_phase0.md](memory/memory_flow_phase0.md) - Phase 0 (planning) flow
- [memory_flow_phases.md](memory/memory_flow_phases.md) - Phase 1-4 (execution) flow
- [memory_flow_post_execution.md](memory/memory_flow_post_execution.md) - Post-execution processing
- [memory_for_paper.md](memory/memory_for_paper.md) - Final memory for paper generation
- [memgpt-features.md](memory/memgpt-features.md) - Available memory features
- [memgpt-implementation.md](memory/memgpt-implementation.md) - Implementation details

### Development

- [testing.md](development/testing.md) - Unit tests and test scope
- [troubleshooting.md](development/troubleshooting.md) - Common failures and fixes

## Suggested Reading Paths

### HPC Operator

For deploying and operating in HPC environments:

1. [requirements.md](getting-started/requirements.md)
2. [execution-modes.md](configuration/execution-modes.md)
3. [configuration.md](configuration/configuration.md)
4. [outputs.md](configuration/outputs.md)
5. [troubleshooting.md](development/troubleshooting.md)

### Prompt/LLM Tuning

For customizing prompts and LLM behavior:

1. [llm-context.md](architecture/llm-context.md)
2. [prompt-structure.md](architecture/prompt-structure.md)
3. [configuration.md](configuration/configuration.md)
4. [resource-files.md](architecture/resource-files.md)

### Understanding the Codebase

For developers new to the project:

1. [file-roles.md](architecture/file-roles.md)
2. [execution_flow_standard.md](architecture/execution_flow_standard.md)
3. [prompt-structure.md](architecture/prompt-structure.md)
4. [llm-context-details.md](architecture/llm-context-details.md)

### Memory System Deep Dive

For implementing or debugging memory features:

1. [memory.md](memory/memory.md) - Start here
2. [memory_flow.md](memory/memory_flow.md) - Architecture overview
3. [memory_flow_phases.md](memory/memory_flow_phases.md) - Execution details
4. [memgpt-implementation.md](memory/memgpt-implementation.md) - Code implementation

## Related Files

| Documentation | Related Code/Config |
|--------------|---------------------|
| Configuration | [bfts_config.yaml](../bfts_config.yaml) |
| CLI entry points | [launch_scientist_bfts.py](../launch_scientist_bfts.py) |
| Prompts | [prompt/](../prompt/) |
| Memory | [ai_scientist/memory/](../ai_scientist/memory/) |
| Tree search | [ai_scientist/treesearch/](../ai_scientist/treesearch/) |
| Ideas | [ai_scientist/ideas/](../ai_scientist/ideas/) |
| Tests | [tests/](../tests/) |

## Images

Diagrams are stored in [images/](images/):

- `memory_flow.png` - Memory system architecture
- `phasing_flow.png` - Phase execution flow
- `citation_context.png` - Citation context diagram

## Citation

See [citation.md](citation.md) for BibTeX and paper links.
