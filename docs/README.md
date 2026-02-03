# Documentation Index

This folder contains documentation for AI-Scientist-v2-HPC, organized by topic.

## Directory Structure

```
docs/
├── overview/            # Core concepts, workflow, glossary
├── getting-started/     # Installation, setup, and first run
├── configuration/       # Configuration files and CLI
├── architecture/        # System architecture and internals
├── memory/              # MemGPT-style memory system
├── visualization/       # Visualization tools (HTML viewers)
├── development/         # Testing, troubleshooting, verification
├── images/              # Diagrams and figures
├── README.md            # This file
└── citation.md          # Citation information
```

## Quick Navigation

### Overview Documents

Start here if you're completely new to this project:

- [concepts.md](overview/concepts.md) - Core concepts: tree search, phases, memory
- [workflow.md](overview/workflow.md) - End-to-end workflow diagram
- [glossary.md](overview/glossary.md) - Terminology glossary

### Getting Started

New users should follow this path:

1. [requirements.md](getting-started/requirements.md) - Host and container dependencies
2. [installation.md](getting-started/installation.md) - Conda/pip/torch setup
3. [credentials.md](getting-started/credentials.md) - API keys and model providers
4. [quickstart.md](getting-started/quickstart.md) - Minimal end-to-end run

### Configuration & Usage

- [configuration.md](configuration/configuration.md) - `bfts_config.yaml` reference
- [cli-entry-points.md](configuration/cli-entry-points.md) - Script entry points
- [fewshot-customization.md](configuration/fewshot-customization.md) - Generate domain-specific review examples
- [review-bias.md](configuration/review-bias.md) - Control automated review strictness (neg/pos/neutral)
- [execution-modes.md](configuration/execution-modes.md) - Split vs single execution
- [outputs.md](configuration/outputs.md) - Run directories and artifacts

### Architecture & Internals

- [execution-flow.md](architecture/execution-flow.md) - Standard execution flow (no memory)
- [llm-context.md](architecture/llm-context.md) - Prompt assembly overview
- [llm-context-details.md](architecture/llm-context-details.md) - Detailed context per phase
- [prompt-structure.md](architecture/prompt-structure.md) - Prompt directory layout
- [file-roles.md](architecture/file-roles.md) - Code file responsibilities
- [resource-files.md](architecture/resource-files.md) - JSON/YAML resource schemas

### Visualization

- [visualization.md](visualization/visualization.md) - HTML visualization tools (`unified_tree_viz.html`, `memory_database.html`)

### Memory System

The memory system documentation is organized by flow:

- [memory.md](memory/memory.md) - **Start here** - Memory system overview and API
- [memory-flow.md](memory/memory-flow.md) - High-level architecture and injection points
- [memory-flow-phase0.md](memory/memory-flow-phase0.md) - Phase 0 (planning) flow
- [memory-flow-phases.md](memory/memory-flow-phases.md) - Phase 1-4 (execution) flow
- [memory-flow-post-execution.md](memory/memory-flow-post-execution.md) - Post-execution processing
- [memory-for-paper.md](memory/memory-for-paper.md) - Final memory for paper generation
- [hardware-info-injection.md](memory/hardware-info-injection.md) - 🆕 Automatic hardware info extraction
- [memgpt-features.md](memory/memgpt-features.md) - Available memory features
- [memgpt-implementation.md](memory/memgpt-implementation.md) - Implementation details

**Recent Updates (2026-02-03)**:
- [IMPLEMENTATION_SUMMARY_20260203.md](memory/IMPLEMENTATION_SUMMARY_20260203.md) - 🆕 Implementation summary (English)
- [IMPLEMENTATION_SUMMARY_20260203_ja.md](memory/IMPLEMENTATION_SUMMARY_20260203_ja.md) - 🆕 実装サマリー (日本語)

### Development

- [testing.md](development/testing.md) - Unit tests and test scope
- [troubleshooting.md](development/troubleshooting.md) - Common failures and fixes
- [verification-report.md](development/verification-report.md) - Documentation vs implementation verification report

## Suggested Reading Paths

### Complete Beginner

For those new to both HPC-AutoResearch and automated research systems:

1. [concepts.md](overview/concepts.md) - Understand core concepts (Tree Search, Phases, Memory)
2. [workflow.md](overview/workflow.md) - See the complete workflow
3. [glossary.md](overview/glossary.md) - Learn the terminology
4. [requirements.md](getting-started/requirements.md) - Check what you need
5. [quickstart.md](getting-started/quickstart.md) - Run your first experiment

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
2. [execution-flow.md](architecture/execution-flow.md)
3. [prompt-structure.md](architecture/prompt-structure.md)
4. [llm-context-details.md](architecture/llm-context-details.md)

### Memory System Deep Dive

For implementing or debugging memory features:

1. [memory.md](memory/memory.md) - Start here
2. [memory-flow.md](memory/memory-flow.md) - Architecture overview
3. [memory-flow-phases.md](memory/memory-flow-phases.md) - Execution details
4. [memgpt-implementation.md](memory/memgpt-implementation.md) - Code implementation

## Related Files

| Documentation | Related Code/Config |
|--------------|---------------------|
| Configuration | [bfts_config.yaml](../bfts_config.yaml) |
| CLI entry points | [launch_scientist_bfts.py](../launch_scientist_bfts.py) |
| Prompts | [prompt/](../prompt/) |
| Memory | [ai_scientist/memory/](../ai_scientist/memory/) |
| Tree search | [ai_scientist/treesearch/](../ai_scientist/treesearch/) |
| Visualization | [ai_scientist/visualization/](../ai_scientist/visualization/) |
| Ideas | [ai_scientist/ideas/](../ai_scientist/ideas/) |
| Tests | [tests/](../tests/) |

## Images

Diagrams are stored in [images/](images/):

- `memory_flow.png` - Memory system architecture
- `phasing_flow.png` - Phase execution flow
- `citation_context.png` - Citation context diagram

## Citation

See [citation.md](citation.md) for BibTeX and paper links.
