# File Roles (各ファイルの役割)

This document describes the role and responsibility of each file in the
HPC-AutoResearch project.

## Top-Level Files

| File | Role |
|------|------|
| `launch_scientist_bfts.py` | Main entry point. Loads ideas, launches tree-search experiments, invokes plot aggregation, writeup, and optional review. Orchestrates the full research automation pipeline. |
| `generate_paper.py` | Standalone script to regenerate plots/writeup/review from an existing experiment run directory (skips the experiment phase). |
| `bfts_config.yaml` | Default configuration file defining paths, execution modes, agent hyperparameters, memory settings, and LLM model selections. Copied into each run directory for self-contained experiments. |
| `data_resources.json` | Resource file defining datasets, GitHub repos, and HuggingFace models to stage into experiments. Supports prompt injection and path binding. |
| `requirements.txt` | Python dependencies for the host environment. |

## ai_scientist/ (Core Package)

### Root Modules

| File | Role |
|------|------|
| `__init__.py` | Package marker. |
| `prompt_loader.py` | Prompt file loader with caching and persona override support. Resolves templates from `prompt/` directory. |
| `persona.py` | Persona system for role-based prompt customization. Replaces `{persona}` placeholders with configured `agent.role_description`. |

### llm/ (LLM Client Package)

| File | Role |
|------|------|
| `__init__.py` | Exports LLM client factory and utilities. |
| `clients.py` | LLM client factory. Supports OpenAI, Anthropic, Ollama, DeepSeek, Gemini, Bedrock, Vertex AI. |
| `constants.py` | LLM-related constants and model configurations. |
| `response.py` | Response parsing and normalization. Includes token tracking decorator. |
| `utils.py` | Utility functions for LLM operations. |

### vlm/ (Vision-Language Model Package)

| File | Role |
|------|------|
| `__init__.py` | Exports VLM client factory and utilities. |
| `clients.py` | VLM client factory for image-based analysis. |
| `constants.py` | VLM-related constants and model configurations. |
| `response.py` | VLM response parsing and normalization. |
| `utils.py` | Utility functions for VLM operations (plot review, figure captioning). |
| `perform_ideation_temp_free.py` | Idea generation module. Generates research ideas from seed topics using LLM. |
| `perform_plotting.py` | Plot aggregation and VLM-based plot analysis. Selects best plots for paper inclusion. |
| `perform_writeup.py` | Paper writeup generation. Produces LaTeX content from experiment results with citation gathering. |
| `perform_llm_review.py` | LLM-based paper review simulation (NeurIPS-style scoring). |
| `perform_vlm_review.py` | VLM-based figure/caption review for visual quality assessment. |

### memory/ (MemGPT-Style Memory)

| File | Role |
|------|------|
| `__init__.py` | Exports `MemoryManager` and resource snapshot utilities. |
| `memgpt_store.py` | Core MemGPT implementation. Hierarchical memory with Core/Recall/Archival layers, SQLite persistence, FTS5 search, LLM-based compression, branch isolation, and promotion. |
| `resource_memory.py` | Resource snapshot management. Tracks resource files in long-term memory with digests, staging info, and usage tracking. |

### treesearch/ (BFTS Agent Manager)

| File | Role |
|------|------|
| `__init__.py` | Package marker. |
| `agent_manager.py` | Agent manager orchestrating tree-search stages (draft/debug/improve/hyperparam/ablation). Manages node creation, stage transitions, and best-node selection. |
| `parallel_agent.py` | Parallel agent worker implementation. Handles split-phase execution (Phase 0-4), code generation, compilation, run execution inside Singularity containers. |
| `perform_experiments_bfts_with_agentmanager.py` | Entry point for BFTS experiments. Initializes agent manager and runs the tree search. |
| `journal.py` | Journal system for logging experiment progress, stage notes, and result tracking. |
| `journal2report.py` | Converts journal entries to structured reports. |
| `log_summarization.py` | Summarizes execution logs using LLM for memory storage and prompt injection. |
| `interpreter.py` | Code interpreter for executing generated Python/C++ code in isolation. |
| `bfts_utils.py` | Utility functions for BFTS operations (idea to markdown, config editing). |
| `worker_plan.py` | Worker planning utilities for parallel execution. |

### treesearch/backend/

| File | Role |
|------|------|
| Backend implementations for different execution environments (Singularity, local). |

### treesearch/utils/

| File | Role |
|------|------|
| Visualization templates, token tracking utilities, and helper functions. |

### utils/

| File | Role |
|------|------|
| `token_tracker.py` | Token usage tracking across LLM calls. Records per-model token counts and costs. |
| `model_params.py` | Model parameter utilities for building token budget kwargs per model. |

### Other Directories

| Directory | Role |
|-----------|------|
| `blank_latex/` | Blank LaTeX template for academic paper generation. |
| `fewshot_examples/` | Few-shot examples for prompting. |
| `ideas/` | Pre-generated research idea JSON files. |
| `tools/` | Agent tools and utilities. |

## prompt/ (Prompt Templates)

See [prompt-structure.md](prompt-structure.md) for detailed structure.

## tests/

| File | Role |
|------|------|
| `test_resource.py` | Tests for resource file handling. |
| `test_llm_compression.py` | Tests for LLM-based memory compression. |
| `test_iterative_compression.py` | Tests for iterative compression. |
| `test_memgpt_branch_inheritance.py` | Tests for MemGPT branch inheritance. |
| `test_final_memory_generation.py` | Tests for final memory generation. |
| `test_writeup_memory_loading.py` | Tests for writeup memory loading. |
| `test_smoke_split.py` | Smoke tests for split execution mode. |
| `test_worker_parallelism.py` | Tests for worker parallelism. |

## template/

| File | Role |
|------|------|
| `README.md` | Instructions for building the base Singularity image. |

## experiments/

Runtime output directory. Each run creates a subdirectory containing:
- `bfts_config.yaml` (frozen config)
- `logs/` (tree visualization, prompt logs, phase logs)
- `memory/` (SQLite database, resource snapshots, final memory exports)
- Generated code, plots, and LaTeX artifacts.
