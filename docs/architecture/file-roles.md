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

### output/ (Paper Generation Pipeline)

| File | Role |
|------|------|
| `__init__.py` | Exports plotting, writeup, and citation utilities. |
| `plotting.py` | Plot aggregation implementation. `aggregate_plots()` for selecting/combining experiment plots. |
| `writeup.py` | Paper writeup generation. `perform_writeup()` for LaTeX generation with reflection steps. |
| `citation.py` | Citation gathering. `gather_citations()` for collecting references from Semantic Scholar. |
| `latex_utils.py` | LaTeX utilities. `compile_latex()`, page detection, PDF processing. |

### Root Modules (ai_scientist/)

| File | Role |
|------|------|
| `perform_ideation_temp_free.py` | Idea generation. Generates research ideas from seed topics using LLM with temperature control. |
| `perform_plotting.py` | CLI wrapper for `output.plotting`. Backward compatibility entry point. |
| `perform_writeup.py` | CLI wrapper for `output.writeup`. Backward compatibility entry point. |
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
| `agent_manager.py` | Agent manager orchestrating tree-search stages (draft/debug/improve/hyperparam/ablation). Manages node creation, stage transitions, best-node selection, and stage cleanup (`_cleanup_stage_node_logs`). |
| `parallel_agent.py` | Parallel agent worker implementation. Handles split-phase execution (Phase 0-4), code generation, compilation, run execution inside Singularity containers. Also manages workspace persistence to `node_logs/` for cross-stage file inheritance. |
| `perform_experiments_bfts_with_agentmanager.py` | Entry point for BFTS experiments. Initializes agent manager and runs the tree search. |
| `journal.py` | Journal system for logging experiment progress, stage notes, and result tracking. |
| `journal2report.py` | Converts journal entries to structured reports. |
| `log_summarization.py` | Summarizes execution logs using LLM for memory storage and prompt injection. |
| `interpreter.py` | Code interpreter for executing generated Python/C++ code in isolation. |
| `bfts_utils.py` | Utility functions for BFTS operations (idea to markdown, config editing). |
| `worker_plan.py` | Worker planning utilities for parallel execution. |
| `worker.py` | Worker management classes (`WorkerTask`, `WorkerResult`, `WorkerManager`) for parallel experiment execution. Handles task distribution and result collection. |
| `gpu.py` | GPU management (`GPUManager`, `parse_cuda_visible_devices`). Handles GPU allocation and CUDA environment configuration. |
| `ablation.py` | Ablation study configuration classes (`AblationConfig`, `AblationIdea`, `HyperparamTuningIdea`). Defines ablation and hyperparameter tuning experiments. |
| `backend.py` | LLM backend interface. Provides `query()` function for LLM calls and `FunctionSpec` for structured output. |

### treesearch/backend/

| File | Role |
|------|------|
| Backend implementations for different execution environments (Singularity, local). |

### treesearch/utils/

| File | Role |
|------|------|
| `phase_execution.py` | Phase 1 execution utilities. `SingularityWorkerContainer` for container management, `ExecutionEnvironment` for environment detection, iterative installer logic. |
| `phase_plan.py` | Phase 2-4 artifact generation. `extract_phase_artifacts()` for parsing LLM output, `apply_workspace_plan()` for file writing. |
| `resource.py` | Resource system implementation. `LocalResource`, `GitHubResource`, `HuggingFaceResource` dataclasses and `load_resources()`, `build_resources_context()` functions. |
| `config.py` | Configuration utilities. `Config` class for YAML config loading with dot-notation access. |
| `metric.py` | Metric handling. `MetricValue` dataclass for experiment metrics, `WorstMetricValue` for failed nodes. |
| `response.py` | LLM response parsing. `extract_code()`, `extract_memory_updates()`, `trim_long_string()` utilities. |
| `parsing.py` | Text parsing utilities. JSON extraction, language normalization, Phase 0 plan parsing. |
| `file_utils.py` | File utilities. `read_text()`, `summarize_file()`, `find_previous_run_dir()`, log summarization. |
| `artifacts.py` | Artifact management. `write_prompt_log()`, `save_phase_execution_artifacts()`, path resolution. |
| `serialize.py` | Serialization utilities. Markdown-to-dict parsing, JSON normalization. |

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
- `node_logs/` (temporary: per-node workspace snapshots for cross-stage inheritance, cleaned up after each stage)
- `stage_best/` (permanent: best node workspace per stage for next-stage inheritance)
- Generated code, plots, and LaTeX artifacts.
