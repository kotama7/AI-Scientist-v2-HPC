# Refactor Notes (split + Singularity cleanup)

## Step0: Split entry chain (current code path)
- `launch_scientist_bfts.py` `__main__` (CLI entry) -> `edit_bfts_config_file` to create per-run config
- `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py:perform_experiments_bfts`
- `ai_scientist/treesearch/agent_manager.py:AgentManager.run`
- `ai_scientist/treesearch/parallel_agent.py:ParallelAgent.step`
- `ai_scientist/treesearch/parallel_agent.py:_process_node_wrapper` (worker process)
- Split phases executed in `_process_node_wrapper`:
  - Phase 1 (download/install iterative) -> `SingularityWorkerContainer.prepare_phase1` (container-only)
  - Phase 2 (coding) -> `apply_workspace_plan`
  - Phase 3 (compile) -> `run_commands_with_logging`
  - Phase 4 (run) -> `run_commands_with_logging`

## Step1: Keyword inventory (rg -l)
### adapter
- `launch_scientist_bfts.py`
- `README.md`
- `bfts_config.yaml`
- `prompt/execution_split_schema.txt`
- `prompt/base_system.txt`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/utils/config.py`
- `prompt/treesearch/parallel_agent/response_format/split_phase.txt`

### interpreter
- `prompt/execution_split_schema.txt`
- `prompt/treesearch/parallel_agent/response_format/split_phase.txt`
- `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py`
- `ai_scientist/treesearch/interpreter.py`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/journal.py`
- `prompt/treesearch/parallel_agent/language_adapter/change_prompt.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/seed_plotting_prompt/introduction.txt`

### router
- `ai_scientist/llm.py`
- `prompt/base_system.txt`
- `prompt/treesearch/parallel_agent/response_format/split_phase.txt`
- `prompt/execution_split_schema.txt`
- `ai_scientist/treesearch/parallel_agent.py`

### dispatch
- `ai_scientist/ideas/i_cant_believe_its_not_better_openai.json`

### backend
- `ai_scientist/utils/model_params.py`
- `ai_scientist/treesearch/journal2report.py`
- `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py`
- `ai_scientist/treesearch/log_summarization.py`
- `ai_scientist/treesearch/backend/__init__.py`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/agent_manager.py`
- `ai_scientist/treesearch/journal.py`

### language
- `LICENSE`
- `bfts_config.yaml`
- `launch_scientist_bfts.py`
- `ai_scientist/vlm.py`
- `ai_scientist/llm.py`
- `README.md`
- `ai_scientist/blank_icml_latex/icml2025.bst`
- `ai_scientist/ideas/i_cant_believe_its_not_better_openai.json`
- `ai_scientist/treesearch/interpreter.py`
- `ai_scientist/ideas/i_cant_believe_its_not_better.json`
- `ai_scientist/fewshot_examples/132_automated_relational.txt`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/utils/viz_templates/template.js`
- `ai_scientist/fewshot_examples/attention.txt`
- `prompt/treesearch/parallel_agent/response_format/hyperparam_tuning.txt`
- `ai_scientist/treesearch/utils/viz_templates/template.html`
- `ai_scientist/blank_icbinb_latex/iclr2025.bst`
- `prompt/treesearch/parallel_agent/response_format/split_phase.txt`
- `ai_scientist/treesearch/utils/config.py`
- `ai_scientist/treesearch/bfts_utils.py`
- `ai_scientist/treesearch/utils/response.py`
- `prompt/treesearch/parallel_agent/response_format/metric_parse.txt`
- `ai_scientist/treesearch/utils/phase_plan.py`
- `prompt/treesearch/parallel_agent/response_format/debug.txt`
- `prompt/treesearch/parallel_agent/response_format/default.txt`
- `prompt/treesearch/journal2report/system_prompt.json`
- `prompt/treesearch/parallel_agent/seed_plotting_prompt/response_format.txt`
- `prompt/treesearch/parallel_agent/response_format/ablation.txt`
- `prompt/domain_neutral.txt`
- `prompt/execution_split_schema.txt`
- `prompt/treesearch/parallel_agent/debug/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/language_decider.txt`
- `prompt/treesearch/parallel_agent/debug/bugfix_improvement_sketch_guideline.txt`
- `prompt/treesearch/parallel_agent/language_adapter/change_prompt.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/response_format/hyperparam_tuning.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/response_format/metric_parse.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/response_format/debug.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/seed_plotting_prompt/response_format.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/ablation_node/introduction_prefix.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/response_format/default.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/hyperparam_node/introduction_prefix.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/response_format/ablation.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/environment/message_cpp.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/define_global_metrics/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/environment/packages.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/debug/bugfix_improvement_sketch_guideline.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/debug/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/execution_review/introduction.txt`

### executor
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/log_summarization.py`

### strategy
- `ai_scientist/ideas/i_cant_believe_its_not_better_openai.json`
- `ai_scientist/ideas/i_cant_believe_its_not_better_ollama.json`
- `ai_scientist/ideas/i_cant_believe_its_not_better.json`

### TheAIScientist / ai_scientist / paper_writeup / citation / model_training / inference
- `suppliment_information.txt`
- `bfts_config.yaml`
- `README.md`
- `ai_scientist/perform_llm_review.py`
- `ai_scientist/perform_ideation_temp_free.py`
- `ai_scientist/perform_writeup.py`
- `ai_scientist/perform_icbinb_writeup.py`
- `ai_scientist/perform_vlm_review.py`
- `ai_scientist/perform_plotting.py`
- `ai_scientist/llm.py`
- `ai_scientist/prompt_loader.py`
- `ai_scientist/vlm.py`
- `ai_scientist/tools/semantic_scholar.py`
- `ai_scientist/blank_icbinb_latex/iclr2025.sty`
- `ai_scientist/blank_icbinb_latex/iclr2025.bst`
- `ai_scientist/ideas/i_cant_believe_its_not_better_openai.json`
- `ai_scientist/fewshot_examples/attention.txt`
- `ai_scientist/fewshot_examples/132_automated_relational.txt`
- `ai_scientist/blank_icbinb_latex/natbib.sty`
- `ai_scientist/treesearch/log_summarization.py`
- `ai_scientist/fewshot_examples/2_carpe_diem.json`
- `ai_scientist/treesearch/agent_manager.py`
- `ai_scientist/treesearch/journal2report.py`
- `ai_scientist/treesearch/journal.py`
- `ai_scientist/treesearch/backend/__init__.py`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/blank_icml_latex/icml2025.sty`
- `ai_scientist/blank_icml_latex/icml2025.bst`
- `ai_scientist/treesearch/backend/utils.py`
- `ai_scientist/treesearch/utils/config.py`
- `launch_scientist_bfts.py`
- `prompt/icbinb_writeup/system_message.txt`
- `prompt/icbinb_writeup/reflection_prompt.txt`
- `prompt/icbinb_writeup/citation_system_message.txt`
- `prompt/icbinb_writeup/citation_first_prompt.txt`
- `prompt/writeup/citation_first_prompt.txt`
- `prompt/writeup/citation_second_prompt.txt`
- `prompt/icbinb_writeup/citation_second_prompt.txt`
- `prompt/icbinb_writeup/writeup_prompt.txt`
- `prompt/writeup/writeup_prompt.txt`
- `prompt/writeup/system_message.txt`
- `prompt/writeup/citation_system_message.txt`

### deprecated / legacy / unused / experimental / old / v1
- `suppliment_information.txt`
- `LICENSE`
- `launch_scientist_bfts.py`
- `bfts_config.yaml`
- `README.md`
- `ai_scientist/llm.py`
- `ai_scientist/vlm.py`
- `ai_scientist/blank_icbinb_latex/math_commands.tex`
- `ai_scientist/perform_writeup.py`
- `ai_scientist/ideas/i_cant_believe_its_not_betterrealworld.py`
- `ai_scientist/blank_icml_latex/template.tex`
- `ai_scientist/blank_icml_latex/icml2025.sty`
- `ai_scientist/blank_icbinb_latex/fancyhdr.sty`
- `ai_scientist/fewshot_examples/attention.json`
- `ai_scientist/utils/model_params.py`
- `ai_scientist/perform_vlm_review.py`
- `ai_scientist/persona.py`
- `ai_scientist/tools/semantic_scholar.py`
- `ai_scientist/fewshot_examples/132_automated_relational.txt`
- `ai_scientist/ideas/i_cant_believe_its_not_better_openai.json`
- `ai_scientist/fewshot_examples/2_carpe_diem.json`
- `ai_scientist/blank_icbinb_latex/template.tex`
- `ai_scientist/perform_icbinb_writeup.py`
- `ai_scientist/fewshot_examples/attention.txt`
- `ai_scientist/perform_plotting.py`
- `ai_scientist/ideas/i_cant_believe_its_not_better.json`
- `prompt/review/neurips_form.txt`
- `prompt/icbinb_writeup/image_reflection_prompt.txt`
- `prompt/icbinb_writeup/reflection_prompt.txt`
- `ai_scientist/treesearch/journal.py`
- `prompt/icbinb_writeup/writeup_prompt.txt`
- `prompt/writeup/writeup_prompt.txt`
- `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py`
- `ai_scientist/treesearch/utils/metric.py`
- `prompt/treesearch/journal/stage_notes_intro.txt`
- `prompt/treesearch/journal/stage_notes_user_message.txt`
- `prompt/treesearch/journal/summary_user_message.txt`
- `prompt/treesearch/log_summarization/stage_aggregate_prompt.txt`
- `prompt/writeup/system_message.txt`
- `ai_scientist/treesearch/log_summarization.py`
- `prompt/treesearch/journal/summary_intro.txt`
- `prompt/writeup/reflection_prompt.txt`
- `ai_scientist/treesearch/utils/viz_templates/template.js`
- `ai_scientist/treesearch/backend/backend_openai.py`
- `ai_scientist/treesearch/utils/viz_templates/template.html`
- `prompt/treesearch/parallel_agent/improve/introduction.txt`
- `ai_scientist/treesearch/utils/phase_plan.py`
- `ai_scientist/treesearch/utils/tree_export.py`
- `ai_scientist/treesearch/utils/config.py`
- `ai_scientist/treesearch/utils/response.py`
- `prompt/treesearch/parallel_agent/summary/introduction.txt`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/agent_manager.py`
- `prompt/treesearch/parallel_agent/select_plots/introduction.txt`
- `prompt/treesearch/parallel_agent/parse_metrics_prompt/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/change_prompt.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/summary/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/execution_review/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/hyperparam_node/introduction_prefix.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/improve/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/parse_metrics_prompt/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/ablation_node/introduction_prefix.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/response_format/debug.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/hyperparam_tuning_prompt/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/determine_datasets/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/environment/message_cpp.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/environment/packages.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/debug/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/debug/bugfix_improvement_sketch_guideline.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/implementation_guideline/post.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/select_plots/introduction.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/implementation_guideline/dataset.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/define_global_metrics/instructions.txt`
- `prompt/treesearch/parallel_agent/language_adapter/static_cpp/define_global_metrics/introduction.txt`

## Step2: Usage notes for removal candidates
- Language adapter prompts (`prompt/treesearch/parallel_agent/language_adapter/**`), `language_decider.txt`, and related adapter code in `launch_scientist_bfts.py` were only referenced by the commented-out prompt adaptation path. They are not imported or used in split execution.
- `prompt/treesearch/parallel_agent/response_format/split_phase.txt` was not referenced by code (split mode uses `prompt/execution_split_schema.txt`).
- `prompt_adapter` config was only referenced in `launch_scientist_bfts.py` (commented) and in the config schema; split mode never reads it.
- `exec.container_runtime` was not read by the split runner (runtime detection always forces Singularity).

## Step3: Classification (A-E) and decisions
### A) Language adaptation layer (split unreachable)
- Removed adapter logic from `launch_scientist_bfts.py` (language decider, prompt override, prompt snapshotting).
- Removed `prompt/treesearch/parallel_agent/language_adapter/**` prompt overlays.
- Dropped `prompt_adapter` from `bfts_config.yaml` and config schema.

### B) AI Scientist fixed paths (kept)
- Writeup/review/citation modules remain because they are still invoked from `launch_scientist_bfts.py` (guarded by `--skip_writeup` / `--skip_review`).

### C) Unused CLI flags/config
- Removed `--load_code` and `--add_dataset_ref` (no live call sites after split path cleanup).
- Removed `--container_runtime` and `exec.container_runtime` (split runner is Singularity-only).
- Removed `exec.cpp_compile_flags` and `exec.cpp_compiler` (split runner uses LLM compile commands; single-mode C++ no longer configurable here).

### D) Unused utilities/prompts
- Removed `prompt/treesearch/parallel_agent/response_format/split_phase.txt` (unreferenced).
- Removed `redirect_stdout_stderr_to_file` (unused helper in `launch_scientist_bfts.py`).

### E) Unused dependencies
- Removed `wandb` and `seaborn` from `requirements.txt` (no imports in codebase).
