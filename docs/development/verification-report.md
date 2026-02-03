# Documentation Verification Report

Updated: 2026-02-02

This report summarizes the current consistency between `docs/` and the
implementation as of this update pass.

## Summary

| Component | Docs | Implementation | Status | Notes |
|-----------|------|----------------|--------|-------|
| Tree search stages | overview/concepts.md, overview/workflow.md | agent_manager.py | ✅ | Stage names aligned to `initial_implementation`, `baseline_tuning`, `creative_research`, `ablation_studies` (substage prefixes `stage_1_...`) |
| Log/output layout | configuration/outputs.md | bfts_utils.py, run_io.py, parallel_agent.py | ✅ | `experiments/<run>/logs/` root, `stage_<stage_name>/`, `phase_logs/node_<id>/`, `unified_tree_viz.html` |
| Experiment results | configuration/outputs.md | parallel_agent.py | ✅ | Stored under `experiments/logs/<run>/experiment_results/...` |
| Memory defaults | configuration/configuration.md, memory/memory.md | bfts_config.yaml, memgpt_store.py | ✅ | Defaults: core 2000, recall 5, retrieval 4, max_compression_iterations 5 |
| Memory update schema | memory/* docs | prompt/*, memgpt_store.py | ✅ | LLM uses `mem_*` keys; internal normalization documented |
| Resource snapshot | architecture/resource-files.md, memory/* docs | resource_memory.py | ✅ | Snapshot/index is **optional** and not auto-generated |
| CLI flags | configuration/cli-entry-points.md | launch_scientist_bfts.py, generate_paper.py | ✅ | underscore flags on launcher, hyphen flags on generate_paper |
| Review bias | configuration/review-bias.md, cli-entry-points.md | generate_paper.py, launch_scientist_bfts.py | ✅ | `--review-bias` / `--review_bias`: neg/pos/neutral (default: neutral) |
| Prompt logs | architecture/llm-context.md | parallel_agent.py | ✅ | `logs/phase_logs/node_<id>/prompt_logs/` plus `runs/workers/.../prompt_logs/` |

## Known implementation quirks (not doc bugs)

These are code-path mismatches or legacy paths that remain in the codebase:

- `launch_scientist_bfts.py` and `ai_scientist/output/writeup.py` still look for
  summaries and experiment results under legacy `logs/0-run/` paths; current
  runs write summaries to `experiments/<run>/logs/` and experiment results to
  `experiments/logs/<run>/experiment_results/`.
- `phase0_internal_info.json` is referenced by `memgpt_store.py` for fallback
  Phase 0 summaries, but the main pipeline does not generate it.

## Notes

- Docs now reflect the actual log directory structure, memory defaults, and
  memory update schema used by the prompts and `apply_llm_memory_updates`.
- Resource index behavior is documented as optional because no default
  snapshotting hook exists in the current pipeline.
- Review bias feature (added 2026-02-02): Changed default from `neg` (strict)
  to `neutral` (balanced) to provide fairer automated reviews. Three modes
  available via command-line flag.
