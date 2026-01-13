# LLM Context (What We Pass In)

This section summarizes what each LLM call receives. The goal is to make prompt
construction transparent when debugging behavior or adjusting resources.


## Where prompts live

Prompt templates are stored under `prompt/` and assembled per stage. The most
commonly edited files include:

- `prompt/base_system.txt` (global system instructions).
- `prompt/treesearch/parallel_agent/` (stage introductions and response formats).
- `prompt/writeup/` and `prompt/icbinb_writeup/` (writeup pipelines).
- `prompt/review/` (paper and plot review prompts).

## Split-phase planning and install

- Phase 0 planning (split): Introduction + Task + History (phase summaries,
  compile/run logs and errors, prior LLM outputs) + Environment snapshot
  (OS/CPU/GPU, compilers/libs, network, container) + optional Resources.
- Phase 1 iterative install (split): Introduction + Task + Phase plan
  (download/compile/run) + Constraints + Progress history + optional Phase 0
  guidance + Environment injection + Resources.

## Phase-by-phase context injection summary

Each phase receives specific context to accomplish its goal. The table below
summarizes what is injected per phase:

| Phase | Role | Injected Context |
|-------|------|------------------|
| **Phase 0** | Planning | Introduction (`phase0_whole_planning.txt`) + Task + History (phase summaries, compile/run logs, errors, prior LLM outputs) + Environment (OS, CPU, GPU, compilers, libs, network, container info) + Memory + Resources |
| **Phase 1** | Download/Install | Introduction (`phase1_iterative_installer.txt`) + Task + Phase plan (download/compile/run commands from Phase 0) + Constraints + Progress (step index, max steps, command history with exit codes/stdout/stderr) + Phase 0 plan snippet (phase1 guidance: targets, preferred_commands, done_conditions) + Environment injection + Resources + Memory |
| **Phase 2** | Coding | Stage-specific sections + Phase 0 plan snippet (goal_summary, implementation_strategy, dependencies, phase2-4 guidance) + System/Domain prompts + Environment injection + Resources + Memory + Instructions (guidelines, response format, implementation guidance) |
| **Phase 3** | Compile | Same as Phase 2 (combined in single LLM call with Phase 2/4; `coding/compile/run` artifacts produced together per `execution_split_schema.txt`) |
| **Phase 4** | Run | Same as Phase 2/3 (combined in single LLM call) |

### Phase 0: Planning context details

- **Introduction**: `prompt/phase0_whole_planning.txt` (schema and constraints)
- **Task**: Research idea/experiment description
- **History**: Collected from previous runs via `_collect_phase0_history()`:
  - Phase summaries (phase0-4 outcomes from prior attempts)
  - Compile log summary and errors
  - Run log summary and errors
  - Prior LLM output summary
- **Environment**: Collected at runtime from Singularity container:
  - `os_release`, `cpu_info`, `memory_info`, `gpu_info`
  - `available_compilers` (gcc, clang, etc. with versions/paths/flags)
  - `available_libs` (pkg-config detected libraries)
  - `network_access` (available/blocked)
  - `container_runtime`, `singularity_image`, `workspace_mount`
- **Memory**: Optional memory context from MemoryManager
- **Resources**: Resource file context (local data, GitHub repos, HuggingFace models)

### Phase 1: Iterative installer context details

- **Introduction**: `prompt/phase1_iterative_installer.txt` (rules and command categories)
- **Task**: Research idea/experiment description
- **Phase plan**: From Phase 0 output:
  - `download_commands_seed`: Initial download commands
  - `compile_plan`: Build configuration (language, compiler, flags)
  - `compile_commands`: Build commands
  - `run_commands`: Execution commands
- **Constraints**: From `execution_split_schema.txt` (sudo allowed, write paths, etc.)
- **Progress**: Step-by-step history:
  - `step`: Current step index
  - `max_steps`: Maximum allowed steps (default 12)
  - `history`: List of `{command, exit_code, stdout_summary, stderr_summary}`
- **Phase 0 guidance for Phase 1**:
  - `targets`: Dependencies to install/fetch
  - `preferred_commands`: Ordered command list
  - `done_conditions`: Verification commands (dpkg, ls, python -c import)
- **Environment injection**: Compiler/library availability rendered via `environment_injection` template
- **Resources**: Resource bindings and context for the current phase
- **Memory**: Memory context from MemoryManager (task_hint: `phase1_iterative`)

### Phase 2/3/4: Coding/Compile/Run context details

These phases are combined in a single LLM call during tree-search stages
(draft/debug/improve/hyperparam/ablation). The response includes structured
JSON following `prompt/execution_split_schema.txt` with:
- `coding.workspace`: File tree and contents to generate
- `compile.build_plan`: Language, compiler, flags, output path
- `compile.commands`: Build commands
- `run.commands`: Execution commands
- `run.expected_outputs`: Output files to validate

Context injected via `_apply_split_prompt_layers()`:
- **System**: `prompt/base_system.txt`
- **Domain**: Domain-neutral prompt
- **Environment injection**: Available compilers/libs as JSON
- **Resources**: Resource context for phase2
- **Memory**: Memory context from MemoryManager (if not already injected by stage;
  uses stage-specific task_hint: `phase2_draft`, `phase2_debug`, `phase2_improve`,
  `phase2_hyperparam`, `phase2_ablation`)
- **Phase 0 plan snippet** (excluding phase1 guidance):
  - `goal_summary`
  - `implementation_strategy`
  - `dependencies` (apt, pip, source)
  - `phase_guidance` for phase2, phase3, phase4
  - `risks_and_mitigations` (if present)
- **Instructions**: Phase guidance + response format + implementation guidelines

## Split-phase coding/compile/run (Phase 2/3/4)

In split mode, Phase 2/3/4 artifacts are produced in the same LLM call that
drives each tree-search stage (draft/debug/improve/hyperparam/ablation). There
is no separate prompt per phase; instead the response includes structured JSON
for `coding`, `compile`, and `run` (plus `download` for Phase 1) following
`prompt/execution_split_schema.txt`.

The prompt content for these calls includes:

- Stage-specific sections (Introduction + Research idea + Memory, plus previous
  code/execution output/plot feedback depending on the stage).
- Stage-specific Instructions (guidelines, response format for the stage, and
  implementation guidance).
- Split-mode layers: System + Domain prompts, Environment injection, Resources
  (injected for Phase 2), and the Phase 0 plan snippet (goal summary,
  implementation strategy, dependencies, phase guidance for Phase 2-4, and
  risks if present).

## Tree search stages

- Stage 1/3 draft/debug/improve: Introduction + Research idea + Memory, plus
  prior code/execution output/plot+time feedback as applicable; Instructions
  (guidelines + response format + impl/phase guidance); optional Data Overview;
  split mode adds System/Domain + Environment injection + Resources + Phase 0
  plan snippet.
- Stage 2 hyperparam + Stage 4 ablation: Introduction (idea) + base code + tried
  history + stage requirements/response format; split mode adds System/Domain +
  Environment injection + Resources + Phase 0 plan snippet.

## Plotting and VLM passes

- Plotting: Response format + plotting guideline (experiment code + optional
  base plotting code).
- Plot selection: Introduction + plot paths.
- VLM analysis: research idea text + selected plot images (base64).
- Dataset success check: plot analyses + VLM summary + original plotting code +
  response format.

## Execution review and metrics

- Execution review: Introduction + research idea + implementation + execution
  output.
- Parse-metrics plan: original code + prior parse errors/code + instructions +
  example parser + response format.
- Metric extraction: parser execution output.

## Node summary

- Introduction + research idea + implementation + plan + execution output +
  analysis + metric + plot analyses + VLM feedback.

## Resource injection

Resource files can mount data and inject context into prompts. See
`docs/resource-files.md` for the schema and injection rules.

## Inspecting prompt logs

When `exec.log_prompts` is true (default), per-node prompt logs are written to:

- `experiments/<run>/logs/<index>-<exp_name>/phase_logs/node_<id>/prompt_logs/`

Each directory includes JSON and Markdown renderings of the prompts used for
that node and stage, which is the fastest way to verify what context was
actually injected.
