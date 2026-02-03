# Core Concepts

This document explains the core concepts behind HPC-AutoResearch, including the
system design philosophy and the role of each component.

## 1. Tree Search for Automated Research

### Why tree search?

Research is inherently exploratory. The first idea is rarely the best solution,
and iterative trial-and-error is required to converge on strong results.
HPC-AutoResearch formalizes this process as **tree search**.

```
                          Root (Idea)
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
         Draft 1           Draft 2           Draft 3
            │                 │                 │
       ┌────┼────┐       ┌────┼────┐           ✗
       │    │    │       │    │    │
    Debug Improve ✗   Debug Improve ✗
       │    │           │    │
       ✗   Best        ✗    ─┘
            │
       Hyperparam
            │
       Ablation
            │
         Final
```

### Benefits of the tree structure

1. **Parallel exploration**: multiple implementations can be tried in parallel
2. **Rollback**: if a node fails, the system can return to a parent node and try a different approach
3. **Best-first focus**: successful nodes are prioritized for deeper exploration
4. **History reuse**: prior failures inform future attempts

### Node types

| Node Type | Purpose | Inputs from Parent |
|-----------|---------|--------------------|
| **Draft** | Generate an initial implementation | Idea description |
| **Debug** | Fix errors | Failing code + error logs |
| **Improve** | Improve performance | Working code + metrics |
| **Hyperparam** | Tune parameters | Best code + tuning history |
| **Ablation** | Evaluate component impact | Final code |

### Stage Definition Details

Four main stages are defined in the implementation (`agent_manager.py`):

| Stage | Internal Name | Purpose |
|-------|---------------|---------|
| Stage 1 | `initial_implementation` | Generate initial implementation draft, verify functionality |
| Stage 2 | `baseline_tuning` | Baseline tuning, evaluation on additional datasets |
| Stage 3 | `creative_research` | Creative improvements, execute experiment plans |
| Stage 4 | `ablation_studies` | Ablation studies, verify risk factors |

Stage goals are defined in prompt files:
- `prompt/agent/manager/stages/stage1_goals.txt`
- `prompt/agent/manager/stages/stage2_goals.txt`
- `prompt/agent/manager/stages/stage3_goals.txt`
- `prompt/agent/manager/stages/stage4_goals.txt`

### Node Detailed Attributes

Each node is defined by the `Node` dataclass in `journal.py` with the following key attributes:

| Category | Attributes | Description |
|----------|------------|-------------|
| Basic Info | `id`, `step`, `ctime` | Unique ID, step number, creation time |
| Relations | `parent`, `children`, `branch_id` | Parent node, child nodes, memory branch |
| Code | `code`, `plan`, `phase_artifacts` | Generated code, plan, per-phase artifacts |
| Execution | `_term_out`, `exec_time`, `exc_type` | Output, execution time, exception info |
| Evaluation | `metric`, `analysis`, `is_buggy` | Metrics, analysis results, buggy flag |
| VLM | `plot_analyses`, `vlm_feedback_summary` | Plot analyses, VLM feedback |
| Special | `ablation_name`, `is_seed_node` | Ablation name, seed node flag |
| Inheritance | `inherited_from_node_id`, `worker_sif_path` | Source node ID, reused SIF path |

### Multi-Seed Evaluation

At the completion of each main stage, the best node undergoes multi-seed evaluation:

```
Best Node ──┬──▶ Seed 1 ──▶ Run ──▶ Metrics
            ├──▶ Seed 2 ──▶ Run ──▶ Metrics
            └──▶ Seed 3 ──▶ Run ──▶ Metrics
                           ↓
                 Seed Aggregation Node
                 (mean/std/statistical plots)
```

Configuration (`bfts_config.yaml`):
```yaml
agent:
  multi_seed_eval:
    num_seeds: 3  # Number of evaluation seeds
```

## 2. Split-Phase Architecture

### Why split phases?

In HPC environments, each step from dependency installation to execution is
complex. Splitting phases provides:

1. **Clear responsibilities**: each phase has a single responsibility
2. **Error localization**: it is easy to see which phase failed
3. **Retry strategies**: different retries can be applied per phase
4. **Container-first design**: phases are designed for Singularity execution

### Phase Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase 0: Planning                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purpose: Analyze the environment and produce the Phase 1-4 execution plan    │
│                                                                              │
│ Inputs:                                                                      │
│   - Idea/task description                                                    │
│   - Environment info (OS, CPU, GPU, compilers, libraries)                    │
│   - Prior execution history (if available)                                   │
│                                                                              │
│ Output (phase0_plan.json):                                                   │
│   - goal_summary: summary of objectives                                      │
│   - implementation_strategy: implementation strategy                         │
│   - dependencies: required deps (apt, pip, source)                           │
│   - download_commands_seed: download commands                                │
│   - compile_plan: build configuration                                        │
│   - compile_commands: build commands                                         │
│   - run_commands: execution commands                                         │
│   - phase_guidance: guidance for each phase                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Phase 1: Download/Install                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purpose: Install dependencies and prepare the runtime environment             │
│                                                                              │
│ Execution environment: inside a Singularity container                         │
│                                                                              │
│ Characteristics:                                                             │
│   - Iterative installer: up to 100 steps, incremental build-up                │
│   - Command types: apt-get, pip install, source builds                        │
│   - Progress tracking: step results (exit_code, stdout, stderr)               │
│   - Error recovery: LLM analyzes failures and selects next commands           │
│                                                                              │
│ Write modes:                                                                  │
│   - tmpfs: fast but memory-limited                                            │
│   - overlay: persistent but slower                                            │
│   - none: read-only (deps are pre-baked into base image)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Phase 2: Coding                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purpose: Generate experiment code and write to the workspace                  │
│                                                                              │
│ Output layout:                                                               │
│   workspace/                                                                  │
│   ├── src/                                                                    │
│   │   ├── main.c                                                              │
│   │   └── utils.h                                                             │
│   ├── Makefile                                                                │
│   └── working/                                                                │
│       └── (runtime outputs are generated here)                               │
│                                                                              │
│ LLM output format:                                                           │
│   - file_tree: directory structure                                            │
│   - files: list of {path, mode, content}                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase 3: Compile                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purpose: Build the source code                                                │
│                                                                              │
│ Supported languages/tools:                                                   │
│   - C/C++ (gcc, g++, clang)                                                   │
│   - CUDA (nvcc)                                                               │
│   - Fortran (gfortran)                                                        │
│   - Make/CMake                                                                │
│                                                                              │
│ build_plan structure:                                                        │
│   - language: programming language                                            │
│   - compiler_selected: compiler to use                                        │
│   - cflags: compile flags                                                     │
│   - ldflags: link flags                                                       │
│   - output: output binary path                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Phase 4: Run                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purpose: Execute the built program and collect results                        │
│                                                                              │
│ Expected outputs:                                                            │
│   - working/{experiment_name}_data.npy (dynamic filename)                     │
│   - or files listed in expected_outputs                                       │
│                                                                              │
│ Post-run processing:                                                         │
│   1. Metrics extraction: parse speedup, accuracy, etc. from stdout/stderr     │
│   2. Plot generation: generate and execute visualization code from .npy data  │
│   3. VLM analysis: assess image quality with VLM                              │
│   4. Node summary: generate a consolidated summary                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. MemGPT-Style Hierarchical Memory

### Why hierarchical memory?

LLMs have a **context window** limit. In long-running research workflows, it is
impossible to include everything in every prompt. MemGPT-style memory addresses
this by:

1. **Context management**: injects the most relevant information within budget
2. **Long-term retention**: preserves patterns of success and failure
3. **Branch inheritance**: children inherit parent learning

### Memory layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Memory (Core Layer)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Characteristics: always-injected key context                                 │
│ Capacity: core_max_chars (default 2000 chars, code fallback 2000 chars)       │
│                                                                              │
│ Typical content:                                                             │
│   - RESOURCE_INDEX: digest of available resources (if snapshot created)       │
│   - RESOURCE_INDEX_JSON: JSON form of the resource index                      │
│   - RESOURCE_DIGEST: resource digest                                          │
│   - LLM-set keys: optimal_threads, best_flags, etc. (no reserved keys)        │
│                                                                              │
│ Management:                                                                  │
│   - Importance (1-5): higher stays longer                                     │
│   - TTL: entries with expiration                                              │
│   - Eviction: overflow moves low-importance entries to Archival               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Recall Memory (Recall Layer)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Characteristics: recent event timeline                                       │
│ Capacity: recall_max_events (default 5, code fallback 20)                     │
│                                                                              │
│ Recorded events:                                                             │
│   - node_created: node created                                                │
│   - phase1_complete/failed: Phase 1 results                                   │
│   - compile_complete/failed: compile results                                  │
│   - run_complete/failed: run results                                          │
│   - metrics_extracted: metrics extraction                                     │
│   - node_result: final node result                                            │
│                                                                              │
│ Management:                                                                  │
│   - FIFO: oldest events removed first                                         │
│   - Consolidation: similar events can be summarized                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       Archival Memory (Archival Layer)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Characteristics: long-term searchable storage                                │
│ Capacity: unlimited (SQLite-backed)                                          │
│                                                                              │
│ Search:                                                                      │
│   - FTS5 full-text search (if available)                                      │
│   - Keyword search (fallback)                                                 │
│   - Tag-based search                                                         │
│                                                                              │
│ Common tags:                                                                 │
│   - PHASE0_INTERNAL: Phase 0 plan details (LLM-saved as needed)               │
│   - IDEA_MD: full idea markdown                                               │
│   - PERFORMANCE: performance findings                                        │
│   - ERROR: error patterns                                                     │
│   - LLM_INSIGHT: LLM-recorded insights                                       │
│                                                                              │
│ Injection:                                                                   │
│   - retrieval_k (default 4, code fallback 8): top-k injected to prompt        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Branch inheritance model

```
             ROOT (branch_id: 0)
            ┌─────────────────┐
            │ Core: {...}     │
            │ Recall: [e1,e2] │
            │ Archival: {...} │
            └────────┬────────┘
                     │ fork
        ┌────────────┴────────────┐
        │                         │
   Node A (branch_id: 1)     Node B (branch_id: 2)
  ┌─────────────────┐       ┌─────────────────┐
  │ Core: inherited │       │ Core: inherited │
  │      + updates  │       │      + updates  │
  │ Recall: [e1,e2, │       │ Recall: [e1,e2, │
  │         e3_A]   │       │         e3_B]   │
  │ Archival: read  │       │ Archival: read  │
  └─────────────────┘       └─────────────────┘
        │                         │
        │ Write isolation         │ Write isolation
        │ (A cannot see B)        │ (B cannot see A)
```

## 4. LLM Memory Operations

The LLM can directly manipulate memory using `<memory_update>` blocks:

```json
<memory_update>
{
  "mem_core_set": {
    "optimal_threads": "8",
    "best_compiler_flags": "-O3 -march=native"
  },
  "mem_archival_write": [
    {
      "text": "8 threads are optimal for this workload",
      "tags": ["PERFORMANCE", "THREADING"]
    }
  ],
  "mem_archival_search": {
    "query": "compilation errors",
    "k": 3
  }
}
</memory_update>
```

### Operation Types

| Operation | Target | Description |
|-----------|--------|-------------|
| `mem_core_set` | Core | Set key/value pairs |
| `mem_core_get` | Core | Retrieve key values |
| `mem_core_del` | Core | Delete keys |
| `mem_archival_write` | Archival | Add a record |
| `mem_archival_search` | Archival | Search records |
| `mem_recall_append` | Recall | Append an event |
| `mem_recall_search` | Recall | Search events |
| `consolidate` | All | Consolidate memory |

## 5. Resource System

Injects external data, repositories, and models into prompts and containers:

```json
{
  "local": [
    {
      "name": "dataset",
      "host_path": "/shared/data",
      "mount_path": "/workspace/input/data",
      "read_only": true
    }
  ],
  "github": [
    {
      "name": "library",
      "repo": "https://github.com/org/lib.git",
      "dest": "/workspace/third_party/lib"
    }
  ],
  "items": [
    {
      "name": "template_code",
      "class": "template",
      "source": "local",
      "resource": "dataset",
      "path": "baseline",
      "include_files": ["main.c", "Makefile"]
    }
  ]
}
```

### Class-based injection rules

All classes are present in every phase, but their content injection and tree
rendering differ:

| Class | Phase 0 content | Phase 1 content | Phase 2 content | Phase 3/4 content |
|-------|----------------|----------------|----------------|-------------------|
| template | ✓ (tree+content) | ✓ (tree+content) | ✓ (content) | - |
| document | ✓ (content) | ✓ (content) | ✓ (content) | ✓ (content) |
| setup | ✓ (tree+content) | ✓ (tree+content) | - | - |
| library | meta only | meta only | - | - |
| dataset | meta only | meta only | - | - |
| model | meta only | meta only | - | - |

## 6. Persona System

Customizes the agent's role:

```yaml
agent:
  role_description: "HPC Researcher"
```

Effects:
- `{persona}` tokens are replaced with the configured role
- Applied recursively to all prompts

## 7. Post-Processing Pipeline

After Phase 4 execution, the following post-processing is automatically performed:

### VLM Analysis Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VLM Analysis Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Plot Code Generation                                                     │
│     └── LLM generates visualization code from .npy data                      │
│                                                                              │
│  2. Plot Execution                                                           │
│     └── Execute generated Python code to produce PNG images                  │
│                                                                              │
│  3. Image Encoding                                                           │
│     └── Base64 encode generated images                                       │
│                                                                              │
│  4. VLM Invocation (vlm/clients.py)                                          │
│     Input:                                                                   │
│       - Research idea text                                                   │
│       - Base64 encoded images                                                │
│       - VLM_ANALYSIS_PROMPT_TEMPLATE                                         │
│     Output:                                                                  │
│       - Image quality assessment                                             │
│       - Data visualization appropriateness                                   │
│       - Improvement suggestions                                              │
│                                                                              │
│  5. Result Storage                                                           │
│     └── Store in Node attributes:                                            │
│         - plot_analyses: Analysis results for each plot                      │
│         - vlm_feedback_summary: VLM feedback summary                         │
│         - datasets_successfully_tested: Successfully tested datasets         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Paper Generation Pipeline (Writeup Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Paper Generation Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Collect Experiment Summaries (load_exp_summaries)                        │
│     └── Collect results, metrics, and plots from each node                   │
│                                                                              │
│  2. Plot Aggregation (aggregate_plots)                                       │
│     └── LLM selects best plots and generates aggregation script              │
│     └── Improvement through reflection steps                                 │
│                                                                              │
│  3. Citation Gathering (gather_citations)                                    │
│     └── Uses Semantic Scholar API                                            │
│     └── Automatically search and collect related papers                      │
│                                                                              │
│  4. LaTeX Generation (perform_writeup)                                       │
│     Structure:                                                               │
│       - Abstract                                                             │
│       - Introduction                                                         │
│       - Method                                                               │
│       - Experiments                                                          │
│       - Conclusion                                                           │
│     └── Quality improvement through reflection steps                         │
│                                                                              │
│  5. PDF Generation (compile_latex)                                           │
│     └── Compile with pdflatex                                                │
│     └── Page limit check (detect_pages_before_impact)                        │
│                                                                              │
│  6. Review (Optional)                                                        │
│     └── LLM Review: NeurIPS-style evaluation                                 │
│     └── VLM Review: Figure quality assessment                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metrics Extraction Flow

```
Execution Output (stdout/stderr)
        │
        ▼
   Parse Metrics Plan
   (LLM generates parser code)
        │
        ▼
   Execute Parser
        │
        ▼
   Store MetricValue
   - name: Metric name (e.g., "speedup")
   - value: Numeric value
   - direction: "higher_is_better" / "lower_is_better"
```

## Related Documents

- [glossary.md](glossary.md) - Terminology
- [workflow.md](workflow.md) - Workflow overview
- [../architecture/execution-flow.md](../architecture/execution-flow.md) - Standard execution flow
- [../memory/memory.md](../memory/memory.md) - Memory system details
- [../architecture/resource-files.md](../architecture/resource-files.md) - Resource file details
