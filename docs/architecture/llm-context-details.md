# LLM Context Details (LLM Call Context Breakdown)

This document provides detailed technical information about what context is
passed to LLM calls at each stage of the research automation pipeline.

## Context Assembly Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Context Assembly                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ System      │   │ Environment │   │ Memory      │           │
│  │ Prompt      │ + │ Injection   │ + │ Context     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                 │                 │                   │
│         v                 v                 v                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Base Context Layer                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              v                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Task/Idea   │   │ Resources   │   │ History     │           │
│  │ Description │ + │ Context     │ + │ Context     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                 │                 │                   │
│         v                 v                 v                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Stage-Specific Layer                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              v                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Instructions│   │ Response    │   │ Prior       │           │
│  │ /Guidelines │ + │ Format      │ + │ Outputs     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Context Components

### 1. System Prompt Layer

**Source**: `prompt/core/system.txt`, `prompt/core/domain_neutral.txt`

```python
# Loaded via prompt_loader.py
system_prompt = load_prompt("core/system")
domain_neutral = load_prompt("core/domain_neutral")
```

**Contents**:
- Agent role definition (researcher, coder, reviewer)
- Persona override from `agent.role_description`
- General behavioral constraints
- Output formatting rules

### 2. Environment Injection

**Source**: `prompt/config/environment/injection.txt`

**Collected at runtime**:
```python
environment_context = {
    "os_release": "...",           # /etc/os-release content
    "cpu_info": "...",             # lscpu output (model, sockets, NUMA)
    "memory_info": "...",          # Total RAM
    "gpu_info": "...",             # nvidia-smi output
    "available_compilers": [...],  # gcc/clang with versions/paths
    "available_libs": [...],       # pkg-config detected libraries
    "network_access": "...",       # available/blocked
    "container_runtime": "...",    # singularity/docker/none
    "singularity_image": "...",    # SIF path
    "workspace_mount": "..."       # /workspace mount point
}
```

### 3. Memory Context (MemGPT)

**Source**: `ai_scientist/memory/memgpt_store.py`

**Injected when `memory.enabled=true`**:
```python
memory_context = {
    "resource_index": "...",        # Separate "Resource Index" section (if available)
    "core": {
        # All keys are LLM-managed via <memory_update> (no reserved keys)
        "optimal_threads": "8",
        "best_compiler": "-O3",
        # ...
    },
    "recall": [...],                # Recent events (window)
    "archival": [...]               # Retrieved archival hits
}
```

**Note**: `RESOURCE_INDEX` is only injected if a resource snapshot or core entry
exists. It is rendered as a separate "Resource Index" section, not within
"Core Memory". All Core Memory keys are LLM-managed.

**Character budgets** (from `bfts_config.yaml`):
- `memory.memory_budget_chars`: total memory context budget.
- `memory.archival_snippet_budget_chars`: per-archival excerpt cap.

### 4. Task/Idea Description

**Source**: `idea.md` (loaded from ideas JSON)

**Contents**:
```markdown
## Abstract
Research objective and hypothesis

## Task goal
Specific task to accomplish

## Experiments
Variables and experimental design

## Evaluation
Metrics and success criteria

## Risk Factors And Limitations
Known challenges and mitigations
```

### 5. Resources Context

**Source**: `data_resources.json`, `resource_memory.py`

**Structure**:
```python
resources_context = {
    "RESOURCE_INDEX": {  # present only if resource index was built
        "digest": "sha256:...",
        "items": [
            {
                "id": "dataset-1",
                "class": "local_data",
                "path": "/workspace/data/...",
                "fetch_status": "available",
                "tree_summary": "...",
                "content_excerpt": "..."
            },
            # ...
        ]
    },
    "resource_items": [...]  # Top-K relevant items
}
```

### 6. History Context

**Source**: Prior execution outputs, phase summaries

**Collected via `_collect_phase0_history()`**:
```python
history_context = {
    "phase_summaries": {
        "phase0": "Planning completed...",
        "phase1": "Dependencies installed...",
        "phase2": "Code generated...",
        "phase3": "Compilation successful...",
        "phase4": "Run completed..."
    },
    "compile_logs": {
        "summary": "...",
        "errors": [...]
    },
    "run_logs": {
        "summary": "...",
        "errors": [...]
    },
    "prior_llm_output": "..."  # Previous LLM response summary
}
```

## Phase-Specific Context

### Phase 0: Planning

```python
context = {
    "introduction": load_prompt("config/phases/phase0_planning")  # or phase0_planning_with_memory when enabled,
    "task": idea_markdown,
    "environment": environment_snapshot,
    "history": prior_phase_summaries,  # From previous attempts
    "resources": resources_context     # If available
}
```

**Output schema**:
When memory is enabled, the response begins with a required
`<memory_update>...</memory_update>` block, followed immediately by the Phase 0
plan JSON. When memory is disabled, the response is JSON only. The JSON plan
follows the schema documented in `prompt/config/phases/phase0_planning.txt`.

Legacy example (plan fields only):
```json
{
    "goal_summary": "...",
    "implementation_strategy": "...",
    "dependencies": {"apt": [...], "pip": [...], "source": [...]},
    "download_commands_seed": [...],
    "compile_plan": {...},
    "compile_commands": [...],
    "run_commands": [...],
    "phase_guidance": {
        "phase1": {"targets": [...], "preferred_commands": [...], "done_conditions": [...]},
        "phase2": {...},
        "phase3": {...},
        "phase4": {...}
    },
    "risks_and_mitigations": [...]
}
```

### Phase 1: Iterative Installer

```python
context = {
    "introduction": load_prompt("config/phases/phase1_installer"),
    "task": idea_markdown,
    "phase_plan": phase0_output,  # Download/compile/run plan
    "constraints": execution_constraints,
    "progress": {
        "step": current_step,
        "max_steps": 12,
        "history": [
            {"command": "apt-get install ...", "exit_code": 0, "stdout_summary": "...", "stderr_summary": "..."},
            # ...
        ]
    },
    "phase0_guidance": phase0_output["phase_guidance"]["phase1"],
    "environment": environment_injection,
    "resources": resources_context,
    "memory": memory_context
}
```

### Phase 2/3/4: Coding/Compile/Run (Combined)

```python
context = {
    # Stage-specific sections
    "introduction": load_prompt(f"agent/parallel/tasks/{stage}/introduction"),
    "research_idea": idea_markdown,
    "memory": memory_context,

    # Prior outputs (varies by stage)
    "prior_code": current_code,           # For debug/improve stages
    "execution_output": run_output,       # stdout/stderr from prior run
    "plot_feedback": vlm_analysis,        # For improve stages
    "time_feedback": execution_time,      # Performance data

    # Split-mode layers
    "system": load_prompt("core/system"),
    "domain": load_prompt("core/domain_neutral"),
    "environment": environment_injection,
    "resources": resources_context,
    "phase0_plan_snippet": {
        "goal_summary": "...",
        "implementation_strategy": "...",
        "dependencies": {...},
        "phase_guidance": {
            "phase2": {...},
            "phase3": {...},
            "phase4": {...}
        },
        "risks_and_mitigations": [...]
    },

    # Instructions
    "guidelines": load_prompt(f"agent/parallel/guidelines/{guideline_type}"),
    "response_format": load_prompt("agent/parallel/response_format/execution_split"),
    "implementation_guidance": task_specific_guidance
}
```

**Output schema** (`prompt/agent/parallel/response_format/execution_split.txt`):
```json
{
    "coding": {
        "workspace": {
            "file_tree": [...],
            "files": {"path": "content", ...}
        }
    },
    "compile": {
        "build_plan": {
            "language": "cpp",
            "compiler": "g++",
            "flags": [...],
            "output_path": "..."
        },
        "commands": [...]
    },
    "run": {
        "commands": [...],
        "expected_outputs": [...]
    }
}
```

## Tree Search Stage Contexts

### Stage 1: Draft

```python
stage_context = {
    "introduction": "Generate initial implementation draft",
    "research_idea": idea_markdown,
    "data_overview": dataset_description,  # Optional
    "memory": memory_context,
    "instructions": {
        "guidelines": implementation_guidelines,
        "response_format": draft_response_format
    }
}
```

### Stage 2: Hyperparam Tuning

```python
stage_context = {
    "introduction": "Optimize hyperparameters for best performance",
    "research_idea": idea_markdown,
    "base_code": best_node_code,
    "tried_history": [
        {"params": {...}, "metrics": {...}},
        # ...
    ],
    "memory": memory_context,
    "instructions": {
        "guidelines": hyperparam_guidelines,
        "response_format": hyperparam_response_format
    }
}
```

### Stage 3: Debug/Improve

```python
stage_context = {
    "introduction": "Debug and improve the implementation",
    "research_idea": idea_markdown,
    "current_code": failing_code,
    "execution_output": error_output,
    "traceback": error_traceback,
    "plot_feedback": vlm_analysis,  # For improve
    "memory": memory_context,
    "instructions": {
        "guidelines": debug_guidelines,
        "response_format": debug_response_format
    }
}
```

### Stage 4: Ablation

```python
stage_context = {
    "introduction": "Conduct ablation study",
    "research_idea": idea_markdown,
    "base_code": best_node_code,
    "ablation_plan": {
        "components_to_ablate": [...],
        "expected_impact": [...]
    },
    "memory": memory_context,
    "instructions": {
        "guidelines": ablation_guidelines,
        "response_format": ablation_response_format
    }
}
```

## Plotting and Review Contexts

### Plot Aggregation

```python
context = {
    "system": load_prompt("output/plotting/system_message"),
    "experiment_code": code_from_best_node,
    "base_plotting_code": optional_template,
    "response_format": plotting_response_format,
    "plotting_guidelines": load_prompt("agent/parallel/guidelines/plotting")
}
```

### VLM Analysis

```python
context = {
    "research_idea": idea_text,
    "plot_images": [base64_encoded_images],  # Multimodal input
    "analysis_prompt": load_prompt("agent/parallel/vlm_analysis")
}
```

### Paper Writeup

```python
context = {
    "system": load_prompt("output/writeup/system_message"),
    "research_idea": idea_markdown,
    "experiment_results": aggregated_results,
    "figures": selected_plots,
    "metrics": extracted_metrics,
    "memory": memory_context,  # Final memory for paper
    "response_format": writeup_format,
    "reflection_rounds": 3
}
```

### LLM Review

```python
context = {
    "system": load_prompt("review/llm/system_base"),
    "paper_content": latex_content,
    "review_form": load_prompt("review/llm/neurips_form"),
    "fewshot_examples": optional_examples,
    "reflection_prompt": load_prompt("review/llm/reflection")
}
```

## Debugging Context Assembly

Enable prompt logging to inspect actual contexts:

```yaml
exec:
  log_prompts: true
```

Prompt logs are written to:
```
experiments/<run>/logs/phase_logs/node_<id>/prompt_logs/
├── phase0_prompt.json
├── phase0_prompt.md
├── phase1_step0_prompt.json
├── phase1_step0_prompt.md
├── stage1_draft_prompt.json
├── stage1_draft_prompt.md
└── ...
```

Each log includes:
- Full assembled prompt text
- System message
- User message(s)
- Response format
- Memory context (if enabled)
- Resource context (if available)
