# Prompt Structure (Prompt File Layout)

This document describes the organization and contents of prompt templates in the
`prompt/` directory.

## Directory Overview

```
prompt/
├── core/                    # Shared base prompts
│   ├── system.txt           # Global system instructions
│   ├── domain_neutral.txt   # Domain-agnostic guidance
│   └── ai_optional.txt      # Optional AI-specific instructions
├── config/                  # Configuration and environment
│   ├── environment/         # Environment context injection
│   │   ├── injection.txt    # Environment snapshot template
│   │   ├── history_injection.txt  # History injection template
│   │   └── resources_injection.txt # Resource context template
│   ├── phases/              # Phase-specific prompts
│   │   ├── phase0_planning.txt    # Phase 0 planning instructions
│   │   └── phase1_installer.txt   # Phase 1 installer instructions
│   └── memory/              # Memory-related prompts
│       └── compression.txt  # LLM compression instructions
├── agent/                   # Agent-related prompts
│   ├── manager/             # Agent manager prompts
│   │   ├── task_description.txt   # Task description template
│   │   ├── feedback/        # Feedback prompts
│   │   ├── stages/          # Stage-specific prompts
│   │   └── substage/        # Sub-stage prompts
│   └── parallel/            # Parallel agent prompts
│       ├── data_source/     # Data source prompts
│       ├── environment/     # Environment prompts
│       ├── guidelines/      # Implementation guidelines
│       ├── nodes/           # Node-specific prompts
│       ├── response_format/ # Response format templates
│       ├── tasks/           # Task-specific prompts
│       ├── seed_injection.txt    # Seed injection template
│       └── vlm_analysis.txt # VLM analysis prompt
├── ideation/                # Idea generation prompts
│   ├── system_prompt.txt    # Ideation system prompt
│   ├── generation_prompt.txt # Idea generation prompt
│   ├── reflection_prompt.txt # Idea reflection prompt
│   └── finalize_tool_description.txt # Tool finalization
├── journal/                 # Journal and logging prompts
│   ├── best_node/           # Best node selection
│   ├── journal2report/      # Journal to report conversion
│   ├── log_summarization/   # Log summarization prompts
│   ├── stage_notes/         # Stage notes templates
│   └── summary/             # Summary prompts
├── output/                  # Output generation prompts
│   ├── plotting/            # Plot generation prompts
│   │   ├── system_message.txt     # Plotting system prompt
│   │   ├── aggregator_prompt.txt  # Plot aggregation prompt
│   │   └── reflection_prompt.txt  # Plot reflection prompt
│   └── writeup/             # Paper writeup prompts
│       ├── system_message.txt     # Writeup system prompt
│       ├── writeup_prompt.txt     # Main writeup prompt
│       ├── reflection_prompt.txt  # Writeup reflection prompt
│       └── citation/        # Citation prompts
└── review/                  # Review prompts
    ├── llm/                 # LLM paper review
    │   ├── neurips_form.txt # NeurIPS review form
    │   ├── reflection.txt   # Review reflection
    │   └── ...              # Other review templates
    └── vlm/                 # VLM image review
        ├── img_review.txt   # Image review prompt
        ├── img_cap_selection.txt  # Caption selection
        └── ...              # Other VLM templates
```

## Core Prompts (prompt/core/)

### system.txt
Global system instructions injected into all LLM calls.

```
You are an AI researcher conducting experiments. Follow instructions precisely.
Generate well-structured, executable code. Document your reasoning.
```

### domain_neutral.txt
Domain-agnostic guidance for scientific research.

```
Focus on reproducibility. Use standard metrics. Report both positive and
negative results. Follow best practices for your domain.
```

## Phase Prompts (prompt/config/phases/)

### phase0_planning.txt
Instructions for Phase 0 planning. Defines the planning schema and constraints
for generating download/compile/run plans.

Key sections:
- Environment analysis instructions
- Plan output format (JSON schema)
- Dependency resolution rules
- Risk assessment guidance

### phase1_installer.txt
Instructions for Phase 1 iterative installation. Guides step-by-step dependency
installation and environment setup.

Key sections:
- Command categories (apt-get, pip, source builds)
- Progress tracking format
- Error handling rules
- Done conditions

## Agent Prompts (prompt/agent/)

### manager/stages/
Stage-specific goal prompts:
- `stage1_goals.txt`: Initial exploration goals
- `stage2_goals.txt`: Refinement goals
- `stage3_goals.txt`: Optimization goals
- `stage4_goals.txt`: Final validation goals
- `stage_progression_eval.txt`: Stage advancement criteria

### parallel/tasks/
Task-specific prompts for parallel agent:

| Directory | Purpose |
|-----------|---------|
| `draft/` | Initial code drafting |
| `debug/` | Bug fixing and debugging |
| `improve/` | Performance improvement |
| `hyperparam_tuning/` | Hyperparameter optimization |
| `ablation_analysis/` | Ablation study design |
| `parse_metrics/` | Metric extraction |
| `seed_plotting/` | Multi-seed plot generation |
| `select_plots/` | Plot selection for paper |

### parallel/response_format/
Response format templates:
- `default.txt`: Standard response format
- `execution_split.txt`: Split execution JSON schema
- `debug.txt`: Debug response format
- `ablation.txt`: Ablation response format
- `hyperparam.txt`: Hyperparameter response format

## Memory Prompts (prompt/config/memory/)

### compression.txt
LLM compression instructions for intelligent memory truncation.

Template placeholders:
- `{text}`: Original text to compress
- `{max_chars}`: Target character limit
- `{current_chars}`: Current text length
- `{context_hint}`: Description of text context

```
Compress the following {context_hint} text to fit within {max_chars} characters.
Preserve key facts, metrics, and numerical values. Output only compressed text.
```

## Output Prompts (prompt/output/)

### writeup/system_message.txt
Paper writeup system prompt. Defines the role of paper author and writing style.

### writeup/writeup_prompt.txt
Main writeup generation prompt. Includes:
- Section structure (Abstract, Introduction, Method, Experiments, Conclusion)
- LaTeX formatting guidelines
- Citation requirements
- Figure/table integration rules

### plotting/aggregator_prompt.txt
Plot aggregation instructions for selecting and combining experiment plots.

## Review Prompts (prompt/review/)

### llm/neurips_form.txt
NeurIPS-style review form template. Includes:
- Summary section
- Strengths/weaknesses evaluation
- Technical soundness scoring
- Novelty/significance assessment
- Recommendation (Accept/Reject)

### vlm/img_review.txt
VLM-based image review prompt for figure quality assessment.

## Placeholder Variables

Common placeholders across prompt templates:

| Placeholder | Description |
|-------------|-------------|
| `{persona}` | Agent role from `agent.role_description` |
| `{idea}` | Research idea markdown content |
| `{code}` | Generated/current code |
| `{output}` | Execution output/logs |
| `{metrics}` | Extracted metrics |
| `{plots}` | Plot descriptions/paths |
| `{memory}` | MemGPT memory context |
| `{resources}` | Resource file context |
| `{environment}` | Environment snapshot |
| `{history}` | Execution history |

## Loading Prompts

Prompts are loaded via `ai_scientist/prompt_loader.py`:

```python
from ai_scientist.prompt_loader import load_prompt, format_prompt

# Load a prompt file
system = load_prompt("core/system")  # .txt extension optional

# Load and format with placeholders
prompt = format_prompt("agent/parallel/tasks/draft/introduction",
                       idea=idea_text, persona="HPC Researcher")
```

## Customization

To customize prompts:

1. Edit files directly in `prompt/` directory
2. Set `AI_SCIENTIST_PROMPT_DIR` environment variable to use custom prompt root
3. Use `agent.role_description` in config to change persona without editing files
