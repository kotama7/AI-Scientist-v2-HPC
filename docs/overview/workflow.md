# Workflow Overview

This document describes the end-to-end HPC-AutoResearch workflow from ideation
through paper review.

## High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HPC-AutoResearch Workflow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ 1. Ideation      │  perform_ideation_temp_free.py (optional)             │
│  │    (Ideation)    │  Workshop description → idea JSON                      │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 2. Launcher      │  launch_scientist_bfts.py                              │
│  │    (Launch)      │  Load ideas → create experiment directory               │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 3. BFTS Experiments (Tree Search)                                     │   │
│  │                                                                       │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │  │ Stage 1      │──▶│ Stage 2      │──▶│ Stage 3      │──▶│ Stage 4      │ │
│  │  │ Initial Impl │   │ Baseline     │   │ Creative     │   │ Ablation     │ │
│  │  │              │   │ Tuning       │   │ Research     │   │ Studies      │ │
│  │  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│  │       │                                                               │   │
│  │       ▼ Per-node execution                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ Phase 0: Planning                                          │    │   │
│  │  │ Phase 1: Download/Install (inside Singularity)             │    │   │
│  │  │ Phase 2: Coding                                            │    │   │
│  │  │ Phase 3: Compile                                           │    │   │
│  │  │ Phase 4: Run                                               │    │   │
│  │  │ → Metrics → Plotting → VLM analysis → Summary              │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  └────────┬─────────────────────────────────────────────────────────────┘   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4. Plot Aggreg.  │  perform_plotting.py                                  │
│  │ (Plot Aggreg.)   │  Select plots from best node                           │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 5. Writeup       │  perform_writeup.py                                   │
│  │    (Writeup)     │  LaTeX generation + citations                          │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 6. Review        │  perform_llm_review.py + perform_vlm_review.py        │
│  │    (Review)      │  NeurIPS review + figure review                        │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Flow

### Step 1: Ideation (optional)

Skip if you already have an idea JSON.

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/himeno_benchmark_challenge.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 3 \
  --num-reflections 5
```

**Input**: workshop description markdown (`*.md`)  
**Output**: idea JSON (`*.json`) in the same directory

### Step 2: Launch experiments

The main entry point.

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 4 \
  --enable_memgpt
```

**What it does**:
1. Loads ideas from the JSON file
2. Creates `experiments/<timestamp>_<idea>_attempt_<id>/`
3. Writes `idea.md`, `idea.json`, `bfts_config.yaml`
4. Starts the BFTS experiment loop

### Step 3: BFTS experiments

Tree search executes experiments automatically.

**Stages**:
- **Stage 1 (`initial_implementation`)**: Initial implementation and validation
- **Stage 2 (`baseline_tuning`)**: Baseline tuning and extra datasets
- **Stage 3 (`creative_research`)**: Creative improvements and planned experiments
- **Stage 4 (`ablation_studies`)**: Ablation studies and risk-factor validation

**Per-node phase flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Node Execution Flow                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 0: Planning                                               │
│  ├── Environment collection (OS, CPU, GPU, compilers, libs)       │
│  ├── Prior execution history                                      │
│  └── Phase 1-4 plan (JSON)                                        │
│           │                                                      │
│           ▼                                                      │
│  Phase 1: Download/Install (inside Singularity)                  │
│  ├── apt-get, pip install                                        │
│  ├── Build from source                                           │
│  └── Iterative installer (max 100 steps)                         │
│           │                                                      │
│           ▼                                                      │
│  Phase 2: Coding                                                  │
│  ├── LLM generates source code                                    │
│  └── Write files to workspace                                     │
│           │                                                      │
│           ▼                                                      │
│  Phase 3: Compile                                                 │
│  ├── Build with gcc/g++/nvcc, etc.                                │
│  └── On error, spawn debug nodes                                  │
│           │                                                      │
│           ▼                                                      │
│  Phase 4: Run                                                     │
│  ├── Execute program                                              │
│  └── Collect outputs (.npy)                                       │
│           │                                                      │
│           ▼                                                      │
│  Post-processing                                                  │
│  ├── Metrics extraction (speedup, accuracy, etc.)                 │
│  ├── Plot code generation + execution                             │
│  ├── VLM analysis (plot quality)                                  │
│  └── Node summary generation                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 4: Plot aggregation

Select and aggregate plots across nodes.

```bash
# Run automatically via launcher, or run manually:
python ai_scientist/perform_plotting.py \
  --folder experiments/<run>
```

**Output**: `experiments/<run>/figures/` and `auto_plot_aggregator.py`

### Step 5: Writeup generation

Generate the LaTeX paper from experiment results.

```bash
# Run automatically via launcher, or:
python generate_paper.py \
  --experiment-dir experiments/<run> \
  --writeup-type normal
```

**Output**: `experiments/<run>/<run>.pdf`

### Step 6: Review

Run a NeurIPS-style review on the generated paper.

**Outputs**:
- `review_text.txt`: text review
- `review_img_cap_ref.json`: figure/caption review

## Skip options

Each stage can be skipped:

```bash
python launch_scientist_bfts.py \
  --skip_plot \       # skip plot aggregation
  --skip_writeup \    # skip writeup generation
  --skip_review \     # skip review
  ...
```

## Minimal verification run

A minimal run configuration:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 2 \
  --skip_plot --skip_writeup --skip_review
```

## Regenerate from an existing experiment

Rebuild plots and writeup from an existing experiment directory:

```bash
python generate_paper.py \
  --experiment-dir experiments/<run> \
  --writeup-type normal \
  --model-agg-plots o3-mini-2025-01-31 \
  --model-writeup o1-preview-2024-09-12
```

## Memory-enabled flow

When `--enable_memgpt` is set, memory management is active in each phase:

```
┌─────────────────────────────────────────────────────────────────┐
│ Additional processing with memory enabled                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. On each phase completion                                     │
│     └── Event recorded in Recall memory                          │
│     └── Error/success details stored in Archival memory           │
│                                                                  │
│  2. During prompt assembly                                       │
│     └── Context injected from Core/Recall/Archival                │
│     └── LLM can write memory with <memory_update> blocks          │
│                                                                  │
│  3. LLM-managed memory                                           │
│     └── idea_md_summary: saved to Core when needed                │
│     └── phase0_summary: saved to Core when needed                 │
│     └── Other important facts saved as LLM decides                │
│                                                                  │
│  4. End of run                                                   │
│     └── final_memory_for_paper.md/json generated                  │
│                                                                  │
│  Note: idea_md_summary and phase0_summary are not auto-injected.  │
│        The LLM must save them via <memory_update>.               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Related Documents

- [quickstart.md](getting-started/quickstart.md) - Quickstart guide
- [execution-modes.md](configuration/execution-modes.md) - Split/Single mode details
