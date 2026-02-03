#!/bin/bash
#SBATCH --job-name=<JOB_NAME>
#SBATCH -p <PARTITION_NAME>
#SBATCH -N <NUM_NODES>
#SBATCH -t <TIME_LIMIT>

# === Environment ===
source ~/miniconda3/bin/activate
conda activate ai_scientist

export S2_API_KEY="YOUR_S2_API_KEY_HERE"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

export AI_SCIENTIST_ROOT="<PATH_TO_YOUR_REPO_ROOT>"

# Singularity check
if command -v module >/dev/null 2>&1; then
  module load singularity >/dev/null 2>&1 || true
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "[FATAL] singularity not found in PATH." >&2
  exit 1
fi

# Singularity container setup (CPU version - no CUDA binds)
export AI_SCIENTIST_SIF="${AI_SCIENTIST_SIF:-$HOME/workplace/docker/ai-scientist-worker-nv.sif}"
if [ ! -f "$AI_SCIENTIST_SIF" ]; then
  echo "[FATAL] AI_SCIENTIST_SIF not found at: $AI_SCIENTIST_SIF" >&2
  exit 1
fi

export AI_SCIENTIST_USE_INSTANCE=0
export SINGULARITYENV_PATH="<YOUR_PATH_HERE>"
export SINGULARITYENV_DEBIAN_FRONTEND=noninteractive

# Singularity cache/tmp directories
LOCAL_TMP_BASE="/var/tmp/${USER:-tmp}/ai-sci-singularity"
mkdir -p "$LOCAL_TMP_BASE/cache" "$LOCAL_TMP_BASE/tmp"
chmod 700 "$LOCAL_TMP_BASE" "$LOCAL_TMP_BASE/cache" "$LOCAL_TMP_BASE/tmp"
export TMPDIR="$LOCAL_TMP_BASE/tmp"
export SINGULARITY_CACHEDIR="$LOCAL_TMP_BASE/cache"
export SINGULARITY_TMPDIR="$LOCAL_TMP_BASE/tmp"

export AI_SCIENTIST_RUN_ROOT="${AI_SCIENTIST_RUN_ROOT:-/var/tmp/${USER:-tmp}/ai-sci-runs}"
mkdir -p "$AI_SCIENTIST_RUN_ROOT"

export AI_SCIENTIST_WRITABLE_MODE="${AI_SCIENTIST_WRITABLE_MODE:-tmpfs}"

# Resources file
RESOURCES_FILE="${RESOURCES_FILE:-$REPO_ROOT/data_resources.json}"
if [ ! -f "$RESOURCES_FILE" ]; then
  echo "[FATAL] resources file not found: $RESOURCES_FILE" >&2
  exit 1
fi

# Run experiment (CPU mode)
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/himeno_benchmark_challenge_extra.json" \
  --writeup-type auto \
  --writeup-reflections 15 \
  --model_writeup gpt-5.2 \
  --model_writeup_small gpt-5.2 \
  --model_citation gpt-5.2 \
  --model_review gpt-5.2 \
  --model_agg_plots gpt-5.2 \
  --model_agg_plots_ref 20 \
  --num_cite_rounds 20 \
  --resources "$RESOURCES_FILE" \
  --phase1_max_steps 100 \
  --num_workers 4 \
  --enable_memgpt \
  --singularity_image "$AI_SCIENTIST_SIF" \
  --per_worker_sif true \
  --use_fakeroot true \
  --writable_mode "$AI_SCIENTIST_WRITABLE_MODE" \
  --keep_sandbox false \
  --use_gpu false
