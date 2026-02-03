# Worker-Node Transition Flow

This document describes how workers transition between nodes during tree search,
including the Singularity container execution environment setup.

## Overview

The HPC-AutoResearch system uses a parallel worker architecture where multiple
worker processes execute nodes concurrently. Each worker:

1. Receives a parent node (or `None` for root node creation)
2. Creates a child node through LLM-guided code generation
3. Executes the child node in a Singularity container
4. Returns results to the main process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN PROCESS (Coordinator)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Node Selection (_select_parallel_nodes)                          │   │
│  │                                                                      │   │
│  │    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐        │   │
│  │    │ Draft Nodes │  OR  │ Debug Nodes │  OR  │ Best Nodes  │        │   │
│  │    │ (new roots) │      │ (buggy fix) │      │ (improve)   │        │   │
│  │    └──────┬──────┘      └──────┬──────┘      └──────┬──────┘        │   │
│  │           │                    │                    │               │   │
│  │           └────────────────────┼────────────────────┘               │   │
│  │                                ▼                                    │   │
│  │                 [N nodes selected for parallel processing]          │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. Task Submission (step → worker_manager.submit)                   │   │
│  │                                                                      │   │
│  │    for each selected node:                                          │   │
│  │      node_data = node.to_dict()                                     │   │
│  │      task_id = worker_manager.submit(_process_node_wrapper, ...)    │   │
│  │                                                                      │   │
│  │    ┌─────────────────────────────────────────────────────────────┐  │   │
│  │    │ WorkerManager (multiprocessing.Process pool)                │  │   │
│  │    │                                                              │  │   │
│  │    │   task_queue ──────────────────────────────────────────────>│  │   │
│  │    │                                                              │  │   │
│  │    │   Worker-0    Worker-1    Worker-2    Worker-N              │  │   │
│  │    │      │           │           │           │                  │  │   │
│  │    │      └───────────┴───────────┴───────────┘                  │  │   │
│  │    │                       │                                      │  │   │
│  │    │   <──────────────────────────────────────── result_queue    │  │   │
│  │    └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. Result Collection (wait_for_results)                             │   │
│  │                                                                      │   │
│  │    results = worker_manager.wait_for_results(task_ids, timeout)     │   │
│  │    for result in results:                                           │   │
│  │        child_node = Node.from_dict(result_data, journal)            │   │
│  │        journal.append(child_node)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Worker Process Flow

Each worker process executes `_process_node_wrapper` which handles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WORKER PROCESS (_process_node_wrapper)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. GPU Assignment                                                           │
│     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)                        │
│                                                                              │
│  2. Memory Manager Initialization                                            │
│     memory_manager = MemoryManager(db_path, memory_cfg)                     │
│     memory_manager.set_root_branch_id(root_branch_id)                       │
│                                                                              │
│  3. Workspace Setup                                                          │
│     workspace = cfg.workspace_dir / f"worker_{worker_id}"                   │
│     os.makedirs(workspace, exist_ok=True)                                   │
│                                                                              │
│  4. Parent Workspace Inheritance                                             │
│     if parent_workspace_path:                                               │
│         shutil.copytree(parent_workspace, workspace, ...)                   │
│         # Copies: src/, bin/, working/                                      │
│         # Skips: input/data (mounted), .pydeps, __pycache__                 │
│                                                                              │
│  5. Container Environment Discovery                                          │
│     ExecutionEnvironment → collect_available_compilers, libs, etc.          │
│                                                                              │
│  6. WorkerAgent Creation                                                     │
│     worker_agent = WorkerAgent(cfg, workspace, memory_manager, ...)         │
│                                                                              │
│  7. Memory Branch Fork                                                       │
│     child_branch_id = uuid.uuid4().hex                                      │
│     memory_manager.mem_node_fork(parent_branch_id, child_branch_id)         │
│     worker_agent.branch_id = child_branch_id                                │
│                                                                              │
│  8. Phase 0 Planning (if split mode)                                        │
│     phase0_plan = _execute_phase0_planning(...)                             │
│                                                                              │
│  9. Node Processing (based on parent state)                                  │
│     ┌────────────────────────────────────────────────────────┐              │
│     │ if parent_node is None:                                │              │
│     │     child_node = worker_agent._draft()                 │              │
│     │ elif parent_node.is_buggy:                             │              │
│     │     child_node = worker_agent._debug(parent_node)      │              │
│     │ elif stage_name.startswith("2_"):                      │              │
│     │     child_node = worker_agent._generate_hyperparam_... │              │
│     │ elif stage_name.startswith("4_"):                      │              │
│     │     child_node = worker_agent._generate_ablation_node  │              │
│     │ else:                                                  │              │
│     │     child_node = worker_agent._improve(parent_node)    │              │
│     └────────────────────────────────────────────────────────┘              │
│                                                                              │
│  10. Code Execution in Singularity                                           │
│      exec_env.run(commands, cwd=workspace)                                  │
│                                                                              │
│  11. Result Return                                                           │
│      return child_node.to_dict()                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Node Selection Algorithm

The `_select_parallel_nodes` method selects N nodes for parallel processing:

```python
def _select_parallel_nodes(self) -> List[Optional[Node]]:
    nodes_to_process = []
    processed_trees = set()

    while len(nodes_to_process) < self.num_workers:
        # 1. Check if more draft nodes needed (root creation)
        if len(journal.draft_nodes) < num_drafts:
            nodes_to_process.append(None)  # Signal: create new root
            continue

        # 2. Debug phase (with probability)
        if random.random() < debug_prob:
            debuggable_nodes = [n for n in buggy_nodes if n.is_leaf and ...]
            if debuggable_nodes:
                nodes_to_process.append(random.choice(debuggable_nodes))
                continue

        # 3. Stage-specific handling
        if stage_name.startswith("4_"):
            nodes_to_process.append(best_stage3_node)
        elif stage_name.startswith("2_"):
            nodes_to_process.append(best_stage1_node)
        else:
            # Best-first search for improvement
            best_node = journal.get_best_node(cfg=cfg)
            nodes_to_process.append(best_node)

    return nodes_to_process
```

## Singularity Container Execution

### Container Setup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ExecutionEnvironment (Singularity)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Configuration:                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ workspace: /path/to/experiments/<run>/worker_0                         │ │
│  │ image: /path/to/template/base.sif                                      │ │
│  │ workspace_mount: /workspace                                             │ │
│  │ gpu_id: 0 (or None)                                                     │ │
│  │ enable_writable_tmpfs: true                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Bind Mounts:                                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Host                           Container                                │ │
│  │ ────                           ─────────                                │ │
│  │ worker_0/                  →   /workspace/                              │ │
│  │ worker_0/src/              →   /workspace/src/                          │ │
│  │ worker_0/working/          →   /workspace/working/                      │ │
│  │ worker_0/input/data/       →   /workspace/input/data/ (if configured)  │ │
│  │ <resource_binds>           →   (additional data mounts)                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Environment Variables:                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CUDA_VISIBLE_DEVICES=0                                                  │ │
│  │ PATH=/usr/local/cuda/bin:/usr/local/sbin:...                           │ │
│  │ LD_LIBRARY_PATH=/usr/local/cuda/lib64:...                              │ │
│  │ PYTHONPATH=/workspace/.pydeps                                          │ │
│  │ PYTHONNOUSERSITE=1                                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Command Execution Flow

```
Host Process                          Singularity Container
─────────────                         ─────────────────────

exec_env.run(cmd, cwd)
        │
        ▼
    ┌───────────────────┐
    │ Build command:    │
    │ singularity exec  │
    │   --bind ...      │
    │   --nv            │
    │   --writable-tmpfs│
    │   image.sif       │
    │   bash -lc "cmd"  │
    └─────────┬─────────┘
              │
              ▼
    subprocess.run(...)  ──────────────>  ┌─────────────────────┐
                                          │ Container Process   │
                                          │                     │
                                          │ 1. Load environment │
                                          │ 2. cd /workspace    │
                                          │ 3. Execute command  │
                                          │ 4. Write outputs    │
                                          │                     │
                                          └─────────┬───────────┘
                                                    │
    <──────────────────────────────────────────────┘
    │
    ▼
CompletedProcess(
    returncode=...,
    stdout=...,
    stderr=...
)
```

## Memory Branch Fork

When a worker creates a child node, it also forks the memory branch:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY BRANCH HIERARCHY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          root_branch_id                                      │
│                               │                                              │
│            ┌──────────────────┼──────────────────┐                          │
│            │                  │                  │                          │
│            ▼                  ▼                  ▼                          │
│      child_branch_1     child_branch_2     child_branch_3                   │
│      (worker_0)         (worker_1)         (worker_2)                       │
│            │                  │                  │                          │
│     ┌──────┴──────┐          │           ┌──────┴──────┐                    │
│     ▼             ▼          ▼           ▼             ▼                    │
│  grandchild_1  grandchild_2  ...      grandchild_4  grandchild_5            │
│                                                                              │
│  Inheritance:                                                                │
│  - Child inherits Core Memory (visible)                                     │
│  - Child inherits Recall Memory (visible)                                   │
│  - Child can search Archival Memory (including ancestors)                   │
│  - Child writes are ISOLATED (siblings don't see each other)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Code:
```python
# In _process_node_wrapper
child_branch_id = uuid.uuid4().hex
parent_branch_id = parent_node.branch_id if parent_node else root_branch_id

# Build ancestor chain for inheritance
ancestor_chain = []
current = parent_node
while current:
    if current.branch_id:
        ancestor_chain.append(current.branch_id)
    current = current.parent
ancestor_chain = ancestor_chain[::-1]  # root-to-parent order

# Fork memory branch
memory_manager.mem_node_fork(parent_branch_id, child_branch_id, ancestor_chain)
worker_agent.branch_id = child_branch_id
```

## Workspace Inheritance

Cross-stage file inheritance preserves compiled binaries and source code:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKSPACE INHERITANCE FLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1 (Creative Research)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ node_abc/                                                           │    │
│  │   src/                  # Source code generated by LLM              │    │
│  │   bin/                  # Compiled binaries                         │    │
│  │   working/              # Experiment outputs (results, data)        │    │
│  │   input/data/           # Mounted dataset (read-only)               │    │
│  └──────────────────────────────────┬──────────────────────────────────┘    │
│                                     │                                        │
│                          (best node selected)                                │
│                                     │                                        │
│                                     ▼                                        │
│  Stage 2 (Hyperparam Tuning)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ node_def/                                                           │    │
│  │   src/        ← COPIED from node_abc/src/                          │    │
│  │   bin/        ← COPIED from node_abc/bin/                          │    │
│  │   working/    ← COPIED from node_abc/working/                      │    │
│  │   input/data/ ← PRESERVED (symlink to original mount)              │    │
│  └──────────────────────────────────┬──────────────────────────────────┘    │
│                                     │                                        │
│                          (best node selected)                                │
│                                     │                                        │
│                                     ▼                                        │
│  Stage 3 (Creative Research 2)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ node_ghi/                                                           │    │
│  │   src/        ← COPIED from node_def/src/                          │    │
│  │   bin/        ← COPIED from node_def/bin/                          │    │
│  │   working/    ← COPIED from node_def/working/                      │    │
│  │   input/data/ ← PRESERVED                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Skipped during copy:                                                        │
│  - .workspace.lock   (lock file for current workspace)                       │
│  - .pydeps           (pip packages, reinstalled per worker)                  │
│  - __pycache__       (bytecode cache, auto-regenerated)                      │
│  - input/data/       (mounted data, not copied)                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage Best Preservation

After each stage completes, the best node's workspace is preserved:

```
experiments/<run>/<exp>/
├── stage_best/
│   ├── 1_creative_research_1_first_attempt/
│   │   ├── src/
│   │   ├── bin/
│   │   └── working/
│   ├── 2_hyperparam_tuning/
│   │   ├── src/
│   │   ├── bin/
│   │   └── working/
│   ├── 3_creative_research_2/
│   │   └── ...
│   └── 4_ablation_studies/
│       └── ...
└── node_logs/          # Temporary (cleaned after stage)
    └── node_abc123/
```

## Timeout Recovery

Workers save intermediate results for timeout recovery:

```python
def _save_intermediate_result(child_node, stage: str = "unknown"):
    """Save intermediate result for timeout recovery."""
    intermediate_data = {
        "plan": child_node.plan,
        "code": child_node.code,
        "analysis": child_node.analysis,
        "exc_type": child_node.exc_type,
        "is_buggy": child_node.is_buggy,
        "metric": child_node.metric.to_dict() if child_node.metric else None,
        "stage": stage,
        "node_id": child_node.id,
        "branch_id": child_node.branch_id,
        # ... more fields
    }
    intermediate_result_path.write_text(json.dumps(intermediate_data))

# Called at key points:
_save_intermediate_result(child_node, stage="after_node_creation")
_save_intermediate_result(child_node, stage="after_compile")
_save_intermediate_result(child_node, stage="after_run")
_save_intermediate_result(child_node, stage="after_plots")
```

## Complete Node Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NODE LIFECYCLE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Selection (Main Process)                                                 │
│     │                                                                        │
│     ├─ _select_parallel_nodes() → [parent_node_0, parent_node_1, ...]       │
│     │                                                                        │
│  2. Serialization (Main Process)                                             │
│     │                                                                        │
│     ├─ node_data = parent_node.to_dict()                                    │
│     │                                                                        │
│  3. Task Submission (Main Process → Worker)                                  │
│     │                                                                        │
│     ├─ worker_manager.submit(_process_node_wrapper, node_data, ...)         │
│     │                                                                        │
│  4. Worker Initialization (Worker Process)                                   │
│     │                                                                        │
│     ├─ Set CUDA_VISIBLE_DEVICES                                             │
│     ├─ Initialize MemoryManager                                             │
│     ├─ Create workspace directory                                           │
│     ├─ Copy files from parent workspace                                     │
│     │                                                                        │
│  5. Branch Fork (Worker Process)                                             │
│     │                                                                        │
│     ├─ child_branch_id = uuid.uuid4().hex                                   │
│     ├─ memory_manager.mem_node_fork(parent_branch_id, child_branch_id)      │
│     │                                                                        │
│  6. Phase 0 Planning (Worker Process)                                        │
│     │                                                                        │
│     ├─ _execute_phase0_planning() if split mode                             │
│     │                                                                        │
│  7. Node Creation (Worker Process)                                           │
│     │                                                                        │
│     ├─ _draft() / _debug() / _improve() / _generate_*_node()                │
│     ├─ LLM generates code                                                   │
│     ├─ child_node.branch_id = child_branch_id                               │
│     │                                                                        │
│  8. Container Setup (Worker Process)                                         │
│     │                                                                        │
│     ├─ ExecutionEnvironment(image=singularity_image, ...)                   │
│     ├─ Collect environment info (compilers, libs, GPU)                      │
│     │                                                                        │
│  9. Phase Execution (Worker Process in Singularity)                          │
│     │                                                                        │
│     ├─ Phase 1: Download/Install dependencies                               │
│     ├─ Phase 2: Compile code                                                │
│     ├─ Phase 3: Run experiment                                              │
│     ├─ Phase 4: Parse results                                               │
│     ├─ (Optional) Plot generation                                           │
│     │                                                                        │
│  10. Result Return (Worker Process → Main Process)                           │
│      │                                                                       │
│      ├─ return child_node.to_dict()                                         │
│                                                                              │
│  11. Result Processing (Main Process)                                        │
│      │                                                                       │
│      ├─ child_node = Node.from_dict(result_data, journal)                   │
│      ├─ journal.append(child_node)                                          │
│      ├─ Update hyperparam/ablation state                                    │
│                                                                              │
│  12. Next Iteration                                                          │
│      │                                                                       │
│      └─ Go to step 1 (until termination conditions met)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| [parallel_agent.py](../../ai_scientist/treesearch/parallel_agent.py) | Main worker agent, node selection, task dispatch |
| [worker/manager.py](../../ai_scientist/treesearch/worker/manager.py) | WorkerManager for multiprocessing |
| [journal.py](../../ai_scientist/treesearch/journal.py) | Node and Journal classes |
| [utils/phase_execution.py](../../ai_scientist/treesearch/utils/phase_execution.py) | ExecutionEnvironment (Singularity wrapper) |
| [memory/memgpt_store.py](../../ai_scientist/memory/memgpt_store.py) | MemoryManager for branch fork |

## Configuration

```yaml
# bfts_config.yaml
agent:
  search:
    num_drafts: 3        # Number of root nodes to create
    debug_prob: 0.3      # Probability of selecting debug node
    max_debug_depth: 3   # Maximum debug iterations

exec:
  phase_mode: split                    # "split" for Singularity
  singularity_image: template/base.sif # Path to SIF image
  workspace_mount: /workspace          # Mount point in container
  use_gpu: true
  timeout: 10800                       # Execution timeout (seconds)

# Worker count
agent:
  num_workers: 4  # Number of parallel workers
```
