# Troubleshooting

## Collect debug info first

- `experiments/<run>/logs/` for phase logs and errors.
- `experiments/<run>/logs/unified_tree_viz.html` for node history.
- `experiments/<run>/logs/phase_logs/node_<id>/prompt_logs/` for prompt inputs.
- `experiments/<run>/memory/` if memory is enabled.

## "Singularity image is required"

Split mode needs a base image. Pass `--singularity_image` or update
`exec.singularity_image` in `bfts_config.yaml`.

## `singularity` command not found

Ensure Apptainer is aliased or symlinked to `singularity`, or update your PATH
so the launcher can invoke it.

## Phase 1 cannot write inside the container

Try one of the following:

- `--writable_mode overlay` with `--container_overlay /path/to/overlay.img`.
- Disable tmpfs via `--disable_writable_tmpfs`.

## `singularity build` fails due to permissions

Try `--use_fakeroot false`.

## `torch` import fails on the host

Install Torch in the control-plane environment. The launcher imports Torch to
map workers to GPUs even in split mode.

## CUDA is not detected

- Confirm `CUDA_VISIBLE_DEVICES` is set correctly in your job environment.
- Verify `python -c "import torch; print(torch.cuda.is_available())"` on the
  host.

## Resource validation errors

Double-check:

- `mount_path`/`dest` are under `/workspace`.
- Local `host_path` exists.
- `items.path` is relative and does not include `..`.

## Hugging Face downloads fail

Install `huggingface_hub` inside the worker image and provide
`HUGGINGFACE_API_KEY` if needed.

## Phase 1 repeatedly fails to install deps

- Increase `--phase1_max_steps` so the iterative installer can recover.
- Switch to `--writable_mode overlay` with a larger overlay image.
- Bake the dependency into the base SIF if it is always needed.

## Tree visualization is blank

- Ensure `unified_tree_viz.html` and the per-stage `tree_data.json` files exist.
- Open the HTML file from the `logs/` directory so relative paths resolve.

## Database is locked errors

SQLite "database is locked" errors occur when multiple parallel workers try to
write to the memory database simultaneously. The system includes a centralized
database writer to prevent this, but issues can still occur in edge cases.

### Symptoms

```
WARNING  Auto-consolidation of inherited memory failed: database is locked
WARNING  Failed to write node_created event: database is locked
WARNING  Failed to apply Phase 1 memory updates: database is locked
```

### Solutions

1. **Verify DatabaseWriterProcess is running**

   Check logs for:
   ```
   DatabaseWriterProcess started (pid=XXXXX)
   ```

   If not present, check for initialization errors in the main process.

2. **Verify workers are using the writer queue**

   Each worker should log:
   ```
   [Worker N] Using centralized database writer process
   ```

   If not present, the `writer_queue` may not be passed correctly.

3. **Check for process crashes**

   If the writer process crashes, workers will fall back to local connections
   which can cause locking. Look for:
   ```
   DatabaseWriterProcess exited
   ```

4. **Increase timeouts (temporary workaround)**

   In `memgpt_store.py`, the following settings control retry behavior:
   ```python
   # _execute_with_retry defaults
   max_retries: int = 10
   base_delay: float = 0.5

   # SQLite connection
   PRAGMA busy_timeout=60000  # 60 seconds
   ```

5. **Reduce parallelism**

   If issues persist, try reducing `num_workers` or running with
   `num_workers=1` to isolate the problem.

### Architecture

The centralized writer serializes all writes through a single process:

```
Worker 1 ─┐
Worker 2 ─┼─> Write Queue ─> DatabaseWriterProcess ─> SQLite
Worker N ─┘
```

This avoids SQLite's concurrent write limitations while maintaining throughput
through batched commits. See [memory.md](../memory/memory.md#centralized-database-writer-parallel-execution)
for details.

### Related files

- `ai_scientist/memory/db_writer.py`: DatabaseWriterProcess implementation
- `ai_scientist/memory/memgpt_store.py`: MemoryManager with writer integration
- `ai_scientist/treesearch/parallel_agent.py`: Writer initialization and shutdown
