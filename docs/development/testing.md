# Testing

Run individual test modules with `python -m unittest`. The tests are organized
by feature area so you can target the parts you changed.

## Testing workflow

```bash
python -m unittest tests/test_smoke_split.py
python -m unittest tests/test_resource.py
python -m unittest tests/test_memgpt_branch_inheritance.py
python -m unittest tests/test_final_memory_generation.py
python -m unittest tests/test_worker_parallelism.py
python -m unittest tests/test_llm_compression.py
python -m unittest tests/test_iterative_compression.py
python -m unittest tests/test_writeup_memory_loading.py
```

## What each test covers

- `test_smoke_split`: split-mode smoke test.
- `test_resource`: resource file validation and staging.
- `test_memgpt_branch_inheritance`: memory behavior across branches.
- `test_final_memory_generation`: final memory artifacts.
- `test_worker_parallelism`: worker count and GPU assignment.
- `test_llm_compression`: LLM-based memory compression.
- `test_iterative_compression`: iterative LLM-based compression logic.
- `test_writeup_memory_loading`: memory loading for paper writeup.

## Notes

- Some tests assume split mode and may need Singularity available.
- If you changed prompts or writeup flows, consider running a small end-to-end
  run instead of unit tests.

## Minimal validation set

If you only want to sanity-check a change:

- `python -m unittest tests/test_smoke_split.py`
- `python -m unittest tests/test_resource.py`
