# Automatic Hardware Information Injection

## Overview

The memory system automatically extracts hardware and environment information from archival memory (PHASE0_INTERNAL entries) and injects it into Core Memory during final memory generation. This ensures hardware information is always available for paper generation, preventing "Not found in memory" issues.

## Implementation

### Location

- **File**: `ai_scientist/memory/memgpt_store.py`
- **Methods**:
  - `_extract_hardware_info_from_archival()`: Extracts hardware info from PHASE0_INTERNAL entries
  - `_inject_hardware_info_to_core()`: Injects extracted info into Core Memory
  - `generate_final_memory_for_paper()`: Automatically calls injection before generating sections

### Execution Flow

```
generate_final_memory_for_paper()
    │
    ├─ _inject_hardware_info_to_core(root_branch_id)
    │   │
    │   ├─ Check if hardware info already in Core Memory
    │   │   └─ If exists: Skip extraction (log and return)
    │   │
    │   └─ If missing:
    │       ├─ _extract_hardware_info_from_archival(branch_id)
    │       │   ├─ Query archival for PHASE0_INTERNAL entries
    │       │   └─ Extract via regex: CPU, sockets, cores, NUMA, OS, compiler, tools
    │       │
    │       └─ set_core() for each extracted field:
    │           ├─ hardware_cpu
    │           ├─ hardware_os
    │           ├─ hardware_compiler
    │           └─ hardware_tools
    │
    └─ _build_paper_sections() (uses injected Core Memory)
```

## Extracted Information

| Core Memory Key | Example Value | Extraction Pattern |
|----------------|---------------|-------------------|
| `hardware_cpu` | `AMD EPYC 9554, 2 socket(s), 128 cores, 2 NUMA node(s)` | CPU model, sockets, cores, NUMA nodes |
| `hardware_os` | `Ubuntu 22.04` | Ubuntu version |
| `hardware_compiler` | `gcc 11.4.0` | gcc/gfortran version |
| `hardware_tools` | `numactl, perf, taskset, hwloc-ls, lscpu, numastat` | Tool names |

## Behavior

### When Hardware Info Already Exists

If all four hardware keys (`hardware_cpu`, `hardware_os`, `hardware_compiler`, `hardware_tools`) already exist in Core Memory:
- Skips extraction (logs: "Hardware info already present in Core Memory, skipping extraction")
- Uses existing values
- No database queries performed

### When Hardware Info Missing

If any hardware key is missing:
1. Queries archival memory for PHASE0_INTERNAL entries (up to 20 entries)
2. Extracts information using regex patterns
3. Injects into Core Memory for missing keys only
4. Logs each injection (e.g., "Injected hardware_cpu: AMD EPYC 9554, 2 socket(s), 128 cores, 2 NUMA node(s)")

## Consistency Across Execution Paths

Both execution paths now use the **same implementation**:

### Path 1: `launch_scientist_bfts.py`

```python
# perform_experiments_bfts_with_agentmanager.py:324
generate_paper_memory_from_manager(
    memory_manager=memory_manager,
    manager=manager,
    ...
)
# → Calls generate_final_memory_for_paper()
# → Automatically injects hardware info
```

### Path 2: `regenerate_memory_env.sh`

```python
# regenerate_memory_env.sh:264
generate_paper_memory_from_manager(
    memory_manager=store,
    manager=manager,
    ...
)
# → Calls generate_final_memory_for_paper()
# → Automatically injects hardware info (same code path)
```

**Result**: Both paths produce identical output with no "Not found in memory" for hardware information.

## Usage in Paper Generation

After injection, hardware information is available via:

1. **Core Memory** (always visible):
   - `_append_experimental_setup_facts()` reads from Core Memory first
   - Section generation queries Core Memory directly

2. **LLM Context**:
   - Core Memory is included in every LLM prompt
   - LLM can reference hardware info when generating sections

## Example Output

### Before Injection
```markdown
### Hardware and Environment Details
- **CPU**: Not found in memory.
- **Operating System**: Not found in memory.
- **Compiler**: Not found in memory.
- **Tools**: Not found in memory.
```

### After Injection
```markdown
### Hardware and Environment Details
- **CPU**: AMD EPYC 9554, 2 socket(s), 128 cores, 2 NUMA node(s)
- **Operating System**: Ubuntu 22.04
- **Compiler**: gcc 11.4.0.
- **Tools**: numactl, perf, taskset, hwloc-ls, lscpu, numastat
```

## Limitations

### Information Not Extracted

The following information cannot be extracted automatically and may still show "Not found in memory":
- OpenMP runtime version (not consistently recorded)
- PAPI hardware counters (if not used)
- Vendor-specific profiling tools (AMD uProf, Intel PCM)
- Exact memory capacity (not in standard format)
- Kernel version (not consistently recorded)
- Background load control mechanisms

These require explicit recording during Phase 0 or are truly missing from the experiment.

## Troubleshooting

### Hardware Info Still Missing

If hardware information is still missing after regeneration:

1. **Check PHASE0_INTERNAL entries exist**:
   ```bash
   sqlite3 <run_dir>/memory/memory.sqlite \
     "SELECT COUNT(*) FROM archival WHERE tags LIKE '%PHASE0_INTERNAL%';"
   ```

2. **Check extraction logs**:
   Look for "Extracting hardware info from archival memory..." in output

3. **Manually verify patterns**:
   ```bash
   sqlite3 <run_dir>/memory/memory.sqlite \
     "SELECT text FROM archival WHERE tags LIKE '%PHASE0_INTERNAL%' LIMIT 5;"
   ```
   Check if text contains: "AMD EPYC", "Ubuntu", "gcc", "numactl"

4. **Check regex patterns**:
   Patterns are in `_extract_hardware_info_from_archival()` (line ~4015)
   Adjust if your format differs

## Related Documentation

- [memory-for-paper.md](memory-for-paper.md) - Final memory generation overview
- [memory.md](memory.md) - Memory system architecture
- [memgpt-implementation.md](memgpt-implementation.md) - Core Memory implementation
