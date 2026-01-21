# Supplementary Information: Himeno Benchmark Performance Analysis on Modern HPC Systems

## Keywords
NUMA topology, cache hierarchy, thread pinning, memory bandwidth, TLB, OpenMP environment variables, CPU frequency scaling, turbo boost, performance metrics, statistical sampling, loop tiling, SIMD vectorization, system noise, OS jitter

## TL;DR
This supplement provides technical background for understanding Himeno benchmark instability. Key factors include NUMA effects, cache residency differences across problem sizes, and runtime variability from CPU frequency scaling and thread migration. For the three problem sizes (**64×64×128** ~50MB, **128×128×256** ~400MB, **256×256×512** ~3.2GB), memory footprint determines whether performance is cache-bound or memory-bound. Recommended mitigations include explicit thread affinity (`OMP_PROC_BIND=close`), fixed CPU frequency (`cpupower frequency-set -g performance`), and NUMA-local allocation (`numactl --localalloc`). Optimization strategies target three metrics separately: cache blocking improves **median**, thread pinning reduces **CV**, and OS noise isolation suppresses **p99/median** tail degradation.

## Abstract
This supplementary document provides detailed technical context for the main analysis of Himeno benchmark performance stability.

The Himeno benchmark, developed at RIKEN, solves a 3D Poisson equation using a 19-point Jacobi stencil. Memory footprints vary significantly across problem sizes: **64×64×128** (~50MB) may fit in L3 cache, **128×128×256** (~400MB) exceeds typical cache sizes, and **256×256×512** (~3.2GB) stresses main memory bandwidth. These differences cause each size to exhibit distinct sensitivity profiles to system conditions.

Performance variability arises from multiple sources. NUMA effects include first-touch page placement and remote memory access penalties (1.5–3× latency). Runtime factors include CPU turbo boost (5–15% CV impact), thread migration (3–10%), OS interrupts (1–5%), and thermal throttling (5–20%). These are not independent—interactions between factors can amplify instability.

Experimental methodology requires statistical rigor: minimum 30 runs per configuration, warm-up run exclusion, and reporting with/without outliers. Three metrics capture distribution behavior: **median MFLOPS** (central tendency), **CV** (relative dispersion), and **p99/median** (tail degradation ratio).

Optimization strategies map to specific metrics. For **median improvement**: cache blocking, loop reordering, SIMD vectorization, and prefetching. For **CV reduction**: explicit thread pinning via `OMP_PROC_BIND` and `OMP_PLACES`, NUMA-aware allocation, and fixed CPU frequency. For **p99/median suppression**: minimizing synchronization, using deterministic allocators, CPU isolation via cgroups, and IRQ affinity tuning.

Recommended system settings include: turbo boost disabled, hyper-threading disabled, automatic NUMA balancing disabled, and transparent huge pages enabled. These configurations reduce non-determinism while maintaining representative performance levels. Platform topology (single vs. multi-socket, memory channels, interconnect bandwidth) further modulates which factors dominate for each problem size.