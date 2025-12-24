# Title: Challenges in Achieving Stable Himeno Benchmark Performance on Modern HPC Systems

## Keywords
Himeno benchmark, 3D Poisson solver, Jacobi iteration, 3D stencil, HPC performance, performance variability, memory bandwidth, cache, NUMA, OpenMP, thread affinity, CPU frequency scaling, reproducibility

## TL;DR
The Himeno benchmark (Himeno Bench) is a widely used 3D Poisson/Jacobi-style stencil benchmark whose performance is often dominated by the memory hierarchy rather than peak compute.  
On modern HPC nodes, measured results can fluctuate significantly due to runtime and system factors, making stable evaluation difficult.  
This work focuses on improving Himeno Bench performance and stability as a benchmark methodology for reliable, comparable HPC evaluation.

## Abstract
The Himeno benchmark (Himeno Bench) is a widely used benchmark based on an iterative method for solving a 3D Poisson equation using a Jacobi-like stencil update over a structured grid. Although its arithmetic operations are straightforward, the benchmark is highly sensitive to the memory hierarchy and runtime execution conditions, and therefore frequently exhibits substantial performance variability on modern HPC systems. In practice, measured performance and scaling may deviate from simple expectations such as stable per-node throughput or near-linear speedup with increasing core count, especially on multi-socket CPU nodes and in production environments.

This work investigates why stable and reproducible Himeno Bench performance is difficult to achieve and highlights recurring categories of factors that influence results, including memory hierarchy behavior, cache effects, NUMA placement, OpenMP runtime interactions, thread/process affinity, CPU frequency dynamics, microarchitectural differences across CPU generations, and system noise at scale. We argue that for Himeno Bench, performance outcomes depend not only on the computational kernel but also on the discipline and consistency of execution conditions, which strongly affect comparability across systems and across runs.

The final objective of this work is to support **improved Himeno Bench performance and stability** for modern HPC evaluation. Rather than treating variability as an unavoidable nuisance, we position it as a practical target for benchmark refinement and reproducible methodology. By clarifying the sources of instability and framing them in terms of benchmark-driven evaluation needs, this work aims to contribute toward Himeno Bench–based measurements that are more predictable, comparable, and useful for assessing contemporary HPC platforms.
