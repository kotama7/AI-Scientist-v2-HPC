# Title: Challenges in Achieving Stable OpenBLAS Performance on Modern HPC Systems

## Keywords
OpenBLAS, BLAS kernels, HPC performance, performance variability, CPU optimization

## TL;DR
OpenBLAS is widely used in HPC environments, yet its real-world performance often deviates from expected peak values.  
This work examines the performance inconsistencies and underlying factors across modern CPU architectures.

## Abstract
OpenBLAS is one of the most commonly used open-source BLAS libraries in academic and industrial HPC environments. Its architecture-specific optimizations and portability have made it a standard choice for many simulation, numerical linear algebra, and AI-related pipelines. However, practitioners often observe that OpenBLAS underperforms relative to theoretical peak FLOPS or vendor-provided benchmarks when deployed on heterogeneous, production-grade HPC systems.

This work investigates these unexpected performance behaviors by analyzing negative results encountered across various real-world workloads within the HPC Asia community. We document issues such as unstable scaling across NUMA domains, inconsistent kernel selection, performance drops caused by compiler-toolchain interactions, and sensitivity to CPU frequency scaling and microarchitectural differences. Additionally, we examine how modern CPU design trends—such as hybrid cores, AVX2/AVX-512 trade-offs, and deeper memory hierarchies—create challenges for static-kernel BLAS libraries like OpenBLAS.

Through a cross-domain analysis of numerical workloads, we highlight recurring patterns in performance degradation and variability. These findings illustrate broader challenges faced by open-source numerical kernels, including difficulties in keeping pace with rapidly evolving CPU architectures and balancing portability with performance. By openly discussing these negative results and their root causes, this work aims to support more predictable and reproducible HPC research and to inform future directions for OpenBLAS optimization within the HPC Asia community.
