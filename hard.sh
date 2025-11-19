echo "===== CPU ====="
lscpu
echo ""
echo "===== CACHE ====="
lscpu --cache
echo ""
echo "===== MEMORY ====="
free -h
numactl --hardware
echo ""
echo "===== OS ====="
uname -a
cat /etc/os-release
echo ""
echo "===== COMPILER ====="
gcc --version
gcc -march=native -Q --help=target | grep enabled
echo ""
echo "===== OPENBLAS ====="
python3 -c "import numpy as np; np.__config__.show()"
echo ""
echo "===== THREAD SETTINGS ====="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "GOTO_NUM_THREADS=$GOTO_NUM_THREADS"
echo ""
echo "===== GPU ====="
nvidia-smi
echo ""
echo "===== CONDA ENVIRONMENT ====="
conda info
conda list
echo ""
echo "===== COMPILER FLAG =====
-O2
-std=c++20
-I/home/users/takanori.kotama/miniconda3/envs/ai_scientist/include
-I/home/users/takanori.kotama/miniconda3/envs/ai_scientist/include/python3.11
-I/home/users/takanori.kotama/miniconda3/envs/ai_scientist/include/eigen3
-L/home/users/takanori.kotama/miniconda3/envs/ai_scientist/lib
-lpython3.11
-lcnpy
-lcudart
-lblas
-lfmt
-fopenmp
-larmadillo
-lmkl_intel_lp64
-lmkl_core
-lmkl_sequential
-lpthread
-lm
-ldl
-Wl,-rpath,~/miniconda3/envs/ai_scientist/lib/python3.11/site-packages/torch/lib"
echo ""