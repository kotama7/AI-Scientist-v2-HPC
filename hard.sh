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
-ccbin,
/usr/bin/g++,
-O2,
-std=c++20,
--cudart,
shared,
-Xcompiler, 
-fopenmp,
-I$CONDA_PREFIX/include,
-I$CONDA_PREFIX/include/python3.11,
-I$CONDA_PREFIX/include/eigen3,
-L/usr/lib64,
-L/lib64,
-L$CONDA_PREFIX/lib,
-lcnpy,
-lfmt,
-lopenblas,
-lcublas,
-lpython3.11,
-lpthread,
-lm,
-ldl,
-lstdc++fs,
-Xlinker,
'-rpath,$CONDA_PREFIX/lib',
-Xlinker,
'-rpath,$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib'"
echo ""