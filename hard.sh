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
-fopenmp
-I$CONDA_PREFIX/include
-I$CONDA_PREFIX/libtorch/include/torch/csrc/api/include
-I$CONDA_PREFIX/targets/x86_64-linux/include
-I$CONDA_PREFIX/include/python3.11
-I$CONDA_PREFIX/include/eigen3
-I$CONDA_PREFIX/include/xtensor
-I$CONDA_PREFIX/include/xtl
-I$CONDA_PREFIX/include/matplotlibcpp
-I$LIBTORCH_GPU/include
-I$LIBTORCH_GPU/include/torch/csrc/api/include
-L$CONDA_PREFIX/lib
-L$LIBTORCH_GPU/lib
-L$CONDA_PREFIX/libtorch/lib
-L/usr/lib64
-L/lib64
-lcnpy 
-lcudart 
-lmkl_rt 
-lcublas 
-lopenblas 
-lpython3.11 
-lpthread 
-ltorch 
-ltorch_cpu 
-ltorch_cuda 
-lc10 
-lnuma 
-ldl 
-lspdlog 
-lfmt 
-lm 
-lhwloc
-lpthread
-lstdc++fs" 
echo ""