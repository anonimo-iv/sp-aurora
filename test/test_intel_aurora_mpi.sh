#!/bin/bash
# Test script for Intel GPU Ring Flash Attention on Aurora with MPI

echo "🚀 Testing Intel GPU Ring Flash Attention on Aurora"
echo "=============================================="

# Aurora-specific Intel GPU environment variables
export CCL_PROCESS_LAUNCHER=pmix  
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE_SCALEOUT="direct:0-1048576;rabenseifner:1048577-max"
export CCL_BCAST=double_tree

export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600 

export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_KVS_USE_MPI_RANKS=1

export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_ZE_ENABLE=1
export CCL_LOG_LEVEL=info
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:*
export SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK=2
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1

# Aurora-specific settings
export CCL_ALLREDUCE=topo
export FI_CXI_DISABLE_HOST_REGISTER=1

# MPI settings for Aurora
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=262144
export MPIR_CVAR_CH4_IPC_GPU_ENGINE_TYPE=compute
export MPIR_CVAR_CH4_IPC_ZE_SHAREABLE_HANDLE=drmfd
export MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD=262144

echo "Environment variables set for Aurora Intel GPU"
echo "Starting with mpiexec..."

# For single node with 6 GPUs on Aurora
# Use mpiexec with 6 processes, all on one node
mpiexec -n 6 -ppn 6 python ./test_intel_gpu_mpi.py

echo "Test completed."