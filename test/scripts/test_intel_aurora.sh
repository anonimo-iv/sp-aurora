#!/bin/bash
# Test script for Intel GPU Ring Flash Attention on Aurora with MPI

echo "🚀 Testing Intel GPU Ring Flash Attention on Aurora"
echo "=============================================="

# Aurora-specific Intel GPU environment variables
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_LOG_LEVEL=info
export CCL_ZE_ENABLE=1
export CCL_KVS_USE_MPI_RANKS=1
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1

# Additional environment variables
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:*
export MPIR_CVAR_ENABLE_GPU=1

# MPI settings for Aurora
# export MPIR_CVAR_ENABLE_GPU=1
# export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=262144
# export MPIR_CVAR_CH4_IPC_GPU_ENGINE_TYPE=compute
# export MPIR_CVAR_CH4_IPC_ZE_SHAREABLE_HANDLE=drmfd
# export MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD=262144

echo "Environment variables set for Aurora Intel GPU"
echo "Starting with mpiexec..."

# For single node with 6 GPUs on Aurora
# Use mpiexec with 6 processes, all on one node
echo "Testing fixed implementation..."
mpirun -n 4 python test_intel_sp_aurora.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "❌ Test failed!"
    exit 1
fi