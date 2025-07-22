#!/bin/bash
# Minimal test script for CCL communication

echo "🚀 Testing CCL Backend Initialization"
echo "====================================="

# Add Intel Extension for PyTorch library path
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Set Intel GPU environment variables (from test_intel_fixed.sh)
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

# Additional debugging variables
export CCL_ALLREDUCE=topo
export FI_CXI_DISABLE_HOST_REGISTER=1

echo "Environment variables set for Intel GPU"
echo "Running backend check..."

# First check available backends
# python test/check_backends.py

echo -e "\nRunning minimal CCL test..."
torchrun --nproc_per_node=2 test_ccl_minimal.py

echo "Test completed."