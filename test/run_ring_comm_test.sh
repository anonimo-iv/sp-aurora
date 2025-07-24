#!/bin/bash
# Run ring communication test with proper CCL environment

echo "🚀 Testing Ring Communication with CCL"
echo "====================================="

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

# Add Intel Extension library path - MUST be first in path for CCL to work
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Verify the library can be found
if [ -f "/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib/libintel-ext-pt-gpu.so" ]; then
    echo "✅ Found libintel-ext-pt-gpu.so"
else
    echo "❌ Cannot find libintel-ext-pt-gpu.so"
    exit 1
fi

echo "Environment variables set for Intel GPU"
echo "Running ring communication test..."

# Run the test with explicit master address and port
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 test_ring_comm_mpi.py

echo "Test completed."