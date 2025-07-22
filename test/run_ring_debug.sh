#!/bin/bash
# Run the fixed ring debug test with proper CCL environment

echo "🚀 Testing Fixed Ring Debug with Enhanced Logging"
echo "==============================================="

# Set Intel GPU environment variables
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
echo "Running fixed ring debug test..."

# Run with timeout
timeout 120s torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 test_ring_debug_fixed.py

echo "Test completed."