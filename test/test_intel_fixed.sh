#!/bin/bash
# Test script for fixed Intel GPU Ring Flash Attention

echo "🚀 Testing Fixed Intel GPU Ring Flash Attention"
echo "=============================================="

# Set Intel GPU environment variables
export CCL_BACKEND=native
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_ZE_ENABLE=1
export CCL_LOG_LEVEL=info
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:*
export SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK=2

# Additional debugging variables to prevent P2P issues
export CCL_P2P_ACCESS_POLICY=off
export CCL_ALLREDUCE=ring
export FI_CXI_DISABLE_HOST_REGISTER=1

echo "Environment variables set for Intel GPU"
echo "Starting torchrun with 2 processes..."

# Run the fixed test
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 test_intel_gpu_fixed.py

echo "Test completed."