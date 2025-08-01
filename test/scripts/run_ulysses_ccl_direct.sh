#!/bin/bash
# Script to run Ulysses attention test with CCL backend directly
# Attempting to avoid the all-to-all hang issue

echo "=================================================="
echo "Setting up oneCCL environment for direct CCL usage"
echo "=================================================="

# Set up Intel oneAPI environment if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "Loading Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
fi

# Set library path for Intel Extension for PyTorch
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Source oneCCL environment
if [ -f /opt/intel/oneapi/ccl/latest/env/vars.sh ]; then
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
fi

# Critical: Use specific CCL settings to avoid hangs
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_ENABLE=1

# Use specific alltoall algorithm that might avoid hang
export CCL_ALLTOALL=direct  # Try direct algorithm instead of naive
export CCL_ALLTOALLV=direct
export CCL_ALLTOALL_SCALEOUT=direct

# Disable potentially problematic features
export CCL_ATL_SYNC_COLL=0
export CCL_OP_SYNC=0
export CCL_WORKER_WAIT=0
export CCL_ZE_BARRIER=0
export CCL_ZE_BIDIR_ALGO=0

# Set higher timeout for debugging
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_WORKER_TIMEOUT=600

# Use specific process launcher settings
unset CCL_PROCESS_LAUNCHER  # Let it auto-detect

# Enable verbose logging to debug
export CCL_LOG_LEVEL=debug
export CCL_LOG_FILE=ccl_ulysses_direct.log

# Intel GPU settings
export SYCL_DEVICE_FILTER=level_zero:gpu
export ZE_AFFINITY_MASK=0,1  # Use first 2 GPUs
export FI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072

echo ""
echo "CCL Configuration:"
echo "=================="
echo "CCL_ALLTOALL: ${CCL_ALLTOALL}"
echo "CCL_ATL_TRANSPORT: ${CCL_ATL_TRANSPORT}"
echo "CCL_CONFIGURATION: ${CCL_CONFIGURATION}"
echo "CCL_ZE_ENABLE: ${CCL_ZE_ENABLE}"
echo ""

# Test with modified ulysses test that uses CCL
echo "Running test with CCL backend..."
echo "Command: mpirun -n 2 python test_intel_ulysses_attn.py"
echo ""

# Run with timeout to prevent indefinite hang
timeout 300 mpirun -n 2 python test_intel_ulysses_attn.py 2>&1 | tee ulysses_ccl_direct.log

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 124 ]; then
    echo ""
    echo "❌ Test timed out after 5 minutes - CCL all-to-all likely hung"
    echo "Check ccl_ulysses_direct.log for CCL debug information"
elif [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully with CCL backend!"
else
    echo ""
    echo "❌ Test failed with exit code: $EXIT_CODE"
fi

echo ""
echo "To check CCL debug log:"
echo "cat ccl_ulysses_direct.log"