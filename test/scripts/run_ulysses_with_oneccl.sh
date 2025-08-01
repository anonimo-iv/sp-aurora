#!/bin/bash
# Script to run Ulysses attention test with proper oneCCL environment setup

echo "=================================================="
echo "Setting up oneCCL environment for Intel GPU"
echo "=================================================="

# First, check if we're on an Aurora node with Intel GPU
if command -v sycl-ls &> /dev/null; then
    echo "Checking available SYCL devices..."
    sycl-ls
fi

# Set up Intel oneAPI environment if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "Loading Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
elif [ -f ~/intel/oneapi/setvars.sh ]; then
    echo "Loading Intel oneAPI environment from home..."
    source ~/intel/oneapi/setvars.sh
fi

# Set library path for Intel Extension for PyTorch
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Source oneCCL environment if available
if [ -f /opt/intel/oneapi/ccl/latest/env/vars.sh ]; then
    echo "Loading oneCCL environment..."
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
elif [ -f ~/intel/oneapi/ccl/latest/env/vars.sh ]; then
    echo "Loading oneCCL environment from home..."
    source ~/intel/oneapi/ccl/latest/env/vars.sh
fi

# Essential oneCCL environment variables
export CCL_ROOT=${CCL_ROOT:-$(python -c "import oneccl_bindings_for_pytorch; print(oneccl_bindings_for_pytorch.cwd)" 2>/dev/null || echo "")}

# MPI-specific settings
export I_MPI_HYDRA_BOOTSTRAP=ssh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=ssh

# oneCCL configuration for Intel GPU
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_TRANSPORT_MPI=1
export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_ENABLE=1
export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_LOG_LEVEL=info

# Use simple algorithms to avoid hangs
export CCL_ALLREDUCE=ring
export CCL_ALLTOALL=naive
export CCL_BCAST=double_tree
export CCL_ALLGATHER=ring

# Synchronization settings
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1
export CCL_WORKER_WAIT=1

# Intel GPU optimizations
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:gpu
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS=0

# Fabric interface settings
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072

# MPI GPU support
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=1

# Print diagnostic information
echo ""
echo "Environment diagnostics:"
echo "========================"
echo "CCL_ROOT: ${CCL_ROOT}"
echo "LD_LIBRARY_PATH contains:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -E "(ccl|mpi|oneapi)" | head -5

# Check if oneccl_bindings_for_pytorch is available
echo ""
echo "Checking oneCCL bindings..."
python -c "
try:
    import oneccl_bindings_for_pytorch
    print('✓ oneccl_bindings_for_pytorch found')
    print(f'  Version: {oneccl_bindings_for_pytorch.__version__ if hasattr(oneccl_bindings_for_pytorch, \"__version__\") else \"unknown\"}')
except ImportError as e:
    print('✗ oneccl_bindings_for_pytorch not found:', e)
"

# Check MPI
echo ""
echo "Checking MPI..."
if command -v mpirun &> /dev/null; then
    echo "✓ mpirun found at: $(which mpirun)"
    mpirun --version | head -1
else
    echo "✗ mpirun not found"
fi

# Run the test
echo ""
echo "=================================================="
echo "Running Ulysses attention test with MPI"
echo "=================================================="
echo "Command: mpirun -n 2 python test_intel_ulysses_attn.py"
echo ""

# Run with error handling
set -e
mpirun -n 2 python test_intel_ulysses_attn.py 2>&1 | tee ulysses_test_output.log

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
else
    echo ""
    echo "❌ Test failed! Check ulysses_test_output.log for details"
    echo ""
    echo "Common issues:"
    echo "1. oneCCL not properly installed - install with: pip install oneccl_bind_pt"
    echo "2. Intel GPU not available - check with: sycl-ls"
    echo "3. MPI not configured - ensure Intel MPI or MPICH is installed"
    echo "4. Missing Intel Extension for PyTorch - install with: pip install intel-extension-for-pytorch"
    exit 1
fi