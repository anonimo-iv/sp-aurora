#!/bin/bash
# Examples of running Ring Flash Attention with mpiexec
# This script demonstrates mpiexec compatibility alongside existing torchrun usage

echo "🚀 Ring Flash Attention MPI Examples"
echo "====================================="

# Function to run a command with proper error handling
run_test() {
    local description="$1"
    local command="$2"
    
    echo ""
    echo "🧪 $description"
    echo "Command: $command"
    echo "---"
    
    if eval "$command"; then
        echo "✅ $description: SUCCESS"
    else
        echo "❌ $description: FAILED"
        return 1
    fi
}

# Check if we have MPI available
if ! command -v mpiexec &> /dev/null; then
    echo "❌ mpiexec not found. Please install MPI (e.g., OpenMPI, Intel MPI)"
    echo "   Ubuntu/Debian: sudo apt-get install openmpi-bin"
    echo "   RHEL/CentOS: sudo yum install openmpi"
    echo "   Intel MPI: source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

echo "✅ MPI detected: $(mpiexec --version | head -1)"

# Test 1: Basic MPI compatibility test
run_test "Basic MPI Compatibility (2 processes)" \
    "mpiexec -n 2 python test_mpi_ring_flash_attn.py"

# Test 2: MPI with explicit master address (useful for multi-node)
run_test "MPI with explicit master address" \
    "MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 mpiexec -n 2 python test_mpi_ring_flash_attn.py"

# Test 3: Intel MPI with CCL backend (if available)
if command -v mpirun &> /dev/null && [[ "$MPIRUN" == *"intel"* ]] || [[ -n "$I_MPI_ROOT" ]]; then
    run_test "Intel MPI with CCL backend" \
        "mpiexec -n 2 -genv CCL_BACKEND=native -genv CCL_ATL_TRANSPORT=ofi -genv FI_PROVIDER=cxi python test_mpi_ring_flash_attn.py"
fi

# Test 4: Compare with torchrun (if available)
if command -v torchrun &> /dev/null; then
    echo ""
    echo "🔄 Comparing with torchrun..."
    
    run_test "TorchRun (for comparison)" \
        "torchrun --nproc_per_node=2 test_mpi_ring_flash_attn.py"
fi

# Test 5: Single process (should work with both)
run_test "Single process with mpiexec" \
    "mpiexec -n 1 python test_mpi_ring_flash_attn.py"

echo ""
echo "=================================================="
echo "📖 USAGE EXAMPLES"
echo "=================================================="
echo ""
echo "1. Basic usage with OpenMPI:"
echo "   mpiexec -n 4 python your_script.py"
echo ""
echo "2. Multi-node with OpenMPI:"
echo "   mpiexec -n 8 -hostfile hosts python your_script.py"
echo ""
echo "3. Intel MPI with optimizations:"
echo "   mpiexec -n 4 -genv CCL_BACKEND=native \\"
echo "     -genv CCL_ATL_TRANSPORT=ofi python your_script.py"
echo ""
echo "4. Slurm with MPI:"
echo "   srun --mpi=pmix -n 4 python your_script.py"
echo ""
echo "5. Setting master address explicitly:"
echo "   MASTER_ADDR=node001 MASTER_PORT=29500 \\"
echo "     mpiexec -n 4 python your_script.py"
echo ""
echo "💡 TIP: Your script needs to use ring_flash_attn.mpi_utils.setup_mpi_distributed()"
echo "    for automatic MPI compatibility!"

echo ""
echo "🎯 Integration in your code:"
echo "----------------------------"
cat << 'EOF'
from ring_flash_attn.mpi_utils import setup_mpi_distributed, cleanup_distributed

def main():
    # Setup distributed environment (works with both torchrun and mpiexec)
    setup_info = setup_mpi_distributed()
    
    rank = setup_info['rank']
    world_size = setup_info['world_size']
    device = setup_info['device']
    
    print(f"Rank {rank}/{world_size} on device {device}")
    
    # Your ring attention code here...
    from ring_flash_attn import ring_flash_attn_func
    # ... rest of your code
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
EOF

echo ""
echo "✅ MPI compatibility examples completed!"