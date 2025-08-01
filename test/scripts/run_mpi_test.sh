#!/bin/bash
# Script to run MPI test with proper environment

# For interactive testing on a compute node, use:
# 1. Get an interactive allocation: qsub -I -l select=1:system=aurora,walltime=00:30:00
# 2. Then run: mpiexec -n 2 python test_intel_ulysses_attn.py

# For testing without a job allocation, we need to use a different approach
echo "Running single process test (no MPI)..."
python -c "
import os
os.environ['MPICH_SINGLE_HOST'] = '1'
os.environ['PMI_RANK'] = '0'
os.environ['PMI_SIZE'] = '1'
exec(open('test_intel_ulysses_attn.py').read())
"