#!/usr/bin/env python3
"""Test MPI initialization issue"""

import os
import sys

# Set MPI4PY to not initialize MPI automatically
os.environ['MPI4PY_RC_INITIALIZE'] = 'false'

from mpi4py import MPI

# Initialize MPI manually
if not MPI.Is_initialized():
    MPI.Init_thread(MPI.THREAD_SINGLE)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank} of {size} - Success!")

# Finalize MPI
MPI.Finalize()