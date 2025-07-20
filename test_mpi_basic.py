#!/usr/bin/env python3
"""
Basic test to verify MPI compatibility is working
"""

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mpi_setup():
    """Test basic MPI setup"""
    try:
        from ring_flash_attn.mpi_utils import setup_distributed_environment
        
        env_info = setup_distributed_environment()
        rank = env_info['rank']
        world_size = env_info['world_size']
        launcher = env_info['launcher']
        
        print(f"[Rank {rank}] MPI setup successful!")
        print(f"[Rank {rank}] Launcher: {launcher}")
        print(f"[Rank {rank}] World size: {world_size}")
        
        return True
    except Exception as e:
        print(f"MPI setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mpi_setup()
    sys.exit(0 if success else 1)