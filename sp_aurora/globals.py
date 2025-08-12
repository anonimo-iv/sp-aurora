"""Global process group management for sequence parallelism.

This module provides yunchang-compatible process group management for 
Intel GPU distributed training using oneCCL backend.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any


class ProcessGroupSingleton:
    """Singleton class to manage Ulysses and Ring process groups.
    
    Compatible with yunchang's PROCESS_GROUP interface.
    """
    
    def __init__(self):
        self.ULYSSES_PG: Optional[dist.ProcessGroup] = None
        self.RING_PG: Optional[dist.ProcessGroup] = None
        self.initialized: bool = False
        self._ulysses_degree: int = 1
        self._ring_degree: int = 1
        self._rank: int = 0
        self._world_size: int = 1
        
    def __repr__(self):
        return (f"ProcessGroupSingleton(initialized={self.initialized}, "
                f"ulysses_degree={self._ulysses_degree}, "
                f"ring_degree={self._ring_degree})")
    
    @property
    def ulysses_degree(self) -> int:
        """Get Ulysses parallelism degree."""
        return self._ulysses_degree
    
    @property
    def ring_degree(self) -> int:
        """Get Ring parallelism degree."""
        return self._ring_degree


# Global process group instance (yunchang-compatible)
PROCESS_GROUP = ProcessGroupSingleton()


def set_seq_parallel_pg(
    sp_ulysses_degree: int,
    sp_ring_degree: int, 
    rank: int,
    world_size: int,
    backend: str = "ccl"
) -> None:
    """Initialize sequence parallel process groups.
    
    This function is compatible with yunchang's set_seq_parallel_pg API.
    Creates separate process groups for Ulysses and Ring parallelism patterns.
    
    Args:
        sp_ulysses_degree: Degree of Ulysses (sequence) parallelism
        sp_ring_degree: Degree of Ring parallelism  
        rank: Current process rank
        world_size: Total number of processes
        backend: Communication backend (default: "ccl" for Intel GPUs)
        
    Raises:
        ValueError: If degrees don't match world size or already initialized
    """
    global PROCESS_GROUP
    
    if PROCESS_GROUP.initialized:
        raise ValueError("Process groups already initialized")
        
    if sp_ulysses_degree * sp_ring_degree != world_size:
        raise ValueError(
            f"Ulysses degree ({sp_ulysses_degree}) * Ring degree ({sp_ring_degree}) "
            f"must equal world size ({world_size})"
        )
    
    # Store configuration
    PROCESS_GROUP._ulysses_degree = sp_ulysses_degree
    PROCESS_GROUP._ring_degree = sp_ring_degree
    PROCESS_GROUP._rank = rank
    PROCESS_GROUP._world_size = world_size
    
    # Intel GPU optimizations
    if backend == "ccl":
        # Set Intel-specific environment variables for optimal performance
        os.environ.setdefault('CCL_PROCESS_LAUNCHER', 'pmix')
        os.environ.setdefault('CCL_ATL_TRANSPORT', 'mpi')
        os.environ.setdefault('CCL_ZE_IPC_EXCHANGE', 'drmfd')
        os.environ.setdefault('CCL_WORKER_COUNT', str(min(4, sp_ring_degree)))
        
    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    # Create Ulysses process groups (column-wise grouping)
    if sp_ulysses_degree > 1:
        ulysses_groups = []
        for ring_idx in range(sp_ring_degree):
            ulysses_ranks = [
                ring_idx + i * sp_ring_degree 
                for i in range(sp_ulysses_degree)
            ]
            ulysses_groups.append(dist.new_group(ulysses_ranks, backend=backend))
            
        # Find current process's Ulysses group
        ring_idx = rank % sp_ring_degree
        PROCESS_GROUP.ULYSSES_PG = ulysses_groups[ring_idx]
    else:
        # Single process Ulysses group
        PROCESS_GROUP.ULYSSES_PG = dist.new_group([rank], backend=backend)
    
    # Create Ring process groups (row-wise grouping)
    if sp_ring_degree > 1:
        ring_groups = []
        for ulysses_idx in range(sp_ulysses_degree):
            ring_ranks = [
                ulysses_idx * sp_ring_degree + i 
                for i in range(sp_ring_degree)
            ]
            ring_groups.append(dist.new_group(ring_ranks, backend=backend))
            
        # Find current process's Ring group
        ulysses_idx = rank // sp_ring_degree
        PROCESS_GROUP.RING_PG = ring_groups[ulysses_idx]
    else:
        # Single process Ring group
        PROCESS_GROUP.RING_PG = dist.new_group([rank], backend=backend)
    
    PROCESS_GROUP.initialized = True
    
    # Synchronize all processes
    dist.barrier()


def get_ulysses_pg() -> Optional[dist.ProcessGroup]:
    """Get the Ulysses process group."""
    return PROCESS_GROUP.ULYSSES_PG


def get_ring_pg() -> Optional[dist.ProcessGroup]:
    """Get the Ring process group."""
    return PROCESS_GROUP.RING_PG


def is_initialized() -> bool:
    """Check if process groups are initialized."""
    return PROCESS_GROUP.initialized


def destroy_seq_parallel_pg() -> None:
    """Destroy sequence parallel process groups and reset state."""
    global PROCESS_GROUP
    
    if PROCESS_GROUP.initialized:
        # Note: PyTorch doesn't provide explicit process group destruction
        # Groups will be cleaned up when the process exits
        PROCESS_GROUP.ULYSSES_PG = None
        PROCESS_GROUP.RING_PG = None
        PROCESS_GROUP.initialized = False
        PROCESS_GROUP._ulysses_degree = 1
        PROCESS_GROUP._ring_degree = 1
        PROCESS_GROUP._rank = 0
        PROCESS_GROUP._world_size = 1


# Feature detection flags (yunchang-compatible)
HAS_LONG_CTX_ATTN = True
HAS_FLASH_ATTN = True
HAS_SPARSE_SAGE_ATTENTION = False  # Not supported on Intel GPUs