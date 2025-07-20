"""
MPI compatibility utilities for Ring Flash Attention

This module provides utilities to make the code compatible with both torchrun and mpiexec
for distributed training on different hardware platforms.
"""

import os
import sys
import socket
from typing import Optional, Dict, Any
import torch
import torch.distributed as dist


def detect_mpi_environment() -> bool:
    """
    Detect if running under MPI environment (mpiexec, mpirun, etc.)
    
    Returns:
        bool: True if MPI environment detected, False otherwise
    """
    # Check for common MPI environment variables
    mpi_vars = [
        'OMPI_COMM_WORLD_RANK',     # Open MPI
        'PMI_RANK',                 # Intel MPI, Slurm
        'PMIX_RANK',                # PMIx-based launchers
        'MV2_COMM_WORLD_RANK',      # MVAPICH2
        'MPI_RANK',                 # Generic MPI
    ]
    
    return any(var in os.environ for var in mpi_vars)


def get_mpi_info() -> Dict[str, Any]:
    """
    Extract MPI process information from environment variables
    
    Returns:
        Dict containing rank, world_size, local_rank, and local_size
    """
    rank = None
    world_size = None
    local_rank = None
    local_size = None
    
    # Try different MPI implementations
    # Open MPI
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
        local_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))
    
    # Intel MPI / Slurm with PMI
    elif 'PMI_RANK' in os.environ:
        rank = int(os.environ['PMI_RANK'])
        world_size = int(os.environ['PMI_SIZE'])
        # For Intel MPI, try to get local rank
        local_rank = int(os.environ.get('MPI_LOCALRANKID', '0'))
        local_size = int(os.environ.get('MPI_LOCALNRANKS', '1'))
    
    # PMIx-based launchers
    elif 'PMIX_RANK' in os.environ:
        rank = int(os.environ['PMIX_RANK'])
        world_size = int(os.environ['PMIX_SIZE'])
        local_rank = int(os.environ.get('PMIX_LOCAL_RANK', '0'))
        local_size = int(os.environ.get('PMIX_LOCAL_SIZE', '1'))
    
    # MVAPICH2
    elif 'MV2_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['MV2_COMM_WORLD_RANK'])
        world_size = int(os.environ['MV2_COMM_WORLD_SIZE'])
        local_rank = int(os.environ.get('MV2_COMM_WORLD_LOCAL_RANK', '0'))
        local_size = int(os.environ.get('MV2_COMM_WORLD_LOCAL_SIZE', '1'))
    
    # Generic fallback
    elif 'MPI_RANK' in os.environ:
        rank = int(os.environ['MPI_RANK'])
        world_size = int(os.environ.get('MPI_SIZE', os.environ.get('MPI_WORLD_SIZE', '1')))
        local_rank = int(os.environ.get('MPI_LOCAL_RANK', '0'))
        local_size = int(os.environ.get('MPI_LOCAL_SIZE', '1'))
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'local_size': local_size
    }


def setup_distributed_environment() -> Dict[str, Any]:
    """
    Setup distributed environment for both torchrun and mpiexec compatibility
    
    Returns:
        Dict with distributed environment information
    """
    
    # Check if we're already in a distributed environment (torchrun)
    if all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']):
        return {
            'launcher': 'torchrun',
            'rank': int(os.environ['RANK']),
            'world_size': int(os.environ['WORLD_SIZE']),
            'local_rank': int(os.environ['LOCAL_RANK']),
            'master_addr': os.environ['MASTER_ADDR'],
            'master_port': int(os.environ['MASTER_PORT'])
        }
    
    # Check for MPI environment
    if detect_mpi_environment():
        mpi_info = get_mpi_info()
        
        if mpi_info['rank'] is None or mpi_info['world_size'] is None:
            raise RuntimeError("Could not determine MPI rank and world size")
        
        # Set up PyTorch distributed environment variables
        os.environ['RANK'] = str(mpi_info['rank'])
        os.environ['WORLD_SIZE'] = str(mpi_info['world_size'])
        os.environ['LOCAL_RANK'] = str(mpi_info['local_rank'] or 0)
        
        # Determine master address and port
        master_addr = os.environ.get('MASTER_ADDR')
        master_port = os.environ.get('MASTER_PORT')
        
        if not master_addr:
            if mpi_info['rank'] == 0:
                # If we're rank 0, use our hostname
                master_addr = socket.gethostname()
                # Broadcast master_addr to other ranks would require MPI calls
                # For simplicity, assume rank 0 hostname is accessible
            else:
                # This is a limitation - in pure MPI we'd need to communicate
                # the master address. For now, assume it's set externally
                master_addr = '127.0.0.1'  # localhost fallback
            
            os.environ['MASTER_ADDR'] = master_addr
        
        if not master_port:
            master_port = 29500  # Default PyTorch port
            os.environ['MASTER_PORT'] = str(master_port)
        
        return {
            'launcher': 'mpiexec',
            'rank': mpi_info['rank'],
            'world_size': mpi_info['world_size'],
            'local_rank': mpi_info['local_rank'] or 0,
            'master_addr': master_addr,
            'master_port': int(master_port)
        }
    
    # Single process fallback
    return {
        'launcher': 'single',
        'rank': 0,
        'world_size': 1,
        'local_rank': 0,
        'master_addr': '127.0.0.1',
        'master_port': 29500
    }


def init_distributed_backend(backend: Optional[str] = None, timeout_seconds: int = 1800) -> bool:
    """
    Initialize distributed backend with proper fallback for different hardware
    
    Args:
        backend: Specific backend to use. If None, auto-detect
        timeout_seconds: Timeout for distributed initialization
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    import datetime
    
    # Skip if already initialized
    if dist.is_initialized():
        return True
    
    # Setup environment
    env_info = setup_distributed_environment()
    
    # Single process case
    if env_info['world_size'] == 1:
        print("Single process detected, skipping distributed initialization")
        return True
    
    # Auto-detect backend if not specified
    if backend is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel GPU - try CCL first
            try:
                import oneccl_bindings_for_pytorch
                backend = 'ccl'
                # Set Intel-specific environment variables
                if env_info['launcher'] == 'mpiexec':
                    os.environ.setdefault('CCL_BACKEND', 'native')
                    os.environ.setdefault('CCL_ATL_TRANSPORT', 'ofi')
                    os.environ.setdefault('FI_PROVIDER', 'cxi')
                    os.environ.setdefault('CCL_ZE_IPC_EXCHANGE', 'drmfd')
                    os.environ.setdefault('CCL_ZE_ENABLE', '1')
            except ImportError:
                backend = 'gloo'  # Fallback for Intel GPU
        elif torch.cuda.is_available():
            backend = 'nccl'  # NVIDIA GPU
        else:
            backend = 'gloo'  # CPU fallback
    
    print(f"Initializing distributed with {backend} backend...")
    print(f"Launcher: {env_info['launcher']}")
    print(f"Rank: {env_info['rank']}/{env_info['world_size']}")
    print(f"Master: {env_info['master_addr']}:{env_info['master_port']}")
    
    try:
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout_seconds)
        )
        print(f"✅ Distributed initialization successful with {backend}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to initialize {backend} backend: {e}")
        
        # Try fallback backends
        fallback_backends = []
        if backend == 'ccl':
            fallback_backends = ['gloo']
        elif backend == 'nccl':
            fallback_backends = ['gloo']
        
        for fallback in fallback_backends:
            try:
                print(f"Trying fallback backend: {fallback}")
                dist.init_process_group(
                    backend=fallback,
                    timeout=datetime.timedelta(seconds=timeout_seconds)
                )
                print(f"✅ Fallback initialization successful with {fallback}")
                return True
            except Exception as fallback_e:
                print(f"❌ Fallback {fallback} also failed: {fallback_e}")
        
        print("❌ All backend initialization attempts failed")
        return False


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("✅ Distributed cleanup successful")
        except Exception as e:
            print(f"⚠️ Warning during distributed cleanup: {e}")


def get_device_for_rank(rank: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for the given rank
    
    Args:
        rank: Process rank. If None, use current rank
        
    Returns:
        torch.device: Device to use for this rank
    """
    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    
    # Intel GPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        if device_count > 1:
            device_id = rank % device_count
            return torch.device(f'xpu:{device_id}')
        else:
            return torch.device('xpu')
    
    # NVIDIA GPU
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = rank % device_count
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cuda')
    
    # CPU fallback
    else:
        return torch.device('cpu')


# Convenience function for easy integration
def setup_mpi_distributed(backend: Optional[str] = None) -> Dict[str, Any]:
    """
    One-stop function to setup distributed training with MPI compatibility
    
    Args:
        backend: Distributed backend to use
        
    Returns:
        Dict with setup information
    """
    env_info = setup_distributed_environment()
    
    # Initialize distributed if needed
    if env_info['world_size'] > 1:
        success = init_distributed_backend(backend)
        if not success:
            raise RuntimeError("Failed to initialize distributed backend")
    
    # Get device
    device = get_device_for_rank()
    
    return {
        'rank': env_info['rank'],
        'world_size': env_info['world_size'],
        'local_rank': env_info['local_rank'],
        'device': device,
        'launcher': env_info['launcher'],
        'backend': dist.get_backend() if dist.is_initialized() else None
    }