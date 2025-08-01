"""
All-to-all fallback implementations for systems where all_to_all_single doesn't work
(e.g., Intel GPUs with CCL backend)
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Callable
import warnings


# Cache for which methods work on current system
_WORKING_METHODS_CACHE = {}


def all_to_all_single_native(output: torch.Tensor, input: torch.Tensor, group=None) -> torch.Tensor:
    """Native PyTorch all_to_all_single - may not work on all backends"""
    dist.all_to_all_single(output, input, group=group)
    return output


def all_to_all_using_allgather(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int, 
    group=None
) -> torch.Tensor:
    """
    Implement all-to-all using allgather collective.
    This gathers all data then extracts the relevant portion.
    More communication overhead but works on more backends.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if world_size == 1:
        return input
    
    # For forward pass (scatter_idx=2, gather_idx=1)
    if scatter_idx == 2 and gather_idx == 1:
        # Input: (P, seq_len/P, bs, num_heads/P, head_dim)
        # Need to gather from all ranks and reorganize
        
        # Gather all tensors
        tensor_list = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(tensor_list, input, group=group)
        
        # tensor_list[i] has data from rank i
        # We need to extract the i-th chunk from each rank's data
        P, shard_seq, bs, shard_heads, head_dim = input.shape
        
        # Create output tensor
        output_chunks = []
        for src_rank in range(world_size):
            # Extract the chunk meant for this rank from src_rank's data
            # tensor_list[src_rank][rank] is what we need
            chunk = tensor_list[src_rank][rank]  # Shape: (shard_seq, bs, shard_heads, head_dim)
            output_chunks.append(chunk)
        
        # Stack chunks along sequence dimension
        output = torch.cat(output_chunks, dim=0)  # (seq_len, bs, shard_heads, head_dim)
        return output
        
    elif scatter_idx == 1 and gather_idx == 2:
        # Backward pass - similar logic but different dimensions
        # Input: (P, shard_heads, shard_seq, bs, head_dim)
        
        # Gather all tensors
        tensor_list = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(tensor_list, input, group=group)
        
        P, shard_heads, shard_seq, bs, head_dim = input.shape
        
        # Extract and reorganize
        output_chunks = []
        for src_rank in range(world_size):
            chunk = tensor_list[src_rank][rank]  # (shard_heads, shard_seq, bs, head_dim)
            output_chunks.append(chunk)
        
        # Stack along heads dimension
        output = torch.cat(output_chunks, dim=0)  # (num_heads, shard_seq, bs, head_dim)
        
        # Reshape to match expected output
        output = output.reshape(world_size, shard_heads, shard_seq, bs, head_dim)
        output = output.transpose(0, 2).reshape(shard_seq, world_size * shard_heads, bs, head_dim)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()
        
        return output
    else:
        raise ValueError(f"Unsupported scatter_idx={scatter_idx}, gather_idx={gather_idx}")


def all_to_all_using_send_recv(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group=None
) -> torch.Tensor:
    """
    Implement all-to-all using point-to-point send/recv operations.
    Most compatible but potentially slower.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if world_size == 1:
        return input
    
    # Split input into chunks for each rank
    chunks_to_send = list(input.chunk(world_size, dim=0))
    chunks_received = [None] * world_size
    
    # Exchange chunks with all ranks
    for i in range(world_size):
        if i == rank:
            # Keep own chunk
            chunks_received[i] = chunks_to_send[i]
        else:
            # Create receive buffer
            chunks_received[i] = torch.empty_like(chunks_to_send[i])
            
            # Use deterministic ordering to avoid deadlock
            if rank < i:
                # Send first, then receive
                dist.send(chunks_to_send[i], dst=i, group=group)
                dist.recv(chunks_received[i], src=i, group=group)
            else:
                # Receive first, then send
                dist.recv(chunks_received[i], src=i, group=group)
                dist.send(chunks_to_send[i], dst=i, group=group)
    
    # Concatenate received chunks
    output = torch.cat(chunks_received, dim=0)
    
    # Handle dimension reshaping based on scatter/gather indices
    if scatter_idx == 2 and gather_idx == 1:
        # Forward pass reshaping
        _, shard_seq, bs, shard_heads, head_dim = chunks_to_send[0].shape
        seq_len = shard_seq * world_size
        output = output.reshape(seq_len, bs, shard_heads, head_dim)
    elif scatter_idx == 1 and gather_idx == 2:
        # Backward pass reshaping
        _, shard_heads, shard_seq, bs, head_dim = chunks_to_send[0].shape
        num_heads = shard_heads * world_size
        output = output.reshape(num_heads, shard_seq, bs, head_dim)
        output = output.transpose(0, 1).reshape(shard_seq, num_heads, bs, head_dim)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()
    
    return output


def all_to_all_with_cpu_fallback(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group=None
) -> torch.Tensor:
    """
    Perform all-to-all on CPU to avoid GPU backend issues.
    This is the most compatible but slowest option.
    """
    orig_device = input.device
    orig_dtype = input.dtype
    
    # Move to CPU
    if input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu):
        input_cpu = input.cpu().float()  # Use float32 on CPU
    else:
        input_cpu = input.float() if input.dtype == torch.float16 else input
    
    # Perform all-to-all on CPU using the native method
    output_cpu = torch.empty_like(input_cpu)
    dist.all_to_all_single(output_cpu, input_cpu, group=group)
    
    # Move back to original device and dtype
    output = output_cpu.to(device=orig_device, dtype=orig_dtype)
    
    return output


def test_all_to_all_method(method: Callable, input_tensor: torch.Tensor, group=None) -> bool:
    """Test if a specific all-to-all method works on current system"""
    try:
        if method == all_to_all_single_native:
            output = torch.empty_like(input_tensor)
            method(output, input_tensor, group)
        else:
            # For other methods that handle dimensions internally
            output = method(input_tensor, scatter_idx=2, gather_idx=1, group=group)
        return True
    except Exception as e:
        return False


def get_working_all_to_all_method(device_type: str, backend: str) -> Callable:
    """
    Determine which all-to-all method works on current system.
    Results are cached for efficiency.
    """
    cache_key = f"{device_type}_{backend}"
    
    if cache_key in _WORKING_METHODS_CACHE:
        return _WORKING_METHODS_CACHE[cache_key]
    
    # Create small test tensor
    test_tensor = torch.ones(4, 4, device=device_type)
    
    # Test methods in order of preference
    methods = [
        ("native", all_to_all_single_native),
        ("allgather", all_to_all_using_allgather),
        ("send_recv", all_to_all_using_send_recv),
        ("cpu_fallback", all_to_all_with_cpu_fallback)
    ]
    
    for name, method in methods:
        if test_all_to_all_method(method, test_tensor):
            warnings.warn(f"Using {name} method for all-to-all on {device_type} with {backend} backend")
            _WORKING_METHODS_CACHE[cache_key] = method
            return method
    
    raise RuntimeError(f"No working all-to-all method found for {device_type} with {backend} backend")


def all_to_all_fallback(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    method: Optional[str] = None
) -> torch.Tensor:
    """
    All-to-all with automatic fallback to working implementation.
    
    Args:
        input: Input tensor
        scatter_idx: Dimension to scatter across
        gather_idx: Dimension to gather across
        group: Process group
        method: Force specific method ('native', 'allgather', 'send_recv', 'cpu')
                If None, automatically selects working method
    
    Returns:
        Output tensor after all-to-all operation
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
    
    # Get device and backend info
    device_type = input.device.type
    backend = dist.get_backend(group)
    
    # Select method
    if method is None:
        # Check environment variable override
        method = os.environ.get('ULYSSES_ALLTOALL_METHOD', None)
    
    if method == 'native':
        # Prepare for native all-to-all shape requirements
        if scatter_idx == 2 and gather_idx == 1:
            # Forward pass transformation
            output = torch.empty_like(input)
            return all_to_all_single_native(output, input, group)
        else:
            # Need to handle dimension transformations
            raise NotImplementedError("Native method only supports specific dimension patterns")
    
    elif method == 'allgather':
        return all_to_all_using_allgather(input, scatter_idx, gather_idx, group)
    
    elif method == 'send_recv':
        return all_to_all_using_send_recv(input, scatter_idx, gather_idx, group)
    
    elif method == 'cpu':
        return all_to_all_with_cpu_fallback(input, scatter_idx, gather_idx, group)
    
    else:
        # Auto-detect working method
        working_method = get_working_all_to_all_method(device_type, backend)
        if working_method == all_to_all_single_native:
            output = torch.empty_like(input)
            return working_method(output, input, group)
        else:
            return working_method(input, scatter_idx, gather_idx, group)