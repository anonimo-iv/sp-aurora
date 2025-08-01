"""
Modified intel_all_to_all_4d that falls back to CPU for all-to-all operations
to work around CCL GPU hangs
"""

import torch
import torch.distributed as dist

def intel_all_to_all_4d_with_cpu_fallback(
    input: torch.Tensor, 
    scatter_idx: int = 2, 
    gather_idx: int = 1, 
    group=None
) -> torch.Tensor:
    """
    All-to-all operation that moves to CPU for the collective operation
    then back to original device. This works around CCL GPU hangs.
    """
    assert input.dim() == 4, f"Input must be 4D tensor, got {input.dim()}"
    
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
    
    # Remember original device
    orig_device = input.device
    
    # Move to CPU for all-to-all if on GPU
    if input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu):
        input_cpu = input.cpu()
    else:
        input_cpu = input
        
    if scatter_idx == 2 and gather_idx == 1:
        # Forward: (bs, seq_len/P, num_heads, head_dim) -> (bs, seq_len, num_heads/P, head_dim)
        bs, shard_seq_len, num_heads, head_dim = input_cpu.shape
        seq_len = shard_seq_len * world_size
        shard_heads = num_heads // world_size
        
        # Reshape to prepare for all-to-all
        input_reshaped = input_cpu.reshape(bs, shard_seq_len, world_size, shard_heads, head_dim)
        
        # Transpose to put world_size dimension first for all-to-all
        input_t = input_reshaped.transpose(0, 2).contiguous()
        
        # Perform all-to-all on CPU
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        
        # Reshape back to desired output shape
        output = output.reshape(seq_len, bs, shard_heads, head_dim)
        output = output.transpose(0, 1).contiguous()
        
    elif scatter_idx == 1 and gather_idx == 2:
        # Backward: (bs, seq_len, num_heads/P, head_dim) -> (bs, seq_len/P, num_heads, head_dim)
        bs, seq_len, shard_heads, head_dim = input_cpu.shape
        num_heads = shard_heads * world_size
        shard_seq_len = seq_len // world_size
        
        # Reshape to prepare for all-to-all
        input_reshaped = input_cpu.reshape(bs, world_size, shard_seq_len, shard_heads, head_dim)
        
        # Transpose for all-to-all
        input_t = input_reshaped.transpose(0, 3).transpose(0, 1).contiguous()
        input_t = input_t.reshape(world_size, shard_heads, shard_seq_len, bs, head_dim)
        
        # Perform all-to-all on CPU
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        
        # Reshape back
        output = output.reshape(num_heads, shard_seq_len, bs, head_dim)
        output = output.transpose(0, 2).contiguous()
        
    else:
        raise ValueError(f"Unsupported scatter_idx={scatter_idx}, gather_idx={gather_idx}")
    
    # Move back to original device
    if orig_device != output.device:
        output = output.to(orig_device)
        
    return output