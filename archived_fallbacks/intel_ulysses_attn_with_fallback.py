"""
Intel-optimized Ulysses (Sequence Parallel) Attention for XPU/GPU
With fallback support for all-to-all operations
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Function
from typing import Optional, Tuple

# Import the fallback all-to-all implementation
from .comm.all_to_all_fallback import all_to_all_fallback

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Import attention implementations
try:
    from .intel_flash_attn_interface import flash_attn_func
except ImportError:
    # Use simple scaled dot product attention as fallback
    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, 
                       window_size=(-1, -1), alibi_slopes=None, deterministic=False):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )


def intel_all_to_all_4d_with_fallback(
    input: torch.Tensor, 
    scatter_idx: int = 2, 
    gather_idx: int = 1, 
    group=None,
    method: Optional[str] = None
) -> torch.Tensor:
    """
    4D tensor all-to-all operation with fallback support.
    
    This function reshapes the input tensor and uses the fallback-enabled
    all-to-all implementation that works on systems where native all_to_all_single
    fails (e.g., Intel GPUs with CCL backend).
    
    Args:
        input: 4D input tensor
        scatter_idx: dimension to scatter (split) across processes
        gather_idx: dimension to gather (concatenate) across processes  
        group: process group for communication
        method: force specific all-to-all method ('native', 'allgather', 'send_recv', 'cpu')
    
    Returns:
        Output tensor after all-to-all redistribution
    """
    assert input.dim() == 4, f"Expected 4D tensor, got {input.dim()}D"
    
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
        
    if scatter_idx == 2 and gather_idx == 1:
        # Forward: (bs, seq_len/P, num_heads, head_dim) -> (bs, seq_len, num_heads/P, head_dim)
        bs, shard_seq_len, num_heads, head_dim = input.shape
        seq_len = shard_seq_len * world_size
        shard_heads = num_heads // world_size
        
        # Reshape to prepare for all-to-all
        input_reshaped = input.reshape(bs, shard_seq_len, world_size, shard_heads, head_dim)
        
        # Transpose to put world_size dimension first
        input_t = input_reshaped.transpose(0, 2).contiguous()
        
        # Use fallback all-to-all
        output = all_to_all_fallback(input_t, scatter_idx=0, gather_idx=0, group=group, method=method)
        
        # Reshape back to desired output shape
        output = output.reshape(seq_len, bs, shard_heads, head_dim)
        output = output.transpose(0, 1).contiguous()
        
        return output
        
    elif scatter_idx == 1 and gather_idx == 2:
        # Backward: (bs, seq_len, num_heads/P, head_dim) -> (bs, seq_len/P, num_heads, head_dim)
        bs, seq_len, shard_heads, head_dim = input.shape
        num_heads = shard_heads * world_size
        shard_seq_len = seq_len // world_size
        
        # Reshape to prepare for all-to-all
        input_reshaped = input.reshape(bs, world_size, shard_seq_len, shard_heads, head_dim)
        
        # Transpose for all-to-all
        input_t = input_reshaped.transpose(0, 3).transpose(0, 1).contiguous()
        input_t = input_t.reshape(world_size, shard_heads, shard_seq_len, bs, head_dim)
        
        # Use fallback all-to-all
        output = all_to_all_fallback(input_t, scatter_idx=0, gather_idx=0, group=group, method=method)
        
        # Reshape back
        output = output.reshape(num_heads, shard_seq_len, bs, head_dim)
        output = output.transpose(0, 1).transpose(1, 2).contiguous()
        
        return output
    else:
        raise ValueError(f"Invalid scatter_idx ({scatter_idx}) and gather_idx ({gather_idx}) combination")


class IntelSeqAllToAll4DWithFallback(Function):
    """Autograd function for 4D all-to-all with gradient support and fallback"""
    
    @staticmethod
    def forward(ctx, input, scatter_idx, gather_idx, group=None, method=None):
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.group = group
        ctx.method = method
        return intel_all_to_all_4d_with_fallback(input, scatter_idx, gather_idx, group, method)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse scatter and gather indices for backward pass
        return (
            intel_all_to_all_4d_with_fallback(
                grad_output, 
                ctx.gather_idx, 
                ctx.scatter_idx, 
                ctx.group,
                ctx.method
            ),
            None, None, None, None
        )


class IntelUlyssesAttentionWithFallback:
    """
    Ulysses Attention implementation with fallback support for all-to-all operations.
    This allows the attention to work on Intel GPUs even when CCL doesn't support
    all_to_all_single on XPU devices.
    """
    
    def __init__(self, process_group=None, all_to_all_method=None):
        """
        Args:
            process_group: Process group for distributed communication
            all_to_all_method: Force specific all-to-all method 
                              ('native', 'allgather', 'send_recv', 'cpu', None for auto)
        """
        self.process_group = process_group
        self.all_to_all_method = all_to_all_method
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    
    def all_to_all_qkv(self, q, k, v):
        """Redistribute QKV tensors using sequence parallelism with fallback"""
        scatter_idx = 2  # scatter across num_heads dimension
        gather_idx = 1   # gather across sequence dimension
        
        q = IntelSeqAllToAll4DWithFallback.apply(
            q, scatter_idx, gather_idx, self.process_group, self.all_to_all_method
        )
        k = IntelSeqAllToAll4DWithFallback.apply(
            k, scatter_idx, gather_idx, self.process_group, self.all_to_all_method
        )
        v = IntelSeqAllToAll4DWithFallback.apply(
            v, scatter_idx, gather_idx, self.process_group, self.all_to_all_method
        )
        
        return q, k, v
    
    def all_to_all_output(self, out):
        """Redistribute output tensor back with fallback"""
        scatter_idx = 1  # scatter across sequence dimension
        gather_idx = 2   # gather across num_heads dimension
        
        return IntelSeqAllToAll4DWithFallback.apply(
            out, scatter_idx, gather_idx, self.process_group, self.all_to_all_method
        )


def intel_ulysses_flash_attn_forward_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    process_group=None,
    all_to_all_method: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ulysses Flash Attention forward pass with fallback all-to-all support.
    
    Additional Args:
        all_to_all_method: Force specific all-to-all method 
                          ('native', 'allgather', 'send_recv', 'cpu', None for auto)
    """
    if process_group is None:
        # No process group = single GPU, just use regular attention
        out = flash_attn_func(
            q, k, v, dropout_p, softmax_scale, causal, 
            window_size, alibi_slopes, deterministic
        )
        lse = None
        return out, lse
    
    # Create Ulysses attention instance with fallback support
    ulysses_attn = IntelUlyssesAttentionWithFallback(process_group, all_to_all_method)
    
    # Redistribute QKV tensors across sequence dimension
    q_redistributed, k_redistributed, v_redistributed = ulysses_attn.all_to_all_qkv(q, k, v)
    
    # Perform local attention on redistributed tensors
    out_local = flash_attn_func(
        q_redistributed, k_redistributed, v_redistributed,
        dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic
    )
    
    # Redistribute output back
    out = ulysses_attn.all_to_all_output(out_local)
    
    # For now, return None for lse (log-sum-exp) as it requires additional handling
    lse = None
    
    return out, lse


# Convenience functions that match the original API
def ulysses_flash_attn_func_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    process_group=None,
    all_to_all_method: Optional[str] = None
) -> torch.Tensor:
    """
    Ulysses Flash Attention with automatic fallback for all-to-all operations.
    
    This function automatically selects a working all-to-all implementation
    based on the current system configuration.
    """
    out, _ = intel_ulysses_flash_attn_forward_with_fallback(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic,
        process_group, all_to_all_method
    )
    return out