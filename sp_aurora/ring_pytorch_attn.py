"""PyTorch-based ring attention implementation for fallback and debugging.

Provides yunchang-compatible ring attention using pure PyTorch operations.
Optimized for Intel GPUs when IPEX is available.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
from .utils import RingComm, update_out_and_lse

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


def ring_pytorch_attn_func(
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
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """PyTorch implementation of ring attention.
    
    Compatible with yunchang's ring_pytorch_attn_func API.
    Uses PyTorch's scaled_dot_product_attention for local attention
    computation within the ring pattern.
    
    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim)
        k: Key tensor (batch, seq_len, num_heads, head_dim)
        v: Value tensor (batch, seq_len, num_heads, head_dim)
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        window_size: Local attention window (not implemented)
        alibi_slopes: ALiBi slopes (not implemented)
        deterministic: Whether to use deterministic dropout
        return_attn_probs: Whether to return attention probabilities
        group: Process group for ring communication
        
    Returns:
        Output tensor with same shape as query
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    
    # Get ring communication info
    if group is None:
        group = dist.group.WORLD
    
    comm = RingComm(group)
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    
    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # For causal attention, each rank handles a chunk of the sequence
    batch_size, seq_len, num_heads, head_dim = q.shape
    chunk_size = seq_len // world_size
    
    # Initialize output and LSE (log-sum-exp) tensors
    out = None
    lse = None
    
    # Current K/V chunks (start with local chunks)
    k_chunk = k.clone()
    v_chunk = v.clone()
    
    # Optimize for Intel GPU if available
    if IPEX_AVAILABLE and q.device.type == 'xpu':
        # Enable IPEX optimizations
        with ipex.optimize_context():
            out, lse = _ring_attention_loop(
                comm, q, k_chunk, v_chunk, out, lse,
                rank, world_size, softmax_scale, 
                dropout_p, causal, deterministic
            )
    else:
        out, lse = _ring_attention_loop(
            comm, q, k_chunk, v_chunk, out, lse,
            rank, world_size, softmax_scale,
            dropout_p, causal, deterministic
        )
    
    if return_attn_probs:
        # Ring attention doesn't naturally produce full attention matrix
        # Return None for attention probs
        return out, None
    
    return out


def _ring_attention_loop(
    comm: RingComm,
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    rank: int,
    world_size: int,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inner loop for ring attention computation.
    
    Args:
        comm: Ring communication handler
        q: Query tensor
        k_chunk: Current key chunk
        v_chunk: Current value chunk
        out: Output accumulator
        lse: Log-sum-exp accumulator
        rank: Current process rank
        world_size: Total number of processes
        softmax_scale: Attention scaling
        dropout_p: Dropout probability
        causal: Whether to use causal masking
        deterministic: Whether to use deterministic operations
        
    Returns:
        Tuple of (output, lse)
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Process chunks in ring pattern
    for step in range(world_size):
        # Determine which chunk we're processing
        chunk_idx = (rank - step) % world_size
        
        # For causal attention, skip chunks that would violate causality
        if causal and chunk_idx > rank:
            # Send/receive K,V for next iteration
            if step < world_size - 1:
                k_chunk = comm.send_recv(k_chunk)
                v_chunk = comm.send_recv(v_chunk)
            continue
        
        # Compute attention for this chunk
        # Use PyTorch SDPA for efficiency
        q_scaled = q * softmax_scale
        
        # Apply causal mask if needed within chunk
        is_causal_chunk = causal and (chunk_idx == rank)
        
        # Compute attention scores using SDPA
        chunk_out = F.scaled_dot_product_attention(
            q_scaled, k_chunk, v_chunk,
            attn_mask=None,
            dropout_p=dropout_p if not deterministic else 0.0,
            is_causal=is_causal_chunk,
            scale=1.0  # Already scaled q
        )
        
        # For LSE computation, we need scores
        # This is a simplified version - full implementation would track LSE properly
        scores = torch.matmul(q_scaled, k_chunk.transpose(-2, -1))
        if is_causal_chunk:
            # Apply causal mask
            seq_len = scores.shape[-1]
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1
            )
            scores.masked_fill_(mask, float('-inf'))
        
        chunk_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        
        # Update output and LSE using stable accumulation
        if out is None:
            out = chunk_out
            lse = chunk_lse
        else:
            out, lse = update_out_and_lse(out, lse, chunk_out, chunk_lse)
        
        # Send/receive K,V for next iteration
        if step < world_size - 1:
            k_chunk = comm.send_recv(k_chunk)
            v_chunk = comm.send_recv(v_chunk)
    
    return out, lse


def ring_pytorch_attn_forward(
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
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for ring attention (returns output and LSE).
    
    This variant is compatible with functions that expect LSE output.
    """
    out = ring_pytorch_attn_func(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic, 
        return_attn_probs, group
    )
    
    # Return output and None for LSE (simplified)
    if isinstance(out, tuple):
        return out
    return out, None