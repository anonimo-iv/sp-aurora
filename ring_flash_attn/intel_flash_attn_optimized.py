"""
Optimized Intel GPU flash attention implementation with LSE support
"""

import torch
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
from typing import Tuple, Optional
import math


def _flash_attn_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tiled flash attention implementation for Intel GPUs.
    Uses blocking to reduce memory usage while computing LSE.
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[-2]
    
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    
    # Initialize output and LSE accumulators
    out = torch.zeros_like(q)
    lse = torch.full((batch_size, num_heads, seq_len_q), float('-inf'), device=q.device, dtype=torch.float32)
    
    # Compute attention in blocks to reduce memory usage
    for i in range(0, seq_len_q, block_size):
        i_end = min(i + block_size, seq_len_q)
        q_block = q[:, :, i:i_end, :]
        
        # Initialize block accumulators
        block_out = torch.zeros_like(q_block)
        block_lse = torch.full((batch_size, num_heads, i_end - i), float('-inf'), 
                               device=q.device, dtype=torch.float32)
        
        for j in range(0, seq_len_k, block_size):
            j_end = min(j + block_size, seq_len_k)
            
            # Skip blocks that are masked out by causal mask
            if causal and j >= i_end:
                continue
            
            k_block = k[:, :, j:j_end, :]
            v_block = v[:, :, j:j_end, :]
            
            # Compute attention scores for this block
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * softmax_scale
            
            # Apply causal mask within block
            if causal and j < i_end:
                block_seq_len_q = i_end - i
                block_seq_len_k = j_end - j
                causal_mask = torch.triu(
                    torch.ones(block_seq_len_q, block_seq_len_k, device=scores.device, dtype=torch.bool),
                    diagonal=j - i + 1
                )
                scores.masked_fill_(causal_mask, float('-inf'))
            
            # Compute block LSE
            block_lse_update = torch.logsumexp(scores, dim=-1)
            
            # Update LSE using log-sum-exp trick
            max_lse = torch.maximum(block_lse, block_lse_update)
            block_lse = max_lse + torch.log(
                torch.exp(block_lse - max_lse) + torch.exp(block_lse_update - max_lse)
            )
            
            # Compute attention weights
            attn_weights = torch.exp(scores - block_lse_update.unsqueeze(-1))
            
            # Apply dropout if needed
            if dropout_p > 0 and q.requires_grad:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
            # Update output accumulator
            scale_factor = torch.exp(block_lse_update - block_lse).unsqueeze(-1)
            block_out = block_out * scale_factor + torch.matmul(attn_weights, v_block)
        
        # Write back to output
        out[:, :, i:i_end, :] = block_out
        lse[:, :, i:i_end] = block_lse
    
    return out, lse


def intel_flash_attn_forward_optimized(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized Intel GPU flash attention with multiple implementation strategies.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    # Device handling - ensure all tensors are on the same device
    device = q.device
    
    # If XPU is available and tensors aren't on XPU, move them
    if hasattr(torch, 'xpu') and torch.xpu.is_available() and device.type != 'xpu':
        device = torch.device('xpu')
        q = q.to(device)
        k = k.to(device) 
        v = v.to(device)
    
    # Ensure all tensors are on the same device
    if k.device != device:
        k = k.to(device)
    if v.device != device:
        v = v.to(device)
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Strategy 1: Try IPEX optimized SDPA if available
    if hasattr(ipex, 'llm') and hasattr(ipex.llm, 'functional'):
        try:
            # Check if IPEX has flash attention with LSE support
            if hasattr(ipex.llm.functional, 'scaled_dot_product_attention'):
                # Use IPEX's optimized kernel
                result = ipex.llm.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=dropout_p if q.requires_grad else 0.0,
                    is_causal=causal,
                    scale=softmax_scale,
                    return_debug_mask=True  # This might return LSE
                )
                
                if isinstance(result, tuple) and len(result) >= 2:
                    attn_output, debug_info = result[0], result[1]
                    # Extract LSE from debug info if available
                    if hasattr(debug_info, 'logsumexp') or isinstance(debug_info, dict) and 'lse' in debug_info:
                        lse = debug_info.logsumexp if hasattr(debug_info, 'logsumexp') else debug_info['lse']
                        return attn_output, lse.to(torch.float32)
        except Exception:
            pass
    
    # Strategy 2: Use PyTorch's SDPA with separate LSE computation
    if hasattr(F, 'scaled_dot_product_attention') and not return_softmax and seq_len <= 2048:
        try:
            # Use SDPA for efficient attention computation
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
            ):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=dropout_p if q.requires_grad else 0.0,
                    is_causal=causal,
                    scale=softmax_scale
                )
            
            # Compute LSE efficiently with tiling for large sequences
            if seq_len > 512:
                # Use tiled computation for LSE only
                _, lse = _flash_attn_forward_tiled(
                    q, k, v, 
                    dropout_p=0.0,  # Don't apply dropout in LSE computation
                    softmax_scale=softmax_scale, 
                    causal=causal,
                    block_size=256
                )
                return attn_output, lse
            else:
                # For smaller sequences, compute LSE directly
                scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
                if causal:
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), 
                        diagonal=1
                    )
                    scores = scores.masked_fill(causal_mask, float('-inf'))
                lse = torch.logsumexp(scores, dim=-1, keepdim=False)
                return attn_output, lse.to(torch.float32)
        except Exception:
            pass
    
    # Strategy 3: Use tiled implementation for memory efficiency
    if seq_len > 1024:
        return _flash_attn_forward_tiled(
            q, k, v, 
            dropout_p=dropout_p, 
            softmax_scale=softmax_scale, 
            causal=causal,
            block_size=128
        )
    
    # Strategy 4: Fallback to standard implementation for small sequences
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), 
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    lse = torch.logsumexp(scores, dim=-1, keepdim=False)
    attn_weights = torch.exp(scores - lse.unsqueeze(-1))
    
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    attn_output = torch.matmul(attn_weights, v)
    
    return attn_output, lse.to(torch.float32)