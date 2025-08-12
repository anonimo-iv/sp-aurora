"""PyTorch attention implementations for fallback and compatibility.

Provides yunchang-compatible attention functions using PyTorch operations.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from enum import Enum


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch implementation of attention forward pass.
    
    Compatible with yunchang's pytorch_attn_forward API.
    Uses torch.nn.functional.scaled_dot_product_attention for efficiency.
    
    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim)
        v: Value tensor of shape (batch, seq_len, num_heads, head_dim)
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        window_size: Window size for local attention (not implemented)
        softcap: Soft capping value (not implemented)
        alibi_slopes: ALiBi slopes (not implemented)
        deterministic: Whether to use deterministic dropout
        return_attn_probs: Whether to return attention probabilities
        
    Returns:
        Tuple of (output, attention_probs or None)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    
    # Ensure correct shape for SDPA (batch, num_heads, seq_len, head_dim)
    if q.dim() == 4 and q.shape[2] != k.shape[2]:
        # Assume shape is (batch, seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        need_transpose = True
    else:
        need_transpose = False
    
    # Apply scaling to query
    q_scaled = q * softmax_scale
    
    # Use PyTorch's optimized SDPA
    out = F.scaled_dot_product_attention(
        q_scaled, k, v,
        attn_mask=None,
        dropout_p=dropout_p if not deterministic else 0.0,
        is_causal=causal,
        scale=1.0  # Already scaled q
    )
    
    # Transpose back if needed
    if need_transpose:
        out = out.transpose(1, 2)
    
    if return_attn_probs:
        # SDPA doesn't return attention probs, compute them if needed
        if need_transpose:
            attn_probs = _compute_attention_probs(
                q.transpose(1, 2), k.transpose(1, 2), 
                softmax_scale, causal
            )
        else:
            attn_probs = _compute_attention_probs(q, k, softmax_scale, causal)
        return out, attn_probs
    
    return out, None


def pytorch_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor, 
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    """PyTorch attention function (forward only).
    
    Compatible with yunchang's pytorch_attn_func API.
    
    Args:
        Same as pytorch_attn_forward
        
    Returns:
        Output tensor
    """
    out, _ = pytorch_attn_forward(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, softcap, alibi_slopes, deterministic,
        return_attn_probs=False
    )
    return out


def _compute_attention_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    causal: bool
) -> torch.Tensor:
    """Compute attention probabilities for debugging/visualization.
    
    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim)
        k: Key tensor (batch, seq_len, num_heads, head_dim)
        softmax_scale: Scaling factor
        causal: Whether to apply causal mask
        
    Returns:
        Attention probabilities (batch, num_heads, seq_len, seq_len)
    """
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    # Apply causal mask if needed
    if causal:
        seq_len = q.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores.masked_fill_(mask, float('-inf'))
    
    # Compute probabilities
    attn_probs = F.softmax(scores, dim=-1)
    
    return attn_probs