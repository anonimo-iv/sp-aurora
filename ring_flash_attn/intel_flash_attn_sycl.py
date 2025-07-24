"""
Intel GPU Flash Attention with SYCL kernels
This module provides a drop-in replacement for the existing flash attention
implementation with optimized SYCL kernels.
"""

import torch
from typing import Tuple, Optional
import warnings

# Try to import the SYCL module
try:
    import ring_flash_attn.sycl_flash_attn as sycl_fa
    HAS_SYCL = True
except ImportError:
    HAS_SYCL = False
    warnings.warn("SYCL Flash Attention module not found. Please build the SYCL kernels first.")

# Fall back to existing implementation
from .intel_flash_attn import intel_flash_attn_forward as _fallback_forward
from .intel_flash_attn import intel_flash_attn_backward as _fallback_backward


def is_sycl_available() -> bool:
    """Check if SYCL flash attention is available"""
    return HAS_SYCL and sycl_fa.is_available()


def get_sycl_device_info() -> dict:
    """Get Intel GPU device information"""
    if not HAS_SYCL:
        return {"error": "SYCL module not available"}
    
    try:
        return sycl_fa.get_device_info()
    except Exception as e:
        return {"error": str(e)}


def intel_flash_attn_forward_sycl(
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
    Intel GPU Flash Attention forward pass using SYCL kernels.
    
    This function automatically falls back to the Python implementation
    if SYCL kernels are not available.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len_q, head_dim]
        k: Key tensor [batch, num_heads, seq_len_k, head_dim]
        v: Value tensor [batch, num_heads, seq_len_k, head_dim]
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for QK^T
        causal: Whether to apply causal masking
        window_size: Local attention window size
        alibi_slopes: ALiBi slopes (not supported in SYCL version)
        return_softmax: Whether to return softmax weights (not supported)
        
    Returns:
        output: Attention output [batch, num_heads, seq_len_q, head_dim]
        lse: Log-sum-exp values [batch, num_heads, seq_len_q]
    """
    
    # Check if we can use SYCL
    if not is_sycl_available():
        return _fallback_forward(
            q, k, v, dropout_p, softmax_scale, causal, 
            window_size, alibi_slopes, return_softmax, **kwargs
        )
    
    # Validate unsupported features
    if alibi_slopes is not None:
        warnings.warn("ALiBi slopes not supported in SYCL version, falling back to Python")
        return _fallback_forward(
            q, k, v, dropout_p, softmax_scale, causal,
            window_size, alibi_slopes, return_softmax, **kwargs
        )
    
    if return_softmax:
        warnings.warn("return_softmax not supported in SYCL version, falling back to Python")
        return _fallback_forward(
            q, k, v, dropout_p, softmax_scale, causal,
            window_size, alibi_slopes, return_softmax, **kwargs
        )
    
    # Ensure tensors are float32 (SYCL kernel requirement)
    orig_dtype = q.dtype
    if orig_dtype != torch.float32:
        q = q.float()
        k = k.float()
        v = v.float()
    
    # Calculate softmax scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    try:
        # Call SYCL kernel
        output, lse = sycl_fa.forward(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1]
        )
        
        # Convert back to original dtype if needed
        if orig_dtype != torch.float32:
            output = output.to(orig_dtype)
            # LSE should remain float32 for numerical stability
        
        return output, lse
        
    except Exception as e:
        warnings.warn(f"SYCL kernel failed: {e}, falling back to Python implementation")
        return _fallback_forward(
            q, k, v, dropout_p, softmax_scale, causal,
            window_size, alibi_slopes, return_softmax, **kwargs
        )


def intel_flash_attn_backward_sycl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    **kwargs
):
    """
    Intel GPU Flash Attention backward pass.
    
    Currently falls back to autograd as SYCL backward is not implemented.
    """
    # For now, just use the fallback
    return _fallback_backward(
        dout, q, k, v, out, softmax_lse, dq, dk, dv,
        dropout_p, softmax_scale, causal, window_size,
        alibi_slopes, deterministic, **kwargs
    )


# Compatibility wrappers
def _flash_attn_forward(*args, **kwargs):
    """Compatibility wrapper for flash_attn interface"""
    return intel_flash_attn_forward_sycl(*args, **kwargs)


def _flash_attn_backward(*args, **kwargs):
    """Compatibility wrapper for flash_attn interface"""
    return intel_flash_attn_backward_sycl(*args, **kwargs)


# Auto-select best implementation
def auto_select_flash_attn_forward(*args, **kwargs):
    """
    Automatically select the best flash attention implementation.
    Prefers SYCL > Optimized Python > Basic Python
    """
    if is_sycl_available():
        return intel_flash_attn_forward_sycl(*args, **kwargs)
    else:
        # Try to use optimized Python version
        try:
            from .intel_flash_attn_optimized import intel_flash_attn_forward_optimized
            return intel_flash_attn_forward_optimized(*args, **kwargs)
        except ImportError:
            # Fall back to basic version
            return _fallback_forward(*args, **kwargs)