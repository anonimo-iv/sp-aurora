"""
Intel GPU compatible Ulysses (Sequence Parallel) Flash Attention implementation

Based on the Ulysses attention pattern from yunchang library, adapted for Intel GPUs
using oneCCL for all-to-all communication and Intel Extension for PyTorch (IPEX).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from .utils import update_out_and_lse, get_default_args
import intel_extension_for_pytorch as ipex

# Import the new class-based implementation
from .ulysses.attn_layer import UlyssesAttention, AttnType

try:
    import oneccl_bindings_for_pytorch
    ONECCL_AVAILABLE = True
except ImportError:
    ONECCL_AVAILABLE = False


class IntelSeqAllToAll4D(torch.autograd.Function):
    """
    All-to-all communication for 4D tensors with Intel GPU support.
    Redistributes tensors across sequence or head dimensions for Ulysses attention.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        
        # Handle single process case
        if not dist.is_initialized() or dist.get_world_size(group) == 1:
            return input
            
        return intel_all_to_all_4d(input, scatter_idx, gather_idx, group)
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        # Reverse the scatter and gather indices for backward pass
        return (
            None,
            IntelSeqAllToAll4D.apply(ctx.group, grad_output, ctx.gather_idx, ctx.scatter_idx),
            None,
            None,
        )


def intel_all_to_all_4d(
    input: torch.Tensor, 
    scatter_idx: int = 2, 
    gather_idx: int = 1, 
    group=None
) -> torch.Tensor:
    """
    All-to-all operation for 4D tensors on Intel GPUs.
    
    Args:
        input: Input tensor with shape (bs, seq_len, num_heads, head_dim)
        scatter_idx: Dimension to scatter (split) across processes
        gather_idx: Dimension to gather (concatenate) across processes
        group: Process group for communication
        
    Returns:
        Redistributed tensor
    """
    assert input.dim() == 4, f"Input must be 4D tensor, got {input.dim()}"
    
    # Keep tensor on its current device - don't force XPU
    # This allows the function to work with CPU/gloo backend as well
    
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
        
    if scatter_idx == 2 and gather_idx == 1:
        # Forward: (bs, seq_len/P, num_heads, head_dim) -> (bs, seq_len, num_heads/P, head_dim)
        bs, shard_seq_len, num_heads, head_dim = input.shape
        seq_len = shard_seq_len * world_size
        shard_heads = num_heads // world_size
        
        # Reshape to prepare for all-to-all
        # (bs, seq_len/P, num_heads, head_dim) -> (bs, seq_len/P, P, num_heads/P, head_dim)
        input_reshaped = input.reshape(bs, shard_seq_len, world_size, shard_heads, head_dim)
        
        # Transpose to put world_size dimension first for all-to-all
        # (bs, seq_len/P, P, num_heads/P, head_dim) -> (P, seq_len/P, bs, num_heads/P, head_dim)
        input_t = input_reshaped.transpose(0, 2).contiguous()
        
        # Perform all-to-all
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        
        # Reshape back to desired output shape
        # (P, seq_len/P, bs, num_heads/P, head_dim) -> (seq_len, bs, num_heads/P, head_dim)
        output = output.reshape(seq_len, bs, shard_heads, head_dim)
        
        # (seq_len, bs, num_heads/P, head_dim) -> (bs, seq_len, num_heads/P, head_dim)
        output = output.transpose(0, 1).contiguous()
        
        return output
        
    elif scatter_idx == 1 and gather_idx == 2:
        # Backward: (bs, seq_len, num_heads/P, head_dim) -> (bs, seq_len/P, num_heads, head_dim)
        bs, seq_len, shard_heads, head_dim = input.shape
        num_heads = shard_heads * world_size
        shard_seq_len = seq_len // world_size
        
        # Reshape to prepare for all-to-all
        # (bs, seq_len, num_heads/P, head_dim) -> (bs, P, seq_len/P, num_heads/P, head_dim)
        input_reshaped = input.reshape(bs, world_size, shard_seq_len, shard_heads, head_dim)
        
        # Transpose for all-to-all
        # (bs, P, seq_len/P, num_heads/P, head_dim) -> (P, num_heads/P, seq_len/P, bs, head_dim)
        input_t = input_reshaped.transpose(0, 3).transpose(0, 1).contiguous()
        input_t = input_t.reshape(world_size, shard_heads, shard_seq_len, bs, head_dim)
        
        # Perform all-to-all
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        
        # Reshape back
        # (P, num_heads/P, seq_len/P, bs, head_dim) -> (num_heads, seq_len/P, bs, head_dim)
        output = output.reshape(num_heads, shard_seq_len, bs, head_dim)
        
        # (num_heads, seq_len/P, bs, head_dim) -> (bs, seq_len/P, num_heads, head_dim)
        output = output.transpose(0, 2).contiguous()
        
        return output
        
    else:
        raise ValueError(f"Unsupported scatter_idx={scatter_idx}, gather_idx={gather_idx}")


class IntelUlyssesComm:
    """
    Intel GPU compatible Ulysses communication handler using oneCCL backend
    """
    
    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if process_group else 1
        self.rank = dist.get_rank(process_group) if process_group else 0
        
    def all_to_all_qkv(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        scatter_idx: int = 2,
        gather_idx: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform all-to-all on Q, K, V tensors for Ulysses attention
        """
        q = IntelSeqAllToAll4D.apply(self.process_group, q, scatter_idx, gather_idx)
        k = IntelSeqAllToAll4D.apply(self.process_group, k, scatter_idx, gather_idx)
        v = IntelSeqAllToAll4D.apply(self.process_group, v, scatter_idx, gather_idx)
        return q, k, v
        
    def all_to_all_output(
        self,
        output: torch.Tensor,
        scatter_idx: int = 1,
        gather_idx: int = 2
    ) -> torch.Tensor:
        """
        Perform all-to-all on output tensor to restore original partitioning
        """
        return IntelSeqAllToAll4D.apply(self.process_group, output, scatter_idx, gather_idx)


def intel_ulysses_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Intel GPU compatible Ulysses flash attention forward pass.
    
    Redistributes sequences across GPUs, computes local attention, then redistributes back.
    """
    
    # Ensure tensors are on Intel GPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        q = q.to('xpu')
        k = k.to('xpu')
        v = v.to('xpu')
    
    # Handle single process case
    if not dist.is_initialized() or (process_group is None and dist.get_world_size() == 1):
        # Use PyTorch SDPA for single GPU
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale
        )
        # Calculate dummy LSE for compatibility
        batch_size, num_heads, seq_len, head_dim = q.shape
        lse = torch.zeros(batch_size, num_heads, seq_len, dtype=torch.float32, device=q.device)
        return output, lse
    
    comm = IntelUlyssesComm(process_group)
    
    # CCL backend requires a collective operation before all-to-all (following Ring pattern)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dummy_tensor = torch.tensor([1.0], device=q.device, dtype=torch.float32)
        dist.all_reduce(dummy_tensor)
    
    # Redistribute tensors: scatter sequence dimension, gather head dimension
    # Input: (bs, seq_len/P, num_heads, head_dim)
    # After all-to-all: (bs, seq_len, num_heads/P, head_dim)
    q_redistributed, k_redistributed, v_redistributed = comm.all_to_all_qkv(q, k, v)
    
    # Compute local attention on redistributed tensors
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        # Use IPEX optimized attention if available
        try:
            from .intel_flash_attn_sycl import intel_flash_attention_forward
            output, lse = intel_flash_attention_forward(
                q_redistributed, k_redistributed, v_redistributed,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic
            )
        except ImportError:
            # Fallback to PyTorch SDPA
            # Note: After all-to-all, shape is (bs, seq_len, num_heads/P, head_dim)
            # SDPA expects (batch, num_heads, seq_len, head_dim)
            batch_size, seq_len, num_heads_per_rank, head_dim = q_redistributed.shape
            
            # Transpose to SDPA format
            q_sdpa = q_redistributed.transpose(1, 2)  # (bs, num_heads/P, seq_len, head_dim)
            k_sdpa = k_redistributed.transpose(1, 2)
            v_sdpa = v_redistributed.transpose(1, 2)
            
            output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale
            )
            
            # Transpose back to (bs, seq_len, num_heads/P, head_dim)
            output = output.transpose(1, 2)
            
            # Calculate LSE manually if needed
            scores = torch.matmul(q_sdpa, k_sdpa.transpose(-2, -1)) * softmax_scale
            if causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    
    # Redistribute output back: scatter head dimension, gather sequence dimension
    # Input: (bs, seq_len, num_heads/P, head_dim)
    # After all-to-all: (bs, seq_len/P, num_heads, head_dim)
    output = comm.all_to_all_output(output)
    
    # Also redistribute LSE if needed
    if lse.dim() == 4 and lse.shape[-1] == 1:
        lse = lse.squeeze(-1)  # Remove last dimension if it's 1
    
    return output, lse


def intel_ulysses_flash_attn_backward(
    process_group,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Intel GPU compatible Ulysses flash attention backward pass.
    """
    
    # Ensure tensors are on Intel GPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
        dout = dout.to(device)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        out = out.to(device)
        lse = lse.to(device) if lse is not None else lse
    
    # Handle single process case
    if not dist.is_initialized() or (process_group is None and dist.get_world_size() == 1):
        # Use autograd for single GPU backward
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        
        # Recompute forward pass for gradients
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale
        )
        
        # Compute gradients manually if autograd doesn't work
        if q.grad is None:
            # Manual gradient computation fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
            if causal:
                causal_mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            if dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
            # Gradient w.r.t v
            dv = attn_weights.transpose(-2, -1) @ dout
            
            # Gradient w.r.t attention weights
            dattn = dout @ v.transpose(-2, -1)
            
            # Gradient w.r.t scores
            dscores = attn_weights * (dattn - (dattn * attn_weights).sum(dim=-1, keepdim=True))
            if causal:
                dscores.masked_fill_(causal_mask, 0)
            dscores = dscores * softmax_scale
            
            # Gradient w.r.t q and k
            dq = dscores @ k
            dk = dscores.transpose(-2, -1) @ q
            
            return dq, dk, dv
        else:
            # Use autograd gradients if available
            output.backward(dout)
            return q.grad, k.grad, v.grad
    
    comm = IntelUlyssesComm(process_group)
    
    # CCL backend requires a collective operation before all-to-all (following Ring pattern)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dummy_tensor = torch.tensor([1.0], device=dout.device, dtype=torch.float32)
        dist.all_reduce(dummy_tensor)
    
    # Redistribute tensors for backward pass
    q_redistributed, k_redistributed, v_redistributed = comm.all_to_all_qkv(q, k, v)
    # For dout, we need to redistribute it alone
    dout_redistributed = IntelSeqAllToAll4D.apply(comm.process_group, dout, 2, 1)
    out_redistributed = IntelSeqAllToAll4D.apply(comm.process_group, out, 2, 1) if out is not None else None
    
    # Enable gradients for backward computation
    q_redistributed.requires_grad_(True)
    k_redistributed.requires_grad_(True)
    v_redistributed.requires_grad_(True)
    
    # Compute local attention backward
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            from .intel_flash_attn_sycl import intel_flash_attention_backward
            dq, dk, dv = intel_flash_attention_backward(
                dout_redistributed,
                q_redistributed, k_redistributed, v_redistributed,
                out_redistributed, lse,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic
            )
        except ImportError:
            # Fallback: manual gradient computation
            scores = torch.matmul(q_redistributed, k_redistributed.transpose(-2, -1)) * softmax_scale
            if causal:
                seq_len = scores.shape[-1]
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            if dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
            # Gradient w.r.t v
            dv = attn_weights.transpose(-2, -1) @ dout_redistributed
            
            # Gradient w.r.t attention weights
            dattn = dout_redistributed @ v_redistributed.transpose(-2, -1)
            
            # Gradient w.r.t scores
            dscores = attn_weights * (dattn - (dattn * attn_weights).sum(dim=-1, keepdim=True))
            if causal:
                dscores.masked_fill_(causal_mask, 0)
            dscores = dscores * softmax_scale
            
            # Gradient w.r.t q and k
            dq = dscores @ k_redistributed
            dk = dscores.transpose(-2, -1) @ q_redistributed
    
    # Redistribute gradients back
    dq = comm.all_to_all_output(dq)
    dk = comm.all_to_all_output(dk)
    dv = comm.all_to_all_output(dv)
    
    return dq, dk, dv


class IntelUlyssesFlashAttnFunc(torch.autograd.Function):
    """
    Autograd function for Intel Ulysses Flash Attention
    """
    
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
            
        out, lse = intel_ulysses_flash_attn_forward(
            group,
            q, k, v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        
        return out if not return_attn_probs else (out, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        
        dq, dk, dv = intel_ulysses_flash_attn_backward(
            ctx.group,
            dout,
            q, k, v,
            out,
            lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        
        return dq, dk, dv, None, None, None, None, None, None, None, None


def intel_ulysses_flash_attn_func(
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
    """
    Intel Ulysses Flash Attention function.
    
    Implements sequence parallel attention using all-to-all communication.
    Compatible with Intel GPUs using oneCCL backend.
    
    Args:
        q: Query tensor of shape (batch, seq_len/P, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len/P, num_heads, head_dim)
        v: Value tensor of shape (batch, seq_len/P, num_heads, head_dim)
        dropout_p: Dropout probability
        softmax_scale: Softmax scale factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Local attention window size
        alibi_slopes: ALiBi slopes for position bias
        deterministic: Whether to use deterministic algorithms
        return_attn_probs: Whether to return attention probabilities
        group: Process group for distributed communication
        
    Returns:
        Output tensor of shape (batch, seq_len/P, num_heads, head_dim)
    """
    return IntelUlyssesFlashAttnFunc.apply(
        q, k, v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group
    )


def intel_ulysses_flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Intel Ulysses Flash Attention with packed QKV tensor.
    
    Args:
        qkv: Packed QKV tensor of shape (batch, seq_len/P, 3, num_heads, head_dim)
        Other args same as intel_ulysses_flash_attn_func
        
    Returns:
        Output tensor of shape (batch, seq_len/P, num_heads, head_dim)
    """
    batch, seq_len, three, num_heads, head_dim = qkv.shape
    assert three == 3, f"qkv must have size 3 in dim 2, got {three}"
    
    q, k, v = qkv.unbind(dim=2)
    return intel_ulysses_flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        group=group
    )


def intel_ulysses_flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Intel Ulysses Flash Attention with packed KV tensor.
    
    Args:
        q: Query tensor of shape (batch, seq_len/P, num_heads, head_dim)
        kv: Packed KV tensor of shape (batch, seq_len/P, 2, num_heads, head_dim)
        Other args same as intel_ulysses_flash_attn_func
        
    Returns:
        Output tensor of shape (batch, seq_len/P, num_heads, head_dim)
    """
    batch, seq_len, two, num_heads, head_dim = kv.shape
    assert two == 2, f"kv must have size 2 in dim 2, got {two}"
    
    k, v = kv.unbind(dim=2)
    return intel_ulysses_flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        group=group
    )