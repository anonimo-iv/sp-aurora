"""
Intel GPU compatible Ring Flash Attention implementation
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .utils import IntelRingComm, update_out_and_lse, get_default_args
from .intel_flash_attn import _flash_attn_forward, _flash_attn_backward
import intel_extension_for_pytorch as ipex
import time
import os
import math
from typing import Optional, Tuple, Any


def intel_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    """Intel GPU compatible ring flash attention forward pass"""
    
    # Ensure tensors are on Intel GPU (XPU)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        q = q.to('xpu')
        k = k.to('xpu') 
        v = v.to('xpu')
    
    # Convert from [batch, seq_len, num_heads, head_dim] to [batch, num_heads, seq_len, head_dim]
    if q.dim() == 4 and q.shape[1] != k.shape[1]:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    
    # Handle single process case (no distributed communication needed)
    if not dist.is_initialized() or (process_group is None and dist.get_world_size() == 1):
        # Single process - use PyTorch SDPA
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale
        )
        # Calculate LSE for compatibility
        batch_size, num_heads, seq_len, head_dim = q.shape
        lse = torch.zeros(batch_size, num_heads, seq_len, dtype=torch.float32, device=q.device)
        return output, lse
    
    comm = IntelRingComm(process_group)
    
    # CCL backend requires a collective operation before P2P communication
    # Only do this if we have multiple ranks in the ring
    if comm.world_size > 1:
        dummy_tensor = torch.tensor([1.0], device='xpu')
        dist.all_reduce(dummy_tensor, group=process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    
    for step in range(comm.world_size):
        
        # Communication phase
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        # Computation phase
        if not causal or step <= comm.rank:
            # Try to use Intel flash attention which returns LSE
            try:
                from .intel_flash_attn import intel_flash_attn_forward
                
                # Apply causal mask logic based on ring step
                # In ring attention, causal mask should consider global position
                # Step 0: Each rank processes its own K,V - use causal mask
                # Step > 0: Processing K,V from other ranks - only use causal if it's from an earlier rank
                is_causal_step = causal and (step == 0)
                
                block_out, block_lse = intel_flash_attn_forward(
                    q, k, v,
                    dropout_p=dropout_p,
                    causal=is_causal_step,
                    softmax_scale=softmax_scale
                )
            except Exception:
                # Fallback: compute attention manually with proper LSE
                batch_size, num_heads, seq_len_q, head_dim = q.shape
                seq_len_k = k.shape[2]
                
                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
                
                # Apply causal mask considering ring position
                if causal:
                    if step == 0:
                        # First step: standard causal mask
                        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=scores.device), diagonal=1)
                        scores.masked_fill_(causal_mask.bool(), float('-inf'))
                    elif step > comm.rank:
                        # Future steps: mask everything (no attention to future positions)
                        scores.fill_(float('-inf'))
                
                # Calculate LSE
                block_lse = torch.logsumexp(scores, dim=-1, keepdim=False)
                
                # Compute attention weights
                attn_weights = torch.exp(scores - block_lse.unsqueeze(-1))
                
                # Apply dropout if needed
                if dropout_p > 0:
                    attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
                
                # Compute output
                block_out = torch.matmul(attn_weights, v)
            
            # Update accumulated output and LSE
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        # Synchronization phase
        if step + 1 != comm.world_size:
            comm.commit()
            comm.wait()
            k, v = next_k, next_v
        

    # Final processing
    # out is currently in [batch, num_heads, seq_len, head_dim] format
    # The update_out_and_lse function returns in [batch, num_heads, seq_len, head_dim] format
    # Convert to [batch, seq_len, num_heads, head_dim] for compatibility with the expected interface
    if out.dim() == 4 and out.shape[1] != out.shape[2]:  # Likely [batch, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous()
    out = out.to(q.dtype)
    
    if lse.dim() == 4:
        lse = lse.squeeze(dim=-1)
    if lse.dim() == 3 and lse.shape[1] != lse.shape[2]:  # Likely [batch, num_heads, seq_len]
        lse = lse.transpose(1, 2).contiguous()
    
    return out, lse


def intel_ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    """Intel GPU compatible ring flash attention backward pass"""
    
    # Ensure tensors are on Intel GPU (XPU) 
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
        dout = dout.to(device)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        out = out.to(device)
        softmax_lse = softmax_lse.to(device)
    
    kv_comm = IntelRingComm(process_group)
    d_kv_comm = IntelRingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            params = get_default_args(_flash_attn_backward).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": bwd_causal,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )
            
            # For Intel GPU, use autograd-based backward computation
            # Create requires_grad tensors for backward pass
            q_grad = q.clone().detach().requires_grad_(True)
            k_grad = k.clone().detach().requires_grad_(True) 
            v_grad = v.clone().detach().requires_grad_(True)
            
            # Forward pass to compute attention
            attn_out, _ = _flash_attn_forward(
                q_grad, k_grad, v_grad,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=bwd_causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes
            )
            
            # Compute gradients via autograd
            attn_out.backward(dout)
            
            block_dq_buffer.copy_(q_grad.grad if q_grad.grad is not None else torch.zeros_like(q))
            block_dk_buffer.copy_(k_grad.grad if k_grad.grad is not None else torch.zeros_like(k))
            block_dv_buffer.copy_(v_grad.grad if v_grad.grad is not None else torch.zeros_like(v))

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                # Communication is now synchronous, no need to wait
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            # Communication is now synchronous, no need to wait
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            # Communication is now synchronous, no need to wait
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    # Communication is now synchronous, no need to wait

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class IntelRingFlashAttnFunc(torch.autograd.Function):
    """Intel GPU compatible Ring Flash Attention Function"""
    
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        
        out, softmax_lse = intel_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = intel_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def intel_ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    """Intel GPU compatible QKV packed ring flash attention"""
    return IntelRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def intel_ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    """Intel GPU compatible KV packed ring flash attention"""
    return IntelRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def intel_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    """Intel GPU compatible ring flash attention function"""
    
    # Handle single-process case
    if not dist.is_initialized() or (group is None and dist.get_world_size() == 1):
        # Use regular Intel flash attention for single process
        from .intel_flash_attn import intel_flash_attn_forward
        
        # Convert to the format expected by single-GPU implementation
        q_single = q.transpose(1, 2)  # (batch, seqlen, nheads, d) -> (batch, nheads, seqlen, d)
        k_single = k.transpose(1, 2)
        v_single = v.transpose(1, 2)
        
        out, lse = intel_flash_attn_forward(
            q_single, k_single, v_single, 
            dropout_p=dropout_p,
            causal=causal, 
            softmax_scale=softmax_scale
        )
        # Convert back to expected format
        out = out.transpose(1, 2)  # (batch, nheads, seqlen, d) -> (batch, seqlen, nheads, d)
        if return_attn_probs:
            return out, lse, None
        else:
            return out
    
    return IntelRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


# Backward compatibility aliases for drop-in replacement
RingFlashAttnFunc = IntelRingFlashAttnFunc
ring_flash_attn_func = intel_ring_flash_attn_func
ring_flash_attn_kvpacked_func = intel_ring_flash_attn_kvpacked_func
ring_flash_attn_qkvpacked_func = intel_ring_flash_attn_qkvpacked_func