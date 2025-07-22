"""
Intel GPU compatible Ring Flash Attention implementation with fixed P2P communication
"""

import torch
import torch.distributed as dist
try:
    from .intel_flash_attn import _flash_attn_forward, _flash_attn_backward
except ImportError:
    # Fallback to standard flash attention if intel version not available
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .intel_utils_fixed import IntelRingComm, update_out_and_lse, get_default_args
import intel_extension_for_pytorch as ipex


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
    """Intel GPU compatible ring flash attention forward pass with fixed P2P"""
    
    # Ensure tensors are on Intel GPU (XPU)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        q = q.to('xpu')
        k = k.to('xpu') 
        v = v.to('xpu')
    
    # Handle single process case (no distributed communication needed)
    if not dist.is_initialized() or (process_group is None and dist.get_world_size() == 1):
        # Single process - use regular flash attention
        from .intel_flash_attn import intel_flash_attn_forward
        return intel_flash_attn_forward(q, k, v, causal=causal, softmax_scale=softmax_scale)
    
    comm = IntelRingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            # Use synchronous P2P communication
            next_k, next_v = comm.send_recv_kv(k, v)

        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
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
            
            # Use Intel compatible flash attention
            block_out, block_lse = _flash_attn_forward(**params)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            # No need to call commit/wait - communication is already synchronous
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
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
    """Intel GPU compatible ring flash attention backward pass with fixed P2P"""
    
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
            # Use synchronous P2P communication
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
                # Communication is synchronous
                if step + 1 != kv_comm.world_size:
                    next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
                else:
                    dk = dk + block_dk_buffer
                    dv = dv + block_dv_buffer
        elif step != 0:
            # Receive gradients from next step
            if step + 1 != kv_comm.world_size:
                dk, dv = d_kv_comm.send_recv_kv(dk, dv)

        if step + 1 != kv_comm.world_size:
            # Communication is synchronous - no need to wait
            k, v = next_k, next_v

    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)
    return dq, dk, dv


class IntelRingFlashAttnFunc(torch.autograd.Function):
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
        return_attn_probs,
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
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_attn_probs else (out, softmax_lse, None)

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
    """Intel GPU compatible ring flash attention function with fixed P2P"""
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