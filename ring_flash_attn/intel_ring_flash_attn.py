"""
Intel GPU compatible Ring Flash Attention implementation
"""

import torch
import torch.distributed as dist
from .intel_flash_attn import _flash_attn_forward, _flash_attn_backward
from .intel_utils import IntelRingComm, update_out_and_lse, get_default_args
import intel_extension_for_pytorch as ipex
import time
import os


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
    
    start_time = time.time()
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"\n[Rank {rank}] ========== intel_ring_flash_attn_forward START ==========")
    print(f"[Rank {rank}] Time: {start_time}")
    print(f"[Rank {rank}] PID: {os.getpid()}")
    print(f"[Rank {rank}] Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    print(f"[Rank {rank}] Input dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
    print(f"[Rank {rank}] Input devices - q: {q.device}, k: {k.device}, v: {v.device}")
    print(f"[Rank {rank}] Parameters - dropout_p: {dropout_p}, causal: {causal}, softmax_scale: {softmax_scale}")
    print(f"[Rank {rank}] dist.is_initialized: {dist.is_initialized()}")
    if dist.is_initialized():
        print(f"[Rank {rank}] Backend: {dist.get_backend()}")
        print(f"[Rank {rank}] World size: {dist.get_world_size()}")
    
    # Ensure tensors are on Intel GPU (XPU)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_start = time.time()
        q = q.to('xpu')
        k = k.to('xpu') 
        v = v.to('xpu')
        device_elapsed = time.time() - device_start
        print(f"[Rank {rank}] Tensors moved to XPU in {device_elapsed:.3f}s")
    
    # Handle single process case (no distributed communication needed)
    if not dist.is_initialized() or (process_group is None and dist.get_world_size() == 1):
        # Single process - use regular flash attention
        print(f"[Rank {rank}] Single process mode - using regular flash attention")
        from .intel_flash_attn import intel_flash_attn_forward
        return intel_flash_attn_forward(q, k, v, causal=causal, softmax_scale=softmax_scale)
    
    comm_start = time.time()
    print(f"[Rank {rank}] Creating IntelRingComm with process_group={process_group}")
    comm = IntelRingComm(process_group)
    comm_elapsed = time.time() - comm_start
    print(f"[Rank {rank}] IntelRingComm created in {comm_elapsed:.3f}s")
    
    # CCL backend requires a collective operation before P2P communication
    print(f"[Rank {rank}] Performing dummy all_reduce to initialize CCL communicators...")
    dummy_tensor = torch.tensor([1.0], device='xpu')
    dist.all_reduce(dummy_tensor)
    print(f"[Rank {rank}] CCL communicators initialized")

    out = None
    lse = None

    next_k, next_v = None, None

    loop_start = time.time()
    print(f"[Rank {comm.rank}] Starting main ring communication loop with {comm.world_size} steps")
    
    for step in range(comm.world_size):
        step_start = time.time()
        print(f"\n[Rank {comm.rank}] ===== Step {step}/{comm.world_size-1} START =====")
        
        # Communication phase
        if step + 1 != comm.world_size:
            comm_phase_start = time.time()
            print(f"[Rank {comm.rank}] Step {step}: Preparing to send k.shape={k.shape}, v.shape={v.shape}")
            print(f"[Rank {comm.rank}] Step {step}: k.device={k.device}, v.device={v.device}")
            print(f"[Rank {comm.rank}] Step {step}: Calling send_recv_kv...")
            
            try:
                next_k, next_v = comm.send_recv_kv(k, v)
                comm_phase_elapsed = time.time() - comm_phase_start
                print(f"[Rank {comm.rank}] Step {step}: send_recv_kv returned in {comm_phase_elapsed:.3f}s")
                print(f"[Rank {comm.rank}] Step {step}: next_k.shape={next_k.shape}, next_v.shape={next_v.shape}")
            except Exception as e:
                print(f"[Rank {comm.rank}] Step {step}: ERROR in send_recv_kv: {e}")
                raise

        # Computation phase
        if not causal or step <= comm.rank:
            comp_phase_start = time.time()
            print(f"[Rank {comm.rank}] Step {step}: Starting computation phase (causal={causal}, step={step}, rank={comm.rank})")
            
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
            print(f"[Rank {comm.rank}] Step {step}: Calling _flash_attn_forward...")
            try:
                block_out, block_lse = _flash_attn_forward(**params)
                print(f"[Rank {comm.rank}] Step {step}: _flash_attn_forward completed")
                print(f"[Rank {comm.rank}] Step {step}: block_out.shape={block_out.shape}, block_lse.shape={block_lse.shape}")
                
                print(f"[Rank {comm.rank}] Step {step}: Calling update_out_and_lse...")
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                print(f"[Rank {comm.rank}] Step {step}: update_out_and_lse completed")
                
                comp_phase_elapsed = time.time() - comp_phase_start
                print(f"[Rank {comm.rank}] Step {step}: Computation phase completed in {comp_phase_elapsed:.3f}s")
            except Exception as e:
                print(f"[Rank {comm.rank}] Step {step}: ERROR in computation phase: {e}")
                raise
        else:
            print(f"[Rank {comm.rank}] Step {step}: Skipping computation (causal={causal}, step={step}, rank={comm.rank})")

        # Synchronization phase
        if step + 1 != comm.world_size:
            sync_phase_start = time.time()
            print(f"[Rank {comm.rank}] Step {step}: Starting synchronization phase")
            
            try:
                print(f"[Rank {comm.rank}] Step {step}: Calling comm.commit()...")
                comm.commit()
                print(f"[Rank {comm.rank}] Step {step}: comm.commit() completed")
                
                print(f"[Rank {comm.rank}] Step {step}: Calling comm.wait()...")
                comm.wait()
                print(f"[Rank {comm.rank}] Step {step}: comm.wait() completed")
                
                k, v = next_k, next_v
                print(f"[Rank {comm.rank}] Step {step}: Updated k and v for next iteration")
                
                sync_phase_elapsed = time.time() - sync_phase_start
                print(f"[Rank {comm.rank}] Step {step}: Synchronization phase completed in {sync_phase_elapsed:.3f}s")
            except Exception as e:
                print(f"[Rank {comm.rank}] Step {step}: ERROR in synchronization phase: {e}")
                raise
        
        step_elapsed = time.time() - step_start
        print(f"[Rank {comm.rank}] ===== Step {step} END - elapsed: {step_elapsed:.3f}s =====")

    loop_elapsed = time.time() - loop_start
    print(f"[Rank {comm.rank}] Main ring communication loop completed in {loop_elapsed:.3f}s")
    
    # Final processing
    finalize_start = time.time()
    print(f"[Rank {comm.rank}] Finalizing output...")
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    finalize_elapsed = time.time() - finalize_start
    print(f"[Rank {comm.rank}] Output finalized in {finalize_elapsed:.3f}s")
    
    total_elapsed = time.time() - start_time
    print(f"[Rank {rank}] ========== intel_ring_flash_attn_forward END - total elapsed: {total_elapsed:.3f}s ==========\n")
    
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