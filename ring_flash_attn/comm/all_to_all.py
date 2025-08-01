# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# Adapted for Intel GPU support

import torch
from typing import Any, Tuple
from torch import Tensor
import torch.distributed as dist

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


def all_to_all_4D(
    input: torch.Tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.Tensor:
    """
    All-to-all communication for 4D tensors with Intel GPU support.
    
    Args:
        input: Input tensor with shape (bs, seq_len, num_heads, head_dim)
        scatter_idx: Dimension to scatter (split) across processes
        gather_idx: Dimension to gather (concatenate) across processes
        group: Process group for communication
        use_sync: Whether to synchronize after all-to-all
        
    Returns:
        Redistributed tensor
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"
    
    # Handle single process case
    if not dist.is_initialized() or (group is None and dist.get_world_size() == 1):
        return input
    
    seq_world_size = dist.get_world_size(group)
    if seq_world_size == 1:
        return input
    
    # Ensure tensor is on appropriate device (Intel GPU support)
    if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        if input.device.type != 'xpu':
            input = input.to('xpu')
    
    if scatter_idx == 2 and gather_idx == 1:
        # Forward: (bs, seq_len/P, num_heads, head_dim) -> (bs, seq_len, num_heads/P, head_dim)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size
        
        # Reshape to prepare for all-to-all
        # (bs, seq_len/P, hc, hs) -> (bs, seq_len/P, P, hc/P, hs) -> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )
        
        output = torch.empty_like(input_t)
        # Perform all-to-all
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                if hasattr(torch, 'xpu') and input.device.type == 'xpu':
                    torch.xpu.synchronize()
                elif input.device.type == 'cuda':
                    torch.cuda.synchronize()
        else:
            output = input_t
        
        # Reshape back to desired output shape
        output = output.reshape(seqlen, bs, shard_hc, hs)
        # (seq_len, bs, hc/P, hs) -> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        
        return output
        
    elif scatter_idx == 1 and gather_idx == 2:
        # Backward: (bs, seq_len, num_heads/P, head_dim) -> (bs, seq_len/P, num_heads, head_dim)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        
        # Reshape for all-to-all
        # (bs, seqlen, hc/P, hs) -> (bs, P, seq_len/P, hc/P, hs) -> (hc/P, P, seq_len/P, bs, hs) -> (P, hc/P, seq_len/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )
        
        output = torch.empty_like(input_t)
        # Perform all-to-all
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                if hasattr(torch, 'xpu') and input.device.type == 'xpu':
                    torch.xpu.synchronize()
                elif input.device.type == 'cuda':
                    torch.cuda.synchronize()
        else:
            output = input_t
        
        # Reshape back
        output = output.reshape(hc, shard_seqlen, bs, hs)
        # (hc, seq_len/N, bs, hs) -> (bs, seq_len/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
        
        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )


def all_to_all_5D(
    input: torch.Tensor, scatter_idx: int = 3, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.Tensor:
    """
    All-to-all for QKV packed tensors (5D)
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)
    
    Args:
        input: Input tensor with shape (bs, seq_len/N, 3, num_heads, head_dim)
        scatter_idx: Dimension to scatter (default 3 for heads)
        gather_idx: Dimension to gather (default 1 for sequence)
        group: Process group for communication
        use_sync: Whether to synchronize after all-to-all
        
    Returns:
        Redistributed tensor
    """
    assert input.dim() == 5, f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"
    
    # Handle single process case
    if not dist.is_initialized() or (group is None and dist.get_world_size() == 1):
        return input
        
    seq_world_size = dist.get_world_size(group)
    if seq_world_size == 1:
        return input
    
    # Ensure tensor is on appropriate device (Intel GPU support)
    if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        if input.device.type != 'xpu':
            input = input.to('xpu')
    
    if scatter_idx == 3 and gather_idx == 1:
        # Forward: (bs, seqlen/P, 3, hc, hs) -> (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape
        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size
        
        # Reshape for all-to-all
        # (bs, seqlen/P, 3, hc, hs) -> (bs, seq_len/P, 3, P, hc/P, hs) -> (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs)
            .transpose(0, 3)
            .contiguous()
        )
        
        output = torch.empty_like(input_t)
        # Perform all-to-all
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                if hasattr(torch, 'xpu') and input.device.type == 'xpu':
                    torch.xpu.synchronize()
                elif input.device.type == 'cuda':
                    torch.cuda.synchronize()
        else:
            output = input_t
        
        # Reshape back
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)
        # (seq_len, 3, bs, hc/P, hs) -> (bs, seq_len, 3, hc/P, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()
        
        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
        
    elif scatter_idx == 1 and gather_idx == 3:
        # Backward: (bs, seqlen, 3, hc/P, hs) -> (bs, seqlen/P, 3, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        
        # Reshape for all-to-all
        # (bs, seqlen, 3, hc/P, hs) -> (bs, P, seq_len/P, 3, hc/P, hs) -> (hc/P, P, seq_len/P, 3, bs, hs) -> (P, hc/P, seq_len/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )
        
        output = torch.empty_like(input_t)
        # Perform all-to-all
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                if hasattr(torch, 'xpu') and input.device.type == 'xpu':
                    torch.xpu.synchronize()
                elif input.device.type == 'cuda':
                    torch.cuda.synchronize()
        else:
            output = input_t
        
        # Reshape back
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)
        # (hc, seq_len/N, 3, bs, hs) -> (bs, seq_len/N, 3, hc, hs)
        output = output.transpose(0, 3).contiguous()
        
        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        
        return all_to_all_5D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None, None]:
        return (
            None,
            SeqAllToAll5D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )