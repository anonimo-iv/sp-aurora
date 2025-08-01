# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# Adapted for Intel GPU support

import torch
from typing import Any, Optional, Tuple
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from ..comm.all_to_all import SeqAllToAll4D
from enum import Enum

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


class AttnType(Enum):
    """Attention implementation types"""
    TORCH = "torch"
    INTEL_SYCL = "intel_sycl"
    INTEL_ONEDNN = "intel_onednn"


def select_flash_attn_impl(attn_type: AttnType):
    """Select the appropriate attention implementation based on the type"""
    
    if attn_type == AttnType.INTEL_SYCL:
        try:
            from ring_flash_attn.intel_flash_attn_sycl import (
                intel_flash_attention_forward,
                intel_flash_attention_backward
            )
            
            def attn_fn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, 
                       window_size=(-1, -1), softcap=0.0, alibi_slopes=None, 
                       deterministic=False, return_attn_probs=False):
                if softmax_scale is None:
                    softmax_scale = q.shape[-1] ** -0.5
                
                out, lse = intel_flash_attention_forward(
                    q, k, v,
                    softmax_scale=softmax_scale,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic
                )
                
                if return_attn_probs:
                    return out, lse
                return out
            
            return attn_fn
        except ImportError:
            pass
    
    # Default to PyTorch SDPA
    def pytorch_attn_fn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                       window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                       deterministic=False, return_attn_probs=False):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
            
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale
        )
        
        if return_attn_probs:
            # Calculate LSE for compatibility
            batch_size, seq_len, num_heads, head_dim = q.shape
            scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
            if causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            return out, lse
            
        return out
    
    return pytorch_attn_fn


class UlyssesAttention(torch.nn.Module):
    """
    Ulysses Attention module for sequence parallel attention computation.
    
    This module redistributes sequences across GPUs using all-to-all communication,
    computes local attention, then redistributes back.
    
    Arguments:
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm (default: 2)
        gather_idx (int): gather_idx for all2all comm (default: 1)
        use_sync (bool): whether to synchronize after all-to-all
        attn_type (AttnType): attention implementation type
    """
    
    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.TORCH,
    ) -> None:
        super(UlyssesAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.attn_type = attn_type
        
        # Auto-detect device and select appropriate implementation
        if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device("xpu")
            gpu_name = torch.xpu.get_device_properties(device).name
            
            # Try to use Intel SYCL implementation if available
            if attn_type == AttnType.TORCH:
                try:
                    from ring_flash_attn.intel_flash_attn_sycl import intel_flash_attention_forward
                    self.attn_type = AttnType.INTEL_SYCL
                except ImportError:
                    pass
        
        self.attn_fn = select_flash_attn_impl(self.attn_type)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        alibi_slopes: Optional[Tensor] = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        *args: Any
    ) -> Tensor:
        """
        Forward pass of Ulysses attention.
        
        Arguments:
            query (Tensor): query input to the layer (bs, seq_len/P, num_heads, head_dim)
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            dropout_p: dropout probability
            softmax_scale: scale factor for softmax
            causal: whether to apply causal masking
            window_size: local attention window size
            softcap: softmax capping value
            alibi_slopes: ALiBi slopes for position bias
            deterministic: whether to use deterministic algorithms
            return_attn_probs: whether to return attention probabilities
            
        Returns:
            output (Tensor): context output (bs, seq_len/P, num_heads, head_dim)
        """
        # Handle single GPU case
        if not dist.is_initialized() or dist.get_world_size(self.spg) == 1:
            return self.attn_fn(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
        
        # All-to-all to redistribute tensors
        # (bs, seq_len/P, num_heads, head_dim) -> (bs, seq_len, num_heads/P, head_dim)
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync)
        
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        
        # Compute local attention
        context_layer = self.attn_fn(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
        
        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]
        
        # All-to-all to redistribute back
        # (bs, seq_len, num_heads/P, head_dim) -> (bs, seq_len/P, num_heads, head_dim)
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync
        )
        
        return output