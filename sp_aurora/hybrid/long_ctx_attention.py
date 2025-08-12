"""Long context attention implementation combining Ulysses and Ring patterns.

Provides yunchang-compatible LongContextAttention class for hybrid parallelism.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union
from enum import Enum

from ..globals import PROCESS_GROUP
from ..ulysses.attn_layer import UlyssesAttention, AttnType
from ..intel_ring_flash_attn import intel_ring_flash_attn_func
from ..kernels.attention import pytorch_attn_func
from ..comm.all_to_all import SeqAllToAll4D


class LongContextAttention(nn.Module):
    """Hybrid Ulysses and Ring attention for long context.
    
    Compatible with yunchang's LongContextAttention API. Combines:
    - Ulysses (sequence parallel) attention across one dimension
    - Ring attention across another dimension
    
    This enables handling very long sequences by distributing both
    computation and memory across multiple GPUs.
    """
    
    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        sp_ulysses_degree: Optional[int] = None,
        sp_ring_degree: Optional[int] = None,
        use_ulysses: bool = True,
        use_ring: bool = True,
        ulysses_pg: Optional[dist.ProcessGroup] = None,
        ring_pg: Optional[dist.ProcessGroup] = None,
        attn_type: Union[str, AttnType] = AttnType.FA,
        use_pack_qkv: bool = False,
        use_sync: bool = False,
    ):
        """Initialize LongContextAttention.
        
        Args:
            scatter_idx: Dimension index for scattering in all-to-all
            gather_idx: Dimension index for gathering in all-to-all
            sp_ulysses_degree: Ulysses parallelism degree (auto-detect if None)
            sp_ring_degree: Ring parallelism degree (auto-detect if None)
            use_ulysses: Whether to use Ulysses parallelism
            use_ring: Whether to use Ring parallelism
            ulysses_pg: Ulysses process group (use global if None)
            ring_pg: Ring process group (use global if None)
            attn_type: Attention implementation type
            use_pack_qkv: Whether to pack q,k,v tensors for optimized all-to-all
            use_sync: Whether to synchronize after all-to-all operations
        """
        super().__init__()
        
        # Convert string to AttnType enum if needed
        if isinstance(attn_type, str):
            attn_type = AttnType(attn_type)
        self.attn_type = attn_type
        
        # Get process groups from globals if not provided
        if ulysses_pg is None and PROCESS_GROUP.initialized:
            ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        if ring_pg is None and PROCESS_GROUP.initialized:
            ring_pg = PROCESS_GROUP.RING_PG
            
        # Get degrees from globals if not provided
        if sp_ulysses_degree is None and PROCESS_GROUP.initialized:
            sp_ulysses_degree = PROCESS_GROUP.ulysses_degree
        if sp_ring_degree is None and PROCESS_GROUP.initialized:
            sp_ring_degree = PROCESS_GROUP.ring_degree
            
        self.sp_ulysses_degree = sp_ulysses_degree or 1
        self.sp_ring_degree = sp_ring_degree or 1
        self.use_ulysses = use_ulysses and self.sp_ulysses_degree > 1
        self.use_ring = use_ring and self.sp_ring_degree > 1
        
        self.ulysses_pg = ulysses_pg
        self.ring_pg = ring_pg
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_pack_qkv = use_pack_qkv
        self.use_sync = use_sync
        
        # Initialize Ulysses attention if needed
        if self.use_ulysses:
            self.ulysses_attn = UlyssesAttention(
                sequence_process_group=ulysses_pg,
                scatter_idx=scatter_idx,
                gather_idx=gather_idx,
                attn_type=attn_type
            )
        else:
            self.ulysses_attn = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        joint_tensor_key_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for hybrid attention.
        
        Args:
            query: Query tensor (batch, seq_len, num_heads, head_dim)
            key: Key tensor (batch, seq_len, num_heads, head_dim)
            value: Value tensor (batch, seq_len, num_heads, head_dim)
            dropout_p: Dropout probability
            softmax_scale: Attention scaling factor
            causal: Whether to use causal masking
            window_size: Local attention window size
            softcap: Soft capping value
            alibi_slopes: ALiBi slopes
            deterministic: Whether to use deterministic dropout
            return_attn_probs: Whether to return attention probabilities
            joint_tensor_key_value: Optional joint KV tensor
            
        Returns:
            Output tensor with same shape as query
        """
        # Handle joint KV tensor if provided
        if joint_tensor_key_value is not None:
            # Assume joint tensor has shape (batch, seq_len, 2, num_heads, head_dim)
            key = joint_tensor_key_value[:, :, 0]
            value = joint_tensor_key_value[:, :, 1]
        
        # Hybrid Ulysses + Ring approach (following yunchang pattern)
        if self.use_ulysses and self.ulysses_pg is not None and dist.get_world_size(self.ulysses_pg) > 1:
            # Apply Ulysses all-to-all communication
            if self.use_pack_qkv:
                # Pack q, k, v for optimized communication
                qkv = torch.cat([query, key, value]).contiguous()
                # Apply all-to-all on packed tensor
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx, self.use_sync
                )
                # Unpack
                qkv_chunks = torch.chunk(qkv, 3, dim=0)
                query_layer, key_layer, value_layer = qkv_chunks[0], qkv_chunks[1], qkv_chunks[2]
            else:
                # Apply all-to-all separately on q, k, v
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx, self.use_sync
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx, self.use_sync
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx, self.use_sync
                )
            
            # Apply ring attention on redistributed tensors
            if self.use_ring and self.ring_pg is not None:
                output = intel_ring_flash_attn_func(
                    query_layer, key_layer, value_layer,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    group=self.ring_pg
                )
            else:
                # Local attention on Ulysses-distributed tensors
                output = pytorch_attn_func(
                    query_layer, key_layer, value_layer,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs
                )
            
            if return_attn_probs and isinstance(output, tuple):
                output = output[0]
            
            # Apply reverse all-to-all to gather results
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, output, self.gather_idx, self.scatter_idx, self.use_sync
            )
        
        # Apply Ring attention only (no Ulysses)
        elif self.use_ring and self.ring_pg is not None:
            output = intel_ring_flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg
            )
            if return_attn_probs and isinstance(output, tuple):
                output = output[0]
        
        # Apply Ulysses attention only (using existing implementation)
        elif self.use_ulysses and self.ulysses_attn is not None:
            output = self.ulysses_attn(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs
            )
            if return_attn_probs and isinstance(output, tuple):
                output = output[0]
        
        # Fallback to local attention
        else:
            output = pytorch_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs
            )
            if return_attn_probs and isinstance(output, tuple):
                output = output[0]
        
        return output
    
    def get_sequence_parallel_degree(self) -> int:
        """Get the total sequence parallelism degree."""
        return self.sp_ulysses_degree * self.sp_ring_degree
    
    def get_ulysses_degree(self) -> int:
        """Get Ulysses parallelism degree."""
        return self.sp_ulysses_degree
    
    def get_ring_degree(self) -> int:
        """Get Ring parallelism degree."""
        return self.sp_ring_degree