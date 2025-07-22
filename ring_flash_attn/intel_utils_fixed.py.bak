"""
Intel GPU compatible utilities with synchronous oneCCL communication
Fixed version that prevents P2P deadlocks
"""

from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
from functools import cache

# Import oneCCL bindings
try:
    import oneccl_bindings_for_pytorch
    ONECCL_AVAILABLE = True
except ImportError:
    ONECCL_AVAILABLE = False
    print("Warning: oneCCL bindings not available, falling back to default backend")

import intel_extension_for_pytorch as ipex


__all__ = ["update_out_and_lse", "IntelRingComm", "get_default_args"]


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class IntelRingComm:
    """
    Intel GPU compatible ring communication using synchronous oneCCL backend
    Fixed version with immediate P2P execution to prevent deadlocks
    """
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        
        # Handle single process case
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized. Use intel_flash_attn_forward for single process.")
        
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

        # Initialize oneCCL backend if available
        self._init_oneccl_backend()

    def _init_oneccl_backend(self):
        """Initialize oneCCL backend for Intel GPU communication"""
        if ONECCL_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = 'xpu'
        else:
            # Fallback to default backend
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _ensure_xpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on XPU device for Intel GPU operations"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return tensor.to('xpu')
        return tensor

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synchronous send/receive for k and v tensors to prevent deadlock.
        Uses immediate execution instead of batched operations.
        """
        # Debug: Print communication info for first iteration only
        if not hasattr(self, '_debug_printed'):
            print(f"[Rank {self.rank}] Starting synchronous P2P: send to {self.send_rank}, recv from {self.recv_rank}")
            self._debug_printed = True
            
        # Ensure tensors are on the correct device
        k = self._ensure_xpu_tensor(k)
        v = self._ensure_xpu_tensor(v)
        print(f"[Rank {self.rank}] k: {k.shape}, v: {v.shape}")
        
        # Create receive buffers
        if k_buffer is None:
            next_k = torch.empty_like(k)
        else:
            next_k = self._ensure_xpu_tensor(k_buffer)
            
        if v_buffer is None:
            next_v = torch.empty_like(v)
        else:
            next_v = self._ensure_xpu_tensor(v_buffer)
        
        # Execute P2P operations immediately using batch_isend_irecv
        # This ensures both ranks execute their operations simultaneously
        ops = []
        ops.append(dist.P2POp(dist.isend, k, self.send_rank, group=self._process_group))
        ops.append(dist.P2POp(dist.irecv, next_k, self.recv_rank, group=self._process_group))
        ops.append(dist.P2POp(dist.isend, v, self.send_rank, group=self._process_group))
        ops.append(dist.P2POp(dist.irecv, next_v, self.recv_rank, group=self._process_group))
        print(f"[Rank {self.rank}] ops: {ops}")
        
        # Execute all operations at once and wait for completion
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
            
        if not hasattr(self, '_debug_printed_complete'):
            print(f"[Rank {self.rank}] P2P communication completed successfully")
            self._debug_printed_complete = True
            
        return next_k, next_v

    # Deprecated methods kept for compatibility but not used
    def commit(self):
        """No-op for compatibility - operations are now synchronous"""
        pass

    def wait(self):
        """No-op for compatibility - operations are now synchronous"""
        pass


class IntelAllGatherComm:
    """Intel GPU compatible all-gather communication"""
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def _ensure_xpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on XPU device for Intel GPU operations"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return tensor.to('xpu')
        return tensor

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        output_tensor = self._ensure_xpu_tensor(output_tensor)
        input_tensor = self._ensure_xpu_tensor(input_tensor)
        
        handle = dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.group, async_op=True
        )
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []


class IntelRingCommFixed:
    """
    Fixed Intel GPU ring communication with multiple deadlock prevention strategies
    """
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized.")
        
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

        # Debug info
        self._debug_printed = False
        self._comm_count = 0

    def _ensure_xpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on XPU device"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return tensor.to('xpu')
        return tensor

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fixed send/receive with multiple strategies to prevent deadlock
        """
        # Debug print only once
        if not self._debug_printed:
            print(f"[Rank {self.rank}] IntelRingCommFixed: send to {self.send_rank}, recv from {self.recv_rank}")
            self._debug_printed = True
            
        # Ensure tensors are on the correct device
        k = self._ensure_xpu_tensor(k)
        v = self._ensure_xpu_tensor(v)
        
        # Create receive buffers
        if k_buffer is None:
            next_k = torch.empty_like(k)
        else:
            next_k = self._ensure_xpu_tensor(k_buffer)
            
        if v_buffer is None:
            next_v = torch.empty_like(v)
        else:
            next_v = self._ensure_xpu_tensor(v_buffer)
        
        # Strategy selection based on world size and iteration
        if self.world_size == 2:
            # For 2 ranks, use special handling to avoid deadlock
            return self._send_recv_kv_two_ranks(k, v, next_k, next_v)
        else:
            # For >2 ranks, use the standard batch approach
            return self._send_recv_kv_batch(k, v, next_k, next_v)
    
    def _send_recv_kv_two_ranks(self, k, v, next_k, next_v):
        """
        Special handling for 2-rank case using alternating send/recv pattern
        """
        # Method 1: Even/odd pattern to break symmetry
        if self.rank == 0:
            # Rank 0: send k first, then receive k
            req_send_k = dist.isend(k, dst=self.send_rank, group=self._process_group)
            req_recv_k = dist.irecv(next_k, src=self.recv_rank, group=self._process_group)
            req_send_k.wait()
            req_recv_k.wait()
            
            # Then handle v
            req_send_v = dist.isend(v, dst=self.send_rank, group=self._process_group)
            req_recv_v = dist.irecv(next_v, src=self.recv_rank, group=self._process_group)
            req_send_v.wait()
            req_recv_v.wait()
        else:
            # Rank 1: receive k first, then send k
            req_recv_k = dist.irecv(next_k, src=self.recv_rank, group=self._process_group)
            req_send_k = dist.isend(k, dst=self.send_rank, group=self._process_group)
            req_recv_k.wait()
            req_send_k.wait()
            
            # Then handle v
            req_recv_v = dist.irecv(next_v, src=self.recv_rank, group=self._process_group)
            req_send_v = dist.isend(v, dst=self.send_rank, group=self._process_group)
            req_recv_v.wait()
            req_send_v.wait()
            
        return next_k, next_v
    
    def _send_recv_kv_batch(self, k, v, next_k, next_v):
        """
        Standard batch approach for >2 ranks
        """
        # Use batch_isend_irecv but with careful ordering
        ops = []
        
        # Order operations to minimize deadlock risk:
        # First add all sends, then all receives
        ops.append(dist.P2POp(dist.isend, k, self.send_rank, group=self._process_group))
        ops.append(dist.P2POp(dist.isend, v, self.send_rank, group=self._process_group))
        ops.append(dist.P2POp(dist.irecv, next_k, self.recv_rank, group=self._process_group))
        ops.append(dist.P2POp(dist.irecv, next_v, self.recv_rank, group=self._process_group))
        
        # Execute all operations
        reqs = dist.batch_isend_irecv(ops)
        
        # Wait for all to complete
        for req in reqs:
            req.wait()
            
        return next_k, next_v
    
    def send_recv_kv_allreduce(self, k, v, k_buffer=None, v_buffer=None):
        """
        Alternative implementation using all-reduce for debugging
        This is less efficient but guaranteed deadlock-free
        """
        # Stack all ranks' k and v tensors
        batch_size = k.shape[0]
        k_flat = k.flatten()
        v_flat = v.flatten()
        
        # Create a tensor to hold all ranks' data
        all_k = torch.zeros(self.world_size * k_flat.numel(), device=k.device, dtype=k.dtype)
        all_v = torch.zeros(self.world_size * v_flat.numel(), device=v.device, dtype=v.dtype)
        
        # Place own data in correct position
        start_idx = self.rank * k_flat.numel()
        end_idx = (self.rank + 1) * k_flat.numel()
        all_k[start_idx:end_idx] = k_flat
        all_v[start_idx:end_idx] = v_flat
        
        # All-reduce to share data
        dist.all_reduce(all_k, group=self._process_group)
        dist.all_reduce(all_v, group=self._process_group)
        
        # Extract data from previous rank
        prev_start = self.recv_rank * k_flat.numel()
        prev_end = (self.recv_rank + 1) * k_flat.numel()
        
        next_k = all_k[prev_start:prev_end].reshape_as(k)
        next_v = all_v[prev_start:prev_end].reshape_as(v)
        
        return next_k, next_v

    # Compatibility methods
    def commit(self):
        pass

    def wait(self):
        pass


# Backward compatibility aliases
RingComm = IntelRingComm
AllGatherComm = IntelAllGatherComm