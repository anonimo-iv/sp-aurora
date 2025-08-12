"""
Intel GPU compatible utilities with oneCCL communication backend
"""

from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
from functools import cache
import time
import os

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

    # Ensure block_out matches the dimension ordering of out
    # Handle different tensor layouts more robustly
    if out.ndim == 4 and block_out.ndim == 4:
        # Check if dimensions are swapped between seq_len and num_heads
        if (out.shape[1] != block_out.shape[1] and 
            out.shape[2] != block_out.shape[2] and
            out.shape[1] == block_out.shape[2] and 
            out.shape[2] == block_out.shape[1]):
            # Transpose dimensions 1 and 2 to match
            block_out = block_out.transpose(1, 2)
    
    block_out = block_out.to(torch.float32)
    
    # Final shape check for out and block_out
    if out.shape != block_out.shape:
        raise RuntimeError(f"Shape mismatch after block_out transformation: out.shape={out.shape}, block_out.shape={block_out.shape}")
    
    # Handle block_lse shape transformation to match lse format
    # First, ensure block_lse has 4 dimensions
    if block_lse.dim() == 3:
        # If [batch, seq_len, num_heads] -> add dimension to get [batch, seq_len, num_heads, 1]
        block_lse = block_lse.unsqueeze(dim=-1)
    
    # Now both tensors should be 4D. Check if dimensions need to be transposed to match lse
    if lse.shape != block_lse.shape:
        # Check if block_lse needs to be transposed to match lse's dimension ordering
        # lse typically has shape [batch, num_heads, seq_len, 1] 
        # block_lse might have shape [batch, seq_len, num_heads, 1]
        if (lse.shape[1] == block_lse.shape[2] and 
            lse.shape[2] == block_lse.shape[1] and
            lse.shape[0] == block_lse.shape[0] and
            lse.shape[3] == block_lse.shape[3]):
            # Transpose dimensions 1 and 2 to match lse's ordering
            block_lse = block_lse.transpose(1, 2)
        
        # If still not matching, try other reshape strategies
        elif block_lse.numel() == lse.numel():
            # Same number of elements, try direct reshape
            block_lse = block_lse.view_as(lse)
        
        # Final shape check
        if lse.shape != block_lse.shape:
            raise RuntimeError(f"Shape mismatch after transformation: lse.shape={lse.shape}, block_lse.shape={block_lse.shape}")

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
        # Transpose block_out to match the dimension ordering we'll use for lse
        # block_out: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        out = block_out.transpose(1, 2).to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        # Ensure block_out has the same dimension ordering as out
        # out is [batch, num_heads, seq_len, head_dim] after first iteration
        # block_out might be [batch, seq_len, num_heads, head_dim]
        if block_out.shape[1] != out.shape[1] and block_out.shape[2] == out.shape[1]:
            # Need to transpose block_out to match out's dimension ordering
            block_out = block_out.transpose(1, 2)
        
        # Similarly handle block_lse
        if block_lse.dim() == 3 and lse.dim() == 4:
            # block_lse might be [batch, seq_len, num_heads]
            # lse is [batch, num_heads, seq_len, 1]
            if block_lse.shape[1] == lse.shape[2] and block_lse.shape[2] == lse.shape[1]:
                block_lse = block_lse.transpose(1, 2).unsqueeze(-1)
            elif block_lse.shape[-1] != 1:
                block_lse = block_lse.unsqueeze(-1)
        
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
    Intel GPU compatible ring communication using oneCCL backend
    """
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._reqs = []  # Store individual requests instead of ops
        
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
            # Set XPU device for communication
            if not dist.is_initialized():
                # Use MPI-compatible initialization
                try:
                    from .mpi_utils import init_distributed_backend
                    init_distributed_backend(backend='ccl')
                except ImportError:
                    # Fallback to direct initialization
                    dist.init_process_group(backend='ccl')
            self.device = 'xpu'
        else:
            # Fallback to default backend
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _ensure_xpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on XPU device for Intel GPU operations"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return tensor.to('xpu')
        return tensor

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Ensure tensors are on the correct device and contiguous
        to_send = self._ensure_xpu_tensor(to_send).contiguous()
        
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = self._ensure_xpu_tensor(recv_tensor)

        # Fix for deadlock: Order operations based on rank parity to break circular dependency
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            send_req = dist.isend(to_send, self.send_rank, group=self._process_group)
            recv_req = dist.irecv(res, self.recv_rank, group=self._process_group)
        else:
            # Odd ranks: receive first, then send
            recv_req = dist.irecv(res, self.recv_rank, group=self._process_group)
            send_req = dist.isend(to_send, self.send_rank, group=self._process_group)
        
        # Store requests for later wait
        self._reqs.append(send_req)
        self._reqs.append(recv_req)
        
        return res

    def commit(self):
        # No-op since operations start immediately when created
        pass

    def wait(self):
        if not self._reqs:
            return
            
        try:
            for req in self._reqs:
                req.wait()
        except Exception as e:
            raise
        
        self._reqs = []


    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simply call send_recv for each tensor - operations start immediately
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        return next_k, next_v

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


# Backward compatibility aliases
RingComm = IntelRingComm
AllGatherComm = IntelAllGatherComm