"""
Intel GPU compatible utilities with oneCCL communication backend
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
    Intel GPU compatible ring communication using oneCCL backend
    """
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        
        # Handle single process case
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized. Use intel_flash_attn_forward for single process.")
        
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

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
        # Ensure tensors are on the correct device
        to_send = self._ensure_xpu_tensor(to_send)
        
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = self._ensure_xpu_tensor(recv_tensor)

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure tensors are on Intel GPU
        k = self._ensure_xpu_tensor(k)
        v = self._ensure_xpu_tensor(v)
        
        # Use direct P2P operations to avoid deadlock
        # The original batch_isend_irecv approach has timing issues on Intel GPU
        if k_buffer is None:
            next_k = torch.empty_like(k)
        else:
            next_k = self._ensure_xpu_tensor(k_buffer)
            
        if v_buffer is None:
            next_v = torch.empty_like(v)
        else:
            next_v = self._ensure_xpu_tensor(v_buffer)
        
        # Perform P2P operations directly with proper synchronization
        # Debug: Print communication info for first iteration only
        if not hasattr(self, '_debug_printed'):
            print(f"[Rank {self.rank}] Starting P2P: send to {self.send_rank}, recv from {self.recv_rank}")
            self._debug_printed = True
            
        send_req_k = dist.isend(k, self.send_rank, group=self._process_group)
        recv_req_k = dist.irecv(next_k, self.recv_rank, group=self._process_group)
        send_req_v = dist.isend(v, self.send_rank, group=self._process_group)
        recv_req_v = dist.irecv(next_v, self.recv_rank, group=self._process_group)
        
        # Wait for all operations to complete before returning
        send_req_k.wait()
        recv_req_k.wait() 
        send_req_v.wait()
        recv_req_v.wait()
        
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