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
        start_time = time.time()
        print(f"[IntelRingComm.__init__] Starting initialization at {start_time}")
        print(f"[IntelRingComm.__init__] Process group: {process_group}")
        print(f"[IntelRingComm.__init__] Environment variables:")
        for key in ['CCL_BACKEND', 'CCL_ATL_TRANSPORT', 'CCL_LOG_LEVEL', 'CCL_PROCESS_LAUNCHER']:
            print(f"  {key}={os.environ.get(key, 'NOT SET')}")
        
        self._process_group = process_group
        self._ops = []
        
        # Handle single process case
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized. Use intel_flash_attn_forward for single process.")
        
        print(f"[IntelRingComm.__init__] Getting rank and world size")
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None
        
        print(f"[IntelRingComm.__init__] Rank: {self.rank}, World size: {self.world_size}")
        print(f"[IntelRingComm.__init__] Backend: {dist.get_backend()}")

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)
        
        print(f"[IntelRingComm.__init__] Send rank: {self.send_rank}, Recv rank: {self.recv_rank}")

        # Initialize oneCCL backend if available
        self._init_oneccl_backend()
        elapsed = time.time() - start_time
        print(f"[IntelRingComm.__init__] Initialization complete in {elapsed:.3f}s")

    def _init_oneccl_backend(self):
        """Initialize oneCCL backend for Intel GPU communication"""
        print(f"[Rank {self.rank}] _init_oneccl_backend: ONECCL_AVAILABLE={ONECCL_AVAILABLE}")
        print(f"[Rank {self.rank}] _init_oneccl_backend: has XPU={hasattr(torch, 'xpu')}")
        if hasattr(torch, 'xpu'):
            print(f"[Rank {self.rank}] _init_oneccl_backend: XPU available={torch.xpu.is_available()}")
            print(f"[Rank {self.rank}] _init_oneccl_backend: XPU device count={torch.xpu.device_count()}")
        
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
            print(f"[Rank {self.rank}] _init_oneccl_backend: Using XPU device")
        else:
            # Fallback to default backend
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[Rank {self.rank}] _init_oneccl_backend: Using {self.device} device")

    def _ensure_xpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on XPU device for Intel GPU operations"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return tensor.to('xpu')
        return tensor

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        start_time = time.time()
        print(f"[Rank {self.rank}] send_recv: Starting with to_send.shape={to_send.shape}, dtype={to_send.dtype}, device={to_send.device}")
        
        # Ensure tensors are on the correct device
        to_send = self._ensure_xpu_tensor(to_send)
        
        if recv_tensor is None:
            res = torch.empty_like(to_send)
            print(f"[Rank {self.rank}] send_recv: Created recv tensor with shape={res.shape}")
        else:
            res = self._ensure_xpu_tensor(recv_tensor)
            print(f"[Rank {self.rank}] send_recv: Using provided recv tensor with shape={res.shape}")

        print(f"[Rank {self.rank}] send_recv: Creating P2POp for send to rank {self.send_rank}")
        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        print(f"[Rank {self.rank}] send_recv: Creating P2POp for recv from rank {self.recv_rank}")
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        
        elapsed = time.time() - start_time
        print(f"[Rank {self.rank}] send_recv: Completed in {elapsed:.3f}s, total ops={len(self._ops)}")
        return res

    def commit(self):
        start_time = time.time()
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        print(f"[Rank {self.rank}] IntelRingComm.commit() - Starting batch_isend_irecv with {len(self._ops)} ops at {start_time}")
        for i, op in enumerate(self._ops):
            op_type = "send" if hasattr(op.op, '__name__') and 'send' in op.op.__name__ else "recv"
            print(f"[Rank {self.rank}] IntelRingComm.commit() - Op {i}: {op_type} to/from rank {op.peer}")
        
        try:
            print(f"[Rank {self.rank}] IntelRingComm.commit() - Calling dist.batch_isend_irecv...")
            self._reqs = dist.batch_isend_irecv(self._ops)
            elapsed = time.time() - start_time
            print(f"[Rank {self.rank}] IntelRingComm.commit() - batch_isend_irecv returned {len(self._reqs)} requests in {elapsed:.3f}s")
        except Exception as e:
            print(f"[Rank {self.rank}] IntelRingComm.commit() - ERROR in batch_isend_irecv: {e}")
            raise

    def wait(self):
        start_time = time.time()
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        print(f"[Rank {self.rank}] IntelRingComm.wait() - Waiting for {len(self._reqs)} requests at {start_time}")
        
        try:
            for i, req in enumerate(self._reqs):
                req_start = time.time()
                print(f"[Rank {self.rank}] IntelRingComm.wait() - Waiting for request {i+1}/{len(self._reqs)}...")
                req.wait()
                req_elapsed = time.time() - req_start
                print(f"[Rank {self.rank}] IntelRingComm.wait() - Request {i+1} completed in {req_elapsed:.3f}s")
        except Exception as e:
            print(f"[Rank {self.rank}] IntelRingComm.wait() - ERROR waiting for request: {e}")
            raise
        
        self._reqs = None
        self._ops = []
        elapsed = time.time() - start_time
        print(f"[Rank {self.rank}] IntelRingComm.wait() - All requests completed in {elapsed:.3f}s")


    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_time = time.time()
        print(f"[Rank {self.rank}] send_recv_kv: Starting P2P - send to {self.send_rank}, recv from {self.recv_rank}")
        print(f"[Rank {self.rank}] send_recv_kv: k.shape={k.shape}, v.shape={v.shape}")
        print(f"[Rank {self.rank}] send_recv_kv: k.dtype={k.dtype}, v.dtype={v.dtype}")
        print(f"[Rank {self.rank}] send_recv_kv: k.device={k.device}, v.device={v.device}")
        print(f"[Rank {self.rank}] send_recv_kv: Number of ops before: {len(self._ops)}")
            
        # Use the same pattern as original RingComm
        print(f"[Rank {self.rank}] send_recv_kv: Calling send_recv for k tensor...")
        next_k = self.send_recv(k, k_buffer)
        print(f"[Rank {self.rank}] send_recv_kv: send_recv for k completed")
        
        print(f"[Rank {self.rank}] send_recv_kv: Calling send_recv for v tensor...")
        next_v = self.send_recv(v, v_buffer)
        print(f"[Rank {self.rank}] send_recv_kv: send_recv for v completed")
        
        elapsed = time.time() - start_time
        print(f"[Rank {self.rank}] send_recv_kv: Number of ops after: {len(self._ops)}")
        print(f"[Rank {self.rank}] send_recv_kv: Completed in {elapsed:.3f}s")
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