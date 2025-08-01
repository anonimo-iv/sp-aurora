import torch

# Intel GPU backend only
try:
    import intel_extension_for_pytorch as ipex
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise RuntimeError("Intel GPU not available")
except ImportError:
    raise ImportError("Intel Extension for PyTorch not installed")

# Import Intel GPU implementations
from .intel_ring_flash_attn import (
    intel_ring_flash_attn_func as ring_flash_attn_func,
    intel_ring_flash_attn_kvpacked_func as ring_flash_attn_kvpacked_func,
    intel_ring_flash_attn_qkvpacked_func as ring_flash_attn_qkvpacked_func,
)

# Import Intel Ulysses (Sequence Parallel) implementations
from .intel_ulysses_attn import (
    intel_ulysses_flash_attn_func as ulysses_flash_attn_func,
    intel_ulysses_flash_attn_kvpacked_func as ulysses_flash_attn_kvpacked_func,
    intel_ulysses_flash_attn_qkvpacked_func as ulysses_flash_attn_qkvpacked_func,
    IntelUlyssesComm as UlyssesComm,
    IntelSeqAllToAll4D as SeqAllToAll4D,
)

# Import new class-based Ulysses attention
from .ulysses.attn_layer import UlyssesAttention
from .comm.all_to_all import SeqAllToAll4D as SeqAllToAll4D_v2, SeqAllToAll5D

# Defer print to avoid MPI initialization issues
# print("Using Intel GPU backend for Ring Flash Attention and Ulysses Attention")

try:
    from .adapters import (
        substitute_hf_flash_attn,
        update_ring_flash_attn_params,
    )
except ImportError:
    print("Warning: adapters not available")
    def substitute_hf_flash_attn(*args, **kwargs):
        raise NotImplementedError("substitute_hf_flash_attn not available on this backend")
    def update_ring_flash_attn_params(*args, **kwargs):
        raise NotImplementedError("update_ring_flash_attn_params not available on this backend")

# MPI utilities for enhanced distributed compatibility
try:
    from .mpi_utils import (
        setup_mpi_distributed,
        init_distributed_backend,
        cleanup_distributed,
        get_device_for_rank,
        detect_mpi_environment,
    )
except ImportError:
    print("Warning: MPI utilities not available")
    def setup_mpi_distributed(*args, **kwargs):
        raise NotImplementedError("MPI utilities not available")
    def init_distributed_backend(*args, **kwargs):
        raise NotImplementedError("MPI utilities not available")
    def cleanup_distributed(*args, **kwargs):
        pass
    def get_device_for_rank(*args, **kwargs):
        return torch.device('cpu')
    def detect_mpi_environment(*args, **kwargs):
        return False
