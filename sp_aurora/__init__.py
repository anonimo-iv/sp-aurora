import os
import site
import torch

# Intel GPU backend only
try:
    import intel_extension_for_pytorch as ipex
    
    # Set up library path for CCL backend
    for path in site.getsitepackages():
        ipex_lib = os.path.join(path, 'intel_extension_for_pytorch', 'lib')
        if os.path.exists(ipex_lib):
            ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if ipex_lib not in ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{ipex_lib}:{ld_path}"
            break
    
    # Import oneCCL bindings to register CCL backend
    try:
        import oneccl_bindings_for_pytorch
    except ImportError:
        pass  # CCL backend optional
    
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise RuntimeError("Intel GPU not available")
    
    # Set Intel-specific environment variables for optimal performance
    # These are set early to ensure they're available before any distributed operations
    
    # CCL (oneCCL) environment variables
    os.environ.setdefault('CCL_PROCESS_LAUNCHER', 'pmix')
    os.environ.setdefault('CCL_ATL_TRANSPORT', 'mpi')
    os.environ.setdefault('CCL_KVS_MODE', 'mpi')
    os.environ.setdefault('CCL_LOG_LEVEL', 'info')
    os.environ.setdefault('CCL_ZE_ENABLE', '1')
    os.environ.setdefault('CCL_KVS_USE_MPI_RANKS', '1')
    os.environ.setdefault('CCL_ATL_SYNC_COLL', '1')
    os.environ.setdefault('CCL_OP_SYNC', '1')
    
    # Network fabric and IPC settings
    os.environ.setdefault('FI_PROVIDER', 'cxi')
    os.environ.setdefault('CCL_ZE_IPC_EXCHANGE', 'drmfd')
    
    # Intel Extension for PyTorch settings
    os.environ.setdefault('IPEX_XPU_ONEDNN_LAYOUT', '1')
    os.environ.setdefault('IPEX_OFFLINE_COMPILER', '1')
    
    # SYCL settings
    os.environ.setdefault('SYCL_CACHE_PERSISTENT', '1')
    os.environ.setdefault('SYCL_DEVICE_FILTER', 'level_zero:*')
    
    # MPI settings for GPU support
    os.environ.setdefault('MPIR_CVAR_ENABLE_GPU', '1')
    
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
        update_sp_aurora_params,
    )
except ImportError:
    print("Warning: adapters not available")
    def substitute_hf_flash_attn(*args, **kwargs):
        raise NotImplementedError("substitute_hf_flash_attn not available on this backend")
    def update_sp_aurora_params(*args, **kwargs):
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

# Import yunchang-compatible features
from .globals import PROCESS_GROUP, set_seq_parallel_pg, HAS_LONG_CTX_ATTN, HAS_FLASH_ATTN, HAS_SPARSE_SAGE_ATTENTION
from .kernels import AttnType, select_flash_attn_impl, pytorch_attn_forward, pytorch_attn_func
from .hybrid import LongContextAttention
from .ring_pytorch_attn import ring_pytorch_attn_func, ring_pytorch_attn_forward
from .intel_ring_flash_attn import RingFlashAttnFunc  # Already exists as alias
from .utils import update_out_and_lse, RingComm
from .comm.all_to_all import SeqAllToAll4D
