"""Kernel implementations for attention mechanisms.

This module provides yunchang-compatible kernel selection and implementations.
"""

from ..ulysses.attn_layer import AttnType, select_flash_attn_impl
from .attention import pytorch_attn_forward, pytorch_attn_func

__all__ = [
    "AttnType",
    "select_flash_attn_impl", 
    "pytorch_attn_forward",
    "pytorch_attn_func",
]