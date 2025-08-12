# Communication Module

This module contains communication utilities for distributed operations.

## Files

- `all_to_all.py` - General all-to-all implementation with both 4D and 5D tensor support
  - Used by `ulysses/attn_layer.py` for the class-based Ulysses implementation
  - Provides `SeqAllToAll4D` autograd function

## Note

The main Ulysses implementation in `intel_ulysses_attn.py` has its own `IntelSeqAllToAll4D` and `intel_all_to_all_4d` functions that are specifically optimized for Intel GPUs. The implementations here are kept for compatibility with the class-based Ulysses attention layer.