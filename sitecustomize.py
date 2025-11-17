"""
sitecustomize.py

Automatically imported by Python on startup if present on sys.path.

We use it to ensure torch.distributed.tensor.parallel exposes
SequenceParallel so NeMo 2.5.3 stops crashing on import. For our
single-GPU/CPU inference use case, SequenceParallel is not actually
used at runtime; we just need the symbol to exist.
"""

import torch

try:
    # If this works, great — no patch needed.
    from torch.distributed.tensor.parallel import SequenceParallel  # type: ignore
except Exception:
    # Try to get it from the style module if it exists there.
    try:
        from torch.distributed.tensor.parallel.style import (  # type: ignore
            SequenceParallel as _SequenceParallel,
        )
    except Exception:
        # Last resort: define a no-op stub. This is fine for inference.
        class _SequenceParallel:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass

    import torch.distributed.tensor.parallel as tp  # type: ignore
    tp.SequenceParallel = _SequenceParallel  # type: ignore