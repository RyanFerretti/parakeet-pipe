"""
sitecustomize.py
----------------

Automatically imported by Python on startup if present on sys.path.

NeMo 2.5.3 expects torch.distributed.tensor.parallel.SequenceParallel to
exist, but certain PyTorch builds (including Deepnote's default runtime)
ship it under torch.distributed.tensor.parallel.style or omit it entirely.
For our single-GPU / CPU inference workflow we don't actually use
SequenceParallel, so it's enough to ensure the attribute exists.
"""


def _ensure_sequence_parallel_symbol() -> None:
    try:
        from torch.distributed.tensor.parallel import SequenceParallel  # type: ignore  # noqa: F401
        return  # symbol is already available
    except Exception:
        pass

    try:
        from torch.distributed.tensor.parallel.style import (  # type: ignore
            SequenceParallel as _SequenceParallel,
        )
    except Exception:
        # Fallback stub that satisfies NeMo's import without doing anything.
        class _SequenceParallel:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass

    import torch.distributed.tensor.parallel as tp  # type: ignore

    tp.SequenceParallel = _SequenceParallel  # type: ignore


try:
    # Only run when torch is available. If torch isn't installed yet,
    # importing it here would raise and unnecessarily break startup.
    import torch  # noqa: F401
except Exception:
    pass
else:
    _ensure_sequence_parallel_symbol()

