from __future__ import annotations

from typing import Any

try:
    from lightning.pytorch.callbacks import Callback
except Exception:  # pragma: no cover - lightning might not be installed
    class Callback:  # type: ignore
        """Fallback so the stub can be imported without Lightning."""

        pass


class TimeEventCallback(Callback):
    """No-op Lightning callback used solely to satisfy NeMo imports."""

    def __init__(self, provider, call_on_app_start: bool = True) -> None:
        super().__init__()
        self.provider = provider
        self.call_on_app_start = call_on_app_start

    # Lightning will happily ignore callbacks that don't implement hooks, so we
    # intentionally leave the rest of the interface empty.
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - defensive
        # Return a harmless lambda for any hook that Lightning might probe.
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop

