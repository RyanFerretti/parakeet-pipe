from __future__ import annotations

from types import SimpleNamespace

from .config import TrainingTelemetryConfig


class TrainingTelemetryProvider:
    """Bare-bones singleton that mirrors the public API NeMo expects."""

    _instance: TrainingTelemetryProvider | None = None

    def __init__(self) -> None:
        self._base_config = None
        self._export_config = None
        self.config = SimpleNamespace(telemetry_config=None)

    @classmethod
    def instance(cls) -> "TrainingTelemetryProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def with_base_config(self, config) -> "TrainingTelemetryProvider":
        self._base_config = config
        return self

    def with_export_config(self, config: Optional[dict] = None) -> "TrainingTelemetryProvider":
        self._export_config = config or {}
        return self

    def configure_provider(self) -> "TrainingTelemetryProvider":
        # Nothing to configure in the stub, but keep the fluent API.
        return self

    def set_training_telemetry_config(self, config: TrainingTelemetryConfig) -> None:
        self.config.telemetry_config = config

    def get_training_telemetry_config(self) -> TrainingTelemetryConfig | None:
        return self.config.telemetry_config

