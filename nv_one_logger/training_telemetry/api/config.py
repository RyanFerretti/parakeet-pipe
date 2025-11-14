from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TrainingTelemetryConfig:
    """Loose container for telemetry metadata.

    The production class exposes dozens of strongly typed fields. NeMo only
    requires that the object be instantiable with arbitrary keyword arguments
    and later stored on the provider singleton, so we simply hold the payload
    in a dict.
    """

    payload: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.payload = kwargs

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.payload)

