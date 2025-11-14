from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


Scalar = Union[int, float, str, bool]


@dataclass
class OneLoggerConfig:
    """Minimal config container used by NeMo.

    The real nv-one-logger library exposes a richer dataclass. We only need the
    handful of attributes NeMo touches while importing, so this simplified
    version stores arbitrary keyword arguments for forward compatibility.
    """

    application_name: str
    session_tag_or_fn: Union[str, Callable[[], str]]
    enable_for_current_rank: bool = True
    world_size_or_fn: Union[int, Callable[[], int]] = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.application_name = kwargs.pop("application_name", "nemo")
        self.session_tag_or_fn = kwargs.pop("session_tag_or_fn", "nemo-run")
        self.enable_for_current_rank = kwargs.pop("enable_for_current_rank", True)
        self.world_size_or_fn = kwargs.pop("world_size_or_fn", 1)
        self.extra = kwargs

    def as_dict(self) -> dict[str, Any]:
        return {
            "application_name": self.application_name,
            "session_tag_or_fn": self.session_tag_or_fn,
            "enable_for_current_rank": self.enable_for_current_rank,
            "world_size_or_fn": self.world_size_or_fn,
            **self.extra,
        }

