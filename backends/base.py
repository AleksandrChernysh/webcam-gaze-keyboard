from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackingResult:
    method_name: str
    valid: bool
    features: dict[str, Any] = field(default_factory=dict)
    annotated_frame: Any | None = None
    landmarks: Any | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    error: str | None = None


class TrackerBackend(ABC):
    @abstractmethod
    def process(self, frame) -> TrackingResult:
        pass