from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional, Protocol
import torch

from .camera import CameraRecord


@dataclass
class AeroFixInput:
    noisy_rgb: torch.Tensor
    reference_rgb: torch.Tensor
    novel_camera: Optional[CameraRecord] = None
    reference_camera: Optional[CameraRecord] = None
    reference_depth: Optional[torch.Tensor] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class AeroFixResult:
    fixed_rgb: torch.Tensor
    quality: float = 1.0
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.quality = float(max(0.0, min(1.0, self.quality)))


class AeroFixProtocol(Protocol):
    def fix(self, item: AeroFixInput) -> AeroFixResult: ...


class IdentityAeroFix:
    """Smoke-test adapter. Do not use for actual ProDiG experiments."""

    def fix(self, item: AeroFixInput) -> AeroFixResult:
        return AeroFixResult(item.noisy_rgb.detach().clone(), quality=1.0)


def load_aerofix_factory(spec: str) -> AeroFixProtocol:
    if not spec or spec == "identity":
        return IdentityAeroFix()
    if ":" not in spec:
        raise ValueError("AeroFix factory must be 'module:function' or 'identity'")
    module_name, func_name = spec.split(":", 1)
    module = import_module(module_name)
    factory = getattr(module, func_name)
    instance = factory()
    if not hasattr(instance, "fix"):
        raise TypeError("AeroFix factory result must provide .fix(AeroFixInput)")
    return instance
