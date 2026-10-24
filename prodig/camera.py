from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import torch


@dataclass
class CameraRecord:
    name: str
    c2w: torch.Tensor
    K: torch.Tensor
    width: int
    height: int
    image_path: Optional[Path]
    image_id: int
    appearance_id: int
    weight: float = 1.0
    source: str = "aerial"
    reference_name: Optional[str] = None
    stage_factor: Optional[float] = None

    def __post_init__(self) -> None:
        self.c2w = torch.as_tensor(self.c2w, dtype=torch.float32).clone()
        self.K = torch.as_tensor(self.K, dtype=torch.float32).clone()
        if self.c2w.shape != (4, 4):
            raise ValueError("c2w must have shape [4,4]")
        if self.K.shape != (3, 3):
            raise ValueError("K must have shape [3,3]")

    @property
    def center(self) -> torch.Tensor:
        return self.c2w[:3, 3]

    @property
    def forward(self) -> torch.Tensor:
        # COLMAP/OpenCV camera convention: +z points forward.
        v = self.c2w[:3, 2]
        return v / torch.clamp(torch.linalg.norm(v), min=1e-8)

    def with_pose(self, c2w: torch.Tensor, **changes) -> "CameraRecord":
        return replace(self, c2w=c2w, **changes)
