from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class _GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, init_gate: float) -> None:
        super().__init__()
        if not 0.0 < init_gate < 1.0:
            raise ValueError("init_gate must be in (0, 1)")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, math.log(init_gate / (1.0 - init_gate)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistanceAdaptiveModule(nn.Module):
    """ProDiG-style distance-conditioned scale and opacity gates.

    The paper specifies F_sca(f_sca, d_gc) and F_opa(f_opa, d_gc), followed by
    a sigmoid gate. The exact distance normalization is not specified; this
    implementation uses log1p(distance / scene_scale) for numerical stability.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 16,
        init_gate: float = 0.99,
        distance_mode: str = "log_scene",
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.distance_mode = distance_mode
        self.scale_mlp = _GateMLP(feature_dim + 1, hidden_dim, init_gate)
        self.opacity_mlp = _GateMLP(feature_dim + 1, hidden_dim, init_gate)

    def normalized_distance(
        self, means: torch.Tensor, camera_center: torch.Tensor, scene_scale: float
    ) -> torch.Tensor:
        center = camera_center.reshape(1, 3).to(means)
        d = torch.linalg.norm(means - center, dim=-1, keepdim=True)
        denom = max(float(scene_scale), 1e-8)
        if self.distance_mode == "scene":
            return d / denom
        if self.distance_mode == "log_scene":
            return torch.log1p(d / denom)
        if self.distance_mode == "raw":
            return d
        raise ValueError(f"Unknown distance_mode: {self.distance_mode}")

    def forward(
        self,
        means: torch.Tensor,
        camera_center: torch.Tensor,
        f_sca: torch.Tensor,
        f_opa: torch.Tensor,
        scene_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.normalized_distance(means, camera_center, scene_scale)
        if f_sca.shape[:-1] != d.shape[:-1] or f_sca.shape[-1] != self.feature_dim:
            raise ValueError("f_sca shape does not match means/feature_dim")
        if f_opa.shape != f_sca.shape:
            raise ValueError("f_opa must have the same shape as f_sca")
        scale_gate = torch.sigmoid(self.scale_mlp(torch.cat([f_sca, d], dim=-1)))
        opacity_gate = torch.sigmoid(self.opacity_mlp(torch.cat([f_opa, d], dim=-1)))
        return scale_gate, opacity_gate


class AppearanceModule(nn.Module):
    """Per-reference appearance embedding with per-Gaussian view-conditioned RGB correction."""

    def __init__(
        self,
        num_appearances: int,
        gaussian_feature_dim: int = 32,
        embed_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_appearances <= 0:
            raise ValueError("num_appearances must be positive")
        self.embeddings = nn.Embedding(num_appearances, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(gaussian_feature_dim + embed_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        last = self.mlp[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        features: torch.Tensor,
        appearance_id: int | torch.Tensor,
        dirs: torch.Tensor,
    ) -> torch.Tensor:
        if dirs.shape != (features.shape[0], 3):
            raise ValueError("dirs must be [N,3]")
        dirs = F.normalize(dirs, dim=-1)
        if isinstance(appearance_id, int):
            ids = torch.tensor([appearance_id], device=features.device, dtype=torch.long)
        else:
            ids = appearance_id.to(device=features.device, dtype=torch.long).reshape(-1)
        if ids.numel() != 1:
            raise ValueError("This trainer renders one camera at a time")
        emb = self.embeddings(ids)[0].expand(features.shape[0], -1)
        return self.mlp(torch.cat([features, emb, dirs], dim=-1))
