from __future__ import annotations

import math
import warnings
from typing import Optional
import torch
from torch import nn

from .camera import CameraRecord
from .modules import AppearanceModule, DistanceAdaptiveModule


def _logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def _initial_log_scales(points: torch.Tensor, init_scale: float = 1.0) -> torch.Tensor:
    n = points.shape[0]
    if n < 2:
        return torch.zeros((n, 3), dtype=points.dtype, device=points.device)
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        # Without k-NN every Gaussian gets one uniform initial scale, gsplat's
        # oversized-prune step then culls almost the whole model, and the run still
        # exits 0 with dark renders (see docs/USAGE.md). Fail here instead.
        raise ImportError(
            "scikit-learn is required for per-point initial scales: pip install scikit-learn"
        ) from exc
    try:
        k = min(4, n)
        pts = points.detach().cpu().numpy()
        nn_model = NearestNeighbors(n_neighbors=k).fit(pts)
        distances, _ = nn_model.kneighbors(pts)
        d = torch.from_numpy(distances[:, 1:].mean(axis=1)).to(points)
    except Exception as exc:
        warnings.warn(f"k-NN initial scales failed ({exc!r}); using a coarser fallback", RuntimeWarning)
        if n > 4096:
            # Robust fallback that avoids an O(N^2) allocation for large scenes.
            span = torch.linalg.norm(points.max(0).values - points.min(0).values)
            d = torch.full((n,), float(span / max(n ** (1.0 / 3.0), 1.0)), device=points.device)
        else:
            pair = torch.cdist(points, points)
            pair.fill_diagonal_(float("inf"))
            d = pair.min(dim=1).values
    d = torch.clamp(d * init_scale, min=1e-5)
    return torch.log(d).unsqueeze(-1).repeat(1, 3)


def apply_adaptive_gates(
    scales: torch.Tensor,
    opacity: torch.Tensor,
    scale_gate: torch.Tensor,
    opacity_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return scales * scale_gate, opacity * opacity_gate.squeeze(-1)


class ProDiGModel(nn.Module):
    def __init__(
        self,
        splats: nn.ParameterDict,
        num_appearances: int,
        scene_scale: float,
        gaussian_feature_dim: int = 32,
        distance_feature_dim: int = 8,
        distance_hidden_dim: int = 16,
        appearance_embed_dim: int = 16,
        appearance_hidden_dim: int = 64,
        distance_init_gate: float = 0.99,
        distance_mode: str = "log_scene",
    ) -> None:
        super().__init__()
        self.splats = splats
        self.scene_scale = float(scene_scale)
        self.distance = DistanceAdaptiveModule(
            feature_dim=distance_feature_dim,
            hidden_dim=distance_hidden_dim,
            init_gate=distance_init_gate,
            distance_mode=distance_mode,
        )
        self.appearance = AppearanceModule(
            num_appearances=num_appearances,
            gaussian_feature_dim=gaussian_feature_dim,
            embed_dim=appearance_embed_dim,
            hidden_dim=appearance_hidden_dim,
        )

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        rgbs: torch.Tensor,
        num_appearances: int,
        scene_scale: float,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        gaussian_feature_dim: int = 32,
        distance_feature_dim: int = 8,
        **kwargs,
    ) -> "ProDiGModel":
        points = points.float()
        rgbs = rgbs.float().clamp(0.0, 1.0)
        if points.ndim != 2 or points.shape[-1] != 3 or rgbs.shape != points.shape:
            raise ValueError("points and rgbs must both be [N,3]")
        n = points.shape[0]
        quats = torch.zeros((n, 4), dtype=points.dtype, device=points.device)
        quats[:, 0] = 1.0
        splats = nn.ParameterDict(
            {
                "means": nn.Parameter(points.clone()),
                "scales": nn.Parameter(_initial_log_scales(points, init_scale)),
                "quats": nn.Parameter(quats),
                "opacities": nn.Parameter(
                    torch.full((n,), math.log(init_opacity / (1.0 - init_opacity)), device=points.device)
                ),
                "features": nn.Parameter(torch.zeros((n, gaussian_feature_dim), device=points.device)),
                "colors": nn.Parameter(_logit(rgbs)),
                "f_sca": nn.Parameter(torch.zeros((n, distance_feature_dim), device=points.device)),
                "f_opa": nn.Parameter(torch.zeros((n, distance_feature_dim), device=points.device)),
            }
        )
        return cls(
            splats=splats,
            num_appearances=num_appearances,
            scene_scale=scene_scale,
            gaussian_feature_dim=gaussian_feature_dim,
            distance_feature_dim=distance_feature_dim,
            **kwargs,
        )

    def create_splat_optimizers(
        self,
        batch_size: int = 1,
        lrs: Optional[dict[str, float]] = None,
    ) -> dict[str, torch.optim.Optimizer]:
        defaults = {
            "means": 1.6e-4 * self.scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "features": 2.5e-3,
            "colors": 2.5e-3,
            "f_sca": 1e-3,
            "f_opa": 1e-3,
        }
        if lrs:
            defaults.update(lrs)
        bs_scale = math.sqrt(max(batch_size, 1))
        return {
            name: torch.optim.Adam(
                [{"params": [param], "lr": defaults[name] * bs_scale, "name": name}],
                eps=1e-15 / bs_scale,
            )
            for name, param in self.splats.items()
        }

    def create_module_optimizer(self, lr: float = 1e-3) -> torch.optim.Optimizer:
        params = list(self.distance.parameters()) + list(self.appearance.parameters())
        return torch.optim.Adam(params, lr=lr)

    def visible_mask(
        self,
        camera: CameraRecord,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
    ) -> torch.Tensor:
        """Cheap, gradient-free frustum pre-filter using base (un-gated) scales.

        The distance-adaptive gate only ever shrinks scale/opacity (sigmoid in
        (0,1)), so a visibility test against the un-gated base scale is a safe
        superset of what would actually be visible after gating -- it never
        drops a Gaussian that should render.
        """
        try:
            from gsplat import fully_fused_projection
        except ModuleNotFoundError as exc:
            raise RuntimeError("gsplat is required for rendering; install requirements.txt") from exc

        means = self.splats["means"]
        with torch.no_grad():
            base_scales = torch.exp(self.splats["scales"])
            viewmat = torch.linalg.inv(camera.c2w.to(means))[None]
            K = camera.K.to(means)[None]
            radii, _, _, _, _ = fully_fused_projection(
                means,
                None,
                self.splats["quats"],
                base_scales,
                viewmat,
                K,
                width=int(camera.width),
                height=int(camera.height),
                near_plane=near_plane,
                far_plane=far_plane,
            )
            return (radii > 0).any(dim=-1).squeeze(0)

    def adapted_parameters(
        self, camera: CameraRecord, visible: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = self.splats["means"]
        base_scales = torch.exp(self.splats["scales"])
        base_opacity = torch.sigmoid(self.splats["opacities"])
        center = camera.center.to(means)

        if visible is None:
            idx = None
            means_v, scales_v, opacity_v = means, base_scales, base_opacity
            f_sca_v, f_opa_v, features_v, colors_v = (
                self.splats["f_sca"],
                self.splats["f_opa"],
                self.splats["features"],
                self.splats["colors"],
            )
        else:
            idx = torch.nonzero(visible, as_tuple=False).squeeze(-1)
            means_v = means[idx]
            scales_v = base_scales[idx]
            opacity_v = base_opacity[idx]
            f_sca_v = self.splats["f_sca"][idx]
            f_opa_v = self.splats["f_opa"][idx]
            features_v = self.splats["features"][idx]
            colors_v = self.splats["colors"][idx]

        scale_gate, opacity_gate = self.distance(means_v, center, f_sca_v, f_opa_v, self.scene_scale)
        scales_out, opacities_out = apply_adaptive_gates(scales_v, opacity_v, scale_gate, opacity_gate)
        dirs = means_v - center.reshape(1, 3)
        color_delta = self.appearance(features_v, camera.appearance_id, dirs)
        colors_out = torch.sigmoid(colors_v + color_delta)

        if idx is None:
            return scales_out, opacities_out, colors_out

        # Scatter the visible subset's MLP outputs back into full-N tensors so
        # rasterization()/DefaultStrategy still see the same [N, ...] shapes and
        # index alignment as before culling was added. Non-visible slots keep a
        # non-degenerate fallback scale (base_scales, not zero) and zero
        # opacity, so they render as fully transparent without risking a
        # degenerate covariance inside gsplat's own projection math.
        n = means.shape[0]
        scales_full = base_scales.clone()
        opacities_full = base_opacity.new_zeros((n,))
        colors_full = torch.sigmoid(self.splats["colors"]).clone()
        scales_full[idx] = scales_out
        opacities_full[idx] = opacities_out
        colors_full[idx] = colors_out
        return scales_full, opacities_full, colors_full

    def render(
        self,
        camera: CameraRecord,
        packed: bool = False,
        absgrad: bool = False,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        render_mode: str = "RGB",
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        try:
            from gsplat.rendering import rasterization
        except ModuleNotFoundError as exc:
            raise RuntimeError("gsplat is required for rendering; install requirements.txt") from exc

        means = self.splats["means"]
        visible = self.visible_mask(camera, near_plane=near_plane, far_plane=far_plane)
        scales, opacities, colors = self.adapted_parameters(camera, visible=visible)
        c2w = camera.c2w.to(means)
        K = camera.K.to(means)
        renders, alphas, info = rasterization(
            means=means,
            quats=self.splats["quats"],
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w)[None],
            Ks=K[None],
            width=int(camera.width),
            height=int(camera.height),
            packed=packed,
            absgrad=absgrad,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
        )
        return renders, alphas, info
