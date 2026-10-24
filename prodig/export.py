"""Export a ProDiGModel's Gaussians to a standard, viewer-portable PLY file.

The distance-adaptive gates and appearance-embedding color correction are
per-camera (they depend on `d_gc` and a reference appearance id), so there is
no single "correct" gated snapshot for every viewpoint at once. By default
(`camera=None`) this export uses the un-gated base parameters -- i.e. the
persistent per-Gaussian state that is actually optimized and densified,
before any per-view modulation is applied. Pass `camera` to instead bake in
the scale/opacity gate and appearance color correction that `model.render()`
would compute for that specific camera -- the result matches what
`model.render(camera)` actually draws, viewed from anywhere near that
camera's distance/appearance condition (it will drift again for viewpoints
far from it, since the gate is still a single frozen snapshot).

`means`/`quats` are written as-is. `scales` and `opacities` are written in
their *raw* (pre-activation) form -- standard 3DGS-PLY consumers (Supersplat,
antimatter15, gsplat's own loader, etc.) apply `exp()`/`sigmoid()` themselves
on load, exactly mirroring how the reference 3DGS implementation stores them.
Do not pre-apply those activations here: `model.splats["scales"]` is already
log-scale (see `_initial_log_scales`) and `model.splats["opacities"]` is
already a logit, so activating them before writing double-applies the
nonlinearity in every downstream viewer and inflates every Gaussian's scale
to roughly the size of the whole scene. When `camera` is given, the gated
(activated) scale/opacity are converted back to log-scale/logit before
writing, for the same reason.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from .camera import CameraRecord
from .model import ProDiGModel, _logit

# Degree-0 spherical harmonic normalization constant used by the standard
# 3DGS/gsplat PLY convention: color = SH_C0 * sh0 + 0.5.
_SH_C0 = 0.28209479177387814


def export_ply(model: ProDiGModel, path: Path, camera: Optional[CameraRecord] = None) -> Path:
    try:
        from gsplat import export_splats
    except ModuleNotFoundError as exc:
        raise RuntimeError("gsplat is required to export a PLY; install requirements.txt") from exc

    with torch.no_grad():
        means = model.splats["means"].detach().float()
        quats = model.splats["quats"].detach().float()

        if camera is None:
            scales = model.splats["scales"].detach().float()
            opacities = model.splats["opacities"].detach().float()
            colors = torch.sigmoid(model.splats["colors"]).detach().float()
        else:
            device = means.device
            cam = replace(camera, c2w=camera.c2w.to(device), K=camera.K.to(device))
            scales_gated, opacities_gated, colors_gated = model.adapted_parameters(cam)
            # adapted_parameters() returns activated (linear-space) values --
            # invert back to the raw log-scale/logit form the PLY convention
            # expects, same as the camera=None branch above.
            scales = torch.log(scales_gated.clamp_min(1e-8)).detach().float()
            opacities = _logit(opacities_gated).detach().float()
            colors = colors_gated.detach().float()

        sh0 = ((colors - 0.5) / _SH_C0).unsqueeze(1)  # (N, 1, 3)
        shN = means.new_zeros((means.shape[0], 0, 3))  # no higher-order SH

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(path),
    )
    return path
