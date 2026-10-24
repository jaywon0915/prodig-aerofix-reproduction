from __future__ import annotations

import torch
from .camera import CameraRecord


def make_progressive_camera(
    reference: CameraRecord,
    factor: float,
    mode: str,
    scene_center: torch.Tensor,
    forward_mix: float = 0.15,
) -> CameraRecord:
    """Create a lower-distance/altitude proxy camera from a reference camera.

    `factor` is the desired fraction of camera-to-scene distance. In an arbitrary
    COLMAP coordinate frame this is more robust than assuming world-Z is altitude.
    """
    if not 0.0 < factor <= 1.0:
        raise ValueError("factor must be in (0,1]")
    center = reference.center
    scene_center = torch.as_tensor(scene_center, dtype=center.dtype, device=center.device)
    radial = center - scene_center
    radius = torch.linalg.norm(radial)
    c2w = reference.c2w.clone()

    if mode == "scaled":
        new_center = scene_center + radial * factor
    elif mode == "forward":
        move = (1.0 - factor) * radius
        new_center = center + reference.forward * move
    elif mode == "scaled_forward":
        scaled = scene_center + radial * factor
        extra = reference.forward * ((1.0 - factor) * radius * forward_mix)
        new_center = scaled + extra
    else:
        raise ValueError(f"Unsupported trajectory mode: {mode}")

    c2w[:3, 3] = new_center
    suffix = int(round(factor * 100))
    return reference.with_pose(
        c2w,
        name=f"{reference.name.rsplit('.', 1)[0]}_p{suffix:02d}.png",
        image_path=None,
        image_id=-1,
        source="novel",
        reference_name=reference.name,
        stage_factor=float(factor),
    )
