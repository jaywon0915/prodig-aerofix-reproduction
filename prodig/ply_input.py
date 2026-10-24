"""Load a pre-trained Gaussian model from a standard 3DGS PLY.

Why this exists
---------------
The normal pipeline builds Gaussians from the COLMAP point cloud and spends an
initial phase (default 7,000 steps) fitting them to the real aerial photographs
before any descent begins. When a good Gaussian model already exists -- trained
by this pipeline, by graphdeco 3DGS, or by any tool writing the standard 3DGS PLY
layout -- that phase is redundant. This module reconstructs the model directly
from the PLY so the run starts at the first descent stage.

Format
------
Reads the same layout ``prodig.export.export_ply`` writes, which is the common
3DGS convention::

    x, y, z              position
    f_dc_0..2            degree-0 SH  (color = SH_C0 * sh0 + 0.5)
    opacity              RAW logit    (viewers apply sigmoid)
    scale_0..2           RAW log      (viewers apply exp)
    rot_0..3             quaternion
    f_rest_*             higher-order SH -- ignored, this model is degree-0 only

``opacity`` and ``scale_*`` are stored pre-activation by convention and are
copied through unchanged; applying exp/sigmoid here would double-apply the
nonlinearity and inflate every Gaussian to roughly scene size.

Per-Gaussian features the PLY cannot carry (``features``, ``f_sca``, ``f_opa``
-- the appearance and distance-adaptive inputs) are zero-initialised, which is
the same state ``ProDiGModel.from_points`` starts them in.

Coordinate frame
----------------
Critical and easy to get wrong. prodig normalises the scene to a unit sphere
(``colmap_scene.normalize_scene``), so:

* a PLY exported by THIS pipeline is already in the normalized frame
  (``export_ply`` writes ``model.splats["means"]`` verbatim) -> ``frame="normalized"``
* a PLY from an external trainer is in COLMAP world coordinates
  -> ``frame="world"``, and both positions and scales must be transformed

Passing the wrong frame does not raise -- it silently places the Gaussians in the
wrong place relative to the cameras and every render comes out empty or garbage,
so ``describe_ply_frame`` is provided to sanity-check before a long run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

_SH_C0 = 0.28209479177387814
_REQUIRED = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a binary-little-endian float32 PLY into {property: array}."""
    path = Path(path)
    with open(path, "rb") as f:
        header, names, count, fmt = b"", [], None, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            header += line
            text = line.decode("ascii", "replace").strip()
            if text.startswith("format"):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                count = int(text.split()[2])
            elif text.startswith("property"):
                parts = text.split()
                if parts[1] != "float" and parts[1] != "float32":
                    raise ValueError(f"{path}: only float32 properties supported, got '{text}'")
                names.append(parts[2])
            elif text == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: expected binary_little_endian, got '{fmt}'")
        if not count:
            raise ValueError(f"{path}: element vertex is 0")
        data = np.frombuffer(f.read(count * len(names) * 4), dtype="<f4")
    if data.size != count * len(names):
        raise ValueError(f"{path}: truncated -- expected {count * len(names)} floats, got {data.size}")
    data = data.reshape(count, len(names))
    return {n: data[:, i] for i, n in enumerate(names)}


def _logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x) - np.log1p(-x)


def load_ply_splats(
    path: str | Path,
    *,
    frame: str = "world",
    normalization_transform: torch.Tensor | None = None,
    gaussian_feature_dim: int = 32,
    distance_feature_dim: int = 8,
    device: torch.device | str = "cpu",
) -> nn.ParameterDict:
    """Rebuild a ``ProDiGModel``-compatible ParameterDict from a 3DGS PLY.

    Args:
        frame: ``"world"`` if the PLY is in COLMAP coordinates (external
            trainers), ``"normalized"`` if it came from this pipeline's own
            ``export_ply``. ``"world"`` requires ``normalization_transform``.
        normalization_transform: ``SceneData.normalization_transform``.
    """
    if frame not in ("world", "normalized"):
        raise ValueError(f"frame must be 'world' or 'normalized', got {frame!r}")
    props = read_ply(path)
    missing = [k for k in _REQUIRED if k not in props]
    if missing:
        raise ValueError(f"{path}: missing required PLY properties: {missing}")

    means = np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float32)
    scales = np.stack([props[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)
    quats = np.stack([props[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    opacities = props["opacity"].astype(np.float32)
    sh0 = np.stack([props[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)

    if frame == "world":
        if normalization_transform is None:
            raise ValueError("frame='world' requires normalization_transform")
        t = np.asarray(normalization_transform.detach().cpu().numpy(), dtype=np.float32)
        means = means @ t[:3, :3].T + t[:3, 3]
        # Scales are stored as log(size); the transform rescales sizes by s, so
        # in log space that is an additive shift, not a multiply.
        s = float(t[0, 0])
        if s <= 0:
            raise ValueError(f"non-positive normalization scale {s}")
        scales = scales + math.log(s)

    # sh0 -> linear color -> logit, matching ProDiGModel's stored representation.
    colors = _logit(np.clip(_SH_C0 * sh0 + 0.5, 0.0, 1.0))

    n = means.shape[0]
    dev = torch.device(device)
    def P(a):  # noqa: E306
        return nn.Parameter(torch.from_numpy(np.ascontiguousarray(a)).to(dev))
    return nn.ParameterDict({
        "means": P(means),
        "scales": P(scales),
        "quats": P(quats),
        "opacities": P(opacities),
        "colors": P(colors),
        # Not representable in a standard 3DGS PLY; same init as from_points().
        "features": nn.Parameter(torch.zeros((n, gaussian_feature_dim), device=dev)),
        "f_sca": nn.Parameter(torch.zeros((n, distance_feature_dim), device=dev)),
        "f_opa": nn.Parameter(torch.zeros((n, distance_feature_dim), device=dev)),
    })


def describe_ply_frame(path: str | Path, scene) -> str:
    """Human-readable frame check: compare PLY extent against the scene's.

    Wrong ``--input-ply-frame`` fails silently -- the Gaussians land far from the
    cameras and every render is empty -- so this is printed before training as a
    cheap way to notice before hours are spent.
    """
    props = read_ply(path)
    pts = np.stack([props["x"], props["y"], props["z"]], axis=1)
    ply_r = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
    cams = torch.stack([c.c2w[:3, 3] for c in scene.cameras]).numpy()
    cam_r = float(np.linalg.norm(cams - cams.mean(0), axis=1).max())
    guess = "normalized" if ply_r < cam_r * 20 else "world"
    return (f"PLY: {pts.shape[0]:,} Gaussians, extent radius {ply_r:.3f}; "
            f"normalized camera radius {cam_r:.3f} -> looks like frame='{guess}'")
