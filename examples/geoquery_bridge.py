"""GeoQuery bridge backed by anchor_prodig's GeoQuery refiner.

Mirrors examples/real_aerofix_bridge.py's structure but swaps in GeoQuery
(anchor_prodig.diffusion.geoquery.GeoQueryRefiner) in place of AeroFix. Unlike
AeroFix, GeoQuery is geometry-conditioned at inference time: it needs the
reference view's metric depth (RefinementRequest.reference_depth) alongside
both camera poses/intrinsics, and reprojects real reference content via
depth+pose rather than freeform generation. It has no fixed pair-training
resolution the way the AeroFix checkpoint does, so (unlike the AeroFix bridge)
images are kept at their native size -- no forced square resize.

Run with:
    python -m prodig.train ... \
      --aerofix-factory examples.geoquery_bridge:create_geoquery \
      --reference-depth-dir data/park/depths_metric \
      --min-aerofix-quality 0.01

Requires the anchor-prodig conda env (gsplat, pycolmap, anchor_prodig all
installed there) and a CUDA device.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anchor_prodig.config import DiffusionConfig
from anchor_prodig.diffusion.aerofix import create_refiner
from anchor_prodig.diffusion.base import RefinementRequest
from prodig.aerofix import AeroFixInput, AeroFixResult

# Machine-specific paths. Override per machine without editing this file:
#   export GEOQUERY_CHECKPOINT=/path/to/geoquery_window_size3_down1.pkl
#   export GEOQUERY_SOURCE_DIR=/path/to/GeoQuery/src
# There are no defaults: both must be set, and _require_paths() below fails fast
# with an actionable message if either is missing.
GEOQUERY_CHECKPOINT = os.environ.get("GEOQUERY_CHECKPOINT", "")
GEOQUERY_SOURCE_DIR = os.environ.get("GEOQUERY_SOURCE_DIR", "")


def _require_paths() -> None:
    """Fail at construction with an actionable message, not mid-run.

    Without this the first .fix() call dies deep inside the refiner with a
    FileNotFoundError naming a path from someone else's machine, typically after
    the initial 7,000 training steps have already been spent.
    """
    missing = [
        (name, value)
        for name, value in (
            ("GEOQUERY_CHECKPOINT", GEOQUERY_CHECKPOINT),
            ("GEOQUERY_SOURCE_DIR", GEOQUERY_SOURCE_DIR),
        )
        # Path("") is ".", which exists, so an unset variable must be checked explicitly
        if not value or not Path(value).exists()
    ]
    if missing:
        lines = "\n".join(f"  {n} = {v or '(not set)'}" for n, v in missing)
        raise FileNotFoundError(
            "GeoQuery path(s) not set or not found on this machine:\n"
            f"{lines}\n"
            "Set them for your checkout, e.g.\n"
            "  export GEOQUERY_CHECKPOINT=/path/to/geoquery_window_size3_down1.pkl\n"
            "  export GEOQUERY_SOURCE_DIR=/path/to/GeoQuery/src"
        )


# Longest image side handed to GeoQuery. Its geometry-guided cross-view attention
# is quadratic in token count, so cost explodes with resolution: at a native
# 1598x1064 park view, a single .fix() call peaks at ~23.5GB of a 24GB card while
# the Gaussian training it wraps needs only 2.8GB. Capping the diffusion input
# keeps full-resolution training, renders and noisy_views intact -- only the
# correction is computed smaller and resampled back, the same compromise
# real_aerofix_bridge.py makes with PAIR_RESOLUTION=512.
#
# Set GEOQUERY_MAX_SIDE=0 to disable the cap (native, needs ~24GB+).
GEOQUERY_MAX_SIDE = int(os.environ.get("GEOQUERY_MAX_SIDE", "1024"))


def _fit_to_cap(width: int, height: int) -> tuple[int, int]:
    """Scale (w, h) so the longer side is at most GEOQUERY_MAX_SIDE, preserving aspect."""
    longest = max(width, height)
    if GEOQUERY_MAX_SIDE <= 0 or longest <= GEOQUERY_MAX_SIDE:
        return width, height
    s = GEOQUERY_MAX_SIDE / longest
    return max(1, round(width * s)), max(1, round(height * s))


def _rescale_intrinsics(K: np.ndarray, old_wh: tuple[int, int], new_wh: tuple[int, int]) -> np.ndarray:
    """Rescale K to match a resized image (same formula as the aeroFix bridge).

    GeoQuery reprojects through K, so a resized image with unscaled intrinsics
    would place every reprojected sample in the wrong pixel.
    """
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, :] *= new_wh[0] / old_wh[0]
    out[1, :] *= new_wh[1] / old_wh[1]
    out[2, 2] = 1.0
    return out


def _denormalize_c2w(c2w: torch.Tensor, normalization_transform: torch.Tensor) -> np.ndarray:
    """Undo prodig's scene normalization (unit-sphere camera-center scaling).

    GeoQuery's geometry-guided attention reprojects through real metric depth
    values, so poses must be in the same real-world scale that depth is in --
    not prodig's internal normalized frame. See
    real_aerofix_bridge.py::_denormalize_c2w for the derivation; identical here.
    """
    inv_transform = torch.linalg.inv(normalization_transform.to(c2w.dtype))
    normalized_translation = torch.cat([c2w[:3, 3], torch.ones(1, dtype=c2w.dtype)])
    original_translation = (inv_transform @ normalized_translation)[:3]
    out = c2w.clone()
    out[:3, 3] = original_translation
    return out.detach().cpu().numpy()


def _save_chw_rgb(rgb: torch.Tensor, path: Path, size: tuple[int, int] | None = None) -> None:
    rgb = rgb.detach().float().cpu()
    if rgb.ndim != 3:
        raise ValueError("noisy_rgb must be CHW or HWC")
    if rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)
    elif rgb.shape[-1] != 3:
        raise ValueError("noisy_rgb must have 3 color channels")
    arr = (rgb.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(arr)
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(path)


def _save_reference_image(source: Path, dest: Path, size: tuple[int, int]) -> Path:
    """Write the reference at the capped size, or reuse the original when unchanged."""
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        if image.size == size:
            return source
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.resize(size, Image.Resampling.LANCZOS).save(dest)
    return dest


class GeoQueryBridge:
    """Adapter around anchor_prodig's GeoQueryRefiner."""

    def __init__(self) -> None:
        _require_paths()
        cache_dir = Path(tempfile.mkdtemp(prefix="prodig_geoquery_cache_"))
        cfg = DiffusionConfig(
            backend="geoquery",
            checkpoint=GEOQUERY_CHECKPOINT,
            checkpoint_format="geoquery_pkl",
            geoquery_source_dir=Path(GEOQUERY_SOURCE_DIR),
            geoquery_neighborhood_size=3,
            geoquery_low_res_only=True,
            geoquery_allow_unsafe_pickle=True,
            prompt="remove degradation",
            timestep=199,
            inference_steps=1,
            guidance_scale=0.0,
            cache_dir=cache_dir,
        )
        # GeoQueryRefiner lazily loads the checkpoint on first .refine() call.
        self.refiner = create_refiner(cfg, device=torch.device("cuda"))
        self._io_dir = Path(tempfile.mkdtemp(prefix="prodig_geoquery_io_"))
        self._counter = 0

    def fix(self, item: AeroFixInput) -> AeroFixResult:
        if item.novel_camera is None or item.reference_camera is None:
            raise ValueError("GeoQueryBridge requires novel_camera and reference_camera")
        if item.reference_camera.image_path is None:
            raise ValueError("reference_camera.image_path is required for GeoQuery inference")
        if item.reference_depth is None:
            # select_stage_references() (prodig/trainer.py) picks references
            # without regard to depth availability -- it has no concept of a
            # depth-requiring backend, since AeroFix (the package's original
            # target) never needs one. DA3 only has depth for 855/975 real
            # aerial views here, so this reference-view/depth-source combo is
            # a real, expected gap, not a bug. Reject softly via the
            # quality-threshold mechanism instead of raising, so one missing
            # depth file doesn't crash the whole run. The trainer never accepts
            # quality <= 0 (see trainer.accept_refined_view), whatever
            # --min-aerofix-quality is set to.
            return AeroFixResult(
                fixed_rgb=item.noisy_rgb.detach().clone(),
                quality=0.0,
                metadata={"skipped_reason": "no_reference_depth", "reference": item.reference_camera.name},
            )

        self._counter += 1
        noisy_path = self._io_dir / f"noisy_{self._counter:06d}.png"
        depth_path = self._io_dir / f"depth_{self._counter:06d}.npy"
        output_path = self._io_dir / f"fixed_{self._counter:06d}.png"

        native_wh = (int(item.novel_camera.width), int(item.novel_camera.height))
        work_wh = _fit_to_cap(*native_wh)
        ref_native_wh = (int(item.reference_camera.width), int(item.reference_camera.height))
        ref_work_wh = _fit_to_cap(*ref_native_wh)

        _save_chw_rgb(item.noisy_rgb, noisy_path, size=work_wh)
        # Depth must stay pixel-aligned with the reference image it describes, so
        # it is resampled to exactly the size the reference is fed at. Nearest
        # neighbour: averaging across a depth discontinuity invents surfaces that
        # sit between the foreground and the background.
        depth_np = item.reference_depth.detach().cpu().numpy().astype(np.float32)
        if (depth_np.shape[1], depth_np.shape[0]) != ref_work_wh:
            depth_np = np.asarray(
                Image.fromarray(depth_np, mode="F").resize(ref_work_wh, Image.Resampling.NEAREST),
                dtype=np.float32,
            )
        np.save(depth_path, depth_np)

        metadata = item.metadata or {}
        stage = metadata.get("factor_index", "na")
        normalization_transform = metadata.get("normalization_transform")
        if normalization_transform is None:
            raise ValueError(
                "AeroFixInput.metadata['normalization_transform'] is required; "
                "prodig.trainer.ProDiGTrainer.generate_and_fix_stage must supply it."
            )
        query_pose = _denormalize_c2w(item.novel_camera.c2w, normalization_transform)
        reference_pose = _denormalize_c2w(item.reference_camera.c2w, normalization_transform)
        # Scene normalization only rescales camera translations, never K (see
        # prodig/colmap_scene.py::normalize_scene) -- and both cameras are kept
        # at their native resolution here, so K needs no rescale either.
        # Both K's follow their own image to the capped size; the query and the
        # reference can differ in native size, so they are scaled independently.
        query_K = _rescale_intrinsics(item.novel_camera.K.detach().cpu().numpy(), native_wh, work_wh)
        reference_K = _rescale_intrinsics(
            item.reference_camera.K.detach().cpu().numpy(), ref_native_wh, ref_work_wh
        )
        reference_path = _save_reference_image(
            item.reference_camera.image_path,
            self._io_dir / f"reference_{self._counter:06d}.png",
            ref_work_wh,
        )

        request = RefinementRequest(
            input_image=noisy_path,
            output_image=output_path,
            reference_image=reference_path,
            pose=query_pose,
            reference_pose=reference_pose,
            K=query_K,
            reference_K=reference_K,
            reference_depth=depth_path,
            cache_key_extra=f"prodig_stage:{stage}:{item.novel_camera.name}",
        )
        result = self.refiner.refine(request)

        native_size = (int(item.novel_camera.width), int(item.novel_camera.height))
        with Image.open(result.output_image) as handle:
            fixed = handle.convert("RGB")
            if fixed.size != native_size:
                fixed = fixed.resize(native_size, Image.Resampling.LANCZOS)
        arr = np.asarray(fixed, dtype=np.float32) / 255.0
        fixed_rgb = torch.from_numpy(arr.copy()).permute(2, 0, 1)

        # GeoQueryRefiner does not emit a confidence score either; accept every
        # refined view here, same rationale as real_aerofix_bridge.py.
        return AeroFixResult(fixed_rgb=fixed_rgb, quality=1.0, metadata=dict(result.metadata))


def create_geoquery() -> GeoQueryBridge:
    return GeoQueryBridge()
