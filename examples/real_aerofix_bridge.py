"""Real AeroFix bridge backed by anchor_prodig's trained-checkpoint refiner.

This does not reimplement AeroFix inference. It wires prodig's AeroFixBridge
interface directly to `anchor_prodig.diffusion.aerofix.ProDiGPklAeroFixRefiner`,
which already knows how to load a `model_*.pkl` checkpoint produced by the
public ProDiG aeroFix trainer and run pose/Pluecker geometry-conditioned
single-step diffusion inference.

Run with:
    python -m prodig.train ... \
      --aerofix-factory examples.real_aerofix_bridge:create_aerofix

Requires the `aerofix` conda environment (gsplat, pycolmap, anchor_prodig all
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
#   export AEROFIX_CHECKPOINT=/path/to/model_XXXXX.pkl
#   export AEROFIX_SOURCE_DIR=/path/to/ProDiG/aeroFix/src
# There are no defaults: both must be set, and _require_paths() below fails fast
# with an actionable message if either is missing.
CHECKPOINT = os.environ.get("AEROFIX_CHECKPOINT", "")
OFFICIAL_SOURCE_DIR = os.environ.get("AEROFIX_SOURCE_DIR", "")


def _require_paths() -> None:
    """Fail at construction, not after the initial training phase is already spent."""
    missing = [
        (n, v) for n, v in (("AEROFIX_CHECKPOINT", CHECKPOINT),
                            ("AEROFIX_SOURCE_DIR", OFFICIAL_SOURCE_DIR))
        # Path("") is ".", which exists, so an unset variable must be checked explicitly
        if not v or not Path(v).exists()
    ]
    if missing:
        lines = "\n".join(f"  {n} = {v or '(not set)'}" for n, v in missing)
        raise FileNotFoundError(
            "aeroFix path(s) not set or not found on this machine:\n"
            f"{lines}\n"
            "Set them for your checkout, e.g.\n"
            "  export AEROFIX_CHECKPOINT=/path/to/model_10495.pkl\n"
            "  export AEROFIX_SOURCE_DIR=/path/to/ProDiG/aeroFix/src"
        )
# Matches `pairs.resolution: 512` in aerofix_training/configs/park.yaml -- the
# square resolution the checkpoint was actually trained/paired on. Feeding
# full-resolution images/K in caused epipolar-mask indices to overflow the
# model's fixed 512/8 latent grid.
PAIR_RESOLUTION = 512


def _rescale_intrinsics(K: np.ndarray, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> np.ndarray:
    """Same formula as anchor_prodig.geometry.cameras.rescale_intrinsics."""
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    result = np.asarray(K, dtype=np.float32).copy()
    result[0, :] *= new_w / old_w
    result[1, :] *= new_h / old_h
    result[2, 2] = 1.0
    return result


def _denormalize_c2w(c2w: torch.Tensor, normalization_transform: torch.Tensor) -> np.ndarray:
    """Undo prodig's scene normalization (unit-sphere camera-center scaling).

    ``normalization_transform`` maps original COLMAP-scale points to prodig's
    internal normalized frame: ``normalized = scale * original - scale * center``
    (see prodig/colmap_scene.py::normalize_scene). AeroFix's epipolar/Pluecker
    geometry expects real-world-scale poses, so only the translation column
    needs to be mapped back through the inverse of that affine transform; the
    rotation block is untouched because normalization never rotates anything.
    """
    inv_transform = torch.linalg.inv(normalization_transform.to(c2w.dtype))
    normalized_translation = torch.cat([c2w[:3, 3], torch.ones(1, dtype=c2w.dtype)])
    original_translation = (inv_transform @ normalized_translation)[:3]
    out = c2w.clone()
    out[:3, 3] = original_translation
    return out.detach().cpu().numpy()


def _save_chw_rgb_square(rgb: torch.Tensor, path: Path, resolution: int) -> None:
    rgb = rgb.detach().float().cpu()
    if rgb.ndim != 3:
        raise ValueError("AeroFix noisy_rgb must be CHW or HWC")
    if rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)
    elif rgb.shape[-1] != 3:
        raise ValueError("AeroFix noisy_rgb must have 3 color channels")
    arr = (rgb.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(arr)
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    image.save(path)


def _save_file_image_square(source: Path, dest: Path, resolution: int) -> None:
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest)


class RealAeroFixBridge:
    """Adapter around anchor_prodig's ProDiGPklAeroFixRefiner."""

    def __init__(self) -> None:
        _require_paths()
        cache_dir = Path(tempfile.mkdtemp(prefix="prodig_aerofix_cache_"))
        cfg = DiffusionConfig(
            backend="aerofix",
            checkpoint=CHECKPOINT,
            checkpoint_format="prodig_pkl",
            official_source_dir=Path(OFFICIAL_SOURCE_DIR),
            aerofix_base_checkpoint="nvidia/difix_ref",
            aerofix_mv_unet_v="v4",
            aerofix_lora_rank_unet=4,
            aerofix_lora_rank_vae=4,
            aerofix_pose_embed=True,
            aerofix_plucker_embed=True,
            aerofix_require_geometry=True,
            prompt="remove degradation",
            timestep=199,
            inference_steps=1,
            guidance_scale=0.0,
            cache_dir=cache_dir,
        )
        # ProDiGPklAeroFixRefiner lazily loads the checkpoint on first .refine() call.
        self.refiner = create_refiner(cfg, device=torch.device("cuda"))
        self._io_dir = Path(tempfile.mkdtemp(prefix="prodig_aerofix_io_"))
        self._counter = 0

    def fix(self, item: AeroFixInput) -> AeroFixResult:
        if item.novel_camera is None or item.reference_camera is None:
            raise ValueError("RealAeroFixBridge requires novel_camera and reference_camera")
        if item.reference_camera.image_path is None:
            raise ValueError("reference_camera.image_path is required for real AeroFix inference")

        self._counter += 1
        noisy_path = self._io_dir / f"noisy_{self._counter:06d}.png"
        reference_path = self._io_dir / f"reference_{self._counter:06d}.png"
        output_path = self._io_dir / f"fixed_{self._counter:06d}.png"
        _save_chw_rgb_square(item.noisy_rgb, noisy_path, PAIR_RESOLUTION)
        _save_file_image_square(item.reference_camera.image_path, reference_path, PAIR_RESOLUTION)

        metadata = item.metadata or {}
        stage = metadata.get("factor_index", "na")
        normalization_transform = metadata.get("normalization_transform")
        if normalization_transform is None:
            raise ValueError(
                "AeroFixInput.metadata['normalization_transform'] is required; "
                "prodig.trainer.ProDiGTrainer.generate_and_fix_stage must supply it."
            )
        # anchor_prodig's preprocess.py saves query/reference pose directly from
        # original (un-normalized) COLMAP c2w. prodig internally renormalizes
        # camera translations to a unit-sphere frame for gsplat training
        # stability, so that scale must be undone before this pose reaches
        # AeroFix's epipolar/Pluecker geometry code.
        query_pose = _denormalize_c2w(item.novel_camera.c2w, normalization_transform)
        reference_pose = _denormalize_c2w(item.reference_camera.c2w, normalization_transform)
        # K must match the square PAIR_RESOLUTION images actually sent to the
        # model (see anchor_prodig aerofix_factory/preprocess.py: images are
        # resized to (resolution, resolution) and K is rescaled to match).
        query_K = _rescale_intrinsics(
            item.novel_camera.K.detach().cpu().numpy(),
            (int(item.novel_camera.height), int(item.novel_camera.width)),
            (PAIR_RESOLUTION, PAIR_RESOLUTION),
        )
        reference_K = _rescale_intrinsics(
            item.reference_camera.K.detach().cpu().numpy(),
            (int(item.reference_camera.height), int(item.reference_camera.width)),
            (PAIR_RESOLUTION, PAIR_RESOLUTION),
        )
        request = RefinementRequest(
            input_image=noisy_path,
            output_image=output_path,
            reference_image=reference_path,
            pose=query_pose,
            reference_pose=reference_pose,
            K=query_K,
            reference_K=reference_K,
            cache_key_extra=f"prodig_stage:{stage}:{item.novel_camera.name}",
        )
        result = self.refiner.refine(request)

        # AeroFix always returns a PAIR_RESOLUTION-square image (it thinks that
        # square image *is* the original, since that's the size we handed it).
        # novel_camera keeps the source camera's native width/height, and
        # ProgressiveDataset.__getitem__ enforces that the saved fixed-view
        # file matches those dimensions exactly -- so resize back up here.
        native_size = (int(item.novel_camera.width), int(item.novel_camera.height))
        with Image.open(result.output_image) as handle:
            fixed = handle.convert("RGB")
            if fixed.size != native_size:
                fixed = fixed.resize(native_size, Image.Resampling.LANCZOS)
        arr = np.asarray(fixed, dtype=np.float32) / 255.0
        fixed_rgb = torch.from_numpy(arr.copy()).permute(2, 0, 1)

        # ProDiGPklAeroFixRefiner does not emit a confidence score; the paper's
        # exact quality-filter formula is not public either,
        # so this bridge accepts every refined view. Tighten with a real metric
        # (e.g. LPIPS-to-reference) once one is available.
        return AeroFixResult(fixed_rgb=fixed_rgb, quality=1.0, metadata=dict(result.metadata))


def create_aerofix() -> RealAeroFixBridge:
    return RealAeroFixBridge()
