from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .aerofix import AeroFixInput, AeroFixProtocol
from .camera import CameraRecord
from .colmap_scene import SceneData
from .config import RunConfig
from .dataset import ProgressiveDataset, load_rgb
from .depth_calib import calibrate_relative_depth
from .export import export_ply
from .losses import mixed_rgb_loss
from .model import ProDiGModel
from .ply_input import describe_ply_frame, load_ply_splats
from .trajectory import make_progressive_camera


@dataclass
class StageReference:
    """One generated view promoted to serve as the next stage's GeoQuery reference."""

    camera: CameraRecord
    image_path: Path
    depth_path: Path


def _resize_map(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize a 2D map to (H, W).

    DA3 processes at a multiple of 14, so its output is generally not the render
    resolution. The calibration is a per-pixel affine fit, so resampling before
    fitting is harmless -- but the two maps must be pixel-aligned first.
    """
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def camera_to_dict(camera: CameraRecord) -> dict:
    return {
        "name": camera.name,
        "c2w": camera.c2w.cpu().tolist(),
        "K": camera.K.cpu().tolist(),
        "width": camera.width,
        "height": camera.height,
        "image_path": None if camera.image_path is None else str(camera.image_path),
        "image_id": camera.image_id,
        "appearance_id": camera.appearance_id,
        "weight": camera.weight,
        "source": camera.source,
        "reference_name": camera.reference_name,
        "stage_factor": camera.stage_factor,
    }


def camera_from_dict(data: dict) -> CameraRecord:
    path = data.get("image_path")
    return CameraRecord(
        name=data["name"],
        c2w=torch.tensor(data["c2w"], dtype=torch.float32),
        K=torch.tensor(data["K"], dtype=torch.float32),
        width=int(data["width"]),
        height=int(data["height"]),
        image_path=None if path is None else Path(path),
        image_id=int(data["image_id"]),
        appearance_id=int(data["appearance_id"]),
        weight=float(data.get("weight", 1.0)),
        source=data.get("source", "aerial"),
        reference_name=data.get("reference_name"),
        stage_factor=data.get("stage_factor"),
    )


def select_stage_references(
    cameras: list[CameraRecord], max_views: int
) -> list[CameraRecord]:
    if max_views <= 0 or max_views >= len(cameras):
        return list(cameras)
    if max_views == 1:
        return [cameras[0]]
    # Deterministic evenly-spaced sampling, preserving endpoints.
    indices = [round(i * (len(cameras) - 1) / (max_views - 1)) for i in range(max_views)]
    return [cameras[i] for i in indices]


def accept_refined_view(quality: float, min_quality: float) -> bool:
    """Whether a refined view enters training.

    quality <= 0 is a refiner's explicit rejection (e.g. GeoQuery with no reference
    depth) and is never accepted. A plain `quality < min_quality` check against the
    0.0 default would let it through, and the rejected view would then replace that
    camera's real aerial photo in the training set.
    """
    return quality > 0.0 and quality >= min_quality


def _save_chw_rgb(rgb: torch.Tensor, path: Path) -> None:
    rgb = rgb.detach().float().cpu()
    if rgb.ndim != 3:
        raise ValueError("AeroFix output must be CHW or HWC RGB")
    if rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)
    elif rgb.shape[-1] != 3:
        raise ValueError("AeroFix output must have 3 color channels")
    arr = (rgb.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _load_optional_depth(depth_dir: Optional[Path], image_name: str) -> Optional[torch.Tensor]:
    if depth_dir is None:
        return None
    stem = Path(image_name).stem
    npy = depth_dir / f"{stem}.npy"
    if npy.exists():
        return torch.from_numpy(np.load(npy).astype(np.float32))
    for ext in (".png", ".tif", ".tiff"):
        p = depth_dir / f"{stem}{ext}"
        if p.exists():
            return torch.from_numpy(np.asarray(Image.open(p), dtype=np.float32).copy())
    return None


class ProDiGTrainer:
    def __init__(
        self,
        scene: SceneData,
        cfg: RunConfig,
        aerofix: AeroFixProtocol,
        device: str = "cuda",
    ) -> None:
        self.scene = scene
        self.cfg = cfg
        self.aerofix = aerofix
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.cfg.output_dir / "fixed_views").mkdir(exist_ok=True)
        (self.cfg.output_dir / "noisy_views").mkdir(exist_ok=True)
        (self.cfg.output_dir / "logs").mkdir(exist_ok=True)
        with open(self.cfg.output_dir / "config.json", "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)

        self.original_cameras = list(scene.cameras)
        self.dataset = ProgressiveDataset(self.original_cameras)
        mc = cfg.model
        self.model = ProDiGModel.from_points(
            scene.points,
            scene.colors,
            num_appearances=len(self.original_cameras),
            scene_scale=scene.scene_scale,
            init_opacity=mc.init_opacity,
            init_scale=mc.init_scale,
            gaussian_feature_dim=mc.gaussian_feature_dim,
            distance_feature_dim=mc.distance_feature_dim,
            distance_hidden_dim=mc.distance_hidden_dim,
            appearance_embed_dim=mc.appearance_embed_dim,
            appearance_hidden_dim=mc.appearance_hidden_dim,
            distance_init_gate=mc.distance_init_gate,
            distance_mode=mc.distance_mode,
        ).to(self.device)

        if cfg.input_ply is not None:
            # Replace the point-cloud initialisation with the supplied model. The
            # ProDiGModel above is still constructed first so its distance/
            # appearance MLPs exist -- a PLY cannot carry those.
            print(describe_ply_frame(cfg.input_ply, scene), flush=True)
            self.model.splats = load_ply_splats(
                cfg.input_ply,
                frame=cfg.input_ply_frame,
                normalization_transform=scene.normalization_transform,
                gaussian_feature_dim=mc.gaussian_feature_dim,
                distance_feature_dim=mc.distance_feature_dim,
                device=self.device,
            )
            print(f"Loaded {len(self.model.splats['means']):,} Gaussians from "
                  f"{cfg.input_ply} (frame={cfg.input_ply_frame}); "
                  f"initial training phase = {cfg.progressive.initial_steps} steps", flush=True)

        self.global_step = 0
        self.completed_factors = 0
        self.phase = "initial"
        self.active_factor_index: Optional[int] = None
        self.phase_end_global_step = cfg.progressive.initial_steps
        # original camera name -> the preceding stage's generated view for that
        # camera, once it has a validated calibrated depth. Empty at stage 0, so
        # stage 0 always reprojects from the real aerial views.
        self._prev_stage: dict[str, StageReference] = {}
        self._build_training_state()

    def _build_training_state(self) -> None:
        try:
            from gsplat.strategy import DefaultStrategy
        except ModuleNotFoundError as exc:
            raise RuntimeError("gsplat is required for training") from exc
        tc = self.cfg.training
        self.splat_optimizers = self.model.create_splat_optimizers()
        self.module_optimizer = self.model.create_module_optimizer(tc.module_lr)
        self.strategy = DefaultStrategy(
            refine_start_iter=tc.refine_start_iter,
            refine_stop_iter=min(tc.refine_stop_iter, self.cfg.progressive.total_steps),
            refine_every=tc.refine_every,
            reset_every=tc.reset_every,
            prune_opa=tc.prune_opa,
            grow_grad2d=tc.grow_grad2d,
            grow_scale3d=tc.grow_scale3d,
            grow_scale2d=tc.grow_scale2d,
            prune_scale3d=tc.prune_scale3d,
            prune_scale2d=tc.prune_scale2d,
        )
        self.strategy.check_sanity(self.model.splats, self.splat_optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene.scene_scale)
        total = max(self.cfg.progressive.total_steps, 1)
        self.means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.splat_optimizers["means"], gamma=0.01 ** (1.0 / total)
        )

    def _zero_grad(self) -> None:
        for opt in self.splat_optimizers.values():
            opt.zero_grad(set_to_none=True)
        self.module_optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self) -> None:
        for opt in self.splat_optimizers.values():
            opt.step()
        self.module_optimizer.step()
        self.means_scheduler.step()

    def train_steps(self, num_steps: int, description: str) -> None:
        if num_steps <= 0:
            return
        self.model.train()
        pbar = tqdm(range(num_steps), desc=description)
        log_path = self.cfg.output_dir / "logs" / "train.jsonl"
        for local_step in pbar:
            idx = random.randrange(len(self.dataset))
            item = self.dataset[idx]
            camera = self.dataset.records[idx]
            target = item["image"].to(self.device).unsqueeze(0)
            camera_device = replace(camera, c2w=camera.c2w.to(self.device), K=camera.K.to(self.device))

            self._zero_grad()
            renders, alphas, info = self.model.render(
                camera_device,
                packed=self.cfg.training.packed,
                absgrad=bool(getattr(self.strategy, "absgrad", False)),
                near_plane=self.cfg.training.near_plane,
                far_plane=self.cfg.training.far_plane,
            )
            pred = renders[..., :3]

            self.strategy.step_pre_backward(
                params=self.model.splats,
                optimizers=self.splat_optimizers,
                state=self.strategy_state,
                step=self.global_step,
                info=info,
            )

            loss, parts = mixed_rgb_loss(pred, target, self.cfg.training.ssim_lambda)
            weighted_loss = loss * float(item["weight"])
            weighted_loss.backward()

            self._optimizer_step()
            self.strategy.step_post_backward(
                params=self.model.splats,
                optimizers=self.splat_optimizers,
                state=self.strategy_state,
                step=self.global_step,
                info=info,
                packed=self.cfg.training.packed,
            )
            self.global_step += 1

            pbar.set_postfix(
                loss=f"{float(weighted_loss.detach()):.4f}",
                gs=len(self.model.splats["means"]),
                src=item["source"],
            )
            if self.global_step % 50 == 0:
                row = {
                    "step": self.global_step,
                    "loss": float(weighted_loss.detach().cpu()),
                    "l1": float(parts["l1"].detach().cpu()),
                    "ssim_loss": float(parts["ssim_loss"].detach().cpu()),
                    "num_gaussians": len(self.model.splats["means"]),
                    "source": item["source"],
                    "dataset_size": len(self.dataset),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            if (
                self.cfg.training.checkpoint_every > 0
                and self.global_step % self.cfg.training.checkpoint_every == 0
            ):
                self.save_checkpoint(self.cfg.output_dir / "checkpoints" / "latest.pt")

    @torch.no_grad()
    def generate_and_fix_stage(self, factor: float, factor_index: int) -> int:
        self.model.eval()
        refs = select_stage_references(
            self.original_cameras, self.cfg.progressive.max_views_per_stage
        )
        dirs = self._stage_dirs(factor_index, factor)
        stage_dir, noisy_dir = dirs["fixed"], dirs["noisy"]
        # Reproject from the preceding stage's generated view when it exists and
        # has usable calibrated depth. The novel camera below is still built from
        # the ORIGINAL aerial camera, so altitude factors stay absolute and the
        # schedule matches runs made before this option existed -- only which
        # image GeoQuery reprojects FROM changes.
        use_prev = self.cfg.depth.reference_mode == "previous_stage" and bool(self._prev_stage)
        generated: list[tuple[CameraRecord, CameraRecord]] = []
        accepted = 0
        reused_prev = 0
        next_id = max(r.image_id for r in self.dataset.records) + 1
        for ref in tqdm(refs, desc=f"AeroFix {factor:.0%}"):
            novel = make_progressive_camera(
                ref, factor, self.cfg.progressive.trajectory_mode, self.scene.scene_center
            )
            novel_device = replace(novel, c2w=novel.c2w.to(self.device), K=novel.K.to(self.device))
            renders, alphas, _ = self.model.render(
                novel_device,
                packed=self.cfg.training.packed,
                near_plane=self.cfg.training.near_plane,
                far_plane=self.cfg.training.far_plane,
                render_mode="RGB+ED" if self.cfg.depth.enabled else "RGB",
            )
            noisy = renders[0, ..., :3].clamp(0, 1).permute(2, 0, 1).cpu()
            if self.cfg.depth.enabled:
                # Persist the GS-rendered depth/alpha for this exact camera; it is
                # the metric anchor the DA3 fit needs, and the model changes every
                # stage so it cannot be recomputed later.
                stem = Path(novel.name).stem
                dirs["gs_depth"].mkdir(parents=True, exist_ok=True)
                np.save(dirs["gs_depth"] / f"{stem}.npy", renders[0, ..., 3].cpu().numpy())
                np.save(dirs["gs_depth"] / f"{stem}.alpha.npy", alphas[0, ..., 0].cpu().numpy())
            # Persist the raw GS render that AeroFix/GeoQuery is about to be
            # handed, under noisy_views/<stage>/ mirroring fixed_views/<stage>/.
            # Written before the quality gate below, so a rejected view still
            # leaves a record of what the model actually drew -- comparing this
            # against the corrected output is the fastest way to tell a
            # collapsed scene apart from a bad correction.
            _save_chw_rgb(noisy, noisy_dir / novel.name)
            if ref.image_path is None:
                raise ValueError(f"Reference camera {ref.name} has no image")
            # Fall back to the real aerial view whenever the previous stage has
            # nothing usable here -- a calibration that failed closed, or a view
            # that was rejected upstream. Degrading to the old behaviour for one
            # view is far better than dropping it.
            prev = self._prev_stage.get(ref.name) if use_prev else None
            if prev is not None:
                ref_camera, ref_image = prev.camera, prev.image_path
                depth = torch.from_numpy(np.load(prev.depth_path).astype(np.float32))
                reused_prev += 1
            else:
                ref_camera, ref_image = ref, ref.image_path
                depth = _load_optional_depth(self.cfg.reference_depth_dir, ref.name)
            reference = load_rgb(ref_image).permute(2, 0, 1)
            result = self.aerofix.fix(
                AeroFixInput(
                    noisy_rgb=noisy,
                    reference_rgb=reference,
                    novel_camera=novel,
                    reference_camera=ref_camera,
                    reference_depth=depth,
                    metadata={
                        "factor": factor,
                        "factor_index": factor_index,
                        # Camera translations in novel_camera/ref camera are in
                        # the scene-normalized frame used for gsplat training.
                        # Bridges that need real-world-scale poses (e.g. AeroFix's
                        # epipolar/Pluecker geometry) must invert this transform.
                        "normalization_transform": self.scene.normalization_transform,
                    },
                )
            )
            quality = float(result.quality)
            if not accept_refined_view(quality, self.cfg.progressive.min_aerofix_quality):
                continue
            out_path = stage_dir / novel.name
            _save_chw_rgb(result.fixed_rgb, out_path)
            weight = quality * self.cfg.progressive.synthetic_loss_weight
            fixed = replace(
                novel,
                image_path=out_path,
                image_id=next_id,
                source="fixed",
                weight=weight,
                appearance_id=ref.appearance_id,
            )
            # Train on this stage's AeroFix-corrected view instead of the
            # pristine original or any earlier stage's correction for the same
            # camera -- never both at once. AeroFix reference lookups for
            # later stages are unaffected: they read ref.image_path from
            # self.original_cameras, which this dataset mutation never touches.
            self.dataset.drop_reference(ref.name)
            self.dataset.append_fixed(fixed)
            generated.append((ref, fixed))
            next_id += 1
            accepted += 1
        if use_prev:
            print(f"  reprojected {reused_prev}/{len(refs)} views from the previous stage")
        if self.cfg.depth.enabled:
            self._build_stage_depths(dirs, generated)
        self.model.train()
        return accepted

    def _stage_dirs(self, factor_index: int, factor: float) -> dict[str, Path]:
        name = f"stage_{factor_index:02d}_{int(round(factor * 100)):02d}"
        root = self.cfg.output_dir
        return {
            "name": name,
            "fixed": root / "fixed_views" / name,
            "noisy": root / "noisy_views" / name,
            "gs_depth": root / "gs_depth" / name,
            "da3": root / "da3_depth" / name,
            "calib": root / "depths_calibrated" / name,
        }

    def _run_da3(self, images_dir: Path, output_dir: Path) -> None:
        """Shell out to the DA3 interpreter for a whole stage at once.

        DA3 pins numpy<2 and gsplat needs numpy>=2, so this cannot be an import
        (see DepthConfig). Batching per stage keeps model-load cost to one per
        stage rather than one per view.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_da3_depth.py"
        cmd = [
            str(self.cfg.depth.da3_python), str(script),
            "--images", str(images_dir), "--output", str(output_dir),
            "--model", self.cfg.depth.da3_model,
            "--process-res", str(self.cfg.depth.da3_process_res),
        ]
        # DA3 loads its own model into the SAME GPU while this process still holds
        # the Gaussians and GeoQuery's 866M-parameter UNet. At native resolution
        # that is ~21GB of 24GB, leaving no room for DA3's 3-4GB. Returning the
        # caching allocator's unused blocks to the driver first is what makes the
        # subprocess fit; without it the child OOMs and the stage is lost.
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 2**30
        print(f"  running DA3 on {images_dir.name} ... ({free_gb:.1f}GB free on device)", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
            raise RuntimeError(f"DA3 inference failed (exit {proc.returncode}) for {images_dir}")

    def _build_stage_depths(
        self, dirs: dict[str, Path], generated: list[tuple[CameraRecord, CameraRecord]]
    ) -> None:
        """DA3 + GS-anchored calibration over this stage's generated views.

        Populates ``self._prev_stage`` so the NEXT stage can reproject from these
        views. Views whose calibration fails closed are simply absent from that
        map, which makes the next stage fall back to the real aerial reference.
        """
        self._prev_stage = {}
        if not generated:
            return
        use_da3 = self.cfg.depth.source == "da3_calibrated"
        if use_da3:
            self._run_da3(dirs["fixed"], dirs["da3"])
        dirs["calib"].mkdir(parents=True, exist_ok=True)
        norm_scale = float(self.scene.normalization_transform[0, 0])

        stats, failures = [], {}
        for ref, fixed in generated:
            stem = Path(fixed.name).stem
            gs_path = dirs["gs_depth"] / f"{stem}.npy"
            if not gs_path.exists() or (use_da3 and not (dirs["da3"] / f"{stem}.npy").exists()):
                failures["missing input"] = failures.get("missing input", 0) + 1
                continue
            gs_depth = np.load(gs_path)
            gs_alpha = np.load(dirs["gs_depth"] / f"{stem}.alpha.npy")

            if use_da3:
                result = calibrate_relative_depth(
                    _resize_map(np.load(dirs["da3"] / f"{stem}.npy"), gs_depth.shape),
                    gs_depth,
                    gs_alpha=gs_alpha,
                    depth_scale=norm_scale,
                    min_alpha=self.cfg.depth.min_alpha,
                    max_relative_error=self.cfg.depth.max_relative_error,
                )
                if not result.valid:
                    # Bucket by cause, not by the numeric value in the message,
                    # or every rejected view becomes its own summary key.
                    key = result.reason.split(":")[0].split(" (")[0]
                    for marker in ("median relative error", "valid pixels", "trimmed to"):
                        if marker in result.reason:
                            key = marker
                            break
                    failures[key] = failures.get(key, 0) + 1
                    continue
                depth, quality = result.depth, result.relative_error
            else:
                # The rasterizer's depth is already metric in the normalized scene
                # frame, so the only conversion needed is the frame scale -- there
                # is no fit, hence nothing that can fail to converge. The one real
                # failure mode left is a camera that renders almost nothing, whose
                # depth would be meaningless as a reprojection source.
                coverage = float((gs_alpha >= self.cfg.depth.min_alpha).mean())
                if coverage < self.cfg.depth.min_coverage:
                    failures["insufficient render coverage"] = (
                        failures.get("insufficient render coverage", 0) + 1
                    )
                    continue
                depth, quality = (gs_depth / norm_scale).astype(np.float32), 1.0 - coverage

            out = dirs["calib"] / f"{stem}.npy"
            np.save(out, depth)
            self._prev_stage[ref.name] = StageReference(fixed, fixed.image_path, out)
            stats.append(quality)

        summary = {
            "stage": dirs["name"],
            "source": self.cfg.depth.source,
            "generated": len(generated),
            "calibrated": len(stats),
            "median_fit_error": float(np.median(stats)) if stats else None,
            "failures": failures,
        }
        (dirs["calib"] / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
        med = f"{summary['median_fit_error']:.4f}" if stats else "n/a"
        metric = "median fit error" if use_da3 else "median uncovered fraction"
        print(f"  depth for {len(stats)}/{len(generated)} views "
              f"[{self.cfg.depth.source}] ({metric} {med})")
        if failures:
            print(f"  calibration failures: {failures}")

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic = [camera_to_dict(c) for c in self.dataset.records if c.source == "fixed"]
        payload = {
            "model": self.model.state_dict(),
            "splat_optimizers": {k: v.state_dict() for k, v in self.splat_optimizers.items()},
            "module_optimizer": self.module_optimizer.state_dict(),
            "means_scheduler": self.means_scheduler.state_dict(),
            "strategy_state": self.strategy_state,
            "global_step": self.global_step,
            "completed_factors": self.completed_factors,
            "phase": self.phase,
            "active_factor_index": self.active_factor_index,
            "phase_end_global_step": self.phase_end_global_step,
            "synthetic_cameras": synthetic,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["model"]
        splat_tensors = {
            key.split("splats.", 1)[1]: value.to(self.device)
            for key, value in state.items()
            if key.startswith("splats.")
        }
        self.model.splats = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(value.clone()) for name, value in splat_tensors.items()}
        )
        self.model.load_state_dict(state, strict=True)

        self.dataset = ProgressiveDataset(self.original_cameras)
        for data in ckpt.get("synthetic_cameras", []):
            record = camera_from_dict(data)
            if record.image_path is None or not record.image_path.exists():
                raise FileNotFoundError(f"Checkpoint synthetic image missing: {record.image_path}")
            if record.reference_name is not None:
                self.dataset.drop_reference(record.reference_name)
            self.dataset.append_fixed(record)

        self._build_training_state()
        for key, state_dict in ckpt.get("splat_optimizers", {}).items():
            if key in self.splat_optimizers:
                self.splat_optimizers[key].load_state_dict(state_dict)
        if "module_optimizer" in ckpt:
            self.module_optimizer.load_state_dict(ckpt["module_optimizer"])
        if "means_scheduler" in ckpt:
            self.means_scheduler.load_state_dict(ckpt["means_scheduler"])
        if "strategy_state" in ckpt:
            self.strategy_state = ckpt["strategy_state"]

        self.global_step = int(ckpt.get("global_step", 0))
        self.completed_factors = int(ckpt.get("completed_factors", 0))
        self.phase = ckpt.get("phase", "initial")
        self.active_factor_index = ckpt.get("active_factor_index")
        self.phase_end_global_step = int(
            ckpt.get("phase_end_global_step", self.cfg.progressive.initial_steps)
        )
        self._restore_prev_stage()

    def _restore_prev_stage(self) -> None:
        """Rebuild the previous stage's reference map after a --resume.

        The map is derived from files on disk rather than stored in the
        checkpoint, so a resume picks up whatever calibrated depths the earlier
        process already wrote. Any view missing one falls back to its real aerial
        reference, exactly as during a fresh run.
        """
        self._prev_stage = {}
        if not self.cfg.depth.enabled or self.cfg.depth.reference_mode != "previous_stage":
            return
        # Which stage's calibrated depths feed the NEXT generation?
        #   phase="stage": we stopped part-way through stage N's TRAINING, so
        #     stage N's views and depths already exist and stage N+1 must use them.
        #     completed_factors is still N here, so completed_factors-1 would point
        #     one stage too far back and leave _prev_stage empty -- which silently
        #     downgrades every later stage to the original-aerial reference, i.e.
        #     turns a previous_stage run into a baseline run without any error.
        #   phase="idle": the last completed stage is the source.
        if self.phase == "stage" and self.active_factor_index is not None:
            idx = self.active_factor_index
        else:
            idx = self.completed_factors - 1
        if idx < 0:
            return
        factors = self.cfg.progressive.altitude_factors
        if idx >= len(factors):
            return
        dirs = self._stage_dirs(idx, factors[idx])
        by_name = {c.name: c for c in self.original_cameras}
        for rec in self.dataset.records:
            if rec.source != "fixed" or rec.reference_name not in by_name:
                continue
            depth_path = dirs["calib"] / f"{Path(rec.name).stem}.npy"
            if depth_path.exists() and rec.image_path is not None and rec.image_path.exists():
                self._prev_stage[rec.reference_name] = StageReference(rec, rec.image_path, depth_path)
        if self._prev_stage:
            print(f"restored {len(self._prev_stage)} previous-stage references from {dirs['calib']}")

    def _find_stage_camera(self, factor: float) -> Optional[CameraRecord]:
        """Find a fixed-view camera from this altitude stage, to use as the
        reference for a gated PLY export. Looked up from the live dataset
        (rather than threaded through as a return value) so this also works
        after a --resume, where generate_and_fix_stage() is skipped because
        the stage's fixed views were already generated before the restart.
        """
        for rec in self.dataset.records:
            if rec.source == "fixed" and rec.stage_factor == factor:
                return rec
        return None

    def _export_stage_ply(self, name: str, camera: Optional[CameraRecord] = None) -> None:
        plys_dir = self.cfg.output_dir / "plys"
        raw_path = export_ply(self.model, plys_dir / f"{name}.ply")
        print(f"Exported Gaussians (raw): {raw_path}")
        if camera is not None:
            cam = replace(camera, c2w=camera.c2w.to(self.device), K=camera.K.to(self.device))
            gated_path = export_ply(self.model, plys_dir / f"{name}_gated.ply", camera=cam)
            print(f"Exported Gaussians (gated @ {camera.name}): {gated_path}")

    def run(self) -> None:
        pcfg = self.cfg.progressive
        if self.phase == "initial":
            remaining = max(self.phase_end_global_step - self.global_step, 0)
            self.train_steps(remaining, "Initial aerial training")
            self.phase = "idle"
            self.phase_end_global_step = self.global_step
            self.save_checkpoint(self.cfg.output_dir / "checkpoints" / "initial.pt")
            self._export_stage_ply("initial")

        for idx in range(self.completed_factors, len(pcfg.altitude_factors)):
            factor = pcfg.altitude_factors[idx]
            if not (self.phase == "stage" and self.active_factor_index == idx):
                accepted = self.generate_and_fix_stage(factor, idx)
                print(f"Stage {idx} factor={factor:.2f}: accepted {accepted} fixed views")
                self.phase = "stage"
                self.active_factor_index = idx
                self.phase_end_global_step = self.global_step + pcfg.stage_steps
            remaining = max(self.phase_end_global_step - self.global_step, 0)
            self.train_steps(remaining, f"Stage {idx} @ {factor:.0%}")
            self.completed_factors = idx + 1
            self.phase = "idle"
            self.active_factor_index = None
            self.phase_end_global_step = self.global_step
            stage_name = f"stage_{idx:02d}_{int(round(factor*100)):02d}"
            self.save_checkpoint(self.cfg.output_dir / "checkpoints" / f"{stage_name}.pt")
            self._export_stage_ply(stage_name, camera=self._find_stage_camera(factor))

        self.save_checkpoint(self.cfg.output_dir / "checkpoints" / "final.pt")
        last_factor = pcfg.altitude_factors[-1] if pcfg.altitude_factors else None
        self._export_stage_ply(
            "final", camera=self._find_stage_camera(last_factor) if last_factor is not None else None
        )
