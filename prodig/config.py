from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    gaussian_feature_dim: int = 32
    distance_feature_dim: int = 8
    distance_hidden_dim: int = 16
    appearance_embed_dim: int = 16
    appearance_hidden_dim: int = 64
    distance_init_gate: float = 0.99
    distance_mode: str = "log_scene"
    init_opacity: float = 0.1
    init_scale: float = 1.0


@dataclass
class TrainingConfig:
    ssim_lambda: float = 0.2
    module_lr: float = 1e-3
    packed: bool = False
    near_plane: float = 0.01
    far_plane: float = 1e10
    random_background: bool = False
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3_000
    # gsplat.strategy.DefaultStrategy growth/prune thresholds. Defaults match
    # gsplat's own library defaults; raise grow_grad2d / prune_opa or increase
    # refine_every to slow down Gaussian count growth on dense point clouds.
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    checkpoint_every: int = 1_000
    seed: int = 42


@dataclass
class ProgressiveConfig:
    initial_steps: int = 7_000
    stage_steps: int = 3_000  # paper main text does not specify this; tune per dataset.
    altitude_factors: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1)
    trajectory_mode: str = "forward"
    max_views_per_stage: int = 0  # 0 = one lowered view for every original aerial camera.
    min_aerofix_quality: float = 0.0
    synthetic_loss_weight: float = 1.0

    @property
    def total_steps(self) -> int:
        return self.initial_steps + self.stage_steps * len(self.altitude_factors)


@dataclass
class DepthConfig:
    """DA3 depth on generated views, scale/offset-anchored to the GS geometry.

    A generated altitude view has no COLMAP tracks to calibrate monocular depth
    against, so the GS model's own rendered depth supplies the metric anchor
    while DA3 supplies the detail (see prodig.depth_calib). Giving generated
    views their own depth is what lets them serve as GeoQuery references, which
    `reference_mode="previous_stage"` then switches on.

    DA3 runs in a SEPARATE interpreter (`da3_python`): it pins numpy<2 while
    gsplat needs numpy>=2, so they cannot share an env and exchange .npy instead.
    """

    enabled: bool = False
    # "da3_calibrated" -> DA3 relative depth, scale/offset fitted to the GS render.
    # "gs_render"      -> the GS rendered depth itself, converted to the real frame.
    #   Measured on 40 real aerial views against the COLMAP-anchored reference, the
    #   two are within 0.13pp (2.52% vs 2.39% median error), but gs_render costs no
    #   DA3 pass (~25 min/stage) and cannot fail a fit -- so every generated view
    #   stays usable as a reference instead of ~57% being rejected at factor 0.1.
    source: str = "da3_calibrated"
    da3_python: Path | None = None
    da3_model: str = "da3-large"
    da3_process_res: int = 504
    min_alpha: float = 0.5
    # gs_render only: reject a view whose render covers less of the frame than this,
    # so a camera that sees essentially nothing never becomes a reference.
    min_coverage: float = 0.10
    max_relative_error: float = 0.05
    # "original"       -> always reproject from the real aerial view (previous behaviour)
    # "previous_stage" -> reproject from the preceding stage's generated view, which
    #                     sits far closer to the query camera. Novel-camera geometry is
    #                     unchanged either way: factors stay absolute w.r.t. the original
    #                     camera, so the altitude schedule matches earlier runs.
    reference_mode: str = "original"

    def __post_init__(self) -> None:
        if self.reference_mode not in ("original", "previous_stage"):
            raise ValueError(f"invalid reference_mode: {self.reference_mode}")
        if self.source not in ("da3_calibrated", "gs_render"):
            raise ValueError(f"invalid depth source: {self.source}")
        if self.enabled and self.source == "da3_calibrated" and self.da3_python is None:
            raise ValueError("depth source 'da3_calibrated' requires da3_python")


@dataclass
class RunConfig:
    data_dir: Path = Path("data/scene")
    output_dir: Path = Path("outputs/prodig")
    aerofix_factory: str = "identity"
    reference_depth_dir: Path | None = None
    # Start from an existing Gaussian model instead of the COLMAP point cloud.
    # When set, the initial aerial training phase is skipped entirely (see
    # ProgressiveConfig.initial_steps) and the run begins at the first descent.
    input_ply: Path | None = None
    # "world"      -> PLY is in COLMAP coordinates (external trainers)
    # "normalized" -> PLY came from this pipeline's own export_ply
    input_ply_frame: str = "world"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    progressive: ProgressiveConfig = field(default_factory=ProgressiveConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["data_dir"] = str(self.data_dir)
        out["output_dir"] = str(self.output_dir)
        if self.reference_depth_dir is not None:
            out["reference_depth_dir"] = str(self.reference_depth_dir)
        if self.input_ply is not None:
            out["input_ply"] = str(self.input_ply)
        if self.depth.da3_python is not None:
            out["depth"]["da3_python"] = str(self.depth.da3_python)
        return out
