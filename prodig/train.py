from __future__ import annotations

import argparse
from pathlib import Path

from .aerofix import load_aerofix_factory
from .colmap_scene import load_colmap_scene
from .config import DepthConfig, ModelConfig, ProgressiveConfig, RunConfig, TrainingConfig
from .trainer import ProDiGTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ProDiG reimplementation on gsplat")
    p.add_argument("--data", type=Path, required=True, help="COLMAP scene root containing images/ and sparse/0/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--aerofix-factory", default="identity", help="module:function returning an object with .fix(AeroFixInput)")
    p.add_argument("--reference-depth-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="cuda")

    p.add_argument("--initial-steps", type=int, default=None,
                   help="steps of initial aerial training before the first descent. "
                        "Default 7000, or 0 when --input-ply is given (the supplied "
                        "model is already trained, so the phase is redundant).")
    p.add_argument("--input-ply", type=Path, default=None,
                   help="start from this 3DGS PLY instead of the COLMAP point cloud, "
                        "skipping initial training and going straight to the descent.")
    p.add_argument("--input-ply-frame", choices=["world", "normalized"], default="world",
                   help="'world': PLY is in COLMAP coordinates (external trainers). "
                        "'normalized': PLY came from this pipeline's own export_ply.")
    p.add_argument("--stage-steps", type=int, default=3000)
    p.add_argument("--factors", default="0.9,0.7,0.5,0.3,0.1")
    p.add_argument("--trajectory", choices=["forward", "scaled", "scaled_forward"], default="forward")
    p.add_argument("--max-views-per-stage", type=int, default=0)
    p.add_argument("--min-aerofix-quality", type=float, default=0.0)
    p.add_argument("--synthetic-loss-weight", type=float, default=1.0)

    p.add_argument("--distance-feature-dim", type=int, default=8)
    p.add_argument("--distance-hidden-dim", type=int, default=16)
    p.add_argument("--distance-init-gate", type=float, default=0.99)
    p.add_argument("--distance-mode", choices=["raw", "scene", "log_scene"], default="log_scene")
    p.add_argument("--gaussian-feature-dim", type=int, default=32)
    p.add_argument("--appearance-embed-dim", type=int, default=16)
    p.add_argument("--appearance-hidden-dim", type=int, default=64)

    p.add_argument("--ssim-lambda", type=float, default=0.2)
    p.add_argument("--module-lr", type=float, default=1e-3)
    p.add_argument("--packed", action="store_true")
    p.add_argument("--refine-start-iter", type=int, default=500)
    p.add_argument("--refine-stop-iter", type=int, default=15000)
    p.add_argument("--refine-every", type=int, default=100,
                    help="Densify/prune every N steps. Raise to slow Gaussian count growth.")
    p.add_argument("--reset-every", type=int, default=3000)
    p.add_argument("--prune-opa", type=float, default=0.005,
                    help="Opacity below which a Gaussian is pruned. Raise to prune more aggressively.")
    p.add_argument("--grow-grad2d", type=float, default=0.0002,
                    help="Screen-space gradient threshold to grow a Gaussian. Raise to grow less.")
    p.add_argument("--grow-scale3d", type=float, default=0.01)
    p.add_argument("--grow-scale2d", type=float, default=0.05)
    p.add_argument("--prune-scale3d", type=float, default=0.1)
    p.add_argument("--prune-scale2d", type=float, default=0.15)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("depth for generated views (enables --reference-mode previous_stage)")
    g.add_argument("--depth-source", choices=["da3_calibrated", "gs_render"], default="da3_calibrated",
                   help="'da3_calibrated': DA3 relative depth with scale/offset fitted to the GS "
                        "render (needs --da3-python). 'gs_render': use the GS rendered depth "
                        "directly -- no DA3 pass, and no fit that can fail, so every generated "
                        "view stays usable as a reference.")
    g.add_argument("--depth-min-coverage", type=float, default=0.10,
                   help="gs_render only: skip views whose render covers less of the frame than this")
    g.add_argument("--da3-python", type=Path, default=None,
                   help="interpreter with depth_anything_3 installed; enables DA3 depth on "
                        "generated views. Must NOT be the anchor-prodig python (numpy conflict).")
    g.add_argument("--da3-model", default="da3-large", help="DA3 model registry key")
    g.add_argument("--da3-process-res", type=int, default=504,
                   help="DA3 processing resolution (multiple of 14)")
    g.add_argument("--depth-min-alpha", type=float, default=0.5,
                   help="minimum GS alpha for a pixel to constrain the scale/offset fit")
    g.add_argument("--depth-max-rel-error", type=float, default=0.05,
                   help="reject a calibration whose median residual vs GS depth exceeds this; "
                        "0.05 was measured to separate good from bad fits (see depth_calib)")
    g.add_argument("--reference-mode", choices=["original", "previous_stage"], default="original",
                   help="'previous_stage' reprojects from the preceding stage's generated view "
                        "(needs --da3-python); novel-camera geometry is unchanged either way")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # A supplied model is already fitted to the aerial views, so the initial
    # phase would only undo that; default it to 0 unless explicitly set.
    initial_steps = args.initial_steps
    if initial_steps is None:
        initial_steps = 0 if args.input_ply is not None else 7000
    if args.input_ply is not None and not args.input_ply.exists():
        raise SystemExit(f"--input-ply not found: {args.input_ply}")
    factors = tuple(float(x) for x in args.factors.split(",") if x.strip())
    cfg = RunConfig(
        data_dir=args.data,
        output_dir=args.output,
        aerofix_factory=args.aerofix_factory,
        reference_depth_dir=args.reference_depth_dir,
        input_ply=args.input_ply,
        input_ply_frame=args.input_ply_frame,
        model=ModelConfig(
            gaussian_feature_dim=args.gaussian_feature_dim,
            distance_feature_dim=args.distance_feature_dim,
            distance_hidden_dim=args.distance_hidden_dim,
            appearance_embed_dim=args.appearance_embed_dim,
            appearance_hidden_dim=args.appearance_hidden_dim,
            distance_init_gate=args.distance_init_gate,
            distance_mode=args.distance_mode,
        ),
        training=TrainingConfig(
            ssim_lambda=args.ssim_lambda,
            module_lr=args.module_lr,
            packed=args.packed,
            refine_start_iter=args.refine_start_iter,
            refine_stop_iter=args.refine_stop_iter,
            refine_every=args.refine_every,
            reset_every=args.reset_every,
            prune_opa=args.prune_opa,
            grow_grad2d=args.grow_grad2d,
            grow_scale3d=args.grow_scale3d,
            grow_scale2d=args.grow_scale2d,
            prune_scale3d=args.prune_scale3d,
            prune_scale2d=args.prune_scale2d,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
        ),
        progressive=ProgressiveConfig(
            initial_steps=initial_steps,
            stage_steps=args.stage_steps,
            altitude_factors=factors,
            trajectory_mode=args.trajectory,
            max_views_per_stage=args.max_views_per_stage,
            min_aerofix_quality=args.min_aerofix_quality,
            synthetic_loss_weight=args.synthetic_loss_weight,
        ),
        depth=DepthConfig(
            # gs_render needs no external interpreter, so previous-stage referencing
            # alone is enough to switch depth export on.
            enabled=args.da3_python is not None or args.reference_mode == "previous_stage",
            source=args.depth_source,
            da3_python=args.da3_python,
            da3_model=args.da3_model,
            da3_process_res=args.da3_process_res,
            min_alpha=args.depth_min_alpha,
            min_coverage=args.depth_min_coverage,
            max_relative_error=args.depth_max_rel_error,
            reference_mode=args.reference_mode,
        ),
    )
    if args.depth_source == "da3_calibrated" and args.reference_mode == "previous_stage" \
            and args.da3_python is None:
        raise SystemExit(
            "--reference-mode previous_stage with --depth-source da3_calibrated requires "
            "--da3-python (or pass --depth-source gs_render to use the GS depth directly)"
        )
    scene = load_colmap_scene(args.data, normalize=True)
    aerofix = load_aerofix_factory(args.aerofix_factory)
    if args.aerofix_factory == "identity":
        print("WARNING: IdentityAeroFix is for pipeline smoke tests only; it does not refine novel views.")
    trainer = ProDiGTrainer(scene, cfg, aerofix, device=args.device)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
