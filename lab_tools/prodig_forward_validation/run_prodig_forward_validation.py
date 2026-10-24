#!/usr/bin/env python3
"""Run *only* ProDiG's first altitude-descent stage and export a viewable Gaussian ply.

Purpose: validate that a Gaussian model produced from stage-0's aeroFix-"fixed"
renders is good enough to descend further, before committing to the full multi-stage
run. It does not run the whole progressive pipeline -- it deliberately stops after
one stage:
  1. (Re)generate a ProDiG scene manifest from a COLMAP reconstruction, unless one is
     supplied directly with --manifest.
  2. Render a paper-faithful pipeline config restricted to a single progressive stage
     at --stage-altitude-scale (default 0.9, the paper's first stage): render pseudo
     views at that altitude from the starting Gaussian model, correct them with
     aeroFix, then retrain the Gaussian model on {real aerial photos + those
     corrected pseudo views}.
  3. Run `anchor-prodig validate`, then (unless --validate-only) `anchor-prodig run`.
  4. Report the path to `<output-dir>/final_gs.ply` -- the resulting Gaussian model,
     in standard 3DGS ply format, viewable directly in SuperSplat or similar. Open it
     and eyeball the quality before deciding whether to proceed to lower altitudes.

Two defaults intentionally deviate from the checked-in template because they were
found to be outright bugs during validation, not tunable choices:
  - `progressive.terrain_relative_clearance` defaults to false. When true, it snaps
    every descended camera to roughly the same absolute height regardless of the
    scene's own coordinate scale, which produces solid-black renders on scenes whose
    coordinate system isn't in real-world meters. It is also not part of ProDiG's
    published method. Only pass --terrain-relative-clearance if you specifically want
    to reproduce that behavior.
  - `gs.densification.source_aware` / `max_gaussians` are enabled by default with a
    conservative cap, since unbounded densification reliably runs a 24GB GPU out of
    memory partway through the first training phase. Raise --max-gaussians if your
    GPU has more headroom, or pass --no-gaussian-budget to disable the cap entirely.
  - `progressive.adaptive_covisibility_subdivision` is off, so exactly one stage runs
    at exactly --stage-altitude-scale -- no extra stage gets silently inserted.

Example:
    python scripts/run_prodig_forward_validation.py \\
        --dataset-dir /path/to/park \\
        --initial-ply /path/to/park/point_cloud.ply \\
        --aerofix-checkpoint /path/to/checkpoints/aerofix_checkpoints/model_10495.pkl \\
        --official-source-dir third_party/ProDiG/aeroFix/src \\
        --output-dir /path/to/outputs/park_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only ProDiG's stage-0 altitude descent and export a "
        "viewable final_gs.ply, to validate the model before descending further.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Root folder containing images/ and sparse/0/ (a COLMAP reconstruction).",
    )
    dataset.add_argument(
        "--images-dir", type=Path, default=None,
        help="Image directory. Defaults to <dataset-dir>/images.",
    )
    dataset.add_argument(
        "--colmap-model", type=Path, default=None,
        help="COLMAP sparse model directory. Defaults to <dataset-dir>/sparse/0.",
    )
    dataset.add_argument(
        "--manifest", type=Path, default=None,
        help="Existing scene.json to use as-is. If omitted, one is (re)generated from "
        "--colmap-model + --images-dir + --pcd-prior-cloud into <output-dir>/scene.json.",
    )
    dataset.add_argument(
        "--pcd-prior-cloud", type=Path, default=None,
        help="Point cloud used both as the manifest's point cloud and for pcd_prior "
        "supervision. Defaults to <colmap-model>/points3D.ply.",
    )
    dataset.add_argument(
        "--source", choices=["aerial", "real"], default="aerial",
        help="Camera source label written into the manifest.",
    )

    model = parser.add_argument_group("gaussian model")
    model.add_argument(
        "--initial-ply", type=Path, required=True,
        help="Gaussian .ply to use as the starting model (explicit_ply init backend). "
        "If it doesn't exist yet, it will be produced by training graphdeco on "
        "--dataset-dir first (requires --graphdeco-source-dir).",
    )
    model.add_argument("--sh-degree", type=int, default=3)
    model.add_argument("--residual-seed-count", type=int, default=50_000)
    model.add_argument(
        "--initial-steps", type=int, default=7000,
        help="Steps of direct refinement against real aerial photos before the "
        "progressive/aeroFix loop starts. Paper default is 7000; set 0 to skip this "
        "phase entirely if --initial-ply already renders accurately (verify this "
        "first, e.g. by rendering it from a few manifest poses and comparing against "
        "the real photos -- don't assume it).",
    )
    model.add_argument("--stage-steps", type=int, default=2000)
    model.add_argument(
        "--stage-altitude-scale", type=float, default=0.9,
        help="The single altitude-descent scale factor to validate (paper's stage-0 "
        "default is 0.9). Only this one stage runs; the pipeline stops there.",
    )
    # CHANGE (added after the initial version of this script): --views-per-stage, wired
    # through to progressive.views_per_stage below in build_config(). Lets a fast
    # diagnostic pass generate pseudo views for only a spatially-spread subset of the
    # aerial views, instead of every one, since view generation (one render + one
    # aeroFix diffusion call per view) is the dominant cost of a run. To restore the
    # original behavior (always use every aerial view, no subsampling), either omit
    # this flag -- its default is already None, i.e. unchanged -- or delete this
    # argument and remove the `"views_per_stage": args.views_per_stage,` line in
    # build_config()'s progressive dict.
    model.add_argument(
        "--views-per-stage", type=int, default=None,
        help="Cap the number of pseudo views generated for the stage (each one is a "
        "render + aeroFix diffusion call, the dominant cost of a run). Uses a "
        "spatially spread-out subset of the aerial views (spatial_view_sampling), "
        "not just the first N. Default: use every aerial view. Set this to e.g. 100 "
        "for a fast diagnostic pass instead of building a separate smaller dataset.",
    )
    model.add_argument(
        "--max-gaussians", type=int, default=8_500_000,
        help="Gaussian-count budget cap. Growth is masked before it happens once "
        "the count nears this ceiling, instead of running out of GPU memory.",
    )
    model.add_argument("--budget-margin", type=int, default=250_000)
    model.add_argument(
        "--no-gaussian-budget", action="store_true",
        help="Disable the Gaussian-count budget cap entirely (unbounded densification).",
    )
    model.add_argument(
        "--terrain-relative-clearance", action="store_true",
        help="Enable the terrain-clearance camera-pose clamp. Off by default -- see "
        "module docstring for why.",
    )

    graphdeco = parser.add_argument_group(
        "graphdeco fallback (only used if --initial-ply doesn't exist yet)"
    )
    graphdeco.add_argument("--graphdeco-source-dir", type=Path, default=None)
    graphdeco.add_argument(
        "--graphdeco-python", type=str, default=None,
        help="Python executable used to run graphdeco's train.py.",
    )
    graphdeco.add_argument("--graphdeco-iterations", type=int, default=7000)

    diffusion = parser.add_argument_group("aeroFix diffusion")
    diffusion.add_argument("--aerofix-checkpoint", type=Path, required=True)
    diffusion.add_argument(
        "--official-source-dir", type=Path, required=True,
        help="Path to the public ProDiG aeroFix source (e.g. third_party/ProDiG/aeroFix/src).",
    )
    diffusion.add_argument("--checkpoint-format", default="prodig_pkl")
    diffusion.add_argument(
        "--allow-unsafe-pickle", action="store_true",
        help="Only pass this for checkpoints you trust; enables unsafe pickle loading.",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--image-resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    run.add_argument(
        "--anchor-prodig-python", type=str, default=sys.executable,
        help="Python executable with anchor_prodig installed. Defaults to the "
        "interpreter running this script.",
    )
    run.add_argument(
        "--config-out", type=Path, default=None,
        help="Where to write the generated config. Defaults to <output-dir>/generated_config.yaml.",
    )
    run.add_argument(
        "--validate-only", action="store_true",
        help="Only run `anchor-prodig validate`, skip the actual pipeline run.",
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Only generate and print the config; don't validate or run anything.",
    )
    run.add_argument(
        "--overwrite-manifest", action="store_true",
        help="Regenerate the manifest even if <output-dir>/scene.json already exists.",
    )

    return parser.parse_args(argv)


def resolve_manifest(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        manifest = args.manifest.expanduser().resolve()
        if not manifest.exists():
            raise FileNotFoundError(f"--manifest does not exist: {manifest}")
        return manifest

    images_dir = (args.images_dir or args.dataset_dir / "images").expanduser().resolve()
    colmap_model = (args.colmap_model or args.dataset_dir / "sparse" / "0").expanduser().resolve()
    pcd_prior_cloud = (
        args.pcd_prior_cloud or colmap_model / "points3D.ply"
    ).expanduser().resolve()
    manifest_out = (args.output_dir / "scene.json").expanduser().resolve()

    if manifest_out.exists() and not args.overwrite_manifest:
        print(f"Reusing existing manifest: {manifest_out}")
        return manifest_out

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.anchor_prodig_python,
        str(REPO_ROOT / "scripts" / "colmap_to_prodig_manifest.py"),
        "--model", str(colmap_model),
        "--images", str(images_dir),
        "--point-cloud", str(pcd_prior_cloud),
        "--output", str(manifest_out),
        "--source", args.source,
        "--overwrite",
    ]
    print("Generating manifest:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return manifest_out


def build_config(args: argparse.Namespace, manifest: Path) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    initial_ply = args.initial_ply.expanduser().resolve()
    pcd_prior_cloud = (
        args.pcd_prior_cloud
        or (args.colmap_model or args.dataset_dir / "sparse" / "0") / "points3D.ply"
    ).expanduser().resolve()

    densification: dict = {
        "enabled": True,
        "refine_start_iter": 500,
        "refine_stop_iter": 15000,
        "refine_every": 100,
        "reset_every": 3000,
    }
    if not args.no_gaussian_budget:
        densification.update(
            source_aware=True,
            max_gaussians=args.max_gaussians,
            budget_margin=args.budget_margin,
            on_budget_exceeded="warn_stop",
        )

    config: dict = {
        "mode": "baseline",
        "device": {
            "device": args.device,
            "seed": args.seed,
            "amp": True,
            "amp_dtype": "bfloat16",
            "compile_modules": False,
            "data_workers": 4,
            "pin_memory": True,
        },
        "data": {
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            "image_resize": list(args.image_resize),
            "cache_images": True,
            "cache_capacity": 128,
            "up_axis": "z",
        },
        "renderer": {
            "backend": "gsplat",
            "packed": True,
            "sparse_grad": False,
            "antialiased": False,
            "render_depth": True,
        },
        "initial_gs": {
            "enabled": True,
            "backend": "graphdeco",
            "run_if_missing": True,
            "graphdeco_source_dir": str(args.graphdeco_source_dir)
            if args.graphdeco_source_dir
            else str(REPO_ROOT / "third_party" / "gaussian-splatting"),
            "python_executable": args.graphdeco_python or args.anchor_prodig_python,
            "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
            "output_dir": str(output_dir / "initial_gs"),
            "output_ply": str(
                output_dir / "initial_gs" / "point_cloud" / "iteration_7000" / "point_cloud.ply"
            ),
            "iterations": args.graphdeco_iterations,
        },
        "gs": {
            "initialization_backend": "explicit_ply",
            "initial_ply": str(initial_ply),
            "imported_progressive_policy": "residual_geometry",
            "residual_seed_count": args.residual_seed_count,
            "sh_degree": args.sh_degree,
            "initial_steps": args.initial_steps,
            "stage_steps": args.stage_steps,
            "correction_steps": 1000,
            "refresh_steps": 500,
            "batch_size": 1,
            "appearance_enabled": False,
            "freeze_topology_after_initial": False,
            "distance_adaptive": {
                "enabled": True,
                "feature_dim": 8,
                "hidden_dim": 32,
                "mode": "bounded_residual",
            },
            "densification": densification,
        },
        "diffusion": {
            "backend": "aerofix",
            "checkpoint": str(args.aerofix_checkpoint.expanduser().resolve()),
            "checkpoint_format": args.checkpoint_format,
            "aerofix_allow_unsafe_pickle": bool(args.allow_unsafe_pickle),
            "official_source_dir": str(args.official_source_dir.expanduser().resolve()),
            "cache_dir": str(output_dir / "cache" / "diffusion"),
            "prompt": "remove degradation",
            "timestep": 199,
            "inference_steps": 1,
            "guidance_scale": 0.0,
            "dtype": "bfloat16",
            "confidence_scale": 1.0,
        },
        "progressive": {
            "altitude_scales": [args.stage_altitude_scale],
            # CHANGE: wires --views-per-stage through (see its add_argument() above
            # for the full note). Delete this line to restore the original behavior.
            "views_per_stage": args.views_per_stage,
            "strategy": "stochastic_scaled_forward",
            "pseudo_base_weight": 0.35,
            "quality_keep_threshold": 0.35,
            "quality_drop_threshold": 0.1,
            "tangent_forward_shift": True,
            "terrain_relative_clearance": bool(args.terrain_relative_clearance),
            "minimum_clearance_m": 1.0,
            "terrain_neighbors": 32,
            "spatial_view_sampling": True,
            # Off so exactly one stage runs at exactly --stage-altitude-scale, rather
            # than covisibility heuristics silently inserting another one.
            "adaptive_covisibility_subdivision": False,
            "minimum_adjacent_covisibility": 0.2,
            "covisibility_refinement_rounds": 2,
            "visibility_sample_count": 12000,
            "reference_source": "nearest_previous",
        },
        "pcd_prior": {
            "enabled": True,
            "point_cloud": str(pcd_prior_cloud),
            "surface_weight": 1.0,
            "depth_weight": 0.5,
            "normal_weight": 0.1,
        },
        "replay": {
            "enabled": True,
            "stale_refresh_enabled": False,
            "root": str(output_dir / "replay"),
        },
        "pose_refine": {"enabled": False},
        "graph": {
            "enabled": True,
            "root": str(output_dir / "graph"),
            "flush_every": 1_000_000,
        },
        "optimization": {"backward_cycles": 1},
    }
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = resolve_manifest(args)
    config = build_config(args, manifest)

    config_out = (args.config_out or args.output_dir / "generated_config.yaml").expanduser().resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Wrote config: {config_out}")

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        return 0

    validate_cmd = [
        args.anchor_prodig_python, "-m", "anchor_prodig.cli", "validate",
        "--config", str(config_out),
    ]
    print("Validating:", " ".join(validate_cmd))
    result = subprocess.run(validate_cmd)
    if result.returncode != 0:
        print("Validation failed; not running the pipeline.", file=sys.stderr)
        return result.returncode

    if args.validate_only:
        return 0

    run_cmd = [
        args.anchor_prodig_python, "-m", "anchor_prodig.cli", "run",
        "--config", str(config_out),
    ]
    print("Running:", " ".join(run_cmd))
    result = subprocess.run(run_cmd)
    if result.returncode != 0:
        return result.returncode

    final_ply = args.output_dir / "final_gs.ply"
    if final_ply.exists():
        size_mb = final_ply.stat().st_size / (1024 * 1024)
        print(f"\nStage-0 Gaussian model written: {final_ply} ({size_mb:.1f} MB)")
        print("Open it in SuperSplat (or another 3DGS viewer) to judge whether it's")
        print("clean enough to proceed to lower altitudes.")
    else:
        print(
            f"\nWARNING: expected output {final_ply} was not created. "
            "Check the run log above for errors.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
