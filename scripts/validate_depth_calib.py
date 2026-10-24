#!/usr/bin/env python3
"""Validate GS-anchored DA3 calibration against the existing COLMAP-anchored depths.

Why this exists
---------------
The supervisor's change calibrates DA3 depth on *generated* views using the GS
model's own rendered depth as the metric anchor, because a synthesized viewpoint
has no COLMAP tracks. There is no ground truth for a generated view -- but there
IS for the real aerial views: `data/park/depths_metric` already holds DA3
depth calibrated the COLMAP way (scripts/apply_depth_params_correction.py).

So run the NEW method on those same real views and compare against the OLD one.
Agreement means the GS anchor is a sound substitute for COLMAP tracks, which is
the entire premise -- and it can be checked before a single generated view is
produced, rather than discovering a bad premise after a multi-hour training run.

Run in the anchor-prodig env (needs gsplat + pycolmap):

    env PYTHONPATH=. python \
        scripts/validate_depth_calib.py \
          --checkpoint outputs/gsplat_geoquery_sklearn/checkpoints/final.pt \
          --data data/park_lowres \
          --da3-dir <output of run_da3_depth.py> \
          --reference-depth-dir data/park/depths_metric \
          --num-views 40
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prodig.colmap_scene import load_colmap_scene  # noqa: E402
from prodig.depth_calib import calibrate_relative_depth  # noqa: E402
from prodig.model import ProDiGModel  # noqa: E402


def resize_to(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a 2D map to (H, W) using torch (no cv2/PIL dependency)."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    out = torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def load_model(checkpoint: Path, scene, device: torch.device) -> ProDiGModel:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    splats = {k.split("splats.", 1)[1]: v for k, v in state.items() if k.startswith("splats.")}
    n = splats["means"].shape[0]
    model = ProDiGModel.from_points(
        scene.points[:1], scene.colors[:1], num_appearances=len(scene.cameras),
        scene_scale=scene.scene_scale,
    )
    model.splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.clone()) for k, v in splats.items()}
    )
    model.load_state_dict(state, strict=True)
    print(f"loaded {n:,} Gaussians from {checkpoint.name}")
    return model.to(device).eval()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--da3-dir", type=Path, required=True, help="DA3 .npy maps from run_da3_depth.py")
    p.add_argument("--reference-depth-dir", type=Path, required=True, help="COLMAP-anchored .npy depths")
    p.add_argument("--num-views", type=int, default=40)
    p.add_argument("--min-alpha", type=float, default=0.5)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda")
    scene = load_colmap_scene(args.data, normalize=True)
    norm_scale = float(scene.normalization_transform[0, 0])
    print(f"normalization scale = {norm_scale:.6f}  (real = normalized / scale)")
    model = load_model(args.checkpoint, scene, device)

    # Only views that have BOTH a DA3 map and a COLMAP-anchored reference depth
    # can be compared; DA3 covers 975 but the COLMAP calibration only 855.
    usable = [
        c for c in scene.cameras
        if (args.da3_dir / f"{Path(c.name).stem}.npy").exists()
        and (args.reference_depth_dir / f"{Path(c.name).stem}.npy").exists()
    ]
    if not usable:
        print("error: no views have both DA3 and reference depth", file=sys.stderr)
        return 1
    step = max(len(usable) // args.num_views, 1)
    sample = usable[::step][: args.num_views]
    print(f"{len(usable)} comparable views; evaluating {len(sample)}\n")

    rows = []
    for cam in sample:
        stem = Path(cam.name).stem
        cam_d = replace(cam, c2w=cam.c2w.to(device), K=cam.K.to(device))
        with torch.no_grad():
            renders, alphas, _ = model.render(cam_d, render_mode="RGB+ED")
        gs_depth = renders[0, ..., 3].cpu().numpy()
        gs_alpha = alphas[0, ..., 0].cpu().numpy()
        hw = gs_depth.shape

        da3 = resize_to(np.load(args.da3_dir / f"{stem}.npy"), hw)
        res = calibrate_relative_depth(
            da3, gs_depth, gs_alpha=gs_alpha, depth_scale=norm_scale,
            min_alpha=args.min_alpha, max_relative_error=1e9,  # report, don't gate
        )
        if not res.valid:
            rows.append({"view": stem, "valid": False, "reason": res.reason})
            print(f"{stem:16} FAILED: {res.reason}")
            continue

        ref = resize_to(np.load(args.reference_depth_dir / f"{stem}.npy"), hw)
        m = np.isfinite(res.depth) & np.isfinite(ref) & (ref > 1e-6) & (gs_alpha >= args.min_alpha)
        if m.sum() < 100:
            rows.append({"view": stem, "valid": False, "reason": "no overlap with reference"})
            continue
        rel = np.abs(res.depth[m] - ref[m]) / ref[m]
        corr = float(np.corrcoef(res.depth[m], ref[m])[0, 1])
        row = {
            "view": stem, "valid": True,
            "scale": res.scale, "offset": res.offset,
            "fit_rel_err_vs_gs": res.relative_error,
            "median_rel_err_vs_colmap": float(np.median(rel)),
            "corr_vs_colmap": corr,
            "ours_median_depth": float(np.median(res.depth[m])),
            "colmap_median_depth": float(np.median(ref[m])),
            "pixels": int(m.sum()),
        }
        rows.append(row)
        print(f"{stem:16} relerr_vs_colmap={row['median_rel_err_vs_colmap']:.4f} "
              f"corr={corr:.4f}  ours={row['ours_median_depth']:.2f} colmap={row['colmap_median_depth']:.2f}")

    ok = [r for r in rows if r.get("valid")]
    print(f"\n=== {len(ok)}/{len(rows)} views calibrated ===")
    if ok:
        for key, label in [
            ("median_rel_err_vs_colmap", "median relative error vs COLMAP depth"),
            ("corr_vs_colmap", "correlation vs COLMAP depth"),
            ("fit_rel_err_vs_gs", "fit residual vs GS depth"),
        ]:
            v = np.array([r[key] for r in ok])
            print(f"  {label:38} median={np.median(v):.4f}  mean={v.mean():.4f}  "
                  f"p10={np.percentile(v,10):.4f}  p90={np.percentile(v,90):.4f}")
    if args.out_json:
        args.out_json.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
