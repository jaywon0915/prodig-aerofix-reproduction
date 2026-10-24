#!/usr/bin/env python3
"""Measure the DA3+GS calibration acceptance rate on real GENERATED views.

scripts/validate_depth_calib.py checks accuracy on REAL aerial views, where the
GS model was trained on that exact viewpoint. Generated views are extrapolated
cameras the model never saw, so their rendered depth is weaker and the residual
gate will reject more of them. This script measures that rate before committing
to a multi-hour run -- if most views fail, `reference_mode=previous_stage`
silently degrades to the old behaviour and the run is a waste.

It reconstructs the exact novel cameras of a completed run (same
make_progressive_camera call, same factor) and renders depth from the checkpoint
that was live when that stage was generated, so the numbers reflect the real
pipeline rather than a proxy.

    env PYTHONPATH=. python \
      scripts/check_generated_view_calib.py \
        --checkpoint <run>/checkpoints/initial.pt \
        --data data/park_lowres \
        --fixed-views <run>/fixed_views/stage_00_90 \
        --factor 0.9 --num-views 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from prodig.trainer import _resize_map  # noqa: E402
from prodig.trajectory import make_progressive_camera  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fixed-views", type=Path, required=True, help="a completed run's stage dir")
    p.add_argument("--factor", type=float, required=True)
    p.add_argument("--trajectory", default="forward")
    p.add_argument("--num-views", type=int, default=40)
    p.add_argument("--da3-python", type=Path, required=True, help="python of the separate DA3 env (see run_da3_depth.py)")
    p.add_argument("--workdir", type=Path, default=Path("outputs/gen_calib_check"))
    p.add_argument("--max-rel-error", type=float, default=0.05)
    p.add_argument("--min-alpha", type=float, default=0.5)
    args = p.parse_args(argv)

    device = torch.device("cuda")
    scene = load_colmap_scene(args.data, normalize=True)
    norm_scale = float(scene.normalization_transform[0, 0])

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    splats = {k.split("splats.", 1)[1]: v for k, v in state.items() if k.startswith("splats.")}
    model = ProDiGModel.from_points(
        scene.points[:1], scene.colors[:1], num_appearances=len(scene.cameras),
        scene_scale=scene.scene_scale,
    )
    model.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in splats.items()})
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"loaded {splats['means'].shape[0]:,} Gaussians @ step {ckpt.get('global_step')}")

    # Rebuild the same novel cameras the run generated, and keep only those whose
    # fixed view actually exists (some may have been rejected upstream).
    pairs = []
    for cam in scene.cameras:
        novel = make_progressive_camera(cam, args.factor, args.trajectory, scene.scene_center)
        img = args.fixed_views / novel.name
        if img.exists():
            pairs.append((cam, novel, img))
    if not pairs:
        print(f"error: no fixed views matched in {args.fixed_views}", file=sys.stderr)
        return 1
    step = max(len(pairs) // args.num_views, 1)
    sample = pairs[::step][: args.num_views]
    print(f"{len(pairs)} generated views present; checking {len(sample)}")

    img_dir = args.workdir / "images"
    da3_dir = args.workdir / "da3"
    img_dir.mkdir(parents=True, exist_ok=True)
    for _, _, img in sample:
        target = img_dir / img.name
        if not target.exists():
            target.symlink_to(img)

    print("running DA3 ...", flush=True)
    proc = subprocess.run(
        [str(args.da3_python), str(REPO / "scripts" / "run_da3_depth.py"),
         "--images", str(img_dir), "--output", str(da3_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        return 1

    rows, accepted = [], 0
    for cam, novel, img in sample:
        stem = Path(novel.name).stem
        da3_path = da3_dir / f"{stem}.npy"
        if not da3_path.exists():
            continue
        nd = replace(novel, c2w=novel.c2w.to(device), K=novel.K.to(device))
        with torch.no_grad():
            renders, alphas, _ = model.render(nd, render_mode="RGB+ED")
        gs_depth = renders[0, ..., 3].cpu().numpy()
        gs_alpha = alphas[0, ..., 0].cpu().numpy()
        res = calibrate_relative_depth(
            _resize_map(np.load(da3_path), gs_depth.shape), gs_depth, gs_alpha=gs_alpha,
            depth_scale=norm_scale, min_alpha=args.min_alpha, max_relative_error=1e9,
        )
        ok = res.valid and res.relative_error <= args.max_rel_error
        accepted += int(ok)
        rows.append({"view": stem, "residual": res.relative_error if res.valid else None,
                     "accepted": bool(ok), "alpha_cov": float((gs_alpha >= args.min_alpha).mean())})

    good = [r["residual"] for r in rows if r["residual"] is not None]
    print(f"\n=== factor {args.factor} : {accepted}/{len(rows)} would be ACCEPTED "
          f"at threshold {args.max_rel_error} ===")
    if good:
        g = np.array(good)
        print(f"  residual vs GS depth: median={np.median(g):.4f} p25={np.percentile(g,25):.4f} "
              f"p75={np.percentile(g,75):.4f} p90={np.percentile(g,90):.4f}")
        for t in (0.05, 0.08, 0.10, 0.15, 0.20):
            print(f"    threshold {t:.2f} -> would accept {(g <= t).sum():>3}/{len(g)}")
    cov = np.array([r["alpha_cov"] for r in rows])
    print(f"  GS alpha coverage: median={np.median(cov):.3f}")
    (args.workdir / f"calib_check_{int(args.factor*100)}.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
