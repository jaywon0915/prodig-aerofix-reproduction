#!/usr/bin/env python3
"""PSNR/SSIM of each run's Gaussian model against the REAL aerial photographs.

Why against the real photos
---------------------------
A generated altitude view has no ground truth -- it is a synthesized viewpoint
that was never photographed -- so image metrics on the stage outputs cannot be
scored against anything. What every run *can* be scored on is the thing the
pipeline is actually producing: a Gaussian map. Rendering each run's model at the
original COLMAP camera poses and comparing to the real photographs is directly
comparable across runs, because all three consumed the identical real images.

Read it as fidelity-to-reality, not as a held-out test score: the real views were
trained on during the initial phase. They are progressively *replaced* by
generated views as the descent proceeds (see ProgressiveDataset.drop_reference),
so a stage-by-stage sweep shows whether the progressive process preserves or
erodes agreement with the real capture -- which is the interesting question.

    env PYTHONPATH=. python scripts/eval_psnr.py \
        --data data/park_lowres --num-views 60 \
        --run baseline=outputs/gsplat_geoquery_sklearn \
        --run da3_gs=outputs/gsplat_geoquery_da3 \
        --run gs_depth=outputs/gsplat_geoquery_gsdepth \
        --out outputs/psnr_comparison.json
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
from prodig.dataset import load_rgb  # noqa: E402
from prodig.losses import ssim_index  # noqa: E402
from prodig.model import ProDiGModel  # noqa: E402

CHECKPOINTS = [
    ("initial", "initial.pt"),
    ("0.9", "stage_00_90.pt"),
    ("0.7", "stage_01_70.pt"),
    ("0.5", "stage_02_50.pt"),
    ("0.3", "stage_03_30.pt"),
    ("0.1", "stage_04_10.pt"),
    ("final", "final.pt"),
]


def load_model(path: Path, scene, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    splats = {k.split("splats.", 1)[1]: v for k, v in state.items() if k.startswith("splats.")}
    m = ProDiGModel.from_points(
        scene.points[:1], scene.colors[:1],
        num_appearances=len(scene.cameras), scene_scale=scene.scene_scale,
    )
    m.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in splats.items()})
    m.load_state_dict(state, strict=True)
    return m.to(device).eval(), splats["means"].shape[0], int(ckpt.get("global_step", -1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--num-views", type=int, default=60)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda")
    scene = load_colmap_scene(args.data, normalize=True)
    step = max(len(scene.cameras) // args.num_views, 1)
    cams = scene.cameras[::step][: args.num_views]
    print(f"scoring {len(cams)} real aerial views\n")

    # Load the ground-truth photographs once; every model is scored on the same set.
    gts = {}
    for c in cams:
        gt = load_rgb(c.image_path).to(device)
        if gt.shape[:2] != (c.height, c.width):
            raise ValueError(f"image/camera size mismatch for {c.name}")
        gts[c.name] = gt

    results = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        run = Path(path)
        results[label] = {}
        print(f"=== {label} ({run.name}) ===")
        for stage, fname in CHECKPOINTS:
            ck = run / "checkpoints" / fname
            if not ck.exists():
                continue
            model, n_gauss, gstep = load_model(ck, scene, device)
            psnrs, ssims = [], []
            with torch.no_grad():
                for c in cams:
                    cd = replace(c, c2w=c.c2w.to(device), K=c.K.to(device))
                    render, _, _ = model.render(cd)
                    pred = render[0, ..., :3].clamp(0, 1)
                    gt = gts[c.name]
                    mse = torch.mean((pred - gt) ** 2)
                    psnrs.append(float(-10.0 * torch.log10(mse.clamp_min(1e-12))))
                    ssims.append(float(ssim_index(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
            results[label][stage] = {
                "psnr": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
                "ssim": float(np.mean(ssims)), "gaussians": n_gauss, "step": gstep,
            }
            print(f"  {stage:8} PSNR {np.mean(psnrs):6.2f} dB   SSIM {np.mean(ssims):.4f}   "
                  f"{n_gauss:>9,} GS  @step {gstep}")
            del model
            torch.cuda.empty_cache()
        print()

    stages = [s for s, _ in CHECKPOINTS]
    labels = list(results)
    print("=" * (14 + 16 * len(labels)))
    print(f"{'PSNR (dB)':14}" + "".join(f"{l:>16}" for l in labels))
    print("=" * (14 + 16 * len(labels)))
    for s in stages:
        if not any(s in results[l] for l in labels):
            continue
        row = f"{s:14}"
        for l in labels:
            row += f"{results[l][s]['psnr']:>16.2f}" if s in results[l] else f"{'-':>16}"
        print(row)
    print(f"\n{'SSIM':14}" + "".join(f"{l:>16}" for l in labels))
    for s in stages:
        if not any(s in results[l] for l in labels):
            continue
        row = f"{s:14}"
        for l in labels:
            row += f"{results[l][s]['ssim']:>16.4f}" if s in results[l] else f"{'-':>16}"
        print(row)

    if args.out:
        args.out.write_text(json.dumps({"num_views": len(cams), "results": results}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
