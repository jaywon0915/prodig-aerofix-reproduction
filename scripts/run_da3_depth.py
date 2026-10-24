#!/usr/bin/env python3
"""Batch DA3 relative-depth inference over a directory of images -> one .npy each.

MUST run in the DA3 environment, not the anchor-prodig one:

    /path/to/da3_venv/bin/python scripts/run_da3_depth.py \
        --images <dir> --output <dir>

DA3's own requirements pin `numpy<2`, while anchor-prodig (gsplat/pycolmap/torch)
is on numpy>=2. Installing DA3 into anchor-prodig downgrades numpy and breaks
gsplat, so the two never share an interpreter -- they exchange .npy files. Same
split the project already uses for MVSFormer++ and graphdeco.

Output is DA3's *relative* depth at DA3's own processing resolution (a multiple
of 14, so generally NOT the input image size). Resampling to the target frame is
the consumer's job -- see prodig.depth_calib, which fits scale/offset against a
GS-rendered depth map to recover metric scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="directory of input images")
    p.add_argument("--output", type=Path, required=True, help="directory for <stem>.npy depth maps")
    p.add_argument("--model", default="da3-large", help="DA3 model name (registry key)")
    p.add_argument("--process-res", type=int, default=504, help="DA3 processing resolution (multiple of 14)")
    p.add_argument("--batch-size", type=int, default=8, help="images per inference call")
    p.add_argument("--save-conf", action="store_true", help="also write <stem>.conf.npy")
    p.add_argument("--overwrite", action="store_true", help="recompute maps that already exist")
    args = p.parse_args(argv)

    images = sorted(f for f in args.images.iterdir() if f.suffix in IMAGE_EXTS)
    if not images:
        print(f"error: no images found in {args.images}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    todo = [f for f in images if args.overwrite or not (args.output / f"{f.stem}.npy").exists()]
    print(f"{len(images)} images, {len(todo)} to compute -> {args.output}", flush=True)
    if not todo:
        return 0

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model.upper()}").cuda().eval()
    print(f"loaded {args.model}", flush=True)

    meta: dict[str, dict] = {}
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        pred = model.inference(image=[str(f) for f in chunk], process_res=args.process_res)
        depth = np.asarray(pred.depth, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        conf = None if pred.conf is None else np.asarray(pred.conf, dtype=np.float32)
        if conf is not None and conf.ndim == 2:
            conf = conf[None]
        if depth.shape[0] != len(chunk):
            raise RuntimeError(f"DA3 returned {depth.shape[0]} maps for {len(chunk)} images")

        for i, f in enumerate(chunk):
            np.save(args.output / f"{f.stem}.npy", depth[i])
            if args.save_conf and conf is not None:
                np.save(args.output / f"{f.stem}.conf.npy", conf[i])
            meta[f.stem] = {
                "shape": list(depth[i].shape),
                "min": float(depth[i].min()),
                "max": float(depth[i].max()),
                "median": float(np.median(depth[i])),
            }
        print(f"  {min(start + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    # is_metric is an empty dict for the relative models -- record it so a
    # consumer can tell a relative checkpoint from da3metric-* after the fact.
    summary = {
        "model": args.model,
        "process_res": args.process_res,
        "is_metric": bool(getattr(pred, "is_metric", None)),
        "count": len(meta),
        "images": meta,
    }
    (args.output / "da3_meta.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(meta)} depth maps + da3_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
