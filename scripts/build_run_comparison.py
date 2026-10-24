#!/usr/bin/env python3
"""Stage-by-stage comparison contact sheets between two prodig runs.

Built for: baseline run (GeoQuery reprojecting from the ORIGINAL aerial view via
COLMAP-anchored DA3 depth) vs the new run (reprojecting from the PRECEDING
stage's generated view via GS-anchored DA3 depth).

Both runs must share an altitude schedule for this to be meaningful -- the novel
cameras are then identical per stage, so a given filename denotes the same
viewpoint in both and the columns are directly comparable.

Columns: [raw GS render | old run fixed | new run fixed]. The raw column comes
from the NEW run's noisy_views/ (the baseline predates that feature); note it is
the new run's own pre-GeoQuery render, which from stage 1 onward diverges from
the baseline's because the models have diverged.

    env PYTHONPATH=. python scripts/build_run_comparison.py \
        --old-run outputs/gsplat_geoquery_sklearn \
        --new-run outputs/gsplat_geoquery_da3 \
        --output  outputs/comparison_da3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STAGES = [(0, 90), (1, 70), (2, 50), (3, 30), (4, 10)]


def sharpness(a: np.ndarray) -> float:
    g = a.mean(-1) if a.ndim == 3 else a
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def stats(path: Path) -> dict:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return {"brightness": float(a.mean()), "sharpness": sharpness(a), "std": float(a.std())}


def label(img: Image.Image, text: str, sub: str = "") -> Image.Image:
    """Caption a tile with an opaque strip so text stays legible over any image."""
    bar = 34 if sub else 20
    out = Image.new("RGB", (img.width, img.height + bar), (18, 18, 18))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((5, img.height + 3), text, fill=(255, 255, 255))
    if sub:
        d.text((5, img.height + 17), sub, fill=(170, 170, 170))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--new-run", type=Path, required=True)
    p.add_argument("--third-run", type=Path, default=None,
                   help="optional 4th column, e.g. the --depth-source gs_render run")
    p.add_argument("--third-label", default="GS-depth: prev-stage ref + GS render depth")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-views", type=int, default=6, help="views sampled per stage")
    p.add_argument("--tile-width", type=int, default=420)
    p.add_argument("--per-view", action="store_true",
                   help="write one image per view into <output>/<stage>/ instead of stacking "
                        "every view into a single tall contact sheet")
    p.add_argument("--no-sheet", action="store_true", help="skip the combined contact sheet")
    # Both runs use DA3; what differs is which image the depth belongs to and what
    # anchors its scale/offset (COLMAP sparse points vs the GS render). Spell that
    # out -- "COLMAP depth" reads as MVS depth, which is not what this is.
    p.add_argument("--old-label", default="OLD: original aerial ref + DA3 depth (COLMAP-anchored)")
    p.add_argument("--new-label", default="NEW: prev-stage ref + DA3 depth (GS-anchored)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx, pct in STAGES:
        name = f"stage_{idx:02d}_{pct:02d}"
        old_dir = args.old_run / "fixed_views" / name
        new_dir = args.new_run / "fixed_views" / name
        raw_dir = args.new_run / "noisy_views" / name
        if not old_dir.is_dir() or not new_dir.is_dir():
            print(f"{name}: skipped (missing in one run)")
            continue

        third_dir = (args.third_run / "fixed_views" / name) if args.third_run else None
        has_third = third_dir is not None and third_dir.is_dir()
        shared = {f.name for f in old_dir.glob("*.png")} & {f.name for f in new_dir.glob("*.png")}
        if has_third:
            # Intersect so every column shows the same viewpoint; a view missing
            # from any run would otherwise silently compare different cameras.
            shared &= {f.name for f in third_dir.glob("*.png")}
        shared = sorted(shared)
        if not shared:
            print(f"{name}: no shared view names")
            continue

        # Metrics over an even spread of the whole stage, not just the sheet views.
        sample = shared[:: max(len(shared) // 60, 1)][:60]
        om = [stats(old_dir / n) for n in sample]
        nm = [stats(new_dir / n) for n in sample]
        agg = {
            "stage": name,
            "shared_views": len(shared),
            "measured": len(sample),
            "old": {k: float(np.mean([m[k] for m in om])) for k in ("brightness", "sharpness", "std")},
            "new": {k: float(np.mean([m[k] for m in nm])) for k in ("brightness", "sharpness", "std")},
        }
        agg["sharpness_delta_pct"] = 100 * (agg["new"]["sharpness"] - agg["old"]["sharpness"]) / agg["old"]["sharpness"]
        if has_third:
            tm = [stats(third_dir / n) for n in sample]
            agg["third"] = {k: float(np.mean([m[k] for m in tm])) for k in ("brightness", "sharpness", "std")}
            agg["third_delta_pct"] = 100 * (agg["third"]["sharpness"] - agg["old"]["sharpness"]) / agg["old"]["sharpness"]
        for run, key in ((args.new_run, "calibrated"), (args.third_run, "third_calibrated")):
            if run is None:
                continue
            calib = run / "depths_calibrated" / name / "calibration_summary.json"
            if calib.exists():
                try:
                    c = json.loads(calib.read_text())
                    agg[key] = f"{c['calibrated']}/{c['generated']}"
                except Exception:
                    pass
        summary.append(agg)

        sheet_views = shared[:: max(len(shared) // args.num_views, 1)][: args.num_views]
        rows = []
        for vn in sheet_views:
            tiles = []
            cols = [(raw_dir / vn, "RAW GS render"), (old_dir / vn, args.old_label), (new_dir / vn, args.new_label)]
            if has_third:
                cols.append((third_dir / vn, args.third_label))
            for src, cap in cols:
                if not src.exists():
                    tiles.append(label(Image.new("RGB", (args.tile_width, int(args.tile_width * 2 / 3)), (40, 40, 40)), cap, "(missing)"))
                    continue
                im = Image.open(src).convert("RGB")
                h = int(args.tile_width * im.height / im.width)
                s = stats(src)
                tiles.append(label(im.resize((args.tile_width, h), Image.Resampling.LANCZOS), cap,
                                   f"{vn}  bright {s['brightness']:.3f}  sharp {s['sharpness']:.4f}"))
            h = max(t.height for t in tiles)
            row = Image.new("RGB", (sum(t.width for t in tiles) + 2 * 6, h), (18, 18, 18))
            x = 0
            for t in tiles:
                row.paste(t, (x, 0))
                x += t.width + 6
            rows.append(row)

            if args.per_view:
                # Same row, standalone: repeat the stage header so each file is
                # self-describing once it is separated from the others.
                vdir = args.output / name
                vdir.mkdir(parents=True, exist_ok=True)
                single = Image.new("RGB", (row.width, row.height + 22), (18, 18, 18))
                ImageDraw.Draw(single).text(
                    (6, 5), f"{name} (factor {pct/100:.2f})  {vn}", fill=(255, 255, 255))
                single.paste(row, (0, 22))
                single.save(vdir / vn)

        if args.no_sheet:
            print(f"{name}: {len(sheet_views)} per-view images -> {args.output / name}  "
                  f"({len(shared)} shared views, sharpness {agg['sharpness_delta_pct']:+.1f}%)")
            continue

        w = max(r.width for r in rows)
        sheet = Image.new("RGB", (w, sum(r.height for r in rows) + 6 * (len(rows) - 1) + 26), (18, 18, 18))
        d = ImageDraw.Draw(sheet)
        hdr = f"{name}  (factor {pct/100:.2f})   sharpness {agg['old']['sharpness']:.4f} -> {agg['new']['sharpness']:.4f}  ({agg['sharpness_delta_pct']:+.1f}%)"
        if has_third:
            hdr += f" | GS-depth {agg['third']['sharpness']:.4f} ({agg['third_delta_pct']:+.1f}%)"
        if "calibrated" in agg:
            hdr += f"   calibrated {agg['calibrated']}"
        if "third_calibrated" in agg:
            hdr += f" / GS {agg['third_calibrated']}"
        d.text((6, 7), hdr, fill=(255, 255, 255))
        y = 26
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 6
        out = args.output / f"{name}.png"
        sheet.save(out)
        print(f"{name}: {out}  ({len(shared)} shared views, sharpness {agg['sharpness_delta_pct']:+.1f}%)")

    if summary:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        three = any("third" in s for s in summary)
        hdr = f"\n{'stage':14} {'baseline':>10} {'DA3+GS':>10} {'delta':>8}"
        if three:
            hdr += f" {'GS-depth':>10} {'delta':>8} {'GS refs':>10}"
        hdr += f" {'DA3 refs':>10}"
        print(hdr)
        for s in summary:
            row = (f"{s['stage']:14} {s['old']['sharpness']:>10.4f} {s['new']['sharpness']:>10.4f} "
                   f"{s['sharpness_delta_pct']:>7.1f}%")
            if three:
                row += (f" {s['third']['sharpness']:>10.4f} {s['third_delta_pct']:>7.1f}%"
                        f" {s.get('third_calibrated','-'):>10}") if "third" in s else f" {'-':>10} {'-':>8} {'-':>10}"
            row += f" {s.get('calibrated','-'):>10}"
            print(row)
        print(f"\nwrote {args.output}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
