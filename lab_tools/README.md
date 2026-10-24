# Lab tools

Tools I wrote inside the lab's research repository while reproducing AeroFix and
ProDiG. They drive that repository's `anchor_prodig` package and its
`aerofix_training/` scripts, which are private while the project is in development,
so **they don't run on their own**. Paths are placeholders (`/path/to/...`).

| Folder | What it is |
|---|---|
| [`aerofix_data_prep/`](#1-aerofix-training-data-aerofix_data_prep) | Configs that turn 7 GauUScene aerial datasets into AeroFix training pairs |
| [`aerofix_validation/`](#2-aerofix-checkpoint-validation-aerofix_validation) | CLI and pinned environment for measuring a trained AeroFix checkpoint |
| [`prodig_forward_validation/`](#3-prodig-forward-stage-validation-prodig_forward_validation) | Runs and checks ProDiG's first descent stage before committing to all five |

---

## 1. AeroFix training data (`aerofix_data_prep/`)

AeroFix is trained on pairs of a degraded Gaussian Splatting render, the clean
ground-truth photo from the same pose, and a nearby reference photo. The goal was to
generate those pairs for 7 GauUScene aerial datasets, the same way the team already
had for its synthetic scenes.

| Scene | Images |
|---|---|
| CUHK_tower_low | 715 |
| CUHK_tower_up | 715 |
| HAV | 424 |
| LFLS | 1,106 |
| SMBU | 829 |
| SZIIT | 1,215 |
| SZTU | 1,500 |
| **Total** | **6,504** |

**Pipeline, per scene:**

1. `convert_colmap.py` turns an undistorted COLMAP reconstruction (PINHOLE /
   SIMPLE_PINHOLE cameras, matching images, a `.ply` point cloud) into a `scene.json`
   manifest.
2. `prepare.py --config configs/gauUscene_<scene>_graphdeco.yaml` trains a Gaussian
   model with the graphdeco 3DGS backend (7,000 steps), renders degraded views, and
   pairs each query view with a reference view by altitude and co-visibility.

Each config is the team's `citysample_1` template with only the output paths, scene
ID and manifest changed.

### Root cause: COLMAP exports without feature tracks

The first run failed on **all 7 scenes** with
`RuntimeError: At least two generated pairs are required`, and every query view was
logged as `no compatible reference view`. Gaussian training itself finished cleanly
each time, so the failure was entirely in pairing.

The datasets' `sparse/0/` exports had camera poses and point positions but **no
feature tracks**: the `POINTS2D[]` line was blank for every image and every point's
`TRACK[]` was empty. They were most likely exported from photogrammetry software
(the filenames come from DJI Zenmuse L1 missions) as poses plus a dense point cloud.
Pairing requires at least 50 shared COLMAP track observations between a query and a
candidate reference, so with zero tracks every candidate was rejected regardless of
altitude, distance or heading. Re-serializing the model (`colmap model_converter`)
can't help, since it can't create tracks that were never exported.

**Fix:** re-run Structure-from-Motion from the raw images. The scenes are continuous,
frame-ordered drone flights, so sequential matching replaces exhaustive matching
(linear instead of quadratic in image count; exhaustive would have taken 1–2+ days
over up to 1,500 images per scene):

```bash
colmap feature_extractor --database_path "$OUT/distorted/database.db" \
  --image_path "$OUT/input" --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1
colmap sequential_matcher --database_path "$OUT/distorted/database.db" \
  --SiftMatching.use_gpu 1
colmap mapper --database_path "$OUT/distorted/database.db" \
  --image_path "$OUT/input" --output_path "$OUT/distorted/sparse" \
  --Mapper.ba_global_function_tolerance=0.000001
colmap image_undistorter --image_path "$OUT/input" \
  --input_path "$OUT/distorted/sparse/0" --output_path "$OUT" --output_type COLMAP
```

Then rebuild the manifest and **check that `colmap_point3D_ids` is non-empty before
spending GPU time on training**. First rebuilt scene: CUHK_tower_low, 707 / 715
images registered, 447,276 points with real tracks, about 27 minutes on one RTX 3090 Ti.

Two follow-ups worth knowing:

- These captures are near-level orbits (over 1,000 units of horizontal spread but only
  7–30 units of altitude change), so altitude-based pairing may still be thin. Check
  pair counts, not just the exit code.
- `prepare.py` twice reported success on renders with visible streaking artifacts that
  passed its health check. Look at a few `noisy.png` / `clean.png` pairs before
  calling a scene done.

---

## 2. AeroFix checkpoint validation (`aerofix_validation/`)

- `evaluate_aerofix_checkpoint.py` evaluates a trained AeroFix `.pkl` checkpoint on a
  prepared `data.json` and reports PSNR / SSIM / LPIPS for the refined output against
  the unrefined input render.
- `requirements-aerofix-validation.txt` pins the package versions that were verified
  to work together.

**Test set:** 10 MatrixCity test pairs, one per city block. I pulled only those pairs'
folders with targeted `rsync` instead of downloading the full 34 GB shard set, plus
one training pair, because the dataset validator rejects an empty training split even
though evaluation never reads it.

**Result** for the team's step-10,000 checkpoint:

| Metric | Input render | After AeroFix | Change |
|---|---|---|---|
| LPIPS ↓ (perceptual) | 0.322 | **0.289** | better; 8 of 10 pairs improved |
| PSNR ↑ | **27.45** | 25.77 | −1.68 dB |
| SSIM ↑ | **0.805** | 0.776 | −0.029 |

The model trades pixel-exact accuracy for perceptual realism, a known tradeoff for
diffusion-based restoration. It helped most where the input render was badly degraded
(largest LPIPS gain, +0.146, on a water scene) and slightly hurt renders that were
already sharp (−0.107). Ten pairs is a first read, not a verdict.

**Bugs found getting the evaluation to run:**

- **`plucker_embed` passed to the wrong place.** The loader passed it to the AeroFix
  model's constructor, which doesn't accept it, crashing with a `TypeError`. It's a
  per-call `forward()` argument. A first look suggested the pinned fork never
  supported Plücker embeddings at all; a deeper check found the implementation in the
  attention layers and 32 trained `plucker_proj` tensors in the checkpoint. The fix
  was deleting one line.
- **`torch.cuda.set_device()` crashed on a bare `"cuda"` device.** Fixed to default to
  index 0.
- **A config silently overrode the checkpoint path.** Configs set up for the public
  `difix_ref` baseline force that backend regardless of `--checkpoint`; the evaluation
  now sets `backend` and `checkpoint_format` explicitly.
- **DreamSim crashed** on a `peft` version mismatch. Disabled for validation; LPIPS,
  PSNR and SSIM are unaffected.
- **pytorch3d failed to build** against CUDA 13. Building from `main` with a CUDA
  toolkit that matches `torch.version.cuda` works; `stable` doesn't.

---

## 3. ProDiG forward-stage validation (`prodig_forward_validation/`)

`run_prodig_forward_validation.py` runs **only** ProDiG's first descent stage (90% of
the original altitude) through the lab pipeline: render pseudo views at that altitude,
repair them with AeroFix, retrain on real aerial photos plus the repaired views, and
export `final_gs.ply` to inspect in a viewer such as SuperSplat. That shows whether
the model is good enough to keep descending before spending hours on all five stages.
`configs/` holds the park (four iterations) and road scene configs it was run with.

Three defaults deliberately differ from the lab's template, because during validation
they turned out to be bugs rather than tuning choices:

- **Terrain-relative clearance is off.** It snapped every descended camera to roughly
  the same absolute height, which produced solid-black renders on scenes whose
  coordinates aren't in meters. It isn't part of ProDiG's published method either.
- **Gaussian count is capped.** Unbounded densification reliably ran a 24 GB GPU out
  of memory partway through the first training phase.
- **Adaptive stage subdivision is off**, so exactly one stage runs at exactly the
  requested altitude.
