# ProDiG + GeoQuery from a pre-trained PLY

한국어 설명은 [USAGE_KO.md](USAGE_KO.md).

Runs the **same pipeline as the baseline `gsplat_geoquery_sklearn` run** — GeoQuery
reprojects from the **original GT aerial photograph and its depth at every altitude
stage** — with one addition: it can start from an **existing Gaussian PLY** instead
of the COLMAP point cloud, skipping initial training and going straight to
render → fix → descend.

```
input PLY (already good)
      │
      ├─ render novel view at 90% altitude   → noisy_views/
      ├─ GeoQuery fixes it, using the GT aerial photo + its depth as reference
      │                                      → fixed_views/
      ├─ train on the fixed views
      └─ repeat at 70% → 50% → 30% → 10%
```

## What you need

| Input | Required | What it is |
|---|---|---|
| `--data <dir>` | yes | COLMAP scene: `images/` + `sparse/0/`. Camera model must be PINHOLE/SIMPLE_PINHOLE (run `colmap image_undistorter` first) |
| `--input-ply <file>` | no | a trained 3DGS PLY. Omit it to train from the point cloud as usual |
| `--reference-depth-dir <dir>` | yes | metric depth `.npy` per GT image, named by image stem (`H_010_00001.npy`) |
| `--aerofix-factory` | yes | `examples.geoquery_bridge:create_geoquery` |
| GPU | yes | CUDA. ~13 GB at 1598×1064; GeoQuery dominates, not the Gaussians |

**Why reference depth is required.** GeoQuery is geometry-conditioned: it warps the
reference view into the target viewpoint through that reference's depth, so it needs
depth **for the reference image, not for the view being fixed**. Without it every
view is soft-rejected and the run produces nothing.

Environment paths for the GeoQuery checkpoint (required; there are no defaults):

```bash
export GEOQUERY_CHECKPOINT=/path/to/geoquery_window_size3_down1.pkl
export GEOQUERY_SOURCE_DIR=/path/to/GeoQuery/src
```

### `anchor_prodig` — only needed for the GeoQuery and AeroFix bridges

`anchor_prodig` is the main package of the lab's research repository, which stays
private while the project is in development. It is not on PyPI. This pipeline
borrows exactly one thing from it — the code that loads and runs the diffusion
refiner checkpoints:

```python
from anchor_prodig.config import DiffusionConfig          # refiner configuration
from anchor_prodig.diffusion.aerofix import create_refiner # builds GeoQuery / aeroFix
from anchor_prodig.diffusion.base import RefinementRequest # one refinement call
```

Everything else here — Gaussian training, rasterization, distance-adaptive gating,
the progressive descent — is self-contained and does not depend on it. Only
`examples/geoquery_bridge.py` and `examples/real_aerofix_bridge.py` import it; the
default `--aerofix-factory identity` and `examples/aerofix_bridge_example.py` run
without it.

```
prodig/                    the pipeline (self-contained)
        │ calls
examples/*_bridge.py       thin adapters
        │ imports
anchor_prodig.diffusion    loads + runs the refiner checkpoints (lab package)
```

With access to the lab repository, install the package from its checkout and verify:

```bash
pip install -e /path/to/lab-repo   # installs the anchor-prodig package from src/
python -c "import anchor_prodig; print(anchor_prodig.__file__)"
```

### ⚠ scikit-learn is mandatory — check it before every fresh install

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

If that fails, **fix it before running anything**:

```bash
pip install scikit-learn
```

It is a declared dependency in `requirements.txt` and `pyproject.toml`, so
`pip install -e .` covers it — but running via `PYTHONPATH=.` alone does not, and an
editable install done with `--no-deps` leaves it missing. (`scikit-image` is a
different package and does **not** satisfy it.)

**Why it's checked so strictly.** `prodig/model.py` originally imported
`sklearn.neighbors.NearestNeighbors` inside a bare `try/except Exception:` that
printed no warning. Without it, every Gaussian got one uniform initial scale instead
of a per-point k-NN estimate. Measured on the park scene:

| | median initial scale | over gsplat's prune threshold |
|---|---|---|
| with scikit-learn | 0.00103 | 336 / 576,408 = **0.06 %** |
| without | 0.2303 (uniform) | **100 %** |

gsplat gates oversized pruning behind `step > reset_every`, so at the first refine
step past 3,000 the whole model is culled — observed **569,729 → 11 Gaussians**.
Renders go black, the refiner returns dark output from a black input, and those dark
images become training targets. The run then **completed with exit code 0** and no
error anywhere: the only symptom was that every output image was dark. A missing
scikit-learn now raises an `ImportError` at startup instead.

## Running

Run from the **repository root** so `examples.` resolves.

### From a pre-trained PLY (no initial training)

```bash
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m prodig.train \
  --data data/park \
  --output outputs/<run_name> \
  --input-ply /path/to/model.ply \
  --input-ply-frame world \
  --aerofix-factory examples.geoquery_bridge:create_geoquery \
  --reference-depth-dir data/park/depths_metric \
  --min-aerofix-quality 0.01 \
  --stage-steps 7000 \
  --factors 0.9,0.7,0.5,0.3,0.1 --trajectory forward \
  --max-views-per-stage 0 \
  --refine-stop-iter 12000 --refine-every 300 \
  --grow-grad2d 0.0008 --prune-opa 0.02 --checkpoint-every 1000
```

`--initial-steps` defaults to **0** when `--input-ply` is given. Pass it explicitly
if you still want some aerial fitting first.

### Without a PLY (identical to the baseline run)

Drop `--input-ply*` and add `--initial-steps 7000`.

## `--input-ply-frame` — get this right

The pipeline normalises the scene to a unit sphere, so a PLY is in one of two frames:

| Source of the PLY | Use |
|---|---|
| external trainer (graphdeco 3DGS, Scaffold-GS, …), COLMAP coordinates | `world` (default) |
| this pipeline's own `plys/*.ply` | `normalized` |

Choosing wrong **does not error** — the Gaussians land far from the cameras and every
render comes out empty. The run prints a check at startup; make it agree with your flag:

```
PLY: 6,043,753 Gaussians, extent radius 1.94; normalized camera radius 1.00 -> looks like frame='normalized'
```

## Required flags people forget

- **`--min-aerofix-quality 0.01`** — the minimum refiner quality for a view to be
  used. Quality 0 (a refiner's explicit rejection) is always skipped; before that was
  enforced, the `0.0` default let rejected views through, because `0.0 < 0.0` is false.
- **`--reference-depth-dir`** — see above; without it nothing is produced.

## Outputs

```
noisy_views/stage_XX_YY/   raw render, before GeoQuery
fixed_views/stage_XX_YY/   after GeoQuery
plys/                      initial + per-stage + final (raw, and _gated where applicable)
checkpoints/               initial.pt, per-stage, final.pt, latest.pt
logs/train.jsonl           loss + Gaussian count every 50 steps
config.json, run.log
```

Depth coverage is only 855/975 views on the park scene — 120 GT images have no
calibrated depth, so those views are skipped each stage. That is expected, not a bug.

## Notes

- **Resolution follows the dataset.** There is no resize option: renders match the
  COLMAP camera size. `data/park` → 1598×1064, `data/park_lowres` → 576×384.
- **GeoQuery is capped at 1024 px on the long side** (`GEOQUERY_MAX_SIDE`). Uncapped at
  1598×1064 it needs ~23.5 GB of a 24 GB card. Renders and `noisy_views` stay full
  resolution; only the correction is computed smaller and upscaled back.
- **Resume** with `--resume <checkpoint>`; `latest.pt` is written every 1000 steps.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q     # 38 tests
```
