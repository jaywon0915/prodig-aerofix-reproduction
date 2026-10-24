"""Anchor a monocular (DA3) relative depth map to the Gaussian scene's own geometry.

Motivation
----------
GeoQuery is geometry-conditioned: it reprojects reference content through that
view's *metric* depth. Real aerial views get their depth from DA3 calibrated
against COLMAP sparse points (``scripts/apply_depth_params_correction.py``), but a
*generated* altitude view has no COLMAP tracks to calibrate against -- it is a
synthesized viewpoint that never existed in the capture.

The GS model itself supplies the missing metric anchor. Rendering the scene at the
generated camera (``render_mode="RGB+ED"``) yields a depth map that is metrically
consistent with the reconstruction but smooth, blobby, and unreliable wherever
coverage is thin. DA3 gives the opposite: sharp, detailed structure with no
absolute scale. This module fuses the two -- DA3 for detail, GS for scale.

Following the same affine-in-inverse-depth convention graphdeco uses for
``depth_params.json`` (see ``scripts/apply_depth_params_correction.py``)::

    calibrated_inv = scale * da3_inv + offset
    metric_depth   = 1 / calibrated_inv

with (scale, offset) solved per view by least squares against the GS-rendered
depth, restricted to pixels the GS render can actually vouch for.

Frames
------
``gs_depth`` is expected in the SAME frame the caller wants back. prodig
normalizes the scene to a unit sphere (``colmap_scene.normalize_scene``), so a
raw gsplat depth render is in *normalized* units while GeoQuery and
``data/park/depths_metric`` are in real COLMAP units. Convert with
``gs_depth_real = gs_depth_normalized / normalization_transform[0, 0]`` BEFORE
calling this, or pass ``depth_scale`` to have it done here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CalibrationResult:
    """Outcome of one per-view fit. ``valid=False`` means do not use ``depth``."""

    depth: np.ndarray | None
    scale: float
    offset: float
    inlier_fraction: float
    num_pixels: int
    relative_error: float
    valid: bool
    reason: str = ""

    def summary(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "depth"}
        return out


def _to_inverse(depth: np.ndarray, eps: float) -> np.ndarray:
    return 1.0 / np.clip(depth, eps, None)


def calibrate_relative_depth(
    da3_depth: np.ndarray,
    gs_depth: np.ndarray,
    *,
    gs_alpha: np.ndarray | None = None,
    da3_is_inverse: bool = False,
    depth_scale: float = 1.0,
    min_alpha: float = 0.5,
    trim_percentile: float = 10.0,
    max_iters: int = 3,
    min_pixels: int = 500,
    max_relative_error: float = 0.05,
    eps: float = 1e-6,
) -> CalibrationResult:
    """Fit ``scale``/``offset`` mapping DA3 relative depth onto GS metric depth.

    Args:
        da3_depth: DA3 output, HxW. Depth by default; set ``da3_is_inverse``
            if the model emits inverse depth (disparity) instead.
        gs_depth: gsplat ``RGB+ED`` depth channel at the same camera, same HxW.
        gs_alpha: gsplat alpha at that camera. Pixels below ``min_alpha`` carry
            almost no Gaussian coverage, so their depth is meaningless and must
            not steer the fit -- this mask is what keeps sky and unmodelled
            regions out.
        da3_is_inverse: whether ``da3_depth`` is already inverse depth.
        depth_scale: divide ``gs_depth`` by this to reach the target frame; pass
            ``normalization_transform[0, 0]`` to convert normalized -> real.
        trim_percentile: per iteration, discard this percent of worst-residual
            pixels before refitting (robustness against DA3 outliers and GS
            floaters, which are both common and both badly non-Gaussian).
        max_relative_error: reject the fit if median relative depth error
            against GS exceeds this. A bad fit is worse than no depth: it feeds
            GeoQuery confidently wrong geometry.

            The 0.05 default is measured, not guessed. ``scripts/validate_depth_calib.py``
            calibrated 40 real aerial views and compared each against the
            independently-derived COLMAP-anchored depth in
            ``data/park/depths_metric``. This residual (observable at
            runtime) predicts true error against COLMAP at r=+0.912, and the two
            populations barely overlap: views accurate to <15% peak at a residual
            of 0.0611, while inaccurate ones start at 0.0583. Cutting at 0.05
            rejected 4/4 bad views while losing only 2/36 good ones, leaving a
            median error of 1.9%. Raising this toward 0.35 admits views whose GS
            geometry is simply wrong -- typically low-altitude ones, since error
            correlates -0.765 with scene depth.

    Returns:
        CalibrationResult. Callers MUST check ``.valid`` before using ``.depth``.
    """
    if da3_depth.shape != gs_depth.shape:
        raise ValueError(f"shape mismatch: da3 {da3_depth.shape} vs gs {gs_depth.shape}")

    gs = np.asarray(gs_depth, dtype=np.float64) / float(depth_scale)
    da3 = np.asarray(da3_depth, dtype=np.float64)
    da3_inv = da3 if da3_is_inverse else _to_inverse(da3, eps)
    gs_inv = _to_inverse(gs, eps)

    mask = np.isfinite(da3_inv) & np.isfinite(gs_inv) & (gs > eps)
    if gs_alpha is not None:
        if gs_alpha.shape != gs.shape:
            raise ValueError(f"alpha shape {gs_alpha.shape} != depth {gs.shape}")
        mask &= np.asarray(gs_alpha, dtype=np.float64) >= min_alpha

    def _fail(reason: str, scale=0.0, offset=0.0, frac=0.0, npx=0, relerr=float("inf")):
        return CalibrationResult(None, scale, offset, frac, npx, relerr, False, reason)

    total = int(mask.sum())
    if total < min_pixels:
        return _fail(f"only {total} valid pixels (< {min_pixels})", npx=total)

    x_all, y_all = da3_inv[mask], gs_inv[mask]
    sel = np.ones(total, dtype=bool)
    scale = offset = 0.0
    for _ in range(max(max_iters, 1)):
        x, y = x_all[sel], y_all[sel]
        if x.size < min_pixels:
            return _fail(f"trimmed to {x.size} pixels (< {min_pixels})", npx=int(x.size))
        # Guard the degenerate case where DA3 returned a near-constant map:
        # the normal equations are singular and any fit is meaningless.
        if float(np.std(x)) < eps:
            return _fail("DA3 depth is constant; cannot fit", npx=int(x.size))
        A = np.stack([x, np.ones_like(x)], axis=1)
        (scale, offset), *_ = np.linalg.lstsq(A, y, rcond=None)
        residual = np.abs((scale * x_all + offset) - y_all)
        cutoff = np.percentile(residual[sel], 100.0 - trim_percentile)
        new_sel = residual <= max(cutoff, eps)
        if new_sel.sum() < min_pixels:
            break
        sel = new_sel

    if not np.isfinite(scale) or not np.isfinite(offset):
        return _fail("non-finite fit")

    calibrated_inv = scale * da3_inv + offset
    # Any pixel whose calibrated inverse depth is <= 0 sits at or behind the
    # camera -- unrepresentable. Clamp rather than emit negative depth, and let
    # relative_error below decide whether the whole view is salvageable.
    positive = calibrated_inv > eps
    if not positive.any():
        return _fail("calibration produced no positive depths", scale, offset, npx=total)
    depth = np.full(da3_inv.shape, np.nan, dtype=np.float32)
    depth[positive] = (1.0 / calibrated_inv[positive]).astype(np.float32)

    cmp = mask & positive
    if not cmp.any():
        return _fail("no overlap between GS mask and positive depths", scale, offset, npx=total)
    rel = np.abs(depth[cmp] - gs[cmp]) / np.clip(gs[cmp], eps, None)
    relative_error = float(np.median(rel))
    inlier_fraction = float(sel.sum()) / float(total)

    if relative_error > max_relative_error:
        return _fail(
            f"median relative error {relative_error:.3f} > {max_relative_error}",
            scale, offset, inlier_fraction, total, relative_error,
        )
    return CalibrationResult(
        depth=depth,
        scale=float(scale),
        offset=float(offset),
        inlier_fraction=inlier_fraction,
        num_pixels=total,
        relative_error=relative_error,
        valid=True,
    )
