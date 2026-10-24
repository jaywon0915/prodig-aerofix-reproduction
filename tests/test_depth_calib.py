import numpy as np
import pytest

from prodig.depth_calib import calibrate_relative_depth


def _synthetic(h=64, w=64, scale=3.0, offset=0.5, seed=0):
    """Build a GS depth map and the DA3 relative depth it should be recovered from.

    Ground truth: gs_inv = scale * da3_inv + offset, so a correct fit must return
    exactly (scale, offset) in the noise-free case.
    """
    rng = np.random.default_rng(seed)
    da3_inv = rng.uniform(0.1, 1.0, size=(h, w))
    gs_inv = scale * da3_inv + offset
    return 1.0 / da3_inv, 1.0 / gs_inv


def test_recovers_exact_affine_transform():
    da3, gs = _synthetic(scale=3.0, offset=0.5)
    r = calibrate_relative_depth(da3, gs)
    assert r.valid, r.reason
    assert r.scale == pytest.approx(3.0, rel=1e-6)
    assert r.offset == pytest.approx(0.5, rel=1e-6)
    assert r.relative_error < 1e-6
    np.testing.assert_allclose(r.depth, gs, rtol=1e-5)


def test_alpha_mask_excludes_uncovered_pixels():
    """Pixels with no Gaussian coverage hold garbage depth and must not steer the fit."""
    da3, gs = _synthetic(scale=2.0, offset=0.25)
    alpha = np.ones_like(gs)
    alpha[:16, :] = 0.0
    gs_corrupt = gs.copy()
    gs_corrupt[:16, :] = 1e-3  # nonsense depth where alpha is zero

    good = calibrate_relative_depth(da3, gs_corrupt, gs_alpha=alpha)
    assert good.valid, good.reason
    assert good.scale == pytest.approx(2.0, rel=1e-5)

    bad = calibrate_relative_depth(da3, gs_corrupt, gs_alpha=None, max_relative_error=1e9)
    assert abs(bad.scale - 2.0) > 1e-3, "unmasked fit should be dragged off by the garbage region"


def test_outlier_trimming_survives_da3_spikes():
    da3, gs = _synthetic(scale=1.5, offset=0.2, seed=3)
    da3_noisy = da3.copy()
    rng = np.random.default_rng(1)
    idx = rng.choice(da3.size, size=da3.size // 20, replace=False)
    da3_noisy.flat[idx] *= rng.uniform(5, 20, size=idx.size)
    r = calibrate_relative_depth(da3_noisy, gs, trim_percentile=15.0, max_iters=4,
                                 max_relative_error=1e9)
    assert r.valid, r.reason
    assert r.scale == pytest.approx(1.5, rel=0.05)


def test_depth_scale_converts_normalized_to_real_frame():
    """gs_depth in normalized units must yield real-frame depth via depth_scale."""
    da3, gs_real = _synthetic(scale=2.0, offset=0.4)
    norm = 0.125
    gs_normalized = gs_real * norm
    r = calibrate_relative_depth(da3, gs_normalized, depth_scale=norm)
    assert r.valid, r.reason
    np.testing.assert_allclose(r.depth, gs_real, rtol=1e-5)


def test_rejects_constant_da3_map():
    gs = np.full((32, 32), 5.0)
    r = calibrate_relative_depth(np.full((32, 32), 2.0), gs)
    assert not r.valid
    assert "constant" in r.reason


def test_rejects_when_too_few_valid_pixels():
    da3, gs = _synthetic(h=16, w=16)
    r = calibrate_relative_depth(da3, gs, min_pixels=500)
    assert not r.valid
    assert "valid pixels" in r.reason


def test_rejects_bad_fit_rather_than_returning_wrong_geometry():
    """An unfittable pair must fail closed -- wrong depth is worse than none."""
    rng = np.random.default_rng(7)
    da3 = rng.uniform(1.0, 10.0, size=(64, 64))
    gs = rng.uniform(1.0, 10.0, size=(64, 64))  # unrelated
    r = calibrate_relative_depth(da3, gs, max_relative_error=0.35)
    assert not r.valid
    assert r.depth is None


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        calibrate_relative_depth(np.ones((8, 8)), np.ones((8, 9)))


def test_accepts_inverse_depth_input():
    da3, gs = _synthetic(scale=2.5, offset=0.3)
    r = calibrate_relative_depth(1.0 / da3, gs, da3_is_inverse=True)
    assert r.valid, r.reason
    assert r.scale == pytest.approx(2.5, rel=1e-6)
