import math

import numpy as np
import pytest
import torch

from prodig.ply_input import _SH_C0, load_ply_splats, read_ply


def _write_ply(path, n=64, seed=0):
    """Write a standard 3DGS PLY the way prodig.export.export_ply does."""
    rng = np.random.default_rng(seed)
    props = {
        "x": rng.uniform(-2, 2, n), "y": rng.uniform(-2, 2, n), "z": rng.uniform(-2, 2, n),
        "f_dc_0": rng.uniform(-1, 1, n), "f_dc_1": rng.uniform(-1, 1, n), "f_dc_2": rng.uniform(-1, 1, n),
        "opacity": rng.uniform(-3, 3, n),
        "scale_0": rng.uniform(-6, -2, n), "scale_1": rng.uniform(-6, -2, n), "scale_2": rng.uniform(-6, -2, n),
        "rot_0": np.ones(n), "rot_1": np.zeros(n), "rot_2": np.zeros(n), "rot_3": np.zeros(n),
    }
    names = list(props)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {k}\n" for k in names)
              + "end_header\n")
    data = np.stack([props[k] for k in names], axis=1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return props


def test_reads_all_properties(tmp_path):
    p = tmp_path / "m.ply"
    props = _write_ply(p, n=32)
    got = read_ply(p)
    assert set(got) == set(props)
    np.testing.assert_allclose(got["x"], props["x"], rtol=1e-6)


def test_scales_and_opacity_are_copied_pre_activation(tmp_path):
    """They are stored raw by convention; activating here would double-apply it."""
    p = tmp_path / "m.ply"
    props = _write_ply(p)
    s = load_ply_splats(p, frame="normalized")
    np.testing.assert_allclose(s["opacities"].detach().numpy(), props["opacity"], rtol=1e-5)
    np.testing.assert_allclose(s["scales"].detach().numpy()[:, 0], props["scale_0"], rtol=1e-5)


def test_colors_round_trip_through_sh(tmp_path):
    p = tmp_path / "m.ply"
    props = _write_ply(p)
    s = load_ply_splats(p, frame="normalized")
    expected_linear = np.clip(_SH_C0 * props["f_dc_0"] + 0.5, 0, 1)
    got_linear = torch.sigmoid(s["colors"][:, 0]).detach().numpy()
    np.testing.assert_allclose(got_linear, expected_linear, atol=1e-4)


def test_world_frame_applies_the_normalization_transform(tmp_path):
    p = tmp_path / "m.ply"
    props = _write_ply(p)
    scale, center = 0.125, np.array([1.0, -2.0, 3.0], dtype=np.float32)
    t = torch.eye(4)
    t[:3, :3] *= scale
    t[:3, 3] = torch.from_numpy(-center * scale)

    s = load_ply_splats(p, frame="world", normalization_transform=t)
    world = np.stack([props["x"], props["y"], props["z"]], axis=1)
    np.testing.assert_allclose(s["means"].detach().numpy(), (world - center) * scale, atol=1e-4)
    # sizes shrink by `scale`, which in log space is an additive shift
    np.testing.assert_allclose(
        s["scales"].detach().numpy()[:, 0], props["scale_0"] + math.log(scale), atol=1e-4
    )


def test_normalized_frame_leaves_geometry_untouched(tmp_path):
    p = tmp_path / "m.ply"
    props = _write_ply(p)
    s = load_ply_splats(p, frame="normalized")
    world = np.stack([props["x"], props["y"], props["z"]], axis=1)
    np.testing.assert_allclose(s["means"].detach().numpy(), world, atol=1e-5)


def test_world_frame_requires_a_transform(tmp_path):
    p = tmp_path / "m.ply"
    _write_ply(p)
    with pytest.raises(ValueError, match="requires normalization_transform"):
        load_ply_splats(p, frame="world")


def test_unrepresentable_features_are_zero_initialised(tmp_path):
    p = tmp_path / "m.ply"
    _write_ply(p, n=16)
    s = load_ply_splats(p, frame="normalized", gaussian_feature_dim=32, distance_feature_dim=8)
    assert s["features"].shape == (16, 32)
    assert s["f_sca"].shape == (16, 8) and s["f_opa"].shape == (16, 8)
    assert torch.count_nonzero(s["features"]) == 0


def test_keys_match_what_the_model_expects(tmp_path):
    from prodig.model import ProDiGModel
    p = tmp_path / "m.ply"
    _write_ply(p, n=16)
    loaded = load_ply_splats(p, frame="normalized")
    reference = ProDiGModel.from_points(
        torch.rand(16, 3), torch.rand(16, 3), num_appearances=2, scene_scale=1.0
    ).splats
    assert set(loaded) == set(reference), "splat keys must match ProDiGModel.from_points"


def test_rejects_missing_properties(tmp_path):
    p = tmp_path / "bad.ply"
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(p, "wb") as f:
        f.write(header.encode()); f.write(np.zeros((2, 3), dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="missing required PLY properties"):
        load_ply_splats(p, frame="normalized")


def test_rejects_truncated_file(tmp_path):
    p = tmp_path / "trunc.ply"
    _write_ply(p, n=64)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) - 400])
    with pytest.raises(ValueError, match="truncated"):
        read_ply(p)


def test_rejects_ascii_ply(tmp_path):
    p = tmp_path / "ascii.ply"
    p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0.0\n")
    with pytest.raises(ValueError, match="binary_little_endian"):
        read_ply(p)
