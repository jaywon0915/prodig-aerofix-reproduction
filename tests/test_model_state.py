import sys
import torch

from prodig.model import ProDiGModel, apply_adaptive_gates


def test_apply_adaptive_gates():
    scales = torch.ones(2, 3)
    opacity = torch.tensor([0.8, 0.4])
    sg = torch.tensor([[0.5], [0.25]])
    og = torch.tensor([[0.5], [0.5]])
    s2, o2 = apply_adaptive_gates(scales, opacity, sg, og)
    assert torch.allclose(s2, torch.tensor([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]))
    assert torch.allclose(o2, torch.tensor([0.4, 0.2]))


def test_model_has_densification_aligned_per_gaussian_state():
    points = torch.randn(7, 3)
    rgbs = torch.rand(7, 3)
    model = ProDiGModel.from_points(points, rgbs, num_appearances=4, scene_scale=2.0)
    n = len(points)
    required = {"means", "scales", "quats", "opacities", "features", "colors", "f_sca", "f_opa"}
    assert required.issubset(model.splats.keys())
    for name in required:
        assert model.splats[name].shape[0] == n
    opts = model.create_splat_optimizers()
    assert set(opts) == set(model.splats.keys())


def test_initial_scales_fail_loudly_without_sklearn(monkeypatch):
    import pytest
    from prodig.model import _initial_log_scales

    # a None entry in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "sklearn.neighbors", None)
    with pytest.raises(ImportError, match="scikit-learn"):
        _initial_log_scales(torch.rand(10, 3))
