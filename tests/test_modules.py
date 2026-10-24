import torch

from prodig.modules import AppearanceModule, DistanceAdaptiveModule


def test_distance_gates_are_near_identity_and_bounded():
    mod = DistanceAdaptiveModule(feature_dim=8, hidden_dim=16, init_gate=0.99)
    means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    center = torch.tensor([0.0, 0.0, 2.0])
    f_sca = torch.zeros(2, 8)
    f_opa = torch.zeros(2, 8)
    sg, og = mod(means, center, f_sca, f_opa, scene_scale=2.0)
    assert sg.shape == (2, 1)
    assert og.shape == (2, 1)
    assert torch.all((sg > 0) & (sg < 1))
    assert torch.all((og > 0) & (og < 1))
    assert torch.allclose(sg, torch.full_like(sg, 0.99), atol=1e-4)
    assert torch.allclose(og, torch.full_like(og, 0.99), atol=1e-4)


def test_distance_module_backpropagates_to_features_and_mlp():
    mod = DistanceAdaptiveModule(feature_dim=4, hidden_dim=8, init_gate=0.9)
    means = torch.randn(5, 3)
    center = torch.randn(3)
    f_sca = torch.randn(5, 4, requires_grad=True)
    f_opa = torch.randn(5, 4, requires_grad=True)
    sg, og = mod(means, center, f_sca, f_opa, scene_scale=1.0)
    (sg.mean() + og.mean()).backward()
    assert f_sca.grad is not None
    assert f_opa.grad is not None
    assert any(p.grad is not None for p in mod.parameters())


def test_appearance_module_starts_with_zero_correction():
    mod = AppearanceModule(num_appearances=3, gaussian_feature_dim=6, embed_dim=4, hidden_dim=8)
    features = torch.randn(5, 6)
    dirs = torch.randn(5, 3)
    delta = mod(features, appearance_id=1, dirs=dirs)
    assert delta.shape == (5, 3)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)
