import torch
from prodig.losses import mixed_rgb_loss, ssim_index


def test_ssim_identical_is_one():
    x = torch.rand(1, 3, 16, 16)
    assert torch.allclose(ssim_index(x, x), torch.tensor(1.0), atol=1e-5)


def test_mixed_rgb_loss_zero_for_identical():
    x = torch.rand(1, 8, 8, 3)
    loss, parts = mixed_rgb_loss(x, x, ssim_lambda=0.2)
    assert loss.abs() < 1e-6
    assert parts["l1"].abs() < 1e-6
