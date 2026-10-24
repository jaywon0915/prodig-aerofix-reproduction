from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim_index(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Small dependency-free SSIM approximation for BCHW RGB tensors."""
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError("x and y must be BCHW tensors with the same shape")
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return score.mean().clamp(-1.0, 1.0)


def mixed_rgb_loss(
    pred_hwc: torch.Tensor, target_hwc: torch.Tensor, ssim_lambda: float = 0.2
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if pred_hwc.shape != target_hwc.shape or pred_hwc.ndim != 4:
        raise ValueError("pred and target must be BHWC tensors with the same shape")
    l1 = torch.abs(pred_hwc - target_hwc).mean()
    pred = pred_hwc.permute(0, 3, 1, 2)
    target = target_hwc.permute(0, 3, 1, 2)
    ssim_loss = 1.0 - ssim_index(pred, target)
    loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_loss
    return loss, {"l1": l1, "ssim_loss": ssim_loss}
