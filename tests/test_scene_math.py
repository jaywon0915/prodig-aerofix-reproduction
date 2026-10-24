import torch

from prodig.colmap_scene import normalize_scene


def test_scene_normalization_centers_cameras_and_scales_extent():
    points = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    c2ws = torch.eye(4).repeat(2, 1, 1)
    c2ws[0, :3, 3] = torch.tensor([0., 0., 10.])
    c2ws[1, :3, 3] = torch.tensor([0., 0., 20.])
    p2, c2, transform, scale = normalize_scene(points, c2ws)
    centers = c2[:, :3, 3]
    assert torch.allclose(centers.mean(0), torch.zeros(3), atol=1e-6)
    assert torch.max(torch.linalg.norm(centers, dim=-1)) <= 1.0 + 1e-6
    assert p2.shape == points.shape
    assert transform.shape == (4, 4)
    assert scale > 0
