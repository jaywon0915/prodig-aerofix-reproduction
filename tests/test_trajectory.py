import torch

from prodig.camera import CameraRecord
from prodig.trajectory import make_progressive_camera


def _camera():
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.0, 0.0, 10.0])
    # OpenCV camera +z is viewing direction; point it toward scene center.
    c2w[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    return CameraRecord(
        name="a.png", c2w=c2w, K=torch.eye(3), width=100, height=80,
        image_path=None, image_id=0, appearance_id=0, weight=1.0, source="aerial"
    )


def test_forward_trajectory_preserves_rotation_and_moves_toward_scene():
    ref = _camera()
    scene_center = torch.zeros(3)
    novel = make_progressive_camera(ref, 0.5, "forward", scene_center)
    assert torch.allclose(novel.c2w[:3, :3], ref.c2w[:3, :3])
    assert torch.linalg.norm(novel.center - scene_center) < torch.linalg.norm(ref.center - scene_center)
    assert novel.appearance_id == ref.appearance_id


def test_scaled_trajectory_halves_center_radius():
    ref = _camera()
    novel = make_progressive_camera(ref, 0.5, "scaled", torch.zeros(3))
    assert torch.allclose(novel.center, torch.tensor([0.0, 0.0, 5.0]), atol=1e-5)
