from pathlib import Path
import torch
from prodig.camera import CameraRecord
from prodig.trainer import camera_from_dict, camera_to_dict, select_stage_references


def _cam(i):
    c2w = torch.eye(4)
    c2w[0, 3] = float(i)
    return CameraRecord(
        name=f"{i}.png", c2w=c2w, K=torch.eye(3), width=4, height=3,
        image_path=Path(f"/tmp/{i}.png"), image_id=i, appearance_id=i,
    )


def test_camera_manifest_roundtrip():
    c = _cam(2)
    out = camera_from_dict(camera_to_dict(c))
    assert out.name == c.name
    assert out.image_path == c.image_path
    assert torch.allclose(out.c2w, c.c2w)
    assert torch.allclose(out.K, c.K)


def test_reference_selection_is_deterministic_and_capped():
    cams = [_cam(i) for i in range(10)]
    picked = select_stage_references(cams, max_views=4)
    assert [c.image_id for c in picked] == [0, 3, 6, 9]
    assert select_stage_references(cams, 0) == cams
