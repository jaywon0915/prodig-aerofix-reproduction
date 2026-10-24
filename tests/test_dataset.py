from pathlib import Path
import numpy as np
from PIL import Image
import torch

from prodig.camera import CameraRecord
from prodig.dataset import ProgressiveDataset


def cam(path, image_id=0, appearance_id=0, source="aerial", weight=1.0):
    return CameraRecord(
        name=Path(path).name, c2w=torch.eye(4), K=torch.eye(3), width=4, height=3,
        image_path=Path(path), image_id=image_id, appearance_id=appearance_id,
        weight=weight, source=source,
    )


def test_progressive_dataset_appends_fixed_view_and_loads_tensor(tmp_path):
    p0 = tmp_path / "a.png"
    p1 = tmp_path / "fixed.png"
    Image.fromarray(np.full((3, 4, 3), 128, np.uint8)).save(p0)
    Image.fromarray(np.full((3, 4, 3), 64, np.uint8)).save(p1)
    ds = ProgressiveDataset([cam(p0)])
    ref = ds.records[0]
    fixed = cam(p1, image_id=10, appearance_id=ref.appearance_id, source="fixed", weight=0.7)
    ds.append_fixed(fixed)
    item = ds[1]
    assert len(ds) == 2
    assert item["image"].shape == (3, 4, 3)
    assert item["appearance_id"] == 0
    assert item["weight"] == 0.7
    assert item["source"] == "fixed"
