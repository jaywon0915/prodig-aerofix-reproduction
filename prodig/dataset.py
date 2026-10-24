from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .camera import CameraRecord


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy())


class ProgressiveDataset(Dataset):
    def __init__(self, records: Iterable[CameraRecord]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("ProgressiveDataset requires at least one record")

    def __len__(self) -> int:
        return len(self.records)

    def append_fixed(self, record: CameraRecord) -> None:
        if record.image_path is None:
            raise ValueError("Fixed view must have an image_path")
        if record.source != "fixed":
            raise ValueError("Appended synthetic record must use source='fixed'")
        self.records.append(record)

    def drop_reference(self, reference_name: str) -> bool:
        """Remove whichever record currently represents `reference_name`.

        That is either the pristine original (`rec.name == reference_name`,
        `source='aerial'`) or an earlier stage's still-present correction for
        the same camera (`rec.reference_name == reference_name`,
        `source='fixed'`). Call this right before appending a newly accepted
        fixed view so each reference camera contributes at most one training
        image at a time -- the latest accepted AeroFix correction, never the
        original and never a superseded earlier-stage correction. The
        original image file on disk is untouched -- ProDiGTrainer keeps its
        own `original_cameras` list for AeroFix reference lookups across
        every stage, independent of what this dataset currently trains on.
        """
        for i, rec in enumerate(self.records):
            if rec.source == "aerial" and rec.name == reference_name:
                del self.records[i]
                return True
            if rec.source == "fixed" and rec.reference_name == reference_name:
                del self.records[i]
                return True
        return False

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        if rec.image_path is None:
            raise ValueError(f"Camera {rec.name} has no image")
        image = load_rgb(rec.image_path)
        if image.shape[:2] != (rec.height, rec.width):
            raise ValueError(
                f"Image size {tuple(image.shape[:2])} != camera {(rec.height, rec.width)} for {rec.name}"
            )
        return {
            "image": image,
            "c2w": rec.c2w,
            "K": rec.K,
            "image_id": rec.image_id,
            "appearance_id": rec.appearance_id,
            "weight": float(rec.weight),
            "source": rec.source,
            "name": rec.name,
            "record_index": index,
        }
