from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .camera import CameraRecord


@dataclass
class SceneData:
    points: torch.Tensor
    colors: torch.Tensor
    cameras: list[CameraRecord]
    scene_center: torch.Tensor
    scene_scale: float
    normalization_transform: torch.Tensor


def normalize_scene(
    points: torch.Tensor, c2ws: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    points = torch.as_tensor(points, dtype=torch.float32)
    c2ws = torch.as_tensor(c2ws, dtype=torch.float32).clone()
    centers = c2ws[:, :3, 3]
    center = centers.mean(dim=0)
    radius = torch.linalg.norm(centers - center, dim=-1).max()
    scale = 1.0 / max(float(radius), 1e-8)
    points_out = (points - center) * scale
    c2ws[:, :3, 3] = (centers - center) * scale
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    return points_out, c2ws, transform, scale


def _camera_K(camera) -> np.ndarray:
    name = getattr(camera, "model_name", None)
    if name is None:
        model = getattr(camera, "model", None)
        name = getattr(model, "name", str(model))
    params = np.asarray(camera.params, dtype=np.float64)
    w, h = int(camera.width), int(camera.height)
    if name == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif name == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif name == "SIMPLE_RADIAL" and abs(float(params[3])) < 1e-9:
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise ValueError(
            f"Camera model {name} is not supported directly. Use COLMAP image_undistorter so the model is PINHOLE/SIMPLE_PINHOLE."
        )
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _image_w2c(image) -> np.ndarray:
    pose = image.cam_from_world
    if callable(pose):
        pose = pose()
    matrix = pose.matrix() if hasattr(pose, "matrix") else np.asarray(pose)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3] = matrix
        return out
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"Unsupported pycolmap pose matrix shape: {matrix.shape}")


def load_colmap_scene(data_dir: str | Path, normalize: bool = True) -> SceneData:
    try:
        import pycolmap
    except ModuleNotFoundError as exc:
        raise RuntimeError("pycolmap is required to load COLMAP scenes") from exc

    data_dir = Path(data_dir)
    sparse = data_dir / "sparse" / "0"
    images_dir = data_dir / "images"
    if not sparse.exists():
        raise FileNotFoundError(f"Missing COLMAP model: {sparse}")
    rec = pycolmap.Reconstruction(str(sparse))

    pts = []
    cols = []
    for p in rec.points3D.values():
        pts.append(np.asarray(p.xyz, dtype=np.float32))
        cols.append(np.asarray(p.color, dtype=np.float32) / 255.0)
    if not pts:
        raise ValueError("COLMAP reconstruction has no 3D points")
    points = torch.from_numpy(np.stack(pts))
    colors = torch.from_numpy(np.stack(cols))

    raw = []
    for image in sorted(rec.images.values(), key=lambda x: x.name):
        camera = rec.cameras[image.camera_id]
        K = torch.from_numpy(_camera_K(camera))
        w2c = torch.from_numpy(_image_w2c(image)).float()
        c2w = torch.linalg.inv(w2c)
        path = images_dir / image.name
        if not path.exists():
            raise FileNotFoundError(path)
        raw.append((image, camera, K, c2w, path))
    c2ws = torch.stack([x[3] for x in raw])
    transform = torch.eye(4)
    scale = 1.0
    if normalize:
        points, c2ws, transform, scale = normalize_scene(points, c2ws)

    cameras = []
    for idx, (entry, camera, K, _, path) in enumerate(raw):
        cameras.append(
            CameraRecord(
                name=entry.name,
                c2w=c2ws[idx],
                K=K,
                width=int(camera.width),
                height=int(camera.height),
                image_path=path,
                image_id=idx,
                appearance_id=idx,
                source="aerial",
            )
        )
    return SceneData(
        points=points,
        colors=colors,
        cameras=cameras,
        scene_center=torch.zeros(3) if normalize else c2ws[:, :3, 3].mean(0),
        scene_scale=1.0 if normalize else 1.0 / max(scale, 1e-8),
        normalization_transform=transform,
    )
