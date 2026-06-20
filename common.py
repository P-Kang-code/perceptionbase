from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROT_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10]
TRANS_IDX = [3, 7, 11]


def others_work_dir() -> Path:
    return Path(__file__).resolve().parent


def add_others_work_paths() -> None:
    """Reject the old mixed-author import path hook.

    Each baseline in this package must keep its method-specific implementation
    private. Adding all original-author packages to the same ``sys.path`` made it
    possible for one adapter to silently import another author's helpers, which
    violates the audit requirement even when the current call sites do not use
    the hook.
    """
    raise RuntimeError(
        "add_others_work_paths() is disabled: baselines must not share or mix "
        "original-author implementation modules. Import method-private helpers "
        "inside the corresponding baseline package instead."
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def subset_by_max(indices: np.ndarray, max_samples: int | None) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    return indices[: int(max_samples)]


def checkpoint_path(output_dir: Path, baseline: str, suffix: str = ".pt") -> Path:
    return Path(output_dir) / "checkpoints" / f"{baseline}_best{suffix}"


def mat_to_mat_rot_angle(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    small = 1e-4
    trace = (pred_rot * gt_rot).sum(dim=1)
    trace = torch.clamp(trace, -1.0 + small, 3.0 - small)
    return torch.acos((trace - 1.0) / 2.0)


def compute_pose_metrics(pred_pose_flat: np.ndarray, gt_pose_flat: np.ndarray, scale: np.ndarray | None = None) -> Dict[str, float]:
    pred = torch.from_numpy(pred_pose_flat.astype(np.float32))
    gt = torch.from_numpy(gt_pose_flat.astype(np.float32))
    mse = nn.functional.mse_loss(pred, gt).item()
    rot_rad = mat_to_mat_rot_angle(pred[:, ROT_IDX], gt[:, ROT_IDX]).mean().item()
    trans_abs = torch.abs(pred[:, [3, 7, 11]] - gt[:, [3, 7, 11]])
    if scale is None:
        trans_l1 = trans_abs.mean().item()
    else:
        s = torch.from_numpy(np.asarray(scale, dtype=np.float32).reshape(-1, 1)).clamp_min(1e-8)
        trans_l1 = (trans_abs / s).mean().item()
    return {
        "mse": float(mse),
        "rotation_error_deg": float(rot_rad * 180.0 / np.pi),
        "translation_l1": float(trans_l1),
    }


def flatten_pose_matrix(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=np.float32).reshape(16)


def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rz @ ry @ rx).astype(np.float32)


def pose6_to_matrix(state6: np.ndarray) -> np.ndarray:
    state6 = np.asarray(state6, dtype=np.float32).reshape(6)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_xyz_to_matrix(float(state6[3]), float(state6[4]), float(state6[5]))
    T[:3, 3] = state6[:3]
    return T


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    a1 = rot_6d[:, 0:3]
    a2 = rot_6d[:, 3:6]
    b1 = nn.functional.normalize(a1, dim=1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def pose9d_to_poseflat(pose9d: torch.Tensor) -> torch.Tensor:
    trans = pose9d[:, 0:3]
    rot = rotation_6d_to_matrix(pose9d[:, 3:9])
    out = torch.zeros(pose9d.shape[0], 16, device=pose9d.device, dtype=pose9d.dtype)
    out[:, 0:3] = rot[:, 0, :]
    out[:, 4:7] = rot[:, 1, :]
    out[:, 8:11] = rot[:, 2, :]
    out[:, [3, 7, 11]] = trans
    out[:, 15] = 1.0
    return out


def poseflat_to_pose9d(pose_flat: torch.Tensor) -> torch.Tensor:
    rot = pose_flat[:, ROT_IDX].reshape(-1, 3, 3)
    rot6 = torch.cat([rot[:, :, 0], rot[:, :, 1]], dim=1)
    trans = pose_flat[:, [3, 7, 11]]
    return torch.cat([trans, rot6], dim=1)

def handsom_to_tactile_image(hand_som: torch.Tensor, side: int = 64) -> torch.Tensor:
    values = hand_som[:, :, 3] if hand_som.shape[-1] >= 4 else torch.linalg.norm(hand_som[:, :, :3], dim=-1)
    base_side = int(math.ceil(math.sqrt(max(1, int(values.shape[1])))))
    target = base_side * base_side
    if values.shape[1] < target:
        pad = torch.zeros(values.shape[0], target - values.shape[1], device=values.device, dtype=values.dtype)
        values = torch.cat([values, pad], dim=1)
    mask = values[:, :target].reshape(values.shape[0], 1, base_side, base_side)
    if base_side != int(side):
        mask = torch.nn.functional.interpolate(mask, size=(int(side), int(side)), mode="bilinear", align_corners=False)
    min_v = mask.amin(dim=(1, 2, 3), keepdim=True)
    max_v = mask.amax(dim=(1, 2, 3), keepdim=True)
    return (mask - min_v) / (max_v - min_v + 1e-6)



def object_to_stats(object_data: torch.Tensor) -> torch.Tensor:
    pts = object_data.float()
    if pts.ndim == 3:
        mean = pts.mean(dim=1)
        std = pts.std(dim=1, unbiased=False)
        mn = pts.amin(dim=1)
        mx = pts.amax(dim=1)
        return torch.cat([mean, std, mn, mx], dim=1)
    mean = pts.mean(dim=0)
    std = pts.std(dim=0, unbiased=False)
    mn = pts.amin(dim=0)
    mx = pts.amax(dim=0)
    return torch.cat([mean, std, mn, mx], dim=0)


def _decode_if_bytes(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return value


def _nan_to_num_float(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return array


def _pose_any_to_flat16_sample(pose_raw: np.ndarray) -> np.ndarray:
    pose = _nan_to_num_float(np.asarray(pose_raw, dtype=np.float32))
    if pose.ndim == 2 and pose.shape == (4, 4):
        return pose.reshape(16).astype(np.float32, copy=False)
    pose = pose.reshape(-1)
    if pose.shape[0] == 16:
        return pose.astype(np.float32, copy=False)
    if pose.shape[0] not in {9, 12}:
        raise ValueError(f"Unsupported pose dimension {pose.shape[0]}; expected 9, 12, or 16")
    out = np.zeros((16,), dtype=np.float32)
    out[ROT_IDX] = pose[:9]
    if pose.shape[0] == 12:
        out[TRANS_IDX] = pose[9:12]
    out[15] = 1.0
    return out


def _pose_any_to_flat16_batch(pose_raw: np.ndarray) -> np.ndarray:
    pose = _nan_to_num_float(np.asarray(pose_raw, dtype=np.float32))
    if pose.ndim == 3 and pose.shape[1:] == (4, 4):
        return pose.reshape(pose.shape[0], 16).astype(np.float32, copy=False)
    if pose.ndim != 2:
        raise ValueError(f"Expected pose array [N, D] or [N, 4, 4], got {pose.shape}")
    if pose.shape[1] == 16:
        return pose.astype(np.float32, copy=False)
    if pose.shape[1] not in {9, 12}:
        raise ValueError(f"Unsupported pose dimension {pose.shape[1]}; expected 9, 12, or 16")
    out = np.zeros((pose.shape[0], 16), dtype=np.float32)
    out[:, ROT_IDX] = pose[:, :9]
    if pose.shape[1] == 12:
        out[:, TRANS_IDX] = pose[:, 9:12]
    out[:, 15] = 1.0
    return out


def _force_any_to_force2_sample(force_raw: np.ndarray) -> np.ndarray:
    force = _nan_to_num_float(np.asarray(force_raw, dtype=np.float32)).reshape(1, -1)
    if force.shape[1] < 2:
        raise ValueError(f"Expected force sample with at least 2 values, got {force.shape}")
    return force[0, :2].astype(np.float32, copy=False)


def _force_any_to_force2_batch(force_raw: np.ndarray) -> np.ndarray:
    force = _nan_to_num_float(np.asarray(force_raw, dtype=np.float32))
    if force.ndim == 1:
        force = force.reshape(-1, 1)
    if force.ndim != 2 or force.shape[1] < 2:
        raise ValueError(f"Expected force array [N, >=2], got {force.shape}")
    return force[:, :2].astype(np.float32, copy=False)


def _read_h5_rows(dataset: h5py.Dataset, indices: np.ndarray, dtype: Any | None = None, max_slice_rows: int = 8192) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        shape = (0,) + tuple(int(x) for x in dataset.shape[1:])
        return np.empty(shape, dtype=dtype or dataset.dtype)
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    out_dtype = np.dtype(dtype or dataset.dtype)
    out = np.empty((len(indices),) + tuple(int(x) for x in dataset.shape[1:]), dtype=out_dtype)
    cursor = 0
    while cursor < len(sorted_indices):
        raw_start = int(sorted_indices[cursor])
        raw_limit = raw_start + max(1, int(max_slice_rows))
        next_cursor = int(np.searchsorted(sorted_indices, raw_limit, side="left"))
        next_cursor = max(next_cursor, cursor + 1)
        raw_stop = int(sorted_indices[next_cursor - 1]) + 1
        arr = np.asarray(dataset[raw_start:raw_stop])
        selected = arr[sorted_indices[cursor:next_cursor] - raw_start]
        out[order[cursor:next_cursor]] = selected.astype(out_dtype, copy=False)
        cursor = next_cursor
    return out


def _stack_pose_force_samples(source_dataset: Dataset, local_indices: np.ndarray):
    if local_indices.size == 0:
        return (
            torch.empty((0, 0, 4), dtype=torch.float32),
            torch.empty((0, 0, 3), dtype=torch.float32),
            torch.empty((0, 16), dtype=torch.float32),
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )
    samples = [source_dataset[int(idx)] for idx in local_indices]
    columns = list(zip(*samples))
    return tuple(torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x) for x in col], dim=0) for col in columns)


def _batch_from_cached_dataset(source_dataset: Dataset, local_indices: np.ndarray, include_object: bool):
    batch_from_positions = getattr(source_dataset, "batch_from_positions", None)
    if batch_from_positions is None:
        return None
    try:
        return batch_from_positions(local_indices, include_object=include_object)
    except TypeError:
        batch = batch_from_positions(local_indices)
        if not include_object:
            hand_som, _, pose, force, mask_pose, mask_force, scale = batch
            empty_object = torch.empty((hand_som.shape[0], 0, 3), dtype=hand_som.dtype)
            return (hand_som, empty_object, pose, force, mask_pose, mask_force, scale)
        return batch


def read_pose_force_batch(source_dataset: Dataset, indices: np.ndarray, *, include_object: bool = True):
    local_indices = np.asarray(indices, dtype=np.int64)
    cached = _batch_from_cached_dataset(source_dataset, local_indices, include_object)
    if cached is not None:
        return cached
    lazy_h5 = all(
        hasattr(source_dataset, name)
        for name in (
            "global_indices",
            "_ensure_h5",
            "hand_key",
            "pose_key",
            "force_key",
            "pose_mask_name",
            "force_mask_name",
        )
    )
    if not lazy_h5:
        batch = _stack_pose_force_samples(source_dataset, local_indices)
        if not include_object:
            hand_som, _, pose, force, mask_pose, mask_force, scale = batch
            return (hand_som, torch.empty((hand_som.shape[0], 0, 3), dtype=hand_som.dtype), pose, force, mask_pose, mask_force, scale)
        return batch

    f: h5py.File = source_dataset._ensure_h5()
    global_indices = np.asarray(source_dataset.global_indices[local_indices], dtype=np.int64)
    n = int(len(global_indices))
    hand_som = _nan_to_num_float(_read_h5_rows(f[source_dataset.hand_key], global_indices, dtype=np.float32))
    if include_object:
        object_data = _nan_to_num_float(_read_h5_rows(f["object_data"], global_indices, dtype=np.float32))
    else:
        object_data = np.empty((n, 0, 3), dtype=np.float32)
    pose = _pose_any_to_flat16_batch(_read_h5_rows(f[source_dataset.pose_key], global_indices, dtype=np.float32))
    force = _force_any_to_force2_batch(_read_h5_rows(f[source_dataset.force_key], global_indices, dtype=np.float32))
    scale = (
        _nan_to_num_float(_read_h5_rows(f["scale"], global_indices, dtype=np.float32)).reshape(n, 1)
        if "scale" in f
        else np.ones((n, 1), dtype=np.float32)
    )
    if source_dataset.pose_mask_name in f:
        mask_pose = _read_h5_rows(f[source_dataset.pose_mask_name], global_indices).reshape(n, 1).astype(np.float32)
    else:
        mask_pose = np.ones((n, 1), dtype=np.float32)
    if source_dataset.force_mask_name in f:
        mask_force = _read_h5_rows(f[source_dataset.force_mask_name], global_indices).reshape(n, 1).astype(np.float32)
    else:
        mask_force = np.ones((n, 1), dtype=np.float32)
    if "section" in f:
        sections = np.asarray([str(_decode_if_bytes(x)) for x in _read_h5_rows(f["section"], global_indices)])
        mask_pose = mask_pose * (sections.reshape(n, 1) == "pose").astype(np.float32)
        mask_force = mask_force * (sections.reshape(n, 1) == "force").astype(np.float32)

    return (
        torch.as_tensor(hand_som, dtype=torch.float32),
        torch.as_tensor(object_data, dtype=torch.float32),
        torch.as_tensor(pose, dtype=torch.float32),
        torch.as_tensor(force, dtype=torch.float32),
        torch.as_tensor(mask_pose, dtype=torch.float32),
        torch.as_tensor(mask_force, dtype=torch.float32),
        torch.as_tensor(scale, dtype=torch.float32),
    )


def read_aux_batch(source_dataset: Dataset, indices: np.ndarray) -> Dict[str, Any]:
    local_indices = np.asarray(indices, dtype=np.int64)
    aux_from_positions = getattr(source_dataset, "aux_from_positions", None)
    if aux_from_positions is not None:
        return aux_from_positions(local_indices)

    n = int(len(local_indices))
    lazy_h5 = all(hasattr(source_dataset, name) for name in ("global_indices", "_ensure_h5"))
    if lazy_h5:
        f: h5py.File = source_dataset._ensure_h5()
        global_indices = np.asarray(source_dataset.global_indices[local_indices], dtype=np.int64)
        def read_float(names: tuple[str, ...], width: int) -> torch.Tensor:
            for name in names:
                if name in f:
                    arr = _nan_to_num_float(_read_h5_rows(f[name], global_indices, dtype=np.float32))
                    return torch.as_tensor(arr.reshape(n, -1), dtype=torch.float32)
            return torch.zeros((n, int(width)), dtype=torch.float32)

        def read_str(name: str, default: str = "") -> list[str]:
            if name not in f:
                return [default] * n
            return [str(_decode_if_bytes(x)) for x in _read_h5_rows(f[name], global_indices)]

        return {
            "global_indices": torch.as_tensor(global_indices, dtype=torch.long),
            "wiseglove_angle": read_float(("wiseglove_angle_normalized", "wiseglove_angle_raw"), 19),
            "wiseglove_tactile": read_float(("wiseglove_tactile_normalized", "wiseglove_tactile_raw"), 19),
            "trial_key": read_str("trial_key"),
            "participant": read_str("participant"),
            "block": read_str("block"),
            "section": read_str("section"),
            "_n": n,
        }

    global_indices = getattr(source_dataset, "global_indices", None)
    if global_indices is None:
        global_selected = np.asarray(local_indices, dtype=np.int64)
    else:
        global_selected = np.asarray(global_indices)[local_indices]
    return {
        "global_indices": torch.as_tensor(global_selected, dtype=torch.long),
        "wiseglove_angle": torch.zeros((n, 19), dtype=torch.float32),
        "wiseglove_tactile": torch.zeros((n, 19), dtype=torch.float32),
        "trial_key": [""] * n,
        "participant": [""] * n,
        "block": [""] * n,
        "section": [""] * n,
        "_n": n,
    }


def handsom_tactile_stats(hand_som: torch.Tensor, target: int = 28 * 28) -> torch.Tensor:
    values = hand_som[:, 3] if hand_som.shape[-1] >= 4 else torch.linalg.norm(hand_som[:, :3], dim=-1)
    values = values.float()
    if values.numel() < target:
        values = torch.cat([values, torch.zeros(target - values.numel(), dtype=values.dtype)], dim=0)
    values = values[:target]
    values = (values - values.amin()) / (values.amax() - values.amin() + 1e-6)
    return torch.stack(
        [
            values.mean(),
            values.std(unbiased=False),
            values.amax(),
            values.amin(),
        ]
    )


class HandOnlyPoseForceDataset(Dataset):
    """View that avoids reading object_data for hand-only baseline adapters."""

    def __init__(self, source_dataset: Dataset, indices: np.ndarray):
        self.source_dataset = source_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self._lazy_h5 = all(
            hasattr(source_dataset, name)
            for name in (
                "global_indices",
                "_ensure_h5",
                "hand_key",
                "pose_key",
                "force_key",
                "pose_mask_name",
                "force_mask_name",
            )
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[int(i)])
        cached = _batch_from_cached_dataset(self.source_dataset, np.asarray([idx], dtype=np.int64), include_object=False)
        if cached is not None:
            return tuple(value[0] for value in cached)
        if not self._lazy_h5:
            hand_som, _, pose, force, mask_pose, mask_force, scale = self.source_dataset[idx]
            return hand_som, torch.empty((0, 3), dtype=hand_som.dtype), pose, force, mask_pose, mask_force, scale

        f: h5py.File = self.source_dataset._ensure_h5()
        global_idx = int(self.source_dataset.global_indices[idx])
        hand_som = _nan_to_num_float(np.asarray(f[self.source_dataset.hand_key][global_idx], dtype=np.float32))
        pose = _pose_any_to_flat16_sample(np.asarray(f[self.source_dataset.pose_key][global_idx], dtype=np.float32))
        force = _force_any_to_force2_sample(np.asarray(f[self.source_dataset.force_key][global_idx], dtype=np.float32))
        scale = (
            _nan_to_num_float(np.asarray(f["scale"][global_idx], dtype=np.float32)).reshape(1)
            if "scale" in f
            else np.ones((1,), dtype=np.float32)
        )
        section = str(_decode_if_bytes(f["section"][global_idx])) if "section" in f else "pose_force"
        mask_pose = float(np.asarray(f[self.source_dataset.pose_mask_name][global_idx]).reshape(-1)[0]) if self.source_dataset.pose_mask_name in f else 1.0
        mask_force = float(np.asarray(f[self.source_dataset.force_mask_name][global_idx]).reshape(-1)[0]) if self.source_dataset.force_mask_name in f else 1.0
        if "section" in f:
            mask_pose = mask_pose if section == "pose" else 0.0
            mask_force = mask_force if section == "force" else 0.0

        return (
            torch.as_tensor(hand_som, dtype=torch.float32),
            torch.empty((0, 3), dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(force, dtype=torch.float32),
            torch.as_tensor([mask_pose], dtype=torch.float32),
            torch.as_tensor([mask_force], dtype=torch.float32),
            torch.as_tensor(scale, dtype=torch.float32),
        )


class BatchedHandPoseForceDataset(Dataset):
    def __init__(self, source_dataset: Dataset, indices: np.ndarray, batch_size: int, tag: str = "hand"):
        self.source_dataset = source_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = max(1, int(batch_size))
        self.tag = str(tag)

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_i: int):
        start = int(batch_i) * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        pick = self.indices[start:stop]
        print(f"[data:{self.tag}] loading batch {int(batch_i) + 1}/{len(self)} samples={len(pick)}", flush=True)
        return read_pose_force_batch(self.source_dataset, pick, include_object=False)


def make_hand_pose_loader(
    source_dataset: Dataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    tag: str,
) -> DataLoader:
    dataset = BatchedHandPoseForceDataset(source_dataset, indices, batch_size=batch_size, tag=tag)
    return DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=num_workers)


def print_evaluation_block(tag: str, metrics: Mapping[str, Any]) -> None:
    print(f"\n----- evaluation [{tag}] -----")
    if "rotation_error_deg" in metrics:
        print(f"angle error (deg)       {float(metrics.get('rotation_error_deg', float('nan'))):8.3f}")
    if "translation_l1" in metrics:
        print(f"distance error          {float(metrics.get('translation_l1', float('nan'))):8.4f}")
    if "force_mae" in metrics:
        print(f"force error             {float(metrics.get('force_mae', float('nan'))):8.4f}")
    if "torque_mae" in metrics:
        print(f"torque error            {float(metrics.get('torque_mae', float('nan'))):8.4f}")






