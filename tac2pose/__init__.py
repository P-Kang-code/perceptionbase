from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .runtime import checkpoint_path, compute_pose_metrics, flatten_pose_matrix, read_pose_force_batch, save_json, subset_by_max


NAME = "tac2pose"


def _contact_descriptor_from_handsom(hand_som: torch.Tensor, bins: int = 40) -> np.ndarray:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((bins * bins,), dtype=np.float32)
    pts = arr[:, :3]
    pressure = np.maximum(arr[:, 3], 0.0) if arr.shape[1] >= 4 else np.linalg.norm(pts, axis=1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(pressure)
    pts = pts[finite]
    pressure = pressure[finite]
    if len(pts) == 0:
        return np.zeros((bins * bins,), dtype=np.float32)
    xy = pts[:, :2]
    mn = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - mn, 1e-6)
    uv = np.clip(((xy - mn) / span * (bins - 1)).round().astype(np.int64), 0, bins - 1)
    img = np.zeros((bins, bins), dtype=np.float32)
    for (u, v), w in zip(uv, pressure):
        img[int(v), int(u)] += float(max(w, 0.0))
    img = img / (float(img.max(initial=0.0)) + 1e-6)
    desc = img.reshape(-1)
    desc = desc - desc.mean()
    norm = float(np.linalg.norm(desc))
    return (desc / norm).astype(np.float32) if norm > 1e-8 else desc.astype(np.float32)


def _contact_centroid_from_handsom(hand_som: torch.Tensor) -> np.ndarray:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros(3, dtype=np.float32)
    pts = arr[:, :3]
    pressure = np.maximum(arr[:, 3], 0.0) if arr.shape[1] >= 4 else np.linalg.norm(pts, axis=1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(pressure)
    pts = pts[finite]
    pressure = pressure[finite]
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    if float(pressure.max(initial=0.0)) > float(pressure.min(initial=0.0)):
        keep = pressure >= np.quantile(pressure, 0.70)
        if keep.sum() >= 3:
            pts = pts[keep]
            pressure = pressure[keep]
    weights = pressure + 1e-6
    return ((pts * weights[:, None]).sum(axis=0) / weights.sum()).astype(np.float32)


def _object_geometry_key(object_points: np.ndarray) -> str:
    obj = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    obj = obj[np.isfinite(obj).all(axis=1)]
    if len(obj) == 0:
        return "empty"
    stats = np.concatenate([obj.mean(axis=0), obj.std(axis=0), obj.min(axis=0), obj.max(axis=0)], axis=0)
    return ",".join(f"{float(v):.4f}" for v in np.round(stats, 4))


def _empty_codebook() -> Dict[str, np.ndarray]:
    return {
        "descriptor": np.zeros((0, 1600), dtype=np.float32),
        "rotation": np.zeros((0, 3, 3), dtype=np.float32),
        "anchor": np.zeros((0, 3), dtype=np.float32),
    }


def _rot_axis(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(float(angle)), np.sin(float(angle))
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _euler_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(float(roll)), np.sin(float(roll))
    cy, sy = np.cos(float(pitch)), np.sin(float(pitch))
    cz, sz = np.cos(float(yaw)), np.sin(float(yaw))
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _rotation_templates(args=None) -> list[np.ndarray]:
    yaw_steps = max(4, int(getattr(args, "tac2pose_yaw_steps", 16)))
    pitch_steps = max(3, int(getattr(args, "tac2pose_pitch_steps", 5)))
    roll_steps = max(3, int(getattr(args, "tac2pose_roll_steps", 5)))
    pitch_limit = float(getattr(args, "tac2pose_pitch_limit_deg", 60.0)) * np.pi / 180.0
    roll_limit = float(getattr(args, "tac2pose_roll_limit_deg", 60.0)) * np.pi / 180.0

    rots: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    def add(R: np.ndarray) -> None:
        key = tuple(np.round(np.asarray(R, dtype=np.float32).reshape(-1), 5).tolist())
        if key not in seen:
            seen.add(key)
            rots.append(np.asarray(R, dtype=np.float32))

    add(np.eye(3, dtype=np.float32))
    for axis in range(3):
        for angle in (np.pi / 2, np.pi, 3 * np.pi / 2):
            add(_rot_axis(axis, angle))
    for yaw in np.linspace(0.0, 2.0 * np.pi, yaw_steps, endpoint=False):
        for pitch in np.linspace(-pitch_limit, pitch_limit, pitch_steps):
            for roll in np.linspace(-roll_limit, roll_limit, roll_steps):
                add(_euler_rotation(float(roll), float(pitch), float(yaw)))
    return rots


def _render_contact_descriptor(object_points: np.ndarray, rotation: np.ndarray, bins: int = 40, keep_quantile: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    obj = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if len(obj) == 0:
        return np.zeros((bins * bins,), dtype=np.float32), np.zeros(3, dtype=np.float32)
    if len(obj) > 4096:
        pick = np.linspace(0, len(obj) - 1, 4096).round().astype(np.int64)
        obj = obj[pick]
    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    pts = obj @ R.T
    depth = pts[:, 2]
    keep = depth >= np.quantile(depth, float(keep_quantile))
    if keep.sum() >= 8:
        pts = pts[keep]
        depth = depth[keep]
    anchor = pts.mean(axis=0).astype(np.float32) if len(pts) else np.zeros(3, dtype=np.float32)
    xy = pts[:, :2]
    mn = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - mn, 1e-6)
    uv = np.clip(((xy - mn) / span * (bins - 1)).round().astype(np.int64), 0, bins - 1)
    weights = depth - depth.min(initial=0.0)
    weights = weights / (weights.max(initial=0.0) + 1e-6) + 1e-3
    img = np.zeros((bins, bins), dtype=np.float32)
    for (u, v), w in zip(uv, weights):
        img[int(v), int(u)] += float(w)
    img = img / (float(img.max(initial=0.0)) + 1e-6)
    desc = img.reshape(-1)
    desc = desc - desc.mean()
    norm = float(np.linalg.norm(desc))
    desc = (desc / norm).astype(np.float32) if norm > 1e-8 else desc.astype(np.float32)
    return desc, anchor


def _build_geometry_codebook(dataset, indices: np.ndarray, args, max_objects: int = 24) -> Dict[str, object]:
    picked = subset_by_max(indices, int(getattr(args, "tac2pose_template_samples", 256)))
    if len(picked) == 0:
        return {"by_object": {}, "global": _empty_codebook()}
    _, objects, _, _, _, _, _ = read_pose_force_batch(dataset, picked, include_object=True)
    by_object: Dict[str, Dict[str, list[np.ndarray]]] = {}
    global_desc, global_rot, global_anchor = [], [], []
    object_limit = min(int(getattr(args, "tac2pose_max_objects", max_objects)), int(objects.shape[0]))
    rotations = _rotation_templates(args)
    for i in range(object_limit):
        obj_np = objects[i].detach().cpu().numpy().astype(np.float32)
        bucket = by_object.setdefault(_object_geometry_key(obj_np), {"descriptor": [], "rotation": [], "anchor": []})
        for R in rotations:
            desc, anchor = _render_contact_descriptor(obj_np, R)
            bucket["descriptor"].append(desc)
            bucket["rotation"].append(R.astype(np.float32))
            bucket["anchor"].append(anchor.astype(np.float32))
            global_desc.append(desc)
            global_rot.append(R.astype(np.float32))
            global_anchor.append(anchor.astype(np.float32))
    if not global_desc:
        return {"by_object": {}, "global": _empty_codebook()}
    packed = {}
    for key, bucket in by_object.items():
        packed[key] = {
            "descriptor": np.stack(bucket["descriptor"], axis=0).astype(np.float32),
            "rotation": np.stack(bucket["rotation"], axis=0).astype(np.float32),
            "anchor": np.stack(bucket["anchor"], axis=0).astype(np.float32),
        }
    return {
        "by_object": packed,
        "global": {
            "descriptor": np.stack(global_desc, axis=0).astype(np.float32),
            "rotation": np.stack(global_rot, axis=0).astype(np.float32),
            "anchor": np.stack(global_anchor, axis=0).astype(np.float32),
        },
        "top_k": int(getattr(args, "tac2pose_retrieval_top_k", 5)),
        "temperature": float(getattr(args, "tac2pose_retrieval_temperature", 0.07)),
        "rotation_grid": {
            "yaw_steps": int(getattr(args, "tac2pose_yaw_steps", 16)),
            "pitch_steps": int(getattr(args, "tac2pose_pitch_steps", 5)),
            "roll_steps": int(getattr(args, "tac2pose_roll_steps", 5)),
            "pitch_limit_deg": float(getattr(args, "tac2pose_pitch_limit_deg", 60.0)),
            "roll_limit_deg": float(getattr(args, "tac2pose_roll_limit_deg", 60.0)),
        },
    }


def _retrieve_pose(hand_som: torch.Tensor, object_points: torch.Tensor, codebook: Dict[str, object]) -> np.ndarray | None:
    by_object = codebook.get("by_object", {}) if isinstance(codebook, dict) else {}
    key = _object_geometry_key(object_points.detach().cpu().numpy().astype(np.float32))
    object_book = by_object.get(key) if isinstance(by_object, dict) else None
    if object_book is None:
        object_book = codebook.get("global", _empty_codebook()) if isinstance(codebook, dict) else _empty_codebook()
    descs = np.asarray(object_book.get("descriptor", np.zeros((0, 1600), dtype=np.float32)), dtype=np.float32)
    rotations = np.asarray(object_book.get("rotation", np.zeros((0, 3, 3), dtype=np.float32)), dtype=np.float32)
    anchors = np.asarray(object_book.get("anchor", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if descs.shape[0] == 0:
        return None
    query = _contact_descriptor_from_handsom(hand_som).reshape(1, -1)
    scores = (query @ descs.T).reshape(-1)
    top_k = min(max(1, int(codebook.get("top_k", 5) if isinstance(codebook, dict) else 5)), int(scores.shape[0]))
    top = np.argpartition(-scores, top_k - 1)[:top_k]
    top = top[np.argsort(-scores[top])]
    temp = max(1e-3, float(codebook.get("temperature", 0.07) if isinstance(codebook, dict) else 0.07))
    logits = (scores[top] - float(scores[top].max())) / temp
    weights = np.exp(logits).astype(np.float32)
    weights = weights / (float(weights.sum()) + 1e-8)
    best = int(top[0])
    R = rotations[best]
    anchor = (anchors[top] * weights[:, None]).sum(axis=0)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = _contact_centroid_from_handsom(hand_som) - anchor
    if not np.isfinite(T).all():
        return None
    return flatten_pose_matrix(T)


def _evaluate(dataset, indices: np.ndarray, max_samples: int | None, args, codebook: Dict[str, object], batch_size: int = 64) -> Dict[str, float]:
    picked = subset_by_max(indices, max_samples)
    preds, gts, scales, positions = [], [], [], []
    skipped_invalid = 0
    print(f"[{NAME}] geometry-rendered contact codebook evaluate samples={len(picked)}", flush=True)
    for start in range(0, len(picked), max(1, int(batch_size))):
        stop = min(start + int(batch_size), len(picked))
        hand, objects, pose, _, mask_pose, _, scale = read_pose_force_batch(dataset, picked[start:stop], include_object=True)
        for i in range(stop - start):
            if float(mask_pose[i].reshape(-1)[0].item()) <= 0.5:
                continue
            pred = _retrieve_pose(hand[i], objects[i], codebook)
            if pred is None:
                skipped_invalid += 1
                continue
            preds.append(pred)
            gts.append(pose[i].numpy())
            scales.append(scale[i].numpy())
            positions.append(int(picked[start + i]))
    if not preds:
        return {"mse": float("inf"), "rotation_error_deg": float("inf"), "translation_l1": float("inf"), "num_test_samples": 0, "skipped_invalid_samples": int(skipped_invalid)}
    metrics = compute_pose_metrics(np.stack(preds, axis=0), np.stack(gts, axis=0), scale=np.stack(scales, axis=0))
    metrics["num_test_samples"] = int(len(preds))
    metrics["skipped_invalid_samples"] = int(skipped_invalid)
    if getattr(args, "save_per_sample", True):
        metrics["per_sample_predictions"] = {
            "positions": np.asarray(positions, dtype=np.int64),
            "pred_pose": np.stack(preds, axis=0).astype(np.float32),
            "target_pose": np.stack(gts, axis=0).astype(np.float32),
            "mask_pose": np.ones((len(preds), 1), dtype=np.float32),
            "scale": np.stack(scales, axis=0).astype(np.float32),
        }
    return metrics


def train(args, dataset, train_idx, val_idx, test_idx, device: torch.device) -> Path:
    del val_idx, test_idx, device
    train_pick = subset_by_max(train_idx, args.max_train_samples)
    codebook = _build_geometry_codebook(dataset, train_pick, args)
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline": NAME,
            "method": "dense geometry-rendered contact codebook retrieval",
            "paper_structure": "contact mask/codebook pose retrieval with object-specific dense rendered templates and top-k distribution-style matching",
            "r2_adaptation": "R2 hand_som pressure/contact samples replace Tac2Pose tactile-to-contact-mask calibration; object_data points render dense contact templates over a configurable SO(3) grid.",
            "codebook": codebook,
            "train_size": int(len(train_pick)),
            "num_templates": int(codebook.get("global", _empty_codebook())["descriptor"].shape[0]),
            "num_object_codebooks": int(len(codebook.get("by_object", {}))),
            "split_file": str(args.split_file),
            "args": vars(args),
        },
        ckpt,
    )
    return ckpt


def test(args, dataset, test_idx, device: torch.device, checkpoint: Path | None = None) -> Dict[str, float]:
    del device
    ckpt = checkpoint or checkpoint_path(args.output_dir, NAME, suffix=".pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"{NAME} checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    metrics = _evaluate(dataset, test_idx, args.max_test_samples, args, payload.get("codebook", {}), batch_size=args.batch_size)
    metrics["checkpoint"] = str(ckpt)
    return metrics



