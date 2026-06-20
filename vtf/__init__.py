from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .runtime import checkpoint_path, compute_pose_metrics, flatten_pose_matrix, load_json, read_pose_force_batch, save_json, subset_by_max


NAME = "vtf"


def _rot_axis(axis: int, angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _initial_rotations() -> list[np.ndarray]:
    rots = [np.eye(3, dtype=np.float32)]
    for axis in range(3):
        for angle in (90.0, 180.0, 270.0):
            rots.append(_rot_axis(axis, angle))
    return rots


def _weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 1e-8)
    w = w / w.sum()
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    ca = (a * w[:, None]).sum(axis=0)
    cb = (b * w[:, None]).sum(axis=0)
    aa = a - ca
    bb = b - cb
    h = (aa * w[:, None]).T @ bb
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = cb - r @ ca
    return r.astype(np.float32), t.astype(np.float32)


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) <= int(max_points):
        return points
    pick = np.linspace(0, len(points) - 1, int(max_points)).round().astype(np.int64)
    return points[pick]


def _target_from_handsom(hand_som: torch.Tensor, max_points: int, pressure_quantile: float = 0.50, weight_power: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    pts = arr[:, :3] if arr.shape[1] >= 3 else np.zeros((arr.shape[0], 3), dtype=np.float32)
    if arr.shape[1] >= 4:
        weights = np.maximum(arr[:, 3], 0.0)
    else:
        weights = np.linalg.norm(pts, axis=1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(weights)
    pts = pts[finite]
    weights = weights[finite]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if float(weights.max(initial=0.0)) > float(weights.min(initial=0.0)):
        q = float(np.clip(pressure_quantile, 0.0, 0.90))
        keep = weights >= np.quantile(weights, q)
        if keep.sum() >= 8:
            pts = pts[keep]
            weights = weights[keep]
    if len(pts) > max_points:
        pick = np.linspace(0, len(pts) - 1, int(max_points)).round().astype(np.int64)
        pts = pts[pick]
        weights = weights[pick]
    weights = weights - weights.min(initial=0.0)
    weights = weights / (weights.max(initial=0.0) + 1e-6) + 1e-3
    weights = np.power(weights, max(0.25, float(weight_power)))
    return pts.astype(np.float32), weights.astype(np.float32)


def _nearest_neighbors(src_tf: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = torch.as_tensor(src_tf, dtype=torch.float32)
    tgt = torch.as_tensor(target, dtype=torch.float32)
    best_idx = []
    best_dist = []
    chunk = 2048
    for start in range(0, len(src), chunk):
        dist = torch.cdist(src[start:start + chunk], tgt)
        d, idx = dist.min(dim=1)
        best_idx.append(idx.cpu().numpy())
        best_dist.append(d.cpu().numpy())
    return np.concatenate(best_idx), np.concatenate(best_dist)


def _weighted_icp(
    object_points: np.ndarray,
    target_points: np.ndarray,
    target_weights: np.ndarray,
    *,
    iterations: int = 50,
    tolerance: float = 1e-5,
    max_model_points: int = 4096,
    correspondence_trim_quantile: float = 1.0,
) -> tuple[np.ndarray, float]:
    source = _downsample_points(object_points, max_model_points)
    if len(source) < 3 or len(target_points) < 3:
        return np.eye(4, dtype=np.float32), float("inf")
    best_t = np.eye(4, dtype=np.float32)
    best_err = float("inf")
    src_centroid = source.mean(axis=0)
    tgt_centroid = target_points.mean(axis=0)
    for init_r in _initial_rotations():
        r = init_r.astype(np.float32)
        t = (tgt_centroid - r @ src_centroid).astype(np.float32)
        prev = float("inf")
        for _ in range(int(iterations)):
            transformed = (source @ r.T) + t.reshape(1, 3)
            nn_idx, nn_dist = _nearest_neighbors(transformed, target_points)
            matched = target_points[nn_idx]
            weights = target_weights[nn_idx]
            trim_q = float(np.clip(correspondence_trim_quantile, 0.25, 1.0))
            if trim_q < 0.999 and len(nn_dist) >= 8:
                keep = nn_dist <= np.quantile(nn_dist, trim_q)
                if int(keep.sum()) >= 3:
                    src_fit = source[keep]
                    matched_fit = matched[keep]
                    weights_fit = weights[keep]
                    dist_fit = nn_dist[keep]
                else:
                    src_fit, matched_fit, weights_fit, dist_fit = source, matched, weights, nn_dist
            else:
                src_fit, matched_fit, weights_fit, dist_fit = source, matched, weights, nn_dist
            r_new, t_new = _weighted_kabsch(src_fit, matched_fit, weights_fit)
            mean_err = float(np.average(dist_fit, weights=np.maximum(weights_fit, 1e-6)))
            r, t = r_new, t_new
            if abs(prev - mean_err) < tolerance:
                break
            prev = mean_err
        transformed = (source @ r.T) + t.reshape(1, 3)
        _, nn_dist = _nearest_neighbors(transformed, target_points)
        trim_q = float(np.clip(correspondence_trim_quantile, 0.25, 1.0))
        if trim_q < 0.999 and len(nn_dist) >= 8:
            nn_dist = nn_dist[nn_dist <= np.quantile(nn_dist, trim_q)]
        err = float(nn_dist.mean())
        if err < best_err:
            best_err = err
            best_t = np.eye(4, dtype=np.float32)
            best_t[:3, :3] = r
            best_t[:3, 3] = t
    return best_t, best_err


def _cfg_get(args, overrides: Dict[str, float] | None, name: str, default):
    if overrides and name in overrides:
        return overrides[name]
    return getattr(args, name, default)


def _evaluate(dataset, indices: np.ndarray, max_samples: int | None, args, batch_size: int = 64, config_overrides: Dict[str, float] | None = None) -> Dict[str, float]:
    picked = subset_by_max(indices, max_samples)
    preds, gts, scales, positions = [], [], [], []
    skipped_invalid = 0
    print(f"[{NAME}] weighted ICP evaluate samples={len(picked)} batch_size={batch_size}", flush=True)
    for start in range(0, len(picked), max(1, int(batch_size))):
        stop = min(start + int(batch_size), len(picked))
        hand_batch, object_batch, pose_batch, _, mask_pose_batch, _, scale_batch = read_pose_force_batch(dataset, picked[start:stop], include_object=True)
        for local_i in range(stop - start):
            if float(mask_pose_batch[local_i].reshape(-1)[0].item()) <= 0.5:
                continue
            target, weights = _target_from_handsom(
                hand_batch[local_i],
                int(_cfg_get(args, config_overrides, "vtf_max_target_points", 1024)),
                pressure_quantile=float(_cfg_get(args, config_overrides, "vtf_pressure_quantile", 0.50)),
                weight_power=float(_cfg_get(args, config_overrides, "vtf_weight_power", 1.0)),
            )
            object_points = object_batch[local_i].detach().cpu().numpy().astype(np.float32)
            pred_t, icp_err = _weighted_icp(
                object_points,
                target,
                weights,
                iterations=int(_cfg_get(args, config_overrides, "vtf_icp_iterations", 50)),
                tolerance=float(_cfg_get(args, config_overrides, "vtf_icp_tolerance", 1e-5)),
                max_model_points=int(_cfg_get(args, config_overrides, "vtf_max_model_points", 4096)),
                correspondence_trim_quantile=float(_cfg_get(args, config_overrides, "vtf_correspondence_trim_quantile", 1.0)),
            )
            if not np.isfinite(icp_err) or not np.isfinite(pred_t).all():
                skipped_invalid += 1
                continue
            preds.append(flatten_pose_matrix(pred_t))
            gts.append(pose_batch[local_i].numpy())
            scales.append(scale_batch[local_i].numpy())
            positions.append(int(picked[start + local_i]))
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
    del test_idx, device
    candidates = [
        {"vtf_pressure_quantile": 0.50, "vtf_weight_power": 1.0, "vtf_correspondence_trim_quantile": 1.0},
        {"vtf_pressure_quantile": 0.60, "vtf_weight_power": 1.0, "vtf_correspondence_trim_quantile": 0.90},
        {"vtf_pressure_quantile": 0.70, "vtf_weight_power": 1.5, "vtf_correspondence_trim_quantile": 0.85},
        {"vtf_pressure_quantile": 0.40, "vtf_weight_power": 0.75, "vtf_correspondence_trim_quantile": 0.95},
    ]
    best_config: Dict[str, float] | None = None
    best_train_metrics: Dict[str, float] | None = None
    best_val_metrics: Dict[str, float] | None = None
    for i, candidate in enumerate(candidates, start=1):
        train_metrics = _evaluate(dataset, train_idx, args.max_train_samples, args, batch_size=args.batch_size, config_overrides=candidate)
        val_metrics = _evaluate(dataset, val_idx, args.max_val_samples, args, batch_size=args.batch_size, config_overrides=candidate)
        val_score = float(val_metrics.get("mse", float("inf")))
        print(
            f"[{NAME}] candidate {i}/{len(candidates)} "
            f"train-mse={float(train_metrics.get('mse', float('inf'))):.6f} "
            f"val-mse={val_score:.6f} config={candidate}",
            flush=True,
        )
        if best_val_metrics is None or val_score < float(best_val_metrics.get("mse", float("inf"))):
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            best_config = dict(candidate)
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".json")
    save_json(
        ckpt,
        {
            "baseline": NAME,
            "method": "adapted weighted point-cloud ICP",
            "paper_structure": "modality-weighted point-cloud fusion + validation-selected weighted ICP + robust correspondence trimming + 90-degree multi-start rotations",
            "r2_adaptation": "R2 lacks RGB-D and GelSight point clouds; hand_som 3D samples are used as weighted tactile/proprioceptive observations, with train/validation split selection of contact filtering and ICP weighting.",
            "best_config": best_config or {},
            "train_metrics": {k: v for k, v in (best_train_metrics or {}).items() if k != "per_sample_predictions"},
            "val_metrics": {k: v for k, v in (best_val_metrics or {}).items() if k != "per_sample_predictions"},
            "icp_iterations": int(getattr(args, "vtf_icp_iterations", 50)),
            "icp_tolerance": float(getattr(args, "vtf_icp_tolerance", 1e-5)),
            "split_file": str(args.split_file),
            "args": vars(args),
        },
    )
    return ckpt


def test(args, dataset, test_idx, device: torch.device, checkpoint: Path | None = None) -> Dict[str, float]:
    del device
    ckpt = checkpoint or checkpoint_path(args.output_dir, NAME, suffix=".json")
    if not ckpt.exists():
        raise FileNotFoundError(f"{NAME} checkpoint not found: {ckpt}")
    config = load_json(ckpt).get("best_config", {})
    metrics = _evaluate(dataset, test_idx, args.max_test_samples, args, batch_size=args.batch_size, config_overrides=config)
    metrics["checkpoint"] = str(ckpt)
    return metrics



