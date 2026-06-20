from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .runtime import (
    checkpoint_path,
    compute_pose_metrics,
    flatten_pose_matrix,
    read_aux_batch,
    read_pose_force_batch,
    save_json,
    subset_by_max,
    sync_if_cuda,
)


NAME = "tegtrack"


def _handsom_to_tactile_image(hand_som: torch.Tensor, side: int = 64) -> torch.Tensor:
    values = hand_som[..., 3] if hand_som.shape[-1] >= 4 else torch.linalg.norm(hand_som[..., :3], dim=-1)
    base_side = int(math.ceil(math.sqrt(max(1, int(values.shape[1])))))
    target = base_side * base_side
    if values.shape[1] < target:
        pad = torch.zeros(values.shape[0], target - values.shape[1], dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad], dim=1)
    img = values[:, :target].reshape(values.shape[0], 1, base_side, base_side)
    img = torch.nn.functional.interpolate(img, size=(side, side), mode="bilinear", align_corners=False)
    mn = img.amin(dim=(1, 2, 3), keepdim=True)
    mx = img.amax(dim=(1, 2, 3), keepdim=True)
    return (img - mn) / (mx - mn + 1e-6)


def _object_stats(object_data: torch.Tensor) -> torch.Tensor:
    pts = object_data.float()
    return torch.cat([pts.mean(dim=1), pts.std(dim=1, unbiased=False), pts.amin(dim=1), pts.amax(dim=1)], dim=1)


def _rotmat_to_rotvec_np(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32)


def _rotvec_to_rotmat_np(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = rv / theta
    kx = np.asarray([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)
    return R.astype(np.float32)


def _compose_velocity(pose_flat: np.ndarray, velocity6: np.ndarray) -> np.ndarray:
    T = np.asarray(pose_flat, dtype=np.float32).reshape(4, 4)
    dT = np.eye(4, dtype=np.float32)
    dT[:3, 3] = np.asarray(velocity6[:3], dtype=np.float32)
    dT[:3, :3] = _rotvec_to_rotmat_np(np.asarray(velocity6[3:6], dtype=np.float32))
    return (dT @ T).astype(np.float32).reshape(16)


def _relative_velocity_np(prev_pose_flat: np.ndarray, pose_flat: np.ndarray) -> np.ndarray:
    prev = np.asarray(prev_pose_flat, dtype=np.float32).reshape(4, 4)
    cur = np.asarray(pose_flat, dtype=np.float32).reshape(4, 4)
    dT = cur @ np.linalg.pinv(prev)
    return np.concatenate([dT[:3, 3], _rotmat_to_rotvec_np(dT[:3, :3])], axis=0).astype(np.float32)


class TegTrackSlipVelocityPredictor(nn.Module):
    """TEG-Track-private slip velocity predictor.

    The original method uses a ResNet on adjacent tactile-image differences when
    marker-flow constraints are unreliable during slip. R2 lacks RGB GelSight
    frames, so this keeps only the allowed sensor substitution: pressure maps and
    WiseGlove increments replace tactile RGB image differences. The head predicts
    frame-to-frame [v, omega], never absolute pose, and is not shared.
    """

    def __init__(self, image_side: int = 32, angle_dim: int = 19):
        super().__init__()
        self.image_side = int(image_side)
        self.angle_dim = int(angle_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + self.angle_dim + 3, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 6),
        )

    def forward(self, tactile_pair: torch.Tensor, angle_delta: torch.Tensor, flow_stats: torch.Tensor) -> torch.Tensor:
        z_img = self.image_encoder(tactile_pair).flatten(1)
        z = torch.cat([z_img, angle_delta, flow_stats], dim=1)
        return self.head(z)


def _contact_points(hand_som: torch.Tensor, force: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.size == 0 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    pts = arr[:, :3]
    pressure = arr[:, 3] if arr.shape[1] >= 4 else np.linalg.norm(pts - pts.mean(axis=0, keepdims=True), axis=1)
    if force is not None:
        f = force.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if f.size:
            pressure = pressure * (1.0 + float(np.linalg.norm(f)))
    pressure = np.maximum(pressure, 0.0)
    if float(pressure.max(initial=0.0)) <= 1e-8:
        pressure = np.ones_like(pressure, dtype=np.float32)
    thresh = max(float(np.quantile(pressure, 0.75)), float(pressure.mean()))
    keep = pressure >= thresh
    if int(keep.sum()) < 4:
        keep = pressure >= np.partition(pressure, -min(4, pressure.size))[-min(4, pressure.size)]
    weights = pressure[keep].astype(np.float32)
    weights = weights / (float(weights.sum()) + 1e-8)
    return pts[keep].astype(np.float32), weights


def _weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(3, dtype=np.float32)
    return np.sum(points * weights.reshape(-1, 1), axis=0).astype(np.float32)


def _noslip_velocity_from_contacts(prev_points: np.ndarray, points: np.ndarray, weights: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    n = min(prev_points.shape[0], points.shape[0])
    if n < 3:
        velocity = np.zeros(6, dtype=np.float32)
        velocity[:3] = _weighted_centroid(points, weights) - _weighted_centroid(prev_points, weights[: prev_points.shape[0]])
        return velocity
    p = points[:n].astype(np.float32)
    v_contact = (points[:n] - prev_points[:n]).astype(np.float32)
    w = weights[:n].astype(np.float32)
    w = w / (float(w.sum()) + 1e-8)
    A_blocks = []
    b_blocks = []
    for pi, vi, wi in zip(p, v_contact, w):
        r = pi - pivot.reshape(3)
        cross = np.asarray([[0.0, r[2], -r[1]], [-r[2], 0.0, r[0]], [r[1], -r[0], 0.0]], dtype=np.float32)
        A_blocks.append(np.sqrt(wi) * np.concatenate([np.eye(3, dtype=np.float32), cross], axis=1))
        b_blocks.append(np.sqrt(wi) * vi)
    A = np.concatenate(A_blocks, axis=0)
    b = np.concatenate(b_blocks, axis=0)
    try:
        return np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)
    except np.linalg.LinAlgError:
        out = np.zeros(6, dtype=np.float32)
        out[:3] = np.average(v_contact, axis=0, weights=w)
        return out


def _wiseglove_rotation_delta(angle_now: torch.Tensor | None, angle_prev: torch.Tensor | None, gain: float) -> np.ndarray:
    if angle_now is None or angle_prev is None:
        return np.zeros(3, dtype=np.float32)
    delta = (angle_now.detach().cpu().numpy().reshape(-1) - angle_prev.detach().cpu().numpy().reshape(-1)).astype(np.float32)
    if delta.size == 0:
        return np.zeros(3, dtype=np.float32)
    chunks = np.array_split(delta, 3)
    return np.asarray([float(c.mean()) if c.size else 0.0 for c in chunks], dtype=np.float32) * float(gain)


def _nearest_object_surface(object_points: np.ndarray, query_obj: np.ndarray, max_points: int = 2048) -> np.ndarray:
    pts = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] > max_points:
        step = int(math.ceil(pts.shape[0] / max_points))
        pts = pts[::step]
    diff = query_obj[:, None, :] - pts[None, :, :]
    idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    return pts[idx]


def _geometric_contact_refine(pose_flat: np.ndarray, hand_points: np.ndarray, weights: np.ndarray, object_points: np.ndarray, gain: float) -> np.ndarray:
    if hand_points.shape[0] == 0 or object_points.size == 0:
        return pose_flat
    T = np.asarray(pose_flat, dtype=np.float32).reshape(4, 4).copy()
    R = T[:3, :3]
    t = T[:3, 3]
    query_obj = (hand_points - t.reshape(1, 3)) @ R
    nearest_obj = _nearest_object_surface(object_points, query_obj)
    nearest_world = nearest_obj @ R.T + t.reshape(1, 3)
    residual = np.sum((hand_points - nearest_world) * weights.reshape(-1, 1), axis=0)
    T[:3, 3] = t + float(gain) * residual.astype(np.float32)
    return T.reshape(16).astype(np.float32)


def _initialize_pose_from_contacts(hand_points: np.ndarray, weights: np.ndarray, object_points: np.ndarray) -> np.ndarray | None:
    T = np.eye(4, dtype=np.float32)
    pts = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0 or hand_points.shape[0] == 0:
        return None
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    contact_center = np.sum(hand_points * w, axis=0) / float(np.sum(w) + 1e-8)
    object_anchor = np.mean(pts, axis=0)
    T[:3, 3] = (contact_center - object_anchor).astype(np.float32)
    return T.reshape(16)


def _batch_iter(dataset, indices: np.ndarray, batch_size: int, tag: str):
    indices = np.asarray(indices, dtype=np.int64)
    batch_size = max(1, int(batch_size))
    total = int(math.ceil(len(indices) / batch_size)) if len(indices) else 0
    for batch_i, start in enumerate(range(0, len(indices), batch_size), start=1):
        pick = indices[start : start + batch_size]
        print(f"[data:{tag}] loading batch {batch_i}/{total} samples={len(pick)}", flush=True)
        yield pick


def _make_slip_training_batch(dataset, pick: np.ndarray, config: Mapping[str, float]) -> dict[str, torch.Tensor] | None:
    prev_pick = np.maximum(np.asarray(pick, dtype=np.int64) - 1, 0)
    hand, _, pose, force, mask_pose, _, _ = read_pose_force_batch(dataset, pick, include_object=False)
    prev_hand, _, prev_pose, _, prev_mask_pose, _, _ = read_pose_force_batch(dataset, prev_pick, include_object=False)
    aux = read_aux_batch(dataset, pick)
    prev_aux = read_aux_batch(dataset, prev_pick)
    trials = list(aux.get("trial_key", [""] * len(pick)))
    prev_trials = list(prev_aux.get("trial_key", [""] * len(pick)))
    angle = aux.get("wiseglove_angle", torch.zeros((len(pick), 19), dtype=torch.float32))
    prev_angle = prev_aux.get("wiseglove_angle", torch.zeros_like(angle))
    tactile = _handsom_to_tactile_image(hand, side=int(config.get("image_side", 32)))
    prev_tactile = _handsom_to_tactile_image(prev_hand, side=int(config.get("image_side", 32)))
    targets, keep, flow_stats = [], [], []
    threshold = float(config.get("slip_threshold", 0.006))
    for i in range(len(pick)):
        same = trials[i] == prev_trials[i] and trials[i] != "" and int(pick[i]) == int(prev_pick[i]) + 1
        valid = same and float(mask_pose[i].reshape(-1)[0].item()) > 0.5 and float(prev_mask_pose[i].reshape(-1)[0].item()) > 0.5
        pts, weights = _contact_points(hand[i], force[i])
        prev_pts, prev_weights = _contact_points(prev_hand[i], None)
        flow = _weighted_centroid(pts, weights) - _weighted_centroid(prev_pts, prev_weights)
        slip = float(np.linalg.norm(flow)) > threshold
        keep.append(bool(valid and slip))
        targets.append(_relative_velocity_np(prev_pose[i].detach().cpu().numpy(), pose[i].detach().cpu().numpy()))
        flow_stats.append([float(np.linalg.norm(flow)), float(flow.mean()), float(flow.std())])
    keep_t = torch.as_tensor(keep, dtype=torch.bool)
    if int(keep_t.sum().item()) == 0:
        return None
    return {
        "tactile_pair": torch.cat([tactile, tactile - prev_tactile], dim=1)[keep_t].float(),
        "angle_delta": (angle - prev_angle)[keep_t].float(),
        "flow_stats": torch.as_tensor(np.asarray(flow_stats, dtype=np.float32))[keep_t],
        "target_velocity": torch.as_tensor(np.asarray(targets, dtype=np.float32))[keep_t],
    }


def _calibrate_slip_threshold(dataset, indices: np.ndarray, config: Mapping[str, float], args) -> dict[str, float]:
    if bool(getattr(args, "tegtrack_disable_slip_calibration", False)):
        return {"enabled": False, "threshold": float(config.get("slip_threshold", 0.006)), "reason": "disabled"}
    max_samples = getattr(args, "tegtrack_slip_calibration_samples", None)
    if max_samples is None:
        max_samples = getattr(args, "max_train_samples", None)
    max_samples = 0 if max_samples is None else int(max_samples)
    picked = subset_by_max(indices, max_samples)
    if len(picked) == 0:
        return {"enabled": False, "threshold": float(config.get("slip_threshold", 0.006)), "reason": "empty_train_split"}
    batch_size = max(1, int(getattr(args, "batch_size", 32)))
    flow_norms = []
    for pick in _batch_iter(dataset, picked, batch_size, f"{NAME}-slip-calibration"):
        prev_pick = np.maximum(np.asarray(pick, dtype=np.int64) - 1, 0)
        hand, _, pose, force, mask_pose, _, _ = read_pose_force_batch(dataset, pick, include_object=False)
        prev_hand, _, prev_pose, _, prev_mask_pose, _, _ = read_pose_force_batch(dataset, prev_pick, include_object=False)
        aux = read_aux_batch(dataset, pick)
        prev_aux = read_aux_batch(dataset, prev_pick)
        trials = list(aux.get("trial_key", [""] * len(pick)))
        prev_trials = list(prev_aux.get("trial_key", [""] * len(pick)))
        for i in range(len(pick)):
            same = trials[i] == prev_trials[i] and trials[i] != "" and int(pick[i]) == int(prev_pick[i]) + 1
            valid = same and float(mask_pose[i].reshape(-1)[0].item()) > 0.5 and float(prev_mask_pose[i].reshape(-1)[0].item()) > 0.5
            if not valid:
                continue
            pts, weights = _contact_points(hand[i], force[i])
            prev_pts, prev_weights = _contact_points(prev_hand[i], None)
            flow = _weighted_centroid(pts, weights) - _weighted_centroid(prev_pts, prev_weights)
            rel = _relative_velocity_np(prev_pose[i].detach().cpu().numpy(), pose[i].detach().cpu().numpy())
            score = float(np.linalg.norm(flow)) + 0.25 * float(np.linalg.norm(rel[:3]))
            if np.isfinite(score):
                flow_norms.append(score)
    if len(flow_norms) < 8:
        return {"enabled": False, "threshold": float(config.get("slip_threshold", 0.006)), "num_pairs": int(len(flow_norms)), "reason": "too_few_consecutive_pairs"}
    arr = np.asarray(flow_norms, dtype=np.float32)
    quantile = float(getattr(args, "tegtrack_slip_calibration_quantile", 0.75))
    threshold = float(np.quantile(arr, np.clip(quantile, 0.5, 0.95)))
    min_threshold = float(getattr(args, "tegtrack_min_slip_threshold", 1e-4))
    threshold = max(threshold, min_threshold)
    return {
        "enabled": True,
        "threshold": threshold,
        "num_pairs": int(len(flow_norms)),
        "quantile": quantile,
        "median_flow_score": float(np.median(arr)),
        "p90_flow_score": float(np.quantile(arr, 0.90)),
    }


def _train_slip_velocity_predictor(dataset, indices: np.ndarray, config: Mapping[str, float], args, device: torch.device) -> tuple[TegTrackSlipVelocityPredictor, dict[str, float]]:
    model = TegTrackSlipVelocityPredictor(image_side=int(config.get("image_side", 32)), angle_dim=19).to(device)
    opt = optim.Adam(model.parameters(), lr=float(getattr(args, "lr", 1e-4)), weight_decay=float(getattr(args, "weight_decay", 0.0)))
    epochs = max(1, int(getattr(args, "epochs", 1)))
    batch_size = max(1, int(getattr(args, "batch_size", 32)))
    progress_interval = max(0, int(getattr(args, "progress_interval", 0)))
    total_slip = 0
    last_loss = float("inf")
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_ckpt = checkpoint_path(args.output_dir, f"{NAME}_slip_velocity", suffix=".pt")
    for epoch in range(1, epochs + 1):
        sync_if_cuda(device)
        start_t = time.perf_counter()
        run_loss = 0.0
        seen_batches = 0
        seen_samples = 0
        model.train()
        for pick in _batch_iter(dataset, indices, batch_size, f"{NAME}-slip-train"):
            batch = _make_slip_training_batch(dataset, pick, config)
            if batch is None:
                continue
            pred = model(batch["tactile_pair"].to(device), batch["angle_delta"].to(device), batch["flow_stats"].to(device))
            target = batch["target_velocity"].to(device)
            loss = torch.nn.functional.mse_loss(pred[:, :3] * 100.0, target[:, :3] * 100.0) + torch.nn.functional.mse_loss(pred[:, 3:], target[:, 3:]) * 0.25
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = int(target.shape[0])
            total_slip += n
            seen_samples += n
            seen_batches += 1
            run_loss += float(loss.item())
            if progress_interval and seen_batches % progress_interval == 0:
                print(f"[{NAME}] epoch {epoch}/{epochs} slip-batch={seen_batches} slip-samples={seen_samples} velocity-loss={run_loss / max(seen_batches, 1):.6f}", flush=True)
        last_loss = run_loss / max(seen_batches, 1)
        print(f"[{NAME}] epoch {epoch}/{epochs} slip velocity predictor loss={last_loss:.6f} slip_samples={seen_samples} time={time.perf_counter() - start_t:.2f}s", flush=True)
        if seen_batches > 0 and last_loss < best_loss:
            best_loss = last_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "baseline": NAME,
                    "model_class": "TegTrackSlipVelocityPredictor",
                    "state_dict": best_state,
                    "best_loss": float(best_loss),
                    "best_epoch": int(best_epoch),
                    "slip_training_samples_seen": int(total_slip),
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"[{NAME}] ** saved best slip velocity checkpoint: epoch={epoch:4d} loss={best_loss:.6f} path={best_ckpt}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "slip_velocity_loss": float(best_loss if best_state is not None else last_loss),
        "slip_training_samples": int(total_slip),
        "best_slip_velocity_epoch": int(best_epoch),
        "best_slip_velocity_checkpoint": str(best_ckpt) if best_state is not None else "",
    }


def _slip_network_velocity(
    model: TegTrackSlipVelocityPredictor | None,
    tactile_now: torch.Tensor,
    tactile_prev: torch.Tensor,
    angle_now: torch.Tensor,
    angle_prev: torch.Tensor | None,
    flow: np.ndarray,
    device: torch.device,
) -> np.ndarray | None:
    if model is None or angle_prev is None:
        return None
    with torch.no_grad():
        pair = torch.cat([tactile_now, tactile_now - tactile_prev], dim=1).to(device)
        angle_delta = (angle_now.reshape(1, -1) - angle_prev.reshape(1, -1)).float().to(device)
        stats = torch.as_tensor([[float(np.linalg.norm(flow)), float(flow.mean()), float(flow.std())]], dtype=torch.float32, device=device)
        return model(pair, angle_delta, stats).detach().cpu().numpy().reshape(6).astype(np.float32)


def _rollout_sequence(
    dataset,
    indices: np.ndarray,
    config: Mapping[str, float],
    batch_size: int,
    tag: str,
    slip_model: TegTrackSlipVelocityPredictor | None = None,
    device: torch.device | None = None,
) -> Dict[str, np.ndarray]:
    pred_by_pos, gt_by_pos, mask_by_pos, scale_by_pos, force_by_pos = {}, {}, {}, {}, {}
    skipped_invalid = 0
    last_pose = None
    last_pos = None
    last_centroid = None
    last_points = None
    last_weights = None
    last_tactile = None
    last_angle = None
    last_trial = None
    device = device or torch.device("cpu")
    if slip_model is not None:
        slip_model.eval()
    for pick in _batch_iter(dataset, indices, batch_size, tag):
        hand, obj, pose, force, mask_pose, mask_force, scale = read_pose_force_batch(dataset, pick, include_object=True)
        aux = read_aux_batch(dataset, pick)
        trial_keys = list(aux.get("trial_key", [""] * len(pick)))
        angle = aux.get("wiseglove_angle", torch.zeros((len(pick), 19), dtype=torch.float32))
        tactile_imgs = _handsom_to_tactile_image(hand, side=int(config.get("image_side", 32)))
        poses = pose.detach().cpu().numpy().astype(np.float32)
        positions = np.asarray(pick, dtype=np.int64)
        masks = mask_pose.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
        scales = scale.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
        forces = force.detach().cpu().numpy().astype(np.float32)
        objects = obj.detach().cpu().numpy().astype(np.float32)
        for i, pos in enumerate(positions):
            hand_pts, weights = _contact_points(hand[i], force[i])
            centroid = _weighted_centroid(hand_pts, weights)
            same_sequence = last_pose is not None and last_pos is not None and int(pos) == int(last_pos) + 1 and trial_keys[i] == last_trial
            if not same_sequence:
                pred = _initialize_pose_from_contacts(hand_pts, weights, objects[i])
                if pred is None:
                    skipped_invalid += 1
                    last_pose = None
                    last_pos = None
                    last_centroid = None
                    last_points = None
                    last_weights = None
                    last_tactile = None
                    last_angle = None
                    last_trial = None
                    continue
            else:
                flow = centroid - last_centroid
                slip_score = float(np.linalg.norm(flow))
                slip = slip_score > float(config.get("slip_threshold", 0.006))
                if slip:
                    velocity = _slip_network_velocity(slip_model, tactile_imgs[i : i + 1], last_tactile, angle[i], last_angle, flow, device)
                    if velocity is None:
                        velocity = np.zeros(6, dtype=np.float32)
                        velocity[:3] = flow * float(config.get("slip_translation_gain", 1.0))
                else:
                    pivot = np.asarray(last_pose, dtype=np.float32).reshape(4, 4)[:3, 3]
                    velocity = _noslip_velocity_from_contacts(last_points, hand_pts, weights, pivot)
                    velocity[:3] *= float(config.get("noslip_translation_gain", 1.0))
                velocity[3:6] += _wiseglove_rotation_delta(angle[i], last_angle, float(config.get("rotation_gain", 0.02)))
                pred = _compose_velocity(last_pose, velocity)
            pred = _geometric_contact_refine(pred, hand_pts, weights, objects[i], float(config.get("geometry_refine_gain", 0.25)))
            if not np.isfinite(pred).all():
                skipped_invalid += 1
                last_pose = None
                last_pos = None
                last_centroid = None
                last_points = None
                last_weights = None
                last_tactile = None
                last_angle = None
                last_trial = None
                continue
            pred_by_pos[int(pos)] = pred
            gt_by_pos[int(pos)] = poses[i]
            mask_by_pos[int(pos)] = masks[i]
            scale_by_pos[int(pos)] = scales[i]
            force_by_pos[int(pos)] = forces[i]
            last_pose = pred
            last_pos = int(pos)
            last_centroid = centroid
            last_points = hand_pts
            last_weights = weights
            last_tactile = tactile_imgs[i : i + 1]
            last_angle = angle[i].detach().clone()
            last_trial = trial_keys[i]
    ordered = sorted(pred_by_pos)
    return {
        "positions": np.asarray(ordered, dtype=np.int64),
        "pred_pose": np.stack([pred_by_pos[p] for p in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 16), dtype=np.float32),
        "target_pose": np.stack([gt_by_pos[p] for p in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 16), dtype=np.float32),
        "target_force": np.stack([force_by_pos[p] for p in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 2), dtype=np.float32),
        "mask_pose": np.stack([mask_by_pos[p] for p in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 1), dtype=np.float32),
        "scale": np.stack([scale_by_pos[p] for p in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 1), dtype=np.float32),
        "skipped_invalid_samples": np.asarray([skipped_invalid], dtype=np.int64),
    }


def train(args, dataset, train_idx, val_idx, test_idx, device: torch.device) -> Path:
    train_pick = subset_by_max(train_idx, args.max_train_samples)
    val_pick = subset_by_max(val_idx, args.max_val_samples)
    ckpt = checkpoint_path(args.output_dir, NAME)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "slip_threshold": float(getattr(args, "tegtrack_slip_threshold", 0.006)),
        "noslip_translation_gain": float(getattr(args, "tegtrack_noslip_translation_gain", 0.65)),
        "slip_translation_gain": float(getattr(args, "tegtrack_slip_translation_gain", 1.0)),
        "rotation_gain": float(getattr(args, "tegtrack_rotation_gain", 0.02)),
        "geometry_refine_gain": float(getattr(args, "tegtrack_geometry_refine_gain", 0.25)),
        "image_side": int(getattr(args, "tegtrack_image_side", 32)),
    }
    slip_calibration = _calibrate_slip_threshold(dataset, train_pick, config, args)
    if slip_calibration.get("enabled"):
        config["slip_threshold"] = float(slip_calibration["threshold"])
    slip_model, train_metrics = _train_slip_velocity_predictor(dataset, train_pick, config, args, device)
    payload = {
        "baseline": NAME,
        "model_class": "TegTrackKinematicTrackerWithPrivateSlipVelocityPredictor",
        "paper_structure": "slip detector + no-slip least-squares kinematic velocity predictor + learning-based slip velocity predictor + geometric-kinematic rollout/refinement; predicts frame-to-frame velocity, not absolute pose",
        "r2_adaptation": "R2 pressure-map differences and WiseGlove increments replace GelSight tactile RGB/marker-flow channels; the ShapeAlign/visual stream is approximated by object geometry contact refinement because R2 lacks RGB-D tracking input.",
        "config": config,
        "slip_calibration": slip_calibration,
        "slip_velocity_state_dict": slip_model.state_dict(),
        "train_metrics": train_metrics,
        "train_size": int(len(train_pick)),
        "val_size": int(len(val_pick)),
        "test_size": int(len(test_idx)),
        "split_file": str(args.split_file),
        "args": vars(args),
    }
    torch.save(payload, ckpt)
    save_json(ckpt.with_suffix(".json"), {k: v for k, v in payload.items() if k != "slip_velocity_state_dict"})
    print(f"[{NAME}] saved TEG-Track private slip velocity predictor + kinematic tracker: path={ckpt}", flush=True)
    return ckpt


def test(args, dataset, test_idx, device: torch.device, checkpoint: Path | None = None) -> Dict[str, float]:
    ckpt = checkpoint or checkpoint_path(args.output_dir, NAME)
    if not ckpt.exists():
        raise FileNotFoundError(f"{NAME} checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    config = payload.get("config", {})
    slip_model = TegTrackSlipVelocityPredictor(image_side=int(config.get("image_side", 32)), angle_dim=19).to(device)
    state = payload.get("slip_velocity_state_dict")
    if state is not None:
        slip_model.load_state_dict(state)
    else:
        slip_model = None
    picked = subset_by_max(test_idx, args.max_test_samples)
    predictions = _rollout_sequence(dataset, picked, config, args.batch_size, f"{NAME}-test", slip_model=slip_model, device=device)
    valid = predictions["mask_pose"].reshape(-1) > 0.5
    metrics = compute_pose_metrics(predictions["pred_pose"][valid], predictions["target_pose"][valid], scale=predictions["scale"][valid]) if valid.any() else {"mse": float("inf"), "rotation_error_deg": float("inf"), "translation_l1": float("inf")}
    if getattr(args, "save_per_sample", True):
        metrics["per_sample_predictions"] = predictions
    metrics["num_test_samples"] = int(valid.sum()) if valid.size else 0
    metrics["skipped_invalid_samples"] = int(predictions.get("skipped_invalid_samples", np.asarray([0], dtype=np.int64))[0])
    metrics["checkpoint"] = str(ckpt)
    return metrics




