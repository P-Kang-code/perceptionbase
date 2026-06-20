from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .runtime import checkpoint_path, compute_pose_metrics, flatten_pose_matrix, read_aux_batch, read_pose_force_batch, save_json, subset_by_max


NAME = "neuralfeels"


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) <= int(max_points):
        return pts
    pick = np.linspace(0, len(pts) - 1, int(max_points)).round().astype(np.int64)
    return pts[pick]


def _contact_points(hand_som: torch.Tensor, max_points: int) -> np.ndarray:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    pts = arr[:, :3]
    pressure = np.maximum(arr[:, 3], 0.0) if arr.shape[1] >= 4 else np.linalg.norm(pts, axis=1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(pressure)
    pts = pts[finite]
    pressure = pressure[finite]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if float(pressure.max(initial=0.0)) > float(pressure.min(initial=0.0)):
        keep = pressure >= np.quantile(pressure, 0.70)
        if keep.sum() >= 6:
            pts = pts[keep]
    return _downsample(pts, max_points)


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(float(angle)), np.sin(float(angle))
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _initial_pose_candidates(object_points: np.ndarray, contact_points: np.ndarray, prior_pose: np.ndarray | None = None) -> list[np.ndarray]:
    obj_c = object_points.mean(axis=0) if len(object_points) else np.zeros(3, dtype=np.float32)
    tgt_c = contact_points.mean(axis=0) if len(contact_points) else np.zeros(3, dtype=np.float32)
    rotations = [np.eye(3, dtype=np.float32)]
    for axis in range(3):
        for angle in (np.pi / 2, np.pi, 3 * np.pi / 2):
            rotations.append(_axis_rotation(axis, angle))
    out = []
    if prior_pose is not None:
        out.append(np.asarray(prior_pose, dtype=np.float32).reshape(4, 4).copy())
    for R in rotations:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tgt_c - R @ obj_c
        out.append(T)
    return out


def _torch_transform(points: torch.Tensor, rotvec: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec).clamp_min(1e-8)
    k = rotvec / theta
    zero = torch.zeros((), device=points.device, dtype=points.dtype)
    kx = torch.stack(
        [
            torch.stack([zero, -k[2], k[1]]),
            torch.stack([k[2], zero, -k[0]]),
            torch.stack([-k[1], k[0], zero]),
        ],
        dim=0,
    )
    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    R = eye + torch.sin(theta) * kx + (1.0 - torch.cos(theta)) * (kx @ kx)
    return points @ R.T + trans.reshape(1, 3)


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
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


def _angle_delta_to_rotvec(angle_now: torch.Tensor | None, angle_prev: torch.Tensor | None, gain: float) -> np.ndarray:
    if angle_now is None or angle_prev is None:
        return np.zeros(3, dtype=np.float32)
    delta = (angle_now.detach().cpu().numpy().reshape(-1) - angle_prev.detach().cpu().numpy().reshape(-1)).astype(np.float32)
    if delta.size == 0:
        return np.zeros(3, dtype=np.float32)
    chunks = np.array_split(delta, 3)
    return np.asarray([float(c.mean()) if c.size else 0.0 for c in chunks], dtype=np.float32) * float(gain)


def _compose_motion_prior(prev_pose: np.ndarray, prev_contacts: np.ndarray, contacts: np.ndarray, angle_now: torch.Tensor | None, angle_prev: torch.Tensor | None, gain: float) -> np.ndarray:
    T = np.asarray(prev_pose, dtype=np.float32).reshape(4, 4).copy()
    if len(prev_contacts) and len(contacts):
        T[:3, 3] += (contacts.mean(axis=0) - prev_contacts.mean(axis=0)).astype(np.float32)
    dR = _rotvec_to_rotmat_np(_angle_delta_to_rotvec(angle_now, angle_prev, gain))
    T[:3, :3] = dR @ T[:3, :3]
    return T.astype(np.float32)


class PointCloudSDFTeacher:
    """Point-cloud teacher used to train the online NeuralFeels field.

    The NeuralFeels system optimizes poses against an online neural SDF.
    R2 does not provide the original RGB-D stream for live mapping, so each
    sample's object point cloud supplies self-supervised near-surface targets for
    a method-private neural field. Pose labels are never used by this teacher.
    """

    def __init__(self, points: np.ndarray, max_points: int):
        pts = _downsample(points, max_points)
        if len(pts) == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        self.points = torch.as_tensor(pts, dtype=torch.float32)
        self.center = self.points.mean(dim=0, keepdim=True)

    def query(self, query_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = torch.cdist(query_points, self.points.to(query_points.device))
        min_dist, idx = d.min(dim=1)
        surf = self.points.to(query_points.device)[idx]
        normal = torch.nn.functional.normalize(surf - self.center.to(query_points.device), dim=1)
        return min_dist, normal


class OnlineNeuralSDF(nn.Module):
    """Small NeuralFeels-private field trained online from the current object."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Softplus(beta=10.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=10.0),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)

    def query(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points_req = points if points.requires_grad else points.detach().clone().requires_grad_(True)
        sdf = self.forward(points_req)
        grad = torch.autograd.grad(sdf.sum(), points_req, create_graph=True)[0]
        normal = torch.nn.functional.normalize(grad, dim=1)
        return torch.abs(sdf), normal


def _fit_online_neural_sdf(object_points: np.ndarray, *, max_model_points: int, train_steps: int, hidden_dim: int, lr: float) -> OnlineNeuralSDF:
    model_np = _downsample(object_points, max_model_points)
    if len(model_np) == 0:
        model_np = np.zeros((1, 3), dtype=np.float32)
    device = torch.device("cpu")
    teacher = PointCloudSDFTeacher(model_np, max_model_points)
    surface = torch.as_tensor(model_np, dtype=torch.float32, device=device)
    center = surface.mean(dim=0, keepdim=True)
    radius = torch.linalg.norm(surface - center, dim=1).quantile(0.90).clamp_min(1e-3)
    field = OnlineNeuralSDF(hidden_dim=int(hidden_dim)).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=float(lr))
    n = surface.shape[0]
    batch = min(512, max(64, int(n)))
    for _ in range(max(1, int(train_steps))):
        idx = torch.randint(0, n, (batch,), device=device)
        surf = surface[idx]
        noise = torch.randn_like(surf)
        noise = noise / torch.linalg.norm(noise, dim=1, keepdim=True).clamp_min(1e-6)
        near_offset = 0.025 * radius * torch.randn((batch, 1), device=device)
        near = surf + noise * near_offset
        free = center + torch.empty((batch, 3), device=device).uniform_(-1.4, 1.4) * radius
        query = torch.cat([surf, near, free], dim=0)
        with torch.no_grad():
            target, _ = teacher.query(query)
            target[:batch] = 0.0
        pred = torch.abs(field(query))
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return field


def _optimize_sdf_pose(
    object_points: np.ndarray,
    contact_points: np.ndarray,
    *,
    steps: int,
    lr: float,
    max_model_points: int,
    field_steps: int,
    field_hidden_dim: int,
    field_lr: float,
    prior_pose: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> np.ndarray | None:
    model_np = _downsample(object_points, max_model_points)
    target_np = _downsample(contact_points, 128)
    if len(model_np) < 4 or len(target_np) < 3:
        return None
    device = torch.device("cpu")
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    sdf = _fit_online_neural_sdf(
        model_np,
        max_model_points=max_model_points,
        train_steps=field_steps,
        hidden_dim=field_hidden_dim,
        lr=field_lr,
    )
    best_T = np.eye(4, dtype=np.float32)
    best_loss = float("inf")
    prior_rotvec = None
    prior_trans = None
    if prior_pose is not None and float(prior_weight) > 0.0:
        prior = np.asarray(prior_pose, dtype=np.float32).reshape(4, 4)
        prior_rotvec = torch.as_tensor(_rotmat_to_rotvec(prior[:3, :3]), dtype=torch.float32, device=device)
        prior_trans = torch.as_tensor(prior[:3, 3], dtype=torch.float32, device=device)
    for init in _initial_pose_candidates(model_np, target_np, prior_pose=prior_pose):
        rotvec = torch.tensor(_rotmat_to_rotvec(init[:3, :3]), dtype=torch.float32, requires_grad=True)
        trans = torch.tensor(init[:3, 3], dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([rotvec, trans], lr=float(lr))
        for _ in range(int(steps)):
            theta = torch.linalg.norm(rotvec).clamp_min(1e-8)
            k = rotvec / theta
            zero = torch.zeros((), device=device, dtype=torch.float32)
            kx = torch.stack(
                [
                    torch.stack([zero, -k[2], k[1]]),
                    torch.stack([k[2], zero, -k[0]]),
                    torch.stack([-k[1], k[0], zero]),
                ],
                dim=0,
            )
            eye = torch.eye(3, device=device, dtype=torch.float32)
            R = eye + torch.sin(theta) * kx + (1.0 - torch.cos(theta)) * (kx @ kx)
            target_obj = (target - trans.reshape(1, 3)) @ R
            sdf_dist, normal = sdf.query(target_obj)
            center_prior = torch.mean(target_obj - torch.as_tensor(model_np.mean(axis=0), dtype=torch.float32), dim=0)
            normal_consistency = torch.relu(-torch.sum(normal * torch.nn.functional.normalize(target_obj - target_obj.mean(dim=0, keepdim=True), dim=1), dim=1)).mean()
            loss = sdf_dist.mean() + 0.05 * torch.linalg.norm(center_prior) + 0.02 * normal_consistency
            if prior_rotvec is not None and prior_trans is not None:
                pose_prior = torch.linalg.norm(trans - prior_trans) + 0.25 * torch.linalg.norm(rotvec - prior_rotvec)
                loss = loss + float(prior_weight) * pose_prior
            opt.zero_grad()
            loss.backward()
            opt.step()
        final_loss = float(loss.detach().cpu().item())
        if final_loss < best_loss:
            best_loss = final_loss
            best_T = np.eye(4, dtype=np.float32)
            best_T[:3, :3] = _rotvec_to_rotmat_np(rotvec.detach().cpu().numpy())
            best_T[:3, 3] = trans.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(best_loss) or not np.isfinite(best_T).all():
        return None
    return best_T


def _evaluate(dataset, indices: np.ndarray, max_samples: int | None, args, batch_size: int = 64) -> Dict[str, float]:
    picked = subset_by_max(indices, max_samples)
    preds, gts, scales, positions = [], [], [], []
    skipped_invalid = 0
    last_pose_by_trial: Dict[str, np.ndarray] = {}
    last_contacts_by_trial: Dict[str, np.ndarray] = {}
    last_angle_by_trial: Dict[str, torch.Tensor] = {}
    last_pos_by_trial: Dict[str, int] = {}
    prior_weight = float(getattr(args, "neuralfeels_pose_graph_prior_weight", 0.03))
    rotation_gain = float(getattr(args, "neuralfeels_motion_rotation_gain", 0.02))
    print(f"[{NAME}] neural SDF-style pose optimization samples={len(picked)} batch_size={batch_size}", flush=True)
    for start in range(0, len(picked), max(1, int(batch_size))):
        stop = min(start + int(batch_size), len(picked))
        hand_batch, object_batch, pose_batch, _, mask_pose_batch, _, scale_batch = read_pose_force_batch(dataset, picked[start:stop], include_object=True)
        aux = read_aux_batch(dataset, picked[start:stop])
        trial_keys = list(aux.get("trial_key", [""] * (stop - start)))
        angles = aux.get("wiseglove_angle", torch.zeros((stop - start, 19), dtype=torch.float32))
        for local_i in range(stop - start):
            if float(mask_pose_batch[local_i].reshape(-1)[0].item()) <= 0.5:
                continue
            contacts = _contact_points(hand_batch[local_i], int(getattr(args, "neuralfeels_max_contact_points", 128)))
            object_points = object_batch[local_i].detach().cpu().numpy().astype(np.float32)
            pos = int(picked[start + local_i])
            trial = str(trial_keys[local_i]) or f"sample:{pos}"
            sequential = trial in last_pose_by_trial and pos == int(last_pos_by_trial.get(trial, -10**9)) + 1
            prior = None
            if sequential:
                prior = _compose_motion_prior(
                    last_pose_by_trial[trial],
                    last_contacts_by_trial.get(trial, np.zeros((0, 3), dtype=np.float32)),
                    contacts,
                    angles[local_i],
                    last_angle_by_trial.get(trial),
                    rotation_gain,
                )
            pred = _optimize_sdf_pose(
                object_points,
                contacts,
                steps=int(getattr(args, "neuralfeels_opt_steps", 60)),
                lr=float(getattr(args, "neuralfeels_lr", 2e-2)),
                max_model_points=int(getattr(args, "neuralfeels_max_model_points", 2048)),
                field_steps=int(getattr(args, "neuralfeels_field_steps", 80)),
                field_hidden_dim=int(getattr(args, "neuralfeels_field_hidden_dim", 64)),
                field_lr=float(getattr(args, "neuralfeels_field_lr", 1e-3)),
                prior_pose=prior,
                prior_weight=prior_weight if sequential else 0.0,
            )
            if pred is None:
                skipped_invalid += 1
                continue
            preds.append(flatten_pose_matrix(pred))
            gts.append(pose_batch[local_i].numpy())
            scales.append(scale_batch[local_i].numpy())
            positions.append(pos)
            last_pose_by_trial[trial] = pred
            last_contacts_by_trial[trial] = contacts
            last_angle_by_trial[trial] = angles[local_i].detach().clone()
            last_pos_by_trial[trial] = pos
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
    del dataset, train_idx, val_idx, test_idx, device
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".json")
    save_json(
        ckpt,
        {
            "baseline": NAME,
            "method": "online neural-field-style SDF pose optimization with a trial-local sliding-window motion prior",
            "paper_structure": "method-private online neural SDF field fitted from object geometry, then optimized at test time against tactile contact observations with a lightweight pose-graph-style temporal prior",
            "r2_adaptation": "R2 lacks the original RGB-D stream and DIGIT neural-field training data, so object_data points instantiate the object field and hand_som contact samples replace tactile depth contacts; consecutive trial frames add a contact-flow/WiseGlove motion prior to avoid independent single-frame fitting.",
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
    metrics = _evaluate(dataset, test_idx, args.max_test_samples, args, batch_size=args.batch_size)
    metrics["checkpoint"] = str(ckpt)
    return metrics



