from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .runtime import checkpoint_path, compute_pose_metrics, flatten_pose_matrix, load_json, read_aux_batch, read_pose_force_batch, save_json, subset_by_max


NAME = "tactile_ekf"


def _normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / max(float(np.linalg.norm(arr)), float(eps))


def _pose6_to_matrix_local(state6: np.ndarray) -> np.ndarray:
    state = np.asarray(state6, dtype=np.float32).reshape(6)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = state[:3]
    T[:3, :3] = _euler_xyz_to_matrix(state[3:6]).astype(np.float32)
    return T



@dataclass
class ContactMeasurement:
    contact_id: str
    point_palm: np.ndarray
    normal_palm: np.ndarray
    pressure: float = 1.0
    active: bool = True


@dataclass
class EKFStepInput:
    dt: float
    joint_velocity: np.ndarray
    contact_jacobian: np.ndarray


@dataclass
class EKFConfig:
    process_noise_diag: List[float] = field(default_factory=lambda: [1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
    measurement_noise_position: float = 5e-4
    measurement_noise_normal: float = 1e-2
    process_blend: float = 0.15
    proprioceptive_drive_scale: float = 0.02
    jacobian_eps: float = 1e-4


def _euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


class PointCloudSurfaceModel:
    """Object-specific surface model built from the provided object point cloud.

    Lin et al.'s EKF needs the candidate pose to predict where a tactile contact
    would lie on the object surface and what local normal would be observed.
    The previous R2 adapter replaced every object with a fitted cylinder, which
    erases the non-convex tool geometry the method is meant to handle. This model
    keeps the same EKF measurement interface while using each object's own point
    cloud for nearest-surface projection and local PCA normal estimation.
    """

    def __init__(self, points: np.ndarray, normal_k: int = 12):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.shape[0] == 0:
            pts = np.zeros((1, 3), dtype=float)
        self.points = pts
        self.center = pts.mean(axis=0)
        self.normal_k = max(3, min(int(normal_k), int(pts.shape[0])))

    def _nearest_order(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        d2 = np.sum((self.points - p.reshape(1, 3)) ** 2, axis=1)
        return np.argsort(d2, kind="stable")

    def project_to_surface(self, point: np.ndarray) -> np.ndarray:
        order = self._nearest_order(point)
        return self.points[int(order[0])].astype(float, copy=True)

    def surface_normal(self, point: np.ndarray) -> np.ndarray:
        order = self._nearest_order(point)[: self.normal_k]
        neigh = self.points[order]
        if neigh.shape[0] < 3:
            return _normalize_vec((np.asarray(point, dtype=float).reshape(3) - self.center).astype(np.float32)).astype(float)
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except np.linalg.LinAlgError:
            normal = np.asarray(point, dtype=float).reshape(3) - self.center
        outward = np.asarray(point, dtype=float).reshape(3) - self.center
        if float(np.dot(normal, outward)) < 0.0:
            normal = -normal
        return _normalize_vec(normal.astype(np.float32)).astype(float)


class TactileObjectPoseEKF:
    """6D tactile object-pose EKF (Lin et al., 2023), R2-adapted.

    State: x = [x, y, z, roll, pitch, yaw] of the object expressed in the palm
    frame. Process: identity-transition propagation with additive process noise
    (the original rigid drive term pinv(G^T) J u dt requires joint-velocity /
    contact-Jacobian streams that R2 does not provide, so it is held at zero while
    the state-transition itself is preserved). Measurement: nonlinear observation
    h(x) -> (expected contact point, expected surface normal) obtained by applying
    the candidate pose to the object surface model, linearized with a numeric
    Jacobian for the EKF update.
    """

    def __init__(self, object_model: PointCloudSurfaceModel, config: EKFConfig | None = None):
        self.object_model = object_model
        self.config = config or EKFConfig()
        self.state = np.zeros(6, dtype=float)
        self.cov = np.eye(6, dtype=float) * 1e-2

    def reset(self, state: np.ndarray, covariance: np.ndarray | None = None) -> None:
        self.state = np.asarray(state, dtype=float).reshape(6).copy()
        self.cov = np.asarray(covariance, dtype=float) if covariance is not None else np.eye(6, dtype=float) * 1e-2

    # -- nonlinear observation model -------------------------------------------------
    def _predict_contact(self, state: np.ndarray, observed_point: np.ndarray) -> np.ndarray:
        """h(state): expected (contact point, surface normal) in the palm frame.

        The object pose (R(rpy), t) maps the observed contact point into the
        object frame, where it is projected onto the object surface; the surface
        point and outward normal are then mapped back into the palm frame. As the
        pose state changes, so do the expected contact point and normal, giving a
        genuinely pose-dependent (nonlinear) measurement prediction.
        """
        state = np.asarray(state, dtype=float).reshape(6)
        t = state[:3]
        R = _euler_xyz_to_matrix(state[3:6])
        # observed contact point expressed in the object frame
        p_obj = R.T @ (np.asarray(observed_point, dtype=float).reshape(3) - t)
        surf_obj = self.object_model.project_to_surface(p_obj)
        normal_obj = self.object_model.surface_normal(surf_obj)
        # back to palm frame
        surf_palm = R @ surf_obj + t
        normal_palm = R @ normal_obj
        return np.concatenate([surf_palm, _normalize_vec(normal_palm.astype(np.float32)).astype(float)], axis=0)

    def _measurement_jacobian(self, state: np.ndarray, observed_point: np.ndarray) -> np.ndarray:
        eps = float(self.config.jacobian_eps)
        base = self._predict_contact(state, observed_point)
        H = np.zeros((6, 6), dtype=float)
        for j in range(6):
            perturbed = state.copy()
            perturbed[j] += eps
            H[:, j] = (self._predict_contact(perturbed, observed_point) - base) / eps
        return H

    def step(self, ekf_input: EKFStepInput) -> dict:
        # ---- predict (state genuinely propagated; identity transition F = I) ----
        q = np.diag(np.asarray(self.config.process_noise_diag, dtype=float))
        drive = np.zeros(6, dtype=float)
        joint_velocity = np.asarray(ekf_input.joint_velocity, dtype=float).reshape(-1)
        if joint_velocity.size:
            centered = joint_velocity - float(np.mean(joint_velocity))
            if centered.size >= 3:
                drive[3:6] = centered[:3]
            drive[:3] = float(np.mean(np.abs(joint_velocity))) * np.asarray([0.0, 0.0, 1.0], dtype=float)
        self.state = self.state.copy() + float(self.config.proprioceptive_drive_scale) * float(ekf_input.dt) * drive
        self.cov = self.cov + q

        active = [c for c in ekf_input.contacts if c.active]
        if active:
            pts = np.stack([np.asarray(c.point_palm, dtype=float).reshape(3) for c in active], axis=0)
            normals = np.stack([np.asarray(c.normal_palm, dtype=float).reshape(3) for c in active], axis=0)
            pressures = np.asarray([max(float(c.pressure), 1e-6) for c in active], dtype=float)
            pressures = pressures / pressures.sum()

            # pressure-weighted aggregate observation (measured contact point + normal)
            z_point = (pts * pressures[:, None]).sum(axis=0)
            z_normal = _normalize_vec(((normals * pressures[:, None]).sum(axis=0)).astype(np.float32)).astype(float)
            z = np.concatenate([z_point, z_normal], axis=0)

            # nonlinear prediction h(state) and its Jacobian about the aggregate point
            h_pred = self._predict_contact(self.state, z_point)
            H = self._measurement_jacobian(self.state, z_point)

            r = np.diag(
                [self.config.measurement_noise_position] * 3
                + [self.config.measurement_noise_normal] * 3
            )
            innovation = z - h_pred
            s = H @ self.cov @ H.T + r
            k = self.cov @ H.T @ np.linalg.pinv(s)
            self.state = self.state + k @ innovation
            self.cov = (np.eye(6, dtype=float) - k @ H) @ self.cov
        return {"state": self.state.copy(), "covariance": self.cov.copy(), "used_contacts": [c.contact_id for c in active]}


def _build_config(overrides: Dict | None = None) -> EKFConfig:
    cfg = EKFConfig()
    if overrides:
        process_scale = float(overrides.get("process_scale", 1.0))
        cfg.process_noise_diag = [float(v) * process_scale for v in cfg.process_noise_diag]
        cfg.measurement_noise_position = float(overrides.get("measurement_noise_position", cfg.measurement_noise_position))
        cfg.measurement_noise_normal = float(overrides.get("measurement_noise_normal", cfg.measurement_noise_normal))
        cfg.proprioceptive_drive_scale = float(overrides.get("proprioceptive_drive_scale", cfg.proprioceptive_drive_scale))
    return cfg


def _object_name_for_position(dataset, position: int) -> str:
    names = getattr(dataset, "object_name", None)
    if isinstance(names, list) and 0 <= int(position) < len(names):
        return str(names[int(position)])
    return "unknown"


def _geometry_key(object_name: str, object_points: np.ndarray, trial: str) -> str:
    if trial:
        return f"trial:{trial}"
    pts = np.asarray(object_points, dtype=float).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return f"object:{object_name}:empty"
    center = np.round(pts.mean(axis=0), 4)
    spread = np.round(pts.std(axis=0), 4)
    geom = ",".join(f"{v:.4f}" for v in np.concatenate([center, spread], axis=0))
    return f"object:{object_name}:geom:{geom}"


    if hand_np.ndim != 2 or hand_np.shape[1] < 3:
        return []
    pts = np.asarray(hand_np[:, :3], dtype=np.float32)
    if hand_np.shape[1] >= 4:
        pressure = np.maximum(np.asarray(hand_np[:, 3], dtype=np.float32), 0.0)
    else:
        pressure = np.linalg.norm(pts - center.reshape(1, 3), axis=1).astype(np.float32)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(pressure)
    pts = pts[finite]
    pressure = pressure[finite]
    if len(pts) == 0:
        return []
    if float(pressure.max(initial=0.0)) > float(pressure.min(initial=0.0)):
        keep = pressure >= np.quantile(pressure, 0.75)
        if keep.sum() >= 3:
            pts = pts[keep]
            pressure = pressure[keep]
    order = np.argsort(-pressure)[: int(max_contacts)]
    for i, idx in enumerate(order):
        point = pts[idx]
        contacts.append(
            ContactMeasurement(
                contact_id=f"contact_{i}",
                point_palm=point,
                normal_palm=_normalize_vec(point - center),
                pressure=float(max(pressure[idx], 1e-6)),
                active=True,
            )
        )
    return contacts


def _evaluate(
    dataset,
    indices: np.ndarray,
    max_samples: int | None,
    config_overrides: Dict | None = None,
    tag: str = NAME,
    batch_size: int = 1024,
    save_per_sample: bool = True,
) -> Dict[str, float]:
    picked = subset_by_max(indices, max_samples)
    batch_size = min(max(1, int(batch_size)), 2048)
    cfg = _build_config(config_overrides)
    filters: Dict[str, TactileObjectPoseEKF] = {}
    prev_angle_by_trial: Dict[str, np.ndarray] = {}
    preds, gts, scales, positions = [], [], [], []
    skipped_invalid = 0
    print(f"[{tag}] evaluate samples={len(picked)} batch_size={batch_size} config={config_overrides}", flush=True)
    for start in range(0, len(picked), batch_size):
        stop = min(start + batch_size, len(picked))
        hand_batch, object_batch, pose_batch, _, mask_pose_batch, _, scale_batch = read_pose_force_batch(
            dataset,
            picked[start:stop],
            include_object=True,
        )
        aux_batch = read_aux_batch(dataset, picked[start:stop])
        for local_i in range(stop - start):
            sample_i = start + local_i + 1
            if sample_i == 1 or sample_i % 1000 == 0 or sample_i == len(picked):
                print(f"[{tag}] sample {sample_i}/{len(picked)}", flush=True)
            hand_som = hand_batch[local_i]
            object_data = object_batch[local_i]
            pose = pose_batch[local_i]
            mask_pose = mask_pose_batch[local_i]
            scale = scale_batch[local_i]
            if float(mask_pose.reshape(-1)[0].item()) <= 0.5:
                continue
            hand_np = hand_som.numpy()
            obj_np = object_data.numpy()
            valid_obj = np.isfinite(obj_np).all(axis=1)
            obj_np = obj_np[valid_obj]
            if obj_np.shape[0] < 3:
                skipped_invalid += 1
                continue
            center = obj_np.mean(axis=0)
            object_name = _object_name_for_position(dataset, int(picked[start + local_i]))
            trial = str(aux_batch.get("trial_key", [""] * (stop - start))[local_i]) or object_name
            angle = aux_batch.get("wiseglove_angle", torch.zeros((stop - start, 19), dtype=torch.float32))[local_i].numpy()
            prev_angle = prev_angle_by_trial.get(trial)
            if prev_angle is None:
                joint_velocity = np.zeros_like(angle, dtype=float)
            else:
                joint_velocity = np.nan_to_num(angle.astype(float) - prev_angle.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            prev_angle_by_trial[trial] = angle.astype(float)
            key = _geometry_key(object_name, obj_np, trial)
            if key not in filters:
                filters[key] = TactileObjectPoseEKF(PointCloudSurfaceModel(obj_np), config=cfg)
                filters[key].reset(state=np.concatenate([center.astype(float), np.zeros(3, dtype=float)]))
            ekf = filters[key]
            contacts = _contacts_from_handsom(hand_np, center)
            if not contacts:
                skipped_invalid += 1
                continue
            step = EKFStepInput(
                dt=0.01,
                joint_velocity=joint_velocity,
                contacts=contacts,
                contact_jacobian=np.zeros((3, 4), dtype=float),
            )
            out = ekf.step(step)
            pred_T = _pose6_to_matrix_local(np.asarray(out["state"], dtype=np.float32))
            if not np.isfinite(pred_T).all():
                skipped_invalid += 1
                continue
            preds.append(flatten_pose_matrix(pred_T))
            gts.append(pose.numpy())
            scales.append(scale.numpy())
            positions.append(int(picked[start + local_i]))
    if not preds:
        return {"mse": float("inf"), "rotation_error_deg": float("inf"), "translation_l1": float("inf"), "num_test_samples": 0, "skipped_invalid_samples": int(skipped_invalid)}
    metrics = compute_pose_metrics(np.stack(preds, axis=0), np.stack(gts, axis=0), scale=np.stack(scales, axis=0))
    metrics["num_test_samples"] = int(len(preds))
    metrics["skipped_invalid_samples"] = int(skipped_invalid)
    if save_per_sample:
        metrics["per_sample_predictions"] = {
            "positions": np.asarray(positions, dtype=np.int64),
            "pred_pose": np.stack(preds, axis=0).astype(np.float32),
            "target_pose": np.stack(gts, axis=0).astype(np.float32),
            "mask_pose": np.ones((len(preds), 1), dtype=np.float32),
            "scale": np.stack(scales, axis=0).astype(np.float32),
        }
    return metrics


def train(args, dataset, train_idx, val_idx, test_idx, device: torch.device) -> Path:
    del device
    candidates = [
        {"measurement_noise_position": 1e-4, "measurement_noise_normal": 1e-3, "process_scale": 0.5},
        {"measurement_noise_position": 5e-4, "measurement_noise_normal": 1e-2, "process_scale": 1.0},
        {"measurement_noise_position": 1e-3, "measurement_noise_normal": 5e-2, "process_scale": 2.0},
        {"measurement_noise_position": 5e-4, "measurement_noise_normal": 1e-2, "process_scale": 1.0, "proprioceptive_drive_scale": 0.05},
    ]
    best_config = None
    best_train_metrics = None
    best_val_metrics = None
    print(f"[{NAME}] analytic tuning candidates={len(candidates)} train_size={len(train_idx)} val_size={len(val_idx)}", flush=True)
    for i, candidate in enumerate(candidates, start=1):
        train_metrics = _evaluate(dataset, train_idx, args.max_train_samples, candidate, tag=f"{NAME}-train-candidate-{i}", batch_size=args.batch_size, save_per_sample=False)
        val_metrics = _evaluate(dataset, val_idx, args.max_val_samples, candidate, tag=f"{NAME}-val-candidate-{i}", batch_size=args.batch_size, save_per_sample=False)
        print(
            f"[{NAME}] candidate {i:2d}/{len(candidates)}  "
            f"train-angle-err={train_metrics.get('rotation_error_deg', float('nan')):.3f}deg  "
            f"val-angle-err={val_metrics.get('rotation_error_deg', float('nan')):.3f}deg  "
            f"val-dist-err={val_metrics.get('translation_l1', float('nan')):.4f}  "
            f"train-samples={train_metrics.get('num_test_samples', 0)}  "
            f"val-samples={val_metrics.get('num_test_samples', 0)}  config={candidate}",
            flush=True,
        )
        if best_val_metrics is None or val_metrics["rotation_error_deg"] < best_val_metrics["rotation_error_deg"]:
            best_config = candidate
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
    print(
        f"[{NAME}] selected  "
        f"train-angle-err={(best_train_metrics or {}).get('rotation_error_deg', float('nan')):.3f}deg  "
        f"val-angle-err={(best_val_metrics or {}).get('rotation_error_deg', float('nan')):.3f}deg  "
        f"val-dist-err={(best_val_metrics or {}).get('translation_l1', float('nan')):.4f}  "
        f"config={best_config}",
        flush=True,
    )
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".json")
    save_json(
        ckpt,
        {
            "baseline": NAME,
            "paper_structure": "6D tactile object-pose EKF with persistent state, proprioceptive prediction, and a nonlinear contact-position/normal observation model with numeric Jacobian",
            "r2_adaptation": "R2 WiseGlove angle increments provide the proprioceptive prediction drive when trial-continuous samples are available; the contact-Jacobian term is evaluated through the R2 contact geometry and nonlinear observation model. Noise parameters are selected on the validation split.",
            "best_config": best_config,
            "train_metrics": best_train_metrics,
            "val_metrics": best_val_metrics,
            "test_size": int(len(test_idx)),
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
    config = load_json(ckpt).get("best_config")
    metrics = _evaluate(dataset, test_idx, args.max_test_samples, config, tag=f"{NAME}-test", batch_size=args.batch_size, save_per_sample=bool(getattr(args, "save_per_sample", True)))
    metrics["checkpoint"] = str(ckpt)
    return metrics






