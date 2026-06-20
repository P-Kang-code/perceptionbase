from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .runtime import checkpoint_path, read_aux_batch, read_pose_force_batch, save_json, subset_by_max


NAME = "simtact_force"


@dataclass
class SimTactGraphConfig:
    window: int = 8
    tactile_weight: float = 1.0
    motion_weight: float = 0.25
    smooth_weight: float = 0.15
    pressure_to_force: float = 1.0
    lever_to_torque: float = 1.0


def _pressure(hand_som: torch.Tensor) -> np.ndarray:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0,), dtype=np.float32)
    if arr.shape[1] >= 4:
        values = np.maximum(arr[:, 3], 0.0)
    else:
        values = np.linalg.norm(arr[:, :3], axis=1)
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _contact_centroid(hand_som: torch.Tensor) -> np.ndarray:
    arr = hand_som.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros(3, dtype=np.float32)
    pts = arr[:, :3]
    pressure = _pressure(hand_som)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(pressure)
    pts = pts[finite]
    pressure = pressure[finite]
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    weights = pressure + 1e-6
    return ((pts * weights[:, None]).sum(axis=0) / weights.sum()).astype(np.float32)


def _angle_delta(angle: np.ndarray, prev_angle: np.ndarray | None) -> float:
    if prev_angle is None:
        return 0.0
    delta = np.nan_to_num(np.asarray(angle, dtype=np.float32) - np.asarray(prev_angle, dtype=np.float32), nan=0.0)
    return float(np.linalg.norm(delta))


class SlidingWindowFactorGraph:
    """Small SimTact-style factor graph over per-frame 6D contact wrenches.

    Kim et al. formulate tactile estimation/control as variables connected by
    tactile displacement, kinematic motion, wrench, contact, and smoothness
    factors. R2 does not expose gripper commands, GTSAM variables, or GelSlim
    image markers, so this adapter keeps the essential graph structure with the
    available streams: tactile pressure displacement, WiseGlove motion, contact
    lever arm, and temporal smoothness. It does not train a supervised force
    regressor and it does not borrow modules from the tool-force baseline.
    """

    def __init__(self, config: SimTactGraphConfig):
        self.config = config
        self.rows: List[Dict[str, float]] = []

    def reset(self) -> None:
        self.rows.clear()

    def append(self, pressure_sum: float, pressure_delta: float, motion_delta: float, contact_centroid: np.ndarray) -> np.ndarray:
        centroid = np.nan_to_num(np.asarray(contact_centroid, dtype=np.float32).reshape(3), nan=0.0, posinf=0.0, neginf=0.0)
        self.rows.append(
            {
                "pressure_sum": float(max(pressure_sum, 0.0)),
                "pressure_delta": float(pressure_delta),
                "motion_delta": float(max(motion_delta, 0.0)),
                "cx": float(centroid[0]),
                "cy": float(centroid[1]),
                "cz": float(centroid[2]),
            }
        )
        if len(self.rows) > int(self.config.window):
            self.rows = self.rows[-int(self.config.window) :]
        return self._solve_window()[-1]

    def _solve_window(self) -> np.ndarray:
        n = len(self.rows)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Variables are [Fx,Fy,Fz,Tx,Ty,Tz] for every node in the window.
        a_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        for i, row in enumerate(self.rows):
            base = 6 * i
            tactile_force = self.config.pressure_to_force * (row["pressure_sum"] + row["pressure_delta"])
            motion_force = self.config.pressure_to_force * row["motion_delta"]
            contact = np.asarray([row["cx"], row["cy"], row["cz"]], dtype=np.float64)
            lever = max(float(np.linalg.norm(contact)), 1e-6)
            normal = contact / lever
            tangent = np.asarray([-normal[1], normal[0], 0.0], dtype=np.float64)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-6:
                tangent = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                tangent = tangent / tangent_norm
            torque_axis = np.cross(contact, tangent)
            torque_axis_norm = max(float(np.linalg.norm(torque_axis)), 1e-6)
            torque_axis = torque_axis / torque_axis_norm

            r = np.zeros(6 * n, dtype=np.float64)
            r[base : base + 3] = np.sqrt(self.config.tactile_weight) * normal
            a_rows.append(r)
            b_rows.append(np.sqrt(self.config.tactile_weight) * tactile_force)

            r = np.zeros(6 * n, dtype=np.float64)
            r[base : base + 3] = np.sqrt(self.config.motion_weight) * tangent
            a_rows.append(r)
            b_rows.append(np.sqrt(self.config.motion_weight) * motion_force)

            r = np.zeros(6 * n, dtype=np.float64)
            r[base + 3 : base + 6] = np.sqrt(self.config.tactile_weight) * torque_axis
            r[base : base + 3] = -np.sqrt(self.config.tactile_weight) * self.config.lever_to_torque * np.cross(contact, torque_axis)
            a_rows.append(r)
            b_rows.append(0.0)

            if i > 0:
                for offset in range(6):
                    r = np.zeros(6 * n, dtype=np.float64)
                    r[6 * i + offset] = np.sqrt(self.config.smooth_weight)
                    r[6 * (i - 1) + offset] = -np.sqrt(self.config.smooth_weight)
                    a_rows.append(r)
                    b_rows.append(0.0)

        A = np.stack(a_rows, axis=0)
        b = np.asarray(b_rows, dtype=np.float64)
        sol = np.linalg.lstsq(A, b, rcond=None)[0].reshape(n, 6)
        return sol.astype(np.float32)

    @staticmethod
    def force_torque_summary(wrench6: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench6, dtype=np.float32).reshape(6)
        return np.asarray([np.linalg.norm(wrench[:3]), np.linalg.norm(wrench[3:])], dtype=np.float32)


def _build_config(args: Any | None = None) -> SimTactGraphConfig:
    cfg = SimTactGraphConfig()
    if args is not None:
        for key in asdict(cfg):
            arg_name = f"simtact_{key}"
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                setattr(cfg, key, int(value) if key == "window" else float(value))
    return cfg


def _fit_positive_scale(x: np.ndarray, y: np.ndarray, default: float = 1.0) -> float:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(y, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 1e-8) & (y >= 0.0)
    if int(keep.sum()) < 3:
        return float(default)
    denom = float(np.dot(x[keep], x[keep]))
    if denom <= 1e-12:
        return float(default)
    scale = float(np.dot(x[keep], y[keep]) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        return float(default)
    return scale


def _calibrate_config(args: Any, dataset, train_idx: np.ndarray, cfg: SimTactGraphConfig, batch_size: int) -> Dict[str, Any]:
    max_samples = getattr(args, "simtact_calibration_samples", None)
    if max_samples is None:
        max_samples = getattr(args, "max_train_samples", None)
    max_samples = 0 if max_samples is None else int(max_samples)
    picked = subset_by_max(train_idx, max_samples)
    if len(picked) == 0 or bool(getattr(args, "simtact_disable_calibration", False)):
        return {"enabled": False, "num_samples": 0, "reason": "disabled_or_empty"}

    pressure_features: List[float] = []
    torque_features: List[float] = []
    force_targets: List[float] = []
    torque_targets: List[float] = []
    prev_pressure: Dict[str, float] = {}
    prev_angle: Dict[str, np.ndarray] = {}
    batch_size = max(1, int(batch_size))
    for start in range(0, len(picked), batch_size):
        stop = min(start + batch_size, len(picked))
        hand, _, _, force2, _, mask_force, _ = read_pose_force_batch(dataset, picked[start:stop], include_object=False)
        aux = read_aux_batch(dataset, picked[start:stop])
        angles = aux.get("wiseglove_angle", torch.zeros((stop - start, 19), dtype=torch.float32)).numpy()
        trials = aux.get("trial_key", [""] * (stop - start))
        for i in range(stop - start):
            if float(mask_force[i].reshape(-1)[0].item()) <= 0.5:
                continue
            trial = str(trials[i]) or f"sample:{int(picked[start + i])}"
            p = _pressure(hand[i])
            pressure_sum = float(p.sum())
            pressure_delta = pressure_sum - prev_pressure.get(trial, pressure_sum)
            motion_delta = _angle_delta(angles[i], prev_angle.get(trial))
            centroid = _contact_centroid(hand[i])
            pressure_signal = max(pressure_sum + pressure_delta, 0.0) + 0.25 * max(motion_delta, 0.0)
            lever = max(float(np.linalg.norm(centroid)), 1e-6)
            target = force2[i].numpy().astype(np.float32)
            pressure_features.append(float(pressure_signal))
            torque_features.append(float(pressure_signal * lever))
            force_targets.append(float(max(target[0], 0.0)))
            torque_targets.append(float(max(target[1], 0.0)))
            prev_pressure[trial] = pressure_sum
            prev_angle[trial] = angles[i].copy()

    if len(force_targets) < 3:
        return {"enabled": False, "num_samples": int(len(force_targets)), "reason": "too_few_valid_force_labels"}
    pressure_to_force = _fit_positive_scale(np.asarray(pressure_features), np.asarray(force_targets), cfg.pressure_to_force)
    lever_to_torque = _fit_positive_scale(np.asarray(torque_features) * pressure_to_force, np.asarray(torque_targets), cfg.lever_to_torque)
    cfg.pressure_to_force = pressure_to_force
    cfg.lever_to_torque = lever_to_torque
    return {
        "enabled": True,
        "num_samples": int(len(force_targets)),
        "pressure_to_force": float(pressure_to_force),
        "lever_to_torque": float(lever_to_torque),
    }


def train(args, dataset, train_idx, val_idx, test_idx, device: torch.device) -> Path:
    del val_idx, test_idx, device
    cfg = _build_config(args)
    calibration = _calibrate_config(args, dataset, train_idx, cfg, int(getattr(args, "batch_size", 64)))
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".json")
    save_json(
        ckpt,
        {
            "baseline": NAME,
            "method": "sliding-window tactile factor graph",
            "paper_structure": "factor graph over tactile displacement, kinematic motion, contact lever arm, intrinsic wrench, and smoothness factors; no supervised CNN/MLP regressor",
            "r2_adaptation": "R2 hand_som pressure replaces GelSlim displacement fields, WiseGlove angle increments replace gripper command/pose factors, and the train split calibrates pressure/lever-arm scales before held-out evaluation.",
            "config": asdict(cfg),
            "calibration": calibration,
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
    cfg = _build_config(args)
    try:
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        cfg = SimTactGraphConfig(**payload.get("config", asdict(cfg)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Failed to load trained SimTact config from {ckpt}: {exc}") from exc

    picked = subset_by_max(test_idx, args.max_test_samples)
    graphs: Dict[str, SlidingWindowFactorGraph] = {}
    prev_pressure: Dict[str, float] = {}
    prev_angle: Dict[str, np.ndarray] = {}
    preds, gts, positions = [], [], []
    batch_size = max(1, int(args.batch_size))
    print(f"[{NAME}] sliding-window factor graph evaluate samples={len(picked)}", flush=True)
    for start in range(0, len(picked), batch_size):
        stop = min(start + batch_size, len(picked))
        hand, _, _, force2, _, mask_force, _ = read_pose_force_batch(dataset, picked[start:stop], include_object=False)
        aux = read_aux_batch(dataset, picked[start:stop])
        angles = aux.get("wiseglove_angle", torch.zeros((stop - start, 19), dtype=torch.float32)).numpy()
        trials = aux.get("trial_key", [""] * (stop - start))
        for i in range(stop - start):
            if float(mask_force[i].reshape(-1)[0].item()) <= 0.5:
                continue
            trial = str(trials[i]) or f"sample:{int(picked[start + i])}"
            graph = graphs.setdefault(trial, SlidingWindowFactorGraph(cfg))
            p = _pressure(hand[i])
            pressure_sum = float(p.sum())
            pressure_delta = pressure_sum - prev_pressure.get(trial, pressure_sum)
            motion_delta = _angle_delta(angles[i], prev_angle.get(trial))
            centroid = _contact_centroid(hand[i])
            wrench6 = graph.append(pressure_sum, pressure_delta, motion_delta, centroid)
            preds.append(SlidingWindowFactorGraph.force_torque_summary(wrench6))
            gts.append(force2[i].numpy().astype(np.float32))
            positions.append(int(picked[start + i]))
            prev_pressure[trial] = pressure_sum
            prev_angle[trial] = angles[i].copy()

    metrics: Dict[str, float] = {"num_test_samples": int(len(preds)), "checkpoint": str(ckpt)}
    if preds:
        pred_arr = np.stack(preds, axis=0).astype(np.float32)
        gt_arr = np.stack(gts, axis=0).astype(np.float32)
        metrics["force_mae"] = float(np.abs(pred_arr[:, 0] - gt_arr[:, 0]).mean())
        metrics["torque_mae"] = float(np.abs(pred_arr[:, 1] - gt_arr[:, 1]).mean())
        if getattr(args, "save_per_sample", True):
            metrics["per_sample_predictions"] = {
                "positions": np.asarray(positions, dtype=np.int64),
                "target_force": gt_arr,
                "pred_force": pred_arr,
                "mask_force": np.ones((len(preds), 1), dtype=np.float32),
            }
    return metrics



