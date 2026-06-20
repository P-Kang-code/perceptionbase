from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .runtime import checkpoint_path, compute_pose_metrics, pose9d_to_poseflat, poseflat_to_pose9d, read_pose_force_batch, subset_by_max


NAME = "ag_pose"


class GeometricAwarePointAggregator(nn.Module):
    """Thin R2 adapter inspired by AG-Pose's geometric-aware aggregation."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        h = int(hidden_dim)
        self.local = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, h, 1),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(nn.Conv1d(h, h // 2, 1), nn.ReLU(inplace=True), nn.Conv1d(h // 2, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h // 2),
            nn.ReLU(inplace=True),
            nn.Linear(h // 2, 9),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        feat = self.local(points.transpose(1, 2).contiguous())
        weights = torch.softmax(self.attn(feat), dim=2)
        attentive = (feat * weights).sum(dim=2)
        pooled = feat.amax(dim=2)
        return self.head(torch.cat([pooled, attentive], dim=1))


def _sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    x = x.float()
    if x.ndim != 2 or x.shape[0] == 0:
        return torch.zeros((int(n), x.shape[-1] if x.ndim == 2 else 3), dtype=torch.float32, device=x.device)
    if x.shape[0] == int(n):
        return x
    idx = torch.linspace(0, x.shape[0] - 1, int(n), device=x.device).round().long()
    return x.index_select(0, idx)


def _make_points(hand_som: torch.Tensor, object_data: torch.Tensor, num_points: int) -> torch.Tensor:
    rows = []
    n_obj = max(1, int(num_points) // 2)
    n_hand = max(1, int(num_points) - n_obj)
    for hand, obj in zip(hand_som, object_data):
        obj_xyz = _sample_rows(obj[:, :3], n_obj)
        hand_xyz = _sample_rows(hand[:, :3], n_hand)
        hand_pressure = hand[:, 3:4] if hand.shape[-1] >= 4 else torch.linalg.norm(hand[:, :3], dim=1, keepdim=True)
        pressure = _sample_rows(hand_pressure, n_hand)
        obj_center = obj_xyz.mean(dim=0, keepdim=True)
        obj_radius = torch.linalg.norm(obj_xyz - obj_center, dim=1, keepdim=True)
        hand_radius = torch.linalg.norm(hand_xyz - obj_center, dim=1, keepdim=True)
        obj_feat = torch.cat([obj_xyz, torch.zeros((n_obj, 1), device=obj_xyz.device), torch.ones((n_obj, 1), device=obj_xyz.device), torch.zeros((n_obj, 1), device=obj_xyz.device), obj_radius, torch.zeros((n_obj, 1), device=obj_xyz.device)], dim=1)
        hand_feat = torch.cat([hand_xyz, pressure, torch.zeros((n_hand, 1), device=hand_xyz.device), torch.ones((n_hand, 1), device=hand_xyz.device), hand_radius, torch.ones((n_hand, 1), device=hand_xyz.device)], dim=1)
        pts = torch.cat([obj_feat, hand_feat], dim=0)
        pts[:, :3] = pts[:, :3] - pts[:, :3].mean(dim=0, keepdim=True)
        rows.append(torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.stack(rows, dim=0)


def _iter_batches(dataset: Any, indices: np.ndarray, batch_size: int, num_points: int, device: torch.device):
    for start in range(0, len(indices), max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), len(indices))
        hand, obj, pose, _, mask_pose, _, scale = read_pose_force_batch(dataset, indices[start:stop], include_object=True)
        keep = mask_pose.reshape(-1) > 0.5
        if not bool(keep.any()):
            continue
        yield (
            indices[start:stop][keep.cpu().numpy()],
            _make_points(hand[keep].to(device), obj[keep].to(device), num_points),
            pose[keep].to(device),
            scale[keep].to(device),
        )


def _evaluate(model: nn.Module, dataset: Any, indices: np.ndarray, args: Any, device: torch.device) -> Dict[str, Any]:
    picked = subset_by_max(indices, args.max_test_samples)
    num_points = int(getattr(args, "ag_pose_num_points", 1024))
    batch_size = int(getattr(args, "ag_pose_eval_batch_size", args.batch_size))
    preds, gts, scales, positions = [], [], [], []
    model.eval()
    with torch.no_grad():
        for pos, pts, pose, scale in _iter_batches(dataset, picked, batch_size, num_points, device):
            pred = pose9d_to_poseflat(model(pts)).detach().cpu().numpy()
            preds.append(pred)
            gts.append(pose.cpu().numpy())
            scales.append(scale.cpu().numpy())
            positions.append(np.asarray(pos, dtype=np.int64))
    if not preds:
        return {"mse": float("inf"), "rotation_error_deg": float("inf"), "translation_l1": float("inf"), "num_test_samples": 0}
    pred_arr = np.concatenate(preds, axis=0).astype(np.float32)
    gt_arr = np.concatenate(gts, axis=0).astype(np.float32)
    scale_arr = np.concatenate(scales, axis=0).astype(np.float32)
    metrics = compute_pose_metrics(pred_arr, gt_arr, scale=scale_arr)
    metrics["num_test_samples"] = int(pred_arr.shape[0])
    if getattr(args, "save_per_sample", True):
        metrics["per_sample_predictions"] = {
            "positions": np.concatenate(positions, axis=0).astype(np.int64),
            "pred_pose": pred_arr,
            "target_pose": gt_arr,
            "mask_pose": np.ones((pred_arr.shape[0], 1), dtype=np.float32),
            "scale": scale_arr,
        }
    return metrics


def train(args, dataset, train_idx, val_idx, test_idx, device: torch.device) -> Path:
    del test_idx
    train_pick = subset_by_max(train_idx, args.max_train_samples)
    val_pick = subset_by_max(val_idx, args.max_val_samples)
    num_points = int(getattr(args, "ag_pose_num_points", 1024))
    model = GeometricAwarePointAggregator(hidden_dim=int(getattr(args, "ag_pose_hidden_dim", 256))).to(device)
    best_metric = float("inf")
    best_state = None
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    resume_ckpt = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    if resume_ckpt is not None:
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"{NAME} resume checkpoint not found: {resume_ckpt}")
        payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        best_metric = float(payload.get("best_metric", best_metric))
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[{NAME}] resumed training from {resume_ckpt} best_metric={best_metric:.6f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    progress_interval = max(1, int(getattr(args, "progress_interval", 1)))
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for _, pts, pose, _ in _iter_batches(dataset, train_pick, int(args.batch_size), num_points, device):
            target = poseflat_to_pose9d(pose)
            pred = model(pts)
            pose_loss = nn.functional.smooth_l1_loss(pred, target)
            rot_loss = nn.functional.mse_loss(pose9d_to_poseflat(pred)[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]], pose[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]])
            loss = pose_loss + 0.05 * rot_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(args, "grad_clip", 5.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metric = float(np.mean(losses)) if losses else float("inf")
        val_metrics: Dict[str, Any] = {
            "mse": val_metric,
            "rotation_error_deg": float("nan"),
            "translation_l1": float("nan"),
        }
        if len(val_pick) > 0:
            val_metrics = _evaluate(model, dataset, val_pick, args, device)
            val_metric = float(val_metrics.get("mse", val_metric))
        if val_metric < best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "baseline": NAME,
                    "state_dict": best_state,
                    "best_metric": best_metric,
                    "best_epoch": int(epoch),
                    "best_val_metrics": {
                        "mse": float(val_metrics.get("mse", val_metric)),
                        "rotation_error_deg": float(val_metrics.get("rotation_error_deg", float("nan"))),
                        "translation_l1": float(val_metrics.get("translation_l1", float("nan"))),
                    },
                    "num_points": num_points,
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"[{NAME}] saved best checkpoint epoch={epoch} path={ckpt}", flush=True)
        if epoch == 1 or epoch % progress_interval == 0 or epoch == int(args.epochs):
            print(
                f"[{NAME}] epoch={epoch}/{int(args.epochs)} "
                f"train_loss={float(np.mean(losses)) if losses else float('nan'):.6f} "
                f"val_mse={val_metric:.6f} "
                f"val_angle_deg={float(val_metrics.get('rotation_error_deg', float('nan'))):.3f} "
                f"val_distance={float(val_metrics.get('translation_l1', float('nan'))):.4f}",
                flush=True,
            )
    if best_state is None:
        torch.save({"baseline": NAME, "state_dict": model.state_dict(), "best_metric": best_metric, "num_points": num_points, "args": vars(args)}, ckpt)
    return ckpt


def test(args, dataset, test_idx, device: torch.device, checkpoint: Path | None = None) -> Dict[str, Any]:
    ckpt = checkpoint or checkpoint_path(args.output_dir, NAME, suffix=".pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"{NAME} checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    num_points = int(payload.get("num_points", getattr(args, "ag_pose_num_points", 1024)))
    model = GeometricAwarePointAggregator(hidden_dim=int(getattr(args, "ag_pose_hidden_dim", 256))).to(device)
    model.load_state_dict(payload["state_dict"])
    metrics = _evaluate(model, dataset, test_idx, args, device)
    metrics["checkpoint"] = str(ckpt)
    return metrics




