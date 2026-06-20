from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet101

from .runtime import checkpoint_path, count_parameters, read_pose_force_batch, subset_by_max, sync_if_cuda


NAME = "tool_force"


@dataclass
class ModelConfig:
    is_rgb: bool = False
    image_size: int = 224
    output_dim: int = 1
    resnet_backbone: str = "resnet101"
    reference_mode: str = "train_low_force_by_object"


class InitialCNN(nn.Module):
    """ToolForce per-sensor CNN: 2/6 input channels -> 64 -> 128 -> 256."""

    def __init__(self, is_rgb: bool = False):
        super().__init__()
        in_channels = 6 if is_rgb else 2
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = self.pool(F.relu(self.bn3(self.conv3(x)), inplace=True))
        return self.dropout(x)


class GAMAttention(nn.Module):
    """Global Attention Mechanism used in the released ToolForce implementation."""

    def __init__(self, in_channels: int, out_channels: int, rate: int = 4):
        super().__init__()
        hidden = max(1, int(in_channels / rate))
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel = self.channel_attention(x.permute(0, 2, 3, 1).reshape(b, -1, c)).reshape(b, h, w, c)
        x = x * channel.permute(0, 3, 1, 2)
        return x * self.spatial_attention(x).sigmoid()


class ToolForceEstimatorBaseline(nn.Module):
    """ToolForce network.

    Li & Thuruthel use two DIGIT streams, zero-force reference images, per-sensor
    3-layer CNN encoders, GAM attention before/after fusion, a ResNet-101
    backbone, and a 2048-1024-512-1 regressor.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.initial_cnn1 = InitialCNN(is_rgb=self.config.is_rgb)
        self.initial_cnn2 = InitialCNN(is_rgb=self.config.is_rgb)
        self.attention1 = GAMAttention(in_channels=256, out_channels=256)
        self.attention2 = GAMAttention(in_channels=256, out_channels=256)
        self.attention3 = GAMAttention(in_channels=512, out_channels=512)
        self.resnet = resnet101(weights=None)
        self.resnet.conv1 = nn.Conv2d(512, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, int(self.config.output_dim)),
        )

    def forward(
        self,
        current_1: torch.Tensor,
        reference_1: torch.Tensor,
        current_2: torch.Tensor,
        reference_2: torch.Tensor,
    ) -> torch.Tensor:
        features1 = self.initial_cnn1(torch.cat((current_1, reference_1), dim=1))
        features2 = self.initial_cnn2(torch.cat((current_2, reference_2), dim=1))
        features1 = self.attention1(features1)
        features2 = self.attention2(features2)
        fused = self.attention3(torch.cat((features1, features2), dim=1))
        return self.fc(self.resnet(fused))


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    vmin = values.amin(dim=(1, 2, 3), keepdim=True)
    vmax = values.amax(dim=(1, 2, 3), keepdim=True)
    return (values - vmin) / (vmax - vmin + 1e-6)


def _make_sensor_image(values: torch.Tensor, side: int) -> torch.Tensor:
    # Preserve all tactile samples by 1D resampling before reshaping to a dense image.
    b, n = values.shape
    target = side * side
    x = values.float().reshape(b, 1, n)
    x = F.interpolate(x, size=target, mode="linear", align_corners=False).reshape(b, 1, side, side)
    return _normalize_map(x)


def handsom_to_dual_tactile_images(hand_som: torch.Tensor, side: int = 224) -> tuple[torch.Tensor, torch.Tensor]:
    if hand_som.ndim != 3:
        raise ValueError(f"Expected hand_som [B,N,C], got {tuple(hand_som.shape)}")
    values = hand_som[..., 3] if hand_som.shape[-1] >= 4 else torch.linalg.norm(hand_som[..., :3], dim=-1)
    if hand_som.shape[-1] >= 3:
        x = hand_som[..., 0]
        order = torch.argsort(x, dim=1)
        values = torch.gather(values, 1, order)
    n = values.shape[1]
    mid = max(1, n // 2)
    left = values[:, :mid]
    right = values[:, mid:] if mid < n else values[:, :mid]
    return _make_sensor_image(left, side), _make_sensor_image(right, side)


def _object_name_for_position(dataset: Dataset, position: int) -> str:
    names = getattr(dataset, "object_name", None)
    if isinstance(names, list) and 0 <= int(position) < len(names):
        return str(names[int(position)])
    return "unknown"


def _build_reference_bank(dataset: Dataset, indices: np.ndarray, side: int) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    indices = np.asarray(indices, dtype=np.int64)
    best: Dict[str, tuple[float, torch.Tensor, torch.Tensor]] = {}
    batch_size = 512
    for start in range(0, len(indices), batch_size):
        pick = indices[start:start + batch_size]
        hand_som, _, _, force2, _, mask_force, _ = read_pose_force_batch(dataset, pick, include_object=False)
        img1, img2 = handsom_to_dual_tactile_images(hand_som, side=side)
        scores = torch.linalg.norm(force2.float(), dim=1).reshape(-1)
        valid = mask_force.reshape(-1) > 0.5
        for local_i, pos in enumerate(pick):
            if not bool(valid[local_i]):
                continue
            name = _object_name_for_position(dataset, int(pos))
            score = float(scores[local_i].item())
            if name not in best or score < best[name][0]:
                best[name] = (score, img1[local_i:local_i + 1].cpu(), img2[local_i:local_i + 1].cpu())
    if not best:
        return {}
    global_ref1 = torch.stack([value[1].squeeze(0) for value in best.values()], dim=0).mean(dim=0, keepdim=True)
    global_ref2 = torch.stack([value[2].squeeze(0) for value in best.values()], dim=0).mean(dim=0, keepdim=True)
    refs = {name: (value[1], value[2]) for name, value in best.items()}
    refs["__global__"] = (global_ref1.cpu(), global_ref2.cpu())
    return refs


class ToolForceBatchDataset(Dataset):
    def __init__(
        self,
        source_dataset: Dataset,
        indices: np.ndarray,
        target_idx: int,
        batch_size: int,
        tag: str,
        reference_bank: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
        image_size: int,
    ):
        self.source_dataset = source_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_idx = int(target_idx)
        self.batch_size = max(1, int(batch_size))
        self.tag = str(tag)
        self.reference_bank = dict(reference_bank or {})
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def _reference_for_positions(self, positions: np.ndarray, current_1: torch.Tensor, current_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref1, ref2 = [], []
        global_ref = self.reference_bank.get("__global__")
        for local_i, pos in enumerate(positions):
            item = self.reference_bank.get(_object_name_for_position(self.source_dataset, int(pos)), global_ref)
            if item is None:
                # Last-resort modality adaptation: use a strongly smoothed current image as grasp baseline.
                r1 = F.avg_pool2d(current_1[local_i:local_i + 1], kernel_size=31, stride=1, padding=15)
                r2 = F.avg_pool2d(current_2[local_i:local_i + 1], kernel_size=31, stride=1, padding=15)
            else:
                r1, r2 = item
            ref1.append(r1.squeeze(0))
            ref2.append(r2.squeeze(0))
        return torch.stack(ref1, dim=0).float(), torch.stack(ref2, dim=0).float()

    def __getitem__(self, batch_i: int):
        start = int(batch_i) * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        pick = self.indices[start:stop]
        print(f"[data:{self.tag}] loading batch {int(batch_i) + 1}/{len(self)} samples={len(pick)}", flush=True)
        hand_som, _, pose_flat, force2, mask_pose, mask_force, scale = read_pose_force_batch(self.source_dataset, pick, include_object=False)
        current_1, current_2 = handsom_to_dual_tactile_images(hand_som, side=self.image_size)
        reference_1, reference_2 = self._reference_for_positions(pick, current_1, current_2)
        return {
            "current_1": current_1.float(),
            "reference_1": reference_1.float(),
            "current_2": current_2.float(),
            "reference_2": reference_2.float(),
            "pose_flat": pose_flat,
            "target": force2[:, self.target_idx:self.target_idx + 1].float(),
            "mask_pose": mask_pose.reshape(-1, 1).float(),
            "mask_force": mask_force.reshape(-1, 1).float(),
            "scale": scale.reshape(-1, 1).float(),
        }


def _forward_batch(model: nn.Module, batch: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return model(
        batch["current_1"].to(device),
        batch["reference_1"].to(device),
        batch["current_2"].to(device),
        batch["reference_2"].to(device),
    )


def _train_scalar_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args,
    tag: str,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, object],
):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    progress_interval = max(0, int(getattr(args, "progress_interval", 0)))
    effective_batch_size = getattr(getattr(train_loader, "dataset", None), "batch_size", train_loader.batch_size)
    print(
        f"[{tag}] epochs={args.epochs} lr={args.lr} batch_size={effective_batch_size} "
        f"params={count_parameters(model):,} train_batches={len(train_loader)} val_batches={len(val_loader)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        sync_if_cuda(device)
        epoch_start = time.perf_counter()
        model.train()
        run_loss = 0.0
        n = 0
        for batch in train_loader:
            tgt = batch["target"].to(device)
            m = batch["mask_force"].to(device)
            pred = _forward_batch(model, batch, device)
            loss = (F.l1_loss(pred, tgt, reduction="none") * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.item())
            n += 1
            if progress_interval and (n % progress_interval == 0 or n == len(train_loader)):
                print(
                    f"[{tag}] epoch {epoch:4d}/{args.epochs}  batch {n:4d}/{len(train_loader)}  "
                    f"train-loss={run_loss / max(n, 1):.6f}",
                    flush=True,
                )
        sync_if_cuda(device)
        train_elapsed = time.perf_counter() - epoch_start
        eval_start = time.perf_counter()
        pred, gt = _predict_scalar(model, val_loader, device)
        sync_if_cuda(device)
        val = float(np.abs(pred - gt).mean()) if pred.size else float("inf")
        print(
            f"[{tag}] epoch {epoch:4d}/{args.epochs}  "
            f"train-loss={run_loss / max(n, 1):.6f}  val-mae={val:.6f}  "
            f"epoch-time={time.perf_counter() - epoch_start:.2f}s  train-time={train_elapsed:.2f}s  "
            f"eval-time={time.perf_counter() - eval_start:.2f}s",
            flush=True,
        )
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            out = dict(checkpoint_payload)
            out.update({"state_dict": best_state, "best_val_mae": best_val, "best_epoch": best_epoch})
            torch.save(out, checkpoint)
            print(f"[{tag}] ** saved best checkpoint: epoch={epoch:4d}  val-mae={best_val:.6f}  path={checkpoint}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mae": best_val, "best_epoch": best_epoch}


@torch.no_grad()
def _predict_scalar(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        pred = _forward_batch(model, batch, device)
        mask = batch["mask_force"].to(device).reshape(-1) > 0.5
        if mask.any():
            preds.append(pred.reshape(-1)[mask].detach().cpu().numpy())
            gts.append(batch["target"].to(device).reshape(-1)[mask].detach().cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(preds), np.concatenate(gts)


def _make_loader(dataset, indices, target_idx, batch_size, tag, reference_bank, image_size, shuffle, num_workers):
    return DataLoader(
        ToolForceBatchDataset(dataset, indices, target_idx, batch_size, tag, reference_bank, image_size),
        batch_size=None,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train(args, dataset, train_idx, val_idx, test_idx, device: torch.device) -> Path:
    train_pick = subset_by_max(train_idx, args.max_train_samples)
    val_pick = subset_by_max(val_idx, args.max_val_samples)
    cfg = ModelConfig()
    ckpt = checkpoint_path(args.output_dir, NAME)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    reference_banks = {}
    reference_bank = _build_reference_bank(dataset, train_pick, cfg.image_size)
    for target_idx, key in ((0, "force"), (1, "torque")):
        reference_banks[key] = reference_bank
        train_loader = _make_loader(dataset, train_pick, target_idx, args.batch_size, f"{NAME}_{key}", reference_bank, cfg.image_size, True, args.num_workers)
        val_loader = _make_loader(dataset, val_pick, target_idx, args.batch_size, f"{NAME}_{key}-val", reference_bank, cfg.image_size, False, args.num_workers)
        single_ckpt = ckpt.with_name(f"{ckpt.stem}_{key}{ckpt.suffix}")
        models[key] = _train_scalar_model(
            ToolForceEstimatorBaseline(cfg),
            train_loader,
            val_loader,
            device,
            args,
            f"{NAME}_{key}",
            single_ckpt,
            {
                "baseline": NAME,
                "target": key,
                "model_class": "ToolForceEstimatorBaseline",
                "paper_structure": "dual-sensor current/reference CNN + GAM + ResNet-101 + 2048-1024-512-1",
                "r2_adaptation": "hand_som pressure/proprioceptive samples are resampled into two 224x224 grayscale tactile maps; zero-force references use lowest-force training samples by object.",
                "config": asdict(cfg),
                "reference_bank": reference_bank,
                "train_size": int(len(train_pick)),
                "val_size": int(len(val_pick)),
                "test_size": int(len(test_idx)),
                "split_file": str(args.split_file),
                "args": vars(args),
            },
        )
    torch.save(
        {
            "baseline": NAME,
            "model_class": "ToolForceEstimatorBaseline",
            "paper_structure": "dual-sensor current/reference CNN + GAM + ResNet-101 + 2048-1024-512-1",
            "r2_adaptation": "hand_som pressure/proprioceptive samples are resampled into two 224x224 grayscale tactile maps; zero-force references use lowest-force training samples by object.",
            "config": asdict(cfg),
            "state_dict_force": models["force"][0].state_dict(),
            "state_dict_torque": models["torque"][0].state_dict(),
            "reference_bank_force": reference_banks["force"],
            "reference_bank_torque": reference_banks["torque"],
            "train_meta_force": models["force"][1],
            "train_meta_torque": models["torque"][1],
            "test_size": int(len(test_idx)),
            "split_file": str(args.split_file),
            "args": vars(args),
        },
        ckpt,
    )
    return ckpt


def test(args, dataset, test_idx, device: torch.device, checkpoint: Path | None = None) -> Dict[str, float]:
    ckpt = checkpoint or checkpoint_path(args.output_dir, NAME)
    if not ckpt.exists():
        raise FileNotFoundError(f"{NAME} checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload.get("config", {}))
    model_force = ToolForceEstimatorBaseline(cfg).to(device)
    model_torque = ToolForceEstimatorBaseline(cfg).to(device)
    model_force.load_state_dict(payload["state_dict_force"])
    model_torque.load_state_dict(payload["state_dict_torque"])
    picked = subset_by_max(test_idx, args.max_test_samples)
    loader_force = _make_loader(
        dataset,
        picked,
        0,
        args.batch_size,
        f"{NAME}-test-force",
        payload.get("reference_bank_force", {}),
        cfg.image_size,
        False,
        args.num_workers,
    )
    loader_torque = _make_loader(
        dataset,
        picked,
        1,
        args.batch_size,
        f"{NAME}-test-torque",
        payload.get("reference_bank_torque", {}),
        cfg.image_size,
        False,
        args.num_workers,
    )
    pf, gf = _predict_scalar(model_force, loader_force, device)
    pt, gt = _predict_scalar(model_torque, loader_torque, device)
    metrics: Dict[str, float] = {"num_test_samples": int(len(picked)), "checkpoint": str(ckpt)}
    if pf.size:
        metrics["force_mae"] = float(np.abs(pf - gf).mean())
    if pt.size:
        metrics["torque_mae"] = float(np.abs(pt - gt).mean())
    if getattr(args, "save_per_sample", True):
        _, _, _, force2, _, mask_force, scale = read_pose_force_batch(dataset, picked, include_object=False)
        valid = mask_force.reshape(-1).numpy() > 0.5
        if pf.size and pt.size and len(pf) == int(valid.sum()) and len(pt) == int(valid.sum()):
            metrics["per_sample_predictions"] = {
                "positions": picked[valid],
                "target_force": force2.numpy()[valid],
                "pred_force": np.stack([pf, pt], axis=1).astype(np.float32),
                "mask_force": mask_force.numpy()[valid],
                "scale": scale.numpy()[valid],
            }
    return metrics



