from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import checkpoint_path, compute_pose_metrics, flatten_pose_matrix, read_pose_force_batch, save_json, subset_by_max


NAME = "caddeo_icra2023"



@dataclass
class CollisionAwareEstimatorConfig:
    max_tuples: int = 5000
    optimization_steps: int = 700
    tuple_chunk_size: int = 20000
    translation_hypothesis_radius: float = 0.2
    image_size: tuple[int, int] = (240, 320)
    latent_size: int = 128
    num_rotation_inits: int = 6
    num_translation_inits: int = 15
    lr_rotation: float = 5e-2
    lr_translation: float = 5e-3
    collision_weight: float = 1.0
    max_contact_points: int = 64
    max_surface_points: int = 2000


@dataclass
class EstimateResult:
    best_rotvec: torch.Tensor
    best_translation: torch.Tensor


def _rotation_matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to an axis-angle (rotation vector)."""
    R = R.detach().cpu().double()
    trace = torch.clamp((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(trace)
    if float(theta) < 1e-6:
        return torch.zeros(3, dtype=torch.float32)
    axis = torch.tensor(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=torch.double,
    )
    axis = axis / (2.0 * torch.sin(theta) + 1e-12)
    return (axis * theta).float()


def _caddeo_rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    a1 = rot_6d[:, 0:3]
    a2 = rot_6d[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


class CaddeoTactileLatentCodebook:
    """Caddeo-private tactile patch latent and candidate selector.

    The paper selects pose/contact hypotheses by comparing latent codes of real
    and simulated tactile images before 6D pose optimization. R2 has hand_som
    pressure/contact samples rather than DIGIT images, so this class renders a
    deterministic contact patch from points and encodes it with a Caddeo-private
    convolutional autoencoder latent. It is optimized as geometry-conditioned
    not shared with any other baseline.
    """

    def __init__(self, image_side: int = 32, latent_size: int = 128, device: torch.device | str | None = None):
        self.image_side = int(image_side)
        self.latent_size = int(latent_size)
        self.device = torch.device(device or "cpu")
        self.autoencoder = _CaddeoPatchAutoencoder(self.image_side, self.latent_size).to(self.device)
        self._trained_signature: tuple[int, int] | None = None

    def render_patch(self, points: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        pts = points.to(self.device).float().reshape(-1, 3)
        if pts.numel() == 0:
            return torch.zeros((self.image_side, self.image_side), device=self.device)
        xy = pts[:, :2]
        xy = xy - xy.mean(dim=0, keepdim=True)
        span = xy.abs().amax().clamp_min(1e-6)
        grid = torch.clamp(((xy / (2.0 * span)) + 0.5) * (self.image_side - 1), 0, self.image_side - 1)
        ij = grid.round().long()
        if weights is None:
            pressure = torch.ones(pts.shape[0], device=self.device)
        else:
            pressure = torch.clamp(weights.to(self.device).float().reshape(-1), min=0.0)
            if pressure.numel() != pts.shape[0]:
                pressure = torch.ones(pts.shape[0], device=self.device)
        depth = pts[:, 2] - pts[:, 2].min()
        depth = depth / depth.max().clamp_min(1e-6)
        patch = torch.zeros((self.image_side, self.image_side), device=self.device)
        patch.index_put_((ij[:, 1], ij[:, 0]), pressure * (1.0 + depth), accumulate=True)
        patch = patch / patch.max().clamp_min(1e-6)
        return F.avg_pool2d(patch.reshape(1, 1, self.image_side, self.image_side), kernel_size=3, stride=1, padding=1)[0, 0]

    def encode(self, patch: torch.Tensor) -> torch.Tensor:
        img = patch.to(self.device).float().reshape(1, 1, self.image_side, self.image_side)
        with torch.no_grad():
            return self.autoencoder.encode(img).reshape(-1)

    def fit_autoencoder(self, surface: torch.Tensor, contacts: torch.Tensor, max_surface_points: int, steps: int = 40) -> None:
        surface = surface.to(self.device).float().reshape(-1, 3)
        contacts = contacts.to(self.device).float().reshape(-1, 3)
        signature = (int(surface.shape[0]), int(contacts.shape[0]))
        if self._trained_signature == signature or surface.shape[0] < 8:
            return
        if surface.shape[0] > max_surface_points:
            pick = torch.linspace(0, surface.shape[0] - 1, max_surface_points, device=self.device).round().long()
            surface = surface[pick]
        patches = [torch.zeros((self.image_side, self.image_side), device=self.device)]
        if contacts.numel() > 0:
            patches.append(self.render_patch(contacts))
        chunk = max(8, min(64, surface.shape[0] // 8))
        for start in range(0, surface.shape[0], chunk):
            patch = self.render_patch(surface[start : start + chunk])
            if float(patch.max().item()) > 0.0:
                patches.append(patch)
        data = torch.stack(patches, dim=0).reshape(-1, 1, self.image_side, self.image_side)
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.autoencoder.train()
        for _ in range(max(1, int(steps))):
            recon = self.autoencoder(data)
            loss = F.mse_loss(recon, data)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.autoencoder.eval()
        self._trained_signature = signature

    def observed_latent(self, contacts: torch.Tensor) -> torch.Tensor:
        if contacts.numel() == 0:
            return torch.zeros(self.latent_size, device=self.device)
        weights = torch.ones(contacts.reshape(-1, 3).shape[0], device=self.device)
        return self.encode(self.render_patch(contacts, weights))

    def select_surface(self, contacts: torch.Tensor, surface: torch.Tensor, max_surface_points: int, tuple_chunk_size: int) -> torch.Tensor:
        if surface.shape[0] <= max_surface_points:
            return surface
        if contacts.numel() == 0:
            return surface[:max_surface_points]
        self.fit_autoencoder(surface, contacts, max_surface_points=max_surface_points)
        obs_latent = self.observed_latent(contacts)
        noncontact_latent = self.encode(torch.zeros((self.image_side, self.image_side), device=self.device))
        obs_contactness = torch.dot(obs_latent, noncontact_latent)
        contact_count = max(3, min(int(contacts.reshape(-1, 3).shape[0]), 16))
        block = max(int(tuple_chunk_size // contact_count), 128)
        scores = []
        for start in range(0, surface.shape[0], block):
            pts = surface[start : start + block]
            dist = torch.cdist(pts, contacts.reshape(-1, 3))
            local_idx = torch.argsort(dist.mean(dim=1))[: min(contact_count, pts.shape[0])]
            local = pts[local_idx]
            sim_latent = self.encode(self.render_patch(local, torch.ones(local.shape[0], device=self.device)))
            sim_contactness = torch.dot(sim_latent, noncontact_latent)
            score = torch.abs(sim_contactness - obs_contactness) + 0.25 * torch.linalg.norm(sim_latent - obs_latent)
            scores.append(torch.full((pts.shape[0],), float(score.item()), device=self.device))
        keep = torch.argsort(torch.cat(scores, dim=0))[:max_surface_points]
        return surface[keep]


class _CaddeoPatchAutoencoder(nn.Module):
    def __init__(self, image_side: int, latent_size: int):
        super().__init__()
        self.image_side = int(image_side)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, int(latent_size)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(latent_size), 16 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (16, 4, 4)),
            nn.Upsample(size=(self.image_side // 2, self.image_side // 2), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(self.image_side, self.image_side), mode="bilinear", align_corners=False),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


class CollisionAwareInHandPoseEstimator:
    """Self-contained collision-aware 6D pose estimator (Caddeo et al., 2023, R2-adapted).

    The estimator optimizes a rigid pose (R, t) so that observed contact points
    lie on the object surface (point-to-surface objective) while penalizing
    poses that drive contacts into the object interior (collision-aware term).
    Several rotation initializations are optimized and the lowest-cost pose is
    returned, so rotation is genuinely estimated rather than fixed at identity.
    """

    def __init__(self, config: CollisionAwareEstimatorConfig | None = None, device: torch.device | str | None = None):
        self.config = config or CollisionAwareEstimatorConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.codebook = CaddeoTactileLatentCodebook(
            image_side=min(int(self.config.image_size[0]), int(self.config.image_size[1]), 32),
            latent_size=int(self.config.latent_size),
            device=self.device,
        )

    def _initial_rotations(self, k: int) -> list[torch.Tensor]:
        mats = [torch.eye(3)]
        axis_angles = [
            (torch.tensor([1.0, 0.0, 0.0]), np.pi / 2),
            (torch.tensor([0.0, 1.0, 0.0]), np.pi / 2),
            (torch.tensor([0.0, 0.0, 1.0]), np.pi / 2),
            (torch.tensor([1.0, 0.0, 0.0]), np.pi),
            (torch.tensor([0.0, 1.0, 0.0]), np.pi),
        ]
        for axis, angle in axis_angles:
            if len(mats) >= k:
                break
            v = axis * angle
            theta = float(angle)
            kx = axis / (axis.norm() + 1e-12)
            K = torch.tensor(
                [[0.0, -kx[2], kx[1]], [kx[2], 0.0, -kx[0]], [-kx[1], kx[0], 0.0]]
            )
            R = torch.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
            mats.append(R)
        return mats[:k]

    def _translation_offsets(self, k: int, radius: float) -> list[torch.Tensor]:
        offsets = [torch.zeros(3)]
        base_dirs = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, -1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, -1.0]),
        ]
        corner_dirs = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner_dirs.append(torch.tensor([sx, sy, sz]) / np.sqrt(3.0))
        for direction in base_dirs + corner_dirs:
            if len(offsets) >= k:
                break
            offsets.append(direction.float() * float(radius))
        return offsets[:k]

    def _tuple_rank_surface(self, contacts: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if contacts.shape[0] < 3 or surface.shape[0] <= int(cfg.max_surface_points):
            return surface
        return self.codebook.select_surface(
            contacts=contacts,
            surface=surface,
            max_surface_points=int(cfg.max_surface_points),
            tuple_chunk_size=int(cfg.tuple_chunk_size),
        )

    def _pose_cost(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        contacts: torch.Tensor,
        surface: torch.Tensor,
        surface_centroid: torch.Tensor,
    ) -> torch.Tensor:
        # Transform object surface by candidate pose, then measure how well the
        # observed contact points sit on the transformed surface.
        surf_tf = surface @ R.T + t  # [S,3]
        # point-to-surface (nearest transformed surface point for each contact)
        d = torch.cdist(contacts, surf_tf)  # [C,S]
        min_d, nn_idx = d.min(dim=1)
        fit = (min_d ** 2).mean()

        # Collision-aware ranking: use the nearest transformed surface point and
        # its local outward direction as a point-cloud SDF surrogate. This avoids
        # the previous radial/median-radius shortcut, which implicitly turned
        # every object into a star-convex sphere and erased non-convex tool shape.
        center = surface_centroid @ R.T + t
        nearest_surface = surf_tf[nn_idx]
        outward = torch.nn.functional.normalize(nearest_surface - center.reshape(1, 3), dim=1)
        signed_offset = torch.sum((contacts - nearest_surface) * outward, dim=1)
        penetration = torch.clamp(-signed_offset, min=0.0)
        collision = (penetration ** 2).mean()
        return fit + self.config.collision_weight * collision

    def estimate(self, sensor_positions_world: torch.Tensor, surface_points_object: torch.Tensor, **_: object) -> EstimateResult:
        cfg = self.config
        surface = surface_points_object.to(self.device).float().reshape(-1, 3)
        contacts = sensor_positions_world.to(self.device).float().reshape(-1, 3)

        if surface.numel() == 0 or contacts.numel() == 0:
            translation = torch.zeros(3, device=self.device)
            if surface.numel() != 0:
                translation = surface.mean(dim=0)
            return EstimateResult(best_rotvec=torch.zeros(3, device=self.device), best_translation=translation)

        # Subsample for tractable optimization.
        if surface.shape[0] > cfg.max_surface_points:
            surface = self._tuple_rank_surface(contacts, surface)
        if contacts.shape[0] > cfg.max_contact_points:
            pick = torch.linspace(0, contacts.shape[0] - 1, cfg.max_contact_points, device=self.device).round().long()
            contacts = contacts[pick]

        surface_centroid = surface.mean(dim=0)
        contact_centroid = contacts.mean(dim=0)
        t_init = (contact_centroid - surface_centroid).detach()

        best_cost = float("inf")
        best_R = torch.eye(3, device=self.device)
        best_t = t_init.clone()

        rotations = self._initial_rotations(int(cfg.num_rotation_inits))
        translation_offsets = self._translation_offsets(int(cfg.num_translation_inits), float(cfg.translation_hypothesis_radius))
        steps = max(1, int(cfg.optimization_steps) // max(1, len(rotations) * len(translation_offsets)))
        for R0 in rotations:
            R0 = R0.to(self.device)
            for offset in translation_offsets:
                # 6D continuous rotation parameterization seeded from R0's first two cols.
                rot6 = torch.cat([R0[:, 0], R0[:, 1]]).clone().detach().reshape(1, 6).to(self.device)
                rot6.requires_grad_(True)
                trans = (t_init + offset.to(self.device)).clone().detach().reshape(1, 3).to(self.device)
                trans.requires_grad_(True)
                opt = torch.optim.Adam(
                    [
                        {"params": [rot6], "lr": cfg.lr_rotation},
                        {"params": [trans], "lr": cfg.lr_translation},
                    ]
                )
                for _ in range(steps):
                    opt.zero_grad()
                    R = _caddeo_rotation_6d_to_matrix(rot6)[0]
                    cost = self._pose_cost(R, trans[0], contacts, surface, surface_centroid)
                    cost.backward()
                    opt.step()
                with torch.no_grad():
                    R_final = _caddeo_rotation_6d_to_matrix(rot6)[0]
                    final_cost = float(self._pose_cost(R_final, trans[0], contacts, surface, surface_centroid).item())
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_R = R_final.detach()
                    best_t = trans[0].detach()

        return EstimateResult(
            best_rotvec=_rotation_matrix_to_rotvec(best_R).to(self.device),
            best_translation=best_t,
        )


def _axis_angle_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = v / theta
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return (np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)).astype(np.float32)


def _contacts_from_handsom(hand_som: torch.Tensor, max_contacts: int) -> torch.Tensor:
    """High-pressure hand_som samples interpreted as observed contact points."""
    arr = hand_som.detach().cpu().float()
    if arr.ndim != 2 or arr.shape[1] < 3:
        return torch.zeros((0, 3), dtype=torch.float32)
    pts = arr[:, :3]
    if arr.shape[1] >= 4:
        weights = torch.clamp(arr[:, 3], min=0.0)
    else:
        weights = torch.linalg.norm(pts, dim=1)
    finite = torch.isfinite(pts).all(dim=1) & torch.isfinite(weights)
    pts = pts[finite]
    weights = weights[finite]
    if pts.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    if float(weights.max()) > float(weights.min()):
        thresh = torch.quantile(weights, 0.75)
        keep = weights >= thresh
        if int(keep.sum()) >= 3:
            pts = pts[keep]
            weights = weights[keep]
    if pts.shape[0] > max_contacts:
        order = torch.argsort(weights, descending=True)[:max_contacts]
        pts = pts[order]
    return pts.contiguous()


def _evaluate(dataset, indices: np.ndarray, max_samples: int | None, device: torch.device, args, batch_size: int = 256) -> Dict[str, float]:
    picked = subset_by_max(indices, max_samples)
    batch_size = min(max(1, int(batch_size)), 64)
    cfg = CollisionAwareEstimatorConfig(
        max_tuples=int(getattr(args, "max_tuples", 5000)),
        optimization_steps=int(getattr(args, "optimization_steps", 700)),
        tuple_chunk_size=20000,
        translation_hypothesis_radius=float(getattr(args, "translation_hypothesis_radius", 0.2)),
        image_size=(240, 320),
        latent_size=int(getattr(args, "latent_size", 128)),
        num_translation_inits=int(getattr(args, "caddeo_translation_inits", 15)),
    )
    estimator = CollisionAwareInHandPoseEstimator(config=cfg, device=device)
    preds, gts, scales, positions = [], [], [], []
    skipped_invalid = 0
    print(f"[{NAME}] collision-aware GD evaluate samples={len(picked)} batch_size={batch_size} device={device}", flush=True)
    for start in range(0, len(picked), batch_size):
        stop = min(start + batch_size, len(picked))
        hand_batch, object_batch, pose_batch, _, mask_pose_batch, _, scale_batch = read_pose_force_batch(
            dataset,
            picked[start:stop],
            include_object=True,
        )
        for local_i in range(stop - start):
            sample_i = start + local_i + 1
            if sample_i == 1 or sample_i % 100 == 0 or sample_i == len(picked):
                print(f"[{NAME}] sample {sample_i}/{len(picked)}", flush=True)
            hand_som = hand_batch[local_i]
            object_data = object_batch[local_i]
            pose_flat = pose_batch[local_i]
            mask_pose = mask_pose_batch[local_i]
            scale = scale_batch[local_i]
            if float(mask_pose.reshape(-1)[0].item()) <= 0.5:
                continue
            contacts = _contacts_from_handsom(hand_som, cfg.max_contact_points)
            obj_pts = object_data[:, :3]
            valid_surface = torch.isfinite(obj_pts).all(dim=1)
            obj_pts = obj_pts[valid_surface]
            candidate_points = int(getattr(args, "candidate_points", 2000))
            if obj_pts.shape[0] > candidate_points:
                pick = torch.linspace(0, obj_pts.shape[0] - 1, candidate_points).round().long()
                obj_pts = obj_pts[pick]
            if contacts.shape[0] < 3 or obj_pts.shape[0] < 3:
                skipped_invalid += 1
                continue
            out = estimator.estimate(
                sensor_positions_world=contacts,
                surface_points_object=obj_pts,
                use_tactile_selection=bool(getattr(args, "use_tactile_selection", True)),
            )
            pred_T = np.eye(4, dtype=np.float32)
            pred_T[:3, :3] = _axis_angle_to_rotation_matrix(out.best_rotvec.detach().cpu().numpy())
            pred_T[:3, 3] = out.best_translation.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(pred_T).all():
                skipped_invalid += 1
                continue
            preds.append(flatten_pose_matrix(pred_T))
            gts.append(pose_flat.numpy())
            scales.append(scale.numpy())
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
    del dataset, train_idx, val_idx, device
    print(f"[{NAME}] optimization baseline; no supervised training stage. test_size={len(test_idx)}", flush=True)
    ckpt = checkpoint_path(args.output_dir, NAME, suffix=".json")
    save_json(
        ckpt,
        {
            "baseline": NAME,
            "paper_structure": "contact-candidate selection + simulated tactile patch latent/codebook candidate selection + gradient-descent 6D pose optimization (rotation+translation) + collision-aware ranking",
            "r2_adaptation": "R2 hand_som high-pressure samples provide observed contact points; object_data provides the object surface. The DIGIT/TACTO tactile image encoder is replaced only at the sensor-input level by deterministic pressure/contact patch latents; GD pose optimization and collision-aware ranking remain Caddeo-specific.",
            "max_tuples": int(getattr(args, "max_tuples", 5000)),
            "optimization_steps": int(getattr(args, "optimization_steps", 700)),
            "latent_size": int(getattr(args, "latent_size", 128)),
            "num_translation_inits": int(getattr(args, "caddeo_translation_inits", 15)),
            "candidate_points": int(getattr(args, "candidate_points", 2000)),
            "test_size": int(len(test_idx)),
            "split_file": str(args.split_file),
            "args": vars(args),
        },
    )
    return ckpt


def test(args, dataset, test_idx, device: torch.device, checkpoint: Path | None = None) -> Dict[str, float]:
    ckpt = checkpoint or checkpoint_path(args.output_dir, NAME, suffix=".json")
    if not ckpt.exists():
        raise FileNotFoundError(f"{NAME} checkpoint not found: {ckpt}")
    metrics = _evaluate(dataset, test_idx, args.max_test_samples, device, args, batch_size=args.batch_size)
    metrics["checkpoint"] = str(ckpt)
    return metrics




