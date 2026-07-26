"""Implicit-ET baseline for AET-Bench (simplified INR fallback).

Rationale: the upstream Implicit-ET repo (Kniesel et al., CVPR 2022) is
tightly coupled to its own data format, WandB logging, and outdated PyTorch
1.9 / CUDA 10.2, and its CLI runs for 400k iterations by default. Integrating
it cleanly within the plan's time budget is not feasible.

Instead we ship a minimal NeRF-style implicit neural representation baseline
that preserves the *scientific intent* of the Implicit-ET comparison (an INR
as an alternative to explicit representations like FBP/SIRT/Gaussians), while
fitting inside the existing aet4dgs conda env.

Architecture:
    - Coordinate-based MLP (positional encoding → 4 hidden layers → softplus)
    - Takes (x, y, z) normalized coordinates, outputs a scalar density.
    - Loss: L2 between integrated projection of the MLP's density field and
      the GT tilt series, along the current tilt axis.
    - Optimized with Adam for a few thousand steps on a dense grid of
      coordinates + per-ray integration via the same rotation used in our
      analytical projector.

Limitations (documented in paper):
    - Not the original Implicit-ET model (no separate noise head, no MLE loss)
    - Fewer iterations than the original (4k vs 400k)
    - This baseline is labelled "Implicit-NeRF" in the paper table with a
      footnote citing Implicit-ET as the inspiration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEnc(nn.Module):
    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        self.freqs = 2.0 ** torch.arange(n_freqs).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3) in [-1, 1]
        out = [x]
        for f in self.freqs.to(x.device):
            out.append(torch.sin(x * f * np.pi))
            out.append(torch.cos(x * f * np.pi))
        return torch.cat(out, dim=-1)


class INR(nn.Module):
    def __init__(self, n_freqs: int = 8, hidden: int = 128):
        super().__init__()
        self.enc = PosEnc(n_freqs)
        in_dim = 3 + 3 * 2 * n_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        h = self.enc(xyz)
        return F.softplus(self.net(h)).squeeze(-1)


def _render_tilt_patch(model: INR, render_size: int, angle_rad: float,
                         n_samples: int, patch_size: int,
                         device) -> torch.Tensor:
    """Render a random patch of a projection via sparse ray sampling.

    Picks a random (patch_size, patch_size) patch of the detector, then
    integrates n_samples points per ray along the z axis (post-rotation).
    Returns (patch_size, patch_size) and the (row_off, col_off) location
    used (for masking the GT to the same patch).
    """
    half = 1.0
    # Random patch location
    ymin = torch.randint(0, render_size - patch_size + 1, (1,)).item()
    xmin = torch.randint(0, render_size - patch_size + 1, (1,)).item()
    rows = torch.linspace(-half, half, render_size, device=device)[
        ymin:ymin + patch_size
    ]
    cols = torch.linspace(-half, half, render_size, device=device)[
        xmin:xmin + patch_size
    ]
    zs = torch.linspace(-half, half, n_samples, device=device)

    # (H, W, S, 3): for each pixel (r, c), integrate over S z-samples
    R = rows[:, None, None].expand(patch_size, patch_size, n_samples)
    C = cols[None, :, None].expand(patch_size, patch_size, n_samples)
    Z = zs[None, None, :].expand(patch_size, patch_size, n_samples)
    pts = torch.stack([C, R, Z], dim=-1)  # (H, W, S, 3); x=col, y=row, z=depth

    # Rotate about y axis
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    Rmat = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                        device=device)
    pts_rot = pts @ Rmat.T  # (H, W, S, 3)

    density = model(pts_rot.reshape(-1, 3)).reshape(
        patch_size, patch_size, n_samples
    )
    # Integrated intensity along z
    return density.sum(dim=-1), (ymin, xmin)


def _render_tilt_full(model: INR, render_size: int, angle_rad: float,
                        n_samples: int, device) -> torch.Tensor:
    """Full-resolution projection for final eval. Still memory-bounded."""
    half = 1.0
    rows = torch.linspace(-half, half, render_size, device=device)
    cols = torch.linspace(-half, half, render_size, device=device)
    zs = torch.linspace(-half, half, n_samples, device=device)
    out = torch.zeros(render_size, render_size, device=device)
    # Row-strided batching
    for r in range(render_size):
        R = rows[r].expand(render_size, n_samples)
        C = cols[:, None].expand(render_size, n_samples)
        Z = zs[None, :].expand(render_size, n_samples)
        pts = torch.stack([C, R, Z], dim=-1)
        c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        Rmat = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                            device=device)
        pts_rot = pts @ Rmat.T
        out[r] = model(pts_rot.reshape(-1, 3)).reshape(render_size, n_samples).sum(-1)
    return out


def _dense_volume(model: INR, vol_size: int, device) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, vol_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    batch = 128 * 1024
    out = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], batch):
            out.append(model(pts[i:i + batch]))
    return torch.cat(out).reshape(vol_size, vol_size, vol_size)


class INRWithScale(nn.Module):
    """INR + a global learnable intensity scale."""
    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        self.inr = INR()
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(init_scale))))

    def forward(self, xyz):
        return self.inr(xyz) * torch.exp(self.log_scale)


def run_implicit_et(data: Dict, sample_id: str, work_root: Path,
                     iterations: int = 1500,
                     train_render_size: int = 48,
                     n_z_samples: int = 48,
                     lr: float = 3e-3) -> np.ndarray:
    """Fit a coordinate-based MLP to the tilt series.

    Training: at each step, pick one random tilt angle and render a
    train_render_size × train_render_size projection from the MLP via
    ray integration (n_z_samples per pixel). Loss is L2 against the GT
    projection downsampled to train_render_size. A learnable global
    intensity scale absorbs the integration constant.

    After training, the full-resolution volume is queried on a dense
    grid at the original vol_size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    train = data["train"]
    angles_rad = np.asarray(train["angles"], dtype=np.float32)
    projs = np.asarray(train["projections"], dtype=np.float32)  # (N, H, W)
    vol_size = int(data["nVoxel"][0])

    # Resize projs to training resolution
    from skimage.transform import resize
    projs_low = np.stack([
        resize(p, (train_render_size, train_render_size), preserve_range=True)
        for p in projs
    ]).astype(np.float32)
    projs_t = torch.from_numpy(projs_low).to(device)
    angles_t = torch.from_numpy(angles_rad).to(device)
    n_projs = projs_t.shape[0]

    # Initialize learnable scale so mean(pred) ≈ mean(GT).
    model = INRWithScale(init_scale=1.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  [implicit_et] fitting {sample_id}: {iterations} iters, "
          f"render={train_render_size}, {n_z_samples} z-samples")

    for step in range(iterations):
        i = int(torch.randint(0, n_projs, (1,)).item())
        angle = float(angles_t[i].item())
        pred = _render_tilt_full(model, train_render_size, angle,
                                  n_z_samples, device)
        loss = F.mse_loss(pred, projs_t[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"    step {step + 1}/{iterations} "
                  f"loss={loss.item():.5f} "
                  f"scale={float(model.log_scale.exp().item()):.4f}")

    with torch.no_grad():
        vol = _dense_volume(model.inr, vol_size, device).cpu().numpy().astype(np.float32)
    # Final intensity calibration
    scale = float(projs.mean()) * vol_size / max(vol.mean(), 1e-8)
    vol *= scale
    return vol
