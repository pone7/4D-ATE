"""InstaMap baseline for AET-Bench: hash-encoded neural field for tomography.

This implements the core idea from InstaMap (instant-NGP for cryo-EM):
replace explicit multi-resolution feature grids (as in NeAT) with
multi-resolution hash encoding for faster convergence and better
fine-detail reconstruction.

When ``tinycudann`` is available, we use NVIDIA's fused hash-grid
implementation for maximum speed (matching the upstream InstaMap repo at
``/data/aet-bench/instamap/``). Otherwise we fall back to
a pure-PyTorch multi-resolution hash grid that is functionally equivalent
but slower.

Architecture:
    HashEncoding(xyz, L levels) -> concat -> MLP(2 hidden, 64 units) -> density

Training follows the same projection-matching loop used by run_neat.py:
    - Pick a random tilt angle each step
    - Render a low-resolution projection via ray integration
    - MSE loss against downsampled ground-truth projection
    - Adam optimizer, ~2500 iterations

Key hyper-parameters mirror InstaMap defaults:
    - 8 hash-grid levels, log2 table size = 19
    - Base resolution 8, finest resolution = vol_size
    - 2-feature-dim per level (16 total encoding features)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-PyTorch multi-resolution hash grid (fallback when tinycudann absent)
# ---------------------------------------------------------------------------

def _hash_index(coords_int: torch.Tensor, table_size: int) -> torch.Tensor:
    """Spatial hash following instant-NGP: XOR-based prime hashing.

    Parameters
    ----------
    coords_int : (N, 3) long tensor of integer grid coordinates.
    table_size : number of entries in the hash table (power of 2).

    Returns
    -------
    (N,) long tensor of table indices in [0, table_size).
    """
    primes = torch.tensor([1, 2654435761, 805459861],
                           dtype=torch.long, device=coords_int.device)
    h = (coords_int * primes).sum(dim=-1)
    return h % table_size


class _HashLevel(nn.Module):
    """One resolution level of the multi-resolution hash grid."""

    def __init__(self, resolution: int, table_size: int, feat_dim: int):
        super().__init__()
        self.resolution = resolution
        self.table_size = table_size
        self.feat_dim = feat_dim
        self.table = nn.Parameter(torch.zeros(table_size, feat_dim))
        nn.init.uniform_(self.table, -1e-4, 1e-4)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Trilinearly interpolated lookup.

        Parameters
        ----------
        xyz : (N, 3) in [-1, 1].

        Returns
        -------
        (N, feat_dim) feature vectors.
        """
        # Map [-1,1] -> [0, resolution-1]
        scaled = (xyz * 0.5 + 0.5) * (self.resolution - 1)
        floor = scaled.floor().long()
        frac = scaled - floor.float()  # (N, 3), each in [0, 1)

        # 8 corners of the containing voxel
        result = torch.zeros(xyz.shape[0], self.feat_dim,
                             device=xyz.device, dtype=xyz.dtype)
        for dx in (0, 1):
            wx = frac[:, 0] if dx == 1 else (1.0 - frac[:, 0])
            for dy in (0, 1):
                wy = frac[:, 1] if dy == 1 else (1.0 - frac[:, 1])
                for dz in (0, 1):
                    wz = frac[:, 2] if dz == 1 else (1.0 - frac[:, 2])
                    w = (wx * wy * wz).unsqueeze(-1)  # (N, 1)
                    corner = floor + torch.tensor([dx, dy, dz],
                                                   device=xyz.device)
                    idx = _hash_index(corner, self.table_size)
                    result = result + w * self.table[idx]
        return result


class MultiResHashEncoding(nn.Module):
    """Pure-PyTorch multi-resolution hash encoding (instant-NGP style).

    Parameters mirror InstaMap / tinycudann defaults.
    """

    def __init__(self, n_levels: int = 8, base_resolution: int = 8,
                 finest_resolution: int = 128,
                 log2_hashmap_size: int = 19,
                 features_per_level: int = 2):
        super().__init__()
        self.n_levels = n_levels
        table_size = 2 ** log2_hashmap_size

        # Per-level resolution grows geometrically
        if n_levels > 1:
            per_level_scale = math.exp(
                math.log(finest_resolution / base_resolution) / (n_levels - 1)
            )
        else:
            per_level_scale = 1.0

        self.levels = nn.ModuleList()
        for lvl in range(n_levels):
            res = max(int(base_resolution * per_level_scale ** lvl + 0.5), 2)
            # If the grid is small enough to fit without hashing, use it
            # directly; otherwise use the full hash table.
            actual_table = min(res ** 3, table_size)
            self.levels.append(_HashLevel(res, actual_table, features_per_level))

        self.out_dim = n_levels * features_per_level

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz expected in [-1, 1]
        parts = [lvl(xyz) for lvl in self.levels]
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Network: hash encoding + small MLP -> density
# ---------------------------------------------------------------------------

class HashField(nn.Module):
    """Hash-encoded neural density field for tomographic reconstruction.

    Tries tinycudann first; falls back to pure-PyTorch hash grid.
    """

    def __init__(self, finest_resolution: int = 128,
                 n_levels: int = 8, base_resolution: int = 8,
                 log2_hashmap_size: int = 19,
                 features_per_level: int = 2,
                 hidden: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        self._use_tcnn = False
        enc_out_dim: int

        try:
            import tinycudann as tcnn  # type: ignore
            per_level_scale = math.exp(
                math.log(finest_resolution / base_resolution) / max(n_levels - 1, 1)
            )
            self.encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": n_levels,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear",
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden,
                    "n_hidden_layers": n_hidden_layers,
                },
            )
            self._use_tcnn = True
        except Exception:
            # Fallback: pure-PyTorch hash grid + manual MLP
            self.hash_enc = MultiResHashEncoding(
                n_levels=n_levels,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                log2_hashmap_size=log2_hashmap_size,
                features_per_level=features_per_level,
            )
            enc_out_dim = self.hash_enc.out_dim + 3  # concat raw xyz

            layers = [nn.Linear(enc_out_dim, hidden), nn.ReLU()]
            for _ in range(n_hidden_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers.append(nn.Linear(hidden, 1))
            self.mlp = nn.Sequential(*layers)

        # Global learnable intensity scale
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : (N, 3) coordinates in [-1, 1].

        Returns
        -------
        (N,) non-negative density values.
        """
        if self._use_tcnn:
            # tinycudann expects coords in [0, 1]
            inp = xyz * 0.5 + 0.5
            raw = self.encoding(inp).squeeze(-1)
        else:
            feat = self.hash_enc(xyz)
            h = torch.cat([feat, xyz], dim=-1)
            raw = self.mlp(h).squeeze(-1)

        return F.softplus(raw) * torch.exp(self.log_scale)


# ---------------------------------------------------------------------------
# Projection rendering (parallel-beam, same geometry as run_neat.py)
# ---------------------------------------------------------------------------

def _render_projection(model: nn.Module, render_size: int,
                       angle_rad: float, n_z: int,
                       device: torch.device) -> torch.Tensor:
    """Render a single parallel-beam projection at *angle_rad* about the y-axis.

    Returns a (render_size, render_size) image formed by integrating
    density along z-directed rays after rotating the sample.
    """
    rows = torch.linspace(-1.0, 1.0, render_size, device=device)
    cols = torch.linspace(-1.0, 1.0, render_size, device=device)
    zs = torch.linspace(-1.0, 1.0, n_z, device=device)

    X, Y, Z = torch.meshgrid(cols, rows, zs, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1)  # (W, H, S, 3)

    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    Rmat = torch.tensor([[c, 0.0, s],
                          [0.0, 1.0, 0.0],
                          [-s, 0.0, c]], device=device)
    pts_rot = pts @ Rmat.T
    density = model(pts_rot.reshape(-1, 3)).reshape(render_size, render_size, n_z)
    return density.sum(dim=-1)


# ---------------------------------------------------------------------------
# Dense volume query (for final output)
# ---------------------------------------------------------------------------

def _dense_volume(model: nn.Module, vol_size: int,
                  device: torch.device) -> torch.Tensor:
    """Query the field on a dense (vol_size)^3 grid to produce a volume."""
    coords = torch.linspace(-1.0, 1.0, vol_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    batch = 128 * 1024
    out = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], batch):
            out.append(model(pts[i:i + batch]))
    return torch.cat(out).reshape(vol_size, vol_size, vol_size)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_instamap(data: Dict, sample_id: str, out_dir: Path,
                 iterations: int = 2500,
                 train_render_size: int = 64,
                 n_z_samples: int = 64,
                 lr: float = 3e-3) -> np.ndarray:
    """Train a hash-encoded neural field on projections and return a 3D volume.

    This is the InstaMap baseline: essentially NeAT with hash encoding
    instead of explicit multi-resolution feature grids.

    Parameters
    ----------
    data : dict
        NAF-format data dictionary containing at least:
        - data["train"]["angles"]      : (N,) tilt angles in radians
        - data["train"]["projections"] : (N, H, W) projection images
        - data["nVoxel"]               : (3,) target volume dimensions
    sample_id : str
        Identifier for the sample (used in logging).
    out_dir : Path
        Working directory for intermediate files (created if needed).
    iterations : int
        Number of training iterations.
    train_render_size : int
        Resolution at which projections are rendered during training.
    n_z_samples : int
        Number of depth samples per ray during training.
    lr : float
        Learning rate for Adam.

    Returns
    -------
    np.ndarray
        Reconstructed 3D volume of shape (vol_size, vol_size, vol_size).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Unpack data --------------------------------------------------------
    train = data["train"]
    angles_rad = np.asarray(train["angles"], dtype=np.float32)
    projs = np.asarray(train["projections"], dtype=np.float32)  # (N, H, W)
    vol_size = int(data["nVoxel"][0])

    # -- Downsample projections for training --------------------------------
    from skimage.transform import resize
    projs_low = np.stack([
        resize(p, (train_render_size, train_render_size), preserve_range=True)
        for p in projs
    ]).astype(np.float32)

    projs_t = torch.tensor(projs_low, device=device, dtype=torch.float32)
    angles_t = torch.tensor(
        np.ascontiguousarray(angles_rad, dtype=np.float32),
        device=device, dtype=torch.float32,
    )
    n_projs = projs_t.shape[0]

    # -- Build model --------------------------------------------------------
    # Match InstaMap defaults: 8 levels, base 8, finest = vol_size,
    # hash table 2^19 entries, 2 features/level, 2 hidden layers of 64.
    model = HashField(
        finest_resolution=min(vol_size, 256),
        n_levels=8,
        base_resolution=8,
        log2_hashmap_size=19,
        features_per_level=2,
        hidden=64,
        n_hidden_layers=2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    backend = "tinycudann" if model._use_tcnn else "pytorch-hashgrid"
    print(f"  [instamap] fitting {sample_id}: {iterations} iters, "
          f"render={train_render_size}, samples={n_z_samples}, "
          f"backend={backend}")

    # -- Training loop ------------------------------------------------------
    for step in range(iterations):
        i = int(torch.randint(0, n_projs, (1,)).item())
        angle = float(angles_t[i].item())

        pred = _render_projection(model, train_render_size, angle,
                                  n_z_samples, device)
        loss = F.mse_loss(pred, projs_t[i])

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            print(f"    step {step + 1}/{iterations}  loss={loss.item():.5f}")

    # -- Extract dense volume -----------------------------------------------
    with torch.no_grad():
        vol = _dense_volume(model, vol_size, device).cpu().numpy().astype(np.float32)

    # Light smoothing (same as run_neat.py)
    try:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=0.8).astype(np.float32)
    except Exception:
        pass

    # Intensity calibration: match mean projection intensity
    scale = float(projs.mean()) * vol_size / max(vol.mean(), 1e-8)
    vol *= scale

    return vol
