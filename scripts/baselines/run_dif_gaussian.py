"""DIF-Gaussian (TTO-only) baseline for AET-Bench.

DIF-Gaussian (Lin et al., MICCAI 2024) is a supervised method that uses a
UNet encoder to extract features from projections, places 3D Gaussians on a
regular grid, and predicts per-query-point intensity via Gaussian-weighted
interpolation of learned feature vectors passed through a point decoder MLP.

Since the supervised UNet backbone requires pretrained weights from the same
data distribution (CBCT medical scans), it cannot be directly applied to AET
data. Instead, we implement the **test-time optimization (TTO) component**
as a standalone optimization baseline.

The TTO-only variant retains DIF-Gaussian's core innovations:
  1. Deep intensity field: each Gaussian stores a learned feature vector;
     a shared MLP maps features to scalar intensity (not raw opacity).
  2. Gaussian-weighted feature interpolation: query points aggregate features
     from their k-nearest Gaussians weighted by the Gaussian PDF evaluated
     at the query-to-center displacement.
  3. Test-time optimization: all parameters (positions, covariances, features,
     decoder MLP) are jointly optimized via projection consistency loss.

Forward model: parallel-beam projection via dense volume sampling. At each
training step, one tilt angle is sampled, the volume is queried on a 3D grid
rotated by that angle, and the line integral (sum along depth) is compared
to the measured projection.

Reference: dif-gaussian/ (cloned at /data/aet-bench/dif-gaussian/)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gaussian covariance helpers (adapted from dif-gaussian/code/models/gs_utils.py)
# ---------------------------------------------------------------------------

def _build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix.  r: (N, 4)."""
    norm = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-8)
    q = r / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(q.shape[0], 3, 3, device=r.device, dtype=r.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _build_covariance(log_s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build covariance Sigma = L @ L^T from log-scale and quaternion.

    log_s: (K, 3)  -- log of scaling factors
    r:     (K, 4)  -- quaternion (w,x,y,z)

    Returns: (K, 3, 3) positive-definite covariance matrices.
    """
    s = torch.exp(log_s)  # (K, 3), positive
    R = _build_rotation(r)  # (K, 3, 3)
    L = torch.zeros_like(R)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L  # (K, 3, 3)
    return L @ L.transpose(1, 2)


# ---------------------------------------------------------------------------
# Gaussian-weighted feature query (adapted from dif-gaussian gs_utils.query_gs)
# ---------------------------------------------------------------------------

def _query_gaussian_features(
    query_pts: torch.Tensor,   # (N, 3)
    gs_centers: torch.Tensor,  # (K, 3)
    gs_cov_inv: torch.Tensor,  # (K, 3, 3)
    gs_cov_det: torch.Tensor,  # (K,)
    gs_feats: torch.Tensor,    # (K, C)
    k_neighbors: int = 3,
) -> torch.Tensor:
    """For each query point, find k nearest Gaussians and compute
    Gaussian-PDF-weighted sum of their feature vectors.

    Returns: (N, C) interpolated features.
    """
    # Pairwise squared distances for kNN
    # query_pts: (N, 3), gs_centers: (K, 3)
    N = query_pts.shape[0]
    K = gs_centers.shape[0]
    k = min(k_neighbors, K)

    # cdist for kNN (N, K)
    dists = torch.cdist(query_pts.unsqueeze(0), gs_centers.unsqueeze(0)).squeeze(0)
    _, idx = dists.topk(k, dim=1, largest=False)  # (N, k) indices into K

    # Gather neighbor data
    neb_centers = gs_centers[idx]   # (N, k, 3)
    neb_inv = gs_cov_inv[idx]       # (N, k, 3, 3)
    neb_det = gs_cov_det[idx]       # (N, k)
    neb_feats = gs_feats[idx]       # (N, k, C)

    # Displacement: query - neighbor center
    diff = (query_pts.unsqueeze(1) - neb_centers).unsqueeze(-1)  # (N, k, 3, 1)

    # Gaussian weight: (2*pi)^{-3/2} * det^{-1/2} * exp(-0.5 * d^T Sigma^{-1} d)
    mahal = (diff.transpose(-2, -1) @ neb_inv @ diff).squeeze(-1).squeeze(-1)  # (N, k)
    log_w = -0.5 * mahal - 0.5 * torch.log(neb_det.clamp(min=1e-12))
    # Subtract constant -1.5*log(2*pi) for numerical stability (cancels in softmax-like normalization)
    weights = torch.exp(log_w)  # (N, k)

    # Normalize weights (DIF-Gaussian sums without normalizing, but
    # normalizing is more stable for TTO where the covariance is learned)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)  # (N, k)

    # Weighted feature sum
    interp = (weights.unsqueeze(-1) * neb_feats).sum(dim=1)  # (N, C)
    return interp


# ---------------------------------------------------------------------------
# Point decoder MLP (adapted from dif-gaussian/code/models/point_decoder.py)
# ---------------------------------------------------------------------------

class PointDecoderMLP(nn.Module):
    """Small MLP mapping per-point feature vector to scalar intensity.

    Architecture mirrors DIF-Gaussian's PointDecoder but uses standard
    Linear layers (no batch dimension required) with residual connections.
    """

    def __init__(self, feat_dim: int, hidden_dims: tuple = (64, 32)):
        super().__init__()
        layers = []
        in_d = feat_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(in_d, hd), nn.LeakyReLU(inplace=True)])
            in_d = hd
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (N, C) -> (N,) non-negative intensity."""
        return F.softplus(self.net(feats).squeeze(-1))


# ---------------------------------------------------------------------------
# DIF-Gaussian TTO model: bundles all learnable parameters
# ---------------------------------------------------------------------------

class DIFGaussianTTO(nn.Module):
    """Test-time-optimization variant of DIF-Gaussian.

    Learnable parameters:
      - gs_centers:  (K, 3)   Gaussian center positions in [-1, 1]^3
      - gs_log_s:    (K, 3)   Log-scale of covariance (isotropic init)
      - gs_quat:     (K, 4)   Quaternion for covariance rotation
      - gs_feats:    (K, C)   Per-Gaussian feature vectors
      - decoder:     PointDecoderMLP mapping features -> intensity
      - log_scale:   scalar   Global intensity scale
    """

    def __init__(self, n_gaussians: int, feat_dim: int = 16,
                 hidden_dims: tuple = (64, 32),
                 init_positions: np.ndarray | None = None,
                 init_log_scale: float = -3.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.k_neighbors = min(5, n_gaussians)

        # --- Gaussian parameters ---
        if init_positions is not None:
            centers = torch.tensor(init_positions, dtype=torch.float32)
            n_gaussians = centers.shape[0]
        else:
            centers = (torch.rand(n_gaussians, 3) - 0.5) * 2.0
        self.gs_centers = nn.Parameter(centers)

        # Isotropic init: scale ~ inter-Gaussian spacing
        spacing_log = float(np.log(2.0 / max(round(n_gaussians ** (1.0 / 3)), 1)))
        self.gs_log_s = nn.Parameter(
            torch.full((n_gaussians, 3), spacing_log)
        )

        # Identity rotation (w=1, x=y=z=0)
        quat_init = torch.zeros(n_gaussians, 4)
        quat_init[:, 0] = 1.0
        self.gs_quat = nn.Parameter(quat_init)

        # Feature vectors
        self.gs_feats = nn.Parameter(
            torch.randn(n_gaussians, feat_dim) * 0.1
        )

        # Decoder MLP
        self.decoder = PointDecoderMLP(feat_dim, hidden_dims)

        # Global intensity scale
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale))

    def _covariance_inv_det(self):
        """Compute covariance, its inverse, and determinant."""
        cov = _build_covariance(self.gs_log_s, self.gs_quat)   # (K, 3, 3)
        # Add small diagonal for numerical stability
        eye = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0) * 1e-6
        cov = cov + eye
        cov_inv = torch.linalg.inv(cov)         # (K, 3, 3)
        cov_det = torch.linalg.det(cov)          # (K,)
        return cov_inv, cov_det

    def query_density(self, pts: torch.Tensor) -> torch.Tensor:
        """Evaluate intensity at arbitrary 3D points.

        pts: (N, 3) in [-1, 1]^3
        Returns: (N,) non-negative intensity values.
        """
        cov_inv, cov_det = self._covariance_inv_det()

        # Gaussian-weighted feature interpolation
        interp_feats = _query_gaussian_features(
            pts, self.gs_centers, cov_inv, cov_det,
            self.gs_feats, self.k_neighbors,
        )  # (N, C)

        # Decode to intensity
        intensity = self.decoder(interp_feats)  # (N,)
        return intensity * torch.exp(self.log_scale)

    def forward_volume(self, vol_size: int) -> torch.Tensor:
        """Evaluate on a dense (vol_size)^3 grid. Returns (V, V, V)."""
        coords = torch.linspace(-1.0, 1.0, vol_size, device=self.gs_centers.device)
        X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
        pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

        batch = 64 * 1024
        out = []
        with torch.no_grad():
            for i in range(0, pts.shape[0], batch):
                out.append(self.query_density(pts[i:i + batch]))
        return torch.cat(out).reshape(vol_size, vol_size, vol_size)


# ---------------------------------------------------------------------------
# Rendering: parallel-beam projection via dense sampling
# ---------------------------------------------------------------------------

def _render_projection(model: DIFGaussianTTO, render_size: int,
                       angle_rad: float, n_z: int,
                       device: torch.device) -> torch.Tensor:
    """Render a single parallel-beam projection at *angle_rad* (rotation
    about the Y axis).

    Returns: (render_size, render_size) integrated intensity image.
    """
    rows = torch.linspace(-1.0, 1.0, render_size, device=device)
    cols = torch.linspace(-1.0, 1.0, render_size, device=device)
    zs = torch.linspace(-1.0, 1.0, n_z, device=device)
    X, Y, Z = torch.meshgrid(cols, rows, zs, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1)  # (W, H, S, 3)

    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    Rmat = torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        device=device, dtype=torch.float32,
    )
    pts_rot = pts @ Rmat.T
    density = model.query_density(pts_rot.reshape(-1, 3)).reshape(
        render_size, render_size, n_z
    )
    return density.sum(dim=-1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dif_gaussian(data: Dict, sample_id: str, out_dir: Path,
                     iterations: int = 2000,
                     n_gaussians: int = 2000,
                     feat_dim: int = 16,
                     train_render_size: int = 64,
                     n_z_samples: int = 64,
                     lr: float = 5e-3) -> np.ndarray:
    """DIF-Gaussian TTO-only baseline.

    Parameters
    ----------
    data : dict
        NAF-format data dict. Must contain:
          data["train"]["angles"]      -- (N_proj,) array of tilt angles in radians
          data["train"]["projections"] -- (N_proj, H, W) array of projection images
          data["nVoxel"]               -- (3,) voxel grid dimensions
    sample_id : str
        Identifier for logging.
    out_dir : Path
        Working directory for intermediate outputs.
    iterations : int
        Number of optimization steps (default 2000).
    n_gaussians : int
        Number of Gaussians to initialize (default 2000).
    feat_dim : int
        Per-Gaussian feature vector dimensionality (default 16).
    train_render_size : int
        Resolution at which projections are rendered during training.
    n_z_samples : int
        Number of depth samples per ray for line integration.
    lr : float
        Learning rate for Adam optimizer.

    Returns
    -------
    vol : np.ndarray, shape (V, V, V), dtype float32
        Reconstructed 3D volume.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- unpack data ----
    train = data["train"]
    angles_rad = np.asarray(train["angles"], dtype=np.float32)
    projs = np.asarray(train["projections"], dtype=np.float32)  # (N, H, W)
    vol_size = int(data["nVoxel"][0])

    # ---- downsample projections to training resolution ----
    from skimage.transform import resize
    projs_low = np.stack([
        resize(p, (train_render_size, train_render_size), preserve_range=True)
        for p in projs
    ]).astype(np.float32)
    projs_t = torch.tensor(projs_low, device=device, dtype=torch.float32)
    angles_t = torch.tensor(angles_rad, device=device, dtype=torch.float32)
    n_projs = projs_t.shape[0]

    # ---- initialize Gaussian positions ----
    # Try SIRT-based initialization: compute a coarse SIRT volume and place
    # Gaussians at high-intensity peaks. Fall back to uniform grid otherwise.
    init_positions = None
    try:
        init_positions = _sirt_peak_init(projs, angles_rad, n_gaussians,
                                          vol_size)
    except Exception:
        pass

    if init_positions is None:
        # Uniform grid initialization (like DIF-Gaussian's gs_res grid)
        gs_res = max(int(round(n_gaussians ** (1.0 / 3))), 4)
        n_gaussians = gs_res ** 3
        grid = np.mgrid[:gs_res, :gs_res, :gs_res].reshape(3, -1).T
        init_positions = (grid / (gs_res - 1) - 0.5) * 2.0  # [-1, 1]^3
        init_positions = init_positions.astype(np.float32)

    n_gaussians = init_positions.shape[0]

    # ---- build model ----
    model = DIFGaussianTTO(
        n_gaussians=n_gaussians,
        feat_dim=feat_dim,
        hidden_dims=(64, 32),
        init_positions=init_positions,
    ).to(device)

    # Separate lr for positional parameters vs decoder
    opt = torch.optim.Adam([
        {"params": [model.gs_centers, model.gs_log_s, model.gs_quat,
                     model.gs_feats, model.log_scale], "lr": lr},
        {"params": model.decoder.parameters(), "lr": lr * 2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations,
                                                            eta_min=lr * 0.01)

    # ---- training loop ----
    print(f"  [dif_gaussian_tto] fitting {sample_id}: {iterations} iters, "
          f"K={n_gaussians} gaussians, feat_dim={feat_dim}, "
          f"render={train_render_size}, z_samples={n_z_samples}")

    for step in range(iterations):
        i = int(torch.randint(0, n_projs, (1,)).item())
        angle = float(angles_t[i].item())
        pred = _render_projection(model, train_render_size, angle,
                                   n_z_samples, device)
        loss = F.mse_loss(pred, projs_t[i])

        opt.zero_grad()
        loss.backward()
        # Gradient clipping (matches DIF-Gaussian train.py)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (step + 1) % 500 == 0:
            print(f"    step {step + 1}/{iterations} "
                  f"loss={loss.item():.6f} "
                  f"scale={float(model.log_scale.exp().item()):.4f}")

    # ---- extract dense volume ----
    with torch.no_grad():
        vol = model.forward_volume(vol_size).cpu().numpy().astype(np.float32)

    # Intensity calibration: match mean projection integral
    scale = float(projs.mean()) * vol_size / max(vol.mean(), 1e-8)
    vol *= scale

    # Light Gaussian smoothing for denoising (same as run_neat.py)
    try:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=0.5).astype(np.float32)
    except Exception:
        pass

    return vol


# ---------------------------------------------------------------------------
# SIRT-based Gaussian initialization
# ---------------------------------------------------------------------------

def _sirt_peak_init(projs: np.ndarray, angles_rad: np.ndarray,
                    n_gaussians: int, vol_size: int) -> np.ndarray | None:
    """Compute a coarse SIRT reconstruction and place Gaussians at the
    top-intensity voxel locations.

    Returns (K, 3) float32 positions in [-1, 1]^3, or None on failure.
    """
    try:
        import astra
    except ImportError:
        return None

    # Use a small volume for speed
    coarse_size = min(vol_size, 64)
    from skimage.transform import resize
    projs_coarse = np.stack([
        resize(p, (coarse_size, coarse_size), preserve_range=True)
        for p in projs
    ]).astype(np.float32)

    n_proj = projs_coarse.shape[0]
    # Parallel beam 3D geometry
    vol_geom = astra.create_vol_geom(coarse_size, coarse_size, coarse_size)
    angles = angles_rad.astype(np.float64)
    proj_geom = astra.create_proj_geom(
        "parallel3d", 1.0, 1.0, coarse_size, coarse_size, angles
    )

    # SIRT
    sinogram = projs_coarse.transpose(1, 0, 2)  # (det_row, n_angles, det_col)
    sino_id = astra.data3d.create("-sino", proj_geom, sinogram)
    rec_id = astra.data3d.create("-vol", vol_geom)
    alg_cfg = astra.astra_dict("SIRT3D_CUDA")
    alg_cfg["ReconstructionDataId"] = rec_id
    alg_cfg["ProjectionDataId"] = sino_id
    alg_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(alg_id, 50)
    vol = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(rec_id)

    vol = np.clip(vol, 0, None)

    # Pick top-K voxels by intensity
    flat = vol.ravel()
    K = min(n_gaussians, flat.size)
    top_idx = np.argpartition(flat, -K)[-K:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

    iz, iy, ix = np.unravel_index(top_idx, vol.shape)
    # Map voxel indices to [-1, 1]
    positions = np.stack([
        ix / (coarse_size - 1) * 2 - 1,
        iy / (coarse_size - 1) * 2 - 1,
        iz / (coarse_size - 1) * 2 - 1,
    ], axis=1).astype(np.float32)

    return positions
