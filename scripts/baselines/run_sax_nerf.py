"""SAX-NeRF baseline for AET-Bench (reimplementation).

SAX-NeRF (CVPR 2024, "Structure-Aware Sparse-View X-ray 3D
Reconstruction") introduces the Lineformer: a transformer that
performs self-attention on line segments (groups of sample points
along each ray), replacing standard MLP layers with attention blocks.
This structural prior captures inter-sample relationships along
rays, improving reconstruction from sparse views.

Since the original SAX-NeRF codebase requires a custom CUDA hash
encoder (JIT-compiled C++/CUDA kernels), we provide a self-contained
reimplementation that captures the core architectural ideas:

    1. **Frequency positional encoding** as a drop-in for the hash
       encoder (pure PyTorch, no compilation needed).
    2. **Line Attention blocks** faithfully ported from the original
       Lineformer.py -- multi-head self-attention with learnable
       positional bias operating on ray-partitioned sample groups.
    3. **Skip connections** at the midpoint of the network, matching
       the original architecture.
    4. **Parallel-beam ray marching** with stratified sampling and
       density integration for projection rendering.

The reimplementation follows the same training loop pattern as the
existing NeAT wrapper: per-projection MSE loss, Adam optimizer,
short training (2500 iterations) for speed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


# ---------------------------------------------------------------------------
# Frequency Positional Encoding (replaces hash-grid encoder)
# ---------------------------------------------------------------------------

class FreqEncoder(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates.

    Maps (x, y, z) -> (x, y, z, sin(x), cos(x), sin(2x), cos(2x), ...)
    following the standard NeRF positional encoding scheme.
    """

    def __init__(self, input_dim: int = 3, n_freqs: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.n_freqs = n_freqs
        self.output_dim = input_dim + input_dim * n_freqs * 2
        self.freq_bands = 2.0 ** torch.arange(n_freqs).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 3) in [-1, 1] -> (..., output_dim)"""
        freqs = self.freq_bands.to(x.device)
        out = [x]
        for f in freqs:
            out.append(torch.sin(x * f * np.pi))
            out.append(torch.cos(x * f * np.pi))
        return torch.cat(out, dim=-1)


# ---------------------------------------------------------------------------
# Multi-resolution Dense Feature Grid (alternative encoder)
# ---------------------------------------------------------------------------

class FeatureGrid(nn.Module):
    """Two-scale learnable 3D feature volume with trilinear interpolation.

    Provides a multi-resolution spatial encoding without requiring custom
    CUDA kernels. Combines a coarse (low_res^3) and fine (high_res^3)
    grid, each with feat_dim channels.
    """

    def __init__(self, low_res: int = 32, high_res: int = 96,
                 feat_dim: int = 8):
        super().__init__()
        self.low = nn.Parameter(
            torch.zeros(1, feat_dim, low_res, low_res, low_res))
        self.high = nn.Parameter(
            torch.zeros(1, feat_dim, high_res, high_res, high_res))
        nn.init.normal_(self.low, std=0.01)
        nn.init.normal_(self.high, std=0.01)
        self.output_dim = 2 * feat_dim + 3  # concat both grids + raw xyz

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (N, 3) in [-1, 1] -> (N, output_dim)"""
        N = xyz.shape[0]
        # grid_sample expects (N, C, D, H, W) input, (N, D_out, H_out, W_out, 3) grid
        grid = xyz.view(1, N, 1, 1, 3)
        # grid_sample uses (z, y, x) ordering internally
        flow = torch.cat(
            [grid[..., 2:3], grid[..., 1:2], grid[..., 0:1]], dim=-1)
        feat_low = F.grid_sample(
            self.low, flow, align_corners=True, mode="bilinear")
        feat_high = F.grid_sample(
            self.high, flow, align_corners=True, mode="bilinear")
        fl = feat_low.squeeze(-1).squeeze(-1).squeeze(0).T   # (N, feat_dim)
        fh = feat_high.squeeze(-1).squeeze(-1).squeeze(0).T  # (N, feat_dim)
        return torch.cat([fl, fh, xyz], dim=-1)


# ---------------------------------------------------------------------------
# Lineformer: Line-segment Self-Attention (ported from SAX-NeRF)
# ---------------------------------------------------------------------------

class LineAttention(nn.Module):
    """Multi-head self-attention operating on ray-partitioned line segments.

    The key idea from SAX-NeRF: instead of processing each sample point
    independently (as in a standard MLP), group consecutive samples along
    each ray into segments and let them attend to each other. This
    captures structural relationships along the ray (e.g., edges, density
    gradients).
    """

    def __init__(self, dim: int, line_size: int = 16,
                 dim_head: int = 32, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.line_size = line_size

        # Learnable relative positional bias
        self.pos_emb = nn.Parameter(
            torch.zeros(1, heads, line_size, line_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N_ray * N_samples, C) -> (N_ray * N_samples, C)

        Points are partitioned into groups of `line_size` consecutive
        samples. Attention is performed within each group.
        """
        n, c = x.shape
        ls = self.line_size

        # Partition into line segments: (n // ls, ls, c)
        x_seg = x.view(n // ls, ls, c)

        # Project to Q, K, V
        q = self.to_q(x_seg)               # (B, ls, inner_dim)
        k, v = self.to_kv(x_seg).chunk(2, dim=-1)

        # Split heads: (B, ls, inner_dim) -> (B, heads, ls, dim_head)
        def split_heads(t):
            return t.view(t.shape[0], t.shape[1], self.heads,
                          t.shape[2] // self.heads).permute(0, 2, 1, 3)
        q, k, v = map(split_heads, (q, k, v))

        # Scaled dot-product attention with positional bias
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        # Aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Merge heads: (B, heads, ls, dim_head) -> (B, ls, inner_dim) -> (B, ls, C)
        out = out.permute(0, 2, 1, 3).contiguous().view(
            out.shape[0], out.shape[2], -1)
        out = self.to_out(out)

        # Merge back to flat: (n // ls, ls, c) -> (n, c)
        return out.reshape(n, c)


class PreNorm(nn.Module):
    """Layer normalization applied before a sub-layer."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class FFN(nn.Module):
    """Feed-forward network used after attention in each Lineformer block."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * mult, dim * mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LineAttentionBlock(nn.Module):
    """A Lineformer block: PreNorm(LineAttention) + PreNorm(FFN), with
    residual connections."""

    def __init__(self, dim: int, line_size: int = 16,
                 dim_head: int = 32, heads: int = 4,
                 num_blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LineAttention(
                    dim=dim, line_size=line_size,
                    dim_head=dim_head, heads=heads)),
                PreNorm(dim, FFN(dim=dim)),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C) -> (N, C)"""
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ---------------------------------------------------------------------------
# SAX-NeRF Model: Encoder + Lineformer Decoder
# ---------------------------------------------------------------------------

class SAXNeRFModel(nn.Module):
    """Structure-Aware X-ray Neural Radiance Field.

    Architecture (following the original SAX-NeRF paper):
        encoder(xyz) -> features
        Linear(features -> hidden)
        LineAttentionBlock x (num_layers - 2), with skip connection at midpoint
        Linear(hidden -> 1)
        sigmoid activation on output

    The model predicts a scalar density at each 3D coordinate.
    """

    def __init__(self, encoder_type: str = "freq",
                 n_freqs: int = 8,
                 grid_low: int = 32, grid_high: int = 96,
                 num_layers: int = 4, hidden_dim: int = 64,
                 line_size: int = 16, dim_head: int = 16,
                 heads: int = 4, num_blocks: int = 1):
        super().__init__()
        self.line_size = line_size

        # Encoder
        if encoder_type == "grid":
            self.encoder = FeatureGrid(
                low_res=grid_low, high_res=grid_high, feat_dim=8)
        else:
            self.encoder = FreqEncoder(input_dim=3, n_freqs=n_freqs)
        in_dim = self.encoder.output_dim

        # Determine skip layer index (midpoint)
        self.skip_idx = num_layers // 2

        # Build layers: first linear, then attention blocks (with one skip),
        # then output linear
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # Input projection
                self.layers.append(nn.Linear(in_dim, hidden_dim))
                self.activations.append(nn.LeakyReLU())
            elif i == num_layers - 1:
                # Output layer
                self.layers.append(nn.Linear(hidden_dim, 1))
                self.activations.append(nn.Sigmoid())
            elif i == self.skip_idx:
                # Skip connection layer
                self.layers.append(
                    nn.Linear(hidden_dim + in_dim, hidden_dim))
                self.activations.append(nn.LeakyReLU())
            else:
                # Line attention block
                self.layers.append(LineAttentionBlock(
                    dim=hidden_dim, line_size=line_size,
                    dim_head=dim_head, heads=heads,
                    num_blocks=num_blocks))
                self.activations.append(nn.Identity())

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (N, 3) in [-1, 1] -> (N, 1) density."""
        feat = self.encoder(xyz)
        input_feat = feat
        x = feat
        for i, (layer, act) in enumerate(
                zip(self.layers, self.activations)):
            if i == self.skip_idx:
                x = torch.cat([input_feat, x], dim=-1)
            x = layer(x)
            x = act(x)
        return x

    def forward_safe(self, xyz: torch.Tensor) -> torch.Tensor:
        """Forward pass that handles padding for line_size alignment.

        When the total number of points is not divisible by line_size,
        we pad with zeros and strip afterwards. This is used during
        volume querying where point counts are not controlled.
        """
        N = xyz.shape[0]
        ls = self.line_size
        remainder = N % ls
        if remainder != 0:
            pad_n = ls - remainder
            xyz_padded = torch.cat(
                [xyz, torch.zeros(pad_n, 3, device=xyz.device)], dim=0)
        else:
            xyz_padded = xyz
            pad_n = 0

        out = self.forward(xyz_padded)
        if pad_n > 0:
            out = out[:N]
        return out


# ---------------------------------------------------------------------------
# Parallel-beam projection rendering
# ---------------------------------------------------------------------------

def _render_projection_parallel(
    model: SAXNeRFModel,
    render_size: int,
    angle_rad: float,
    n_z: int,
    device: torch.device,
    perturb: bool = True,
) -> torch.Tensor:
    """Render a single parallel-beam projection at the given angle.

    For parallel beam geometry, all rays share the same direction and
    are offset spatially. We rotate the sampling grid by the tilt angle
    and integrate density along the beam direction (z-axis after rotation).

    Args:
        model: SAX-NeRF density model.
        render_size: H=W of the output projection image.
        angle_rad: Tilt angle in radians (rotation about Y axis).
        n_z: Number of integration samples along each ray.
        device: Torch device.
        perturb: Whether to add stratified sampling noise.

    Returns:
        proj: (render_size, render_size) integrated projection.
    """
    rows = torch.linspace(-1.0, 1.0, render_size, device=device)
    cols = torch.linspace(-1.0, 1.0, render_size, device=device)
    zs = torch.linspace(-1.0, 1.0, n_z, device=device)

    if perturb:
        dz = 2.0 / n_z
        zs = zs + (torch.rand_like(zs) - 0.5) * dz

    X, Y, Z = torch.meshgrid(cols, rows, zs, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1)  # (H, W, n_z, 3)

    # Rotation matrix about Y axis
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    Rmat = torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        device=device, dtype=torch.float32)
    pts_rot = pts @ Rmat.T  # (H, W, n_z, 3)

    # Flatten for network, process in sub-batches for memory
    flat = pts_rot.reshape(-1, 3)
    ls = model.line_size

    # We need total points divisible by line_size for the attention blocks.
    # Reshape so that each "ray" (n_z samples) forms one or more line segments.
    # Pad n_z up to next multiple of line_size.
    n_rays = render_size * render_size
    total = n_rays * n_z

    # Process the entire flat tensor, using forward_safe for alignment
    chunk_size = max(ls, 64 * 1024)
    # Round chunk_size to multiple of line_size
    chunk_size = (chunk_size // ls) * ls

    out_parts = []
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        chunk = flat[start:end]
        out_parts.append(model.forward_safe(chunk))
    density = torch.cat(out_parts, dim=0)

    density = density.reshape(render_size, render_size, n_z)
    # Integrate along the ray direction (sum, scaled by step size)
    proj = density.sum(dim=-1) * (2.0 / n_z)
    return proj


# ---------------------------------------------------------------------------
# Dense volume extraction
# ---------------------------------------------------------------------------

def _extract_volume(
    model: SAXNeRFModel,
    vol_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Query the trained model on a regular grid to extract a 3D volume.

    Args:
        model: Trained SAX-NeRF model.
        vol_size: Number of voxels along each axis.
        device: Torch device.

    Returns:
        vol: (vol_size, vol_size, vol_size) density volume.
    """
    coords = torch.linspace(-1.0, 1.0, vol_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    batch = 128 * 1024
    out = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], batch):
            chunk = pts[i:i + batch]
            out.append(model.forward_safe(chunk))
    return torch.cat(out, dim=0).reshape(vol_size, vol_size, vol_size)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sax_nerf(
    data: dict,
    sample_id: str,
    out_dir: Path,
    iterations: int = 2500,
    train_render_size: int = 64,
    n_z_samples: int = 64,
    lr: float = 2e-3,
    encoder_type: str = "freq",
) -> np.ndarray:
    """Train SAX-NeRF (Lineformer) on projections and return a 3D volume.

    This is a self-contained reimplementation of the SAX-NeRF core idea
    (structure-aware line-segment attention) adapted for parallel-beam
    electron tomography.

    Args:
        data: NAF-format data dict with keys:
            - train/val: sub-dicts with "projections" (N, H, W) and
              "angles" (N,) in radians
            - nVoxel: [Nx, Ny, Nz] volume dimensions
            - dVoxel: voxel spacings
        sample_id: Identifier string for logging.
        out_dir: Working directory for intermediate outputs.
        iterations: Number of training iterations.
        train_render_size: Downsample projections to this resolution
            during training for speed.
        n_z_samples: Number of samples along each ray.
        lr: Learning rate for Adam optimizer.
        encoder_type: "freq" for frequency encoding (default, no CUDA
            compilation), or "grid" for multi-resolution dense grid.

    Returns:
        vol: (Nx, Ny, Nz) numpy array, reconstructed 3D volume.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and preprocess data
    # ------------------------------------------------------------------
    train = data["train"]
    angles_rad = np.asarray(train["angles"], dtype=np.float32)
    projs = np.asarray(train["projections"], dtype=np.float32)
    vol_size = int(data["nVoxel"][0])

    # Downsample projections for training speed
    from skimage.transform import resize
    projs_low = np.stack([
        resize(p, (train_render_size, train_render_size),
               preserve_range=True)
        for p in projs
    ]).astype(np.float32)
    projs_low = np.ascontiguousarray(projs_low, dtype=np.float32)
    angles_arr = np.ascontiguousarray(angles_rad, dtype=np.float32)

    projs_t = torch.tensor(projs_low, device=device, dtype=torch.float32)
    angles_t = torch.tensor(angles_arr, device=device, dtype=torch.float32)
    n_projs = projs_t.shape[0]

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    # Choose line_size: must divide n_z_samples evenly
    line_size = _pick_line_size(n_z_samples)

    model = SAXNeRFModel(
        encoder_type=encoder_type,
        n_freqs=8,
        grid_low=32,
        grid_high=min(96, vol_size),
        num_layers=4,
        hidden_dim=64,
        line_size=line_size,
        dim_head=16,
        heads=4,
        num_blocks=1,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=iterations, eta_min=lr * 0.01)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"  [sax-nerf] fitting {sample_id}: {iterations} iters, "
          f"render={train_render_size}, samples={n_z_samples}, "
          f"line_size={line_size}, encoder={encoder_type}")

    for step in range(iterations):
        idx = int(torch.randint(0, n_projs, (1,)).item())
        angle = float(angles_t[idx].item())

        pred = _render_projection_parallel(
            model, train_render_size, angle, n_z_samples,
            device, perturb=(step < iterations * 0.9))

        gt = projs_t[idx]
        loss = F.mse_loss(pred, gt)

        opt.zero_grad()
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        opt.step()
        scheduler.step()

        if (step + 1) % 500 == 0:
            print(f"    step {step + 1}/{iterations} "
                  f"loss={loss.item():.6f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------------
    # Extract volume
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        vol = _extract_volume(model, vol_size, device)
        vol = vol.cpu().numpy().astype(np.float32)

    # Light smoothing to reduce high-frequency noise
    try:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=0.6).astype(np.float32)
    except Exception:
        pass

    # Scale to match projection intensity
    scale = float(projs.mean()) * vol_size / max(float(vol.mean()), 1e-8)
    vol *= scale

    return vol


def _pick_line_size(n_z: int) -> int:
    """Choose a line_size that divides n_z evenly.

    Prefers sizes around 8-16 for a good balance between attention
    receptive field and computational cost.
    """
    preferred = [16, 8, 32, 4, 2]
    for ls in preferred:
        if n_z % ls == 0:
            return ls
    # Fallback: find largest power of 2 that divides n_z
    for p in [16, 8, 4, 2, 1]:
        if n_z % p == 0:
            return p
    return 1
