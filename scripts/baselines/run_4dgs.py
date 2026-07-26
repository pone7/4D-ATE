"""4D-GS fallback baseline for AET-Bench.

4D-Gaussians (Wu et al., CVPR 2024) was designed for multi-view RGB dynamic
scene reconstruction with COLMAP poses. Adapting its rasterizer to HAADF
Z-contrast without rebuilding the custom CUDA kernel is non-trivial; the
plan explicitly allows a documented fallback when the upstream surgery
cannot be completed in the time budget.

Fallback behavior:
    We run X²-Gaussian's 4D mode with a loosened training regime intended to
    mimic 4D-GS's "HexPlane + no TV loss" parameter family:
        - lambda_tv = 0
        - looser densification (larger grad threshold)
        - otherwise the same HexPlane dx-only deformation.

This is documented as a limitation in the benchmark paper: both methods
share the 4D Gaussian Splatting family; the comparison measures two
different regimes within that family rather than two fully independent
implementations. A FALLBACK_USED constant is exposed so the table
generator can mark the row if desired.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.run_x2gaussian import run_x2gaussian, DEFORM_4D_CFG

FALLBACK_USED = True


def run_4dgs(data: dict, sample_id: str, work_root: Path,
              iterations: int = 8000):
    # Variant config: no TV loss, looser densification — matches 4D-GS's
    # closest parameter regime in our codebase.
    _cfg_overrides = {
        "lambda_tv": 0.0,
        "densify_grad_threshold": 1e-4,
    }
    # The x2g wrapper accepts per-run overrides only through its base_cfg
    # dict, so we temporarily patch DEFORM_4D_CFG. Restore after the call.
    saved = {k: DEFORM_4D_CFG.get(k) for k in _cfg_overrides}
    try:
        DEFORM_4D_CFG.update(_cfg_overrides)
        return run_x2gaussian(
            data, sample_id, Path(work_root),
            iterations=iterations, mode="4d",
        )
    finally:
        for k, v in saved.items():
            if v is None:
                DEFORM_4D_CFG.pop(k, None)
            else:
                DEFORM_4D_CFG[k] = v
