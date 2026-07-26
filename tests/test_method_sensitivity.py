"""Tests for the method-level extraction sensitivity sweep.

Reviewer-facing requirement (NeurIPS 3869, AC + wpKw + j8nP): the
SSIM-vs-recall ranking inversion must be shown to persist across
extraction and matching hyperparameters, evaluated on *method
reconstructions* rather than reference volumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analysis.extraction_sensitivity import MODES, NMS_RADIUS_GRID, TAU_GRID, THRESHOLD_GRID  # noqa: E402
from analysis.method_sensitivity import summarize_rankings, sweep_volume  # noqa: E402


def _blob_volume(sigma_vox: float) -> tuple[np.ndarray, np.ndarray]:
    """Two blobs of controllable sharpness on a 32-voxel grid (cell 32 A)."""
    grid = np.arange(32, dtype=float)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    centers = np.array([[10.0, 16.0, 16.0], [22.0, 16.0, 16.0]])
    for cx, cy, cz in centers:
        vol += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
                      / (2.0 * sigma_vox ** 2)).astype(np.float32)
    atoms_gt = np.column_stack([centers - 16.0, np.full(len(centers), 29.0)])
    return vol, atoms_gt


def test_sweep_volume_emits_full_grid():
    vol, atoms_gt = _blob_volume(sigma_vox=1.0)
    rows = list(
        sweep_volume(
            vol,
            atoms_gt,
            cell_size_angstrom=32.0,
            sample_id="syn-sharp",
            method="syn_sharp",
        )
    )
    expected = len(MODES) * len(TAU_GRID) * len(NMS_RADIUS_GRID) * len(THRESHOLD_GRID)
    assert len(rows) == expected
    for row in rows:
        assert row["method"] == "syn_sharp"
        assert set(row["metrics"]) >= {"recall", "precision", "f1", "rmsd", "n_pred"}
    # A sharp volume must be perfectly recovered under the paper-default
    # setting (oracle_count, tau=0.5, nms=2.0).
    default = [
        r
        for r in rows
        if r["mode"] == "oracle_count"
        and r["tau_angstrom"] == 0.5
        and r["nms_radius_angstrom"] == 2.0
        and r["threshold_rel"] == 0.05
    ]
    assert default and default[0]["metrics"]["recall"] == 1.0


def test_summarize_rankings_reports_persistent_ordering():
    vol_sharp, atoms_gt = _blob_volume(sigma_vox=1.0)
    rows = list(
        sweep_volume(vol_sharp, atoms_gt, cell_size_angstrom=32.0,
                     sample_id="s0", method="sharp")
    )
    # Blurred method: single merged blob -> at most one recoverable atom.
    vol_blur, _ = _blob_volume(sigma_vox=6.0)
    rows += list(
        sweep_volume(vol_blur, atoms_gt, cell_size_angstrom=32.0,
                     sample_id="s0", method="blurred")
    )
    summary = summarize_rankings(rows)
    settings = summary["settings"]
    expected = len(MODES) * len(TAU_GRID) * len(NMS_RADIUS_GRID) * len(THRESHOLD_GRID)
    assert len(settings) == expected
    for setting in settings:
        assert set(setting["mean_recall"]) == {"sharp", "blurred"}
    # The sharp method must dominate in every single setting: the ordering
    # (not just the default-setting ordering) is what the paper claims.
    n_dominant = sum(
        1
        for s in settings
        if s["mean_recall"]["sharp"] > s["mean_recall"]["blurred"]
    )
    assert summary["n_settings_ordering_holds"]["sharp>blurred"] == n_dominant
    assert n_dominant == expected
