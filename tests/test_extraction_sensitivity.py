"""Tests for atom-extraction sensitivity sweeps.

The production script is intentionally importable so the paper analysis and
unit tests exercise the same peak-detection and matching code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analysis.extraction_sensitivity import (  # noqa: E402
    evaluate_extraction_setting,
    extract_atoms_from_volume,
)


def _two_atom_volume() -> tuple[np.ndarray, np.ndarray]:
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    peaks = np.array([[10, 16, 16], [22, 16, 16]], dtype=float)
    for x, y, z in peaks.astype(int):
        vol[x, y, z] = 1.0
        vol[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] += 0.18
        vol[x, y, z] = 1.0
    atoms_gt = peaks / 32.0 * 32.0 - 16.0
    atoms_gt = np.column_stack([atoms_gt, np.full(len(atoms_gt), 29)])
    return vol, atoms_gt


def test_matching_radius_increase_does_not_reduce_recall():
    vol, atoms_gt = _two_atom_volume()
    small = evaluate_extraction_setting(
        vol, atoms_gt, cell_size_angstrom=32.0,
        tau_angstrom=0.3, nms_radius_angstrom=1.5,
        threshold_rel=0.05, mode="blind_fixed",
    )
    large = evaluate_extraction_setting(
        vol, atoms_gt, cell_size_angstrom=32.0,
        tau_angstrom=1.0, nms_radius_angstrom=1.5,
        threshold_rel=0.05, mode="blind_fixed",
    )
    assert large["recall"] >= small["recall"]


def test_blind_and_oracle_modes_emit_core_metrics():
    vol, atoms_gt = _two_atom_volume()
    for mode in ("blind_fixed", "oracle_count"):
        metrics = evaluate_extraction_setting(
            vol, atoms_gt, cell_size_angstrom=32.0,
            tau_angstrom=0.5, nms_radius_angstrom=1.5,
            threshold_rel=0.05, mode=mode,
        )
        assert {"recall", "precision", "f1", "rmsd", "n_pred", "n_gt"} <= metrics.keys()
        assert metrics["n_gt"] == 2
        assert 0.0 <= metrics["f1"] <= 1.0


def test_nms_radius_increase_does_not_increase_prediction_count():
    vol, _ = _two_atom_volume()
    close = extract_atoms_from_volume(
        vol, cell_size_angstrom=32.0,
        nms_radius_angstrom=1.0, threshold_rel=0.05,
        mode="blind_fixed", num_expected=None,
    )
    wide = extract_atoms_from_volume(
        vol, cell_size_angstrom=32.0,
        nms_radius_angstrom=4.0, threshold_rel=0.05,
        mode="blind_fixed", num_expected=None,
    )
    assert len(wide) <= len(close)


class _ExplodingExpectedCount:
    def __int__(self):  # pragma: no cover - only called on regression
        raise AssertionError("blind mode must not inspect expected atom count")

    def __le__(self, other):  # pragma: no cover - only called on regression
        raise AssertionError("blind mode must not inspect expected atom count")

    def __mul__(self, other):  # pragma: no cover - only called on regression
        raise AssertionError("blind mode must not inspect expected atom count")


def test_blind_mode_does_not_use_expected_count():
    vol, _ = _two_atom_volume()
    peaks = extract_atoms_from_volume(
        vol, cell_size_angstrom=32.0,
        nms_radius_angstrom=1.5, threshold_rel=0.05,
        mode="blind_fixed", num_expected=_ExplodingExpectedCount(),
    )
    assert len(peaks) == 2


def test_oracle_mode_requires_expected_count():
    vol, _ = _two_atom_volume()
    with pytest.raises(ValueError, match="num_expected"):
        extract_atoms_from_volume(
            vol, cell_size_angstrom=32.0,
            nms_radius_angstrom=1.5, threshold_rel=0.05,
            mode="oracle_count", num_expected=None,
        )
