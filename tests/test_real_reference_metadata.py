"""Tests for real-tier reference metadata used in the paper."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_paper_assets import collect_real_reference_metadata  # noqa: E402


def test_real_tier_reference_categories_are_exhaustive():
    rows = collect_real_reference_metadata()
    counts = Counter(r["reference_type"] for r in rows)
    assert len(rows) == 26
    assert counts == {
        "true_reference": 22,
        "pseudo_peaks": 1,
        "no_atom_list": 3,
    }


def test_true_reference_atom_count_range_matches_filesystem():
    rows = collect_real_reference_metadata()
    true_counts = [
        r["atom_count"] for r in rows
        if r["reference_type"] == "true_reference"
    ]
    assert min(true_counts) == 1534
    assert max(true_counts) == 22382
