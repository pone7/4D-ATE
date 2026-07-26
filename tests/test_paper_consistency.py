"""Reviewer-facing consistency checks for the paper draft."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "aet_bench.tex"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"


def test_paper_has_no_submission_placeholders_or_known_typos():
    text = PAPER.read_text()
    forbidden = [
        r"\\missingresult",
        r"\\submissiontodo",
        "Field to finalize",
        "Conlusion",
        "infomation",
        "held-out test",
    ]
    for pattern in forbidden:
        assert pattern not in text


def test_paper_includes_generated_reviewer_tables_and_sensitivity_figure():
    text = PAPER.read_text()
    assert r"\input{tables/tab_sensitivity}" in text
    assert r"\input{tables/tab_baseline_coverage}" in text
    assert r"\input{tables/tab_real_reference_metadata}" in text
    assert r"figures/fig_threshold_sweep.pdf" in text
    assert (TABLES / "tab_sensitivity.tex").exists()
    assert (TABLES / "tab_baseline_coverage.tex").exists()
    assert (TABLES / "tab_real_reference_metadata.tex").exists()
    assert (FIGURES / "fig_threshold_sweep.pdf").exists()


def test_main_table_real_x2_recall_matches_generated_real_table():
    main = (TABLES / "tab_main_results.tex").read_text()
    real = (TABLES / "tab_real_t1.tex").read_text()
    main_match = re.search(r"Real X\$\^2\$-G recall\s*&\s*([0-9.]+)\\%", main)
    if main_match is None:
        # The main table can be omitted, but must not carry stale Real X2 text.
        assert "Real X$^2$-G recall" not in main
        return
    real_match = re.search(r"X\$\^2\$-G\s*&\s*26\s*&(?:[^&]*&){2}\s*[^&]*&\s*[^&]*?([0-9.]+)\\%", real)
    assert real_match is not None
    assert float(main_match.group(1)) == float(real_match.group(1))


def test_multislice_rmsd_text_not_stale_literal():
    text = PAPER.read_text()
    assert "2.2~\\AA" not in text
    assert "2.2 \\AA" not in text
