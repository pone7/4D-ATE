# AET-Bench — anonymized review bundle

This bundle accompanies the authors' response for submission 3869. It contains
the evaluation code reviewers asked to inspect, plus the two revised teaser
figures.

## Figures

**Figure 1 — pixel-accurate is not atom-accurate.** A reconstruction can be
pixel-accurate while the atom model extracted from it is wrong: atoms can be
missed, merged, hallucinated, or given the wrong species.

![Figure 1](figures/fig1_concept.png)

**Figure 2 — the AET-Bench realism ladder.** Six tiers, each adding one source
of difficulty: localization, chemistry, trajectories, scattering physics,
experimental reference agreement, and dynamic tracking. Renderings are produced
from the benchmark data by `scripts/generate_paper_assets.py`; the 4D tile shows
real per-atom trajectories across the ten phases, zoomed onto one neighbourhood
so the true sub-nanometre displacements are legible.

![Figure 2](figures/fig2_realism_ladder.png)

## Layout

```
scripts/          evaluation harness, method wrappers, metrics, analyses
  run_benchmark.py            unified entry point (METHOD_REGISTRY)
  evaluate_baselines.py       shared volume -> atom extraction + Hungarian matching
  metrics/                    FSC, trajectory / identity-persistence metrics
  analysis/                   extraction sensitivity, ceiling, native extraction
  generate_paper_assets.py    regenerates every paper table and figure
tests/            unit tests, incl. the extraction-ceiling regression guard
configs/          method configuration files
figures/          the two teaser figures above (PNG + PDF)
```

## Quickstart

```bash
conda env create -f environment.yml && conda activate aet-bench
python scripts/run_benchmark.py --subset mono --sample Cu-0165-00 --method wbp --task T1
pytest -q tests/
```

## Note on paths

Absolute paths in this bundle were rewritten to a neutral prefix
(`/data/aet-bench`) for anonymization. To run against a different location,
adjust that prefix, e.g.:

```bash
grep -rl '/data/aet-bench' scripts tests | xargs sed -i "s|/data/aet-bench|$PWD|g"
```

Benchmark data is distributed separately (dataset record cited in the paper);
this bundle is the code only.
