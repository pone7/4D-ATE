# Contributing to AET-Bench

Thank you for helping extend the benchmark.  This guide walks through the
two most common contributions: (1) adding a new reconstruction method and
(2) adding a new data tier.

## Adding a New Method

Every method lives behind a single callable:

```python
def my_method(data: dict, sample_id: str) -> np.ndarray:
    """Return a reconstructed volume of shape (D, D, D), float32."""
    projections = data["train"]["projections"]   # (n_angles, H, W)
    angles       = data["train"]["angles"]        # (n_angles,) radians
    image_gt     = data["image"]                  # for shape reference
    # … your reconstruction …
    return vol
```

Register it in `scripts/run_benchmark.py`:

```python
METHOD_REGISTRY = {
    ...,
    "my_method": my_method,
}
```

Once registered, every subset and task is available without further code:

```bash
python scripts/run_benchmark.py --subset mono --method my_method --task T1
python scripts/run_benchmark.py --subset 4d   --method my_method --task T7
```

Results are appended to `data/aet_bench/results.jsonl` in a schema-stable
format.  Regenerate tables and figures with:

```bash
python scripts/generate_paper_assets.py
```

### Implementation Tiers

In Table 2 of the paper, each method carries an implementation tier:

- **N (native)** — the author's own reference implementation is used.
- **F (fallback)** — a faithful but independent re-implementation.  Use
  this label if you swap in an algorithmic stand-in (e.g. IsoNet(GP)
  replacing IsoNet's TF1 pipeline with a Gaussian-prior regulariser).
- **R (reimplementation)** — a port that differs from the original only
  in engineering details (language, framework, CUDA kernels).

State the tier in your pull request.

### Compute & Memory

- Methods that exceed 48 GB VRAM on any single sample will not run on
  the standard evaluation node (H200 × 4).  Chunk or cache accordingly.
- Classical iterative methods should cap default iterations at 100 and
  expose `max_iter` via a config or environment variable.

## Adding a New Data Tier

1. Generate data under `data/aet_bench/<tier>/<sample>/analytical/data.pkl`
   with the schema defined in `paper/aet_bench.tex` Appendix B.
2. Add the tier to `SUBSET_DIRS` in `scripts/run_benchmark.py`.
3. Add a caption string under `write_all_tables` in
   `scripts/generate_paper_assets.py`.

Each sample directory must contain:

- `analytical/data.pkl`   — projections + scanner config (see below)
- `atoms_gt.npy`          — (N, 4) ground-truth `[x, y, z, Z]` in Å
- `metadata.json`         — `element`, `cell_size_angstrom`, `vol_size`, …

For 4D tiers also provide `atoms_gt_4d.npy` of shape `(T, N, 4)`.

## Style

- Python: ruff (configured), type hints encouraged.
- Scripts are CLI-first.  Expose flags via `argparse`, not env vars where
  possible; env vars are fine for worker sharding (`GPU_SLICE`, `N_GPUS`).
- Keep new runtime dependencies out of the top-level environment unless
  they are required to evaluate a paper-level method.

## Reproducing Paper Results

Every paper figure and LaTeX table has a generating script under
`scripts/` or `scripts/analysis/`.  A full round-trip is:

```bash
python scripts/generate_paper_assets.py            # base tables + figures
python scripts/analysis/fourd_mechanism.py          # 4D paradox
python scripts/analysis/dose_analysis.py            # dose tier
python scripts/analysis/ranking_differential.py     # ranking appendix
cd paper && latexmk -pdf aet_bench.tex
```

`git diff paper/tables` should be empty after regeneration if nothing in
the results file changed.
