# Repository Guidelines

## Project Structure & Module Organization

AET-Bench is a Python benchmark for atomic electron tomography. Core execution lives in `scripts/`: `run_benchmark.py` is the unified CLI, `generate_paper_assets.py` regenerates tables/figures, `metrics/` contains FSC and atom-matching metrics, `baselines/` wraps reconstruction methods, and `analysis/` holds paper-specific analyses. Tests live in `tests/`. The NeurIPS paper and generated assets are under `paper/`, with figures in `paper/figures/` and LaTeX tables in `paper/tables/`. Benchmark data and result logs are under `data/aet_bench/`; large outputs and checkpoints go under `output/` and should not be committed.

## Build, Test, and Development Commands

Create the environment:

```bash
conda env create -f environment.yml
conda activate aet-bench
```

Run one benchmark job:

```bash
python scripts/run_benchmark.py --subset mono --sample Cu-0165-00 --method wbp --task T1
```

Regenerate paper assets and audit results:

```bash
python scripts/validate_results.py data/aet_bench/results.jsonl
python scripts/generate_paper_assets.py --results data/aet_bench/results.jsonl
cd paper && /usr/bin/pdflatex -interaction=nonstopmode aet_bench.tex
```

GPU-heavy supplemental runs use `scripts/start_gpu_extras.sh` or targeted scripts such as `scripts/run_gpu_critical.py`.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, descriptive snake_case names, and type hints for new public helpers. Keep scripts CLI-first with `argparse`; use environment variables only for worker sharding or GPU binding (`GPU_SLICE`, `N_GPUS`, `CUDA_VISIBLE_DEVICES`). Method keys in registries should be lowercase, stable identifiers such as `astra_cgls` or `x2gaussian`.

## Testing Guidelines

Tests use `pytest`. Run focused tests before committing:

```bash
pytest -q tests/test_trajectory_metrics.py tests/test_run_benchmark_smoke.py
```

Add tests for new metrics, result-schema changes, and benchmark dispatch behavior. Name test files `test_*.py` and keep smoke tests small enough for CPU-only environments unless explicitly marked/skipped.

## Commit & Pull Request Guidelines

Recent commits use short scoped subjects, for example `paper: ...`, `data: ...`, or `data+paper: ...`. Follow that style and keep generated-result churn separate from code changes when possible. PRs should state the affected subset/method/task, include exact reproduction commands, note GPU requirements, and mention whether paper tables or figures were regenerated.

## Data & Configuration Notes

Do not commit large generated outputs, raw datasets, or temporary logs. Preserve result provenance: validate JSONL logs before using them in the paper, and keep missing or pilot experiments clearly marked rather than inferring unsupported numbers.
