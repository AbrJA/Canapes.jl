# Validation

This directory contains optional reference-based validation scripts for Gideon.jl.
They are **not part of the test suite** — they compare Gideon against the R
`rsparse` package and the Python `implicit`/scikit-learn/scipy implementations to
answer "is my implementation numerically equivalent to a known-good reference?".

By default, R fixtures are stored in `/tmp/gideon_fixtures`
(override with `GIDEON_R_FIXTURE_DIR`); Python fixtures in
`/tmp/gideon_fixtures/python` (override with `GIDEON_PY_FIXTURE_DIR`).

## Quick Start

Run everything (generate fixtures + compare against both references):

```bash
julia --project=. validation/run.jl --all
```

Run only one reference:

```bash
julia --project=. validation/run.jl --r
julia --project=. validation/run.jl --python
```

Generate fixtures only:

```bash
julia --project=. validation/run.jl --prepare
```

`--prepare` implies both fixture sets unless `--r` / `--python` is given.
The runner exits with a non-zero status if any requested step fails or cannot
run (missing `uvr`/`Rscript`, `python3`, or Python `implicit`), and prints a summary
table of every step.

## Structure

- **`run.jl`** — single entrypoint: prepares fixtures and runs each comparison
  in its own subprocess (exit codes propagate; one failure does not stop the rest)
- **`validate_r.jl`** — compares against R `rsparse`
- **`validate_py.jl`** — compares against Python `implicit` / sklearn / scipy
- **`common.jl`** — shared helpers (fixture readers, correlation, top-k overlap,
  metric parity) and centralized thresholds
- **`fixtures_r.R`** — R fixture generator
- **`fixtures_py.py`** — Python fixture generator

## What Is Compared

### R (rsparse)

| Model | Check | Criterion |
|---|---|---|
| WMF (Cholesky) | converged loss, warm-start loss | ≤ 1.05× R |
| WMF (CG) | converged loss | ≤ 1.05× R |
| FTRL | weight + prediction correlation | ≥ 0.9995 |
| FM (XOR) | solution agreement across 5 seeds | ≥ 4/5 agree |
| FM (sparse high-dim) | independent dense-reference forward pass + held-out recovery of a known rank-2 latent interaction + prediction agreement with R (rsparse FM init is nondeterministic, so gate with margin) | rel. err < 1e-3, cor ≥ 0.95, cor(jl,R) ≥ 0.95 |
| GloVe | final cost (same ½·Σ f·diff² convention as rsparse) | ≤ R × 1.15 |
| SoftImpute / SoftSVD | singular values, Frobenius norm, reconstruction correlation | relative thresholds |
| Ranking metrics | AP@k / NDCG@k | exact (atol 1e-6) |

### Python (implicit / sklearn / scipy)

| Model | Check | Criterion (default) |
|---|---|---|
| WMF-Cholesky vs ALS | score correlation + top-k overlap | ≥ 0.45 / 0.20 |
| BPR | score correlation + top-k overlap + NDCG/Recall@10 parity | ≥ 0.20 / 0.10, Δ ≤ 0.05 / 0.07 |
| IALS | same as BPR | ≥ 0.35 / 0.15, Δ ≤ 0.06 / 0.06 |
| EALS | same as BPR | ≥ 0.15 / 0.10, Δ ≤ 0.06 / 0.06 |
| LogisticMF | correlation + overlap (meaningful bounds, was -1.0/0.08); NDCG/Recall parity diagnostic by design | ≥ 0.40 / 0.40 |
| EASE | relative Frobenius error of the closed-form B matrix | ≤ 1e-6 (exact) |
| SLIM | weight-matrix correlation + metric parity (needs sklearn) | ≥ 0.6, Δ ≤ 0.08 |
| SoftImpute | reconstruction correlation + singular-value error (diagnostic by default) | ≥ 0.75, ≤ 0.40 |
| PureSVD | singular values + reconstruction vs scipy `svds` | rel. err < 0.05, cor ≥ 0.99 |
| ItemKNN | W / score correlation + top-k overlap | ≥ 0.95 / 0.95 / 0.70 |
| ADMMSLIM | W Frobenius error + W/score correlation + overlap | < 0.05, ≥ 0.99 / 0.99 / 0.85 |

All thresholds can be overridden via `GIDEON_PY_*` environment variables
(see `validate_py.jl` for the full list). Diagnostic-only comparisons are
enforced by setting `GIDEON_PY_LMF_STRICT=1` or `GIDEON_PY_SOFT_STRICT=1`.

## Requirements

- R with the `rsparse` package for `--r`. Prefer [uvr](https://github.com/astral-sh/uvr)
  (project R environment defined in `uvr.toml`, run as `uvr run validation/fixtures_r.R`),
  or install R yourself so `Rscript` is on `PATH`. Fixture generation takes 2–5 minutes.
- Python 3 with `numpy`, `scipy`, and `implicit` for `--python` (e.g. via
  `uv run` or the repo's `.venv`); `scikit-learn` for the SLIM and ItemKNN fixtures only.

## Behavior on Missing Pieces

- **Missing core fixtures** (e.g. never prepared): the comparison scripts fail
  loudly with a non-zero exit code — a validation run never silently passes
  with zero assertions.
- **Optional fixtures** (FM, FTRL, GloVe, SLIM, metrics, SoftImpute, PureSVD,
  ItemKNN, ADMMSLIM — generated only when the underlying R/Python model is
  available): skipped with an explicit `@info` message.
- **Missing tooling** (`uvr`/`Rscript`, `python3`, `implicit`): the runner marks the
  affected step as skipped/failed and exits non-zero.

## Notes

- Fixtures are **NOT** stored in git — generate locally when validating
- Each comparison runs in its own Julia subprocess, so a failure in R validation
  does not prevent Python validation from running
- `validation/` is not wired into CI; run it before releases and after any
  change to an algorithm's math or training loop
