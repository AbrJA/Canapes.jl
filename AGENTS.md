# AGENTS.md — Canapes.jl developer notes

Guidance for AI coding agents and contributors working on this repository.

## Project identity & licensing

- **Canapes.jl** — pure-Julia sparse statistical learning / recommender systems,
  MIT-licensed.
- Every algorithm is an **independent Julia implementation of the paper cited in
  its docstring**. R's `rsparse` (GPL) and Python's `implicit` /
  `scikit-surprise` / `scikit-learn` / `scipy` are used **only as numerical
  references in `validation/`** — no source code is derived from them. When
  writing comments, say "validated against the reference", never "matches
  <project>'s source" or "port of <project>".
- LICENSE names the author; keep it current.

## Commands

```bash
# Full test suite (threads required — tests exercise parallelism)
julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'

# Fast suite (skips Aqua/JET, docs validation, GPU) for quick iteration
TEST_SUITE=fast julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'

# Reference validation vs R (rsparse) / Python (implicit, sklearn, scipy) /
# Surprise (scikit-surprise) / MovieLens-1M scale
julia --project=. validation/run.jl --all        # R + Python + Surprise
julia --project=. validation/run.jl --r           # R comparison only
julia --project=. validation/run.jl --python      # Python comparison only
julia --project=. validation/run.jl --surprise    # explicit-rating parity
julia --project=. validation/run.jl --ml1m        # MovieLens-1M vs implicit
# Env: PYTHON / CANAPES_SURPRISE_PYTHON / CANAPES_ML1M_PYTHON point at the
# respective python interpreter with the needed package installed.

# Performance harness (appends JSONL records to benchmark/logs/results.jsonl)
julia --project=. --threads=8 benchmark/run.jl
julia --project=. benchmark/compare.jl --strict   # diff runs across git SHAs
```

The benchmark measures `fit!` and `recommend` time + `@allocated` bytes at three
fixed scales. **Validate any performance change with it.**

## Threading conventions

- Use the shared helpers in `src/utils.jl`:
  - `_thread_chunk_bounds(chunk, n, nt)` — contiguous range for a chunk.
  - `_thread_buffers(f, nt)` — per-chunk work buffers.
- Canonical pattern: `Threads.@threads for chunk in 1:nt` with per-chunk buffers
  indexed by `chunk`. **Never** index buffers by `Threads.threadid()` (can exceed
  `nthreads()` with multiple thread pools); **never** use `Threads.maxthreadid()`.
- Hot loops that mutate shared state must be partitioned by ownership (each
  thread writes only disjoint slices/words) or be race-free by construction.

## Determinism conventions

- Training kernels use `@simd` **reductions** over rank-k dot products (NOT
  strict-serial `muladd` chains and NOT `@fastmath`): reproducible per
  environment, NaN/Inf-correct.
- **GloVe** and **BPR** are Hogwild (lock-free single-pass SGD, documented
  non-deterministic): results may differ across thread counts and runs; a single
  thread with a fixed rng reproduces runs.
- Bit-identity across SIMD widths/BLAS builds is not promised; `save_model` /
  `load_model` move models across environments.

## Pitfalls

- **`Threads.@threads` closure capture**: a variable reassigned *after* being
  captured gets boxed (`Core.Box`, typed `Any`) and type-instabilizes the loop.
  Keep captured variables single-assignment.
- **WMF/IALS NaN at λ=0**: singular gramians (zero-rating items) must fall back
  to a regularized diagonal + retry, zeroing the solve if still singular. Keep
  this guard when touching the Cholesky paths.
- **WMF `score` vs pairwise `score` differ by design** (fold-in via `transform`).
  Don't assert pairwise == full for WMF.
- **GloVe ½-convention**: the loss is `½ Σ f(x)·diff²`; the gradient carries no
  factor of 2 (paper formulation). Keep it so `lr` and `loss_history` remain
  comparable across implementations.
- **LogisticMF negative sampling**: sample `min(n_items, seen·n_neg)` negatives
  from the global observation pool, matching `implicit`'s lmf.pyx distribution
  (a negative may occasionally be an already-seen item).
- **GPU ext**: `fit_gpu!` supports only `CholeskySolver` / `NonNegativeSolver`
  (WMF) and `CholeskySolver` (IALS) — CG throws `ArgumentError`. GPU tests run
  only when CUDA is present.
- **ADMMSLIM**: training is dense (n×n Gram joint solve); fitted `W` is stored
  sparse. `recommend` dispatches adaptively via `_use_sparse_score_path`.
- **IALS requires non-negative ratings** (confidence uses `sqrt(α·x)`); negative
  input throws. Signed feedback goes through the explicit families.
- **JET abstract-analysis artifact**: `zeros(T, n, n)` under JET's free-typevar
  analysis widens to a union with `Array{Float64,3}`; use `fill(zero(T), n, n)`.
- **Column-major contiguity**: when optimizing dense scoring loops, remember
  Julia matrices are column-major — iterate the *first* index for contiguity.

## Structure

- `src/algorithms/` — one file per model (`wrmf.jl`, `ials.jl`, …); the explicit
  rating predictors are `baselineonly.jl` / `slopeone.jl` / `pearsonknn.jl`.
- `src/experimental/` — demoted models (LogisticMF, PMF) behind the
  `Canapes.Experimental` module; not root-exported.
- `validation/` — reference-parity harness (R, Python, Surprise, ml-1m).
- `benchmark/` — tracked perf harness.
- `usage/` — data-loading + benchmark scripts for MovieLens sets.
