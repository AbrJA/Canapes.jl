# AGENTS.md — Gideon.jl developer notes

## Commands

```bash
# Full test suite (threads required — tests exercise parallelism)
julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'

# Single algorithm test file
julia --project=. --threads=8 -e 'using Gideon, SparseArrays, Random, Test, LinearAlgebra, Statistics; include("test/test_glove.jl")'

# Performance harness (appends JSONL records to benchmark/logs/results.jsonl)
julia --project=. --threads=8 benchmark/run.jl

# Compare runs across commits
julia --project=. benchmark/compare.jl --strict

# Reference validation vs R (rsparse) / Python (implicit, sklearn, scipy)
# (not part of the test suite; exits non-zero on any requested-but-failed step)
# R fixtures use the project's managed R env (uvr.toml) when `uvr` is installed.
julia --project=. validation/run.jl --all            # prepare fixtures + run both
julia --project=. validation/run.jl --r             # R comparison only
julia --project=. validation/run.jl --python        # Python comparison only
PYTHON=.venv/bin/python julia --project=. validation/run.jl --all   # venv python
```

The benchmark measures `fit!` and `recommend` time + `@allocated` bytes at 3 fixed
scales (hundreds/thousands/millions). **Validate any perf change with it.**

## Threading conventions (Item 3, commit d33cc74)

- Use the shared helpers in `src/utils.jl`:
  - `_thread_chunk_bounds(chunk, n, nt)` — contiguous range for a chunk.
  - `_thread_buffers(f, nt)` — per-chunk work buffers.
- Canonical pattern: `Threads.@threads for chunk in 1:nt` with per-chunk buffers
  indexed by `chunk`. **Never** index buffers by `Threads.threadid()` (can exceed
  `nthreads()` with multiple thread pools) and **never** use
  `Threads.maxthreadid()`.
- Hot loops that mutate shared state must be partitioned by ownership (each
  thread writes only disjoint slices/words) or be race-free by construction.

## Determinism conventions (commits 28da68a, c11747f, b8dab7c, 2026-08-31)

- Training kernels use `@simd` **reductions** over rank-k dot products (NOT
  strict-serial `muladd` chains and NOT `@fastmath`): reproducible per
  environment (same binary → same result), NaN/Inf-correct, and ~8× faster
  than serial FMA chains on the dot kernels (measured; LogisticMF fit dropped
  19-50% end-to-end). Bit-identity across SIMD widths/builds is not claimed —
  BLAS paths already differ across BLAS builds.
- `@simd` is fine on element-wise loops and in loss-monitoring-only code.
- **GloVe** and **BPR** are Hogwild (lock-free single-pass SGD, documented
  non-deterministic): results may differ across thread counts and runs; a
  single thread with a fixed rng reproduces runs. The GloVe epoch was
  switched from a deterministic 3-phase word-ownership scheme (bit-identical
  across thread counts) to Hogwild in 2026-08-31 — measured 1.26x at
  millions scale with identical convergence; determinism tests for GloVe
  were removed accordingly.

## Pitfalls we hit

- **BPR 4.48 GB leak**: a variable reassigned *after* being captured by
  `Threads.@threads` gets boxed (`Core.Box`, typed `Any`) and type-instabilizes
  the whole loop. Keep captured variables single-assignment (use fresh names).
- **WMF/IALS NaN at λ=0**: singular gramians (zero-rating items) must fall back
  to a regularized diagonal + retry, zeroing the solve if still singular
  (commit 7dcc37e). Keep this guard when touching the Cholesky paths.
- **WMF `score` vs pairwise `score` differ by design** (fold-in via `transform`,
  commit 9690e02). Don't assert pairwise == full for WMF.
- **GloVe ½-convention** (2026-08-29): the loss is `½ Σ f(x)·diff²` and the
  gradient carries no factor of 2 — matching rsparse/Stanford C exactly, so
  `lr` semantics and `loss_history` values are comparable to
  rsparse. Keep this convention when touching glove.jl.
- **LogisticMF negative sampling** (2026-08-29): sample
  `min(n_items, seen·n_neg)` negatives like `implicit` (lmf.pyx) — the old
  `min(k, seen·n_neg)` cap silently starved the gradient. Rejection sampling
  on unobserved items is a deliberate improvement; keep it.
- **GPU ext**: `fit_gpu!` supports only `CholeskySolver` / `NonNegativeSolver`
  (WMF) and `CholeskySolver` (IALS) — CG throws `ArgumentError` rather than
  silently switching to Cholesky. GPU entry points share the CPU validators
  (`_require_*`) and are transactional. GPU tests run only when CUDA is
  present (not in CI — CI has no GPU runner).
- **ADMMSLIM**: training is necessarily dense (ADMM joint solve over the
  n×n Gram); the fitted `W` is stored sparse (soft-thresholded exact zeros,
  `dropzeros!(sparse(Z))`). `recommend` dispatches adaptively via
  `_use_sparse_score_path` — the pure sparse path regressed 8× on dense-ish
  W at benchmark scale, so a fixed sparse path is NOT a safe simplification.
- **IALS requires non-negative ratings**: its confidence uses `sqrt(α·x)`
  (Rendle 2021); negative input throws `DomainError`. Signed feedback only
  through the explicit WMF/EASE/SLIM/ADMMSLIM/ItemKNN/SoftImpute families.
- **LogisticMF needs an unobserved item per user**: negative sampling throws
  "no unobserved entity is available" when a user has a fully-observed row;
  property tests skip that degenerate case for LMF.
- **rsparse quirks**: its FM init uses Armadillo's `std::random_device` RNG
  (nondeterministic — gate FM parity with margin, cor ≥ 0.95); its GloVe
  cost records the ½·loss convention; its FM fails to converge on very sparse
  low-coverage problems (upstream, not ours — the agreement gate is skipped
  when rsparse itself does not recover the structure, since Julia's own
  correctness is enforced by the dense-reference and held-out-recovery
  gates).

## Current state

- All roadmap items are complete (see `roadmap.md`). Only the release
  process gates remain before `v1.0.0`.
- Tests: 17430 passing (includes GPU ext tests when CUDA is present;
  includes Aqua + JET, docs-example validation, and 15k randomized
  property tests).
- Reference validation is green end-to-end (`validation/run.jl --all` with
  `PYTHON=.venv/bin/python`): R parity (WMF, FTRL, GloVe, SoftImpute/SVD,
  metrics, FM) and Python parity (ALS, BPR, IALS, EALS, LMF, EASE, SLIM,
  PureSVD, ItemKNN, ADMMSLIM). See `roadmap.md` session log.
- Remaining known tradeoffs:
  - Reproducibility is per-environment (same seed + binary): cross-SIMD-width
    bit-identity is not promised; persist with `save_model`/`load_model` to
    move models across environments bit-identically.
  - WMF Cholesky can produce NaN factors on scale-mixed extreme inputs
    (gramian near-singular within float eps); candidates for the λ=0
    singular-gramian guard treatment.
  - ADMMSLIM `recommend` adaptively densifies W for the GEMM path
    (~0.18s/154MB at thousands scale vs 0.13s/125MB pre-sparse-W).
  - `benchmark/compare.jl` only diffs across different git SHAs.
- CI: doctests + docs-example validation + `RegistryCI.yml` (package-side
  registry-readiness checks via `validation/registry_check.jl`).

## Pitfall: JET abstract-analysis artifact

- `zeros(T, n, n)` under JET's free-typevar analysis widens the slot to a
  union with `Array{Float64,3}`; use `fill(zero(T), n, n)` instead
  (see admmslim.jl fit!).
