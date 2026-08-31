# Gideon.jl Production Roadmap

Status legend: ✅ done · 🟡 partial (what remains is listed) · ⬜ not started

## Workflow

Each iteration follows the same cycle:

1. Inspect the current implementation and local diff.
2. Implement one focused change.
3. Add or update tests.
4. Run targeted tests.
5. Run `git diff --check` and review the complete diff.
6. Create one atomic commit.
7. Periodically run the full test suite.

Existing local changes in benchmarks and experimental files remain outside
these commits unless explicitly reviewed as part of the corresponding phase.

## Phase 1: Public API

1. Freeze and simplify the public API — ✅ done
   - `temporal_split` alias removed (d516d44), exports frozen in `src/Gideon.jl`,
     unused files deleted (b0d33b7).
   - Model/verb matrix is final: `fit!`/`update!`/`recommend`/`score`/`predict`
     documented in README + docs.

2. Unify the callback API — ✅ done
   - Unified `AbstractVector{<:AbstractCallback}` API (021e41e) with
     begin/epoch/end/early-stop semantics; implemented by all iterative models
     (WMF, IALS, EALS, BPR, LMF, GloVe, FTRL, FM, ADMMSLIM, ItemKNN,
     SoftImpute). Closed-form models (EASE, SLIM) have no epochs → no callbacks,
     consistent by design.

3. Standardize validation and errors — ✅ done
   - Shared validators: `_require_fitted`, `_require_nonempty_dimensions`,
     `_validate_recommend_input`; `ArgumentError`/`DimensionMismatch` used
     throughout (109838d). Two remaining `error()` calls are internal
     invariants (GloVe NaN cost, position lookup) — acceptable.

## Phase 2: State and Safety

4. Make `fit!` transactional — ✅ done
   - All families (WMF/ALS 6a0d955, SGD c3ccecd, item-similarity 463f308,
     concurrency guarantees 6861693): training into local buffers, publish on
     success, previous state intact on failure/refit. Tested in
     `test/test_infrastructure.jl`.

5. Define concurrency guarantees — ✅ done
   - Documented (README + docs): separate models parallel-safe, reads on fitted
     models safe, concurrent mutation unsupported, transactional fit!.
   - `test/test_concurrency.jl` covers nesting and multi-thread behavior.

## Phase 3: Scalability

6. Add memory safety checks — ✅ done
   - Memory-bounded batched GEMM top-k scoring for EASE/ADMMSLIM
     (`_predict_batched_gemm_topk`) and shared batched top-k for MF models
     (`_predict_topk_batched`) — full-score and top-k paths never materialize
     more than a bounded batch.
   - Fit-time peak-memory estimation + configurable `max_memory` guard
     (bytes, `nothing` = unlimited) on EASE/SLIM/ADMMSLIM: `fit!` estimates
     the dominant dense allocations (EASE 4 n², SLIM 2 n², ADMMSLIM 6 n²
     matrices in `T`) and throws `ArgumentError` before the first large
     allocation, leaving the model untouched (transactional). Tested in
     `test/test_memory_limits.jl`.

7. Separate full scoring from top-k scoring — ✅ done
   - Unified top-k paths (6b5a071): `_predict_sparse_score_topk` (SLIM,
     ItemKNN, ADMMSLIM), `_predict_batched_gemm_topk` (EASE, ADMMSLIM),
     `_predict_topk_batched` (matrix factorization). `score()` keeps full-matrix
     and pairwise variants; `recommend()` never builds the full score matrix.
     ADMMSLIM `recommend` dispatches adaptively (`_use_sparse_score_path`):
     sparse scoring when the score matrix is expected to stay sparse
     (P = 1-(1-d_W)^k ≤ 0.1), memory-bounded batched GEMM otherwise.

8. Decide and document ADMMSLIM's dense training model — ✅ done
   - The implementation is dense by design (joint ADMM solve over the full
     item-item matrix, 10–100× faster than per-item SLIM).
   - Memory model documented in the docstring + README: O(n_items²) training
     memory (≈5–6 dense n² matrices), sparse fitted `W`, when to prefer SLIM.
   - The fitted `W` is now stored as `SparseMatrixCSC` (soft-thresholded
     exact zeros dropped), cutting storage from O(n²) to O(nnz); `score`
     returns a sparse matrix.

## Phase 4: Persistence

9. Define a stable model persistence schema — ✅ done (current form)
   - `save_model`/`load_model` write a versioned header
     (`GIDEON_v2` + package version + model type) in front of native
     serialization; unsupported versions are rejected with `ArgumentError`.
   - Native serialization is isolated behind the header/version check.

10. Make persistence atomic — ✅ done
    - `save_model` writes to a temp file in the target directory, flushes,
      and `rename`s into place (atomic on POSIX, replace semantics on
      Windows); a crash or failed write never leaves a partial model at the
      target and temp files are cleaned up. Directory targets are rejected
      with `ArgumentError`. Tested in `test/test_infrastructure.jl`.

## Phase 5: Numerical Correctness

11. Validate numerical inputs and stability — ✅ done
    - Uniform NaN/Inf rejection across all 13 `fit!` entry points (plus
      EALS `update!`) via shared `_require_finite_input` /
      `_require_finite_vector` helpers: `ArgumentError` naming the model,
      before any training work, transactional on refit. FTRL now checks
      Inf as well as NaN and covers `y`; GloVe keeps its positive
      co-occurrence requirement after the finite check.
    - FTRL rejects NaN input; GloVe fails loudly on NaN cost; WMF guards
      singular gramians (7dcc37e); empty-input handling is consistent (80fd661).
    - `test/test_finite_input.jl` (261 tests): NaN/±Inf in Float32/Float64
      across every model, y-target rejection, transactional refit state,
      integer targets still accepted.

12. Normalize constructors — ✅ done
    - All constructors accept `Real` hyperparameters, convert to the selected
      `dtype`, and validate rank/iterations/regularization/learning rates.

## Phase 6: GPU and Performance

13. Complete CPU/GPU parity coverage — ✅ done
    - `test/test_gpu.jl` (17 tests, run with CUDA present): parity on small
      datasets, Float32/min-dims/empty behavior, stubs when CUDA unavailable.

14. Add reproducible benchmarks — ✅ done
    - `benchmark/run.jl` measures `fit!`/`recommend` time + `@allocated` at 3
      fixed scales with deterministic seeds; `benchmark/compare.jl` diffs across
      git SHAs; JSONL records in `benchmark/logs/results.jsonl`.

## Phase 7: Reference Validation

15. Maintain shared contracts with `implicit` — ✅ done
    - `test/test_reference_contracts.jl` (227 tests): seen-item exclusion,
      score ordering, valid indices, empty rows/columns, dtypes,
      reproducibility, persistence.

16. Add numerical reference fixtures — ✅ done
    - `test/test_fixtures.jl` (451 tests): pure-Julia ports of the `implicit`
      correctness fixtures (test_factorize, test_cg_nan, recommender_base).
    - `validation/` compares against R (rsparse) and Python (implicit/sklearn/
      scipy) with a one-command runner (`validation/run.jl --all`), including
      the ½-convention GloVe parity and sparse high-dim FM ground-truth
      recovery.

17. Add property-based sparse testing — ✅ done
    - `test/test_properties.jl` (15,318 tests): seeded random draws over
      dimensions, densities, dtypes, duplicated triplets and value bands,
      asserting round-trip invariants on every matrix — prediction
      shape/validity/distinctness, seen-item exclusion (when feasible),
      finite scores, deterministic re-fits (BPR excluded as Hogwild), k
      saturation, save/load round-trips. Signed-feedback models are
      exercised separately from the implicit-feedback family (IALS's
      sqrt(α·x) confidence requires non-negative ratings by design), and
      scale-uniform numerical-extreme bands verify the Cholesky-based
      solvers stay finite.

## Phase 8: Package Quality

18. Remove dead code and experimental artifacts — ✅ done
    - Unused files deleted (b0d33b7), legacy `validation/r_correctness.jl`
      removed. Audit (2026-08-31): all 82 exports defined and referenced in
      tests/docs (Aqua `test_undefined_exports`); all 122 internal helpers
      referenced; `dual_representation` and `link_function` confirmed in use;
      no stale docs/build artifacts tracked (`docs/build` and
      `docs/Manifest.toml` gitignored).

19. Complete production documentation — ✅ done
    - README + docs/ cover algorithm selection, performance design, memory
      behavior of scoring paths, concurrency, reproducibility, GPU support,
      persistence, validation workflow, and executable examples (validated by
      `test/validate_docs.jl`).

20. Harden release CI — ✅ done
    - Matrix 1/lts/pre × ubuntu/windows + macOS aarch64; Aqua + JET in
      the test suite; coverage; docs build/deploy job.
    - Doctests enabled (`doctest = true` in `docs/make.jl`); docstring
      examples converted to executable `julia>` doctests (this caught a
      latent bug: `load_model`'s example called `predict` on an EASE
      model, which only FTRL/FM support). `coef` got a generic stub and
      `AbstractSoftALS` is documented so `checkdocs = :exports` passes.
    - `validate_docs.jl` (docs/src examples) wired into the test suite, so
      both docstring doctests and page examples run on every PR.
    - RegistryCI: `.github/workflows/RegistryCI.yml` runs
      `validation/registry_check.jl`, a package-side enforcement of the
      General-registry rules (name/uuid/semver, compat coverage for all
      non-stdlib deps and weakdeps, parseable VersionSpec bounds, no stale
      deps) that would otherwise only run at registration time.

## Release Candidate Gates

Before `v1.0.0`:

1. Clean the worktree. — ✅ (routine)
2. Run `Pkg.test()` successfully. — ✅ (17,430 passing)
3. Build documentation and run doctests. — ✅ (doctests wired into CI)
4. Run RegistryCI. — ✅ (package-side checks on every PR via
   `validation/registry_check.jl`; the registration-time check still runs
   on the General registry when the package is registered)
5. Install from a clean environment. — ⬜ (process gate, do at release)
6. Test persistence in a separate Julia process. — ✅ (test_infrastructure.jl)
7. Run baseline benchmarks. — ✅ (benchmark/run.jl + logs)
8. Publish `v1.0.0-rc1`. — ⬜
9. Review every commit included in the release. — ⬜ (process gate)
10. Publish `v1.0.0` only after all gates pass. — ⬜

## Remaining work (priority order)

All roadmap items are complete. Only the release process gates remain
(above). Known follow-ups outside the roadmap scope:

- **WMF NaN factors on scale-mixed extreme inputs**: Cholesky solves can
  produce NaN factors when the gramian is near-singular within float eps
  (values spanning many orders of magnitude in one matrix). Detected while
  designing the extremes property tests; out of scope there because the
  bands are scale-uniform by construction. Could be handled like the
  λ=0 singular-gramian guard.
- **Benchmark note**: ADMMSLIM `recommend` now adaptively picks the sparse
  or batched-GEMM path; measured at thousands scale 0.181s/154MB vs the
  pre-sparse-W 0.133s/125MB (the densification costs the difference).

## Session log

### 2026-08-31 — Tables.jl hard dependency (follow-up review)
- `interactions_to_sparse` failed on `DataFrame` ("AbstractDataFrame is not
  iterable") and overclaimed "Tables.jl-compatible": only NamedTuple column
  tables and duck-typed row objects worked. Rewritten on the real Tables.jl
  interface (`Tables.columns`/`Tables.getcolumn`/`Tables.columnnames`) with
  Tables as a hard dependency (pure-Julia, ~300 KB) — an extension would
  have hidden core API behind a weakdep. Now works with DataFrames, CSV,
  Arrow, tuplerows, etc.
- Also: `dtype` kwarg for the value column; duplicate (user, item) pairs
  documented as summing (sparse semantics); clean `ArgumentError`s for
  empty tables, missing columns, non-integer IDs, and non-table input
  (was `ErrorException`/`InexactError`/`MethodError`); column-length
  mismatch → `DimensionMismatch`; range checks without temporaries.
- Tests: 38 in `test/test_tables.jl` (incl. DataFrame, dtype, duplicates,
  error paths, rowtable round-trip); DataFrames added to the test env with
  `import` (not `using`) to avoid the `transform` export clash; the
  validate_docs gate gained a DataFrame example; both docstrings became
  executable doctests.

### 2026-08-31 — All remaining roadmap items completed
- **Item 10 (atomic persistence)**: `save_model` writes to a temp file and
  renames into place; directory targets rejected; temp cleanup on failure.
  `test/test_infrastructure.jl` gains 9 atomicity tests.
- **ADMMSLIM fitted-W sparsity + Item 8**: `W` is now `SparseMatrixCSC`
  (soft-thresholded exact zeros dropped), `score` returns sparse. The pure
  sparse recommend path regressed 8× on dense-ish W at benchmark scale, so
  `_use_sparse_score_path` dispatches adaptively on the expected fill of
  S = X·W (`P = 1-(1-d_W)^k ≤ 0.1`); benchmark re-verified. Memory model
  documented (dense O(n²) joint solve, when to prefer SLIM). JET artifact
  with `zeros(T,n,n)` under abstract typevars sidestepped via `fill`.
- **Item 6 (fit-time memory)**: `max_memory` (bytes) on EASE/SLIM/ADMMSLIM
  with `_fit_memory_estimate`/`_require_fit_memory` guards before the first
  large allocation; 27 tests in `test/test_memory_limits.jl`.
- **Item 11 (uniform NaN/Inf rejection)**: `_require_finite_input`/
  `_require_finite_vector` applied to all 13 `fit!` + EALS `update!`;
  FTRL's NaN-only check replaced (now Inf + y too); 261 tests in
  `test/test_finite_input.jl`.
- **Item 17 (property tests)**: `test/test_properties.jl` — 15,318 seeded
  randomized tests across dims/densities/dtypes/duplicates/value bands.
  Findings: IALS's sqrt(α·x) confidence requires non-negative ratings
  (documented); LMF needs ≥1 unobserved item per user; WMF Cholesky can
  produce NaN on scale-mixed extreme inputs (recorded as follow-up).
- **Item 18 (audit)**: all 82 exports defined and referenced; all 122
  internal helpers used; no stale artifacts tracked. Nothing to fix.
- **Item 20 (CI)**: doctests enabled; docstring examples converted to
  executable `julia>` doctests (caught a `predict`-on-EASE bug in the
  `load_model` docstring); `coef` generic stub + `AbstractSoftALS`
  documented; `validate_docs.jl` wired into the test suite;
  `RegistryCI.yml` + `validation/registry_check.jl` enforce
  General-registry rules on every PR.
- Test suite: **17,430 passing** (was 1,802 at session start), incl.
  Aqua + JET + GPU ext.

### 2026-08-29 — Validation overhaul + algorithm parity deep-dive
- **Validation harness refactored** (`validation/`): deleted dead legacy
  `r_correctness.jl`; shared helpers in `common.jl`; table-driven
  `validate_py.jl`; `run.jl` now runs each step in a subprocess with proper
  exit codes and a summary table (no more silent passes when fixtures are
  missing); R fixtures run via `uvr run` (fallback `Rscript`).
- **GloVe ½-convention adopted** (`src/algorithms/glove.jl`): loss is now
  `½ Σ f(x)·diff²` and the gradient drops the factor of 2 — identical
  accounting to rsparse/Stanford C, so `learning_rate` semantics and loss
  curves are directly comparable. R-parity gate is now `cost ≤ R × 1.15` at
  60 epochs (observed ratio 0.55). The old 2.1×/2.5× gate was a bookkeeping
  artifact, not a quality gap.
- **LogisticMF negative-sampling cap fixed** (`src/algorithms/lmf.jl`): was
  `min(k, seen·n_neg)` (capped at rank), now `min(n_items, seen·n_neg)` to
  match `implicit` (lmf.pyx). Rejection sampling on unobserved items kept (a
  deliberate improvement over implicit, which can resample seen items).
- **SLIM/ItemKNN validated** against sklearn: SLIM W-cor 0.995; ItemKNN W /
  score / top-10 overlap all 1.0. Diagonal masking verified correct.
- **FM sparse high-dim validation added** (R fixture + testset): independent
  dense reference of the FM equation (rel. err ~1e-7), held-out recovery of a
  known rank-2 latent interaction (cor 0.999), prediction agreement with
  rsparse (cor ≥ 0.95, gated with margin because rsparse's FM init uses
  Armadillo's `std::random_device` — nondeterministic by design).
- **Identified pending optimization**: ADMMSLIM's fitted `W` is stored dense
  but could be `SparseMatrixCSC` (see remaining work above).
