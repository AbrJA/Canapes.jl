# Gideon.jl Production Roadmap

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

1. Freeze and simplify the public API.
   - Remove deprecated aliases such as `temporal_split`.
   - Keep only final, consistent names and exports.
   - Define which models implement `fit!`, `update!`, `recommend`, `score`,
     and `predict`.
   - Remove exports that are not stable API.
   - Commit: `refactor: simplify public API`

2. Unify the callback API.
   - Make callback support consistent across training algorithms, or remove
     callbacks from models that do not support them.
   - Accept `AbstractVector{<:AbstractCallback}`.
   - Define begin, epoch, end, failure, and early-stop semantics.
   - Commit: `refactor: unify training callback API`

3. Standardize validation and errors.
   - Share validation for fitted state, dimensions, indices, `k`, `dtype`,
     and numerical values.
   - Use `ArgumentError`, `DimensionMismatch`, and specific errors instead of
     generic `error` calls.
   - Commit: `refactor: standardize API validation`

## Phase 2: State and Safety

4. Make `fit!` transactional.
   - Train into local buffers.
   - Publish model state only after successful completion.
   - Define refit behavior and failure semantics.
   - Test failures during training and refits.
   - Commit: `fix: make model fitting transactional`

5. Define concurrency guarantees.
   - Document that mutating the same model concurrently is unsupported unless
     explicitly synchronized.
   - Remove global mutable state.
   - Test behavior across different `JULIA_NUM_THREADS` values.
   - Commit: `fix: define model concurrency guarantees`

## Phase 3: Scalability

6. Add memory safety checks.
   - Estimate memory before allocating dense item-item matrices.
   - Add configurable limits for EASE, SLIM, and ADMMSLIM.
   - Document actual complexity and memory requirements.
   - Commit: `feat: add memory safety checks`

7. Separate full scoring from top-k scoring.
   - Keep full scores, pairwise scores, and top-k recommendations explicit.
   - Avoid full dense matrices when only top-k results are needed.
   - Commit: `refactor: separate full and top-k scoring`

8. Decide and document ADMMSLIM's dense training model.
   - Keep it dense with accurate documentation, or design a sparse variant.
   - Commit: `docs: document ADMMSLIM memory model`

## Phase 4: Persistence

9. Define a stable model persistence schema.
   - Store schema version, package version, model type, dtype, dimensions,
     hyperparameters, factors, and required state.
   - Replace native Julia serialization or isolate it behind the schema.
   - Commit: `refactor: add stable model persistence schema`

10. Make persistence atomic.
    - Write to a temporary file, validate it, and rename it into place.
    - Test corrupt and interrupted writes.
    - Commit: `fix: make model persistence atomic`

## Phase 5: Numerical Correctness

11. Validate numerical inputs and stability.
    - Reject NaN and Inf inputs.
    - Define negative and zero-value semantics.
    - Protect exponential and logarithmic paths.
    - Test Float32 and Float64 extremes.
    - Commit: `fix: validate numerical inputs and stability`

12. Normalize constructors.
    - Accept `Real` hyperparameters.
    - Convert values to the selected dtype.
    - Validate rank, iterations, regularization, and learning rates.
    - Commit: `refactor: normalize model constructors`

## Phase 6: GPU and Performance

13. Complete CPU/GPU parity coverage.
    - Compare supported solvers on small datasets.
    - Test Float32, minimum dimensions, and empty inputs.
    - Define behavior when CUDA is unavailable.
    - Commit: `test: expand CPU and GPU parity coverage`

14. Add reproducible benchmarks.
    - Separate benchmarks from functional tests.
    - Measure time, allocations, and memory.
    - Define regression thresholds.
    - Commit: `perf: add reproducible benchmark suite`

## Phase 7: Reference Validation

15. Maintain shared contracts with `implicit`.
    - Test seen-item exclusion, score ordering, valid indices, empty rows and
      columns, dtypes, reproducibility, and persistence.

16. Add numerical reference fixtures.
    - Compare observable outputs, not factor orientations.
    - Use small datasets and explicit tolerances.
    - Commit: `test: add implicit numerical reference fixtures`

17. Add property-based sparse testing.
    - Generate random sparse matrices across dimensions and densities.
    - Test minimum sizes, duplicates, and numerical extremes.
    - Commit: `test: add randomized sparse model properties`

## Phase 8: Package Quality

18. Remove dead code and experimental artifacts.
    - Remove `delete*` files, unused symbols, and unstable exports.
    - Commit: `refactor: remove dead code`

19. Complete production documentation.
    - Document algorithm selection, memory limits, CPU/GPU support,
      persistence, reproducibility, and executable examples.
    - Commit: `docs: complete production usage documentation`

20. Harden release CI.
    - Test Julia LTS, stable, and nightly on Linux, macOS, and Windows.
    - Run Aqua, JET, tests, coverage, documentation, and RegistryCI.
    - Commit: `ci: harden release validation`

## Release Candidate Gates

Before `v1.0.0`:

1. Clean the worktree.
2. Run `Pkg.test()` successfully.
3. Build documentation and run doctests.
4. Run RegistryCI.
5. Install from a clean environment.
6. Test persistence in a separate Julia process.
7. Run baseline benchmarks.
8. Publish `v1.0.0-rc1`.
9. Review every commit included in the release.
10. Publish `v1.0.0` only after all gates pass.
