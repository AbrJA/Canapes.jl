# ──────────────────────────────────────────────────────────────────────────────
# Experimental algorithms
#
# Models that are demoted from the core catalog: implemented, tested, and
# numerically validated against their reference implementations, but known to
# underperform on implicit Top-N benchmarks vs the core models, or to require
# fragile hyperparameter settings. They live in `Canapes.Experimental`, NOT in
# the root exports, so the production API stays clean while the code, tests,
# and reference-validated behavior remain in the package.
# ──────────────────────────────────────────────────────────────────────────────

"""
    Canapes.Experimental

Namespaced experimental algorithms, kept out of the root exports so the
production API stays clean. Currently:

- [`LogisticMF`](@ref) (Johnson 2014) — logistic matrix factorization,
  reference-validated against `implicit`, but fragile under Adagrad and weak
  on implicit Top-N benchmarks.
- [`ProbabilisticMF`](@ref) (Mnih & Salakhutdinov 2007) — probabilistic matrix
  factorization via MAP-SGD (no reference-parity target).

Both are implemented, tested, and numerically validated like the core model
catalog; they are namespaced here only because of their benchmark standing.
"""
module Experimental

import ..Canapes   # parent: needed for the internal helpers below
import ..Canapes: fit!, predict, score
using ..Canapes: AbstractMatrixFactorization, AbstractExplicitModel,
                 AbstractCallback, CallbackInfo,
                 run_callbacks, run_callbacks_train_begin, run_callbacks_train_end,
                 ConvergenceMonitor, record!, elapsed_seconds, log_iteration,
                 to_csr, _require_finite_input, _require_nonempty_dimensions,
                 _require_fitted, _predict_pairwise_scores

using SparseArrays
using LinearAlgebra
using Random
using SparseMatricesCSR

include("experimental/logisticmf.jl")
include("experimental/probabilisticmf.jl")

export LogisticMF, ProbabilisticMF
end # module Experimental
