# ──────────────────────────────────────────────────────────────────────────────
# Experimental algorithms
#
# Models that are demoted from the core catalog: implemented, tested, and
# numerically validated against their reference implementations, but known to
# underperform on implicit Top-N benchmarks vs the core models, or to require
# fragile hyperparameter settings. They live in `Canapes.Experimental`, NOT in
# the root exports, so the production API stays clean while the code, tests,
# and reference-validated behavior remain in the package.
#
# Currently: LogisticMF (Johnson 2014) — calibration-oriented logistic matrix
# factorization (fragile under Adagrad) — and PMF (Mnih & Salakhutdinov 2007)
# — probabilistic MF via MAP-SGD with no reference-parity target.
# ──────────────────────────────────────────────────────────────────────────────

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
include("experimental/pmf.jl")

export LogisticMF, PMF
end # module Experimental
