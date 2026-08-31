# ──────────────────────────────────────────────────────────────────────────────
# SLIM — Sparse Linear Methods for Top-N Recommendations
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Ning & Karypis (2011)
#   "SLIM: Sparse Linear Methods for Top-N Recommender Systems" (ICDM 2011)
#
# Learns a sparse item-item weight matrix W by solving, for each item j:
#   min_wⱼ ½‖xⱼ - X·wⱼ‖² + λ_l2/2‖wⱼ‖² + λ_l1‖wⱼ‖₁
#   subject to wⱼ ≥ 0, wⱼⱼ = 0
#
# This is coordinate descent on an elastic-net regression per item column.
# ──────────────────────────────────────────────────────────────────────────────

"""
    SLIM{T} <: AbstractSparseModel

Sparse Linear Methods (SLIM) for item-based collaborative filtering.

Learns a sparse, non-negative item-item weight matrix using elastic net
regularization (L1 + L2). The sparsity of W makes predictions efficient
and interpretable.

# Constructor
```julia
SLIM(; λ_l1=0.01, λ_l2=0.1, max_iter=50, tol=1e-4, verbose=true,
        max_memory=nothing)
```

# Fields
- `λ_l1::T` — L1 penalty (sparsity)
- `λ_l2::T` — L2 penalty (shrinkage)
- `max_iter::Int` — max coordinate descent iterations per item
- `tol::T` — convergence threshold for coordinate descent
- `nonneg::Bool` — enforce non-negative weights (default: true)
- `max_memory::Union{Nothing,Int}` — fit-time peak-memory limit in bytes
  (`nothing` = unlimited); a fit whose estimated peak exceeds it throws
  `ArgumentError` before any large allocation
"""
mutable struct SLIM{T<:AbstractFloat} <: AbstractItemSimilarity
    const λ_l1::T
    const λ_l2::T
    const max_iter::Int
    const tol::T
    const nonneg::Bool
    const verbose::Bool
    const max_memory::Union{Nothing,Int}
    W::SparseMatrixCSC{T,Int}
    is_fitted::Bool
end

function SLIM(;
    λ_l1::Float64 = 0.01,
    λ_l2::Float64 = 0.1,
    max_iter::Int = 50,
    tol::Float64 = 1e-4,
    nonneg::Bool = true,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
    max_memory::Union{Nothing,Int} = nothing,
)
    λ_l1 >= 0.0 || throw(ArgumentError("λ_l1 must be non-negative, got $λ_l1"))
    λ_l2 >= 0.0 || throw(ArgumentError("λ_l2 must be non-negative, got $λ_l2"))
    max_memory === nothing || max_memory > 0 ||
        throw(ArgumentError("max_memory must be positive, got $max_memory"))
    SLIM{T}(T(λ_l1), T(λ_l2), max_iter, T(tol), nonneg, verbose, max_memory,
            spzeros(T, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::SLIM, X) -> model

Fit SLIM on interaction matrix `X` (users × items).
Solves n_items independent elastic net problems via coordinate descent.
"""
function fit!(model::SLIM{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng()) where {T,Tv,Ti}
    old_W = model.W
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "SLIM")
    _require_finite_input(X, "SLIM")

    # Peak fit memory: G plus the assembled W — at most two dense n_items²
    # equivalents (W is stored sparse, so this is an upper bound).
    _require_fit_memory(_fit_memory_estimate(n_items, 2, T), model.max_memory, "SLIM")

    # Precompute XᵀX (Gram matrix) and column norms
    G = Matrix{T}(X' * X)   # n_items × n_items
    diag_G = [G[j, j] for j in 1:n_items]

    model.verbose && @info "[SLIM] Fitting $(n_items) items via coordinate descent..."

    # Solve per-column elastic net problems in parallel
    W_cols = Vector{SparseVector{T,Int}}(undef, n_items)

    Threads.@threads for j in 1:n_items
        W_cols[j] = _slim_fit_column(G, diag_G, j, n_items, model)
    end

    # Assemble sparse weight matrix
    model.W = hcat(W_cols...)
    model.is_fitted = true

    nnz_w = nnz(model.W)
    density = nnz_w / (n_items * n_items) * 100
    model.verbose && @info "[SLIM] Done. W: $(n_items)×$(n_items), nnz=$(nnz_w) ($(round(density, digits=3))%)"
    model
    catch
        model.W = old_W
        model.is_fitted = old_is_fitted
        rethrow()
    end
end

"""
Fit one column of W using coordinate descent for elastic net.
"""
function _slim_fit_column(G::Matrix{T}, diag_G::Vector{T},
                          j::Int, n_items::Int, model::SLIM{T}) where {T}
    λ_l1 = model.λ_l1
    λ_l2 = model.λ_l2
    max_iter = model.max_iter
    tol = model.tol
    nonneg = model.nonneg

    # Target: Xᵀxⱼ = G[:, j]
    w = zeros(T, n_items)
    target = G[:, j]  # XᵀX[:,j]

    # Precompute residuals: r[i] = target[i] - Σ_k G[i,k]*w[k]
    # Initially r = target since w=0
    residual = copy(target)
    residual[j] = zero(T)  # skip diagonal

    for _ in 1:max_iter
        max_change = zero(T)

        for i in 1:n_items
            i == j && continue  # diagonal constraint

            # Use cached residual + correction for current w[i]
            numerator = residual[i] + diag_G[i] * w[i]

            # Elastic net update with soft-thresholding
            denom = diag_G[i] + λ_l2

            if nonneg
                new_w = max(zero(T), (numerator - λ_l1)) / denom
            else
                new_w = _soft_threshold(numerator, λ_l1) / denom
            end

            # Update residual incrementally: Δw = new_w - w[i]
            delta = new_w - w[i]
            if !iszero(delta)
                @inbounds @simd for k in 1:n_items
                    residual[k] -= G[k, i] * delta
                end
                residual[j] = zero(T)  # keep diagonal zeroed
                w[i] = new_w
                change = abs(delta)
                if change > max_change
                    max_change = change
                end
            end
        end

        max_change < tol && break
    end

    # Return as sparse vector
    sparsevec(w)
end

@inline function _soft_threshold(x::T, λ::T) where {T}
    if x > λ
        x - λ
    elseif x < -λ
        x + λ
    else
        zero(T)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    recommend(model::SLIM, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores = X * W, excluding seen items.
"""
function recommend(model::SLIM{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    _predict_sparse_score_topk(X * model.W, X, k)
end

"""
    score(model::SLIM, X) -> SparseMatrixCSC

Return sparse score matrix S = X * W.
"""
function score(model::SLIM{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    size(X, 2) == size(model.W, 1) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.W, 1))"))
    X * model.W
end
