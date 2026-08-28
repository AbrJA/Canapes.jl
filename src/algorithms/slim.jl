# ──────────────────────────────────────────────────────────────────────────────
# SLIM — Sparse Linear Methods for Top-N Recommendations
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Ning & Karypis (2011)
#   "SLIM: Sparse Linear Methods for Top-N Recommender Systems" (ICDM 2011)
#
# Learns a sparse item-item weight matrix W by solving, for each item j:
#   min_wⱼ ½‖xⱼ - X·wⱼ‖² + λ_2/2‖wⱼ‖² + λ_1‖wⱼ‖₁
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
SLIM(; λ_1=0.01, λ_2=0.1, max_iter=50, convergence_tol=1e-4, verbose=true)
```

# Fields
- `λ_1::T` — L1 penalty (sparsity)
- `λ_2::T` — L2 penalty (shrinkage)
- `max_iter::Int` — max coordinate descent iterations per item
- `convergence_tol::T` — convergence threshold for coordinate descent
- `nonneg::Bool` — enforce non-negative weights (default: true)
"""
mutable struct SLIM{T<:AbstractFloat} <: AbstractItemSimilarity
    const λ_1::T
    const λ_2::T
    const max_iter::Int
    const convergence_tol::T
    const nonneg::Bool
    const verbose::Bool
    W::SparseMatrixCSC{T,Int}
    is_fitted::Bool
end

function SLIM(;
    λ_1::Float64 = 0.01,
    λ_2::Float64 = 0.1,
    max_iter::Int = 50,
    convergence_tol::Float64 = 1e-4,
    nonneg::Bool = true,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    λ_1 >= 0.0 || throw(ArgumentError("λ_1 must be non-negative, got $λ_1"))
    λ_2 >= 0.0 || throw(ArgumentError("λ_2 must be non-negative, got $λ_2"))
    T = dtype
    SLIM{T}(T(λ_1), T(λ_2), max_iter, T(convergence_tol), nonneg, verbose,
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
    λ_1 = model.λ_1
    λ_2 = model.λ_2
    max_iter = model.max_iter
    tol = model.convergence_tol
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
            denom = diag_G[i] + λ_2

            if nonneg
                new_w = max(zero(T), (numerator - λ_1)) / denom
            else
                new_w = _soft_threshold(numerator, λ_1) / denom
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
    n_users = size(X, 1)
    n_items = size(model.W, 1)
    k_out = _validate_recommend_input(X, n_items, k)

    # Compute all scores at once: S = X * W (sparse × sparse)
    S = X * model.W
    S_csr = to_csr(S)
    preds = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)

    nt = Threads.nthreads()
    topk_bufs = _thread_buffers(() -> Vector{Int}(undef, k_out), nt)
    score_bufs = _thread_buffers(() -> zeros(T, n_items), nt)

    Threads.@threads for chunk in 1:nt
        scores = score_bufs[chunk]
        topk = topk_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n_users, nt)

        # Zero out scores
        @inbounds @simd for i in 1:n_items
            scores[i] = zero(T)
        end

        # Fill from CSR row of S
        @inbounds for idx in nzrange(S_csr, u)
            j = Int(S_csr.colval[idx])
            scores[j] = S_csr.nzval[idx]
        end

        # Mask seen items
        @inbounds for idx in nzrange(X_csr, u)
            j = Int(X_csr.colval[idx])
            scores[j] = T(-Inf)
        end

        topk = topk_bufs[chunk]
        _topk_indices!(topk, scores, k_out)
        @inbounds for i in 1:k_out
            preds[u, i] = topk[i]
        end
        end
    end
    preds
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
