# ──────────────────────────────────────────────────────────────────────────────
# ADMM-SLIM — ADMM-based Sparse Linear Methods
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Steck, Liang, et al. (2020)
#   "ADMM SLIM: Sparse Recommendations for Many Users"
#   (WSDM 2020) — arXiv:2003.04710
#
# Instead of solving n_items independent elastic-net problems (standard SLIM),
# ADMM-SLIM solves the full item-item weight matrix jointly using ADMM:
#
#   min_B  ½‖X - XB‖²_F + λ_1‖B‖₁ + λ_2/2‖B‖²_F
#   s.t.   diag(B) = 0
#
# This is equivalent to SLIM but 10-100× faster because:
# 1. Computes the Gram matrix G = XᵀX once (dominant cost)
# 2. Pre-factors (G + ρI)⁻¹ via Cholesky once
# 3. Iterates ADMM updates (matrix-level, not per-column)
#
# The result interpolates between EASE (λ_1=0) and SLIM (λ_1>0).
# ──────────────────────────────────────────────────────────────────────────────

"""
    ADMMSLIM{T} <: AbstractItemSimilarity

ADMM-based Sparse Linear Methods for top-N recommendation.

A dramatically faster alternative to standard SLIM that solves the full
item-item weight matrix jointly using ADMM. Produces the same solution
as coordinate-descent SLIM but in 10-100× less time.

# Constructor
```julia
ADMMSLIM(; λ_1=0.01, λ_2=100.0, ρ=1.0, max_iter=50, convergence_tol=1e-4,
            nonneg=true, verbose=true)
```

# Fields
- `λ_1::T` — L1 penalty (sparsity inducing, via soft-thresholding)
- `λ_2::T` — L2 penalty (shrinkage / regularization)
- `ρ::T` — ADMM penalty parameter (controls convergence speed)
- `max_iter::Int` — max ADMM iterations
- `convergence_tol::T` — relative primal residual tolerance
- `nonneg::Bool` — enforce non-negative weights
- `W::Matrix{T}` — item-item weight matrix (n_items × n_items) after fitting

# Example
```julia
using SparseArrays, Gideon
X = sprand(1000, 500, 0.02)
model = ADMMSLIM(λ_1=0.05, λ_2=200.0)
fit!(model, X)
preds = recommend(model, X; k=10)
```
"""
mutable struct ADMMSLIM{T<:AbstractFloat} <: AbstractItemSimilarity
    const λ_1::T
    const λ_2::T
    const ρ::T
    const max_iter::Int
    const convergence_tol::T
    const nonneg::Bool
    const verbose::Bool
    W::Matrix{T}
    is_fitted::Bool
end

function ADMMSLIM(;
    λ_1::Float64 = 0.01,
    λ_2::Float64 = 100.0,
    ρ::Float64 = 1.0,
    max_iter::Int = 50,
    convergence_tol::Float64 = 1e-4,
    nonneg::Bool = true,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    λ_1 >= 0.0 || throw(ArgumentError("λ_1 must be non-negative, got $λ_1"))
    λ_2 >= 0.0 || throw(ArgumentError("λ_2 must be non-negative, got $λ_2"))
    ρ > 0.0 || throw(ArgumentError("ρ must be positive, got $ρ"))
    T = dtype
    ADMMSLIM{T}(T(λ_1), T(λ_2), T(ρ), max_iter, T(convergence_tol), nonneg, verbose,
                 Matrix{T}(undef, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::ADMMSLIM, X; rng) -> model

Fit ADMM-SLIM on interaction matrix `X` (users × items).

Computes the Gram matrix once, pre-factors `(G + ρI)`, then iterates ADMM:
- B-update: solve linear system (pre-factored Cholesky)
- Z-update: soft-thresholding (proximal L1) + optional non-negativity
- U-update: dual variable accumulation
"""
function fit!(model::ADMMSLIM{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    λ_1 = model.λ_1
    λ_2 = model.λ_2
    ρ = model.ρ

    model.verbose && @info "[ADMM-SLIM] Computing Gram matrix ($n_items × $n_items)..."

    # Gram matrix: G = XᵀX
    G = Matrix{T}(X' * X)

    # Pre-factor: (G + (λ_2 + ρ)I)⁻¹ via Cholesky
    # The B-update solves: (G + (λ_2+ρ)I) B = G + ρ(Z - U)
    # Pre-factor the LHS
    lhs = copy(G)
    @inbounds for j in 1:n_items
        lhs[j, j] += λ_2 + ρ
    end
    C = cholesky(Symmetric(lhs))

    model.verbose && @info "[ADMM-SLIM] Running ADMM ($n_items items, max_iter=$(model.max_iter))..."

    # Initialize ADMM variables
    B = zeros(T, n_items, n_items)
    Z = zeros(T, n_items, n_items)
    U = zeros(T, n_items, n_items)  # scaled dual variable

    for iter in 1:model.max_iter
        # ── B-update: B = (G + (λ₂+ρ)I)⁻¹ (G + ρ(Z - U)) ──
        rhs = G .+ ρ .* (Z .- U)
        B .= C \ rhs

        # Enforce diag(B) = 0
        @inbounds for j in 1:n_items
            B[j, j] = zero(T)
        end

        # ── Z-update: proximal operator (soft-threshold + optional non-neg) ──
        B_plus_U = B .+ U
        threshold = λ_1 / ρ

        if model.nonneg
            # Soft-threshold + clip to non-negative
            @inbounds for idx in eachindex(Z)
                z = B_plus_U[idx] - threshold
                Z[idx] = z > zero(T) ? z : zero(T)
            end
        else
            # Standard soft-thresholding
            @inbounds for idx in eachindex(Z)
                v = B_plus_U[idx]
                if v > threshold
                    Z[idx] = v - threshold
                elseif v < -threshold
                    Z[idx] = v + threshold
                else
                    Z[idx] = zero(T)
                end
            end
        end

        # Enforce diag(Z) = 0
        @inbounds for j in 1:n_items
            Z[j, j] = zero(T)
        end

        # ── U-update: dual variable ──
        U .+= B .- Z

        # ── Convergence check: relative primal residual ──
        primal_resid = zero(T)
        norm_B = zero(T)
        @inbounds for idx in eachindex(B)
            d = B[idx] - Z[idx]
            primal_resid += d * d
            norm_B += B[idx] * B[idx]
        end
        rel_resid = sqrt(primal_resid) / (sqrt(norm_B) + T(1e-12))

        if model.verbose && (iter <= 5 || iter % 10 == 0 || iter == model.max_iter)
            nnz_iter = count(>(T(1e-10)), Z)
            @info "[ADMM-SLIM] iter=$iter  rel_resid=$(round(rel_resid; sigdigits=4))  nnz=$(nnz_iter)"
        end

        if rel_resid < model.convergence_tol
            model.verbose && @info "[ADMM-SLIM] Converged at iteration $iter (rel_resid=$(round(rel_resid; sigdigits=4)))"
            break
        end
    end

    model.W = Z  # use the sparse (thresholded) variable as final weights
    model.is_fitted = true

    nnz_w = count(!iszero, model.W)
    density = nnz_w / (n_items * n_items) * 100
    model.verbose && @info "[ADMM-SLIM] Done. W: $(n_items)×$(n_items), nnz=$(nnz_w) ($(round(density; digits=3))%)"
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# recommend / score
# ──────────────────────────────────────────────────────────────────────────────

"""
    recommend(model::ADMMSLIM, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores = X * W, excluding seen items.
"""
function recommend(model::ADMMSLIM{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    n_users = size(X, 1)
    n_items = size(model.W, 1)
    k_out = min(k, n_items)

    # Batched GEMM: convert sparse X to dense in chunks, multiply via BLAS
    preds = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)

    max_batch_mem = 2 * 1024^3
    batch_size = max(1, min(n_users, Int(floor(max_batch_mem / (2 * n_items * sizeof(T))))))

    nt = Threads.maxthreadid()
    topk_bufs = [Vector{Int}(undef, k_out) for _ in 1:nt]
    scores_buf = Matrix{T}(undef, n_items, batch_size)
    X_batch = Matrix{T}(undef, batch_size, n_items)

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        n_batch = batch_end - batch_start + 1

        Xb = @view X_batch[1:n_batch, :]
        fill!(Xb, zero(T))
        @inbounds for u_local in 1:n_batch
            u = batch_start + u_local - 1
            for idx in nzrange(X_csr, u)
                j = Int(X_csr.colval[idx])
                Xb[u_local, j] = T(X_csr.nzval[idx])
            end
        end

        Sb = @view scores_buf[:, 1:n_batch]
        mul!(Sb, model.W', Xb')

        Threads.@threads for u_local in 1:n_batch
            tid = Threads.threadid()
            u = batch_start + u_local - 1
            @inbounds for idx in nzrange(X_csr, u)
                j = Int(X_csr.colval[idx])
                scores_buf[j, u_local] = T(-Inf)
            end
            col = @view scores_buf[:, u_local]
            topk = topk_bufs[tid]
            _topk_indices!(topk, col, k_out)
            @inbounds for i in 1:k_out
                preds[u, i] = topk[i]
            end
        end
    end
    preds
end

"""
    score(model::ADMMSLIM, X) -> Matrix{T}

Return the full score matrix S = X * W (dense, n_users × n_items).
"""
function score(model::ADMMSLIM{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    Matrix{T}(X * model.W)
end
