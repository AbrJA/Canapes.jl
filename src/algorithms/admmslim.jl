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
            nonneg=true, verbose=true, max_memory=nothing)
```

# Fields
- `λ_1::T` — L1 penalty (sparsity inducing, via soft-thresholding)
- `λ_2::T` — L2 penalty (shrinkage / regularization)
- `ρ::T` — ADMM penalty parameter (controls convergence speed)
- `max_iter::Int` — max ADMM iterations
- `convergence_tol::T` — relative primal residual tolerance
- `nonneg::Bool` — enforce non-negative weights
- `max_memory::Union{Nothing,Int}` — fit-time peak-memory limit in bytes
  (`nothing` = unlimited); a fit whose estimated peak exceeds it throws
  `ArgumentError` before any large allocation
- `W::SparseMatrixCSC{T,Int}` — item-item weight matrix (n_items × n_items) after fitting

# Memory model
Training is necessarily **dense**: the joint ADMM solve keeps the full
item-item Gram matrix and ADMM variables in memory, so fitting uses
O(n_items²) memory (≈5 dense n_items² matrices in `T`). The fitted `W`,
however, is stored as a `SparseMatrixCSC`: soft-thresholding produces exact
zeros, so only the surviving weights are kept (observed density is well
below 100% and shrinks as λ_1 grows), and `score` returns a sparse matrix.
`recommend` picks the scoring path adaptively (sparse when the score matrix
stays sparse, memory-bounded batched GEMM otherwise).

Prefer SLIM when `n_items` is large enough that an O(n_items²) dense solve
is prohibitive (SLIM fits per-item in O(n_items) memory per column), or
when the training set is small enough that SLIM's slower per-item
coordinate descent is affordable.

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
    const max_memory::Union{Nothing,Int}
    W::SparseMatrixCSC{T,Int}
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
    max_memory::Union{Nothing,Int} = nothing,
)
    λ_1 >= 0.0 || throw(ArgumentError("λ_1 must be non-negative, got $λ_1"))
    λ_2 >= 0.0 || throw(ArgumentError("λ_2 must be non-negative, got $λ_2"))
    ρ > 0.0 || throw(ArgumentError("ρ must be positive, got $ρ"))
    max_memory === nothing || max_memory > 0 ||
        throw(ArgumentError("max_memory must be positive, got $max_memory"))
    T = dtype
    ADMMSLIM{T}(T(λ_1), T(λ_2), T(ρ), max_iter, T(convergence_tol), nonneg, verbose,
                 max_memory, spzeros(T, 0, 0), false)
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
    old_W = model.W
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try
    _require_nonempty_dimensions(X, "ADMMSLIM")
    _require_finite_input(X, "ADMMSLIM")

    # Peak fit memory: G, lhs, Cholesky factor, B, Z, U — six dense n_items²
    # matrices (the fitted W is built afterwards as sparse).
    _require_fit_memory(_fit_memory_estimate(n_items, 6, T), model.max_memory, "ADMMSLIM")

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

    # Initialize ADMM variables. Use `fill` (not `zeros`): JET's abstract
    # interpretation of `zeros(T, n, n)` under the free typevar T widens the
    # slot to a union with `Array{Float64,3}`, which then trips the `sparse`
    # conversion below. `fill` infers cleanly and is semantically identical.
    B = fill(zero(T), n_items, n_items)
    Z = fill(zero(T), n_items, n_items)
    U = fill(zero(T), n_items, n_items)  # scaled dual variable

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

    model.W = dropzeros!(sparse(Z))
    model.is_fitted = true

    nnz_w = nnz(model.W)
    density = nnz_w / (n_items * n_items) * 100
    model.verbose && @info "[ADMM-SLIM] Done. W: $(n_items)×$(n_items), nnz=$(nnz_w) ($(round(density; digits=3))%)"
    model
    catch
        model.W = old_W
        model.is_fitted = old_is_fitted
        rethrow()
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# recommend / score
# ──────────────────────────────────────────────────────────────────────────────

"""
    recommend(model::ADMMSLIM, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores = X * W, excluding seen items.

The scoring path is chosen adaptively: when `W` is sparse enough that the
score matrix `X * W` stays sparse, a sparse path is used (shared with
SLIM/ItemKNN); otherwise the densified `W` is scored through the
memory-bounded batched GEMM path. Both produce the same scores up to
floating-point accumulation order.
"""
function recommend(model::ADMMSLIM{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    if _use_sparse_score_path(model.W, X)
        _predict_sparse_score_topk(X * model.W, X, k)
    else
        _predict_batched_gemm_topk(X, Matrix(model.W), k)
    end
end

"""
    score(model::ADMMSLIM, X) -> SparseMatrixCSC

Return sparse score matrix S = X * W.
"""
function score(model::ADMMSLIM{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    size(X, 2) == size(model.W, 1) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.W, 1))"))
    X * model.W
end
