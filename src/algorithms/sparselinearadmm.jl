# ──────────────────────────────────────────────────────────────────────────────
# ADMM-SparseLinearModel — ADMM-based Sparse Linear Methods
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Steck, Liang, et al. (2020)
#   "ADMM SparseLinearModel: Sparse Recommendations for Many Users"
#   (WSDM 2020) — arXiv:2003.04710
#
# Instead of solving n_items independent elastic-net problems (standard SparseLinearModel),
# ADMM-SparseLinearModel solves the full item-item weight matrix jointly using ADMM:
#
#   min_B  ½‖X - XB‖²_F + λ_l1‖B‖₁ + λ_l2/2‖B‖²_F
#   s.t.   diag(B) = 0
#
# This is equivalent to SparseLinearModel but 10-100× faster because:
# 1. Computes the Gram matrix G = XᵀX once (dominant cost)
# 2. Pre-factors (G + ρI)⁻¹ via Cholesky once
# 3. Iterates ADMM updates (matrix-level, not per-column)
#
# The result interpolates between ShallowAutoencoder (λ_l1=0) and SparseLinearModel (λ_l1>0).
# ──────────────────────────────────────────────────────────────────────────────

"""
    SparseLinearADMM{T} <: AbstractItemSimilarity

ADMM-based Sparse Linear Methods for top-N recommendation.

A dramatically faster alternative to standard SparseLinearModel that solves the full
item-item weight matrix jointly using ADMM. Produces the same solution
as coordinate-descent SparseLinearModel but in 10-100× less time.

# Constructor
```julia
SparseLinearADMM(; λ_l1=0.01, λ_l2=100.0, ρ=1.0, max_iter=50, tol=1e-4,
            nonnegative=true, verbose=true, max_memory=nothing)
```

# Fields
- `λ_l1::T` — L1 penalty (sparsity inducing, via soft-thresholding)
- `λ_l2::T` — L2 penalty (shrinkage / regularization)
- `ρ::T` — ADMM penalty parameter (controls convergence speed)
- `max_iter::Int` — max ADMM iterations
- `tol::T` — relative primal-residual and solution-drift tolerance (both must be met)
- `nonnegative::Bool` — enforce non-negative weights
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
below 100% and shrinks as λ_l1 grows), and `score` returns a sparse matrix.
`recommend` picks the scoring path adaptively (sparse when the score matrix
stays sparse, memory-bounded batched GEMM otherwise).

Prefer SparseLinearModel when `n_items` is large enough that an O(n_items²) dense solve
is prohibitive (SparseLinearModel fits per-item in O(n_items) memory per column), or
when the training set is small enough that SparseLinearModel's slower per-item
coordinate descent is affordable.

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> model = SparseLinearADMM(λ_l1=0.05, λ_l2=200.0, max_iter=5, verbose=false);

julia> fit!(model, X; rng=MersenneTwister(2));

julia> preds = recommend(model, X; k=10);

julia> size(preds)
(200, 10)
```
"""
mutable struct SparseLinearADMM{T<:AbstractFloat} <: AbstractItemSimilarity
    const λ_l1::T
    const λ_l2::T
    const ρ::T
    const max_iter::Int
    const tol::T
    const nonnegative::Bool
    const verbose::Bool
    const max_memory::Union{Nothing,Int}
    W::SparseMatrixCSC{T,Int}
    is_fitted::Bool
end

# Dense n_items² working matrices during fit!: G, lhs, Cholesky factor, B,
# Z, Z_prev, U (the fitted W is built afterwards as sparse).
const _ADMMSLIM_DENSE_MATRICES = 7

function SparseLinearADMM(;
    λ_l1::Float64 = 0.01,
    λ_l2::Float64 = 100.0,
    ρ::Float64 = 1.0,
    max_iter::Int = 50,
    tol::Float64 = 1e-4,
    nonnegative::Bool = true,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
    max_memory::Union{Nothing,Int} = nothing,
)
    λ_l1 >= 0.0 || throw(ArgumentError("λ_l1 must be non-negative, got $λ_l1"))
    λ_l2 >= 0.0 || throw(ArgumentError("λ_l2 must be non-negative, got $λ_l2"))
    ρ > 0.0 || throw(ArgumentError("ρ must be positive, got $ρ"))
    max_memory === nothing || max_memory > 0 ||
        throw(ArgumentError("max_memory must be positive, got $max_memory"))
    SparseLinearADMM{T}(T(λ_l1), T(λ_l2), T(ρ), max_iter, T(tol), nonnegative, verbose,
                 max_memory, spzeros(T, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::SparseLinearADMM, X; rng) -> model

Fit ADMM-SparseLinearModel on interaction matrix `X` (users × items).

Computes the Gram matrix once, pre-factors `(G + (λ_l2+ρ)I)`, then iterates ADMM:
- B-update: solve linear system (pre-factored Cholesky)
- Z-update: soft-thresholding (proximal L1) + optional non-negativity
- U-update: dual variable accumulation

Convergence is monitored on the primal residual (`‖B-Z‖`) and on the solution
drift (`‖Z-Z_prev‖/‖Z‖`); the drift criterion makes early stopping actually fire
(the raw ADMM dual residual `ρ‖Z-Z_prev‖` grows with ρ and never meets a fixed
tolerance, so it is not used as a stopping metric). The penalty `ρ` is fixed —
the converged ADMM solution is independent of it, so adapting it mid-run would
only move the iterate away from the reference fixed point without changing the
answer.

The B-update solves the dense positive-definite system `(G + (λ₂+ρ)I) B = rhs`
directly via the pre-factored Cholesky. This is intentional: the Gram matrix
`G = XᵀX` of typical implicit interaction data is nearly dense (99.9% for
MovieLens-1M), so an iterative per-column solver (CG/PCG) would cost
`nnz(G)·cg_iters` per column — 5-10× more than the direct substitution at the
scales where SparseLinearADMM is usable. SparseLinearModel (coordinate descent, per-item) and ShallowAutoencoder
(one-shot dense solve) remain the preferred models at movie-scale item counts.
"""
function fit!(model::SparseLinearADMM{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    λ_l1 = model.λ_l1
    λ_l2 = model.λ_l2
    ρ = model.ρ
    old_W = model.W
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try
    _require_nonempty_dimensions(X, "SparseLinearADMM")
    _require_finite_input(X, "SparseLinearADMM")

    # Peak fit memory: G, lhs, Cholesky factor, B, Z, Z_prev, U — seven dense
    # n_items² matrices (the fitted W is built afterwards as sparse).
    _require_fit_memory(_fit_memory_estimate(n_items, _ADMMSLIM_DENSE_MATRICES, T),
                        model.max_memory, "SparseLinearADMM")

    model.verbose && @info "[ADMM-SparseLinearModel] Computing Gram matrix ($n_items × $n_items)..."

    # Gram matrix: G = XᵀX
    G = Matrix{T}(X' * X)

    # Pre-factor: (G + (λ_l2 + ρ)I)⁻¹ via Cholesky
    # The B-update solves: (G + (λ_l2+ρ)I) B = G + ρ(Z - U)
    # Pre-factor the LHS
    lhs = copy(G)
    @inbounds for j in 1:n_items
        lhs[j, j] += λ_l2 + ρ
    end
    C = cholesky(Symmetric(lhs))

    model.verbose && @info "[ADMM-SparseLinearModel] Running ADMM ($n_items items, max_iter=$(model.max_iter))..."

    # Initialize ADMM variables. Use `fill` (not `zeros`): JET's abstract
    # interpretation of `zeros(T, n, n)` under the free typevar T widens the
    # slot to a union with `Array{Float64,3}`, which then trips the `sparse`
    # conversion below. `fill` infers cleanly and is semantically identical.
    B = fill(zero(T), n_items, n_items)
    Z = fill(zero(T), n_items, n_items)
    U = fill(zero(T), n_items, n_items)  # scaled dual variable
    Z_prev = fill(zero(T), n_items, n_items)  # for the dual residual

    for iter in 1:model.max_iter
        copyto!(Z_prev, Z)

        # ── B-update: B = (G + (λ₂+ρ)I)⁻¹ (G + ρ(Z - U)) ──
        rhs = G .+ ρ .* (Z .- U)
        B .= C \ rhs

        # Enforce diag(B) = 0
        @inbounds for j in 1:n_items
            B[j, j] = zero(T)
        end

        # ── Z-update: proximal operator (soft-threshold + optional non-neg) ──
        B_plus_U = B .+ U
        threshold = λ_l1 / ρ

        if model.nonnegative
            # Soft-threshold + clip to non-negative
            @inbounds @simd for idx in eachindex(Z)
                z = B_plus_U[idx] - threshold
                Z[idx] = z > zero(T) ? z : zero(T)
            end
        else
            # Standard soft-thresholding
            @inbounds @simd for idx in eachindex(Z)
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

        # ── Residuals: primal ‖B-Z‖; drift ‖Z-Z_prev‖ (the raw dual residual
        # ρ‖Z-Z_prev‖ grows with ρ and never meets a fixed tol, so the drift —
        # without the ρ factor — is used as the stopping metric) ──
        primal_resid = zero(T)
        dual_resid = zero(T)
        norm_B = zero(T)
        norm_Z = zero(T)
        @inbounds @simd for idx in eachindex(B)
            db = B[idx] - Z[idx]
            primal_resid += db * db
            dz = Z[idx] - Z_prev[idx]
            dual_resid += dz * dz
            norm_B += B[idx] * B[idx]
            norm_Z += Z[idx] * Z[idx]
        end
        primal_rel = sqrt(primal_resid) / (sqrt(norm_B) + T(1e-12))
        drift_rel = sqrt(dual_resid) / (sqrt(norm_Z) + T(1e-12))

        if model.verbose && (iter <= 5 || iter % 10 == 0 || iter == model.max_iter)
            nnz_iter = count(>(T(1e-10)), Z)
            @info "[ADMM-SparseLinearModel] iter=$iter  primal=$(round(primal_rel; sigdigits=4))  drift=$(round(drift_rel; sigdigits=4))  ρ=$(round(ρ; sigdigits=3))  nnz=$(nnz_iter)"
        end

        if primal_rel < model.tol && drift_rel < model.tol
            model.verbose && @info "[ADMM-SparseLinearModel] Converged at iteration $iter (primal=$(round(primal_rel; sigdigits=4)), drift=$(round(drift_rel; sigdigits=4)))"
            break
        end
    end

    model.W = dropzeros!(sparse(Z))
    model.is_fitted = true

    nnz_w = nnz(model.W)
    density = nnz_w / (n_items * n_items) * 100
    model.verbose && @info "[ADMM-SparseLinearModel] Done. W: $(n_items)×$(n_items), nnz=$(nnz_w) ($(round(density; digits=3))%)"
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
    recommend(model::SparseLinearADMM, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores = X * W, excluding seen items.

The scoring path is chosen adaptively: when `W` is sparse enough that the
score matrix `X * W` stays sparse, a sparse path is used (shared with
SparseLinearModel/ItemKNN); otherwise the densified `W` is scored through the
memory-bounded batched GEMM path. Both produce the same scores up to
floating-point accumulation order.
"""
function recommend(model::SparseLinearADMM{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    if _use_sparse_score_path(model.W, X)
        _predict_sparse_score_topk(X, model.W, k)
    else
        _predict_batched_gemm_topk(X, Matrix(model.W), k)
    end
end

"""
    score(model::SparseLinearADMM, X) -> SparseMatrixCSC

Return sparse score matrix S = X * W.
"""
function score(model::SparseLinearADMM{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    size(X, 2) == size(model.W, 1) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.W, 1))"))
    X * model.W
end
