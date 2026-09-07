# ──────────────────────────────────────────────────────────────────────────────
# ShallowAutoencoder — Embarrassingly Shallow Autoencoders for Sparse Data
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Harald Steck (2019)
#   "Embarrassingly Shallow Autoencoders for Sparse Data" (WWW 2019)
#   arXiv:1905.03375
#
# ShallowAutoencoder learns an item-item weight matrix B by solving:
#   min_B ‖X - XB‖²_F + λ‖B‖²_F  subject to diag(B) = 0
#
# Closed-form solution:
#   P = (XᵀX + λI)⁻¹
#   B = I - P · diag(1/diag(P))
#
# This is a linear autoencoder that achieves state-of-the-art results on
# implicit feedback benchmarks, often outperforming deep models.
# ──────────────────────────────────────────────────────────────────────────────

"""
    ShallowAutoencoder{T} <: AbstractSparseModel

Embarrassingly Shallow Autoencoders (ShallowAutoencoder^R) for collaborative filtering.

A closed-form linear model that learns an item-item similarity matrix B
with the constraint that diag(B) = 0 (items cannot recommend themselves).
Despite its simplicity, ShallowAutoencoder consistently outperforms deep models on
standard benchmarks.

# Constructor
```julia
ShallowAutoencoder(; λ=500.0, max_memory=nothing)
```

# Fields
- `λ::T` — L2 regularization (higher = more smoothing, typical range: 100-1000)
- `max_memory::Union{Nothing,Int}` — fit-time peak-memory limit in bytes
  (`nothing` = unlimited); a fit whose estimated peak exceeds it throws
  `ArgumentError` before any large allocation
- `B::Matrix{T}` — item-item weight matrix (n_items × n_items) after fitting

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);  # users × items

julia> model = ShallowAutoencoder(λ=200.0, verbose=false);

julia> fit!(model, X);

julia> preds = recommend(model, X; k=10);

julia> size(preds)
(200, 10)
```
"""
mutable struct ShallowAutoencoder{T<:AbstractFloat} <: AbstractItemSimilarity
    const λ::T
    const verbose::Bool
    const max_memory::Union{Nothing,Int}
    B::Matrix{T}
    is_fitted::Bool
end

function ShallowAutoencoder(; λ::Float64=500.0, verbose::Bool=true, T::Type{<:AbstractFloat}=Float32,
              max_memory::Union{Nothing,Int}=nothing)
    λ > 0.0 || throw(ArgumentError("λ must be positive, got $λ"))
    max_memory === nothing || max_memory > 0 ||
        throw(ArgumentError("max_memory must be positive, got $max_memory"))
    ShallowAutoencoder{T}(T(λ), verbose, max_memory, Matrix{T}(undef, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::ShallowAutoencoder, X) -> model

Compute the closed-form ShallowAutoencoder solution on interaction matrix `X` (users × items).

Complexity: O(n_items² × n_users) for XᵀX, then O(n_items³) for the inverse.
Memory: O(n_items²) for the weight matrix B.
"""
function fit!(model::ShallowAutoencoder{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng()) where {T,Tv,Ti}
    old_B = model.B
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "ShallowAutoencoder")
    _require_finite_input(X, "ShallowAutoencoder")

    # Peak fit memory: G, Cholesky factor, P, B — four dense n_items² matrices
    _require_fit_memory(_fit_memory_estimate(n_items, 4, T), model.max_memory, "ShallowAutoencoder")

    model.verbose && @info "[ShallowAutoencoder] Computing Gram matrix ($(n_items) items)..."

    # G = XᵀX + λI
    G = Matrix{T}(X' * X)
    @inbounds for i in 1:n_items
        G[i, i] += model.λ
    end

    model.verbose && @info "[ShallowAutoencoder] Computing inverse via Cholesky ($(n_items)×$(n_items))..."

    # Use Cholesky factorization for numerical stability (G is SPD)
    C = cholesky(Symmetric(G))
    P = inv(C)

    # B = I - P · diag(1/diag(P))
    # Equivalent to: B_ij = -P_ij / P_jj for i≠j, B_ii = 0
    B = Matrix{T}(undef, n_items, n_items)
    @inbounds for j in 1:n_items
        inv_pjj = one(T) / P[j, j]
        for i in 1:n_items
            B[i, j] = -P[i, j] * inv_pjj
        end
        B[j, j] = zero(T)
    end

    model.B = B
    model.is_fitted = true

    model.verbose && @info "[ShallowAutoencoder] Fitted. B matrix: $(n_items)×$(n_items)"
    model
    catch
        model.B = old_B
        model.is_fitted = old_is_fitted
        rethrow()
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    recommend(model::ShallowAutoencoder, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores are computed as X * B.
Already-interacted items are excluded.
"""
function recommend(model::ShallowAutoencoder{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    # Batched GEMM via the shared memory-bounded path (see _predict_batched_gemm_topk)
    _predict_batched_gemm_topk(X, model.B, k)
end

"""
    score(model::ShallowAutoencoder, X) -> Matrix{T}

Return the full score matrix S = X * B (dense, n_users × n_items).
"""
function score(model::ShallowAutoencoder{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    Matrix{T}(X * model.B)
end
