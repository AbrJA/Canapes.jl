# ──────────────────────────────────────────────────────────────────────────────
# EASE — Embarrassingly Shallow Autoencoders for Sparse Data
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Harald Steck (2019)
#   "Embarrassingly Shallow Autoencoders for Sparse Data" (WWW 2019)
#   arXiv:1905.03375
#
# EASE learns an item-item weight matrix B by solving:
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
    EASE{T} <: AbstractSparseModel

Embarrassingly Shallow Autoencoders (EASE^R) for collaborative filtering.

A closed-form linear model that learns an item-item similarity matrix B
with the constraint that diag(B) = 0 (items cannot recommend themselves).
Despite its simplicity, EASE consistently outperforms deep models on
standard benchmarks.

# Constructor
```julia
EASE(; λ=500.0)
```

# Fields
- `λ::T` — L2 regularization (higher = more smoothing, typical range: 100-1000)
- `B::Matrix{T}` — item-item weight matrix (n_items × n_items) after fitting

# Example
```julia
using SparseArrays, Gideon
X = sprand(1000, 500, 0.02)  # users × items
model = EASE(λ=200.0)
fit!(model, X)
preds = recommend(model, X; k=10)
```
"""
mutable struct EASE{T<:AbstractFloat} <: AbstractItemSimilarity
    const λ::T
    const verbose::Bool
    B::Matrix{T}
    is_fitted::Bool
end

function EASE(; λ::Float64=500.0, verbose::Bool=true, dtype::Type{<:AbstractFloat}=Float32)
    λ > 0.0 || throw(ArgumentError("λ must be positive, got $λ"))
    T = dtype
    EASE{T}(T(λ), verbose, Matrix{T}(undef, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::EASE, X) -> model

Compute the closed-form EASE solution on interaction matrix `X` (users × items).

Complexity: O(n_items² × n_users) for XᵀX, then O(n_items³) for the inverse.
Memory: O(n_items²) for the weight matrix B.
"""
function fit!(model::EASE{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng()) where {T,Tv,Ti}
    old_B = model.B
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "EASE")

    model.verbose && @info "[EASE] Computing Gram matrix ($(n_items) items)..."

    # G = XᵀX + λI
    G = Matrix{T}(X' * X)
    @inbounds for i in 1:n_items
        G[i, i] += model.λ
    end

    model.verbose && @info "[EASE] Computing inverse via Cholesky ($(n_items)×$(n_items))..."

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

    model.verbose && @info "[EASE] Fitted. B matrix: $(n_items)×$(n_items)"
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
    recommend(model::EASE, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores are computed as X * B.
Already-interacted items are excluded.
"""
function recommend(model::EASE{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    n_users = size(X, 1)
    n_items = size(model.B, 1)
    k_out = min(k, n_items)

    # Batched GEMM: convert sparse X to dense in chunks, multiply via BLAS,
    # then extract top-k from each batch's column-major score buffer.
    # This avoids Julia's single-threaded sparse×dense and leverages multi-threaded BLAS.
    preds = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)

    # Batch sizing: target ≤ 2 GB for the dense X chunk + score buffer
    max_batch_mem = 2 * 1024^3
    batch_size = max(1, min(n_users, Int(floor(max_batch_mem / (2 * n_items * sizeof(T))))))

    nt = Threads.nthreads()
    topk_bufs = _thread_buffers(() -> Vector{Int}(undef, k_out), nt)
    # Score buffer: (n_items × batch_size) — column per user for contiguous top-k
    scores_buf = Matrix{T}(undef, n_items, batch_size)
    # Dense X batch buffer: (batch_size × n_items) for GEMM input
    X_batch = Matrix{T}(undef, batch_size, n_items)

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        n_batch = batch_end - batch_start + 1
        batch_users = batch_start:batch_end

        # Convert sparse chunk to dense
        Xb = @view X_batch[1:n_batch, :]
        fill!(Xb, zero(T))
        @inbounds for u_local in 1:n_batch
            u = batch_start + u_local - 1
            for idx in nzrange(X_csr, u)
                j = Int(X_csr.colval[idx])
                Xb[u_local, j] = T(X_csr.nzval[idx])
            end
        end

        # GEMM: S[:,1:n_batch] = B' * Xb' → (n_items × n_batch)
        Sb = @view scores_buf[:, 1:n_batch]
        mul!(Sb, model.B', Xb')

        # Mask and top-k (threaded, chunked)
        Threads.@threads for chunk in 1:nt
            topk = topk_bufs[chunk]
            @inbounds for u_local in _thread_chunk_bounds(chunk, n_batch, nt)
                u = batch_start + u_local - 1
                for idx in nzrange(X_csr, u)
                    j = Int(X_csr.colval[idx])
                    scores_buf[j, u_local] = T(-Inf)
                end
                col = @view scores_buf[:, u_local]
                _topk_indices!(topk, col, k_out)
                for i in 1:k_out
                    preds[u, i] = topk[i]
                end
            end
        end
    end
    preds
end

"""
    score(model::EASE, X) -> Matrix{T}

Return the full score matrix S = X * B (dense, n_users × n_items).
"""
function score(model::EASE{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    Matrix{T}(X * model.B)
end
