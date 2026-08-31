# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers shared across algorithms
# ──────────────────────────────────────────────────────────────────────────────

"""
    init_factors(rng, rank, n; scale=0.01)

Initialize a `rank × n` dense factor matrix with small random values drawn
from a normal distribution N(0, `scale`²).
"""
function init_factors(rng::AbstractRNG, rank::Int, n::Int; scale::Float64=0.01)
    randn(rng, rank, n) .* scale
end

"""
    sigmoid(x)

Numerically stable logistic sigmoid: σ(x) = 1/(1+exp(-x)).
"""
@inline function sigmoid(x::T) where {T<:AbstractFloat}
    if x >= zero(T)
        z = exp(-x)
        return one(T) / (one(T) + z)
    else
        z = exp(x)
        return z / (one(T) + z)
    end
end

"""
    log1pexp(x)

Compute `log(1 + exp(x))` in a numerically stable way (softplus).
"""
@inline function log1pexp(x::T) where {T<:AbstractFloat}
    if x > T(33.3)
        return x
    elseif x > T(-33.3)
        return log1p(exp(x))
    else
        return exp(x)
    end
end

"""
    safe_inv(x; ε=1e-12)

Safe reciprocal that avoids division by zero.
"""
@inline safe_inv(x::T; ε::T=T(1e-12)) where {T<:AbstractFloat} = one(T) / (x + ε)

"""
    link_function(family::LossFamily, x)

Apply the GLM link function for the given family:
- `LossFamilies.Binomial()` → sigmoid(x)
- `LossFamilies.Gaussian()` → x (identity)
- `LossFamilies.Poisson()` → exp(x)
"""
@inline link_function(::LossFamilies.Binomial, x::T) where {T<:AbstractFloat} = sigmoid(x)
@inline link_function(::LossFamilies.Gaussian, x::T) where {T<:AbstractFloat} = x
@inline link_function(::LossFamilies.Poisson, x::T) where {T<:AbstractFloat} = exp(x)

"""
    _inplace_shuffle!(v, rng) -> v

Fisher-Yates in-place shuffle — O(n) time, zero allocations beyond the vector itself.
"""
function _inplace_shuffle!(v::AbstractVector, rng::AbstractRNG)
    n = length(v)
    @inbounds for i in n:-1:2
        j = rand(rng, 1:i)
        v[i], v[j] = v[j], v[i]
    end
    v
end

"""
Binary search in a sorted Int32 vector. O(log n) and cache-friendly.
Used by BPR and LogisticMF for negative sampling.
"""
@inline function _insorted(sorted::Vector{Int32}, val::Int32)
    lo, hi = 1, length(sorted)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        if sorted[mid] < val
            lo = mid + 1
        elseif sorted[mid] > val
            hi = mid - 1
        else
            return true
        end
    end
    return false
end

"""
    _thread_chunk_bounds(chunk, n, nt) -> UnitRange{Int}

Return the contiguous range of `1:n` assigned to chunk `chunk` (1 ≤ chunk ≤ nt).
Chunks are as equal in size as possible. This is the canonical partitioning used
by all threaded sweeps: `Threads.@threads for chunk in 1:nt` with per-chunk
buffers indexed by `chunk` (never by `Threads.threadid()`, which can exceed
`maxthreadid()` and is non-deterministic under dynamic scheduling).
"""
@inline function _thread_chunk_bounds(chunk::Int, n::Int, nt::Int)
    chunk_size = cld(n, nt)
    start = (chunk - 1) * chunk_size + 1
    stop = min(chunk * chunk_size, n)
    start:stop
end

"""
    _thread_buffers(f, nt) -> Vector

Allocate `nt` per-chunk work buffers by calling `f()` once per chunk.
"""
function _thread_buffers(f::Function, nt::Int)
    [f() for _ in 1:nt]
end

"""
    _topk_indices!(topk, scores, k)

Find indices of the `k` largest elements in `scores`, stored in `topk[1:k]`
in descending order. Single O(n) pass, zero allocations.
"""
@inline function _topk_indices!(topk::AbstractVector{Int}, scores::AbstractVector{T}, k::Int) where T
    n = length(scores)
    # Initialize with first k indices, insertion-sorted descending
    @inbounds for i in 1:k
        topk[i] = i
    end
    @inbounds for i in 2:k
        idx = topk[i]
        val = scores[idx]
        j = i - 1
        while j >= 1 && scores[topk[j]] < val
            topk[j + 1] = topk[j]
            j -= 1
        end
        topk[j + 1] = idx
    end
    @inbounds threshold = scores[topk[k]]
    # Single pass: maintain sorted top-k
    @inbounds for i in (k + 1):n
        s = scores[i]
        if s > threshold
            j = k - 1
            while j >= 1 && scores[topk[j]] < s
                topk[j + 1] = topk[j]
                j -= 1
            end
            topk[j + 1] = i
            threshold = scores[topk[k]]
        end
    end
    nothing
end

"""
    _predict_topk_batched(user_factors, item_factors, X_csr, k) -> Matrix{Int}

Shared batched top-k prediction for bilinear matrix factorization models.
Computes scores via GEMM in memory-bounded batches, masks seen items, and
selects top-k per user using threaded partial sort.

Returns an `n_users × k` matrix of recommended item indices (1-based).
"""
function _predict_topk_batched(user_factors::Matrix{T}, item_factors::Matrix{T},
                               X_csr::SparseMatrixCSR, k::Int) where {T}
    n_users = size(user_factors, 2)
    n_items = size(item_factors, 2)
    k_actual = min(k, n_items)

    predictions = Matrix{Int}(undef, n_users, k_actual)

    # Batch sizing: target ≤ 2 GB for score buffer
    max_batch_mem = 2 * 1024^3
    batch_size = max(1, min(n_users, Int(floor(max_batch_mem / (n_items * sizeof(T))))))

    # Per-thread top-k buffers
    nt = Threads.nthreads()
    topk_bufs = [Vector{Int}(undef, k_actual) for _ in 1:nt]
    scores_buf = Matrix{T}(undef, n_items, batch_size)

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        batch_users = batch_start:batch_end
        n_batch = length(batch_users)

        # GEMM: scores_buf[:,1:n_batch] = item_factors' * user_factors[:,batch_users]
        scores = @view scores_buf[:, 1:n_batch]
        mul!(scores, item_factors', @view(user_factors[:, batch_users]))

        # Mask seen items and extract top-k per user (threaded, chunked)
        Threads.@threads for chunk in 1:nt
            topk = topk_bufs[chunk]
            @inbounds for local_u in _thread_chunk_bounds(chunk, n_batch, nt)
                global_u = batch_users[local_u]
                for idx in nzrange(X_csr, global_u)
                    j = Int(X_csr.colval[idx])
                    scores_buf[j, local_u] = T(-Inf)
                end
                col = @view scores_buf[:, local_u]
                _topk_indices!(topk, col, k_actual)
                for i in 1:k_actual
                    predictions[global_u, i] = topk[i]
                end
            end
        end
    end

    predictions
end

"""
    _predict_pairwise_scores(user_factors, item_factors, user_indices, item_indices) -> Vector

Compute scores for specific (user, item) pairs via inner products.
Shared implementation for bilinear MF models.
"""
function _predict_pairwise_scores(user_factors::Matrix{T}, item_factors::Matrix{T},
                                  user_indices::AbstractVector{<:Integer},
                                  item_indices::AbstractVector{<:Integer}) where {T}
    length(user_indices) == length(item_indices) ||
        throw(ArgumentError("user_indices and item_indices must have the same length"))
    n = length(user_indices)
    k = size(user_factors, 1)
    scores = Vector{T}(undef, n)
    @inbounds for idx in 1:n
        u = user_indices[idx]
        i = item_indices[idx]
        s = zero(T)
        for f in 1:k
            s = muladd(user_factors[f, u], item_factors[f, i], s)
        end
        scores[idx] = s
    end
    scores
end

"""
    _use_sparse_score_path(W, X) -> Bool

Choose between the sparse-score and the dense batched-GEMM top-k paths for
item-similarity models whose fitted weights `W` are stored sparse (ADMMSLIM).
Both paths compute the same scores up to floating-point accumulation order;
the choice minimizes work.

The dense GEMM path wins whenever the score matrix `S = X * W` is expected to
be dense, which happens even for moderately sparse `W` when users interact
with many items. Estimate the per-entry fill probability of `S` as
`P = 1 - (1 - d_W)^k`, where `d_W` is the weight density and `k` the mean
items per user, and prefer sparse scoring only when `S` is expected to stay
sparse (`P ≤ 0.1`).
"""
function _use_sparse_score_path(W::SparseMatrixCSC, X::SparseMatrixCSC)
    n_items = size(W, 1)
    n_users = size(X, 1)
    (n_items == 0 || n_users == 0) && return true
    d_w = nnz(W) / (n_items * n_items)
    k = nnz(X) / n_users
    # S[u, j] ≠ 0 iff some item i observed by u has W[i, j] ≠ 0
    p = 1 - (1 - d_w)^k
    p <= 0.1
end

"""
    _predict_sparse_score_topk(S, X, k) -> Matrix{Int}

Shared top-k recommendation for sparse-score item-similarity models
(SLIM, ItemKNN). Scores `S` (typically `X * W`) are scattered per user into a
dense buffer, seen items from `X` are masked, and the top-k is extracted with
a threaded partial sort. Returns an `n_users × k` matrix of item indices.
"""
function _predict_sparse_score_topk(S::SparseMatrixCSC{Tv,Ti},
                                    X::SparseMatrixCSC,
                                    k::Int) where {Tv,Ti}
    T = eltype(S)
    n_users = size(X, 1)
    n_items = size(S, 2)
    k_out = _validate_recommend_input(X, n_items, k)

    S_csr = to_csr(S)
    X_csr = to_csr(X)
    preds = Matrix{Int}(undef, n_users, k_out)

    nt = Threads.nthreads()
    topk_bufs = _thread_buffers(() -> Vector{Int}(undef, k_out), nt)
    score_bufs = _thread_buffers(() -> zeros(T, n_items), nt)

    Threads.@threads for chunk in 1:nt
        scores = score_bufs[chunk]
        topk = topk_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n_users, nt)
            @inbounds @simd for i in 1:n_items
                scores[i] = zero(T)
            end

            @inbounds for idx in nzrange(S_csr, u)
                j = Int(S_csr.colval[idx])
                scores[j] = S_csr.nzval[idx]
            end

            # Mask seen items
            @inbounds for idx in nzrange(X_csr, u)
                j = Int(X_csr.colval[idx])
                scores[j] = T(-Inf)
            end

            _topk_indices!(topk, scores, k_out)
            @inbounds for i in 1:k_out
                preds[u, i] = topk[i]
            end
        end
    end
    preds
end

"""
    _predict_batched_gemm_topk(X, W, k) -> Matrix{Int}

Shared memory-bounded GEMM top-k recommendation for dense item-similarity
models (EASE, ADMMSLIM). The sparse user-item matrix is densified in batches
(target ≤ 2 GB for the dense chunk + score buffer), scored via BLAS GEMM,
seen items masked, and the top-k per user extracted with a threaded partial
sort. Keeps the full-score and top-k paths separate so huge matrices never
materialize a full dense score matrix.
"""
function _predict_batched_gemm_topk(X::SparseMatrixCSC{Tv,Ti},
                                    W::Matrix{T},
                                    k::Int) where {Tv,Ti,T}
    n_users = size(X, 1)
    n_items = size(W, 1)
    k_out = _validate_recommend_input(X, n_items, k)

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

        # GEMM: S[:,1:n_batch] = W' * Xb' → (n_items × n_batch)
        Sb = @view scores_buf[:, 1:n_batch]
        mul!(Sb, W', Xb')

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

# ──────────────────────────────────────────────────────────────────────────────
# Default recommend/score for AbstractMatrixFactorization
# ──────────────────────────────────────────────────────────────────────────────
# Models with user_factors/item_factors get these for free.
# Override only when special logic is needed (e.g. WMF transform, GloVe embeddings).

@inline function _validate_recommend_input(X::SparseMatrixCSC, n_items::Int, k::Int)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    size(X, 2) == n_items || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $n_items"))
    min(k, n_items)
end

@inline function _require_nonempty_dimensions(X::SparseMatrixCSC, algorithm::AbstractString)
    all(>(0), size(X)) || throw(ArgumentError(
        "$algorithm requires positive user and item dimensions, got $(size(X))"))
    nothing
end

@inline function _require_finite_input(X::SparseMatrixCSC, name::AbstractString)
    all(isfinite, nonzeros(X)) || throw(ArgumentError(
        "$name input contains NaN or Inf values; all values must be finite"))
    nothing
end

@inline function _require_finite_vector(y::AbstractVector{<:Real}, name::AbstractString)
    all(isfinite, y) || throw(ArgumentError(
        "$name target values contain NaN or Inf; all values must be finite"))
    nothing
end

@inline function _require_fitted(fitted::Bool)
    fitted || throw(ArgumentError("model is not fitted; call fit! first"))
    nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# Fit-time memory estimation
# ──────────────────────────────────────────────────────────────────────────────

"""
    _fit_memory_estimate(n_items::Int, n_dense_matrices::Int, ::Type{T}) -> Int

Rough peak-memory estimate (bytes) for training an item-item model that keeps
`n_dense_matrices` dense n_items × n_items matrices of element type `T` in
memory. Sparse intermediates (e.g. the XᵀX product) are bounded by n_items² as
well, so the estimate captures the dominant allocations.
"""
@inline _fit_memory_estimate(n_items::Int, n_dense_matrices::Int, ::Type{T}) where {T} =
    n_items * n_items * n_dense_matrices * sizeof(T)

"""
    _require_fit_memory(estimate::Int, limit::Union{Nothing,Int}, name::AbstractString)

Throw `ArgumentError` when the estimated fit-time peak memory `estimate`
(bytes) exceeds the configured `limit` (bytes). A `nothing` limit allows any
size. Called before the first large allocation so a fit that cannot fit in
memory fails early instead of exhausting the system.
"""
function _require_fit_memory(estimate::Int, limit::Union{Nothing,Int}, name::AbstractString)
    if limit !== nothing && estimate > limit
        throw(ArgumentError(
            "$name fit is estimated to need ≈$(round(estimate / 2^20; digits=1)) MiB of peak " *
            "memory ($estimate bytes), exceeding max_memory=$limit bytes. " *
            "Raise max_memory or fit on fewer items."))
    end
    nothing
end

function recommend(model::AbstractMatrixFactorization, X::SparseMatrixCSC; k::Int=10)
    _require_fitted(model.is_fitted)
    _validate_recommend_input(X, size(model.item_factors, 2), k)
    _predict_topk_batched(model.user_factors, model.item_factors, to_csr(X), k)
end

function score(model::AbstractMatrixFactorization, X::SparseMatrixCSC)
    _require_fitted(model.is_fitted)
    size(X, 2) == size(model.item_factors, 2) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.item_factors, 2))"))
    model.user_factors' * model.item_factors
end

function score(model::AbstractMatrixFactorization,
               user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer})
    _require_fitted(model.is_fitted)
    _predict_pairwise_scores(model.user_factors, model.item_factors, user_indices, item_indices)
end

# ──────────────────────────────────────────────────────────────────────────────
# Similarity queries
# ──────────────────────────────────────────────────────────────────────────────

"""
    _cosine_topk(factors, query_id, k) -> (ids, scores)

Find the k most similar columns to `query_id` in `factors` by cosine similarity.
Excludes the query itself from results.
"""
function _cosine_topk(factors::Matrix{T}, query_id::Int, k::Int) where {T}
    rank, n = size(factors)
    query_id >= 1 && query_id <= n ||
        throw(ArgumentError("query_id=$query_id out of range [1, $n]"))
    k_out = min(k, n - 1)

    # Normalize the query vector
    q = @view factors[:, query_id]
    q_norm = norm(q)
    q_norm > zero(T) || return (Int[], T[])

    # Compute cosine similarities
    sims = Vector{T}(undef, n)
    @inbounds for j in 1:n
        if j == query_id
            sims[j] = T(-Inf)  # exclude self
        else
            col_norm = zero(T)
            dot_val = zero(T)
            for f in 1:rank
                dot_val = muladd(factors[f, query_id], factors[f, j], dot_val)
                col_norm = muladd(factors[f, j], factors[f, j], col_norm)
            end
            col_norm = sqrt(col_norm)
            sims[j] = col_norm > zero(T) ? dot_val / (q_norm * col_norm) : zero(T)
        end
    end

    # Top-k extraction
    topk = Vector{Int}(undef, k_out)
    _topk_indices!(topk, sims, k_out)
    scores_out = T[sims[topk[i]] for i in 1:k_out]
    (topk, scores_out)
end

"""
    similar_items(model::AbstractMatrixFactorization, item_id; k=10)

Find the k most similar items to `item_id` based on cosine similarity
of item embedding vectors. Returns `(ids::Vector{Int}, scores::Vector)`.
"""
function similar_items(model::AbstractMatrixFactorization, item_id::Int; k::Int=10)
    hasproperty(model, :is_fitted) && _require_fitted(model.is_fitted)
    factors = if hasproperty(model, :item_factors)
        model.item_factors
    else
        embeddings(model)
    end
    _cosine_topk(factors, item_id, k)
end

"""
    similar_users(model::AbstractMatrixFactorization, user_id; k=10)

Find the k most similar users to `user_id` based on cosine similarity
of user embedding vectors. Returns `(ids::Vector{Int}, scores::Vector)`.
"""
function similar_users(model::AbstractMatrixFactorization, user_id::Int; k::Int=10)
    hasproperty(model, :is_fitted) && _require_fitted(model.is_fitted)
    factors = if hasproperty(model, :user_factors)
        model.user_factors
    else
        embeddings(model)
    end
    _cosine_topk(factors, user_id, k)
end
