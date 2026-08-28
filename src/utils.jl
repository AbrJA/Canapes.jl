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
    link_function(family::Family, x)

Apply the GLM link function for the given family:
- `Binomial()` → sigmoid(x)
- `Gaussian()` → x (identity)
- `Poisson()` → exp(x)
"""
@inline link_function(::Binomial, x::T) where {T<:AbstractFloat} = sigmoid(x)
@inline link_function(::Gaussian, x::T) where {T<:AbstractFloat} = x
@inline link_function(::Poisson, x::T) where {T<:AbstractFloat} = exp(x)

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
    nt = Threads.maxthreadid()
    topk_bufs = [Vector{Int}(undef, k_actual) for _ in 1:nt]
    scores_buf = Matrix{T}(undef, n_items, batch_size)

    # Use multi-threaded BLAS for the large GEMM
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(Threads.nthreads())

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        batch_users = batch_start:batch_end
        n_batch = length(batch_users)

        # GEMM: scores_buf[:,1:n_batch] = item_factors' * user_factors[:,batch_users]
        scores = @view scores_buf[:, 1:n_batch]
        mul!(scores, item_factors', @view(user_factors[:, batch_users]))

        # Mask seen items and extract top-k per user (threaded)
        Threads.@threads for local_u in 1:n_batch
            tid = Threads.threadid()
            global_u = batch_users[local_u]
            @inbounds for idx in nzrange(X_csr, global_u)
                j = Int(X_csr.colval[idx])
                scores_buf[j, local_u] = T(-Inf)
            end
            col = @view scores_buf[:, local_u]
            topk = topk_bufs[tid]
            _topk_indices!(topk, col, k_actual)
            @inbounds for i in 1:k_actual
                predictions[global_u, i] = topk[i]
            end
        end
    end

    BLAS.set_num_threads(old_blas)
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
        @simd for f in 1:k
            s += user_factors[f, u] * item_factors[f, i]
        end
        scores[idx] = s
    end
    scores
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

function recommend(model::AbstractMatrixFactorization, X::SparseMatrixCSC; k::Int=10)
    model.is_fitted || error("Model not fitted")
    _validate_recommend_input(X, size(model.item_factors, 2), k)
    _predict_topk_batched(model.user_factors, model.item_factors, to_csr(X), k)
end

function score(model::AbstractMatrixFactorization, X::SparseMatrixCSC)
    model.is_fitted || error("Model not fitted")
    size(X, 2) == size(model.item_factors, 2) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.item_factors, 2))"))
    model.user_factors' * model.item_factors
end

function score(model::AbstractMatrixFactorization,
               user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer})
    model.is_fitted || error("Model not fitted")
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
            @simd for f in 1:rank
                dot_val += factors[f, query_id] * factors[f, j]
                col_norm += factors[f, j]^2
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
    hasproperty(model, :is_fitted) && model.is_fitted || error("Model not fitted")
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
    hasproperty(model, :is_fitted) && model.is_fitted || error("Model not fitted")
    factors = if hasproperty(model, :user_factors)
        model.user_factors
    else
        embeddings(model)
    end
    _cosine_topk(factors, user_id, k)
end
