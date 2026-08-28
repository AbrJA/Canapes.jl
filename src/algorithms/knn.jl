# ──────────────────────────────────────────────────────────────────────────────
# ItemKNN — Item-based K-Nearest Neighbors
# ──────────────────────────────────────────────────────────────────────────────
#
# A non-parametric item-based collaborative filtering model.
# Computes item-item similarity (cosine or Jaccard), retains top-k neighbors
# per item, and predicts scores via weighted combination of user history.
#
# Reference: Deshpande & Karypis (2004)
#   "Item-Based Top-N Recommendation Algorithms"
# ──────────────────────────────────────────────────────────────────────────────

"""
    ItemKNN{T} <: AbstractItemSimilarity

Item-based K-Nearest Neighbors recommender.

Computes item-item similarity from the interaction matrix, retains the top `k`
most similar items per column, and scores users via `X * W` where `W` is the
sparse truncated similarity matrix.

No training loop — similarity is computed in a single pass.

# Constructor
```julia
ItemKNN(; k=20, similarity=:cosine, shrinkage=0.0, normalize=true)
```

# Fields
- `k::Int` — number of neighbors to retain per item
- `similarity::Symbol` — `:cosine` or `:jaccard`
- `shrinkage::T` — additive shrinkage to denominator (regularizes rare items)
- `normalize::Bool` — row-normalize the similarity matrix (divide by row sum)
- `W::SparseMatrixCSC{T,Int}` — sparse item-item similarity matrix after fitting

# Example
```julia
using SparseArrays, Gideon
X = sprand(1000, 500, 0.02)
model = ItemKNN(k=50, similarity=:cosine)
fit!(model, X)
preds = recommend(model, X; k=10)
```
"""
mutable struct ItemKNN{T<:AbstractFloat} <: AbstractItemSimilarity
    const k::Int
    const similarity::Symbol
    const shrinkage::T
    const normalize::Bool
    const verbose::Bool
    W::SparseMatrixCSC{T,Int}
    is_fitted::Bool
end

function ItemKNN(;
    k::Int = 20,
    similarity::Symbol = :cosine,
    shrinkage::Float64 = 0.0,
    normalize::Bool = true,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    similarity in (:cosine, :jaccard) || throw(ArgumentError("similarity must be :cosine or :jaccard, got :$similarity"))
    shrinkage >= 0.0 || throw(ArgumentError("shrinkage must be non-negative, got $shrinkage"))
    T = dtype
    ItemKNN{T}(k, similarity, T(shrinkage), normalize, verbose,
               spzeros(T, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::ItemKNN, X; rng) -> model

Compute item-item similarity and retain top-k neighbors per item.
"""
function fit!(model::ItemKNN{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    old_W = model.W
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    _require_nonempty_dimensions(X, "ItemKNN")
    kn = min(model.k, n_items - 1)

    model.verbose && @info "[ItemKNN] Computing $(model.similarity) similarity for $n_items items (k=$kn)..."

    if model.similarity == :cosine
        W = _cosine_knn(X, kn, T(model.shrinkage))
    else  # :jaccard
        W = _jaccard_knn(X, kn, T(model.shrinkage))
    end

    # Optional row-normalization: each row sums to 1
    if model.normalize
        row_sums = vec(sum(W; dims=2))
        rv = rowvals(W)
        nz = nonzeros(W)
        @inbounds for col in axes(W, 2)
            for idx in nzrange(W, col)
                row = rv[idx]
                if row_sums[row] > zero(T)
                    nz[idx] /= row_sums[row]
                end
            end
        end
    end

    model.W = W
    model.is_fitted = true
    model.verbose && @info "[ItemKNN] Done. W has $(nnz(W)) nonzeros (density=$(round(nnz(W)/n_items^2; digits=6)))"
    model
    catch
        model.W = old_W
        model.is_fitted = old_is_fitted
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity with top-k truncation
# ──────────────────────────────────────────────────────────────────────────────

function _cosine_knn(X::SparseMatrixCSC{Tv,Ti}, k::Int, shrinkage::T) where {Tv,Ti,T}
    n_items = size(X, 2)

    # Column norms for cosine denominator
    col_norms = Vector{T}(undef, n_items)
    @inbounds for j in 1:n_items
        s = zero(T)
        for idx in nzrange(X, j)
            s += T(nonzeros(X)[idx])^2
        end
        col_norms[j] = sqrt(s)
    end

    # Gram matrix XᵀX
    G = X' * X  # SparseMatrixCSC

    # Build sparse W by keeping top-k per column (excluding self-similarity)
    # Use COO for construction
    nt = Threads.nthreads()
    # Thread-local storage
    local_rows = _thread_buffers(() -> Int[], nt)
    local_cols = _thread_buffers(() -> Int[], nt)
    local_vals = _thread_buffers(() -> T[], nt)

    Threads.@threads for chunk in 1:nt
        for j in _thread_chunk_bounds(chunk, n_items, nt)
        norm_j = col_norms[j]
        norm_j == zero(T) && continue

        # Collect similarities for column j from sparse Gram row
        sims = Vector{Pair{Int,T}}()
        for idx in nzrange(G, j)
            i = rowvals(G)[idx]
            i == j && continue
            norm_i = col_norms[i]
            norm_i == zero(T) && continue
            dot_ij = T(nonzeros(G)[idx])
            sim = dot_ij / (norm_i * norm_j + shrinkage)
            sim > zero(T) && push!(sims, i => sim)
        end

        # Keep top-k (partialsort is O(n) average vs O(n log n) for full sort)
        if length(sims) > k
            partialsort!(sims, 1:k; by=last, rev=true)
            resize!(sims, k)
        end

        for (i, s) in sims
            push!(local_rows[chunk], i)
            push!(local_cols[chunk], j)
            push!(local_vals[chunk], s)
        end
        end
    end

    all_rows = reduce(vcat, local_rows)
    all_cols = reduce(vcat, local_cols)
    all_vals = reduce(vcat, local_vals)
    sparse(all_rows, all_cols, all_vals, n_items, n_items)
end

# ──────────────────────────────────────────────────────────────────────────────
# Jaccard similarity with top-k truncation
# ──────────────────────────────────────────────────────────────────────────────

function _jaccard_knn(X::SparseMatrixCSC{Tv,Ti}, k::Int, shrinkage::T) where {Tv,Ti,T}
    n_users, n_items = size(X)

    # Column support sizes (nnz per item)
    col_nnz = Vector{Int}(undef, n_items)
    @inbounds for j in 1:n_items
        col_nnz[j] = length(nzrange(X, j))
    end

    # Binary Gram: intersection counts via (X>0)' * (X>0)
    # No copy needed — X_bin shares structure with X but has its own nzval;
    # the multiplication X_bin' * X_bin only reads from both arrays.
    X_bin = SparseMatrixCSC(n_users, n_items, X.colptr, rowvals(X), ones(Int, nnz(X)))
    G = X_bin' * X_bin  # intersection counts

    nt = Threads.nthreads()
    local_rows = _thread_buffers(() -> Int[], nt)
    local_cols = _thread_buffers(() -> Int[], nt)
    local_vals = _thread_buffers(() -> T[], nt)

    Threads.@threads for chunk in 1:nt
        for j in _thread_chunk_bounds(chunk, n_items, nt)
        nj = col_nnz[j]
        nj == 0 && continue

        sims = Vector{Pair{Int,T}}()
        for idx in nzrange(G, j)
            i = rowvals(G)[idx]
            i == j && continue
            ni = col_nnz[i]
            ni == 0 && continue
            intersection = Int(nonzeros(G)[idx])
            union_size = ni + nj - intersection
            sim = T(intersection) / (T(union_size) + shrinkage)
            sim > zero(T) && push!(sims, i => sim)
        end

        if length(sims) > k
            partialsort!(sims, 1:k; by=last, rev=true)
            resize!(sims, k)
        end

        for (i, s) in sims
            push!(local_rows[chunk], i)
            push!(local_cols[chunk], j)
            push!(local_vals[chunk], s)
        end
        end
    end

    all_rows = reduce(vcat, local_rows)
    all_cols = reduce(vcat, local_cols)
    all_vals = reduce(vcat, local_vals)
    sparse(all_rows, all_cols, all_vals, n_items, n_items)
end

# ──────────────────────────────────────────────────────────────────────────────
# recommend / score — same pattern as SLIM (sparse W)
# ──────────────────────────────────────────────────────────────────────────────

"""
    recommend(model::ItemKNN, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores = X * W, excluding seen items.
"""
function recommend(model::ItemKNN{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    n_users = size(X, 1)
    n_items = size(model.W, 1)
    k_out = _validate_recommend_input(X, n_items, k)

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

        @inbounds @simd for i in 1:n_items
            scores[i] = zero(T)
        end

        @inbounds for idx in nzrange(S_csr, u)
            j = Int(S_csr.colval[idx])
            scores[j] = S_csr.nzval[idx]
        end

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
    score(model::ItemKNN, X) -> SparseMatrixCSC

Return sparse score matrix S = X * W.
"""
function score(model::ItemKNN{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    size(X, 2) == size(model.W, 1) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.W, 1))"))
    X * model.W
end
