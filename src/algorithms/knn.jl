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
ItemKNN(; k=20, similarity=:cosine, shrinkage=0.0, asym_alpha=0.5,
          k1=1.2, b=0.75, normalize=true)
```

# Fields
- `k::Int` — number of neighbors to retain per item
- `similarity::Symbol` — `:cosine`, `:jaccard`, `:asym_cosine` or `:bm25`
- `shrinkage::T` — additive shrinkage to denominator (regularizes rare items)
- `asym_alpha::T` — asymmetry exponent for `:asym_cosine` (0.5 = symmetric binary cosine; values below 0.5 favor niche candidates)
- `k1::T`, `b::T` — BM25 parameters for `:bm25` (saturation and length normalization; standard IR values 1.2 / 0.75)
- `normalize::Bool` — row-normalize the similarity matrix (divide by row sum)
- `W::SparseMatrixCSC{T,Int}` — sparse item-item similarity matrix after fitting

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> model = ItemKNN(k=10, similarity=:cosine, verbose=false);

julia> fit!(model, X; rng=MersenneTwister(2));

julia> preds = recommend(model, X; k=10);

julia> size(preds)
(200, 10)
```
"""
mutable struct ItemKNN{T<:AbstractFloat} <: AbstractItemSimilarity
    const k::Int
    const similarity::Symbol
    const shrinkage::T
    const asym_alpha::T
    const k1::T
    const b::T
    const normalize::Bool
    const verbose::Bool
    W::SparseMatrixCSC{T,Int}
    is_fitted::Bool
end

function ItemKNN(;
    k::Int = 20,
    similarity::Symbol = :cosine,
    shrinkage::Float64 = 0.0,
    asym_alpha::Float64 = 0.5,
    k1::Float64 = 1.2,
    b::Float64 = 0.75,
    normalize::Bool = true,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    similarity in (:cosine, :jaccard, :asym_cosine, :bm25) ||
        throw(ArgumentError("similarity must be :cosine, :jaccard, :asym_cosine or :bm25, got :$similarity"))
    shrinkage >= 0.0 || throw(ArgumentError("shrinkage must be non-negative, got $shrinkage"))
    0.0 < asym_alpha <= 1.0 || throw(ArgumentError("asym_alpha must be in (0, 1], got $asym_alpha"))
    k1 > 0.0 || throw(ArgumentError("k1 must be positive, got $k1"))
    0.0 <= b <= 1.0 || throw(ArgumentError("b must be in [0, 1], got $b"))
    ItemKNN{T}(k, similarity, T(shrinkage), T(asym_alpha), T(k1), T(b),
               normalize, verbose, spzeros(T, 0, 0), false)
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
    _require_finite_input(X, "ItemKNN")
    kn = min(model.k, n_items - 1)

    model.verbose && @info "[ItemKNN] Computing $(model.similarity) similarity for $n_items items (k=$kn)..."

    if model.similarity == :cosine
        W = _cosine_knn(X, kn, T(model.shrinkage))
    elseif model.similarity == :jaccard
        W = _jaccard_knn(X, kn, T(model.shrinkage))
    elseif model.similarity == :asym_cosine
        W = _asym_cosine_knn(X, kn, T(model.shrinkage), model.asym_alpha)
    else  # :bm25
        W = _bm25_knn(X, kn, T(model.shrinkage), model.k1, model.b)
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

# Min-heap of size k over (value, index) pairs: the root holds the smallest
# kept value, so any candidate that is not better than the root is discarded
# in O(1). "Better" is lexicographic (larger value, then smaller item index),
# matching the stable top-k the old partialsort path produced — ties between
# equal similarity values must keep the same members.
@inline function _heap_better(v_a::T, i_a::Int, v_b::T, i_b::Int) where {T}
    v_a > v_b || (v_a == v_b && i_a < i_b)
end

@inline function _heap_insert_topk!(heap_v::Vector{T}, heap_i::Vector{Int},
                                    s::T, item::Int, k::Int) where {T}
    _heap_better(s, item, heap_v[1], heap_i[1]) || return
    heap_v[1] = s
    heap_i[1] = item
    # Sift the new root down to restore the min-heap invariant: while the
    # parent is better than its worse child, swap with that child.
    c = 1
    @inbounds while true
        l = 2c
        l > k && break
        r = l + 1
        m = (r > k || _heap_better(heap_v[r], heap_i[r], heap_v[l], heap_i[l])) ? l : r
        _heap_better(heap_v[c], heap_i[c], heap_v[m], heap_i[m]) || break
        heap_v[c], heap_v[m] = heap_v[m], heap_v[c]
        heap_i[c], heap_i[m] = heap_i[m], heap_i[c]
        c = m
    end
    nothing
end

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

    # Column-by-column top-k with a size-k min-heap per thread. The full Gram
    # XᵀX is never materialized: for each column j its dot products with all
    # other items are accumulated in a dense per-thread buffer by scattering
    # each user u ∈ N(j) with their full row (X in CSR), normalized, and the
    # top-k kept directly. Memory is O(n_items + nnz), not O(nnz(XᵀX)).
    X_csr = to_csr(X)

    nt = Threads.nthreads()
    dot_bufs = _thread_buffers(() -> zeros(T, n_items), nt)
    heap_bufs = _thread_buffers(() -> fill(T(-Inf), max(k, 1)), nt)
    heap_idx_bufs = _thread_buffers(() -> zeros(Int, max(k, 1)), nt)
    local_rows = _thread_buffers(() -> Int[], nt)
    local_cols = _thread_buffers(() -> Int[], nt)
    local_vals = _thread_buffers(() -> T[], nt)

    rv = rowvals(X)
    nz = nonzeros(X)

    Threads.@threads for chunk in 1:nt
        dot = dot_bufs[chunk]
        hv = heap_bufs[chunk]
        hi = heap_idx_bufs[chunk]
        @inbounds for j in _thread_chunk_bounds(chunk, n_items, nt)
            norm_j = col_norms[j]
            norm_j == zero(T) && continue

            # Reset the dot-product buffer and the heap
            for i in 1:n_items
                dot[i] = zero(T)
            end
            for e in 1:k
                hv[e] = T(-Inf)
            end

            # dot[i] += X[u,j] · X[u,i] over users u that rated item j
            for idx in nzrange(X, j)
                u = Int(rv[idx])
                xj = T(nz[idx])
                for r in nzrange(X_csr, u)
                    i = Int(X_csr.colval[r])
                    i == j && continue
                    dot[i] += xj * T(X_csr.nzval[r])
                end
            end

            # Normalize and keep the top-k similarities in the heap
            for i in 1:n_items
                i == j && continue
                ni = col_norms[i]
                (ni == zero(T) || dot[i] <= zero(T)) && continue
                _heap_insert_topk!(hv, hi, dot[i] / (ni * norm_j + shrinkage), i, k)
            end

            for e in 1:k
                hv[e] > T(-Inf) || continue
                push!(local_rows[chunk], hi[e])
                push!(local_cols[chunk], j)
                push!(local_vals[chunk], hv[e])
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
# Asymmetric cosine similarity (Aiolli 2013 / RecBole ACoS)
# ──────────────────────────────────────────────────────────────────────────────

function _asym_cosine_knn(X::SparseMatrixCSC{Tv,Ti}, k::Int, shrinkage::T,
                          alpha::T) where {Tv,Ti,T}
    n_users, n_items = size(X)

    col_nnz = Vector{Int}(undef, n_items)
    @inbounds for j in 1:n_items
        col_nnz[j] = length(nzrange(X, j))
    end

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
        denom_j = T(nj)^alpha

        sims = Vector{Pair{Int,T}}()
        for idx in nzrange(G, j)
            i = rowvals(G)[idx]
            i == j && continue
            ni = col_nnz[i]
            ni == 0 && continue
            intersection = Int(nonzeros(G)[idx])
            # asymmetric: the candidate item's popularity is exponentiated
            # with (1-α) and the target's with α; α=0.5 reduces to binary cosine
            sim = T(intersection) / (T(ni)^(one(T) - alpha) * denom_j + shrinkage)
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
# BM25 item similarity
# ──────────────────────────────────────────────────────────────────────────────

function _bm25_knn(X::SparseMatrixCSC{Tv,Ti}, k::Int, shrinkage::T,
                   k1::T, b::T) where {Tv,Ti,T}
    n_users, n_items = size(X)

    # Per-user frequency (document frequency of the "user-as-term" view): how
    # many items the user interacted with.
    user_df = zeros(Int, n_users)
    @inbounds for j in 1:n_items
        for idx in nzrange(X, j)
            user_df[rowvals(X)[idx]] += 1
        end
    end

    # IDF of a user spreads mass over few items: log((N - df + 0.5)/(df + 0.5))
    idf = Vector{T}(undef, n_users)
    @inbounds for u in 1:n_users
        df = user_df[u]
        idf[u] = df > 0 ? T(log((n_items - df + 0.5) / (df + 0.5))) : zero(T)
    end

    # Weighted Gram: G[i, j] = Σ_{u ∈ N(i)∩N(j)} idf_u  = (D^{1/2} X)' (D^{1/2} X)
    rv = rowvals(X)
    Xw = SparseMatrixCSC(n_users, n_items, X.colptr, rv,
                         [T(sqrt(idf[rv[idx]])) for idx in 1:nnz(X)])
    G = Xw' * Xw

    col_nnz = Vector{Int}(undef, n_items)
    @inbounds for j in 1:n_items
        col_nnz[j] = length(nzrange(X, j))
    end
    avgdl = sum(col_nnz) / max(n_items, 1)

    # Document-length normalization of the target item j (saturating at k1)
    scale_j = Vector{T}(undef, n_items)
    @inbounds for j in 1:n_items
        len = col_nnz[j]
        denom = len > 0 ? k1 * (one(T) - b + b * T(len) / T(max(avgdl, 1))) + one(T) : one(T)
        scale_j[j] = (k1 + one(T)) / (denom + shrinkage)
    end

    nt = Threads.nthreads()
    local_rows = _thread_buffers(() -> Int[], nt)
    local_cols = _thread_buffers(() -> Int[], nt)
    local_vals = _thread_buffers(() -> T[], nt)

    Threads.@threads for chunk in 1:nt
        for j in _thread_chunk_bounds(chunk, n_items, nt)
        col_nnz[j] == 0 && continue

        sims = Vector{Pair{Int,T}}()
        for idx in nzrange(G, j)
            i = rowvals(G)[idx]
            i == j && continue
            sim = T(nonzeros(G)[idx]) * scale_j[j]
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
    _predict_sparse_score_topk(X, model.W, k)
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
