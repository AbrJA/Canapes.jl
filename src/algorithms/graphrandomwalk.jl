# ──────────────────────────────────────────────────────────────────────────────
# GraphRandomWalk — 3-step random-walk item similarity with popularity penalty
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Paolino, Boratto, Trevisiol, Castells (2017)
#   "RP3β: A Novel Approach to the Long-tail Problem in Recommender Systems"
#   (RecSys 2017) — the type name is `GraphRandomWalk`; `α`/`β` are the paper's
#   hyperparameters. Lineage: 3-step bipartite-graph walk (P3α, Cooper et al.
#   2014) with an item-popularity penalization. This is the closed-form
#   matrix formulation (as in irspack / similaripy); it is NOT the Pixie
#   algorithm (Eksombatchai et al. 2018), which is a sampled personalized
#   random-walk designed for real-time serving.
#
# Score of item j for a user u:
#   S[u, j] = Σ_{i ∈ N(u)} 1/|N(u)| · 1/|N(i)| · pop(j)^{-β}   (α = 1)
# i.e. W_ij = Σ_{u ∈ N(i)∩N(j)} 1/|N(i)| · 1/|N(u)| · pop(j)^{-β},
# with the α power applied to the transition entries before the product.
# ──────────────────────────────────────────────────────────────────────────────

"""
    GraphRandomWalk{T} <: AbstractItemSimilarity

3-step random-walk item similarity with item-popularity penalization
("RP3β", Paolino et al. 2017). Learns a sparse item-item score matrix:

    W_ij = Σ_{u ∈ N(i)∩N(j)} |N(i)|^{-1} · |N(u)|^{-1} · pop(j)^{-β}

An item's columns are weighted by the inverse popularity raised to `β`, which
penalizes head items and boosts long-tail candidates; `α` powers the
transition entries (a no-op at its default `1.0` on binary interactions but
relevant when the matrix carries weights).

No training loop — the walk matrix is built from the two-hop transition
product. With `k` set (the memory-bounded path) the top-k columns are computed
streaming per item with bounded heaps — O(n_items·threads + nnz) memory, never
materializing the full O(Σ_u deg(u)²)-nonzero two-hop matrix (which is only
built when `k = nothing`). The fitted `W` is sparse; `recommend`/`score` reuse
the sparse item-similarity paths.

Note: this is the closed-form RP3β formulation (as in irspack/similaripy), not
the Pixie real-time sampled-walk system (Eksombatchai et al. 2018).

# Constructor
```julia
GraphRandomWalk(; α=1.0, β=0.6, k=nothing, normalize=false, verbose=true)
```

# Fields
- `α::T` — power applied to transition entries (default 1.0)
- `β::T` — popularity-penalization exponent (default 0.6; larger → stronger long-tail bias)
- `k::Union{Nothing,Int}` — keep the top-k neighbors per item (default `nothing` = keep all)
- `normalize::Bool` — row-normalize W after fit (divide by row sum)
- `W::SparseMatrixCSC{T,Int}` — sparse item-item walk matrix after fitting
- `is_fitted::Bool`

# Example
```julia
julia> using SparseArrays, Random

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> model = GraphRandomWalk(β=0.0, k=50, verbose=false);

julia> fit!(model, X);

julia> size(recommend(model, X; k=10))
(200, 10)
```
"""
mutable struct GraphRandomWalk{T<:AbstractFloat} <: AbstractItemSimilarity
    const α::T
    const β::T
    const k::Union{Nothing,Int}
    const normalize::Bool
    const verbose::Bool
    W::SparseMatrixCSC{T,Int}
    is_fitted::Bool
end

function GraphRandomWalk(;
    α::Float64 = 1.0,
    β::Float64 = 0.6,
    k::Union{Nothing,Int} = nothing,
    normalize::Bool = false,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    α > 0.0 || throw(ArgumentError("α must be positive, got $α"))
    β >= 0.0 || throw(ArgumentError("β must be non-negative, got $β"))
    k === nothing || k >= 1 || throw(ArgumentError("k must be ≥ 1 or nothing, got $k"))
    GraphRandomWalk{T}(T(α), T(β), k, normalize, verbose, spzeros(T, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

function fit!(model::GraphRandomWalk{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    old_W = model.W
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    _require_nonempty_dimensions(X, "GraphRandomWalk")
    _require_finite_input(X, "GraphRandomWalk")
    all(x -> x >= zero(x), nonzeros(X)) || throw(ArgumentError(
        "GraphRandomWalk requires non-negative interaction values, got negative"))

    # Degrees: user popularity d_u (rows), item popularity d_i (columns)
    d_u = vec(sum(X; dims=2))
    d_i = vec(sum(X; dims=1))

    if model.k !== nothing
        # Memory-bounded path: the top-k walk matrix is computed column-by-column
        # with bounded per-thread heaps, never materializing the intermediate
        # two-hop matrix (which holds O(Σ_u deg(u)²) nonzeros). Memory is
        # O(n_items·nthreads + nnz) instead of O(nnz(W_full)).
        W = _walk_topk_streaming(X, model.k, model.α, model.β, T)
    else
        # Full two-hop walk W = S1 * S2 (inherently O(Σ_u deg(u)²) nonzeros)
        # Item→user transition S1 (row = item, col = user): Xᵀ scaled by 1/d_i
        # User→item transition S2 (row = user, col = item): X scaled by 1/d_u
        # W = S1 * S2  is computed as (D_i^{-1} Xᵀ) · (D_u^{-1} X).
        Xt = SparseMatrixCSC(X')
        S1 = SparseMatrixCSC(n_items, n_users, Xt.colptr, rowvals(Xt), copy(nonzeros(Xt)))
        S2 = SparseMatrixCSC(n_users, n_items, X.colptr, rowvals(X), copy(nonzeros(X)))

        d_i_inv = T.(1 ./ max.(d_i, one(eltype(d_i))))
        d_u_inv = T.(1 ./ max.(d_u, one(eltype(d_u))))

        # Scale S1 rows (items): element (i, u) gets 1/d_i
        @inbounds for c in 1:n_users
            for idx in nzrange(S1, c)
                S1.nzval[idx] *= d_i_inv[rowvals(S1)[idx]]
            end
        end
        # Scale S2 rows (users): element (u, i) gets 1/d_u
        @inbounds for c in 1:n_items
            for idx in nzrange(S2, c)
                S2.nzval[idx] *= d_u_inv[rowvals(S2)[idx]]
            end
        end

        # α-power on the transition entries (no-op at α = 1)
        if model.α != one(T)
            S1.nzval .^= model.α
            S2.nzval .^= model.α
        end

        # Two-hop walk: W = S1 * S2  (items × items)
        W = S1 * S2

        # Popularity penalization: scale each column j by pop(j)^{-β}
        pop_inv = T.(1 ./ max.(d_i .^ model.β, one(eltype(d_i))))
        @inbounds for j in 1:n_items
            pj = pop_inv[j]
            pj == one(T) && continue
            for idx in nzrange(W, j)
                W.nzval[idx] *= pj
            end
        end

        # Zero the diagonal (no self-recommendation)
        @inbounds for j in 1:n_items
            for idx in nzrange(W, j)
                rowvals(W)[idx] == j && (W.nzval[idx] = zero(T))
            end
        end
    end

    # Optional row normalization (each row sums to 1)
    if model.normalize
        row_sums = vec(sum(W; dims=2))
        @inbounds for c in axes(W, 2)
            for idx in nzrange(W, c)
                r = rowvals(W)[idx]
                rs = row_sums[r]
                rs > zero(T) && (W.nzval[idx] /= rs)
            end
        end
    end

    model.W = W
    model.is_fitted = true
    model.verbose && @info "[GraphRandomWalk] Done. W has $(nnz(W)) nonzeros (density=$(round(nnz(W)/n_items^2; digits=6)))"
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
# Memory-bounded top-k two-hop walk
# ──────────────────────────────────────────────────────────────────────────────

# For each target item j the walk column is
#     W[i, j] = pop(j)^{-β} · |N(i)|^{-α} · Σ_{u ∈ N(i)∩N(j)} X_ui^α · X_uj^α · |N(u)|^{-α}
# computed column-by-column: a dense per-thread dot buffer scatters each user
# u ∈ N(j) with their full row (CSR), the per-column scaling is applied, and the
# top-k entries are kept with a bounded min-heap — the same streaming pattern as
# ItemKNN's `_cosine_knn`. The full O(Σ_u deg(u)²)-nonzero two-hop matrix is
# never materialized: memory is O(n_items·nthreads + nnz).
function _walk_topk_streaming(X::SparseMatrixCSC{Tv,Ti}, k::Int, α::AbstractFloat,
                              β::AbstractFloat, ::Type{T}) where {Tv,Ti,T}
    n_users, n_items = size(X)

    d_u = vec(sum(X; dims=2))
    d_i = vec(sum(X; dims=1))

    # Inverse transition weights: w_u = |N(u)|^{-α} (columns of S2), row_scale
    # i = |N(i)|^{-α} (rows of S1), col_scale j = pop(j)^{-β}.
    w_u = Vector{T}(undef, n_users)
    @inbounds for u in 1:n_users
        w_u[u] = d_u[u] > 0 ? T(one(T) / d_u[u]^α) : zero(T)
    end
    row_scale = Vector{T}(undef, n_items)
    @inbounds for i in 1:n_items
        row_scale[i] = d_i[i] > 0 ? T(one(T) / d_i[i]^α) : zero(T)
    end
    col_scale = Vector{T}(undef, n_items)
    @inbounds for j in 1:n_items
        col_scale[j] = d_i[j] > 0 ? T(one(T) / d_i[j]^β) : zero(T)
    end

    X_csr = to_csr(X)
    nt = Threads.nthreads()
    dot_bufs = _thread_buffers(() -> zeros(T, n_items), nt)
    heap_bufs = _thread_buffers(() -> fill(T(-Inf), k), nt)
    heap_idx_bufs = _thread_buffers(() -> zeros(Int, k), nt)
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
            d_i[j] == 0 && continue

            for i in 1:n_items
                dot[i] = zero(T)
            end
            for e in 1:k
                hv[e] = T(-Inf)
            end

            # dot[i] += (X_uj/d_u)^α · (X_ui)^α  over users u of item j
            if α == 1.0
                for idx in nzrange(X, j)
                    u = Int(rv[idx])
                    w = T(nz[idx]) * w_u[u]
                    for r in nzrange(X_csr, u)
                        i = Int(X_csr.colval[r])
                        i == j && continue
                        dot[i] += w * T(X_csr.nzval[r])
                    end
                end
            else
                for idx in nzrange(X, j)
                    u = Int(rv[idx])
                    w = (T(nz[idx]) / T(d_u[u])) ^ T(α)
                    for r in nzrange(X_csr, u)
                        i = Int(X_csr.colval[r])
                        i == j && continue
                        dot[i] += w * (T(X_csr.nzval[r]) ^ T(α))
                    end
                end
            end

            # Scale by |N(i)|^{-α} and pop(j)^{-β}, keep the top-k positives
            csj = col_scale[j]
            for i in 1:n_items
                i == j && continue
                d = dot[i]
                (d <= zero(T) || row_scale[i] == zero(T)) && continue
                _heap_insert_topk!(hv, hi, d * row_scale[i] * csj, i, k)
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
# recommend / score — same pattern as ItemKNN (sparse W)
# ──────────────────────────────────────────────────────────────────────────────

"""
    recommend(model::GraphRandomWalk, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores = X * W, excluding seen items.
"""
function recommend(model::GraphRandomWalk{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    _predict_sparse_score_topk(X, model.W, k)
end

"""
    score(model::GraphRandomWalk, X) -> SparseMatrixCSC

Return sparse score matrix S = X * W.
"""
function score(model::GraphRandomWalk{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    size(X, 2) == size(model.W, 1) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.W, 1))"))
    X * model.W
end