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

No training loop — the walk matrix is built with two sparse matrix products.
The fitted `W` is sparse; `recommend`/`score` reuse the sparse item-similarity
paths. Memory is O(nnz(X) + nnz(W)) and scales well beyond ShallowAutoencoder's O(n_items²).

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

    # Optional top-k truncation per item
    if model.k !== nothing
        W = _truncate_topk_sparse(W, model.k)
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

# Keep the top-k (positive) entries per column of the sparse matrix W.
function _truncate_topk_sparse(W::SparseMatrixCSC{T,Int}, k::Int) where {T}
    n_items = size(W, 2)
    rv = rowvals(W)
    nz = nonzeros(W)
    rows = Int[]; cols = Int[]; vals = T[]
    @inbounds for j in 1:n_items
        sims = Vector{Pair{Int,T}}()
        for idx in nzrange(W, j)
            v = nz[idx]
            v > zero(T) && push!(sims, rv[idx] => v)
        end
        if length(sims) > k
            partialsort!(sims, 1:k; by=last, rev=true)
            resize!(sims, k)
        end
        for (i, v) in sims
            push!(rows, i); push!(cols, j); push!(vals, v)
        end
    end
    sparse(rows, cols, vals, n_items, n_items)
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