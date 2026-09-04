# ──────────────────────────────────────────────────────────────────────────────

"""
    PearsonKNN{T} <: AbstractExplicitModel

User-based k-nearest-neighbors rating predictor with user-mean-centered
Pearson similarity (Surprise `KNNWithMeans`):

    ŷ_ui = r̄_u + ( Σ_{v ∈ N_k(u)} sim(u,v) · (r_vi - r̄_v) ) / Σ_{v} |sim(u,v)|

The user means absorb rating-scale differences ("generous"/"critical" users);
similarity is the centered cosine over co-rated items. Neighbors with
non-positive similarity are dropped (`min_k` controls the fallback to the
plain user mean).

# Reference

User-based Pearson nearest-neighbor prediction originates with Resnick et
al. (1994), "GroupLens: An Open Architecture for Collaborative Filtering of
Netnews" (ACM CSCW).

# Constructor
```julia
PearsonKNN(; k=40, min_k=1, verbose=true)
```

# Example
```julia
julia> using SparseArrays, Random

julia> X = sprand(MersenneTwister(1), 100, 50, 0.2); nonzeros(X) .= 1 .+ 4 .* rand(MersenneTwister(2), nnz(X));

julia> model = PearsonKNN(k=20, verbose=false);

julia> fit!(model, X);

julia> size(predict(model, X))
(100, 50)
```
"""
mutable struct PearsonKNN{T<:AbstractFloat} <: AbstractExplicitModel
    const k::Int
    const min_k::Int
    const verbose::Bool
    user_mean::Vector{T}
    centered::SparseMatrixCSC{T,Int}          # r_ui - r̄_u
    neighbors::Vector{Vector{Pair{Int,T}}}    # top-k (v, sim) per user, sim > 0
    is_fitted::Bool
end

function PearsonKNN(;
    k::Int = 40,
    min_k::Int = 1,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    min_k >= 1 || throw(ArgumentError("min_k must be ≥ 1, got $min_k"))
    PearsonKNN{T}(k, min_k, verbose, T[], spzeros(T, 0, 0),
                  Vector{Pair{Int,T}}[], false)
end

function fit!(model::PearsonKNN{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "PearsonKNN")
    _require_finite_input(X, "PearsonKNN")
    old = (model.user_mean, model.centered, model.neighbors)
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    # user means and centered matrix C = X with rows shifted by r̄_u
    user_mean = zeros(T, n_users)
    X_csr = to_csr(X)
    @inbounds for u in 1:n_users
        s = zero(T); c = 0
        for idx in nzrange(X_csr, u)
            s += T(X_csr.nzval[idx]); c += 1
        end
        c > 0 && (user_mean[u] = s / c)
    end
    centered = SparseMatrixCSC(n_users, n_items, X.colptr, rowvals(X),
                               [T(nonzeros(X)[idx]) - user_mean[rowvals(X)[idx]]
                                for idx in 1:nnz(X)])

    # user norms of centered rows
    cnorm = zeros(T, n_users)
    @inbounds for u in 1:n_users
        s = zero(T)
        for idx in nzrange(X_csr, u)
            s += T(X_csr.nzval[idx])^2
        end
        cnorm[u] = sqrt(s)
    end

    # Pearson similarity = centered Gram normalized by the user norms
    G = Matrix{T}(centered * centered')
    neighbors = Vector{Vector{Pair{Int,T}}}(undef, n_users)

    nt = Threads.nthreads()

    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
        nu = cnorm[u]
        if nu == zero(T)
            neighbors[u] = Pair{Int,T}[]
            continue
        end
        sims = Vector{Pair{Int,T}}()
        @inbounds for v in 1:n_users
            v == u && continue
            nv = cnorm[v]
            nv == zero(T) && continue
            s = G[u, v] / (nu * nv)
            s > zero(T) && push!(sims, v => s)
        end
        if length(sims) > model.k
            partialsort!(sims, 1:model.k; by=last, rev=true)
            resize!(sims, model.k)
        end
        # if fewer than min_k positive neighbors, fall back to the user mean
        neighbors[u] = length(sims) < model.min_k ? Pair{Int,T}[] : sims
        end
    end

    model.user_mean = user_mean
    model.centered = centered
    model.neighbors = neighbors
    model.is_fitted = true
    model
    catch
        model.user_mean, model.centered, model.neighbors = old
        model.is_fitted = false
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

score(model::PearsonKNN{T}, X::SparseMatrixCSC) where {T} = predict(model, X)

function score(model::PearsonKNN{T}, user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    length(user_indices) == length(item_indices) ||
        throw(DimensionMismatch("user_indices and item_indices must have the same length"))
    [_pearson_value(model, u, i) for (u, i) in zip(user_indices, item_indices)]
end

@inline function _pearson_value(model::PearsonKNN{T}, u::Int, i::Int) where {T}
    num = zero(T)
    den = zero(T)
    @inbounds for (v, s) in model.neighbors[u]
        cv = model.centered[v, i]
        cv == zero(T) && continue
        num += s * cv
        den += abs(s)
    end
    den > zero(T) && return model.user_mean[u] + num / den
    model.user_mean[u]
end

function predict(model::PearsonKNN{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    n_u, n_i = size(X)
    n_u == length(model.user_mean) || throw(DimensionMismatch(
        "X has $n_u users but the fitted model has $(length(model.user_mean))"))
    n_i == size(model.centered, 2) || throw(DimensionMismatch(
        "X has $n_i items but the fitted model has $(size(model.centered, 2))"))

    # Fused single pass per user (threaded): for each neighbor v, scatter
    # sim(u,v) · C[v, i] into num[i] and sim(u,v) into cnt[i] over the items v
    # rated. ŷ = r̄_u + num/cnt where cnt > 0 (Surprise semantics: a neighbor
    # without a rating contributes to neither side).
    S = Matrix{T}(undef, n_u, n_i)
    cent_csr = to_csr(model.centered)
    nt = Threads.nthreads()
    num_bufs = _thread_buffers(() -> Vector{T}(undef, n_i), nt)
    cnt_bufs = _thread_buffers(() -> Vector{T}(undef, n_i), nt)

    Threads.@threads for chunk in 1:nt
        num = num_bufs[chunk]
        cbuf = cnt_bufs[chunk]
        for u in _thread_chunk_bounds(chunk, n_u, nt)
            μu = model.user_mean[u]
            nbrs = model.neighbors[u]
            if isempty(nbrs)
                fill!(view(S, u, :), μu)
                continue
            end
            fill!(num, zero(T))
            fill!(cbuf, zero(T))
            @inbounds for (v, s) in nbrs
                for idx in nzrange(cent_csr, v)
                    i = Int(cent_csr.colval[idx])
                    c = cent_csr.nzval[idx]
                    num[i] += s * c
                    cbuf[i] += s
                end
            end
            @inbounds for i in 1:n_i
                S[u, i] = cbuf[i] > zero(T) ? μu + num[i] / cbuf[i] : μu
            end
        end
    end
    S
end
