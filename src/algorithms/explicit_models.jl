# ──────────────────────────────────────────────────────────────────────────────
# Explicit (rating-prediction) models — BaselineOnly, SlopeOne, PearsonKNN
# ──────────────────────────────────────────────────────────────────────────────
#
# All three subclass AbstractExplicitModel: they predict continuous ratings,
# expose `predict`/`score` (dense n_users × n_items matrices) and pairwise
# scoring, and are evaluated with rmse/mae — never `recommend`.
#
# References:
#   - BaselineOnly: Surprise's biased baseline (Koren 2009), ALS from the
#     sparse ratings.
#   - SlopeOne: Lemire & Maclachlan (2005), "Slope One Predictors for Online
#     Rating-Based Collaborative Filtering".
#   - PearsonKNN: Surprise's KNNWithMeans — user-mean-centered Pearson
#     neighborhood averaging; no training loop beyond the similarity matrix.
# ──────────────────────────────────────────────────────────────────────────────

"""
    BaselineOnly{T} <: AbstractExplicitModel

Baseline rating predictor `ŷ_ui = μ + b_u + b_i` (global mean + user and item
biases), learned by Alternating Least Squares over the observed ratings —
i.e. the baseline term of BiasedMF without the latent factors. The classic
first point of comparison for rating prediction (Surprise `BaselineOnly`).

The bias update is the exact ALS step on the residual `r_ui - μ - b_u - b_i`
with L2 regularization `λ` in the denominator:
`b_u ← Σ_i (r_ui - μ - b_i) / (|I_u| + λ)`, and symmetrically for items.

# Constructor
```julia
BaselineOnly(; λ=0.02, max_iter=10, verbose=true)
```

# Example
```julia
julia> using SparseArrays, Random

julia> X = sprand(MersenneTwister(1), 100, 50, 0.2); nonzeros(X) .= 1 .+ 4 .* rand(MersenneTwister(2), nnz(X));

julia> model = BaselineOnly(verbose=false);

julia> fit!(model, X; rng=MersenneTwister(3));

julia> size(predict(model, X))
(100, 50)
```
"""
mutable struct BaselineOnly{T<:AbstractFloat} <: AbstractExplicitModel
    const λ::T
    const max_iter::Int
    const verbose::Bool
    global_mean::T
    user_bias::Vector{T}
    item_bias::Vector{T}
    is_fitted::Bool
end

function BaselineOnly(;
    λ::Float64 = 0.02,
    max_iter::Int = 10,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    BaselineOnly{T}(T(λ), max_iter, verbose, zero(T), T[], T[], false)
end

function fit!(model::BaselineOnly{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "BaselineOnly")
    _require_finite_input(X, "BaselineOnly")
    λ = model.λ
    is_fitted = model.is_fitted
    global_mean_old, b_u_old, b_i_old = model.global_mean, model.user_bias, model.item_bias
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    model.global_mean = T(sum(nonzeros(X)) / max(nnz(X), 1))
    b_u = zeros(T, n_users)
    b_i = zeros(T, n_items)
    μ = model.global_mean

    X_csr = to_csr(X)
    for _ in 1:model.max_iter
        # user step (uses updated item biases — Gauss-Seidel)
        @inbounds for u in 1:n_users
            s = zero(T); c = zero(T)
            for idx in nzrange(X_csr, u)
                i = Int(X_csr.colval[idx])
                s += T(X_csr.nzval[idx]) - μ - b_i[i]
                c += one(T)
            end
            b_u[u] = s / (c + λ)
        end
        # item step (uses updated user biases)
        for j in 1:n_items
            s = zero(T); c = zero(T)
            for idx in nzrange(X, j)
                u = Int(rowvals(X)[idx])
                s += T(nonzeros(X)[idx]) - μ - b_u[u]
                c += one(T)
            end
            b_i[j] = s / (c + λ)
        end
    end
    model.user_bias = b_u
    model.item_bias = b_i
    model.is_fitted = true
    model
    catch
        model.global_mean = global_mean_old
        model.user_bias = b_u_old
        model.item_bias = b_i_old
        model.is_fitted = is_fitted
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

@inline function _baseline_value(model::BaselineOnly{T}, u::Int, i::Int) where {T}
    model.global_mean + model.user_bias[u] + model.item_bias[i]
end

function score(model::BaselineOnly{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    n_u, n_i = size(X)
    n_u == length(model.user_bias) || throw(DimensionMismatch(
        "X has $n_u users but the fitted model has $(length(model.user_bias))"))
    n_i == length(model.item_bias) || throw(DimensionMismatch(
        "X has $n_i items but the fitted model has $(length(model.item_bias))"))
    S = Matrix{T}(undef, n_u, n_i)
    @inbounds for i in 1:n_i
        bi = model.item_bias[i]
        for u in 1:n_u
            S[u, i] = model.global_mean + model.user_bias[u] + bi
        end
    end
    S
end

function predict(model::BaselineOnly{T}, X::SparseMatrixCSC) where {T}
    score(model, X)
end

function score(model::BaselineOnly{T}, user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    length(user_indices) == length(item_indices) ||
        throw(DimensionMismatch("user_indices and item_indices must have the same length"))
    [_baseline_value(model, u, i) for (u, i) in zip(user_indices, item_indices)]
end

# ──────────────────────────────────────────────────────────────────────────────

"""
    SlopeOne{T} <: AbstractExplicitModel

Slope One rating predictor — the Surprise formulation (Lemire & Maclachlan
2005): learns the mean rating *difference* between every item pair
`dev(i, j) = mean(r_ui - r_uj)` over users who rated both, then predicts

    ŷ_ui = μ_u + (1/|R_i(u)|) · Σ_{j ∈ R_i(u)} dev(i, j)

where `R_i(u)` is the set of items j rated by `u` that share at least one user
with `i`, and `μ_u` is the user's mean rating. This is Surprise's simplified
(unweighted) variant — not the classic weighted `dev(i,j) + r_uj` form — and
matches the reference implementation exactly. Memory is O(n_items²) for
`dev` + `freq`. Users without any relevant item fall back to their own mean
rating, then the global mean.

# Constructor
```julia
SlopeOne(; verbose=true)
```

# Example
```julia
julia> using SparseArrays, Random

julia> X = sprand(MersenneTwister(1), 100, 50, 0.2); nonzeros(X) .= 1 .+ 4 .* rand(MersenneTwister(2), nnz(X));

julia> model = SlopeOne(verbose=false);

julia> fit!(model, X);

julia> size(predict(model, X))
(100, 50)
```
"""
mutable struct SlopeOne{T<:AbstractFloat} <: AbstractExplicitModel
    const verbose::Bool
    dev::Matrix{T}          # mean rating difference dev(i, j) = mean(r_ui - r_uj)
    freq::Matrix{Int}       # number of users who rated both items
    user_mean::Vector{T}
    is_fitted::Bool
end

function SlopeOne(; verbose::Bool=true, T::Type{<:AbstractFloat}=Float32)
    SlopeOne{T}(verbose, Matrix{T}(undef, 0, 0), Matrix{Int}(undef, 0, 0),
                T[], false)
end

function fit!(model::SlopeOne{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "SlopeOne")
    _require_finite_input(X, "SlopeOne")
    old_dev, old_freq = model.dev, model.freq
    old_um = model.user_mean
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    dev = zeros(T, n_items, n_items)
    freq = zeros(Int, n_items, n_items)

    user_mean = zeros(T, n_users)
    item_cnt = zeros(Int, n_items)
    X_csr = to_csr(X)

    @inbounds for u in 1:n_users
        rng_u = nzrange(X_csr, u)
        n = length(rng_u)
        n == 0 && continue
        # collect the user's (item, rating) pairs from the CSR row
        m = 0
        s = zero(T)
        for idx in rng_u
            i = Int(X_csr.colval[idx])
            r = T(X_csr.nzval[idx])
            s += r
            item_cnt[i] += 1
            m += 1
        end
        user_mean[u] = s / m
        # pair differences for this user
        it = Vector{Int}(undef, m)
        rv = Vector{T}(undef, m)
        p = 1
        for idx in rng_u
            it[p] = Int(X_csr.colval[idx])
            rv[p] = T(X_csr.nzval[idx])
            p += 1
        end
        for a in 1:m, b in 1:m
            a == b && continue
            dev[it[a], it[b]] += rv[a] - rv[b]
            freq[it[a], it[b]] += 1
        end
    end

    # normalize into means (Surprise semantics): divide the upper triangle by
    # the count and mirror with a sign flip; diagonal stays 0
    @inbounds for i in 1:n_items
        dev[i, i] = zero(T)
        for j in (i + 1):n_items
            f = freq[i, j]
            if f > 0
                dev[i, j] /= f
                dev[j, i] = -dev[i, j]
            end
        end
    end

    model.dev = dev
    model.freq = freq
    model.user_mean = user_mean
    model.is_fitted = true
    model
    catch
        model.dev, model.freq = old_dev, old_freq
        model.user_mean = old_um
        model.is_fitted = false
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

# Prediction for user u (with rated items `items`) at target item `i` —
# Surprise's SlopeOne estimate: user mean + average deviation over the
# relevant items j (j rated by u with at least one common user with i).
@inline function _slopeone_value(model::SlopeOne{T}, u::Int, items::Vector{Int},
                                 glm::T, c::Int, i::Int) where {T}
    num = zero(T)
    cnt = 0
    @inbounds for a in 1:c
        j = items[a]
        model.freq[i, j] > 0 || continue
        num += model.dev[i, j]
        cnt += 1
    end
    μu = model.user_mean[u]
    cnt > 0 && return μu + num / cnt
    # no relevant items: the user's own mean, else the global mean
    c > 0 ? μu : glm
end

function predict(model::SlopeOne{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    n_u, n_i = size(X)
    n_u == length(model.user_mean) || throw(DimensionMismatch(
        "X has $n_u users but the fitted model has $(length(model.user_mean))"))
    n_i == size(model.dev, 1) || throw(DimensionMismatch(
        "X has $n_i items but the fitted model has $(size(model.dev, 1))"))

    S = Matrix{T}(undef, n_u, n_i)
    X_csr = to_csr(X)
    # global mean for the cold-user fallback
    glm = zero(T)
    cnt = 0
    @inbounds for u in 1:n_u
        if model.user_mean[u] != zero(T)
            glm += model.user_mean[u]; cnt += 1
        end
    end
    glm = cnt > 0 ? glm / cnt : zero(T)

    nt = Threads.nthreads()
    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_u, nt)
            rng_u = nzrange(X_csr, u)
            c = length(rng_u)
            items = Vector{Int}(undef, c)
            p = 1
            @inbounds for idx in rng_u
                items[p] = Int(X_csr.colval[idx])
                p += 1
            end
            @inbounds for i in 1:n_i
                S[u, i] = _slopeone_value(model, u, items, glm, c, i)
            end
        end
    end
    S
end

score(model::SlopeOne{T}, X::SparseMatrixCSC) where {T} = predict(model, X)

function score(model::SlopeOne{T}, user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    length(user_indices) == length(item_indices) ||
        throw(DimensionMismatch("user_indices and item_indices must have the same length"))
    # SlopeOne needs each user's own ratings to build the pair sums; with only
    # (u, i) indices there is no way to compute the table prediction. Use the
    # fitted matrix scoring instead.
    throw(ArgumentError(
        "pairwise scoring is not defined for SlopeOne (it needs the user's ratings); " *
        "use predict(model, X) / score(model, X) instead"))
end

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
    weights::SparseMatrixCSC{T,Int}           # W[u, v] = sim(u, v)
    denominator::Vector{T}                    # Σ_v |sim(u, v)|
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
                  Vector{Pair{Int,T}}[], spzeros(T, 0, 0), T[], false)
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
    local_rows = _thread_buffers(() -> Int[], nt)
    local_cols = _thread_buffers(() -> Int[], nt)
    local_vals = _thread_buffers(() -> T[], nt)
    local_den  = [zeros(T, n_users) for _ in 1:nt]

    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
        nu = cnorm[u]
        nu == zero(T) && (neighbors[u] = Pair{Int,T}[])
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
        neighbor_count = length(sims)
        # if fewer than min_k positive neighbors, fall back to the user mean
        if neighbor_count < model.min_k
            neighbors[u] = Pair{Int,T}[]
        else
            neighbors[u] = sims
            den = local_den[chunk]
            for (v, s) in sims
                den[u] += abs(s)
                push!(local_rows[chunk], u)
                push!(local_cols[chunk], v)
                push!(local_vals[chunk], s)
            end
        end
        end
    end

    weights = sparse(reduce(vcat, local_rows), reduce(vcat, local_cols),
                     reduce(vcat, local_vals), n_users, n_users)
    denom = sum(local_den)  # Σ|sim| per user

    model.user_mean = user_mean
    model.centered = centered
    model.neighbors = neighbors
    model.weights = weights
    model.denominator = denom
    model.is_fitted = true
    model
    catch
        model.user_mean, model.centered, model.neighbors = old
        model.weights = spzeros(T, 0, 0)
        model.denominator = T[]
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
    # N[u, i] = Σ_v W[u, v] · C[v, i]; ŷ = r̄_u + N[u,i] / denom[u,i], where
    # denom counts only neighbors v that rated item i (Surprise semantics:
    # a neighbor without a rating contributes nothing to either side).
    N = Matrix{T}(model.weights * model.centered)
    israted = SparseMatrixCSC(size(model.centered, 1), size(model.centered, 2),
                              model.centered.colptr, model.centered.rowval,
                              fill(one(T), nnz(model.centered)))
    denom = Matrix{T}(model.weights * israted)
    @inbounds for u in 1:n_u
        μu = model.user_mean[u]
        for i in 1:n_i
            du = denom[u, i]
            N[u, i] = du > zero(T) ? μu + N[u, i] / du : μu
        end
    end
    N
end