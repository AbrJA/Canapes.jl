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
`dev` + `freq`, plus the `mask` scoring cache (also O(n_items²), stored as `T`)
that makes `predict` a contiguous, cache-friendly column accumulation. Users
without any relevant item fall back to their own mean rating, then the global
mean.

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
    mask::Matrix{T}         # scoring cache: mask[i, j] = freq[i,j] > 0 ? 1 : 0
    user_mean::Vector{T}
    is_fitted::Bool
end

function SlopeOne(; verbose::Bool=true, T::Type{<:AbstractFloat}=Float32)
    SlopeOne{T}(verbose, Matrix{T}(undef, 0, 0), Matrix{Int}(undef, 0, 0),
                Matrix{T}(undef, 0, 0), T[], false)
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

    # Scoring cache: dev[i,j] is already 0 wherever freq[i,j] == 0 (no
    # co-occurrence), so the numerator can accumulate over contiguous columns
    # of `dev` directly; only the relevant-count mask is materialized.
    # Column-major iteration (j outer, i inner) keeps freq/mask access contiguous.
    mask = zeros(T, n_items, n_items)
    @inbounds for j in 1:n_items, i in 1:n_items
        freq[i, j] > 0 && (mask[i, j] = one(T))
    end

    model.dev = dev
    model.freq = freq
    model.mask = mask
    model.user_mean = user_mean
    model.is_fitted = true
    model
    catch
        model.dev, model.freq = old_dev, old_freq
        model.mask = Matrix{T}(undef, 0, 0)
        model.user_mean = old_um
        model.is_fitted = false
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

# Prediction via contiguous column accumulation (cache-friendly):
#   num[i] = Σ_{j ∈ I_u} dev[i, j]      (dev is zero where freq == 0)
#   cnt[i] = Σ_{j ∈ I_u} mask[i, j]     (relevant-item count)
#   ŷ_ui = cnt[i] > 0 ? μ_u + num[i]/cnt[i] : μ_u
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
    num_bufs = _thread_buffers(() -> Vector{T}(undef, n_i), nt)
    cnt_bufs = _thread_buffers(() -> Vector{T}(undef, n_i), nt)
    dev = model.dev
    mask = model.mask

    Threads.@threads for chunk in 1:nt
        num = num_bufs[chunk]
        cbuf = cnt_bufs[chunk]
        for u in _thread_chunk_bounds(chunk, n_u, nt)
            rng_u = nzrange(X_csr, u)
            c = length(rng_u)
            if c == 0
                fill!(view(S, u, :), glm)
                continue
            end
            fill!(num, zero(T))
            fill!(cbuf, zero(T))
            @inbounds for idx in rng_u
                j = Int(X_csr.colval[idx])
                @simd for i in 1:n_i
                    num[i] += dev[i, j]
                    cbuf[i] += mask[i, j]
                end
            end
            μu = model.user_mean[u]
            @inbounds for i in 1:n_i
                S[u, i] = cbuf[i] > zero(T) ? μu + num[i] / cbuf[i] : μu
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
