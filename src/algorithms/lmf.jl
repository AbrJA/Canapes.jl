# ──────────────────────────────────────────────────────────────────────────────
# Logistic Matrix Factorization (LogisticMF)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Johnson (2014)
#   "Logistic Matrix Factorization for Implicit Feedback Data"
#
# Loss:
#   L = Σ_{u,i} [r_{ui} · xᵤᵀ yᵢ - (1 + α·r_{ui}) · log(1 + exp(xᵤᵀ yᵢ))]
#       - λ/2 (||X||² + ||Y||²)
# ──────────────────────────────────────────────────────────────────────────────

"""
    LogisticMF{T} <: AbstractMatrixFactorization

Logistic Matrix Factorization for implicit feedback via Adagrad with negative sampling.

# Constructor
```julia
LogisticMF(; rank=10, λ=0.6, α=1.0, learning_rate=1.0, max_iter=30,
    n_negative=30, convergence_tol=-1.0, verbose=true)
```

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> model = LogisticMF(rank=8, max_iter=2, verbose=false);

julia> fit!(model, X; rng=MersenneTwister(2));

julia> top_items = recommend(model, X; k=10);

julia> size(top_items)
(200, 10)
```
"""
mutable struct LogisticMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const α::T
    learning_rate::T
    const max_iter::Int
    const n_negative::Int
    const convergence_tol::T
    const verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function LogisticMF(;
    rank::Int = 10,
    λ::Float64 = 0.6,
    α::Float64 = 1.0,
    learning_rate::Float64 = 1.0,
    max_iter::Int = 30,
    n_negative::Int = 30,
    convergence_tol::Float64 = -1.0,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    learning_rate > 0.0 || throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    n_negative >= 1 || throw(ArgumentError("n_negative must be ≥ 1, got $n_negative"))
    Td = dtype
    LogisticMF{Td}(rank, Td(λ), Td(α), Td(learning_rate), max_iter, n_negative, Td(convergence_tol),
            verbose, Matrix{Td}(undef,0,0), Matrix{Td}(undef,0,0), false)
end

"""
    fit!(model::LogisticMF, X; rng) -> model

Fit the LogisticMF model on user-item interaction matrix `X` (n_users × n_items).
Uses per-user/per-item batched Adagrad updates matching the implicit library's approach:
each epoch alternates a user-update phase and an item-update phase, with one batched
gradient accumulation and single Adagrad step per entity.
"""
function fit!(model::LogisticMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    (n_users > 0 && n_items > 0) || throw(ArgumentError(
        "LogisticMF requires positive user and item dimensions, got $(size(X))"))
    nnz(X) > 0 || throw(ArgumentError("LogisticMF requires at least one observed interaction"))
    _require_finite_input(X, "LogisticMF")
    k = model.rank
    old_user_factors = model.user_factors
    old_item_factors = model.item_factors
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try

    # Standard normal initialization (matching implicit)
    model.user_factors = randn(rng, T, k, n_users)
    model.item_factors = randn(rng, T, k, n_items)

    U = model.user_factors  # k × n_users
    V = model.item_factors  # k × n_items

    # Build CSR for user→item access and CSC (= item→user) access
    X_csr = to_csr(X)
    n_interactions = nnz(X)

    # Flat interaction arrays for negative sampling (popularity-biased, like implicit)
    all_items = Vector{Int32}(undef, n_interactions)
    pos = 1
    for u in 1:n_users
        for idx in nzrange(X_csr, u)
            all_items[pos] = Int32(X_csr.colval[idx])
            pos += 1
        end
    end

    # Item→user CSR (transpose structure)
    Xt = sparse(X')  # n_items × n_users
    Xt_csr = to_csr(Xt)

    # Flat arrays for item-side negative sampling
    all_users = Vector{Int32}(undef, n_interactions)
    pos = 1
    for j in 1:n_items
        for idx in nzrange(Xt_csr, j)
            all_users[pos] = Int32(Xt_csr.colval[idx])
            pos += 1
        end
    end

    lr = model.learning_rate
    λ  = model.λ
    n_neg = model.n_negative
    ada_eps = T(1e-6)

    # Adagrad accumulators
    grad2_U = zeros(T, k, n_users)::Matrix{T}
    grad2_V = zeros(T, k, n_items)::Matrix{T}

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    # Per-thread RNGs and pre-allocated gradient buffers (zero alloc inner loop)
    nt = Threads.nthreads()
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nt]
    deriv_bufs = [Vector{T}(undef, k) for _ in 1:nt]

    for epoch in 1:model.max_iter
        epoch_start = time_ns()

        # ── Phase 1: Update user factors (items fixed) ──
        _lmf_update_users!(U, V, X_csr, all_items, grad2_U,
                           lr, λ, n_neg, ada_eps, k, n_users, n_interactions,
                           thread_rngs, deriv_bufs)

        # ── Phase 2: Update item factors (users fixed) ──
        _lmf_update_items!(V, U, Xt_csr, all_users, grad2_V,
                           lr, λ, n_neg, ada_eps, k, n_items, n_interactions,
                           thread_rngs, deriv_bufs)

        # ── Compute epoch loss (sampled estimate) ──
        loss = _lmf_loss_estimate(U, V, X_csr, n_users, k)

        iter_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = elapsed_seconds(monitor)
        if model.verbose
            log_iteration("LogisticMF", epoch, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[LogisticMF] converged at iteration $epoch"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(epoch, Float64(loss), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end
    model.is_fitted = true
    model
    catch
        model.user_factors = old_user_factors
        model.item_factors = old_item_factors
        model.is_fitted = old_is_fitted
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

"""
Update all user factors with one batched Adagrad step per user.
Matches implicit's lmf_update: accumulate gradient from positives + negatives + reg,
then single Adagrad update. Uses pre-allocated per-chunk buffers for zero allocation.
"""
function _lmf_update_users!(U::Matrix{T}, V::Matrix{T},
                            X_csr::SparseMatricesCSR.SparseMatrixCSR,
                            all_items::Vector{Int32},
                            grad2_U::Matrix{T},
                            lr::T, λ::T, n_neg::Int, ada_eps::T,
                            k::Int, n_users::Int, n_interactions::Int,
                            thread_rngs::Vector, deriv_bufs::Vector{Vector{T}}) where {T}
    n_items = size(V, 2)
    nt = Threads.nthreads()
    chunk_size = cld(n_users, nt)

    Threads.@threads for chunk in 1:nt
        local_rng = thread_rngs[chunk]
        deriv = deriv_bufs[chunk]
        for u in ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, n_users)
        rng_u = nzrange(X_csr, u)
        user_seen = length(rng_u)
        user_seen == 0 && continue

        # Use pre-allocated buffer (zero alloc)
        @inbounds @simd for f in 1:k
            deriv[f] = zero(T)
        end

        # Fused positive pass: deriv += c * (1 - σ(s)) * v = c * σ(-s) * v
        @inbounds for idx in rng_u
            j = Int(X_csr.colval[idx])
            c_uj = T(X_csr.nzval[idx])
            # Compute dot product
            s = zero(T)
            for g in 1:k
                s = muladd(U[g, u], V[g, j], s)
            end
            # c * (1 - σ(s)) = c * σ(-s)
            z = c_uj / (one(T) + exp(s))
            @simd for f in 1:k
                deriv[f] += z * V[f, j]
            end
        end

        # Negatives: deriv -= σ(s) * v
        # Sample min(n_items, seen * n_neg) negatives like implicit (lmf.pyx:
        # `range(min(n_items, user_seen_item * neg_prop))`), with rejection
        # sampling so a user's own interactions are never drawn.
        n_neg_samples = min(n_items, user_seen * n_neg)
        @inbounds for _ in 1:n_neg_samples
            j = _lmf_sample_unobserved(local_rng, n_items, X_csr, u,
                                        all_items, n_interactions)
            s = zero(T)
            for g in 1:k
                s = muladd(U[g, u], V[g, j], s)
            end
            σ_s = one(T) / (one(T) + exp(-s))
            @simd for f in 1:k
                deriv[f] -= σ_s * V[f, j]
            end
        end

        # Regularization + Adagrad update (fused)
        @inbounds @simd for f in 1:k
            d = deriv[f] - λ * U[f, u]
            grad2_U[f, u] += d * d
            U[f, u] += (lr / sqrt(ada_eps + grad2_U[f, u])) * d
        end
        end
    end
end

"""
Update all item factors with one batched Adagrad step per item.
Uses pre-allocated per-chunk buffers for zero allocation.
"""
function _lmf_update_items!(V::Matrix{T}, U::Matrix{T},
                            Xt_csr::SparseMatricesCSR.SparseMatrixCSR,
                            all_users::Vector{Int32},
                            grad2_V::Matrix{T},
                            lr::T, λ::T, n_neg::Int, ada_eps::T,
                            k::Int, n_items::Int, n_interactions::Int,
                            thread_rngs::Vector, deriv_bufs::Vector{Vector{T}}) where {T}
    n_users = size(U, 2)
    nt = Threads.nthreads()
    chunk_size = cld(n_items, nt)

    Threads.@threads for chunk in 1:nt
        local_rng = thread_rngs[chunk]
        deriv = deriv_bufs[chunk]
        for j in ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, n_items)
        rng_j = nzrange(Xt_csr, j)
        item_seen = length(rng_j)
        item_seen == 0 && continue

        @inbounds @simd for f in 1:k
            deriv[f] = zero(T)
        end

        # Fused positive pass: deriv += c * (1 - σ(s)) * u = c * σ(-s) * u
        @inbounds for idx in rng_j
            u = Int(Xt_csr.colval[idx])
            c_uj = T(Xt_csr.nzval[idx])
            s = zero(T)
            for g in 1:k
                s = muladd(U[g, u], V[g, j], s)
            end
            z = c_uj / (one(T) + exp(s))
            @simd for f in 1:k
                deriv[f] += z * U[f, u]
            end
        end

        # Negatives: deriv -= σ(s) * u
        # Sample min(n_users, seen * n_neg) negatives like implicit (lmf.pyx),
        # with rejection sampling so an item's own interactions are never drawn.
        n_neg_samples = min(n_users, item_seen * n_neg)
        @inbounds for _ in 1:n_neg_samples
            u = _lmf_sample_unobserved(local_rng, n_users, Xt_csr, j,
                                        all_users, n_interactions)
            s = zero(T)
            for g in 1:k
                s = muladd(U[g, u], V[g, j], s)
            end
            σ_s = one(T) / (one(T) + exp(-s))
            @simd for f in 1:k
                deriv[f] -= σ_s * U[f, u]
            end
        end

        # Regularization + Adagrad update (fused)
        @inbounds @simd for f in 1:k
            d = deriv[f] - λ * V[f, j]
            grad2_V[f, j] += d * d
            V[f, j] += (lr / sqrt(ada_eps + grad2_V[f, j])) * d
        end
        end
    end
end

"""Sample from `pool` while excluding entries observed by `entity`."""
@inline function _lmf_sample_unobserved(rng::AbstractRNG, limit::Int,
                                        observed::SparseMatricesCSR.SparseMatrixCSR,
                                        entity::Int, pool::Vector{Int32},
                                        n_pool::Int)
    # Rejection sampling is cheap for sparse rows and bounded for dense ones.
    for _ in 1:min(16, limit)
        candidate = Int(pool[rand(rng, 1:n_pool)])
        is_observed = false
        for idx in nzrange(observed, entity)
            if Int(observed.colval[idx]) == candidate
                is_observed = true
                break
            end
        end
        !is_observed && return candidate
    end

    # Start at a random offset so the fallback does not always favor low IDs.
    start = rand(rng, 1:limit)
    for offset in 0:(limit - 1)
        candidate = mod1(start + offset, limit)
        is_observed = false
        for idx in nzrange(observed, entity)
            if Int(observed.colval[idx]) == candidate
                is_observed = true
                break
            end
        end
        !is_observed && return candidate
    end
    throw(ArgumentError("no unobserved entity is available for LogisticMF negative sampling"))
end

function _lmf_loss_estimate(U::Matrix{T}, V::Matrix{T},
                            X_csr::SparseMatricesCSR.SparseMatrixCSR, n_users::Int, k::Int) where {T}
    nt = Threads.nthreads()
    partial = zeros(T, nt)
    chunk_size = cld(n_users, nt)
    Threads.@threads for chunk in 1:nt
        @fastmath @inbounds for u in ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, n_users)
            for idx in nzrange(X_csr, u)
                j = Int(X_csr.colval[idx])
                s = zero(T)
                @simd for f in 1:k
                    s += U[f, u] * V[f, j]
                end
                partial[chunk] -= log(one(T) / (one(T) + exp(-s)) + T(1e-10))
            end
        end
    end
    sum(partial) / max(one(T), T(nnz(X_csr)))
end
