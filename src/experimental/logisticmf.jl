
# ──────────────────────────────────────────────────────────────────────────────
# Logistic Matrix Factorization (LogisticMF)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Johnson (2014)
#   "Logistic Matrix Factorization for Implicit Feedback Data"
#
# Loss:
#   L = Σ_{u,i} [α·r_{ui}·(xᵤᵀ yᵢ + β_u + β_i)
#                − (1 + α·r_{ui})·log(1 + exp(xᵤᵀ yᵢ + β_u + β_i))]
#       − λ/2 (||X||² + ||Y||²)
#
# Layout follows implicit's lmf.pyx: factors are rank+2 dimensional. The user's
# second-to-last column is pinned to 1 and the item's last column is pinned to
# 1, so the dot includes the learned item bias + user bias (paper eq. 1):
#   s = xᵤᵀ yᵢ + β_u + β_i
# ──────────────────────────────────────────────────────────────────────────────

"""
    LogisticMF{T} <: AbstractMatrixFactorization

Logistic Matrix Factorization for implicit feedback via Adagrad with negative sampling.

The score of a user-item pair includes per-user and per-item bias terms
(Johnson 2014, eq. 1): factors are stored with `rank + 2` rows, the user's
second-to-last column and the item's last column are pinned to 1 (matching
implicit's `lmf.pyx` layout), so `score = xᵤᵀ yᵢ + β_u + β_i`.

# Constructor
```julia
LogisticMF(; rank=10, λ=0.6, α=1.0, lr=1.0, max_iter=30,
    n_negative=30, tol=-1.0, optimizer=:adagrad, verbose=true)
```

`α` scales the confidence of positive observations (`c = α·r`, paper eq. 2-3).
The default `α=1.0` reproduces implicit's logisticmatrixfactorization exactly.

# Optimizer
`optimizer=:adagrad` is the default and reproduces implicit's `lmf.pyx`
exactly: per-entity Adagrad with an unbounded squared-gradient accumulator, so
effective steps shrink as `lr/√Σg²` and low learning rates barely move the
factors (implicit itself hardcodes `lr=1.0`). `optimizer=:rmsprop` replaces the
accumulator with an EMA (`v ← 0.9·v + 0.1·g²`, decoupled from iteration count)
— more robust to small learning rates and warm-start friendly, at the cost of
no reference parity with implicit. `recommend`/`score` are unaffected by the
choice.

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
    lr::T
    const max_iter::Int
    const n_negative::Int
    const tol::T
    const optimizer::Symbol
    const verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function LogisticMF(;
    rank::Int = 10,
    λ::Float64 = 0.6,
    α::Float64 = 1.0,
    lr::Float64 = 1.0,
    max_iter::Int = 30,
    n_negative::Int = 30,
    tol::Float64 = -1.0,
    optimizer::Symbol = :adagrad,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    lr > 0.0 || throw(ArgumentError("lr must be positive, got $lr"))
    n_negative >= 1 || throw(ArgumentError("n_negative must be ≥ 1, got $n_negative"))
    optimizer in (:adagrad, :rmsprop) || throw(ArgumentError(
        "optimizer must be :adagrad or :rmsprop, got $optimizer"))
    Td = T
    LogisticMF{Td}(rank, Td(λ), Td(α), Td(lr), max_iter, n_negative, Td(tol),
            optimizer, verbose, Matrix{Td}(undef,0,0), Matrix{Td}(undef,0,0), false)
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
    kf = k + 2  # rank dims + user/item bias (implicit's factors+2 layout)
    old_user_factors = model.user_factors
    old_item_factors = model.item_factors
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try

    # Standard normal initialization (matching implicit)
    model.user_factors = randn(rng, T, kf, n_users)
    model.item_factors = randn(rng, T, kf, n_items)

    U = model.user_factors  # kf × n_users
    V = model.item_factors  # kf × n_items

    # Build CSR for user→item access and CSC (= item→user) access
    X_csr = to_csr(X)

    # Occurrence counts for popularity-weighted negatives (implicit's pool
    # distribution: P(item) ∝ times-observed). Pool draws cost one random
    # dereference into a 20MB+ array per sample; the alias tables below
    # reproduce the exact same distribution from L1/L2-resident arrays
    # (benfred/implicit#745: uniform-over-catalogue sampling measured worse on
    # MovieLens-100k — P@10 0.0853 vs 0.0981, MAP@10 0.0476 vs 0.0505 — so the
    # popularity weighting is kept).
    item_counts = zeros(Int, n_items)
    for u in 1:n_users
        for idx in nzrange(X_csr, u)
            item_counts[Int(X_csr.colval[idx])] += 1
        end
    end
    alias_p_items, alias_items = _lmf_build_alias(item_counts, T)

    # Item→user CSR (transpose structure)
    Xt = sparse(X')  # n_items × n_users
    Xt_csr = to_csr(Xt)

    # User-side alias tables (item phase samples negatives over users)
    user_counts = zeros(Int, n_users)
    for j in 1:n_items
        for idx in nzrange(Xt_csr, j)
            user_counts[Int(Xt_csr.colval[idx])] += 1
        end
    end
    alias_p_users, alias_users = _lmf_build_alias(user_counts, T)

    # Bias dims (implicit's lmf.pyx): the user's second-to-last factor column is
    # pinned to 1 — its dot with the item's last column (the learned item bias)
    # yields β_i — and the item's last column is pinned to 1, yielding the
    # learned user bias β_u:  s = xᵤᵀ yᵢ + β_u + β_i.
    U[kf - 1, :] .= one(T)
    V[kf, :] .= one(T)
    # Entities without observations start at zero (implicit parity); the kernels
    # skip them, so they never evolve.
    for u in 1:n_users
        isempty(nzrange(X_csr, u)) && (U[:, u] .= zero(T))
    end
    for j in 1:n_items
        isempty(nzrange(Xt_csr, j)) && (V[:, j] .= zero(T))
    end

    lr = model.lr
    λ  = model.λ
    α  = model.α
    n_neg = model.n_negative
    opt = model.optimizer
    ada_eps = T(1e-6)

    # Adagrad accumulators
    grad2_U = zeros(T, kf, n_users)::Matrix{T}
    grad2_V = zeros(T, kf, n_items)::Matrix{T}

    monitor = ConvergenceMonitor{T}(tol=T(model.tol), min_iter=2)

    # Per-thread RNGs and pre-allocated gradient buffers (zero alloc inner loop)
    nt = Threads.nthreads()
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nt]
    deriv_bufs = [Vector{T}(undef, kf) for _ in 1:nt]
    # Gather buffers for the BLAS-based entity updates: the columns of the
    # fixed matrix are collected into a contiguous per-thread buffer (kf × max
    # negatives) and processed with GEMVs; the random column loads are paid
    # once per entity update.
    max_seen_user = maximum(length(nzrange(X_csr, u)) for u in 1:n_users; init=0)
    max_seen_item = maximum(length(nzrange(Xt_csr, j)) for j in 1:n_items; init=0)
    max_m = max(min(n_items, max_seen_user * n_neg),
                min(n_users, max_seen_item * n_neg))
    col_bufs = [Matrix{T}(undef, kf, max_m) for _ in 1:nt]
    idx_bufs = [Vector{Int32}(undef, max_m) for _ in 1:nt]
    score_bufs = [Vector{T}(undef, max_m) for _ in 1:nt]

    for epoch in 1:model.max_iter
        epoch_start = time_ns()

        # ── Phase 1: Update user factors (items fixed) ──
        _lmf_update_users!(U, V, X_csr, alias_p_items, alias_items, grad2_U,
                           lr, λ, α, n_neg, ada_eps, opt, kf, n_users, n_items,
                           thread_rngs, deriv_bufs, col_bufs, idx_bufs, score_bufs)
        U[kf - 1, :] .= one(T)  # re-pin user bias column (implicit parity)

        # ── Phase 2: Update item factors (users fixed) ──
        _lmf_update_items!(V, U, Xt_csr, alias_p_users, alias_users, grad2_V,
                           lr, λ, α, n_neg, ada_eps, opt, kf, n_items, n_users,
                           thread_rngs, deriv_bufs, col_bufs, idx_bufs, score_bufs)
        V[kf, :] .= one(T)  # re-pin item bias column (implicit parity)

        # ── Compute epoch loss (sampled estimate) ──
        loss = _lmf_loss_estimate(U, V, X_csr, n_users, kf)

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
                            alias_p::Vector{T}, alias_items::Vector{Int},
                            grad2_U::Matrix{T},
                            lr::T, λ::T, α::T, n_neg::Int, ada_eps::T, opt::Symbol,
                            kf::Int, n_users::Int, n_items::Int,
                            thread_rngs::Vector{Random.Xoshiro}, deriv_bufs::Vector{Vector{T}},
                            col_bufs::Vector{Matrix{T}},
                            idx_bufs::Vector{Vector{Int32}},
                            score_bufs::Vector{Vector{T}}) where {T}
    nt = Threads.nthreads()
    chunk_size = cld(n_users, nt)

    Threads.@threads for chunk in 1:nt
        local_rng = thread_rngs[chunk]
        deriv = deriv_bufs[chunk]
        colbuf = col_bufs[chunk]
        idxbuf = idx_bufs[chunk]
        zbuf = score_bufs[chunk]
        for u in ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, n_users)
        rng_u = nzrange(X_csr, u)
        user_seen = length(rng_u)
        user_seen == 0 && continue

        # ── Positive pass via gather + BLAS ──
        # Gather the user's seen item columns into a contiguous buffer (the
        # random column loads are paid once), then score with GEMV and
        # accumulate the weighted column sum with a second GEMV.
        @inbounds for (t, idx) in enumerate(rng_u)
            j = Int(X_csr.colval[idx])
            for f in 1:kf
                colbuf[f, t] = V[f, j]
            end
        end
        Ucol = @view U[:, u]
        colsub = @view colbuf[:, 1:user_seen]
        BLAS.gemv!('T', one(T), colsub, Ucol, zero(T), @view(zbuf[1:user_seen]))
        r0 = first(rng_u)
        # z = c * σ(-s) with c = α·r (paper confidence; α=1 reproduces implicit)
        @inbounds @simd for t in 1:user_seen
            zbuf[t] = α * T(X_csr.nzval[r0 + t - 1]) / (one(T) + exp(zbuf[t]))
        end
        BLAS.gemv!('N', one(T), colsub, @view(zbuf[1:user_seen]), zero(T), deriv)

        # ── Negatives: sample to buffer first, then gather + GEMV. ──
        # Sample min(n_items, seen * n_neg) negatives like implicit (lmf.pyx),
        # popularity-weighted via the alias tables (identical distribution to
        # implicit's flat-observation-pool draw).
        n_neg_samples = min(n_items, user_seen * n_neg)
        @inbounds for s in 1:n_neg_samples
            idxbuf[s] = Int32(_lmf_sample_alias(local_rng, alias_p, alias_items))
        end
        colsubn = @view colbuf[:, 1:n_neg_samples]
        @inbounds for s in 1:n_neg_samples
            j = Int(idxbuf[s])
            for f in 1:kf
                colbuf[f, s] = V[f, j]
            end
        end
        BLAS.gemv!('T', one(T), colsubn, Ucol, zero(T), @view(zbuf[1:n_neg_samples]))
        # -σ(s): negatives are subtracted (deriv -= σ·v)
        @inbounds @simd for s in 1:n_neg_samples
            zbuf[s] = -one(T) / (one(T) + exp(-zbuf[s]))
        end
        BLAS.gemv!('N', one(T), colsubn, @view(zbuf[1:n_neg_samples]), one(T), deriv)

        # Regularization + optimizer update (fused: :adagrad matches implicit,
        # :rmsprop uses the EMA accumulator v ← 0.9v + 0.1g²)
        if opt === :rmsprop
            @inbounds @simd for f in 1:kf
                d = deriv[f] - λ * U[f, u]
                g2 = grad2_U[f, u]
                g2 = T(0.9) * g2 + T(0.1) * d * d
                grad2_U[f, u] = g2
                U[f, u] += (lr / sqrt(ada_eps + g2)) * d
            end
        else
            @inbounds @simd for f in 1:kf
                d = deriv[f] - λ * U[f, u]
                grad2_U[f, u] += d * d
                U[f, u] += (lr / sqrt(ada_eps + grad2_U[f, u])) * d
            end
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
                            alias_p::Vector{T}, alias_users::Vector{Int},
                            grad2_V::Matrix{T},
                            lr::T, λ::T, α::T, n_neg::Int, ada_eps::T, opt::Symbol,
                            kf::Int, n_items::Int, n_users::Int,
                            thread_rngs::Vector{Random.Xoshiro}, deriv_bufs::Vector{Vector{T}},
                            col_bufs::Vector{Matrix{T}},
                            idx_bufs::Vector{Vector{Int32}},
                            score_bufs::Vector{Vector{T}}) where {T}
    nt = Threads.nthreads()
    chunk_size = cld(n_items, nt)

    Threads.@threads for chunk in 1:nt
        local_rng = thread_rngs[chunk]
        deriv = deriv_bufs[chunk]
        colbuf = col_bufs[chunk]
        idxbuf = idx_bufs[chunk]
        zbuf = score_bufs[chunk]
        for j in ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, n_items)
        rng_j = nzrange(Xt_csr, j)
        item_seen = length(rng_j)
        item_seen == 0 && continue

        # ── Positive pass via gather + BLAS ──
        @inbounds for (t, idx) in enumerate(rng_j)
            u = Int(Xt_csr.colval[idx])
            for f in 1:kf
                colbuf[f, t] = U[f, u]
            end
        end
        Vcol = @view V[:, j]
        colsub = @view colbuf[:, 1:item_seen]
        BLAS.gemv!('T', one(T), colsub, Vcol, zero(T), @view(zbuf[1:item_seen]))
        r0 = first(rng_j)
        # z = c * σ(-s) with c = α·r (paper confidence; α=1 reproduces implicit)
        @inbounds @simd for t in 1:item_seen
            zbuf[t] = α * T(Xt_csr.nzval[r0 + t - 1]) / (one(T) + exp(zbuf[t]))
        end
        BLAS.gemv!('N', one(T), colsub, @view(zbuf[1:item_seen]), zero(T), deriv)

        # ── Negatives: sample to buffer first (same draws), then gather + GEMV ──
        n_neg_samples = min(n_users, item_seen * n_neg)
        @inbounds for s in 1:n_neg_samples
            idxbuf[s] = Int32(_lmf_sample_alias(local_rng, alias_p, alias_users))
        end
        colsubn = @view colbuf[:, 1:n_neg_samples]
        @inbounds for s in 1:n_neg_samples
            u = Int(idxbuf[s])
            for f in 1:kf
                colbuf[f, s] = U[f, u]
            end
        end
        BLAS.gemv!('T', one(T), colsubn, Vcol, zero(T), @view(zbuf[1:n_neg_samples]))
        # -σ(s): negatives are subtracted (deriv -= σ·u)
        @inbounds @simd for s in 1:n_neg_samples
            zbuf[s] = -one(T) / (one(T) + exp(-zbuf[s]))
        end
        BLAS.gemv!('N', one(T), colsubn, @view(zbuf[1:n_neg_samples]), one(T), deriv)

        # Regularization + optimizer update (fused: :adagrad matches implicit,
        # :rmsprop uses the EMA accumulator v ← 0.9v + 0.1g²)
        if opt === :rmsprop
            @inbounds @simd for f in 1:kf
                d = deriv[f] - λ * V[f, j]
                g2 = grad2_V[f, j]
                g2 = T(0.9) * g2 + T(0.1) * d * d
                grad2_V[f, j] = g2
                V[f, j] += (lr / sqrt(ada_eps + g2)) * d
            end
        else
            @inbounds @simd for f in 1:kf
                d = deriv[f] - λ * V[f, j]
                grad2_V[f, j] += d * d
                V[f, j] += (lr / sqrt(ada_eps + grad2_V[f, j])) * d
            end
        end
        end
    end
end

"""Build a Walker/Vose alias table for popularity-weighted negative sampling.

`P(item/entity e) ∝ counts[e]` — the identical distribution to implicit's
flat-observation-pool draw (`pool[rand(1:nnz)]`) — but sampleable in O(1) from
two L1/L2-resident arrays instead of one random dereference into the full-size
interaction pool (which costs an extra cache miss per sample).
"""
function _lmf_build_alias(counts::Vector{Int}, ::Type{T}) where {T<:AbstractFloat}
    n = length(counts)
    total = sum(counts)
    p = Vector{T}(undef, n)
    alias = fill(-1, n)
    small = Int[]
    large = Int[]
    @inbounds for i in 1:n
        p[i] = T(counts[i] * n / total)
        p[i] < 1 ? push!(small, i) : push!(large, i)
    end
    while !isempty(small) && !isempty(large)
        s = pop!(small)
        l = pop!(large)
        alias[s] = l
        p[l] -= (one(T) - p[s])
        p[l] < 1 ? push!(small, l) : push!(large, l)
    end
    # entries left on either side after the balancing loop alias to themselves
    # (their mass is 1 up to rounding; the clamp keeps u < p[i] almost surely)
    @inbounds for i in 1:n
        p[i] = min(p[i], one(T) - eps(T))
        alias[i] < 0 && (alias[i] = i)
    end
    (p, alias)
end

"""Sample a popularity-weighted negative via the alias tables `(p, alias)`."""
@inline function _lmf_sample_alias(rng::AbstractRNG, p::Vector{T}, alias::Vector{Int}) where {T}
    n = length(p)
    i = rand(rng, UInt32(0):UInt32(n - 1)) + UInt32(1)
    u = rand(rng, T)
    Int(u < p[i] ? i : alias[i])
end

function _lmf_loss_estimate(U::Matrix{T}, V::Matrix{T},
                            X_csr::SparseMatricesCSR.SparseMatrixCSR, n_users::Int, kf::Int) where {T}
    nt = Threads.nthreads()
    partial = zeros(T, nt)
    chunk_size = cld(n_users, nt)
    Threads.@threads for chunk in 1:nt
        @fastmath @inbounds for u in ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, n_users)
            for idx in nzrange(X_csr, u)
                j = Int(X_csr.colval[idx])
                s = zero(T)
                @simd for f in 1:kf
                    s += U[f, u] * V[f, j]
                end
                partial[chunk] -= log(one(T) / (one(T) + exp(-s)) + T(1e-10))
            end
        end
    end
    sum(partial) / max(one(T), T(nnz(X_csr)))
end
