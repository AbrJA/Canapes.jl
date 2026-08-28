# ──────────────────────────────────────────────────────────────────────────────
# EALS — Element-wise Alternating Least Squares
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: He, Zhang, Kan, Chua (2016/2017)
#   "Fast Matrix Factorization for Online Recommendation with Implicit Feedback"
#   arXiv:1708.05024
#
# Key innovation: Instead of uniform weighting for missing data (as in WMF),
# EALS uses item-popularity-based non-uniform weighting:
#   c_{ui} = c_i^miss  for unobserved entries (popularity-based)
#   c_{ui} = c_i^obs   for observed entries
#
# Element-wise update rule avoids forming the full Gramian per user.
# Instead, it caches Sᵢ = Σ_j c_j * v_j * v_j^T and updates one
# coordinate at a time with O(d) cost per element (vs O(d²) for Cholesky).
#
# This enables:
# 1. Non-uniform weighting without cubic cost
# 2. Efficient incremental updates for online learning
# 3. Lower memory footprint for very large embedding dimensions
# ──────────────────────────────────────────────────────────────────────────────

"""
    EALS{T} <: AbstractMatrixFactorization

Element-wise Alternating Least Squares for implicit feedback recommendation.

Implements the EALS algorithm from He et al. (2016) which uses popularity-based
non-uniform weighting for missing data and coordinate-descent updates that are
O(d) per element instead of O(d³) per user.

# Key Features
- Non-uniform missing data weighting based on item popularity
- Element-wise coordinate descent (O(d) per coordinate update)
- Support for incremental/online updates via `partial_fit!`
- Precomputed caches for efficient iteration

# Constructor
```julia
EALS(; rank=64, λ=0.01, w0=1.0, max_iter=15, convergence_tol=0.005,
       popularity_exponent=0.5, verbose=true)
```

# Fields
- `rank::Int` — embedding dimension
- `λ::T` — L2 regularization strength
- `w0::T` — overall weight for unobserved entries (scales popularity weights)
- `max_iter::Int` — maximum iterations
- `convergence_tol::T` — relative loss change for early stopping (-1 disables)
- `popularity_exponent::T` — exponent for popularity weighting (0.5 = sqrt)
- `user_factors::Matrix{T}` — (rank × n_users) after fitting
- `item_factors::Matrix{T}` — (rank × n_items) after fitting

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.02)
model = EALS(rank=64, λ=0.01, w0=10.0, max_iter=20)
fit!(model, X)
preds = recommend(model, X; k=10)
```
"""
mutable struct EALS{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const w0::T
    const max_iter::Int
    const convergence_tol::T
    const popularity_exponent::T
    const verbose::Bool
    # Factors (rank × n)
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    # Cached: item popularity weights
    item_weights::Vector{T}
    is_fitted::Bool
end

function EALS(;
    rank::Int = 64,
    λ::Float64 = 0.01,
    w0::Float64 = 1.0,
    max_iter::Int = 15,
    convergence_tol::Float64 = 0.005,
    popularity_exponent::Float64 = 0.5,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    w0 > 0.0 || throw(ArgumentError("w0 must be positive, got $w0"))
    popularity_exponent >= 0.0 || throw(ArgumentError("popularity_exponent must be non-negative, got $popularity_exponent"))
    T = dtype
    EALS{T}(rank, T(λ), T(w0), max_iter, T(convergence_tol), T(popularity_exponent), verbose,
            Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), T[], false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::EALS, X; rng, U_init, V_init) -> model

Fit EALS on sparse interaction matrix `X` (users × items).

Uses element-wise coordinate descent with popularity-based non-uniform
weighting for the missing (unobserved) entries.
"""
function fit!(model::EALS{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              U_init::Union{Nothing,Matrix{T}} = nothing,
               V_init::Union{Nothing,Matrix{T}} = nothing,
               callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "EALS")
    old_user_factors = model.user_factors
    old_item_factors = model.item_factors
    old_item_weights = model.item_weights
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    k = model.rank

    # Initialize factors
    if U_init !== nothing
        model.user_factors = copy(U_init)
    else
        model.user_factors = randn(rng, T, k, n_users) .* T(0.01)
    end
    if V_init !== nothing
        model.item_factors = copy(V_init)
    else
        model.item_factors = randn(rng, T, k, n_items) .* T(0.01)
    end

    U = model.user_factors
    V = model.item_factors

    # Compute item popularity weights with c0-mass normalization:
    # c_i = w0 * freq_i^a / Σ_j freq_j^a
    # This matches the EALS weighting scale where w0 controls total missing-data mass.
    model.item_weights = _eals_item_weights(X, model.w0, model.popularity_exponent)
    c_items = model.item_weights

    # Build CSR for row access
    X_csr = to_csr(X)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    λ_val = model.λ::T

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # ── Update user factors ──
        # Precompute S^V = Σ_j c_j * v_j * v_j^T (weighted item Gramian for missing data)
        SV = _eals_weighted_gramian(V, c_items, k, n_items)::Matrix{T}

        _eals_update_users!(U, V, X_csr, SV, c_items, λ_val, k, n_users)

        # ── Update item factors ──
        # Precompute S^U = Σ_u u_u * u_u^T (uniform weight = 1 for users)
        # Actually, for item update: missing weight per user is uniform
        # But observed entries have enhanced weight
        SU = U * U'  # k × k (Gramian of user factors)

        _eals_update_items!(V, U, X, SU, c_items, λ_val, k, n_items)

        # ── Compute loss ──
        loss = _eals_loss(U, V, X, c_items, λ_val)

        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("EALS", iter, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[EALS] converged at iteration $iter"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(iter, Float64(loss), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end

    model.is_fitted = true
    model
    catch
        model.user_factors = old_user_factors
        model.item_factors = old_item_factors
        model.item_weights = old_item_weights
        model.is_fitted = old_is_fitted
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

"""
    update!(model::EALS, X; n_iter=1, rng) -> model

Incremental update: run additional iterations on new or updated data.
Reuses existing factors as warm start.
"""
function update!(model::EALS{T}, X::SparseMatrixCSC{Tv,Ti};
                      n_iter::Int = 1,
                      rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    if !model.is_fitted
        return fit!(model, X; rng=rng)
    end
    n_users, n_items = size(X)
    k = model.rank

    # Resize factors if needed (new users/items)
    if size(model.user_factors, 2) < n_users
        old_n = size(model.user_factors, 2)
        new_cols = randn(rng, T, k, n_users - old_n) .* T(0.01)
        model.user_factors = hcat(model.user_factors, new_cols)
    end
    if size(model.item_factors, 2) < n_items
        old_n = size(model.item_factors, 2)
        new_cols = randn(rng, T, k, n_items - old_n) .* T(0.01)
        model.item_factors = hcat(model.item_factors, new_cols)
    end

    U = model.user_factors
    V = model.item_factors

    # Recompute item weights with c0-mass normalization.
    model.item_weights = _eals_item_weights(X, model.w0, model.popularity_exponent)
    c_items = model.item_weights

    X_csr = to_csr(X)
    λ_val = model.λ::T

    for _ in 1:n_iter
        SV = _eals_weighted_gramian(V, c_items, k, n_items)::Matrix{T}
        _eals_update_users!(U, V, X_csr, SV, c_items, λ_val, k, n_users)
        SU = U * U'
        _eals_update_items!(V, U, X, SU, c_items, λ_val, k, n_items)
    end
    model
end

function _eals_item_weights(X::SparseMatrixCSC{Tv,Ti}, w0::T, a::T) where {Tv,Ti,T<:AbstractFloat}
    n_items = size(X, 2)
    freq = zeros(T, n_items)
    @inbounds for j in 1:n_items
        freq[j] = T(length(nzrange(X, j)))
    end

    pow_freq = a == zero(T) ? ones(T, n_items) : freq .^ a
    mass = sum(pow_freq)

    if !(mass > eps(T))
        return fill(w0 / max(one(T), T(n_items)), n_items)
    end

    # Keep a tiny positive floor for numerical robustness when an item has zero frequency.
    return max.(w0 .* (pow_freq ./ mass), T(1e-12))
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal update functions
# ──────────────────────────────────────────────────────────────────────────────

"""
Compute weighted Gramian: S = Σ_j c_j * v_j * v_j^T
"""
function _eals_weighted_gramian(V::Matrix{T}, c::Vector{T}, k::Int, n::Int) where {T}
    # Compute V * Diag(c) * V' via BLAS syrk: (V .* sqrt(c)') * (V .* sqrt(c)')'
    Vc = similar(V)  # k × n
    @inbounds for j in 1:n
        sc = sqrt(c[j])
        @simd for p in 1:k
            Vc[p, j] = V[p, j] * sc
        end
    end
    S = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), Vc, zero(T), S)
    # Fill lower triangle from upper
    @inbounds for q in 1:k
        for p in (q+1):k
            S[p, q] = S[q, p]
        end
    end
    S
end

"""
Update user factors using element-wise coordinate descent.
For each user u and factor f, the update is:
  u_{uf} = (Σ_i∈R(u) (c_ui - c_i) * v_{if} * r̂_{ui,-f} + b_f) / (Σ_i∈R(u) (c_ui - c_i) * v_{if}² + s_ff + λ)
where r̂_{ui,-f} = r_{ui} - Σ_{g≠f} u_{ug} * v_{ig}
"""
function _eals_update_users!(U::Matrix{T}, V::Matrix{T},
                             X_csr::SparseMatricesCSR.SparseMatrixCSR,
                             SV::Matrix{T}, c_items::Vector{T},
                             λ::T, k::Int, n_users::Int) where {T}
    # Per-thread prediction cache
    nt = Threads.nthreads()
    pred_bufs = _thread_buffers(() -> Vector{T}(undef, 0), nt)

    Threads.@threads for chunk in 1:nt
        for u in _thread_chunk_bounds(chunk, n_users, nt)
        rng_u = nzrange(X_csr, u)
        n_nz = length(rng_u)
        n_nz == 0 && continue

        # Gather observed items and their indices
        # Compute predictions for this user's observed items
        buf = pred_bufs[chunk]
        if length(buf) < n_nz
            pred_bufs[chunk] = Vector{T}(undef, n_nz)
            buf = pred_bufs[chunk]
        end

        # Compute current predictions for observed items
        @inbounds for (pos, idx) in enumerate(rng_u)
            j = Int(X_csr.colval[idx])
            pred = zero(T)
            for f in 1:k
                pred += U[f, u] * V[f, j]
            end
            buf[pos] = pred
        end

        # Element-wise updates for each factor
        @inbounds for f in 1:k
            # Numerator and denominator from missing data (precomputed via SV)
            numer = zero(T)
            denom = SV[f, f] + λ

            # Subtract contribution of current u_f from predictions
            # and accumulate corrections from observed entries
            for (pos, idx) in enumerate(rng_u)
                j = Int(X_csr.colval[idx])
                r_uj = T(X_csr.nzval[idx])
                v_jf = V[f, j]
                cj = c_items[j]

                # Observed confidence: c_obs = 1 + cj
                # Correction vs missing weight: c_diff = c_obs - cj = 1
                c_obs = one(T) + cj

                # pred_no_f = prediction without factor f contribution
                pred_no_f = buf[pos] - U[f, u] * v_jf

                # EALS uses binary preference p=1 for observed entries.
                # Correct formula: c_obs * p_ui - c_diff * pred_no_f
                # where p_ui = 1 and c_diff = c_obs - c_j = 1
                numer += v_jf * (c_obs - pred_no_f)
                denom += v_jf^2  # c_diff = 1
            end

            # Compute SV contribution to numerator (missing data)
            # numer_missing = - Σ_{g≠f} u_{ug} * SV[f,g]
            for g in 1:k
                g == f && continue
                numer -= U[g, u] * SV[f, g]
            end

            # Update the factor
            old_val = U[f, u]
            new_val = numer / denom

            # Update cached predictions
            if old_val != new_val
                delta = new_val - old_val
                for (pos, idx) in enumerate(rng_u)
                    j = Int(X_csr.colval[idx])
                    buf[pos] += delta * V[f, j]
                end
                U[f, u] = new_val
            end
        end
        end
    end
end

"""
Update item factors using element-wise coordinate descent.
"""
function _eals_update_items!(V::Matrix{T}, U::Matrix{T},
                             X::SparseMatrixCSC, SU::Matrix{T},
                             c_items::Vector{T}, λ::T,
                             k::Int, n_items::Int) where {T}
    rv = rowvals(X)
    nz = nonzeros(X)
    n_users = size(U, 2)

    nt = Threads.nthreads()
    pred_bufs = _thread_buffers(() -> Vector{T}(undef, 0), nt)

    Threads.@threads for chunk in 1:nt
        for j in _thread_chunk_bounds(chunk, n_items, nt)
        rng_j = nzrange(X, j)
        n_nz = length(rng_j)
        cj = c_items[j]

        buf = pred_bufs[chunk]
        if length(buf) < n_nz
            pred_bufs[chunk] = Vector{T}(undef, max(n_nz, 64))
            buf = pred_bufs[chunk]
        end

        # Compute current predictions for observed users of this item
        @inbounds for (pos, idx) in enumerate(rng_j)
            u = rv[idx]
            pred = zero(T)
            for f in 1:k
                pred += U[f, u] * V[f, j]
            end
            buf[pos] = pred
        end

        # Element-wise updates for each factor
        @inbounds for f in 1:k
            # From missing data: weighted by cj * SU[f,f]
            denom = cj * SU[f, f] + λ
            numer = zero(T)

            # Missing data contribution (precomputed)
            for g in 1:k
                g == f && continue
                numer -= cj * V[g, j] * SU[f, g]
            end

            # Observed data contributions
            for (pos, idx) in enumerate(rng_j)
                u = rv[idx]
                r_uj = T(nz[idx])
                u_uf = U[f, u]
                c_obs = one(T) + cj

                # pred_no_f = prediction without factor f
                pred_no_f = buf[pos] - V[f, j] * u_uf

                # EALS: binary preference p=1; c_diff = c_obs - c_j = 1
                numer += u_uf * (c_obs - pred_no_f)
                denom += u_uf^2  # c_diff = 1
            end

            old_val = V[f, j]
            new_val = numer / denom

            if old_val != new_val
                delta = new_val - old_val
                for (pos, idx) in enumerate(rng_j)
                    u = rv[idx]
                    buf[pos] += delta * U[f, u]
                end
                V[f, j] = new_val
            end
        end
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────────────

function _eals_loss(U::Matrix{T}, V::Matrix{T}, X::SparseMatrixCSC,
                    c_items::Vector{T}, λ::T) where {T}
    k = size(U, 1)
    n_items = size(V, 2)

    # Loss from observed entries: Σ_{(u,i)∈Ω} (1 + c_i) * (r_{ui} - u^T v)²
    # Loss from observed entries: Σ_{(u,i)∈Ω} (1 + c_i) * (p_{ui} - u^T v)²
    # where p_{ui} = 1 (binary preference for all observed entries)
    loss_obs = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)
    for j in axes(X, 2)
        cj = c_items[j]
        for idx in nzrange(X, j)
            u = rv[idx]
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, u] * V[f, j]
            end
            loss_obs += (one(T) + cj) * (one(T) - pred)^2
        end
    end

    # Loss from unobserved entries (approximated via Gramian trick):
    # Σ_{(u,i)∉Ω} c_i * (u^T v)² ≈ Σ_f Σ_g (Σ_u u_{uf} * u_{ug}) * (Σ_i c_i * v_{if} * v_{ig})
    # = tr(U*U' * S^V) where S^V = Σ_i c_i * v_i * v_i^T
    SV = _eals_weighted_gramian(V, c_items, k, n_items)
    UU = U * U'
    loss_miss = zero(T)
    @inbounds for f in 1:k
        for g in 1:k
            loss_miss += UU[f, g] * SV[f, g]
        end
    end

    # Subtract the observed part that was double-counted in loss_miss
    for j in axes(X, 2)
        cj = c_items[j]
        for idx in nzrange(X, j)
            u = rv[idx]
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, u] * V[f, j]
            end
            loss_miss -= cj * pred^2
        end
    end

    # Regularization
    reg = λ * (sum(abs2, U) + sum(abs2, V))

    loss_obs + loss_miss + reg
end
