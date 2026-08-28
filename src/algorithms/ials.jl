# ──────────────────────────────────────────────────────────────────────────────
# IALS — Implicit Alternating Least Squares with Subspace Optimization
# ──────────────────────────────────────────────────────────────────────────────
#
# References:
#   - Hu, Koren, Volinsky (2008): "Collaborative Filtering for Implicit Feedback Datasets"
#   - Rendle, Krichene, Zhang, Koren (2021): "IALS++: Speeding up Matrix Factorization
#     with Subspace Optimization" (arXiv:2110.14044)
#
# Key insight from IALS++: Instead of solving the full d×d linear system per user/item,
# use a subspace solver that operates on a smaller k-dimensional block, leveraging
# the structure of the Gramian matrix that is shared across all users/items.
#
# Loss (implicit feedback):
#   L = Σ_{u,i} c_{ui} (p_{ui} - xᵤᵀyᵢ)² + λ(‖X‖² + ‖Y‖²)
#
# where c_{ui} = 1 + α·r_{ui}, p_{ui} = 1 if r_{ui}>0 else 0
# ──────────────────────────────────────────────────────────────────────────────

"""
    IALS{T} <: AbstractMatrixFactorization

Implicit Alternating Least Squares with efficient Gramian caching.

Implements the algorithm from Hu et al. (2008) with the efficient Gramian trick:
instead of forming the full confidence-weighted system per user, we precompute
the item Gramian `YᵀY` and add only the diagonal corrections for non-zero entries.

This avoids the O(n_items × d²) cost per user and replaces it with
O(nnz_per_user × d² + d³) per user, which is dramatically faster for sparse data.

# Solver Options
- `CholeskySolver()` — exact solve via Cholesky decomposition, O(d³) per user. Best for d ≤ 128.
- `ConjugateGradient()` — approximate solve via Conjugate Gradient, O(d² × cg_steps) per user.
  Best for large d (≥ 128). Uses warm-start from previous iteration's solution.

# Constructor
```julia
IALS(; rank=64, λ=0.01, α=40.0, max_iter=15, convergence_tol=0.005,
       solver=CholeskySolver(), cg_steps=3, verbose=true)
```

# Fields
- `rank::Int` — embedding dimension
- `λ::T` — L2 regularization
- `α::T` — confidence scaling: c_{ui} = 1 + α·r_{ui}
- `max_iter::Int` — maximum ALS iterations
- `convergence_tol::T` — relative change in loss for early stopping (-1 disables)
- `solver::ALSSolver` — `CholeskySolver()` or `ConjugateGradient()`
- `cg_steps::Int` — CG inner iterations (only for ConjugateGradient() solver)
- `user_factors::Matrix{T}` — (rank × n_users) after fitting
- `item_factors::Matrix{T}` — (rank × n_items) after fitting
"""
mutable struct IALS{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const α::T
    const max_iter::Int
    const convergence_tol::T
    const solver::ALSSolver
    const cg_steps::Int
    const verbose::Bool
    # Factors (rank × n)
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function IALS(;
    rank::Int = 64,
    λ::Float64 = 0.01,
    α::Float64 = 40.0,
    max_iter::Int = 15,
    convergence_tol::Float64 = 0.005,
    solver::ALSSolver = CholeskySolver(),
    cg_steps::Int = 3,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    α >= 0.0 || throw(ArgumentError("α must be non-negative, got $α"))
    solver isa Union{CholeskySolver, ConjugateGradient} || throw(ArgumentError("IALS supports only CholeskySolver() or ConjugateGradient() solvers"))
    cg_steps >= 1 || throw(ArgumentError("cg_steps must be ≥ 1, got $cg_steps"))
    T = dtype
    IALS{T}(rank, T(λ), T(α), max_iter, T(convergence_tol), solver, cg_steps, verbose,
            Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::IALS, X; rng, U_init, V_init) -> model

Fit IALS on sparse interaction matrix `X` (users × items).

Uses the efficient Gramian-caching approach: precomputes `YᵀY` (or `XᵀX`) once
per iteration, then adds per-user diagonal corrections from non-zero entries.
"""
function fit!(model::IALS{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              U_init::Union{Nothing,AbstractMatrix} = nothing,
               V_init::Union{Nothing,AbstractMatrix} = nothing,
               callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "IALS")
    old_user_factors = model.user_factors
    old_item_factors = model.item_factors
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    k = model.rank
    α = model.α
    λ = model.λ

    # Initialize factors
    if U_init !== nothing
        model.user_factors = Matrix{T}(U_init)
    else
        model.user_factors = randn(rng, T, k, n_users) .* T(0.01)
    end
    if V_init !== nothing
        model.item_factors = Matrix{T}(V_init)
    else
        model.item_factors = randn(rng, T, k, n_items) .* T(0.01)
    end

    U = model.user_factors
    V = model.item_factors

    # CSR for fast row access (user rows)
    X_csr = to_csr(X)
    # CSC is already column-oriented (item columns)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    # Pre-allocate per-thread work buffers
    nt = Threads.nthreads()
    A_bufs = [Matrix{T}(undef, k, k) for _ in 1:nt]
    b_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    # CG-specific buffers
    r_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    p_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    Ap_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    # Batched gather buffers: Z (k × max_nnz_per_entity), w (2*max_nnz_per_entity for CG temp)
    max_nnz = max(maximum(diff(X.colptr)), maximum(diff(X_csr.rowptr)))
    Z_bufs = [Matrix{T}(undef, k, max_nnz) for _ in 1:nt]
    w_bufs = [Vector{T}(undef, 2 * max_nnz) for _ in 1:nt]

    use_cg = model.solver isa ConjugateGradient
    cg_steps = model.cg_steps

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # ── Update users: fix V, solve for U ──
        if use_cg
            _ials_update_factors_cg!(U, V, X_csr, α, λ, k, cg_steps,
                                     A_bufs, b_bufs, r_bufs, p_bufs, Ap_bufs,
                                     Z_bufs, w_bufs)
        else
            _ials_update_factors!(U, V, X_csr, α, λ, k, A_bufs, b_bufs,
                                  Z_bufs, w_bufs)
        end

        # ── Update items: fix U, solve for V ──
        if use_cg
            _ials_update_factors_cg!(V, U, X, α, λ, k, cg_steps,
                                     A_bufs, b_bufs, r_bufs, p_bufs, Ap_bufs,
                                     Z_bufs, w_bufs)
        else
            _ials_update_factors!(V, U, X, α, λ, k, A_bufs, b_bufs,
                                  Z_bufs, w_bufs)
        end

        # ── Compute loss ──
        loss = _ials_loss(U, V, X, α, λ)

        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("IALS", iter, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[IALS] converged at iteration $iter"
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
        model.is_fitted = old_is_fitted
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

"""
Update all factors in `target` given fixed `source` factors and sparse matrix `R`.
Dispatches on CSR (row access for user updates) or CSC (column access for item updates).

Uses batched BLAS: gathers item vectors per entity, then single syrk! + gemv! calls
instead of per-item syr! + axpy! (reduces BLAS call count by ~100x).
"""
function _ials_update_factors!(target::Matrix{T}, source::Matrix{T},
                               R::SparseMatricesCSR.SparseMatrixCSR, α::T, λ::T, k::Int,
                               A_bufs::Vector{Matrix{T}},
                               b_bufs::Vector{Vector{T}},
                               Z_bufs::Vector{Matrix{T}},
                               w_bufs::Vector{Vector{T}}) where {T}
    n = size(target, 2)
    colval = R.colval
    nzval = R.nzval
    nt = Threads.nthreads()

    # Precompute Gramian: SᵀS + λI (upper triangle only)
    gramian = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), source, zero(T), gramian)
    @inbounds for d in 1:k
        gramian[d, d] += λ
    end

    Threads.@threads for chunk in 1:nt
        A = A_bufs[chunk]
        b = b_bufs[chunk]
        Z = Z_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n, nt)
        # Gather rated item vectors, scale for syrk, and accumulate b in one pass
        m = 0
        fill!(b, zero(T))
        @inbounds for idx in nzrange(R, u)
            m += 1
            i = Int(colval[idx])
            c_ui = α * T(nzval[idx])
            sq = sqrt(c_ui)
            coeff = one(T) + c_ui
            for f in 1:k
                sf = source[f, i]
                Z[f, m] = sf * sq
                b[f] += coeff * sf
            end
        end

        if m == 0
            @inbounds for f in 1:k
                target[f, u] = zero(T)
            end
            continue
        end

        # A = gramian + Z*Z' (Z already scaled by sqrt(c))
        copyto!(A, gramian)
        BLAS.syrk!('U', 'N', one(T), @view(Z[:, 1:m]), one(T), A)

        # Solve via CholeskySolver, with a singular-gramian fallback
        # (λ ≈ 0 with very few ratings can produce a singular gram).
        _, info = LAPACK.potrf!('U', A)
        if info != 0
            copyto!(A, gramian)
            @inbounds for d in 1:k
                A[d, d] += max(λ, eps(T))
            end
            LinearAlgebra.copytri!(A, 'U')
            _, info = LAPACK.potrf!('U', A)
        end
        if info == 0
            LAPACK.potrs!('U', A, b)
        else
            fill!(b, zero(T))
        end
        @inbounds for f in 1:k
            target[f, u] = b[f]
        end
        end
    end
end

function _ials_update_factors!(target::Matrix{T}, source::Matrix{T},
                               R::SparseMatrixCSC, α::T, λ::T, k::Int,
                               A_bufs::Vector{Matrix{T}},
                               b_bufs::Vector{Vector{T}},
                               Z_bufs::Vector{Matrix{T}},
                               w_bufs::Vector{Vector{T}}) where {T}
    n = size(target, 2)
    rv = rowvals(R)
    nz = nonzeros(R)
    nt = Threads.nthreads()

    # Precompute Gramian: SᵀS + λI (upper triangle only)
    gramian = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), source, zero(T), gramian)
    @inbounds for d in 1:k
        gramian[d, d] += λ
    end

    Threads.@threads for chunk in 1:nt
        A = A_bufs[chunk]
        b = b_bufs[chunk]
        Z = Z_bufs[chunk]

        for j in _thread_chunk_bounds(chunk, n, nt)

        m = 0
        fill!(b, zero(T))
        @inbounds for idx in nzrange(R, j)
            m += 1
            i = Int(rv[idx])
            c_ui = α * T(nz[idx])
            sq = sqrt(c_ui)
            coeff = one(T) + c_ui
            for f in 1:k
                sf = source[f, i]
                Z[f, m] = sf * sq
                b[f] += coeff * sf
            end
        end

        if m == 0
            @inbounds for f in 1:k
                target[f, j] = zero(T)
            end
            continue
        end

        copyto!(A, gramian)
        BLAS.syrk!('U', 'N', one(T), @view(Z[:, 1:m]), one(T), A)

        # Solve via CholeskySolver, with a singular-gramian fallback
        # (λ ≈ 0 with very few ratings can produce a singular gram).
        _, info = LAPACK.potrf!('U', A)
        if info != 0
            copyto!(A, gramian)
            @inbounds for d in 1:k
                A[d, d] += max(λ, eps(T))
            end
            LinearAlgebra.copytri!(A, 'U')
            _, info = LAPACK.potrf!('U', A)
        end
        if info == 0
            LAPACK.potrs!('U', A, b)
        else
            fill!(b, zero(T))
        end
        @inbounds for f in 1:k
            target[f, j] = b[f]
        end
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# CG solver path — uses implicit matrix-vector products (no A formed)
# Instead of forming A = gramian + Σ c_ui y_i y_iᵀ, compute A*v as:
#   A*v = gramian*v + Σ c_ui * y_i * (y_iᵀ * v)
# This is O(k + mk) per CG step instead of O(k²m) for syrk.
# Much faster for large k with few CG steps.
# ──────────────────────────────────────────────────────────────────────────────

function _ials_update_factors_cg!(target::Matrix{T}, source::Matrix{T},
                                  R::SparseMatricesCSR.SparseMatrixCSR, α::T, λ::T,
                                  k::Int, cg_steps::Int,
                                  A_bufs::Vector{Matrix{T}},
                                  b_bufs::Vector{Vector{T}},
                                  r_bufs::Vector{Vector{T}},
                                  p_bufs::Vector{Vector{T}},
                                  Ap_bufs::Vector{Vector{T}},
                                  Z_bufs::Vector{Matrix{T}},
                                  w_bufs::Vector{Vector{T}}) where {T}
    n = size(target, 2)
    colval = R.colval
    nzval = R.nzval
    nt = Threads.nthreads()

    # Gramian = SᵀS + λI (full symmetric for gemv)
    gramian = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), source, zero(T), gramian)
    LinearAlgebra.copytri!(gramian, 'U')
    @inbounds for d in 1:k
        gramian[d, d] += λ
    end

    Threads.@threads for chunk in 1:nt
        b = b_bufs[chunk]
        r = r_bufs[chunk]
        p = p_bufs[chunk]
        Ap = Ap_bufs[chunk]
        Z = Z_bufs[chunk]
        w = w_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n, nt)

        # Gather item vectors and confidence weights for this user
        m = 0
        fill!(b, zero(T))
        @inbounds for idx in nzrange(R, u)
            m += 1
            i = Int(colval[idx])
            c_ui = α * T(nzval[idx])
            w[m] = c_ui
            coeff = one(T) + c_ui
            @simd for f in 1:k
                Z[f, m] = source[f, i]
                b[f] += coeff * Z[f, m]
            end
        end

        # CG solve with warm-start
        x = @view target[:, u]

        if m == 0
            # No interactions: solve gramian * x = 0 → x = 0
            fill!(x, zero(T))
            continue
        end

        Zm = @view Z[:, 1:m]
        wm = @view w[1:m]

        # r = b - A*x where A*x = gramian*x + Z * diag(w) * Z^T * x
        BLAS.gemv!('N', one(T), gramian, x, zero(T), r)
        # Sparse correction via gathered BLAS: tmp = Z^T * x, tmp .*= w, r += Z * tmp
        tmp_m = @view w[m+1:2m]
        BLAS.gemv!('T', one(T), Zm, x, zero(T), tmp_m)
        @inbounds @simd for j in 1:m
            tmp_m[j] *= wm[j]
        end
        BLAS.gemv!('N', one(T), Zm, tmp_m, one(T), r)

        @inbounds @simd for f in 1:k
            r[f] = b[f] - r[f]
            p[f] = r[f]
        end
        rs_old = dot(r, r)

        for _ in 1:cg_steps
            # Ap = A*p = gramian*p + Z * diag(w) * Z^T * p
            BLAS.gemv!('N', one(T), gramian, p, zero(T), Ap)
            BLAS.gemv!('T', one(T), Zm, p, zero(T), tmp_m)
            @inbounds @simd for j in 1:m
                tmp_m[j] *= wm[j]
            end
            BLAS.gemv!('N', one(T), Zm, tmp_m, one(T), Ap)

            pAp = dot(p, Ap)
            pAp < eps(T) && break
            α_cg = rs_old / pAp
            @inbounds @simd for f in 1:k
                x[f] += α_cg * p[f]
                r[f] -= α_cg * Ap[f]
            end
            rs_new = dot(r, r)
            rs_new < eps(T) && break
            β = rs_new / rs_old
            @inbounds @simd for f in 1:k
                p[f] = r[f] + β * p[f]
            end
            rs_old = rs_new
        end
        end
    end
end

function _ials_update_factors_cg!(target::Matrix{T}, source::Matrix{T},
                                  R::SparseMatrixCSC, α::T, λ::T,
                                  k::Int, cg_steps::Int,
                                  A_bufs::Vector{Matrix{T}},
                                  b_bufs::Vector{Vector{T}},
                                  r_bufs::Vector{Vector{T}},
                                  p_bufs::Vector{Vector{T}},
                                  Ap_bufs::Vector{Vector{T}},
                                  Z_bufs::Vector{Matrix{T}},
                                  w_bufs::Vector{Vector{T}}) where {T}
    n = size(target, 2)
    rv = rowvals(R)
    nz = nonzeros(R)
    nt = Threads.nthreads()

    gramian = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), source, zero(T), gramian)
    LinearAlgebra.copytri!(gramian, 'U')
    @inbounds for d in 1:k
        gramian[d, d] += λ
    end

    Threads.@threads for chunk in 1:nt
        b = b_bufs[chunk]
        r = r_bufs[chunk]
        p = p_bufs[chunk]
        Ap = Ap_bufs[chunk]
        Z = Z_bufs[chunk]
        w = w_bufs[chunk]

        for j in _thread_chunk_bounds(chunk, n, nt)

        # Gather item vectors and confidence weights
        m = 0
        fill!(b, zero(T))
        @inbounds for idx in nzrange(R, j)
            m += 1
            i = Int(rv[idx])
            c_ui = α * T(nz[idx])
            w[m] = c_ui
            coeff = one(T) + c_ui
            @simd for f in 1:k
                Z[f, m] = source[f, i]
                b[f] += coeff * Z[f, m]
            end
        end

        x = @view target[:, j]

        if m == 0
            fill!(x, zero(T))
            continue
        end

        Zm = @view Z[:, 1:m]
        wm = @view w[1:m]
        tmp_m = @view w[m+1:2m]

        # r = b - A*x
        BLAS.gemv!('N', one(T), gramian, x, zero(T), r)
        BLAS.gemv!('T', one(T), Zm, x, zero(T), tmp_m)
        @inbounds @simd for jj in 1:m
            tmp_m[jj] *= wm[jj]
        end
        BLAS.gemv!('N', one(T), Zm, tmp_m, one(T), r)

        @inbounds @simd for f in 1:k
            r[f] = b[f] - r[f]
            p[f] = r[f]
        end
        rs_old = dot(r, r)

        for _ in 1:cg_steps
            # Ap = A*p = gramian*p + Z * diag(w) * Z^T * p
            BLAS.gemv!('N', one(T), gramian, p, zero(T), Ap)
            BLAS.gemv!('T', one(T), Zm, p, zero(T), tmp_m)
            @inbounds @simd for jj in 1:m
                tmp_m[jj] *= wm[jj]
            end
            BLAS.gemv!('N', one(T), Zm, tmp_m, one(T), Ap)

            pAp = dot(p, Ap)
            pAp < eps(T) && break
            α_cg = rs_old / pAp
            @inbounds @simd for f in 1:k
                x[f] += α_cg * p[f]
                r[f] -= α_cg * Ap[f]
            end
            rs_new = dot(r, r)
            rs_new < eps(T) && break
            β = rs_new / rs_old
            @inbounds @simd for f in 1:k
                p[f] = r[f] + β * p[f]
            end
            rs_old = rs_new
        end
        end
    end
end

function _ials_loss(U::Matrix{T}, V::Matrix{T},
                    X::SparseMatrixCSC, α::T, λ::T) where {T}
    rv = rowvals(X)
    nz = nonzeros(X)
    loss = zero(T)
    k = size(U, 1)

    for j in axes(X, 2)
        for idx in nzrange(X, j)
            i = rv[idx]
            r = T(nz[idx])
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, i] * V[f, j]
            end
            c = one(T) + α * r
            loss += c * (one(T) - pred)^2
        end
    end
    loss + λ * (sum(abs2, U) + sum(abs2, V))
end
