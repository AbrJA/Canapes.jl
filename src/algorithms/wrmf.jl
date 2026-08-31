# ──────────────────────────────────────────────────────────────────────────────
# WMF — Weighted Regularized Matrix Factorization (Implicit ALS)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Hu, Koren, Volinsky (2008)
#   "Collaborative Filtering for Implicit Feedback Datasets"
#
# Loss:
#   L = Σ_{u,i} c_{ui}(p_{ui} - xᵤᵀ yᵢ)² + λ(Σ_u ||xᵤ||² + Σ_i ||yᵢ||²)
#
# where c_{ui} = 1 + α·r_{ui}  and  p_{ui} = r_{ui} > 0
#
# Optimisations:
#   • Per-thread pre-allocated gram/rhs/Chol buffers → zero inner-loop allocs
#   • BLAS.syr! for O(k²) rank-1 gram accumulation (vectorised BLAS-2)
#   • BLAS.axpy! for O(k) rhs accumulation
#   • In-place LAPACK.potrf! + LAPACK.potrs! → no extra matrices in Chol path
#   • Coordinate-descent NonNegative (true bounded NonNegative, not a clamp)
#   • BLAS.syrk! for YᵀY (symmetric rank-k update)
#   • Base.Threads.@threads (dynamic) — nestable and safe under concurrency
# ──────────────────────────────────────────────────────────────────────────────

"""
    WMF{T} <: AbstractMatrixFactorization

Weighted Regularized Matrix Factorization via Alternating Least Squares.

Supports implicit feedback (Hu et al. 2008) and explicit feedback (MSE).
Three solvers available: CholeskySolver (exact), Conjugate Gradient (approximate, fast),
and NonNegative (non-negative matrix factorization).

# Constructor
```julia
WMF(; rank=10, λ=0.1, α=1.0, max_iter=10, convergence_tol=0.005,
       solver=ConjugateGradient(), cg_steps=3, feedback=IMPLICIT, verbose=true)
```

# Fields
- `rank::Int`          — latent dimension
- `λ::T`              — regularisation strength
- `α::T`              — confidence weight for implicit feedback
- `max_iter::Int`      — maximum ALS iterations
- `convergence_tol::T` — relative loss tolerance for early stopping (<0 disables)
- `solver::ALSSolver`  — `CholeskySolver()`, `ConjugateGradient()`, or `NonNegative()`
- `cg_steps::Int`      — max CG inner iterations (only for CG solver)
- `feedback::FeedbackType` — `IMPLICIT` or `EXPLICIT`
- `user_factors::Matrix{T}`  — rank × n_users (set after `fit!`)
- `item_factors::Matrix{T}`  — rank × n_items (set after `fit!`)

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> model = WMF(rank=8, λ=0.1, α=40.0, max_iter=2, solver=ConjugateGradient(), verbose=false);

julia> fit!(model, X; rng=MersenneTwister(2));

julia> recommendations = recommend(model, X; k=10);

julia> size(recommendations)
(200, 10)
```
"""
mutable struct WMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const α::T
    const max_iter::Int
    const convergence_tol::T
    const solver::ALSSolver
    const cg_steps::Int
    const feedback::FeedbackType
    const verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function WMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    α::Float64 = 1.0,
    max_iter::Int = 10,
    convergence_tol::Float64 = 0.005,
    solver::ALSSolver = ConjugateGradient(),
    cg_steps::Int = 3,
    feedback::FeedbackType = IMPLICIT,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    α >= 0.0 || throw(ArgumentError("α must be non-negative, got $α"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    cg_steps >= 1 || throw(ArgumentError("cg_steps must be ≥ 1, got $cg_steps"))
    T = dtype
    WMF{T}(
        rank, T(λ), T(α), max_iter, T(convergence_tol), solver, cg_steps, feedback, verbose,
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::WMF, X::SparseMatrixCSC; rng, U_init, V_init) -> model

Fit the WMF model on user-item sparse matrix `X` (n_users × n_items).

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()` — random number generator
- `U_init::Union{Nothing, Matrix}` — warm-start user factors (rank × n_users)
- `V_init::Union{Nothing, Matrix}` — warm-start item factors (rank × n_items)
"""
function fit!(model::WMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              U_init::Union{Nothing, Matrix{T}} = nothing,
               V_init::Union{Nothing, Matrix{T}} = nothing,
               callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    old_user_factors = model.user_factors
    old_item_factors = model.item_factors
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    n_users, n_items = size(X)
    _require_finite_input(X, "WMF")
    k = model.rank

    # Initialise factor matrices
    model.user_factors = isnothing(U_init) ? init_factors(rng, k, n_users) : copy(U_init)
    model.item_factors = isnothing(V_init) ? init_factors(rng, k, n_items) : copy(V_init)

    # Build transpose for fast row access
    Xt = SparseMatrixCSC(X')  # n_items × n_users

    # Per-thread ALS workspaces, allocated once and reused across all sweeps
    # and iterations (avoids re-allocating gram/rhs/Z/CG buffers every sweep).
    max_nnz = max(maximum(length(nzrange(X, u)) for u in 1:n_items; init=0),
                  maximum(length(nzrange(Xt, u)) for u in 1:n_users; init=0))
    ws = _als_workspace(model, k, Threads.nthreads(), max_nnz)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # Update user factors (fixing items)
        _als_sweep!(model, Xt, model.user_factors, model.item_factors, n_users, ws)
        # Update item factors (fixing users)
        _als_sweep!(model, X, model.item_factors, model.user_factors, n_items, ws)

        loss = _compute_loss(model, X)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("WMF", iter, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[WMF] converged at iteration $iter"
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

# ──────────────────────────────────────────────────────────────────────────────
# ALS sweep — update one side of factors, multithreaded
# ──────────────────────────────────────────────────────────────────────────────

function _als_sweep!(
    model::WMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    ws,
) where {T}
    _als_sweep!(model.solver, model, A, factors, fixed, n_entities, ws)
end

_als_sweep!(::ConjugateGradient, model, A, factors, fixed, n_entities, ws) =
    _als_sweep_cg!(model, A, factors, fixed, n_entities, ws)
_als_sweep!(::Union{CholeskySolver, NonNegative}, model, A, factors, fixed, n_entities, ws) =
    _als_sweep_cholesky!(model, A, factors, fixed, n_entities, ws)

"""
    _als_workspace(model::WMF{T}, k, nt, max_nnz)

Per-thread work buffers for one ALS sweep, allocated once per `fit!` and reused
across every sweep/iteration to avoid repeated buffer allocation.
"""
function _als_workspace(model::WMF{T}, k::Int, nt::Int, max_nnz::Int) where {T}
    if model.solver isa ConjugateGradient
        (; rhs_bufs = _thread_buffers(() -> Vector{T}(undef, k), nt),
           idx_bufs = _thread_buffers(() -> Vector{Int}(undef, max_nnz), nt),
           wgt_bufs = _thread_buffers(() -> Vector{T}(undef, max_nnz), nt),
           r_bufs   = _thread_buffers(() -> Vector{T}(undef, k), nt),
           p_bufs   = _thread_buffers(() -> Vector{T}(undef, k), nt),
           Ap_bufs  = _thread_buffers(() -> Vector{T}(undef, k), nt))
    else
        Z_CAP = 4096
        (; gram_bufs = _thread_buffers(() -> Matrix{T}(undef, k, k), nt),
           rhs_bufs = _thread_buffers(() -> Vector{T}(undef, k), nt),
           z_bufs   = _thread_buffers(() -> Matrix{T}(undef, k, min(max_nnz, Z_CAP)), nt))
    end
end

# ──────────────── CholeskySolver path ────────────────

function _als_sweep_cholesky!(
    model::WMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    ws,
) where {T}
    k = model.rank
    λ = model.λ
    α = model.α
    is_implicit = model.feedback == IMPLICIT
    is_nnls     = model.solver isa NonNegative

    # YᵀY via BLAS syrk (symmetric rank-k: C = α·A·Aᵀ + β·C)
    YtY = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), fixed, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')
    base_gram = copy(YtY)
    @inbounds for d in 1:k
        base_gram[d, d] += λ
    end

    rv = rowvals(A)
    nz = nonzeros(A)

    # Per-thread buffers from the fit-level workspace (hoisted).
    # Batched gram assembly (Rendle 2021): gather scaled item vectors into Z and
    # accumulate Z·Zᵀ with a single syrk! call per sub-batch instead of one syr!
    # per item. Z is capped so per-thread memory stays at k × Z_CAP regardless
    # of how dense a single entity is.
    gram_bufs = ws.gram_bufs
    rhs_bufs  = ws.rhs_bufs
    z_bufs    = ws.z_bufs
    Z_CAP = size(z_bufs[1], 2)

    Base.Threads.@threads for chunk in 1:length(gram_bufs)
        gram = gram_bufs[chunk]
        rhs  = rhs_bufs[chunk]
        Z    = z_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n_entities, length(gram_bufs))

        # gram ← YᵀY + λI
        copyto!(gram, base_gram)
        fill!(rhs, zero(T))

        if is_implicit
            # Batched gather + syrk!: gram += Σ (cui - 1) yᵢyᵢᵀ, rhs = Σ cui yᵢ
            m = 0
            for idx in nzrange(A, u)
                i   = rv[idx]
                rui = T(nz[idx])
                cui = max(one(T), one(T) + α * rui)
                sq  = sqrt(cui - one(T))
                @inbounds for f in 1:k
                    sf = fixed[f, i]
                    Z[f, m+1] = sf * sq
                    rhs[f] += cui * sf
                end
                m += 1
                if m == Z_CAP
                    BLAS.syrk!('U', 'N', one(T), Z, one(T), gram)
                    m = 0
                end
            end
            if m > 0
                BLAS.syrk!('U', 'N', one(T), @view(Z[:, 1:m]), one(T), gram)
            end
        else
            @inbounds for idx in nzrange(A, u)
                i   = rv[idx]
                rui = T(nz[idx])
                for f in 1:k
                    rhs[f] += rui * fixed[f, i]
                end
            end
        end

        # Mirror upper triangle
        LinearAlgebra.copytri!(gram, 'U')

        # In-place Cholesky solve, with a singular-gramian fallback
        # (λ ≈ 0 with very few ratings can produce a singular gram).
        _, info = LAPACK.potrf!('U', gram)
        if info != 0
            copyto!(gram, base_gram)
            floor_val = max(λ, eps(T))
            @inbounds for d in 1:k
                gram[d, d] += floor_val
            end
            LinearAlgebra.copytri!(gram, 'U')
            _, info = LAPACK.potrf!('U', gram)
        end
        if info == 0
            LAPACK.potrs!('U', gram, rhs)
        else
            fill!(rhs, zero(T))
        end

        if is_nnls
            rhs .= max.(rhs, zero(T))
            _nnls_cd!(rhs, YtY, fixed, rv, nz, nzrange(A, u), k, α, λ, is_implicit)
        end

        @inbounds factors[:, u] .= rhs
        end
    end
end

"""
    _nnls_cd!(x, YtY, Y, rv, nz, col_range, k, α, λ, is_implicit; max_iter=50)

Block-coordinate descent NonNegative: minimises ‖Ax - b‖² s.t. x ≥ 0.
Updates `x` in-place.
"""
function _nnls_cd!(
    x::Vector{T}, YtY::Matrix{T}, Y::Matrix{T},
    rv, nz, col_range, k::Int,
    α::T, λ::T, is_implicit::Bool; max_iter::Int = 50,
) where {T}
    gram = copy(YtY)
    @inbounds for d in 1:k; gram[d,d] += λ; end
    rhs = zeros(T, k)
    for idx in col_range
        i   = rv[idx]
        rui = T(nz[idx])
        yi  = @view Y[:, i]
        if is_implicit
            cui = max(one(T), one(T) + α * rui)
            BLAS.syr!('U', cui - one(T), yi, gram)
            BLAS.axpy!(cui, yi, rhs)
        else
            BLAS.axpy!(rui, yi, rhs)
        end
    end
    LinearAlgebra.copytri!(gram, 'U')

    # Coordinate descent
    for _ in 1:max_iter
        max_change = zero(T)
        for d in 1:k
            @inbounds numer = rhs[d] - BLAS.dot(k, @view(gram[:,d]), 1, x, 1) + gram[d,d]*x[d]
            new_val = max(zero(T), numer / gram[d, d])
            @inbounds max_change = max(max_change, abs(new_val - x[d]))
            @inbounds x[d] = new_val
        end
        max_change < T(1e-8) && break
    end
end

# ──────────────── Conjugate Gradient path ────────────────

function _als_sweep_cg!(
    model::WMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    ws,
) where {T}
    k = model.rank
    λ = model.λ
    α = model.α
    cg_steps = model.cg_steps
    is_implicit = model.feedback == IMPLICIT

    # Base gram (shared, read-only)
    YtY = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), fixed, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')
    base_gram = copy(YtY)
    @inbounds for d in 1:k; base_gram[d,d] += λ; end

    rv = rowvals(A)
    nz = nonzeros(A)

    # Per-thread CG workspace from the fit-level workspace (hoisted).
    rhs_bufs  = ws.rhs_bufs
    idx_bufs  = ws.idx_bufs
    wgt_bufs  = ws.wgt_bufs
    r_bufs    = ws.r_bufs
    p_bufs    = ws.p_bufs
    Ap_bufs   = ws.Ap_bufs
    nt = length(rhs_bufs)

    Base.Threads.@threads for chunk in 1:nt
        rhs  = rhs_bufs[chunk]
        idxs = idx_bufs[chunk]
        wgts = wgt_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n_entities, nt)

        fill!(rhs, zero(T))
        col_range = nzrange(A, u)
        n_nz = length(col_range)

        for (pos, idx) in enumerate(col_range)
            i   = rv[idx]
            rui = T(nz[idx])
            idxs[pos] = i
            if is_implicit
                cui = max(one(T), one(T) + α * rui)
                wgts[pos] = cui - one(T)
                BLAS.axpy!(cui, @view(fixed[:, i]), rhs)
            else
                wgts[pos] = zero(T)
                BLAS.axpy!(rui, @view(fixed[:, i]), rhs)
            end
        end

        xu = @view factors[:, u]
        _cg_solve!(xu, base_gram, fixed,
                   view(idxs, 1:n_nz), view(wgts, 1:n_nz),
                   rhs, k, cg_steps,
                   r_bufs[chunk], p_bufs[chunk], Ap_bufs[chunk])
        end
    end
end

"""
    _cg_solve!(x, base_gram, Y, indices, weights, b, k, max_steps, r, p, Ap)

Solve `(base_gram + Σ_j w_j y_j y_jᵀ) x = b` via Conjugate Gradient.
All vectors are pre-allocated per-thread — zero heap allocation.
"""
function _cg_solve!(
    x::AbstractVector{T},
    base_gram::Matrix{T},
    Y::Matrix{T},
    indices::AbstractVector{Int},
    weights::AbstractVector{T},
    b::Vector{T},
    k::Int,
    max_steps::Int,
    r::Vector{T}, p::Vector{T}, Ap::Vector{T},
) where {T}
    _implicit_matvec!(Ap, base_gram, Y, indices, weights, x, k)
    @inbounds for a in 1:k
        r[a] = b[a] - Ap[a]
        p[a] = r[a]
    end
    rs_old = dot(r, r)

    for _ in 1:max_steps
        _implicit_matvec!(Ap, base_gram, Y, indices, weights, p, k)
        pAp = dot(p, Ap)
        pAp < eps(T) && break
        α_cg = rs_old / pAp
        BLAS.axpy!(α_cg, p, x)
        BLAS.axpy!(-α_cg, Ap, r)
        rs_new = dot(r, r)
        rs_new < eps(T) && break
        β = rs_new / rs_old
        @inbounds @simd for a in 1:k
            p[a] = r[a] + β * p[a]
        end
        rs_old = rs_new
    end
end

function _implicit_matvec!(
    result::Vector{T},
    base_gram::Matrix{T},
    Y::Matrix{T},
    indices::AbstractVector{Int},
    weights::AbstractVector{T},
    v::AbstractVector{T},
    k::Int,
) where {T}
    BLAS.gemv!('N', one(T), base_gram, v, zero(T), result)
    n_nz = length(indices)
    n_nz == 0 && return

    # For very sparse entities (< 32 nnz), avoid BLAS overhead with manual dot+axpy
    if n_nz < 32
        @inbounds for pos in 1:n_nz
            i = indices[pos]
            w = weights[pos]
            iszero(w) && continue
            d = zero(T)
            @inbounds @simd for f in 1:k
                d += Y[f, i] * v[f]
            end
            wd = w * d
            @simd for f in 1:k
                result[f] += wd * Y[f, i]
            end
        end
    else
        @inbounds for pos in eachindex(indices)
            i = indices[pos]
            w = weights[pos]
            iszero(w) && continue
            d = BLAS.dot(k, @view(Y[:, i]), 1, v, 1)
            BLAS.axpy!(w * d, @view(Y[:, i]), result)
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────────────

function _compute_loss(model::WMF{T}, X::SparseMatrixCSC) where {T}
    U = model.user_factors
    V = model.item_factors
    λ = model.λ
    α = model.α
    k = model.rank

    loss = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)

    for j in axes(X, 2)
        for idx in nzrange(X, j)
            i = rv[idx]
            r = T(nz[idx])
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, i] * V[f, j]
            end
            if model.feedback == IMPLICIT
                c = max(one(T), one(T) + α * r)
                loss += c * (one(T) - pred)^2
            else
                loss += (r - pred)^2
            end
        end
    end

    loss += λ * (sum(abs2, U) + sum(abs2, V))
    loss
end

# ──────────────────────────────────────────────────────────────────────────────
# transform / predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    transform(model::WMF, X::SparseMatrixCSC) -> Matrix

Compute user embeddings for new users given their interaction matrix `X` (n_new × n_items).
Returns a `rank × n_new` factor matrix.
"""
function transform(model::WMF{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    n_users_new = size(X, 1)
    k = model.rank
    new_user_factors = Matrix{T}(undef, k, n_users_new)
    fill!(new_user_factors, zero(T))

    Xt = SparseMatrixCSC(X')
    max_nnz = maximum(length(nzrange(Xt, u)) for u in 1:n_users_new; init=0)
    ws = _als_workspace(model, k, Threads.nthreads(), max_nnz)
    _als_sweep!(model, Xt, new_user_factors, model.item_factors, n_users_new, ws)
    new_user_factors
end

"""
    recommend(model::WMF, X::SparseMatrixCSC; k=10) -> Matrix{Int}

Return top-k item indices for each user. Returns `n_users × k` matrix.
Processes users in batches to avoid allocating the full score matrix.
"""
function recommend(model::WMF{T}, X::SparseMatrixCSC; k::Int = 10) where {T}
    _require_fitted(model.is_fitted)

    # Fold in users from X so recommendations always match score(model, X),
    # including when X contains updated interactions for existing users.
    user_emb = transform(model, X)

    _predict_topk_batched(user_emb, model.item_factors, to_csr(X), k)
end

"""
    score(model::WMF, X) -> Matrix

Return the full score matrix (n_users × n_items) without top-k filtering.
Uses `transform` to embed users, then computes inner products with item factors.
"""
function score(model::WMF{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    user_emb = transform(model, X)
    user_emb' * model.item_factors
end

"""
    score(model::WMF, user_indices, item_indices) -> Vector

Return raw scores for specific (user, item) pairs using pre-fitted factors.
"""
function score(model::WMF{T}, user_indices::AbstractVector{<:Integer},
              item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    _predict_pairwise_scores(model.user_factors, model.item_factors, user_indices, item_indices)
end
