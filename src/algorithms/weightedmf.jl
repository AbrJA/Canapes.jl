# ──────────────────────────────────────────────────────────────────────────────
# WeightedMF — Weighted Regularized Matrix Factorization (Implicit ALS)
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
#   • Coordinate-descent NNLS (true bounded non-negative, not a clamp)
#   • BLAS.syrk! for YᵀY (symmetric rank-k update)
#   • Base.Threads.@threads (dynamic) — nestable and safe under concurrency
# ──────────────────────────────────────────────────────────────────────────────

"""
    WeightedMF{T} <: AbstractMatrixFactorization

Weighted Regularized Matrix Factorization via Alternating Least Squares.

Supports implicit feedback (Hu et al. 2008) and explicit feedback. With
`feedback=Explicit` the model is **BiasedMF** (Koren 2009): the prediction is
`μ + b_u + b_i + x_uᵀ y_i`, where `μ` is the observed global mean and the
user/item biases are learned jointly with the factors via augmented ALS (each
entity's bias occupies an extra slot of its ALS solve that is regularized with
the same `λ`). Score/predict then return ratings; evaluate with [`rmse`](@ref).
Three solvers available: CholeskySolver (exact), CGSolver (approximate, fast),
and NonNegativeSolver (non-negative least squares; implicit only — for
explicit feedback it degrades to the plain augmented ALS solve).

!!! note "Naming"
    Hu et al. (2008) is often called "iALS" (implicit ALS) in the literature
    and in other libraries. It is not the [`CachedALS`](@ref) type of this package,
    which implements the improved ALS of Rendle et al. (2021, "IALS++").

!!! note "Mixed input scales"
    The confidence term `c_ui = 1 + α·r_ui` makes the Gramian
    `A = YᵀCᵘY + λI` scale with the magnitude of the ratings. Mixing ratings
    of very different scales in one matrix (e.g. implicit counts 1–10
    alongside scores 1–1000) makes `A` numerically near-singular within
    float `eps`. Prefer a monotone transformation such as `r_ui' = log(1 + r_ui)`
    before fitting when this is expected. As a safety net, the Cholesky path
    detects non-positive pivots and NaN/Inf solve results and retries with
    adaptive regularisation, reverting the affected factor to its previous
    state if no finite solve is possible — factors never become NaN/Inf. The
    explicit (BiasedMF) path is scale-clean: it only involves `r_ui` residuals.

# Constructor
```julia
WeightedMF(; rank=10, λ=0.1, α=1.0, max_iter=10, tol=0.005,
       solver=CGSolver(), cg_steps=3, feedback=Implicit, verbose=true)
```

# Fields
- `rank::Int`          — latent dimension
- `λ::T`              — regularisation strength
- `α::T`              — confidence weight for implicit feedback
- `max_iter::Int`      — maximum ALS iterations
- `tol::T` — relative loss tolerance for early stopping (<0 disables)
- `solver::ALSSolver`  — `CholeskySolver()`, `CGSolver()`, or `NonNegativeSolver()`
- `cg_steps::Int`      — max CG inner iterations (only for CG solver)
- `feedback::FeedbackType` — `Implicit` or `Explicit`
- `user_factors::Matrix{T}`  — rank × n_users (set after `fit!`)
- `item_factors::Matrix{T}`  — rank × n_items (set after `fit!`)
- `global_mean::T`           — observed-ratings mean (explicit only, set after `fit!`)
- `user_bias::Vector{T}`     — fitted user biases (explicit only)
- `item_bias::Vector{T}`     — fitted item biases (explicit only)

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> model = WeightedMF(rank=8, λ=0.1, α=40.0, max_iter=2, solver=CGSolver(), verbose=false);

julia> fit!(model, X; rng=MersenneTwister(2));

julia> recommendations = recommend(model, X; k=10);

julia> size(recommendations)
(200, 10)
```
"""
mutable struct WeightedMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const α::T
    const max_iter::Int
    const tol::T
    const solver::ALSSolver
    const cg_steps::Int
    const feedback::FeedbackType
    const verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    global_mean::T
    user_bias::Vector{T}
    item_bias::Vector{T}
    is_fitted::Bool
end

function WeightedMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    α::Float64 = 1.0,
    max_iter::Int = 10,
    tol::Float64 = 0.005,
    solver::ALSSolver = CGSolver(),
    cg_steps::Int = 3,
    feedback::FeedbackType = Implicit,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    α >= 0.0 || throw(ArgumentError("α must be non-negative, got $α"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    cg_steps >= 1 || throw(ArgumentError("cg_steps must be ≥ 1, got $cg_steps"))
    WeightedMF{T}(
        rank, T(λ), T(α), max_iter, T(tol), solver, cg_steps, feedback, verbose,
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        zero(T), Vector{T}(), Vector{T}(),
        false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::WeightedMF, X::SparseMatrixCSC; rng, U_init, V_init) -> model

Fit the WeightedMF model on user-item sparse matrix `X` (n_users × n_items).

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()` — random number generator
- `U_init::Union{Nothing, Matrix}` — warm-start user factors (rank × n_users)
- `V_init::Union{Nothing, Matrix}` — warm-start item factors (rank × n_items)
"""
function fit!(model::WeightedMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              U_init::Union{Nothing, Matrix{T}} = nothing,
               V_init::Union{Nothing, Matrix{T}} = nothing,
               callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    old_user_factors = model.user_factors
    old_item_factors = model.item_factors
    old_global_mean = model.global_mean
    old_user_bias = model.user_bias
    old_item_bias = model.item_bias
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    n_users, n_items = size(X)
    _require_finite_input(X, "WeightedMF")
    k = model.rank
    is_explicit = model.feedback == Explicit
    # BiasedMF (explicit): prediction = μ + b_u + b_i + x_uᵀ y_i; the global
    # mean is the observed-ratings mean (fixed), biases are learned together
    # with the factors via augmented ALS (see _als_sweep_*).
    if is_explicit
        model.global_mean = T(sum(nonzeros(X)) / max(nnz(X), 1))
        model.user_bias = zeros(T, n_users)
        model.item_bias = zeros(T, n_items)
    end
    # Augmented solves use rank+1 dimensions (the extra slot is the bias).
    kdim = is_explicit ? k + 1 : k

    # Initialise factor matrices
    model.user_factors = isnothing(U_init) ? init_factors(rng, k, n_users) : copy(U_init)
    model.item_factors = isnothing(V_init) ? init_factors(rng, k, n_items) : copy(V_init)

    # Build transpose for fast row access
    Xt = SparseMatrixCSC(X')  # n_items × n_users

    # Per-thread ALS workspaces, allocated once and reused across all sweeps
    # and iterations (avoids re-allocating gram/rhs/Z/CG buffers every sweep).
    max_nnz = max(maximum(length(nzrange(X, u)) for u in 1:n_items; init=0),
                  maximum(length(nzrange(Xt, u)) for u in 1:n_users; init=0))
    ws = _als_workspace(model, kdim, Threads.nthreads(), max_nnz)

    monitor = ConvergenceMonitor{T}(tol=T(model.tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # Update user factors (fixing items)
        _als_sweep!(model, Xt, model.user_factors, model.item_factors, n_users, ws,
                    model.user_bias, model.item_bias)
        # Update item factors (fixing users)
        _als_sweep!(model, X, model.item_factors, model.user_factors, n_items, ws,
                    model.item_bias, model.user_bias)

        loss = _compute_loss(model, X)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("WeightedMF", iter, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[WeightedMF] converged at iteration $iter"
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
        model.global_mean = old_global_mean
        model.user_bias = old_user_bias
        model.item_bias = old_item_bias
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
    model::WeightedMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    ws,
    bias_out::Vector{T},
    bias_other::Vector{T},
) where {T}
    _als_sweep!(model.solver, model, A, factors, fixed, n_entities, ws, bias_out, bias_other)
end

_als_sweep!(::CGSolver, model, A, factors, fixed, n_entities, ws, bias_out, bias_other) =
    _als_sweep_cg!(model, A, factors, fixed, n_entities, ws, bias_out, bias_other)
_als_sweep!(::Union{CholeskySolver, NonNegativeSolver}, model, A, factors, fixed, n_entities, ws, bias_out, bias_other) =
    _als_sweep_cholesky!(model, A, factors, fixed, n_entities, ws, bias_out, bias_other)

"""
    _als_workspace(model::WeightedMF{T}, k, nt, max_nnz)

Per-thread work buffers for one ALS sweep, allocated once per `fit!` and reused
across every sweep/iteration to avoid repeated buffer allocation.
"""
function _als_workspace(model::WeightedMF{T}, k::Int, nt::Int, max_nnz::Int) where {T}
    if model.solver isa CGSolver
        (; rhs_bufs = _thread_buffers(() -> Vector{T}(undef, k), nt),
           idx_bufs = _thread_buffers(() -> Vector{Int}(undef, max_nnz), nt),
           wgt_bufs = _thread_buffers(() -> Vector{T}(undef, max_nnz), nt),
           r_bufs   = _thread_buffers(() -> Vector{T}(undef, k), nt),
           p_bufs   = _thread_buffers(() -> Vector{T}(undef, k), nt),
           Ap_bufs  = _thread_buffers(() -> Vector{T}(undef, k), nt),
           xu_bufs  = _thread_buffers(() -> Vector{T}(undef, k), nt))  # CG solution scratch (k or k+1)
    else
        Z_CAP = 4096
        (; gram_bufs = _thread_buffers(() -> Matrix{T}(undef, k, k), nt),
           rhs_bufs = _thread_buffers(() -> Vector{T}(undef, k), nt),
           z_bufs   = _thread_buffers(() -> Matrix{T}(undef, k, min(max_nnz, Z_CAP)), nt),
           prev_bufs = _thread_buffers(() -> Vector{T}(undef, k), nt),
           save_bufs = _thread_buffers(() -> Vector{T}(undef, k), nt))
    end
end

# ──────────────── Numerical-stability guard ────────────────

"""
    _gram_scale!(A, k, fallback) -> T

Largest *finite* absolute diagonal entry of the assembled Gramian `A`. Used as
the reference scale for adaptive regularisation so extremes in a single entity
cannot distort the rest of the model.
"""
@inline function _gram_scale!(A::Matrix{T}, k::Int, fallback::T) where {T}
    s = zero(T)
    @inbounds for d in 1:k
        v = abs(A[d, d])
        if v > s && v == v && v != T(Inf)
            s = v
        end
    end
    s > zero(T) ? s : fallback
end

"""
    _guarded_cholesky_solve!(A, b, base, save, scale, k) -> Bool

In-place Cholesky solve of `A x = b` guarded against scale-mixed inputs.

`A` initially holds the assembled Gramian (upper triangle, plus the shared
`base = YᵀY + λI` diagonal already folded in) and is used as scratch.
`save` is a scratch copy of `b` kept for the retry attempts.

The confidence terms `c_ui = 1 + α·r_ui` make the Gramian numerically singular
within float `eps` when ratings of wildly different scales are mixed (e.g.
implicit counts 1–10 with scores 1–1000): `potrf!` can then report a failure
or `potrs!` can return NaN/Inf. When either happens the diagonal is
regularised adaptively — lifted by increasing fractions of the Gramian's own
scale — and the solve retried.

Returns `true` with the solution in `b`, or `false` when no finite solve could
be obtained; the caller then decides between reverting to the previous factor
vector and zeroing it.
"""
function _guarded_cholesky_solve!(
    A::Matrix{T}, b::Vector{T}, base::Matrix{T}, save::Vector{T},
    scale::T, k::Int,
) where {T}
    # First attempt on the assembled Gramian (no reprojection of the scale).
    _, info = LAPACK.potrf!('U', A)
    if info == 0
        LAPACK.potrs!('U', A, b)
        all(isfinite, b) && return true
    end

    # Adaptive regularisation: lift the diagonal by fractions of the Gramian's
    # own scale, retrying up to five escalation levels.
    for eta in (T(1e-10), T(1e-8), T(1e-6), T(1e-4), T(1e-2))
        copyto!(A, base)
        lift = scale * eta
        @inbounds for d in 1:k
            A[d, d] += lift
        end
        LinearAlgebra.copytri!(A, 'U')
        _, info = LAPACK.potrf!('U', A)
        info == 0 || continue
        copyto!(b, save)
        LAPACK.potrs!('U', A, b)
        all(isfinite, b) && return true
    end
    return false
end

# ──────────────── CholeskySolver path ────────────────

# Assemble the per-entity Gram + RHS for one Cholesky ALS entity update.
# Implicit: gram += Σ (c_ui - 1) yᵢyᵢᵀ (batched syrk!), rhs = Σ c_ui yᵢ.
# Explicit (BiasedMF): gram += Σ ỹᵢỹᵢᵀ, rhs = Σ ỹᵢ (r - μ - b_other_i); the
# pinned 1 fills the bias row/col and the corner = n_obs + λ.
@inline function _cholesky_assemble!(
    gram::Matrix{T}, rhs::Vector{T}, Z::Matrix{T},
    fixed_a::Matrix{T}, A::SparseMatrixCSC{Tv,Ti},
    rv::Vector{Ti}, nz::Vector{Tv},
    u::Int, is_implicit::Bool, k::Int, kdim::Int,
    α::T, μ::T, bias_other::Vector{T}, Z_CAP::Int,
) where {T,Tv,Ti}
    if is_implicit
        m = 0
        for idx in nzrange(A, u)
            i   = rv[idx]
            rui = T(nz[idx])
            cui = max(one(T), one(T) + α * rui)
            sq  = sqrt(cui - one(T))
            @inbounds for f in 1:k
                sf = fixed_a[f, i]
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
        m = 0
        for idx in nzrange(A, u)
            i   = rv[idx]
            resid = T(nz[idx]) - μ - bias_other[i]
            @inbounds for f in 1:kdim
                sf = fixed_a[f, i]
                Z[f, m+1] = sf
                rhs[f] += resid * sf
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
    end
    nothing
end

function _als_sweep_cholesky!(
    model::WeightedMF{T},
    A::SparseMatrixCSC{Tv,Ti},
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    ws,
    bias_out::Vector{T},
    bias_other::Vector{T},
) where {T,Tv,Ti}
    k = model.rank
    λ = model.λ
    α = model.α
    is_implicit = model.feedback == Implicit
    is_explicit = !is_implicit
    is_nnls     = model.solver isa NonNegativeSolver
    kdim = is_explicit ? k + 1 : k
    μ = model.global_mean
    # Fixed side augmented with a pinned 1 in the bias slot (explicit only):
    # ỹ_i = [y_i; 1], so a single augmented Gram/RHS solve yields factors AND
    # the entity bias in one shot (BiasedMF ALS).
    fixed_a = is_explicit ? _augment_fixed(fixed, kdim) : fixed

    # YᵀY via BLAS syrk (symmetric rank-k: C = α·A·Aᵀ + β·C); for explicit the
    # augmented Gram's extra row/col is assembled per-entity from the observed
    # sets (Σ y_i and the count), so the base Gram holds only the k×k block
    # plus the λ corner for the bias slot.
    YtY = Matrix{T}(undef, kdim, kdim)
    BLAS.syrk!('U', 'N', one(T), fixed_a, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')
    base_gram = Matrix{T}(undef, kdim, kdim)
    fill!(base_gram, zero(T))
    if is_explicit
        @inbounds for a in 1:k, b in 1:k
            base_gram[a, b] = YtY[a, b]
        end
        base_gram[kdim, kdim] = λ
    else
        base_gram .= YtY
    end
    @inbounds for d in 1:kdim
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
    prev_bufs = ws.prev_bufs
    save_bufs = ws.save_bufs
    Z_CAP = size(z_bufs[1], 2)

    Base.Threads.@threads for chunk in 1:length(gram_bufs)
        gram = gram_bufs[chunk]
        rhs  = rhs_bufs[chunk]
        Z    = z_bufs[chunk]
        prev = prev_bufs[chunk]
        save = save_bufs[chunk]

        for u in _thread_chunk_bounds(chunk, n_entities, length(gram_bufs))

        # gram ← base; rhs ← 0
        copyto!(gram, base_gram)
        fill!(rhs, zero(T))

        _cholesky_assemble!(gram, rhs, Z, fixed_a, A, rv, nz, u,
                            is_implicit, k, kdim, α, μ, bias_other, Z_CAP)

        # Mirror upper triangle
        LinearAlgebra.copytri!(gram, 'U')

        # Reference scale for adaptive regularisation, plus snapshots so a
        # failed solve can be contained without poisoning the model.
        scale = _gram_scale!(gram, kdim, one(T))
        view_factors = @view factors[1:k, u]
        copyto!(prev, view_factors)
        prev_bias = is_explicit ? bias_out[u] : zero(T)
        copyto!(save, rhs)

        ok = _guarded_cholesky_solve!(gram, rhs, base_gram, save, scale, kdim)

        if is_nnls && is_implicit
            rhs .= max.(rhs, zero(T))
            _nnls_cd!(rhs, YtY, fixed_a, rv, nz, nzrange(A, u), k, α, λ, is_implicit)
            ok &= all(isfinite, rhs)
        end

        if ok
            @inbounds view_factors .= rhs[1:k]
            is_explicit && (bias_out[u] = rhs[kdim])
        elseif isempty(nzrange(A, u))
            # Cold entity with no signal: the exact ALS solution is zero.
            fill!(view_factors, zero(T))
            is_explicit && (bias_out[u] = zero(T))
        else
            # Scale-mixed inputs can make the Gramian numerically singular;
            # revert rather than let NaN/Inf propagate through the model.
            @inbounds view_factors .= prev
            is_explicit && (bias_out[u] = prev_bias)
        end
        end
    end
end

# Augment a (k × n) fixed-side factor matrix with a pinned row of ones:
# ỹ = [y; 1] in (k+1) dimensions — the BiasedMF bias slot.
function _augment_fixed(fixed::Matrix{T}, kdim::Int) where {T}
    k, n = size(fixed)
    fa = Matrix{T}(undef, kdim, n)
    @inbounds for j in 1:n, f in 1:k
        fa[f, j] = fixed[f, j]
    end
    @inbounds for j in 1:n
        fa[kdim, j] = one(T)
    end
    fa
end

"""
    _nnls_cd!(x, YtY, Y, rv, nz, col_range, k, α, λ, is_implicit; max_iter=50)

Block-coordinate descent NNLS: minimises ‖Ax - b‖² s.t. x ≥ 0.
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

# Fill the CG RHS, neighbor indices and per-item weights for one entity.
# Implicit: w = c_ui - 1, rhs += c_ui·yᵢ. Explicit: w = 1, rhs += (r-μ-b)·ỹᵢ.
@inline function _cg_assemble!(
    rhs::Vector{T}, idxs::Vector{Int}, wgts::Vector{T},
    fixed_a::Matrix{T}, A::SparseMatrixCSC{Tv,Ti},
    rv::Vector{Ti}, nz::Vector{Tv},
    u::Int, col_range::AbstractRange{<:Integer},
    is_implicit::Bool, α::T, μ::T, bias_other::Vector{T},
) where {T,Tv,Ti}
    for (pos, idx) in enumerate(col_range)
        i   = rv[idx]
        rui = T(nz[idx])
        idxs[pos] = i
        if is_implicit
            cui = max(one(T), one(T) + α * rui)
            wgts[pos] = cui - one(T)
            BLAS.axpy!(cui, @view(fixed_a[:, i]), rhs)
        else
            wgts[pos] = one(T)
            resid = rui - μ - bias_other[i]
            BLAS.axpy!(resid, @view(fixed_a[:, i]), rhs)
        end
    end
    nothing
end

function _als_sweep_cg!(
    model::WeightedMF{T},
    A::SparseMatrixCSC{Tv,Ti},
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    ws,
    bias_out::Vector{T},
    bias_other::Vector{T},
) where {T,Tv,Ti}
    k = model.rank
    λ = model.λ
    α = model.α
    cg_steps = model.cg_steps
    is_implicit = model.feedback == Implicit
    is_explicit = !is_implicit
    kdim = is_explicit ? k + 1 : k
    μ = model.global_mean
    fixed_a = is_explicit ? _augment_fixed(fixed, kdim) : fixed

    # Base gram (shared, read-only): for explicit, keep the k×k block of YᵀY
    # plus a λ corner for the bias slot — the Σ yᵢ cross terms are per-entity.
    YtY = Matrix{T}(undef, kdim, kdim)
    BLAS.syrk!('U', 'N', one(T), fixed_a, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')
    base_gram = Matrix{T}(undef, kdim, kdim)
    fill!(base_gram, zero(T))
    if is_explicit
        @inbounds for a in 1:k, b in 1:k
            base_gram[a, b] = YtY[a, b]
        end
        base_gram[kdim, kdim] = λ
    else
        base_gram .= YtY
    end
    @inbounds for d in 1:kdim
        base_gram[d, d] += λ
    end


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

        _cg_assemble!(rhs, idxs, wgts, fixed_a, A, rv, nz, u, col_range,
                      is_implicit, α, μ, bias_other)

        # Warm start: previous factor column (and bias slot) as the CG
        # initial guess — converges faster and never starts from garbage.
        xu = ws.xu_bufs[chunk]
        @inbounds for f in 1:k
            xu[f] = factors[f, u]
        end
        is_explicit && (xu[kdim] = bias_out[u])
        _cg_solve!(xu, base_gram, fixed_a,
                   view(idxs, 1:n_nz), view(wgts, 1:n_nz),
                   rhs, kdim, cg_steps,
                   r_bufs[chunk], p_bufs[chunk], Ap_bufs[chunk])
        @inbounds for f in 1:k
            factors[f, u] = xu[f]
        end
        is_explicit && (bias_out[u] = xu[kdim])
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

function _compute_loss(model::WeightedMF{T}, X::SparseMatrixCSC) where {T}
    U = model.user_factors
    V = model.item_factors
    λ = model.λ
    α = model.α
    k = model.rank
    μ = model.global_mean
    b_u = model.user_bias
    b_i = model.item_bias

    loss = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)

    if model.feedback == Implicit
        for j in axes(X, 2)
            for idx in nzrange(X, j)
                i = rv[idx]
                r = T(nz[idx])
                pred = zero(T)
                @inbounds @simd for f in 1:k
                    pred += U[f, i] * V[f, j]
                end
                c = max(one(T), one(T) + α * r)
                loss += c * (one(T) - pred)^2
            end
        end
        loss += λ * (sum(abs2, U) + sum(abs2, V))
    else
        for j in axes(X, 2)
            for idx in nzrange(X, j)
                i = rv[idx]
                r = T(nz[idx])
                pred = μ + b_u[i] + b_i[j]
                @inbounds @simd for f in 1:k
                    pred += U[f, i] * V[f, j]
                end
                loss += (r - pred)^2
            end
        end
        loss += λ * (sum(abs2, U) + sum(abs2, V) + sum(abs2, b_u) + sum(abs2, b_i))
    end
    loss
end

# ──────────────────────────────────────────────────────────────────────────────
# transform / predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    transform(model::WeightedMF, X::SparseMatrixCSC) -> Matrix

Compute user embeddings for new users given their interaction matrix `X` (n_new × n_items).
Returns a `rank × n_new` factor matrix.
"""
function transform(model::WeightedMF{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    emb, _ = _foldin_embeddings(model, X)
    emb
end

# Fold-in embeddings for the rows of X, returning (factors, folded biases).
# For explicit feedback the augmented ALS solve also produces each folded
# user's bias (bias dimension of the augmented solution vector); the residuals
# use the fixed-side biases (item_bias) and the global mean.
function _foldin_embeddings(model::WeightedMF{T}, X::SparseMatrixCSC) where {T}
    n_users_new = size(X, 1)
    k = model.rank
    is_explicit = model.feedback == Explicit
    kdim = is_explicit ? k + 1 : k
    new_user_factors = Matrix{T}(undef, k, n_users_new)
    fill!(new_user_factors, zero(T))
    folded_bias = is_explicit ? zeros(T, n_users_new) : T[]

    Xt = SparseMatrixCSC(X')
    max_nnz = maximum(length(nzrange(Xt, u)) for u in 1:n_users_new; init=0)
    ws = _als_workspace(model, kdim, Threads.nthreads(), max_nnz)
    _als_sweep!(model, Xt, new_user_factors, model.item_factors, n_users_new, ws,
                folded_bias, model.item_bias)
    (new_user_factors, folded_bias)
end

"""
    recommend(model::WeightedMF, X::SparseMatrixCSC; k=10) -> Matrix{Int}

Return top-k item indices for each user. Returns `n_users × k` matrix.
Processes users in batches to avoid allocating the full score matrix.
"""
function recommend(model::WeightedMF{T}, X::SparseMatrixCSC; k::Int = 10) where {T}
    _require_fitted(model.is_fitted)

    if model.feedback == Explicit
        # BiasedMF: scores include μ + b_u + b_i; the dense score path is the
        # explicit contract (ratings), so top-k is computed from it directly.
        scores = score(model, X)
        _predict_dense_topk(scores, X, k)
    else
        # Fold in users from X so recommendations always match score(model, X),
        # including when X contains updated interactions for existing users.
        user_emb = transform(model, X)
        _predict_topk_batched(user_emb, model.item_factors, to_csr(X), k)
    end
end

"""
    score(model::WeightedMF, X) -> Matrix

Return the full score matrix (n_users × n_items) without top-k filtering.
Uses `transform` to embed users, then computes inner products with item factors.

For `feedback=Explicit` the scores are the predicted ratings
`x_uᵀ y_i + μ + b_u + b_i`, with the user bias taken from the fold-in of each
row of `X` (→ matches `recommend`). Use `predict(model, X)` for predictions
based on the fitted users' biases (the canonical RMSE evaluation path).
"""
function score(model::WeightedMF{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    if model.feedback == Explicit
        emb, b_fold = _foldin_embeddings(model, X)
        S = emb' * model.item_factors
        S .+= model.global_mean
        @inbounds for j in 1:size(S, 2)
            bj = model.item_bias[j]
            for i in 1:size(S, 1)
                S[i, j] += bj + b_fold[i]
            end
        end
        S
    else
        user_emb = transform(model, X)
        user_emb' * model.item_factors
    end
end

"""
    score(model::WeightedMF, user_indices, item_indices) -> Vector

Return raw scores for specific (user, item) pairs using pre-fitted factors.
For `feedback=Explicit` the score is the predicted rating
`x_uᵀ y_i + μ + b_u + b_i` with the fitted biases.
"""
function score(model::WeightedMF{T}, user_indices::AbstractVector{<:Integer},
              item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    if model.feedback == Explicit
        length(user_indices) == length(item_indices) ||
            throw(DimensionMismatch("user_indices and item_indices must have the same length"))
        vals = _predict_pairwise_scores(model.user_factors, model.item_factors,
                                        user_indices, item_indices)
        for t in eachindex(vals)
            vals[t] += model.global_mean + model.user_bias[user_indices[t]] +
                       model.item_bias[item_indices[t]]
        end
        vals
    else
        _predict_pairwise_scores(model.user_factors, model.item_factors, user_indices, item_indices)
    end
end

"""
    predict(model::WeightedMF, X) -> Matrix

Predicted ratings for explicit feedback (`feedback=Explicit`): the dense
`n_users × n_items` matrix `x_uᵀ y_i + μ + b_u + b_i` built from the *fitted*
user factors and biases (rows of `X` must be the training users). Evaluate with
`rmse(predict(model, X_train), X_test)`. For implicit feedback this is an alias
for [`score`](@ref).
"""
function predict(model::WeightedMF{T}, X::SparseMatrixCSC) where {T}
    if model.feedback == Explicit
        _require_fitted(model.is_fitted)
        n_users_f, n_items_f = size(model.item_factors)
        size(X, 1) == size(model.user_factors, 2) || throw(DimensionMismatch(
            "X has $(size(X, 1)) users but the fitted model has $(size(model.user_factors, 2))"))
        size(X, 2) == n_items_f || throw(DimensionMismatch(
            "X has $(size(X, 2)) items but the fitted model has $n_items_f"))
        S = model.user_factors' * model.item_factors
        S .+= model.global_mean
        @inbounds for j in 1:n_items_f
            bj = model.item_bias[j]
            for i in 1:size(S, 1)
                S[i, j] += bj + model.user_bias[i]
            end
        end
        S
    else
        score(model, X)
    end
end

# Dense top-k for the explicit scoring path: mask each user's seen items in S
# (S is a fresh dense matrix) and extract the k highest-scoring items per row.
function _predict_dense_topk(S::Matrix{T}, X::SparseMatrixCSC, k::Int) where {T}
    n_users, n_items = size(S)
    k_out = min(k, n_items)
    k_out >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    preds = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)
    nt = Threads.nthreads()
    topk_bufs = [Vector{Int}(undef, k_out) for _ in 1:nt]
    Threads.@threads for chunk in 1:nt
        topk = topk_bufs[chunk]
        for u in _thread_chunk_bounds(chunk, n_users, nt)
            for idx in nzrange(X_csr, u)
                S[u, Int(X_csr.colval[idx])] = T(-Inf)
            end
            _topk_indices!(topk, @view(S[u, :]), k_out)
            @inbounds for j in 1:k_out
                preds[u, j] = topk[j]
            end
        end
    end
    preds
end
