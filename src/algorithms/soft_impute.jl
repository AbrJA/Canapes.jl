# ──────────────────────────────────────────────────────────────────────────────
# SoftImpute / SoftSVD — matrix completion via alternating least squares
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Hastie, Mazumder, Lee, Zadeh (2014)
#   "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares"
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractSoftALS{T} <: AbstractMatrixFactorization

Abstract parent for ALS-based nuclear-norm matrix completion.
Concrete subtypes: [`SoftImpute`](@ref), [`SoftSVD`](@ref).
"""
abstract type AbstractSoftALS{T<:AbstractFloat} <: AbstractMatrixFactorization end

# ──────────────────────────────────────────────────────────────────────────────
# SoftImpute
# ──────────────────────────────────────────────────────────────────────────────

"""
    SoftImpute{T} <: AbstractSoftALS{T}

Low-rank matrix completion via alternating least squares with nuclear-norm
regularization and observed-entry correction. Learns a factorization
`U * Diagonal(d) * V'` from partially observed entries.

At each iteration, the residual at observed positions is computed and folded
into the ALS step (full imputation correction). This is the stronger mode
that handles missing data properly.

# Constructor
```julia
SoftImpute(; rank=10, λ=0.0, max_iter=100, convergence_tol=1e-3,
             final_svd=true, verbose=true)
```

# Fields
- `rank::Int` — target rank for the low-rank approximation
- `λ::T` — nuclear-norm penalty (soft-threshold on singular values)
- `max_iter::Int` — maximum iterations
- `convergence_tol::T` — relative Frobenius norm change for early stopping
- `final_svd::Bool` — re-factorize result via full SVD at end
- `U::Matrix{T}` — left singular vectors (m × rank) after fitting
- `d::Vector{T}` — singular values after fitting
- `V::Matrix{T}` — right singular vectors (n × rank) after fitting

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.05)
model = SoftImpute(rank=20, λ=1.0, max_iter=50)
fit!(model, X)
reconstruction = model.U * Diagonal(model.d) * model.V'
```
"""
mutable struct SoftImpute{T<:AbstractFloat} <: AbstractSoftALS{T}
    const rank::Int
    const λ::T
    const max_iter::Int
    const convergence_tol::T
    const final_svd::Bool
    const verbose::Bool
    U::Matrix{T}
    d::Vector{T}
    V::Matrix{T}
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function SoftImpute(;
    rank::Int = 10,
    λ::Float64 = 0.0,
    max_iter::Int = 100,
    convergence_tol::Float64 = 1e-3,
    final_svd::Bool = true,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    T = dtype
    SoftImpute{T}(rank, T(λ), max_iter, T(convergence_tol), final_svd, verbose,
                  Matrix{T}(undef,0,0), T[], Matrix{T}(undef,0,0),
                  Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# SoftSVD
# ──────────────────────────────────────────────────────────────────────────────

"""
    SoftSVD{T} <: AbstractSoftALS{T}

Low-rank SVD via alternating least squares with Ridge damping. Power-iteration
style — no imputation correction at observed entries. Faster per iteration than
[`SoftImpute`](@ref) but assumes the observed entries are representative.

# Constructor
```julia
SoftSVD(; rank=10, λ=0.0, max_iter=100, convergence_tol=1e-3,
          final_svd=true, verbose=true)
```

# Fields
Same as [`SoftImpute`](@ref).

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.05)
model = SoftSVD(rank=20, λ=1.0, max_iter=50)
fit!(model, X)
reconstruction = model.U * Diagonal(model.d) * model.V'
```
"""
mutable struct SoftSVD{T<:AbstractFloat} <: AbstractSoftALS{T}
    const rank::Int
    const λ::T
    const max_iter::Int
    const convergence_tol::T
    const final_svd::Bool
    const verbose::Bool
    U::Matrix{T}
    d::Vector{T}
    V::Matrix{T}
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function SoftSVD(;
    rank::Int = 10,
    λ::Float64 = 0.0,
    max_iter::Int = 100,
    convergence_tol::Float64 = 1e-3,
    final_svd::Bool = true,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    T = dtype
    SoftSVD{T}(rank, T(λ), max_iter, T(convergence_tol), final_svd, verbose,
               Matrix{T}(undef,0,0), T[], Matrix{T}(undef,0,0),
               Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# PureSVD — convenience alias for truncated SVD (SoftSVD with λ=0, no final_svd)
# ──────────────────────────────────────────────────────────────────────────────

"""
    PureSVD(; rank=10, max_iter=100, convergence_tol=1e-3, verbose=true)

Truncated SVD via power iteration. Equivalent to `SoftSVD(λ=0, final_svd=false)`.
Computes the top-`rank` singular triplets of a sparse matrix.

# Example
```julia
using SparseArrays, Gideon
X = sprand(1000, 500, 0.05)
model = PureSVD(rank=20)
fit!(model, X)
# model.U * Diagonal(model.d) * model.V' ≈ best rank-20 approximation
```
"""
function PureSVD(; rank::Int=10, max_iter::Int=100, convergence_tol::Float64=1e-3,
                  verbose::Bool=true, dtype::Type{<:AbstractFloat}=Float32)
    SoftSVD(rank=rank, λ=0.0, max_iter=max_iter, convergence_tol=convergence_tol,
            final_svd=false, verbose=verbose, dtype=dtype)
end

# ──────────────────────────────────────────────────────────────────────────────
# ALS step dispatch — the only algorithmic difference between the two types
# ──────────────────────────────────────────────────────────────────────────────

function _als_half_step(::SoftImpute{T}, X::SparseMatrixCSC{Tv,Ti},
                        other_vecs::Matrix{T}, d::Vector{T},
                        self_vecs::Matrix{T}, λ::T) where {T,Tv,Ti}
    _softimpute_step(X, other_vecs, d, self_vecs, λ)
end

function _als_half_step(::SoftSVD{T}, X::SparseMatrixCSC{Tv,Ti},
                        other_vecs::Matrix{T}, d::Vector{T},
                        self_vecs::Matrix{T}, λ::T) where {T,Tv,Ti}
    B_hat = X * other_vecs
    B_hat .*= (d ./ (d .+ λ))'
    B_hat
end

# ──────────────────────────────────────────────────────────────────────────────
# fit! — shared implementation for both types
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::AbstractSoftALS, X; rng, callbacks) -> model

Fit a SoftImpute or SoftSVD model on sparse matrix `X`.

Implements the SoftImpute-ALS algorithm from Hastie, Mazumder, Lee, Zadeh (2014)
"Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares".
"""
function fit!(model::AbstractSoftALS{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    m, n = size(X)
    _require_nonempty_dimensions(X, "SoftImpute")
    k = min(model.rank, m, n)
    λ = T(model.λ)
    algo_name = model isa SoftImpute ? "SoftImpute" : "SoftSVD"
    run_callbacks_train_begin(callbacks, model)
    try

    # Initialize with random orthonormal bases (rsparse style)
    U_cur = Matrix{T}(qr(randn(rng, T, m, k)).Q)[:, 1:k]::Matrix{T}
    d_cur = ones(T, k)::Vector{T}
    V_cur = zeros(T, n, k)::Matrix{T}

    Xt::SparseMatrixCSC{Tv,Ti} = SparseMatrixCSC(X')  # n × m for the item-side step

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # ── Step 1: Update V (items) ──
        B_hat = Matrix{T}(_als_half_step(model, Xt, U_cur, d_cur, V_cur, λ))
        Bsvd = svd(B_hat)
        kk = min(k, length(Bsvd.S))
        V_cur = Matrix{T}(Bsvd.U[:, 1:kk])
        d_cur = Vector{T}(Bsvd.S[1:kk])
        U_cur = Matrix{T}(U_cur * Bsvd.V[:, 1:kk])  # rotate U to stay consistent

        # ── Step 2: Update U (users) ──
        A_hat = Matrix{T}(_als_half_step(model, X, V_cur, d_cur, U_cur, λ))
        Asvd = svd(A_hat)
        kk2 = min(kk, length(Asvd.S))
        U_cur = Matrix{T}(Asvd.U[:, 1:kk2])
        d_cur = Vector{T}(Asvd.S[1:kk2])
        V_cur = Matrix{T}(V_cur * Asvd.V[:, 1:kk2])  # rotate V to stay consistent

        cur_frob = sum(abs2, d_cur)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration(algo_name, iter, model.max_iter, cur_frob,
                         iter_seconds, total_seconds)
        end

        if record!(monitor, T(cur_frob))
            model.verbose && @info "[$algo_name] converged at iteration $iter"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(iter, Float64(cur_frob), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end

    if model.final_svd
        # Final SVD with soft-thresholding (only place λ shrinks singular values)
        M = _final_svd_input(model, X, U_cur, d_cur, V_cur)
        F_final = svd(M)
        final_d = max.(F_final.S .- λ, zero(T))
        n_nz = count(>(zero(T)), final_d)
        n_nz = max(n_nz, 1)  # keep at least 1
        model.U = F_final.U[:, 1:n_nz]
        model.d = final_d[1:n_nz]
        model.V = V_cur * F_final.V[:, 1:n_nz]
    else
        model.U = U_cur
        model.d = d_cur
        model.V = V_cur
    end

    # Set user/item factors for compatibility with AbstractMatrixFactorization
    model.user_factors = (model.U .* sqrt.(model.d)')'  # rank × m
    model.item_factors = (model.V .* sqrt.(model.d)')'  # rank × n
    model.is_fitted = true
    model
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Final SVD input — dispatched by type
# ──────────────────────────────────────────────────────────────────────────────

function _final_svd_input(::SoftImpute, X::SparseMatrixCSC{Tv,Ti},
                          U_cur::Matrix{T}, d_cur::Vector{T}, V_cur::Matrix{T}) where {Tv,Ti,T}
    # M = X_corrected * V + U * diag(d)  where X_corrected = X - approx at observed
    approx_vals = _sparse_approx_values(X, U_cur .* sqrt.(d_cur)', V_cur .* sqrt.(d_cur)')
    X_corrected = copy(X)
    nonzeros(X_corrected) .= nonzeros(X) .- Tv.(approx_vals)
    X_corrected * V_cur .+ U_cur .* d_cur'
end

function _final_svd_input(::SoftSVD, X::SparseMatrixCSC{Tv,Ti},
                          U_cur::Matrix{T}, d_cur::Vector{T}, V_cur::Matrix{T}) where {Tv,Ti,T}
    X * V_cur
end

# ──────────────────────────────────────────────────────────────────────────────
# SoftImpute-ALS step (Hastie et al. 2014, Algorithm 4)
#
# Solves the Ridge-regularized subproblem incorporating λ as damping d/(d+λ)
# rather than explicit soft-thresholding.
# ──────────────────────────────────────────────────────────────────────────────
function _softimpute_step(
    X::SparseMatrixCSC{Tv,Ti},
    other_vecs::Matrix{T},   # the "fixed" side's vectors (e.g., V when solving for U)
    d::Vector{T},            # current singular values
    self_vecs::Matrix{T},    # the side being updated (e.g., U)
    λ::T,
) where {Tv, Ti, T}
    sqrt_d = sqrt.(d)
    A_scaled = self_vecs .* sqrt_d'
    B_scaled = other_vecs .* sqrt_d'

    approx_vals = _sparse_approx_values(X, A_scaled, B_scaled)
    X_delta = copy(X)
    nonzeros(X_delta) .= nonzeros(X) .- Tv.(approx_vals)

    # first = X_delta * other_vecs * diag(sqrt(d) / (d + λ))
    damp1 = sqrt_d ./ (d .+ λ)
    first = X_delta * other_vecs
    first .*= damp1'

    # second = self_vecs * diag(d * sqrt(d) / (d + λ))
    damp2 = d .* sqrt_d ./ (d .+ λ)
    second = self_vecs .* damp2'

    # result = (first + second) * diag(sqrt(d))
    result = (first .+ second) .* sqrt_d'
    result
end

"""
    _sparse_approx_values(X, A, B)

Compute the values of `A * B'` at the non-zero positions of sparse matrix `X`.
"""
function _sparse_approx_values(X::SparseMatrixCSC{Tv,Ti}, A::Matrix{T}, B::Matrix{T}) where {Tv,Ti,T}
    k = size(A, 2)
    rv = rowvals(X)
    result = Vector{T}(undef, nnz(X))
    pos = 1
    @inbounds for col in axes(X, 2)
        for idx in nzrange(X, col)
            row = rv[idx]
            val = zero(T)
            for f in 1:k
                val += A[row, f] * B[col, f]
            end
            result[pos] = val
            pos += 1
        end
    end
    result
end
