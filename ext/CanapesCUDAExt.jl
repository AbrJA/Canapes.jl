# ──────────────────────────────────────────────────────────────────────────────
# CanapesCUDAExt — GPU acceleration for Canapes.jl algorithms
# ──────────────────────────────────────────────────────────────────────────────
#
# This extension is loaded automatically when CUDA.jl is available.
# It provides GPU-accelerated versions of key operations:
# - ShallowAutoencoder: Sparse Gramian computation via cuSPARSE + dense inverse on GPU
# - CachedALS: GPU Gramian caching with pre-allocated per-thread CPU solves
# - WeightedMF: GPU Gramian (cuBLAS syrk) with per-thread CPU ALS solves
# - Score computation: Batch user-item scoring on GPU
# ──────────────────────────────────────────────────────────────────────────────

module CanapesCUDAExt

using Canapes
using CUDA
using CUDA.CUSPARSE
using CUDA.CUBLAS
using LinearAlgebra
using SparseArrays
using Random

# ──────────────────────────────────────────────────────────────────────────────
# GPU utility kernels
# ──────────────────────────────────────────────────────────────────────────────

function _gpu_diag_add_kernel!(A, val)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= min(size(A, 1), size(A, 2))
        @inbounds A[i, i] += val
    end
    return nothing
end

function _gpu_diag_zero_kernel!(A)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= min(size(A, 1), size(A, 2))
        @inbounds A[i, i] = zero(eltype(A))
    end
    return nothing
end

function _gpu_add_to_diag!(A::CuMatrix{T}, val) where T
    n = min(size(A, 1), size(A, 2))
    threads = min(256, n)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _gpu_diag_add_kernel!(A, T(val))
    return A
end

function _gpu_set_diag_zero!(A::CuMatrix)
    n = min(size(A, 1), size(A, 2))
    threads = min(256, n)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _gpu_diag_zero_kernel!(A)
    return A
end

function _gpu_compute_B_kernel!(B, P, n)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= n && j <= n
        if i == j
            @inbounds B[i, j] = zero(eltype(B))
        else
            @inbounds B[i, j] = -P[i, j] / P[j, j]
        end
    end
    return nothing
end

"""
    _estimate_gpu_memory(n_floats, T) -> Int

Estimate GPU memory required for n_floats of type T in bytes.
"""
_estimate_gpu_memory(n_floats::Int, ::Type{T}) where T = n_floats * sizeof(T)

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated ShallowAutoencoder
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit_gpu!(model::ShallowAutoencoder, X) -> model

GPU-accelerated ShallowAutoencoder fitting. Uses cuSPARSE for the sparse Gramian XᵀX
and cuSOLVER for the Cholesky inverse, keeping all heavy computation on GPU.
Falls back to CPU if insufficient GPU memory.
"""
function Canapes.fit_gpu!(model::Canapes.ShallowAutoencoder{T}, X::SparseMatrixCSC{Tv,Ti}) where {T,Tv,Ti}
    n_users, n_items = size(X)
    Canapes._require_nonempty_dimensions(X, "ShallowAutoencoder-GPU")
    Canapes._require_finite_input(X, "ShallowAutoencoder-GPU")
    # Same fit-time peak estimate as the CPU path (G, factor, P, B) plus the
    # densified X copy on the GPU.
    Canapes._require_fit_memory(Canapes._fit_memory_estimate(n_items, 4, T),
                               model.max_memory, "ShallowAutoencoder-GPU")

    old_B = model.B
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try

    model.verbose && @info "[ShallowAutoencoder-GPU] Computing Gramian via cuSPARSE ($(n_items) items)..."

    # Memory check: dense n_users×n_items X + ~3 dense n_items×n_items on GPU
    free_mem = CUDA.free_memory()
    estimated_mem = _estimate_gpu_memory(n_items * n_items * 3 + n_users * n_items, T)

    if estimated_mem > free_mem * 0.8
        @warn "[ShallowAutoencoder-GPU] Insufficient GPU memory (need ~$(estimated_mem ÷ 1_000_000)MB, " *
              "have ~$(free_mem ÷ 1_000_000)MB), falling back to CPU"
        Canapes.fit!(model, X)
        return model
    end

    # Transfer sparse matrix to GPU and compute Gramian G = XᵀX
    # Use dense representation for XᵀX since the result is dense anyway
    X_gpu = CuSparseMatrixCSC{T}(X)
    X_dense_gpu = CuMatrix{T}(X_gpu)
    G_gpu = X_dense_gpu' * X_dense_gpu
    X_gpu = nothing
    X_dense_gpu = nothing
    CUDA.reclaim()

    # Add regularization: G += λI
    _gpu_add_to_diag!(G_gpu, model.λ)

    model.verbose && @info "[ShallowAutoencoder-GPU] Computing Cholesky inverse on GPU ($(n_items)×$(n_items))..."

    # Cholesky factorization and inversion on GPU
    C_gpu = cholesky(Symmetric(G_gpu))
    P_gpu = inv(C_gpu)
    G_gpu = nothing
    CUDA.reclaim()

    # Compute B entirely on GPU: B_ij = -P_ij / P_jj, B_ii = 0
    B_gpu = CuMatrix{T}(undef, n_items, n_items)
    threads_2d = (16, 16)
    blocks_2d = (cld(n_items, 16), cld(n_items, 16))
    @cuda threads=threads_2d blocks=blocks_2d _gpu_compute_B_kernel!(B_gpu, P_gpu, n_items)
    P_gpu = nothing
    CUDA.reclaim()

    # Transfer result back to CPU
    model.B = Array(B_gpu)
    B_gpu = nothing
    CUDA.reclaim()

    model.is_fitted = true
    model.verbose && @info "[ShallowAutoencoder-GPU] Done. B matrix: $(n_items)×$(n_items)"
    model
    catch
        model.B = old_B
        model.is_fitted = old_is_fitted
        rethrow()
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated CachedALS
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit_gpu!(model::CachedALS, X; rng) -> model

GPU-accelerated CachedALS. Uses cuBLAS syrk for Gramian computation (V*Vᵀ and U*Uᵀ),
with pre-allocated per-thread buffers for the CPU-side per-user Cholesky solves.
"""
function Canapes.fit_gpu!(model::Canapes.CachedALS{T}, X::SparseMatrixCSC{Tv,Ti};
                         rng::Random.AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_users, n_items = size(X)
    Canapes._require_nonempty_dimensions(X, "CachedALS-GPU")
    Canapes._require_finite_input(X, "CachedALS-GPU")
    model.solver isa Canapes.CholeskySolver || throw(ArgumentError(
        "fit_gpu!(CachedALS) supports only CholeskySolver, got $(model.solver)"))

    old_U = model.user_factors
    old_V = model.item_factors
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try

    k = model.rank
    α = model.α
    λ = model.λ

    U = randn(rng, T, k, n_users) .* T(0.01)
    V = randn(rng, T, k, n_items) .* T(0.01)

    model.verbose && @info "[CachedALS-GPU] Training rank=$k, $(n_users) users × $(n_items) items"

    X_csr = Canapes.to_csr(X)
    monitor = Canapes.ConvergenceMonitor{T}(tol=T(model.tol), min_iter=2)

    # Pre-allocate per-thread buffers (avoids allocation inside @threads loop)
    nt = Threads.nthreads()
    A_bufs = [Matrix{T}(undef, k, k) for _ in 1:nt]
    b_bufs = [Vector{T}(undef, k) for _ in 1:nt]

    # Persistent GPU buffer for Gramian
    gramian_gpu = CuMatrix{T}(undef, k, k)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # ── Compute item Gramian on GPU: VVᵀ + λI ──
        V_gpu = CuMatrix{T}(V)
        CUBLAS.syrk!('U', 'N', one(T), V_gpu, zero(T), gramian_gpu)
        CUDA.@sync gramian_gpu
        gramian = Array(gramian_gpu)
        LinearAlgebra.copytri!(gramian, 'U')
        @inbounds for i in 1:k
            gramian[i, i] += λ
        end
        V_gpu = nothing

        # ── Update users with pre-allocated buffers ──
        _gpu_ials_update_buffered!(U, V, X_csr, gramian, α, k, A_bufs, b_bufs,
                                   u -> nzrange(X_csr, u),
                                   idx -> Int(X_csr.colval[idx]),
                                   idx -> T(X_csr.nzval[idx]))

        # ── Compute user Gramian on GPU: UUᵀ + λI ──
        U_gpu = CuMatrix{T}(U)
        CUBLAS.syrk!('U', 'N', one(T), U_gpu, zero(T), gramian_gpu)
        CUDA.@sync gramian_gpu
        gramian = Array(gramian_gpu)
        LinearAlgebra.copytri!(gramian, 'U')
        @inbounds for i in 1:k
            gramian[i, i] += λ
        end
        U_gpu = nothing

        # ── Update items with pre-allocated buffers ──
        _gpu_ials_update_buffered!(V, U, X, gramian, α, k, A_bufs, b_bufs,
                                   j -> nzrange(X, j),
                                   idx -> Int(rowvals(X)[idx]),
                                   idx -> T(nonzeros(X)[idx]))

        loss = _ials_loss(U, V, X, α, λ)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = Canapes.elapsed_seconds(monitor)

        if model.verbose
            Canapes.log_iteration("CachedALS-GPU", iter, model.max_iter, Float64(loss),
                                iter_seconds, total_seconds)
        end

        if Canapes.record!(monitor, loss)
            model.verbose && @info "[CachedALS-GPU] converged at iteration $iter"
            break
        end
    end

    gramian_gpu = nothing
    CUDA.reclaim()

    model.user_factors = U
    model.item_factors = V
    model.is_fitted = true
    model
    catch
        model.user_factors = old_U
        model.item_factors = old_V
        model.is_fitted = old_is_fitted
        rethrow()
    end
end

"""
Per-user/item ALS update with pre-allocated per-thread Gramian and RHS buffers.
Avoids allocation inside the inner loop — critical for performance.
"""
function _gpu_ials_update_buffered!(target::Matrix{T}, source::Matrix{T}, R,
                                    gramian::Matrix{T}, α::T, k::Int,
                                    A_bufs::Vector{Matrix{T}},
                                    b_bufs::Vector{Vector{T}},
                                    get_range, get_col, get_val) where {T}
    n = size(target, 2)
    nt = length(A_bufs)
    Threads.@threads for chunk in 1:nt
        A = A_bufs[chunk]
        b = b_bufs[chunk]

        for u in Canapes._thread_chunk_bounds(chunk, n, nt)

        # A ← gramian (copy into pre-allocated buffer)
        copyto!(A, gramian)
        fill!(b, zero(T))

        @inbounds for idx in get_range(u)
            i = get_col(idx)
            r_ui = get_val(idx)
            c_ui = α * r_ui
            for q in 1:k
                sq = source[q, i]
                b[q] += sq * (one(T) + c_ui)
                for p in 1:k
                    A[p, q] += c_ui * source[p, i] * sq
                end
            end
        end

        # Solve via in-place CholeskySolver
        x = cholesky!(Symmetric(A)) \ b
        @inbounds for f in 1:k
            target[f, u] = x[f]
        end
        end
    end
end

"""
Compute reconstruction loss for CachedALS (CPU, used for convergence monitoring).
"""
function _ials_loss(U::Matrix{T}, V::Matrix{T}, X::SparseMatrixCSC, α::T, λ::T) where {T}
    k = size(U, 1)
    loss = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            u = rv[idx]
            r = T(nz[idx])
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, u] * V[f, j]
            end
            c = one(T) + α * r
            loss += c * (one(T) - pred)^2
        end
    end
    loss += λ * (sum(abs2, U) + sum(abs2, V))
    loss
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated WeightedMF
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit_gpu!(model::WeightedMF, X; rng, U_init, V_init) -> model

GPU-accelerated WeightedMF. Uses cuBLAS syrk for Gramian computation (YᵀY)
on GPU, then performs per-user/item Cholesky solves on CPU with
pre-allocated per-thread buffers.

This provides significant speedup for large item/user counts where the
Gramian computation (O(k² × n_items)) dominates iteration cost.
"""
function Canapes.fit_gpu!(model::Canapes.WeightedMF{T}, X::SparseMatrixCSC{Tv,Ti};
                         rng::Random.AbstractRNG = Random.default_rng(),
                         U_init::Union{Nothing, Matrix{T}} = nothing,
                         V_init::Union{Nothing, Matrix{T}} = nothing) where {T,Tv,Ti}
    n_users, n_items = size(X)
    Canapes._require_nonempty_dimensions(X, "WeightedMF-GPU")
    Canapes._require_finite_input(X, "WeightedMF-GPU")
    model.solver isa Union{Canapes.CholeskySolver, Canapes.NonNegativeSolver} || throw(
        ArgumentError("fit_gpu!(WeightedMF) supports only CholeskySolver and NonNegativeSolver, got $(model.solver)"))

    old_U = model.user_factors
    old_V = model.item_factors
    old_is_fitted = model.is_fitted
    model.is_fitted = false
    try

    k = model.rank
    λ = model.λ
    α = model.α
    is_implicit = model.feedback == Canapes.Implicit

    # Initialize factors
    model.user_factors = isnothing(U_init) ? Canapes.init_factors(rng, k, n_users) : copy(U_init)
    model.item_factors = isnothing(V_init) ? Canapes.init_factors(rng, k, n_items) : copy(V_init)

    model.verbose && @info "[WeightedMF-GPU] Training rank=$k, solver=$(model.solver), $(n_users) users × $(n_items) items"

    # Build transpose for row access
    Xt = SparseMatrixCSC(X')

    monitor = Canapes.ConvergenceMonitor{T}(tol=T(model.tol), min_iter=2)

    # Pre-allocate per-thread buffers
    nt = Threads.nthreads()
    gram_bufs = [Matrix{T}(undef, k, k) for _ in 1:nt]
    rhs_bufs  = [Vector{T}(undef, k) for _ in 1:nt]

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # ── Update users: fix items, compute item Gramian on GPU ──
        _gpu_weightedmf_sweep!(model, Xt, model.user_factors, model.item_factors,
                         n_users, gram_bufs, rhs_bufs)

        # ── Update items: fix users, compute user Gramian on GPU ──
        _gpu_weightedmf_sweep!(model, X, model.item_factors, model.user_factors,
                         n_items, gram_bufs, rhs_bufs)

        loss = Canapes._compute_loss(model, X)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = Canapes.elapsed_seconds(monitor)

        if model.verbose
            Canapes.log_iteration("WeightedMF-GPU", iter, model.max_iter, Float64(loss),
                                iter_seconds, total_seconds)
        end

        if Canapes.record!(monitor, loss)
            model.verbose && @info "[WeightedMF-GPU] converged at iteration $iter"
            break
        end
    end

    model.is_fitted = true
    model
    catch
        model.user_factors = old_U
        model.item_factors = old_V
        model.is_fitted = old_is_fitted
        rethrow()
    end
end

"""
Single ALS sweep with GPU-accelerated Gramian computation via cuBLAS syrk.
The Gramian YᵀY is computed on GPU, then per-entity solves run on CPU.
"""
function _gpu_weightedmf_sweep!(
    model::Canapes.WeightedMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
    gram_bufs::Vector{Matrix{T}},
    rhs_bufs::Vector{Vector{T}},
) where {T}
    k = model.rank
    λ = model.λ
    α = model.α
    is_implicit = model.feedback == Canapes.Implicit
    is_nnls = model.solver isa Canapes.NonNegativeSolver

    # ── Compute YᵀY on GPU via cuBLAS syrk ──
    fixed_gpu = CuMatrix{T}(fixed)
    YtY_gpu = CuMatrix{T}(undef, k, k)
    CUBLAS.syrk!('U', 'N', one(T), fixed_gpu, zero(T), YtY_gpu)
    CUDA.@sync YtY_gpu
    YtY = Array(YtY_gpu)
    LinearAlgebra.copytri!(YtY, 'U')
    fixed_gpu = nothing
    YtY_gpu = nothing

    rv = rowvals(A)
    nz = nonzeros(A)

    # ── Per-entity Cholesky solves on CPU with pre-allocated buffers ──
    nt = length(gram_bufs)
    Base.Threads.@threads for chunk in 1:nt
        gram = gram_bufs[chunk]
        rhs = rhs_bufs[chunk]

        for u in Canapes._thread_chunk_bounds(chunk, n_entities, nt)

        # gram ← YᵀY + λI
        copyto!(gram, YtY)
        @inbounds for d in 1:k
            gram[d, d] += λ
        end
        fill!(rhs, zero(T))

        for idx in nzrange(A, u)
            i = rv[idx]
            rui = T(nz[idx])
            yi = @view fixed[:, i]

            if is_implicit
                cui = one(T) + α * rui
                BLAS.syr!('U', cui - one(T), yi, gram)
                BLAS.axpy!(cui, yi, rhs)
            else
                BLAS.axpy!(rui, yi, rhs)
            end
        end

        # Mirror upper triangle
        LinearAlgebra.copytri!(gram, 'U')

        if is_nnls
            Canapes._nnls_cd!(rhs, YtY, fixed, rv, nz, nzrange(A, u),
                             k, α, λ, is_implicit)
            @inbounds factors[:, u] .= rhs
        else
            # In-place Cholesky solve
            _, info = LAPACK.potrf!('U', gram)
            if info == 0
                LAPACK.potrs!('U', gram, rhs)
                @inbounds for f in 1:k
                    factors[f, u] = rhs[f]
                end
            else
                # Fallback: add more regularization
                @inbounds for d in 1:k
                    gram[d, d] += λ
                end
                LinearAlgebra.copytri!(gram, 'U')
                _, info2 = LAPACK.potrf!('U', gram)
                if info2 == 0
                    LAPACK.potrs!('U', gram, rhs)
                end
                @inbounds for f in 1:k
                    factors[f, u] = rhs[f]
                end
            end
        end
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated score computation
# ──────────────────────────────────────────────────────────────────────────────

"""
    score_gpu(model, X) -> Matrix

Compute full score matrix U'V on GPU via cuBLAS gemm, transfer back to CPU.
Works for any model with `user_factors` and `item_factors` fields.

Memory: O(n_users × n_items) on GPU. For very large problems, use
`recommend_gpu` which streams results in batches.
"""
function Canapes.score_gpu(model, X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    Canapes._require_fitted(model.is_fitted)
    T = eltype(model.user_factors)
    n_users = size(model.user_factors, 2)
    n_items = size(model.item_factors, 2)
    size(X, 2) == n_items || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $n_items"))

    # Check GPU memory
    free_mem = CUDA.free_memory()
    needed = _estimate_gpu_memory(n_users * n_items + size(model.user_factors, 1) *
             (n_users + n_items), T)
    if needed > free_mem * 0.8
        @warn "[score_gpu] Insufficient GPU memory, falling back to CPU"
        return model.user_factors' * model.item_factors
    end

    U_gpu = CuMatrix{T}(model.user_factors)
    V_gpu = CuMatrix{T}(model.item_factors)
    S_gpu = U_gpu' * V_gpu
    S = Array(S_gpu)
    U_gpu = nothing
    V_gpu = nothing
    S_gpu = nothing
    CUDA.reclaim()
    S
end

"""
    recommend_gpu(model, X; k=10, batch_size=0) -> Matrix{Int}

GPU-accelerated top-k prediction. Computes scores on GPU in batches
(auto-sized to available GPU memory), masks seen items, and selects
top-k on CPU.

# Arguments
- `k::Int` — number of items to recommend per user
- `batch_size::Int` — users per batch (0 = auto based on GPU memory)
"""
function Canapes.recommend_gpu(model, X::SparseMatrixCSC; k::Int=10, batch_size::Int=0)
    Canapes._require_fitted(model.is_fitted)
    T_elem = eltype(model.user_factors)
    n_users = size(model.user_factors, 2)
    n_items = size(model.item_factors, 2)
    size(X, 2) == n_items || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $n_items"))
    k_out = min(k, n_items)
    rank = size(model.user_factors, 1)

    # Determine batch size based on GPU memory
    if batch_size <= 0
        free_mem = CUDA.free_memory()
        bytes_per_user = sizeof(T_elem) * n_items
        factor_bytes = sizeof(T_elem) * rank * (n_users + n_items)
        available = floor(Int, free_mem * 0.7) - factor_bytes
        batch_size = max(1, available ÷ bytes_per_user)
        batch_size = min(batch_size, n_users)
    end

    # Transfer item factors to GPU once
    V_gpu = CuMatrix{T_elem}(model.item_factors)
    preds = Matrix{Int}(undef, n_users, k_out)

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        batch_range = batch_start:batch_end
        batch_n = length(batch_range)

        # Compute scores for this batch on GPU
        U_batch_gpu = CuMatrix{T_elem}(model.user_factors[:, batch_range])
        S_batch_gpu = U_batch_gpu' * V_gpu
        S_batch = Array(S_batch_gpu)
        U_batch_gpu = nothing
        S_batch_gpu = nothing

        # Mask seen items (CPU — fast O(nnz_batch) operation)
        rv = rowvals(X)
        for j in axes(X, 2)
            for idx in nzrange(X, j)
                u = rv[idx]
                if u in batch_range
                    @inbounds S_batch[u - batch_start + 1, j] = T_elem(-Inf)
                end
            end
        end

        # Top-k selection (CPU, parallelized)
        Threads.@threads for local_u in 1:batch_n
            @inbounds preds[batch_start + local_u - 1, :] .=
                partialsortperm(@view(S_batch[local_u, :]), 1:k_out; rev=true)
        end
    end

    V_gpu = nothing
    CUDA.reclaim()
    preds
end

end # module CanapesCUDAExt
