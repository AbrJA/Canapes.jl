# ──────────────────────────────────────────────────────────────────────────────
# GloVe — Global Vectors for co-occurrence matrix factorization
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Pennington, Socher, Manning (2014)
#   "GloVe: Global Vectors for Word Representation"
#
# Loss (½ convention, matching the Stanford C implementation, Python ports,
# and rsparse's GloVe.cpp):
#   L = ½ Σ_{i,j} f(X_{ij}) (wᵢᵀ w̃ⱼ + bᵢ + b̃ⱼ - log X_{ij})²
#
# where f(x) = (x/x_max)^α if x < x_max, else 1.
# The ½ makes the gradient of each squared term equal to the residual itself
# (no floating factor of 2), so `lr` has the same semantics as the
# reference implementations and loss curves are directly comparable.
# ──────────────────────────────────────────────────────────────────────────────

"""
    GloVe{T} <: AbstractMatrixFactorization

GloVe matrix factorization with deterministic AdaGrad-based SGD.

Learns word/item embeddings from a co-occurrence matrix by factorizing
the log-count matrix with a weighting function that caps frequent pairs.

# Constructor
```julia
GloVe(; rank=50, x_max=100.0, lr=0.05, α=0.75, λ=0.0,
        max_iter=25, tol=-1.0, shuffle=false, verbose=true)
```

# Example
```julia
julia> using SparseArrays

julia> C = sprand(MersenneTwister(1), 60, 60, 0.2);  # co-occurrence matrix

julia> C = C + C';

julia> nonzeros(C) .= abs.(nonzeros(C)) .+ 0.1;  # positive, symmetric

julia> model = GloVe(rank=8, max_iter=2, verbose=false);

julia> fit!(model, C; rng=MersenneTwister(2));

julia> size(recommend(model, C; k=5))
(60, 5)

julia> size(embeddings(model))
(8, 60)
```
"""
mutable struct GloVe{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const x_max::T
    lr::T
    const α::T
    const λ::T
    const max_iter::Int
    const tol::T
    const shuffle::Bool
    const verbose::Bool
    # Embeddings (rank × n)
    W_main::Matrix{T}
    W_ctx::Matrix{T}
    b_main::Vector{T}
    b_ctx::Vector{T}
    # AdaGrad accumulators
    grad_W_main::Matrix{T}
    grad_W_ctx::Matrix{T}
    grad_b_main::Vector{T}
    grad_b_ctx::Vector{T}
    loss_history::Vector{T}
    is_fitted::Bool
end

function GloVe(;
    rank::Int = 50,
    x_max::Float64 = 100.0,
    lr::Float64 = 0.05,
    α::Float64 = 0.75,
    λ::Float64 = 0.0,
    max_iter::Int = 25,
    tol::Float64 = -1.0,
    shuffle::Bool = false,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    x_max > 0.0 || throw(ArgumentError("x_max must be positive, got $x_max"))
    lr > 0.0 || throw(ArgumentError("lr must be positive, got $lr"))
    GloVe{T}(
        rank, T(x_max), T(lr), T(α), T(λ), max_iter, T(tol), shuffle, verbose,
        Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
        T[], T[],
        Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
        T[], T[],
        T[], false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::GloVe, X; rng, callbacks) -> model

Fit GloVe on a square co-occurrence matrix `X` (all values must be positive).

Parallelism uses a deterministic reordered scheme with word ownership
(no Hogwild-style races):

1. *Gradient pass* — all co-occurrences are scored against the epoch-start
   embeddings (read-only), storing each pair's gradient.
2. *Main-vector pass* — each thread updates only the main vectors/AdaGrad
   state of its owned word block.
3. *Context-vector pass* — each thread updates only the context vectors of its
   owned word block.

Each pass is embarrassingly parallel over words, so the learned embeddings are
bit-identical regardless of the number of threads. The loss curve is the
epoch-start objective, which stays monotone-decreasing in practice and matches
a single-threaded run exactly.
"""
function fit!(model::GloVe{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n = size(X, 1)
    size(X, 1) == size(X, 2) || throw(ArgumentError("GloVe requires a square co-occurrence matrix, got $(size(X, 1))×$(size(X, 2))"))
    n > 0 || throw(ArgumentError("GloVe requires a non-empty co-occurrence matrix"))
    nnz(X) > 0 || throw(ArgumentError("GloVe requires at least one co-occurrence"))
    _require_finite_input(X, "GloVe")
    all(x -> x > 0, nonzeros(X)) || throw(ArgumentError("All co-occurrence values must be positive"))

    k = model.rank
    run_callbacks_train_begin(callbacks, model)
    try

    # Initialize embeddings
    model.W_main    = (rand(rng, T, k, n) .- T(0.5))
    model.W_ctx     = (rand(rng, T, k, n) .- T(0.5))
    model.b_main    = (rand(rng, T, n) .- T(0.5))
    model.b_ctx     = (rand(rng, T, n) .- T(0.5))
    model.grad_W_main = ones(T, k, n)
    model.grad_W_ctx  = ones(T, k, n)
    model.grad_b_main = ones(T, n)
    model.grad_b_ctx  = ones(T, n)
    model.loss_history = T[]

    nnz_count = nnz(X)
    X_csr = to_csr(X)

    # Per-pair buffers indexed by CSR position: stored gradient and cost term.
    grad_buf = Vector{T}(undef, nnz_count)
    cost_buf = Vector{T}(undef, nnz_count)
    # CSC (context-major) position → CSR (main-major) position mapping.
    perm = _glove_csc_to_csr_pos(X_csr, X)
    # Optional within-word pair permutation (seeded, deterministic).
    shuffle_perm = model.shuffle ? _glove_within_word_perm(X_csr, rng) : nothing

    monitor = ConvergenceMonitor{T}(tol=T(model.tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()
        epoch_cost = _glove_epoch!(model, X_csr, X, grad_buf, cost_buf, perm,
                                   shuffle_perm)

        if isnan(epoch_cost)
            error("GloVe: cost became NaN — try a smaller lr")
        end

        avg_cost = epoch_cost / nnz_count
        push!(model.loss_history, avg_cost)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("GloVe", iter, model.max_iter, Float64(avg_cost),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, avg_cost)
            model.verbose && @info "[GloVe] converged at iteration $iter"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(iter, Float64(avg_cost), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end
    model.is_fitted = true
    model
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

"""
    _glove_epoch!(model, X_csr, X, grad_buf, cost_buf, perm, shuffle_perm; nt) -> cost

Run one deterministic reordered GloVe epoch:

- Phase 1 (gradient): score every co-occurrence against the epoch-start state,
  storing per-pair gradient in `grad_buf` and the cost term in `cost_buf`.
- Phase 2 (main): per-word AdaGrad updates of `W_main`/`b_main` by word owner.
- Phase 3 (context): per-word AdaGrad updates of `W_ctx`/`b_ctx` by word owner.

The result is bit-identical for any `nt` because each phase is embarrassingly
parallel over words and phases read only state they do not write.
"""
function _glove_epoch!(model::GloVe{T},
                       X_csr::SparseMatricesCSR.SparseMatrixCSR,
                       X::SparseMatrixCSC,
                       grad_buf::Vector{T}, cost_buf::Vector{T},
                       perm::Vector{Int32},
                       shuffle_perm::Union{Nothing,Vector{Int32}};
                       nt::Int = Threads.nthreads()) where {T}
    k  = model.rank
    lr = model.lr
    x_max = model.x_max
    α  = model.α
    λ  = model.λ
    n  = size(X, 1)

    W  = model.W_main
    Wc = model.W_ctx
    b  = model.b_main
    bc = model.b_ctx
    gW  = model.grad_W_main
    gWc = model.grad_W_ctx
    gb  = model.grad_b_main
    gbc = model.grad_b_ctx

    # ── Phase 1: gradient pass (epoch-start state, read-only) ──
    Threads.@threads for chunk in 1:nt
        for i in _thread_chunk_bounds(chunk, n, nt)
            @inbounds for p in nzrange(X_csr, i)
                j = Int(X_csr.colval[p])
                x_ij = T(X_csr.nzval[p])
                weight = x_ij < x_max ? (x_ij / x_max)^α : one(T)
                diff = b[i] + bc[j] - log(x_ij)
                @inbounds @simd for f in 1:k
                    diff += W[f, i] * Wc[f, j]
                end
                # ½-convention loss; gradient = weight·diff (no factor of 2).
                cost_buf[p] = T(0.5) * weight * diff^2
                grad_buf[p] = weight * diff
            end
        end
    end
    cost = sum(cost_buf)

    # ── Phase 2: main-vector AdaGrad updates (word ownership) ──
    Threads.@threads for chunk in 1:nt
        for i in _thread_chunk_bounds(chunk, n, nt)
            positions = shuffle_perm === nothing ? nzrange(X_csr, i) :
                @view(shuffle_perm[X_csr.rowptr[i]:X_csr.rowptr[i+1]-1])
            @inbounds for p in positions
                j = Int(X_csr.colval[p])
                g = grad_buf[p]
                @simd for f in 1:k
                    g_main = g * Wc[f, j] + λ * W[f, i]
                    gW[f, i] += g_main * g_main
                    W[f, i]  -= lr * g_main / (sqrt(gW[f, i]) + T(1e-8))
                end
                gb[i] += g * g
                b[i]  -= lr * g / (sqrt(gb[i]) + T(1e-8))
            end
        end
    end

    # ── Phase 3: context-vector AdaGrad updates (word ownership) ──
    rv = rowvals(X)
    Threads.@threads for chunk in 1:nt
        for j in _thread_chunk_bounds(chunk, n, nt)
            @inbounds for q in nzrange(X, j)
                i = Int(rv[q])
                g = grad_buf[perm[q]]
                @simd for f in 1:k
                    g_ctx = g * W[f, i] + λ * Wc[f, j]
                    gWc[f, j] += g_ctx * g_ctx
                    Wc[f, j]  -= lr * g_ctx / (sqrt(gWc[f, j]) + T(1e-8))
                end
                gbc[j] += g * g
                bc[j]  -= lr * g / (sqrt(gbc[j]) + T(1e-8))
            end
        end
    end

    cost
end

"""
    _glove_csc_to_csr_pos(X_csr, X) -> Vector{Int32}

For each CSC position `q` (context-major), return the CSR position of the same
co-occurrence `(i, j)`. Used to look up per-pair gradients in the context pass.
"""
function _glove_csc_to_csr_pos(X_csr::SparseMatricesCSR.SparseMatrixCSR,
                               X::SparseMatrixCSC)
    n = size(X, 2)
    perm = Vector{Int32}(undef, nnz(X))
    rv = rowvals(X)
    @inbounds for j in 1:n
        for q in nzrange(X, j)
            i = Int(rv[q])
            rng_i = nzrange(X_csr, i)
            lo, hi = first(rng_i), last(rng_i)
            found = false
            while lo <= hi
                mid = (lo + hi) >>> 1
                cv = X_csr.colval[mid]
                if cv < j
                    lo = mid + 1
                elseif cv > j
                    hi = mid - 1
                else
                    perm[q] = Int32(mid)
                    found = true
                    break
                end
            end
            found || error("GloVe: internal error — co-occurrence ($i, $j) not found")
        end
    end
    perm
end

"""
    _glove_within_word_perm(X_csr, rng) -> Vector{Int32}

Seeded, deterministic permutation of the co-occurrence positions within each
main word's block. Only built when `shuffle=true`; the identity order is used
otherwise. This preserves the `shuffle` option's SGD-noise role while keeping
the word-ownership grouping intact.
"""
function _glove_within_word_perm(X_csr::SparseMatricesCSR.SparseMatrixCSR,
                                 rng::AbstractRNG)
    n = length(X_csr.rowptr) - 1
    perm = Vector{Int32}(undef, nnz(X_csr))
    @inbounds for i in 1:n
        rng_i = nzrange(X_csr, i)
        for (t, p) in enumerate(rng_i)
            perm[X_csr.rowptr[i] + t - 1] = Int32(p)
        end
        _inplace_shuffle!(@view(perm[X_csr.rowptr[i]:X_csr.rowptr[i+1]-1]), rng)
    end
    perm
end

"""
    embeddings(model::GloVe) -> Matrix

Return the combined word embeddings `W_main + W_ctx` (each column is an embedding vector).
"""
function embeddings(model::GloVe{T}) where {T}
    _require_fitted(model.is_fitted)
    model.W_main .+ model.W_ctx
end

"""
    recommend(model::GloVe, X; k=10) -> Matrix{Int}

Return top-k most similar items per row, excluding self-interactions.
Uses the combined GloVe embeddings for scoring.
"""
function recommend(model::GloVe{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    _require_fitted(model.is_fitted)
    E = embeddings(model)
    _validate_recommend_input(X, size(E, 2), k)
    _predict_topk_batched(E, E, to_csr(X), k)
end

"""
    score(model::GloVe, X) -> Matrix

Return the full score matrix using combined embeddings: E' * E.
"""
function score(model::GloVe{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    E = embeddings(model)
    size(X, 2) == size(E, 2) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(E, 2))"))
    E' * E
end

"""
    score(model::GloVe, row_indices, col_indices) -> Vector

Return pairwise similarity scores for specific (row, col) index pairs.
"""
function score(model::GloVe{T}, row_indices::AbstractVector{<:Integer},
               col_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    E = embeddings(model)
    _predict_pairwise_scores(E, E, row_indices, col_indices)
end
