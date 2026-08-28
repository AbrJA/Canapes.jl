# ──────────────────────────────────────────────────────────────────────────────
# GloVe — Global Vectors for co-occurrence matrix factorization
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Pennington, Socher, Manning (2014)
#   "GloVe: Global Vectors for Word Representation"
#
# Loss:
#   L = Σ_{i,j} f(X_{ij}) (wᵢᵀ w̃ⱼ + bᵢ + b̃ⱼ - log X_{ij})²
#
# where f(x) = (x/x_max)^α if x < x_max, else 1.
# ──────────────────────────────────────────────────────────────────────────────

"""
    GloVe{T} <: AbstractMatrixFactorization

GloVe matrix factorization with AdaGrad-based SGD (Hogwild parallel).

Learns word/item embeddings from a co-occurrence matrix by factorizing
the log-count matrix with a weighting function that caps frequent pairs.

# Constructor
```julia
GloVe(; rank=50, x_max=100.0, learning_rate=0.05, α=0.75, λ=0.0,
        max_iter=25, convergence_tol=-1.0, shuffle=false, verbose=true)
```

# Example
```julia
using SparseArrays, Gideon

# Co-occurrence matrix (must be square, positive values)
cooccur = sprand(10000, 10000, 0.001)
cooccur = cooccur + cooccur'
nonzeros(cooccur) .= abs.(nonzeros(cooccur)) .+ 0.1

model = GloVe(rank=100, x_max=100.0, learning_rate=0.05, max_iter=25)
fit!(model, cooccur)
embeddings = embeddings(model)
```
"""
mutable struct GloVe{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const x_max::T
    learning_rate::T
    const α::T
    const λ::T
    const max_iter::Int
    const convergence_tol::T
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
    learning_rate::Float64 = 0.05,
    α::Float64 = 0.75,
    λ::Float64 = 0.0,
    max_iter::Int = 25,
    convergence_tol::Float64 = -1.0,
    shuffle::Bool = false,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    x_max > 0.0 || throw(ArgumentError("x_max must be positive, got $x_max"))
    learning_rate > 0.0 || throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    T = dtype
    GloVe{T}(
        rank, T(x_max), T(learning_rate), T(α), T(λ), max_iter, T(convergence_tol), shuffle, verbose,
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
Uses Hogwild-style parallel SGD with AdaGrad.
"""
function fit!(model::GloVe{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n = size(X, 1)
    size(X, 1) == size(X, 2) || throw(ArgumentError("GloVe requires a square co-occurrence matrix, got $(size(X, 1))×$(size(X, 2))"))
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

    # Extract COO triplets
    rows, cols, vals = findnz(X)
    nnz_count = length(rows)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()
        order = if model.shuffle
            _inplace_shuffle!(collect(1:nnz_count), rng)
        else
            1:nnz_count
        end
        epoch_cost = _glove_epoch!(model, rows, cols, vals, order)

        if isnan(epoch_cost)
            error("GloVe: cost became NaN — try a smaller learning_rate")
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

function _glove_epoch!(model::GloVe{T}, rows, cols, vals, order) where {T}
    k  = model.rank
    lr = model.learning_rate
    x_max = model.x_max
    α  = model.α
    λ  = model.λ

    W  = model.W_main
    Wc = model.W_ctx
    b  = model.b_main
    bc = model.b_ctx
    gW  = model.grad_W_main
    gWc = model.grad_W_ctx
    gb  = model.grad_b_main
    gbc = model.grad_b_ctx

    # Thread-local cost accumulators
    nt = Threads.maxthreadid()
    local_costs = zeros(T, nt)

    # Hogwild parallel SGD
    Base.Threads.@threads :static for idx in order
        tid  = Threads.threadid()
        i = rows[idx]
        j = cols[idx]
        x_ij = T(vals[idx])

        # Weighting function
        weight = x_ij < x_max ? (x_ij / x_max)^α : one(T)

        # Inner product + biases
        diff = b[i] + bc[j] - log(x_ij)
        @inbounds for f in 1:k
            diff += W[f, i] * Wc[f, j]
        end

        local_costs[tid] += weight * diff^2
        grad_common = T(2) * weight * diff

        # AdaGrad update (Hogwild writes) with numerical floor
        @inbounds @simd for f in 1:k
            g_main = grad_common * Wc[f, j] + λ * W[f, i]
            g_ctx  = grad_common * W[f, i]  + λ * Wc[f, j]
            gW[f, i]  += g_main * g_main
            gWc[f, j] += g_ctx * g_ctx
            W[f, i]  -= lr * g_main / (sqrt(gW[f, i]) + T(1e-8))
            Wc[f, j] -= lr * g_ctx  / (sqrt(gWc[f, j]) + T(1e-8))
        end

        g_bi = grad_common
        g_bj = grad_common
        @inbounds begin
            gb[i]  += g_bi^2
            gbc[j] += g_bj^2
            b[i]  -= lr * g_bi / (sqrt(gb[i]) + T(1e-8))
            bc[j] -= lr * g_bj / (sqrt(gbc[j]) + T(1e-8))
        end
    end
    sum(local_costs)
end

"""
    embeddings(model::GloVe) -> Matrix

Return the combined word embeddings `W_main + W_ctx` (each column is an embedding vector).
"""
function embeddings(model::GloVe{T}) where {T}
    model.is_fitted || error("Model not fitted")
    model.W_main .+ model.W_ctx
end

"""
    recommend(model::GloVe, X; k=10) -> Matrix{Int}

Return top-k most similar items per row, excluding self-interactions.
Uses the combined GloVe embeddings for scoring.
"""
function recommend(model::GloVe{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    model.is_fitted || error("Model not fitted")
    E = embeddings(model)
    _predict_topk_batched(E, E, to_csr(X), k)
end

"""
    score(model::GloVe, X) -> Matrix

Return the full score matrix using combined embeddings: E' * E.
"""
function score(model::GloVe{T}, X::SparseMatrixCSC) where {T}
    model.is_fitted || error("Model not fitted")
    E = embeddings(model)
    E' * E
end

"""
    score(model::GloVe, row_indices, col_indices) -> Vector

Return pairwise similarity scores for specific (row, col) index pairs.
"""
function score(model::GloVe{T}, row_indices::AbstractVector{<:Integer},
               col_indices::AbstractVector{<:Integer}) where {T}
    model.is_fitted || error("Model not fitted")
    E = embeddings(model)
    _predict_pairwise_scores(E, E, row_indices, col_indices)
end
