# ──────────────────────────────────────────────────────────────────────────────
# Explicit (rating-prediction) models — BaselineOnly, SlopeOne, PearsonKNN
# ──────────────────────────────────────────────────────────────────────────────
#
# All three subclass AbstractExplicitModel: they predict continuous ratings,
# expose `predict`/`score` (dense n_users × n_items matrices) and pairwise
# scoring, and are evaluated with rmse/mae — never `recommend`.
#
# References:
#   - BaselineOnly: Surprise's biased baseline (Koren 2009), ALS from the
#     sparse ratings.
#   - SlopeOne: Lemire & Maclachlan (2005), "Slope One Predictors for Online
#     Rating-Based Collaborative Filtering".
#   - PearsonKNN: Surprise's KNNWithMeans — user-mean-centered Pearson
#     neighborhood averaging; no training loop beyond the similarity matrix.
# ──────────────────────────────────────────────────────────────────────────────

"""
    BaselineOnly{T} <: AbstractExplicitModel

Baseline rating predictor `ŷ_ui = μ + b_u + b_i` (global mean + user and item
biases), learned by Alternating Least Squares over the observed ratings —
i.e. the baseline term of BiasedMF without the latent factors. The classic
first point of comparison for rating prediction (Surprise `BaselineOnly`).

The bias update is the exact ALS step on the residual `r_ui - μ - b_u - b_i`
with L2 regularization `λ` in the denominator:
`b_u ← Σ_i (r_ui - μ - b_i) / (|I_u| + λ)`, and symmetrically for items.

# Constructor
```julia
BaselineOnly(; λ=0.02, max_iter=10, verbose=true)
```

# Example
```julia
julia> using SparseArrays, Random

julia> X = sprand(MersenneTwister(1), 100, 50, 0.2); nonzeros(X) .= 1 .+ 4 .* rand(MersenneTwister(2), nnz(X));

julia> model = BaselineOnly(verbose=false);

julia> fit!(model, X; rng=MersenneTwister(3));

julia> size(predict(model, X))
(100, 50)
```
"""
mutable struct BaselineOnly{T<:AbstractFloat} <: AbstractExplicitModel
    const λ::T
    const max_iter::Int
    const verbose::Bool
    global_mean::T
    user_bias::Vector{T}
    item_bias::Vector{T}
    is_fitted::Bool
end

function BaselineOnly(;
    λ::Float64 = 0.02,
    max_iter::Int = 10,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    BaselineOnly{T}(T(λ), max_iter, verbose, zero(T), T[], T[], false)
end

function fit!(model::BaselineOnly{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "BaselineOnly")
    _require_finite_input(X, "BaselineOnly")
    λ = model.λ
    is_fitted = model.is_fitted
    global_mean_old, b_u_old, b_i_old = model.global_mean, model.user_bias, model.item_bias
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    model.global_mean = T(sum(nonzeros(X)) / max(nnz(X), 1))
    b_u = zeros(T, n_users)
    b_i = zeros(T, n_items)
    μ = model.global_mean

    X_csr = to_csr(X)
    for _ in 1:model.max_iter
        # user step (uses updated item biases — Gauss-Seidel)
        @inbounds for u in 1:n_users
            s = zero(T); c = zero(T)
            for idx in nzrange(X_csr, u)
                i = Int(X_csr.colval[idx])
                s += T(X_csr.nzval[idx]) - μ - b_i[i]
                c += one(T)
            end
            b_u[u] = s / (c + λ)
        end
        # item step (uses updated user biases)
        for j in 1:n_items
            s = zero(T); c = zero(T)
            for idx in nzrange(X, j)
                u = Int(rowvals(X)[idx])
                s += T(nonzeros(X)[idx]) - μ - b_u[u]
                c += one(T)
            end
            b_i[j] = s / (c + λ)
        end
    end
    model.user_bias = b_u
    model.item_bias = b_i
    model.is_fitted = true
    model
    catch
        model.global_mean = global_mean_old
        model.user_bias = b_u_old
        model.item_bias = b_i_old
        model.is_fitted = is_fitted
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

@inline function _baseline_value(model::BaselineOnly{T}, u::Int, i::Int) where {T}
    model.global_mean + model.user_bias[u] + model.item_bias[i]
end

function score(model::BaselineOnly{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    n_u, n_i = size(X)
    n_u == length(model.user_bias) || throw(DimensionMismatch(
        "X has $n_u users but the fitted model has $(length(model.user_bias))"))
    n_i == length(model.item_bias) || throw(DimensionMismatch(
        "X has $n_i items but the fitted model has $(length(model.item_bias))"))
    S = Matrix{T}(undef, n_u, n_i)
    @inbounds for i in 1:n_i
        bi = model.item_bias[i]
        for u in 1:n_u
            S[u, i] = model.global_mean + model.user_bias[u] + bi
        end
    end
    S
end

function predict(model::BaselineOnly{T}, X::SparseMatrixCSC) where {T}
    score(model, X)
end

function score(model::BaselineOnly{T}, user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    length(user_indices) == length(item_indices) ||
        throw(DimensionMismatch("user_indices and item_indices must have the same length"))
    [_baseline_value(model, u, i) for (u, i) in zip(user_indices, item_indices)]
end
