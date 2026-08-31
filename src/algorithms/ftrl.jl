# ──────────────────────────────────────────────────────────────────────────────
# FTRL — Follow The Regularized Leader (Proximal SGD)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: McMahan et al. (2013)
#   "Ad Click Prediction: a View from the Trenches"
#
# Supports Elastic-Net (L1 + L2) regularization with per-coordinate
# adaptive learning rates. Multiple GLM families: binomial, gaussian, poisson.
# ──────────────────────────────────────────────────────────────────────────────

"""
    FTRL{T} <: AbstractSparseRegression

Follow The Regularized Leader proximal SGD for generalized linear models on sparse data.

Supports three families:
- `LossFamilies.Binomial()` — logistic regression (predictions in [0,1])
- `LossFamilies.Gaussian()` — linear regression (identity link)
- `LossFamilies.Poisson()`  — LossFamilies.Poisson regression (log link, predictions > 0)

# Constructor
```julia
FTRL(; lr=0.1, lr_decay=0.5, λ=0.0, l1_ratio=1.0,
       dropout=0.0, family=LossFamilies.Binomial(), grad_clip=1000.0, verbose=true)
```

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> y = rand(MersenneTwister(3), [0.0, 1.0], 200);

julia> model = FTRL(lr=0.1, λ=0.01, l1_ratio=0.5, max_iter=2, verbose=false);

julia> fit!(model, X, y; rng=MersenneTwister(2));

julia> size(predict(model, X))
(200,)

julia> length(coef(model))
100
```
"""
mutable struct FTRL{T<:AbstractFloat} <: AbstractSparseRegression
    lr::T
    const lr_decay::T
    const λ::T
    const l1_ratio::T
    const dropout::T
    const family::LossFamily
    const grad_clip::T
    const max_iter::Int
    const verbose::Bool
    n_features::Int
    z::Vector{T}
    n::Vector{T}
    is_initialized::Bool
end

function FTRL(;
    lr::Float64 = 0.1,
    lr_decay::Float64 = 0.5,
    λ::Float64 = 0.0,
    l1_ratio::Float64 = 1.0,
    dropout::Float64 = 0.0,
    family::LossFamily = LossFamilies.Binomial(),
    grad_clip::Float64 = 1000.0,
    max_iter::Int = 1,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    0.0 <= dropout < 1.0 || throw(ArgumentError("dropout must be in [0, 1), got $dropout"))
    0.0 <= l1_ratio <= 1.0 || throw(ArgumentError("l1_ratio must be in [0, 1], got $l1_ratio"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    lr > 0.0 || throw(ArgumentError("lr must be positive, got $lr"))
    lr_decay > 0.0 || throw(ArgumentError("lr_decay must be positive, got $lr_decay"))
    grad_clip > 0.0 || throw(ArgumentError("grad_clip must be positive, got $grad_clip"))
    FTRL{T}(T(lr), T(lr_decay), T(λ), T(l1_ratio), T(dropout),
            family, T(grad_clip), max_iter, verbose,
            0, T[], T[], false)
end

# ──────────────────────────────────────────────────────────────────────────────
# update! — single epoch (online / streaming)
# ──────────────────────────────────────────────────────────────────────────────

"""
    update!(model::FTRL, X, y; weights, rng) -> model

Run a single epoch of proximal SGD over the data.
Supports online/streaming learning — can be called repeatedly.
"""
function update!(model::FTRL{T}, X::SparseMatrixCSC{Tv,Ti}, y::AbstractVector;
                      weights::AbstractVector{T} = ones(T, length(y)),
                      rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    iter_start = time_ns()
    n_samples, n_features = size(X)
    n_samples == length(y) || throw(DimensionMismatch("X rows ($n_samples) ≠ length(y) ($(length(y)))"))
    _require_finite_input(X, "FTRL")
    _require_finite_vector(y, "FTRL")

    if !model.is_initialized
        model.n_features = n_features
        model.z = zeros(T, n_features)
        model.n = zeros(T, n_features)
        model.is_initialized = true
    end
    n_features == model.n_features || throw(DimensionMismatch("Feature dimension mismatch: got $n_features, expected $(model.n_features)"))

    Xt = SparseMatrixCSC(X')  # n_features × n_samples

    z = model.z
    n_acc = model.n
    lr = model.lr
    β  = model.lr_decay
    λ  = model.λ
    λ1 = λ * model.l1_ratio
    λ2 = λ * (one(T) - model.l1_ratio)
    do_dropout = model.dropout > zero(T)
    clip = model.grad_clip
    family = model.family

    rv = rowvals(Xt)
    nzv = nonzeros(Xt)

    for s in 1:n_samples
        # Compute prediction from z,n state
        pred = zero(T)
        col_range = nzrange(Xt, s)

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            if do_dropout && rand(rng) < model.dropout
                continue
            end
            wj = _ftrl_weight(z[j], n_acc[j], lr, β, λ1, λ2)
            pred += wj * xval
        end
        pred = link_function(family, pred)

        # Gradient: (pred - y) * x_j * weight
        err = (pred - T(y[s])) * weights[s]

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            gj = err * xval
            # Gradient clipping (matches R rsparse)
            gj = clamp(gj, -clip, clip)
            σj = (sqrt(n_acc[j] + gj^2) - sqrt(n_acc[j])) / lr
            z[j] += gj - σj * _ftrl_weight(z[j], n_acc[j], lr, β, λ1, λ2)
            n_acc[j] += gj^2
        end
    end

    if model.verbose
        pass_seconds = (time_ns() - iter_start) / 1e9
        @info @sprintf("[FTRL] update: %d samples, %d features | time=%s",
                       n_samples, n_features, elapsed_str(pass_seconds))
    end
    model
end

"""
    fit!(model::FTRL, X, y; kwargs...) -> model

Train the FTRL model for `model.max_iter` epochs over the full dataset.
"""
function fit!(model::FTRL{T}, X::SparseMatrixCSC, y::AbstractVector;
              weights::AbstractVector{T}=ones(T, length(y)),
              rng::AbstractRNG=Random.default_rng(),
              callbacks::AbstractVector{<:AbstractCallback}=AbstractCallback[]) where {T}
    train_start = time_ns()
    run_callbacks_train_begin(callbacks, model)
    try
    for i in 1:model.max_iter
        epoch_start = time_ns()
        update!(model, X, y; weights=weights, rng=rng)
        epoch_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = (time_ns() - train_start) / 1e9
        if model.verbose
            @info @sprintf("[FTRL] epoch %d/%d | epoch=%s | total=%s",
                           i, model.max_iter, elapsed_str(epoch_seconds), elapsed_str(total_seconds))
        end
        if !isempty(callbacks)
            loss = _ftrl_training_loss(model, X, y)
            info = CallbackInfo(i, Float64(loss), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end
    model
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

function _ftrl_training_loss(model::FTRL{T}, X::SparseMatrixCSC,
                             y::AbstractVector) where {T}
    preds = predict(model, X)
    if model.family isa LossFamilies.Binomial
        -sum(y .* log.(preds .+ T(1e-10)) .+
             (one(T) .- y) .* log.(one(T) .- preds .+ T(1e-10))) / length(y)
    else
        sum((preds .- y).^2) / length(y)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# predict / coef
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::FTRL, X) -> Vector

Generate predictions using the fitted model. Output depends on family:
- `LossFamilies.Binomial()` → probabilities in [0,1]
- `LossFamilies.Gaussian()` → real-valued predictions
- `LossFamilies.Poisson()`  → positive count predictions
"""
function predict(model::FTRL{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_initialized)
    n_samples = size(X, 1)
    size(X, 2) == model.n_features || throw(DimensionMismatch("Feature dimension mismatch: expected $(model.n_features), got $(size(X, 2))"))

    w = coef(model)
    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)
    family = model.family

    preds = Vector{T}(undef, n_samples)
    @inbounds for s in 1:n_samples
        v = zero(T)
        for idx in nzrange(Xt, s)
            j = rv[idx]
            v += w[j] * T(nzv[idx])
        end
        preds[s] = link_function(family, v)
    end
    preds
end

"""
    coef(model::FTRL) -> Vector

Return the model coefficient vector derived from the FTRL state.
"""
function coef(model::FTRL{T}) where {T}
    _require_fitted(model.is_initialized)
    w = Vector{T}(undef, model.n_features)
    lr = model.lr
    β  = model.lr_decay
    λ1 = model.λ * model.l1_ratio
    λ2 = model.λ * (one(T) - model.l1_ratio)
    @inbounds for j in 1:model.n_features
        w[j] = _ftrl_weight(model.z[j], model.n[j], lr, β, λ1, λ2)
    end
    w
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal: compute effective weight from FTRL state
# ──────────────────────────────────────────────────────────────────────────────

@inline function _ftrl_weight(zj::T, nj::T, lr::T, β::T, λ1::T, λ2::T) where {T}
    if abs(zj) <= λ1
        return zero(T)
    end
    sign_z = zj > zero(T) ? one(T) : -one(T)
    -((zj - sign_z * λ1) / ((β + sqrt(nj)) / lr + λ2))
end
