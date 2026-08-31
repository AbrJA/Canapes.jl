# ──────────────────────────────────────────────────────────────────────────────
# Factorization Machines (2nd-order) — SGD with AdaGrad
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Rendle (2010)
#   "Factorization Machines"
#
# Prediction:
#   ŷ(x) = w₀ + Σ_j wⱼ xⱼ + ½ Σ_{f=1}^{k} [ (Σ_j v_{j,f} xⱼ)² - Σ_j v²_{j,f} x²ⱼ ]
# ──────────────────────────────────────────────────────────────────────────────

"""
    FM{T} <: AbstractSparseRegression

Second-order Factorization Machine trained via SGD with AdaGrad.

Supports both classification (`Links.Binomial()`) and regression (`Links.Gaussian()`) via
the `family` parameter. Uses per-coordinate adaptive learning rates (AdaGrad).

# Constructor
```julia
FM(; rank=4, lr_w=0.2, lr_v=lr_w,
                       λ_w=0.0, λ_v=0.0, family=Links.Binomial(), intercept=true,
                       max_iter=10, tol=-1.0, verbose=true)
```

# Example
```julia
julia> using SparseArrays

julia> X = sprand(MersenneTwister(1), 200, 100, 0.05);

julia> y = rand(MersenneTwister(3), [0.0, 1.0], 200);

julia> model = FM(rank=4, max_iter=2, verbose=false);

julia> fit!(model, X, y; rng=MersenneTwister(2));

julia> size(predict(model, X))
(200,)
```
"""
mutable struct FM{T<:AbstractFloat} <: AbstractSparseRegression
    const rank::Int
    lr_w::T
    lr_v::T
    const λ_w::T
    const λ_v::T
    const family::LossFamily
    const intercept::Bool
    const max_iter::Int
    const tol::T
    const verbose::Bool
    n_features::Int
    w0::T
    w::Vector{T}
    V::Matrix{T}            # rank × n_features
    grad_w2::Vector{T}      # AdaGrad accumulators
    grad_v2::Matrix{T}
    is_initialized::Bool
end

function FM(;
    rank::Int = 4,
    lr_w::Float64 = 0.2,
    lr_v::Float64 = lr_w,
    λ_w::Float64 = 0.0,
    λ_v::Float64 = 0.0,
    family::LossFamily = Links.Binomial(),
    intercept::Bool = true,
    max_iter::Int = 10,
    tol::Float64 = -1.0,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    family isa Union{Links.Binomial, Links.Gaussian} || throw(ArgumentError("FM supports Links.Binomial() or Links.Gaussian() families"))
    FM{T}(
        rank, T(lr_w), T(lr_v), T(λ_w), T(λ_v), family, intercept,
        max_iter, T(tol), verbose,
        0, T(0), T[], Matrix{T}(undef,0,0),
        T[], Matrix{T}(undef,0,0), false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# update! — single SGD epoch
# ──────────────────────────────────────────────────────────────────────────────

"""
    update!(model::FM, X, y; weights, rng) -> model

Run a single SGD epoch over the data.
"""
function update!(model::FM{T}, X::SparseMatrixCSC{Tv,Ti},
                      y::AbstractVector;
                      weights::AbstractVector{T} = ones(T, length(y)),
                      rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    iter_start = time_ns()
    n_samples, n_features = size(X)
    n_samples == length(y) || throw(DimensionMismatch("X rows ($n_samples) ≠ length(y) ($(length(y)))"))
    _require_finite_input(X, "FM")
    _require_finite_vector(y, "FM")

    if !model.is_initialized
        model.n_features = n_features
        model.w0 = zero(T)
        model.w  = randn(rng, T, n_features) .* T(0.001)
        model.V  = randn(rng, T, model.rank, n_features) .* T(0.001)
        model.grad_w2 = ones(T, n_features)
        model.grad_v2 = ones(T, model.rank, n_features)
        model.is_initialized = true
    end
    n_features == model.n_features || throw(DimensionMismatch("Feature dimension mismatch: got $n_features, expected $(model.n_features)"))

    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)
    k = model.rank

    # Pre-allocate per-sample buffers
    sum_vx   = Vector{T}(undef, k)
    sum_v2x2 = Vector{T}(undef, k)

    for s in 1:n_samples
        col_range = nzrange(Xt, s)
        # ---- Forward pass ----
        pred = model.intercept ? model.w0 : zero(T)

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            pred += model.w[j] * xval
        end

        # Interaction term: ½ Σ_f [ (Σ_j v_{jf} xⱼ)² - Σ_j v²_{jf} x²ⱼ ]
        fill!(sum_vx, zero(T))
        fill!(sum_v2x2, zero(T))
        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            @inbounds for f in 1:k
                vfj = model.V[f, j]
                sum_vx[f]   += vfj * xval
                sum_v2x2[f] += vfj^2 * xval^2
            end
        end
        interaction = zero(T)
        @inbounds for f in 1:k
            interaction += sum_vx[f]^2 - sum_v2x2[f]
        end
        pred += interaction / 2

        # ---- Compute gradient multiplier ----
        if model.family isa Links.Binomial
            y_s = T(y[s]) > zero(T) ? one(T) : -one(T)
            grad_mult = -y_s * sigmoid(-y_s * pred) * weights[s]
        else  # Gaussian family
            grad_mult = (pred - T(y[s])) * weights[s]
        end

        # ---- Backward pass (AdaGrad updates) ----
        if model.intercept
            model.w0 -= model.lr_w * grad_mult
        end

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            gj = grad_mult * xval + model.λ_w * model.w[j]
            model.grad_w2[j] += gj^2
            model.w[j] -= model.lr_w * gj / sqrt(model.grad_w2[j])
        end

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            @inbounds for f in 1:k
                g_vfj = grad_mult * (sum_vx[f] * xval - model.V[f, j] * xval^2) + model.λ_v * model.V[f, j]
                model.grad_v2[f, j] += g_vfj^2
                model.V[f, j] -= model.lr_v * g_vfj / sqrt(model.grad_v2[f, j])
            end
        end
    end

    if model.verbose
        pass_seconds = (time_ns() - iter_start) / 1e9
        @info @sprintf("[FM] update: %d samples, %d features | time=%s",
                       n_samples, n_features, elapsed_str(pass_seconds))
    end
    model
end

"""
    fit!(model::FM, X, y; kwargs...) -> model

Train the FM for `model.max_iter` epochs.
"""
function fit!(model::FM{T}, X::SparseMatrixCSC, y::AbstractVector;
              weights::AbstractVector{T}=ones(T, length(y)),
              rng::AbstractRNG=Random.default_rng(),
              callbacks::AbstractVector{<:AbstractCallback}=AbstractCallback[]) where {T}
    train_start = time_ns()
    prev_loss = T(Inf)
    run_callbacks_train_begin(callbacks, model)
    try

    for i in 1:model.max_iter
        epoch_start = time_ns()
        update!(model, X, y; weights=weights, rng=rng)
        epoch_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = (time_ns() - train_start) / 1e9

        # Compute training loss for convergence check
        if model.tol > zero(T) || !isempty(callbacks)
            preds = predict(model, X)
            loss = if model.family isa Links.Binomial
                -sum(y .* log.(preds .+ T(1e-10)) .+ (one(T) .- y) .* log.(one(T) .- preds .+ T(1e-10))) / length(y)
            else
                sum((preds .- y).^2) / length(y)
            end
            if model.verbose
                log_iteration("FM", i, model.max_iter, Float64(loss), epoch_seconds, total_seconds)
            end
            if i > 1 && abs(prev_loss - loss) / (abs(prev_loss) + T(1e-12)) < model.tol
                model.verbose && @info "[FM] converged at epoch $i"
                break
            end
            prev_loss = loss
            if !isempty(callbacks)
                info = CallbackInfo(i, Float64(loss), total_seconds, model)
                run_callbacks(callbacks, info) && break
            end
        elseif model.verbose
            @info @sprintf("[FM] epoch %d/%d | epoch=%s | total=%s",
                           i, model.max_iter, elapsed_str(epoch_seconds), elapsed_str(total_seconds))
        end
    end
    model
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::FM, X) -> Vector

Generate predictions. Output depends on family:
- `Links.Binomial()` → probabilities in [0,1]
- `Links.Gaussian()` → real-valued predictions
"""
function predict(model::FM{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_initialized)
    n_samples = size(X, 1)
    size(X, 2) == model.n_features || throw(DimensionMismatch("Feature dimension mismatch: expected $(model.n_features), got $(size(X, 2))"))

    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)
    k = model.rank

    preds = Vector{T}(undef, n_samples)
    for s in 1:n_samples
        col_range = nzrange(Xt, s)
        pred = model.intercept ? model.w0 : zero(T)

        for idx in col_range
            j = rv[idx]
            pred += model.w[j] * T(nzv[idx])
        end

        interaction = zero(T)
        @inbounds for f in 1:k
            s_vx  = zero(T)
            s_v2x2 = zero(T)
            for idx in col_range
                j = rv[idx]
                xval = T(nzv[idx])
                vfj = model.V[f, j]
                s_vx   += vfj * xval
                s_v2x2 += vfj^2 * xval^2
            end
            interaction += s_vx^2 - s_v2x2
        end
        pred += interaction / 2

        if model.family isa Links.Binomial
            preds[s] = sigmoid(pred)
        else
            preds[s] = pred
        end
    end
    preds
end
