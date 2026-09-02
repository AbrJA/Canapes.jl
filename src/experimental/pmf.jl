
# ──────────────────────────────────────────────────────────────────────────────
# PMF — Probabilistic Matrix Factorization (SGD, Gaussian prior)
# ──────────────────────────────────────────────────────────────────────────────


"""
    PMF{T} <: Canapes.AbstractExplicitModel

Probabilistic Matrix Factorization (Mnih & Salakhutdinov 2007, NIPS):
Gaussian-observation model `r_ui ~ N(x_uᵀ y_i, σ²)` with Gaussian priors on the
factors, solved by MAP SGD (L2-regularized) over the observed ratings.

Experimental status: no reference-parity target (neither Surprise nor rsparse
implements PMF), it is not an accuracy leader — BiasedMF (`WMF(feedback=Explicit)`)
dominates it on rating prediction — and it is sensitive to the learning rate.
Use the experimental namespace consciously.

# Constructor
```julia
PMF(; rank=10, λ=0.1, lr=0.01, max_iter=30, verbose=true)
```

# Example
```julia
julia> using SparseArrays, Random

julia> X = sprand(MersenneTwister(1), 100, 50, 0.2); nonzeros(X) .= 1 .+ 4 .* rand(MersenneTwister(2), nnz(X));

julia> m = PMF(rank=8, max_iter=5, verbose=false);

julia> fit!(m, X; rng=MersenneTwister(3));

julia> size(predict(m, X))
(100, 50)
```
"""
mutable struct PMF{T<:AbstractFloat} <: AbstractExplicitModel
    const rank::Int
    const λ::T
    const lr::T
    const max_iter::Int
    const verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function PMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    lr::Float64 = 0.01,
    max_iter::Int = 30,
    verbose::Bool = true,
    T::Type{<:AbstractFloat} = Float32,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    lr > 0.0 || throw(ArgumentError("lr must be positive, got $lr"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    PMF{T}(rank, T(λ), T(lr), max_iter, verbose,
           Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), false)
end

function fit!(model::PMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG=Random.default_rng(),
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    _require_nonempty_dimensions(X, "PMF")
    _require_finite_input(X, "PMF")
    k = model.rank
    λ, lr, T_ = model.λ, model.lr, T
    old_U, old_V = model.user_factors, model.item_factors
    model.is_fitted = false
    run_callbacks_train_begin(callbacks, model)
    try
    model.user_factors = randn(rng, T_, k, n_users) .* T_(0.1)
    model.item_factors = randn(rng, T_, k, n_items) .* T_(0.1)
    U = model.user_factors
    V = model.item_factors

    X_csr = to_csr(X)
    # per-entity update lists (item, rating) from the CSR rows
    lists = [Tuple{Int32,T}[] for _ in 1:n_users]
    @inbounds for u in 1:n_users
        rng_u = nzrange(X_csr, u)
        isempty(rng_u) && continue
        lst = lists[u]
        sizehint!(lst, length(rng_u))
        for idx in rng_u
            push!(lst, (Int32(X_csr.colval[idx]), T_(X_csr.nzval[idx])))
        end
    end

    for epoch in 1:model.max_iter
        # single-threaded SGD for bit-reproducibility (experimental model)
        loss = zero(T_)
        @inbounds for u in 1:n_users
            for (i, r) in lists[u]
                pred = zero(T_)
                for f in 1:k
                    pred += U[f, u] * V[f, i]
                end
                err = r - pred
                loss += err * err
                for f in 1:k
                    uf = U[f, u]
                    U[f, u] += lr * (err * V[f, i] - λ * uf)
                    V[f, i] += lr * (err * uf - λ * V[f, i])
                end
            end
        end
        model.verbose && log_iteration("PMF", epoch, model.max_iter,
                                       Float64(loss / max(nnz(X), 1)), NaN, NaN)
    end
    model.is_fitted = true
    model
    catch
        model.user_factors, model.item_factors = old_U, old_V
        model.is_fitted = false
        rethrow()
    finally
        run_callbacks_train_end(callbacks, model)
    end
end

function predict(model::PMF{T}, X::SparseMatrixCSC) where {T}
    _require_fitted(model.is_fitted)
    size(X, 1) == size(model.user_factors, 2) || throw(DimensionMismatch(
        "X has $(size(X, 1)) users but the fitted model has $(size(model.user_factors, 2))"))
    size(X, 2) == size(model.item_factors, 2) || throw(DimensionMismatch(
        "X has $(size(X, 2)) items but the fitted model has $(size(model.item_factors, 2))"))
    model.user_factors' * model.item_factors
end

score(model::PMF{T}, X::SparseMatrixCSC) where {T} = predict(model, X)

function score(model::PMF{T}, user_indices::AbstractVector{<:Integer},
               item_indices::AbstractVector{<:Integer}) where {T}
    _require_fitted(model.is_fitted)
    length(user_indices) == length(item_indices) ||
        throw(DimensionMismatch("user_indices and item_indices must have the same length"))
    _predict_pairwise_scores(model.user_factors, model.item_factors, user_indices, item_indices)
end
