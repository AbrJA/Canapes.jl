# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation and hyperparameter search for recommendation models
# ──────────────────────────────────────────────────────────────────────────────

"""
    random_holdout(X::SparseMatrixCSC; test_fraction=0.2, rng=Random.default_rng())

Split a user-item matrix into train/test by randomly holding out a fraction
of each user's interactions for testing. This is a random holdout, not a
temporal split, because the matrix does not contain timestamps.

Returns `(X_train, X_test)` as sparse matrices.
"""
function random_holdout(X::SparseMatrixCSC{Tv,Ti};
                        test_fraction::Real=0.2,
                        rng::AbstractRNG=Random.default_rng()) where {Tv,Ti}
    0.0 < test_fraction < 1.0 || throw(ArgumentError("test_fraction must be in (0, 1), got $test_fraction"))

    n_users, n_items = size(X)
    train_rows = Int[]; train_cols = Int[]; train_vals = Tv[]
    test_rows  = Int[]; test_cols  = Int[]; test_vals  = Tv[]

    rv = rowvals(X)
    nz = nonzeros(X)

    # Build per-user item lists
    user_entries = [Tuple{Int,Tv}[] for _ in 1:n_users]
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            push!(user_entries[rv[idx]], (j, nz[idx]))
        end
    end

    for u in 1:n_users
        entries = user_entries[u]
        n_entries = length(entries)
        n_entries == 0 && continue

        n_test = max(1, round(Int, n_entries * test_fraction))
        if n_entries <= 1
            # Keep everything in train
            for (j, v) in entries
                push!(train_rows, u); push!(train_cols, j); push!(train_vals, v)
            end
            continue
        end

        # Random permutation for this user's items
        perm = randperm(rng, n_entries)
        for (k, p) in enumerate(perm)
            j, v = entries[p]
            if k <= n_test
                push!(test_rows, u); push!(test_cols, j); push!(test_vals, v)
            else
                push!(train_rows, u); push!(train_cols, j); push!(train_vals, v)
            end
        end
    end

    X_train = sparse(train_rows, train_cols, train_vals, n_users, n_items)
    X_test  = sparse(test_rows, test_cols, test_vals, n_users, n_items)
    (X_train, X_test)
end

"""
    crossval(model_fn, X; n_folds=5, k=10, metric=map_at_k, rng=default_rng())

K-fold cross-validation for recommendation models.

# Arguments
- `model_fn` — a zero-argument function that returns a fresh model instance
- `X` — full interaction matrix (users × items)
- `n_folds` — number of folds
- `k` — cutoff for ranking metrics
- `metric` — ranking metric function (e.g., `map_at_k`, `ndcg_at_k`)

# Returns
- `(mean_score, std_score, fold_scores)`

# Example
```julia
mean_map, std_map, scores = crossval(
    () -> WMF(rank=10, λ=0.1, α=40.0, max_iter=10, verbose=false),
    X; n_folds=5, k=10, metric=map_at_k
)
```
"""
function crossval(model_fn, X::SparseMatrixCSC;
                     n_folds::Int=5,
                     k::Int=10,
                     metric=map_at_k,
                     rng::AbstractRNG=Random.default_rng())
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2, got $n_folds"))

    fold_scores = Float64[]

    for fold in 1:n_folds
        X_train, X_test = random_holdout(X; test_fraction=1.0/n_folds,
                                         rng=MersenneTwister(rand(rng, UInt)))

        model = model_fn()
        fit!(model, X_train; rng=rng)
        preds = recommend(model, X_train; k=k)
        metric_val = _scalarize_metric(metric(preds, X_test; k=k))
        push!(fold_scores, metric_val)
    end

    mean_score = sum(fold_scores) / length(fold_scores)
    std_score = sqrt(sum((s - mean_score)^2 for s in fold_scores) / (n_folds - 1))

    (mean_score, std_score, fold_scores)
end

"""
    grid_search(model_fn, X, param_grid; k=10, metric=map_at_k,
                test_fraction=0.2, rng=default_rng(), verbose=true)

Grid search over hyperparameters with train/test split.

# Arguments
- `model_fn(params::NamedTuple)` — function that takes hyperparams and returns a model
- `X` — interaction matrix
- `param_grid` — `Dict{Symbol, Vector}` mapping param names to values

# Returns
- `(best_params::NamedTuple, best_score::Float64, results::Vector)`

# Example
```julia
best, score, results = grid_search(
    p -> WMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => [10, 20, 50], :λ => [0.01, 0.1, 1.0]);
    k=10
)
```
"""
function grid_search(model_fn, X::SparseMatrixCSC,
                     param_grid::Dict{Symbol,<:AbstractVector};
                     k::Int=10,
                     metric=map_at_k,
                     test_fraction::Float64=0.2,
                     rng::AbstractRNG=Random.default_rng(),
                     verbose::Bool=true)
    X_train, X_test = random_holdout(X; test_fraction=test_fraction, rng=rng)

    # Generate all combinations
    keys_vec = collect(keys(param_grid))
    vals_vec = [param_grid[k] for k in keys_vec]
    combos = vec(collect(Iterators.product(vals_vec...)))

    results = Vector{NamedTuple{(:params, :score), Tuple{NamedTuple, Float64}}}()
    best_score = -Inf
    best_params = NamedTuple()

    for combo in combos
        params = NamedTuple{Tuple(keys_vec)}(combo)

        model = model_fn(params)
        try
            fit!(model, X_train; rng=rng)
            preds = recommend(model, X_train; k=k)
            metric_val = _scalarize_metric(metric(preds, X_test; k=k))
            push!(results, (params=params, score=metric_val))

            if metric_val > best_score
                best_score = metric_val
                best_params = params
            end

            verbose && @info "[GridSearch] $(params) → $(round(metric_val, digits=6))"
        catch e
            verbose && @warn "[GridSearch] $(params) failed: $(e)"
            push!(results, (params=params, score=-Inf))
        end
    end

    (best_params, best_score, results)
end

"""
    random_search(model_fn, X, param_samplers; n_trials=20, k=10,
                  metric=map_at_k, test_fraction=0.2, rng=default_rng(), verbose=true)

Random search over hyperparameters.

# Arguments
- `model_fn(params::NamedTuple)` — function returning a model
- `param_samplers` — `Dict{Symbol, Function}` mapping param names to sampling functions
  (each function takes an RNG and returns a value)

# Example
```julia
best, score, _ = random_search(
    p -> WMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => rng -> rand(rng, [10,20,50,100]),
         :λ => rng -> 10.0^(rand(rng)*3 - 2));  # log-uniform [0.01, 10]
    n_trials=30
)
```
"""
function random_search(model_fn, X::SparseMatrixCSC,
                       param_samplers::Dict{Symbol,<:Function};
                       n_trials::Int=20,
                       k::Int=10,
                       metric=map_at_k,
                       test_fraction::Float64=0.2,
                       rng::AbstractRNG=Random.default_rng(),
                       verbose::Bool=true)
    X_train, X_test = random_holdout(X; test_fraction=test_fraction, rng=rng)

    keys_vec = collect(keys(param_samplers))
    results = Vector{NamedTuple{(:params, :score), Tuple{NamedTuple, Float64}}}()
    best_score = -Inf
    best_params = NamedTuple()

    for trial in 1:n_trials
        # Sample parameters
        vals = [param_samplers[k](rng) for k in keys_vec]
        params = NamedTuple{Tuple(keys_vec)}(Tuple(vals))

        model = model_fn(params)
        try
            fit!(model, X_train; rng=rng)
            preds = recommend(model, X_train; k=k)
            metric_val = _scalarize_metric(metric(preds, X_test; k=k))
            push!(results, (params=params, score=metric_val))

            if metric_val > best_score
                best_score = metric_val
                best_params = params
            end

            verbose && @info "[RandomSearch $trial/$n_trials] $(params) → $(round(metric_val, digits=6))"
        catch e
            verbose && @warn "[RandomSearch $trial/$n_trials] $(params) failed: $(e)"
            push!(results, (params=params, score=-Inf))
        end
    end

    (best_params, best_score, results)
end

# Metrics that return per-user vectors (e.g. ndcg_at_k) are reduced to their
# mean so cross-validation and search report a single scalar per trial.
@inline _scalarize_metric(x::Real) = x
@inline _scalarize_metric(v::AbstractVector{<:Real}) = isempty(v) ? NaN : sum(v) / length(v)
