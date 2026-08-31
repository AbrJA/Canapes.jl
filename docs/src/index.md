# Gideon.jl

A high-performance Julia package for sparse matrix factorization, collaborative filtering, and recommendation systems. Julia port and enhancement of R's [rsparse](https://github.com/rexyai/rsparse).

## Features

### Matrix Factorization
- **WMF** — Weighted Regularized Matrix Factorization (Cholesky, CG & NNLS solvers)
- **IALS** — Implicit ALS with Gramian caching (Rendle et al. 2021)
- **EALS** — Element-wise ALS with popularity-based weighting (He et al. 2016)
- **BPR** — Bayesian Personalized Ranking (pairwise learning)
- **LogisticMF** — Logistic Matrix Factorization with negative sampling
- **GloVe** — Global Vectors for word/item embeddings
- **SoftImpute / SoftSVD / PureSVD** — Nuclear-norm regularized matrix completion

### Item Similarity
- **EASE** — Embarrassingly Shallow Autoencoders (closed-form)
- **SLIM** — Sparse Linear Methods (elastic-net, coordinate descent)
- **ADMMSLIM** — ADMM-based SLIM (10-100× faster than SLIM)
- **ItemKNN** — Item-based K-Nearest Neighbors (cosine / Jaccard)

### Regression
- **FTRL** — Follow The Regularized Leader (Binomial, Gaussian, Poisson)
- **FM** — Factorization Machines (second-order feature interactions)

### Infrastructure
- **Ranking Metrics** — MAP@k, NDCG@k, Precision@k, Recall@k
- **Cross-validation** — temporal split, k-fold, grid search, random search
- **Callbacks** — early stopping, loss history, checkpointing, learning rate schedules
- **GPU acceleration** — CUDA.jl extension for EASE, IALS, WMF
- **Tables.jl integration** — accept interaction data as (user, item, value) triplets
- **Serialization** — versioned save/load for all models

## Quick Start

```julia
using Gideon, SparseArrays, Random

# Create a sparse user-item interaction matrix (1000 users, 500 items)
X = sprand(MersenneTwister(42), 1000, 500, 0.02)

# Split into train/test
X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))

# Fit a model
model = WMF(rank=10, λ=0.1, α=40.0, max_iter=15, verbose=false)
fit!(model, X_train)

# Get top-10 recommendations (seen items automatically masked)
recommendations = recommend(model, X_train; k=10)

# Evaluate
map_score  = mean_ap_at_k(recommendations, X_test; k=10)
ndcg_score = ndcg_at_k(recommendations, X_test; k=10)
println("MAP@10: $(round(map_score, digits=4))")
println("NDCG@10: $(round(ndcg_score, digits=4))")

# Full score matrix
scores = score(model, X_train)

# Scores for specific (user, item) pairs
pair_scores = score(model, [1, 2, 3], [10, 20, 30])

# Find similar items/users by cosine similarity
ids, sims = similar_items(model, 42; k=5)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/AbrJA/Gideon.jl")
```

## API Design

Gideon separates **recommender models** from **regression models** with domain-appropriate verbs:

| Model type | Top-k predictions | Raw scores | Regression |
|---|---|---|---|
| Recommenders (WMF, IALS, EASE, ...) | `recommend(model, X; k)` | `score(model, X)` | — |
| Regression (FTRL, FM) | — | — | `predict(model, X)` |

All models share `fit!(model, X)` for training. Matrix factorization models additionally support:

- `transform(model, X)` — embed new users into the latent space
- `similar_items(model, id; k)` / `similar_users(model, id; k)` — cosine-based neighbors

The type hierarchy uses Julia's dispatch to provide default implementations:

```
AbstractSparseModel
├── AbstractRecommender
│   ├── AbstractMatrixFactorization   # WMF, IALS, EALS, LogisticMF, BPR, GloVe
│   │   └── AbstractSoftALS           # SoftImpute, SoftSVD, PureSVD
│   └── AbstractItemSimilarity        # EASE, SLIM, ADMMSLIM, ItemKNN
└── AbstractSparseRegression          # FTRL, FM
```

New models inheriting from `AbstractMatrixFactorization` automatically get `recommend`,
`score`, `similar_items`, and `similar_users` with no boilerplate required.

## Cross-Validation & Hyperparameter Search

```julia
using Gideon, SparseArrays, Random

X = sprand(MersenneTwister(42), 1000, 500, 0.02)

# 5-fold cross-validation
mean_map, std_map, scores = cross_validate(
    () -> WMF(rank=10, λ=0.1, α=40.0, max_iter=10, verbose=false),
    X; n_folds=5, k=10, metric=mean_ap_at_k
)

# Grid search
best_params, best_score, results = grid_search(
    p -> EASE(λ=p.λ, verbose=false),
    X,
    Dict(:λ => [10.0, 100.0, 500.0, 1000.0]);
    k=10
)

# Random search with log-uniform sampling
best_params, best_score, _ = random_search(
    p -> WMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => rng -> rand(rng, [10, 20, 50, 100]),
         :λ    => rng -> 10.0^(rand(rng) * 3 - 2));  # log-uniform [0.01, 10]
    n_trials=30
)
```

## Callbacks

```julia
using Gideon, SparseArrays, Random

X = sprand(MersenneTwister(42), 1000, 500, 0.02)

# Early stopping: stop if loss doesn't improve for 3 iterations
early_stop = EarlyStoppingCallback(patience=3)

# Track loss history
loss_history = LossHistoryCallback()

# Checkpoint every 5 iterations
checkpoint = CheckpointCallback(path="checkpoints/wmf", every=5)

model = WMF(rank=10, λ=0.1, α=40.0, max_iter=50)
fit!(model, X; callbacks=[early_stop, loss_history, checkpoint])

# Inspect loss curve
loss_history.losses  # Vector of per-iteration losses
```

## Concurrency

- **Separate models, in parallel** — fitting independent models concurrently
  (e.g. inside a `Threads.@threads` loop) is supported; training loops are
  dynamic and safe to nest.
- **Reads on a fitted model** — `recommend` / `score` are read-only and safe
  to call concurrently from multiple threads.
- **Mutating one model** — calling `fit!` on the same model instance from
  multiple threads is unsupported unless you synchronize access externally.
- **Transactional `fit!`** — a failed `fit!` or refit leaves the previous
  fitted state intact.
- **Reproducibility** — fits are deterministic for a given `rng` seed.
  Training kernels use `muladd` reductions (strict scalar order), so results
  match across builds and platforms; `GloVe` is additionally bit-identical
  across thread counts. `BPR` is the exception: Hogwild! lock-free SGD is
  intentionally racy.

## Tables.jl Integration

```julia
using Gideon

# From a NamedTuple of vectors (column table)
data = (user=[1, 1, 2, 2, 3], item=[1, 3, 2, 4, 1], value=[1.0, 1.0, 1.0, 1.0, 1.0])
X = triplets_to_sparse(data; user_col=:user, item_col=:item, value_col=:value)

# Binary interactions (no value column)
clicks = (user_id=[1, 1, 2, 3], item_id=[10, 20, 10, 30])
X = triplets_to_sparse(clicks; user_col=:user_id, item_col=:item_id, value_col=nothing)

# Convert back to triplets
triplets = sparse_to_triplets(X)
```

## Serialization

```julia
using Gideon

# Save any fitted model
save_model(model, "my_model.jls")

# Load it back
model = load_model("my_model.jls")
```
