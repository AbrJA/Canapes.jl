# Algorithms

## Matrix Factorization

### WMF — Weighted Regularized Matrix Factorization

```@docs
WMF
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)

# Cholesky solver (default for small-medium datasets)
model = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CholeskySolver())
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=5)

# Conjugate gradient solver (better for large datasets)
model_cg = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CGSolver(), cg_steps=3)
fit!(model_cg, X; rng=MersenneTwister(42))

# Non-negative solver
model_nn = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=NonNegativeSolver())
fit!(model_nn, X; rng=MersenneTwister(42))

# Score matrix and similarity search
scores = score(model, X)
ids, sims = similar_items(model, 42; k=5)
```

### IALS — Implicit ALS with Gramian Caching

```@docs
IALS
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = IALS(rank=32, λ=0.01, α=1.0, max_iter=15)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
scores = score(model, X)
```

### EALS — Element-wise ALS

```@docs
EALS
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = EALS(rank=64, λ=0.01, unobserved_weight=10.0, max_iter=20)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)

# Incremental update with new interactions
X_new = sprand(MersenneTwister(2), 1000, 500, 0.01)
update!(model, X_new; n_iters=3)
```

### BPR — Bayesian Personalized Ranking

```@docs
BPR
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = BPR(rank=32, λ_user=0.01, λ_pos=0.01, λ_neg=0.01, lr=0.05, max_iter=50)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
```

### LogisticMF — Logistic Matrix Factorization *(experimental)*

Demoted from the core catalog: reference-validated against `implicit`, but it
ranks at the bottom of implicit Top-N benchmarks and requires fragile tuning.
Access it via the Experimental namespace — it is not a root export.

```@docs
Canapes.Experimental.LogisticMF
```

#### Example

```julia
using Canapes, SparseArrays, Random
using Canapes.Experimental: LogisticMF
X = sprand(MersenneTwister(1), 800, 300, 0.03)
model = LogisticMF(rank=15, α=1.0, λ=0.1, lr=0.01, max_iter=20, n_negative=5)
# optimizer=:adagrad (default) reproduces implicit's lmf.pyx exactly;
# optimizer=:rmsprop uses an EMA squared-gradient accumulator — recommended
# when training with small learning rates for longer runs.
model_rp = LogisticMF(rank=15, α=1.0, λ=0.1, lr=0.05, max_iter=50, optimizer=:rmsprop)
fit!(model, X; rng=MersenneTwister(42))
fit!(model_rp, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
scores = score(model, X)
```

### Explicit rating prediction — BiasedMF, BaselineOnly, SlopeOne, PearsonKNN

```@docs
BaselineOnly
SlopeOne
PearsonKNN
```

The biased matrix factorization lives in the dual model
[`WMF`](@ref) with `feedback=Explicit` (BiasedMF: `μ + b_u + b_i + x_uᵀ y_i`,
learned via augmented ALS), loss `WMF`'s `predict` returns the dense fitted
ratings. Experimental: [`PMF`](@ref) (`Canapes.Experimental.PMF`) is a MAP-SGD
Probabilistic Matrix Factorization with no reference-parity target.

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.1)
nonzeros(X) .= 1.0 .+ 4.0 .* rand(MersenneTwister(2), nnz(X))   # ratings 1-5
X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(3))

model = WMF(rank=16, λ=0.1, max_iter=15, feedback=Explicit, verbose=false)
fit!(model, X_train; rng=MersenneTwister(4))
preds = predict(model, X_train)
rmse(preds, X_train)         # training error; hold out for honest evaluation

for m in (BaselineOnly(verbose=false), SlopeOne(verbose=false),
          PearsonKNN(k=40, verbose=false))
    fit!(m, X_train)
    rmse(predict(m, X_train), X_test)
end
```

### RandomWalk — RP3β Graph Random Walk

```@docs
RandomWalk
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 2000, 1000, 0.01)

# 3-step bipartite random walk with popularity penalization (RP3β).
# No training loop; memory O(nnz); β > 0 boosts long-tail items.
model = RandomWalk(α=1.0, β=0.6, k=200, verbose=false)
fit!(model, X)
preds = recommend(model, X; k=10)
W = model.W   # sparse item×item walk matrix
```

### GloVe — Global Vectors

```@docs
GloVe
```

#### Example

```julia
using Canapes, SparseArrays, Random

# GloVe expects a symmetric co-occurrence matrix
X = sprand(MersenneTwister(1), 100, 100, 0.1)
X = X + X'
nonzeros(X) .= abs.(nonzeros(X))  # ensure positive counts

model = GloVe(rank=50, x_max=100.0, lr=0.05, max_iter=25)
fit!(model, X; rng=MersenneTwister(42))
E = embeddings(model)  # rank × n_words
```

### SoftImpute / SoftSVD / PureSVD — Low-rank Matrix Completion

```@docs
SoftImpute
SoftSVD
PureSVD
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 200, 150, 0.3)
nonzeros(X) .= 1.0 .+ 4.0 .* rand(MersenneTwister(2), nnz(X))   # ratings 1-5
X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(3))

# SoftImpute: full imputation correction (explicit-ratings / completion only;
# ranks below the popularity baseline on binarized implicit data)
model = SoftImpute(rank=10, λ=0.5, max_iter=100)
fit!(model, X_train; rng=MersenneTwister(1))

# The explicit contract: score/predict give the dense predicted-ratings
# matrix; evaluate with rmse/mae — completion models have no recommend().
preds = predict(model, X_train)          # n_users × n_items predicted ratings
println("RMSE = ", round(rmse(preds, X_test), digits=3))
```

## Item Similarity

### EASE — Embarrassingly Shallow Autoencoders

```@docs
EASE
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 200, 0.05)

# Closed-form solution — no iterations, just λ controls regularization
model = EASE(λ=500.0)
fit!(model, X)
preds = recommend(model, X; k=10)
scores = score(model, X)  # dense Matrix
```

### SLIM — Sparse Linear Methods

```@docs
SLIM
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 100, 0.05)

# λ_l1 controls L1 sparsity, λ_l2 controls L2 shrinkage
model = SLIM(λ_l1=0.01, λ_l2=0.1, max_iter=100, nonnegative=true)
fit!(model, X)
preds = recommend(model, X; k=10)
scores = score(model, X)  # sparse SparseMatrixCSC
```

### ADMMSLIM — ADMM-based SLIM

```@docs
ADMMSLIM
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 100, 0.05)

# Same objective as SLIM but 10-100× faster via ADMM
# ρ controls ADMM convergence speed and adapts automatically (primal/dual
# residual rebalancing, Boyd et al. 2011); convergence requires the primal
# residual AND the solution drift below tol. The direct dense solve is used
# deliberately: the Gram matrix of implicit data is ~99.9% dense, where
# per-column CG/PCG is 5-10× more expensive than the Cholesky substitution.
# Training memory is O(n_items²) (dense joint solve); prefer SLIM for very
# large item counts. The fitted W is stored sparse: score returns SparseMatrixCSC.
model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, ρ=1.0, max_iter=50, nonnegative=true)
fit!(model, X)
preds = recommend(model, X; k=10)
```

### ItemKNN — Item-based K-Nearest Neighbors

```@docs
ItemKNN
```

#### Example

```julia
using Canapes, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 200, 0.05)

# Cosine similarity with top-20 neighbors
model = ItemKNN(k=20, similarity=:cosine, normalize=true)
fit!(model, X)
preds = recommend(model, X; k=10)

# Jaccard similarity with shrinkage for rare items
model_jac = ItemKNN(k=20, similarity=:jaccard, shrinkage=10.0)
fit!(model_jac, X)
```

## Regression

### FTRL — Follow The Regularized Leader

```@docs
FTRL
```

#### Example

```julia
using Canapes, SparseArrays, Random

# Binary classification
X = sprand(MersenneTwister(1), 1000, 50, 0.1)
y = rand(MersenneTwister(2), [0.0, 1.0], 1000)
model = FTRL(lr=0.1, λ=0.01, family=Links.Binomial())
fit!(model, X, y)
p = predict(model, X)  # probabilities in [0, 1]

# Online/streaming: single-epoch update
update!(model, X, y)

# Gaussian regression
model_reg = FTRL(lr=0.1, family=Links.Gaussian())

# Poisson regression (count data)
model_pois = FTRL(lr=0.1, family=Links.Poisson())

# Access coefficients
w = coef(model)
```

### FM — Factorization Machines

```@docs
FM
```

#### Example

```julia
using Canapes, SparseArrays, Random

# XOR problem with second-order interactions
X = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
y = [0.0, 1.0, 1.0, 0.0]
model = FM(rank=4, family=Links.Binomial(), max_iter=100, lr_w=0.2)
fit!(model, X, y; rng=MersenneTwister(42))
predict(model, X)

# Gaussian regression with feature interactions
X_reg = sprand(MersenneTwister(1), 500, 20, 0.3)
y_reg = randn(MersenneTwister(2), 500)
model_reg = FM(rank=8, family=Links.Gaussian(), max_iter=50)
fit!(model_reg, X_reg, y_reg; rng=MersenneTwister(42))
```
