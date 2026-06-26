# Algorithms

## Matrix Factorization

### WMF — Weighted Regularized Matrix Factorization

```@docs
WMF
```

#### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)

# Cholesky solver (default for small-medium datasets)
model = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CholeskySolver())
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=5)

# Conjugate gradient solver (better for large datasets)
model_cg = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=ConjugateGradient(), cg_steps=3)
fit!(model_cg, X; rng=MersenneTwister(42))

# Non-negative solver
model_nn = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=NonNegative())
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
using Gideon, SparseArrays, Random
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
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = EALS(rank=64, λ=0.01, w0=10.0, max_iter=20)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)

# Incremental update with new interactions
X_new = sprand(MersenneTwister(2), 1000, 500, 0.01)
update!(model, X_new; n_iter=3)
```

### BPR — Bayesian Personalized Ranking

```@docs
BPR
```

#### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = BPR(rank=32, λ_user=0.01, λ_pos=0.01, λ_neg=0.01, learning_rate=0.05, max_iter=50)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
```

### LogisticMF — Logistic Matrix Factorization

```@docs
LogisticMF
```

#### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 800, 300, 0.03)
model = LogisticMF(rank=15, α=1.0, λ=0.1, learning_rate=0.01, max_iter=20, n_negative=5)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
scores = score(model, X)
```

### GloVe — Global Vectors

```@docs
GloVe
```

#### Example

```julia
using Gideon, SparseArrays, Random

# GloVe expects a symmetric co-occurrence matrix
X = sprand(MersenneTwister(1), 100, 100, 0.1)
X = X + X'
nonzeros(X) .= abs.(nonzeros(X))  # ensure positive counts

model = GloVe(rank=50, x_max=100.0, learning_rate=0.05, max_iter=25)
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
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 200, 150, 0.3)

# SoftImpute: full imputation correction (better for missing data recovery)
model = SoftImpute(rank=10, λ=0.5, max_iter=100)
fit!(model, X; rng=MersenneTwister(1))

# SoftSVD: power-iteration style (faster per iteration, good for implicit)
model_svd = SoftSVD(rank=10, λ=0.5, max_iter=100)
fit!(model_svd, X; rng=MersenneTwister(1))

# PureSVD: truncated SVD (no regularization, just rank reduction)
model_pure = PureSVD(rank=10, max_iter=100)
fit!(model_pure, X; rng=MersenneTwister(1))

# Access factors: model.U * Diagonal(model.d) * model.V'
preds = recommend(model, X; k=10)
```

## Item Similarity

### EASE — Embarrassingly Shallow Autoencoders

```@docs
EASE
```

#### Example

```julia
using Gideon, SparseArrays, Random
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
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 100, 0.05)

# λ_1 controls L1 sparsity, λ_2 controls L2 shrinkage
model = SLIM(λ_1=0.01, λ_2=0.1, max_iter=100, nonneg=true)
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
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 100, 0.05)

# Same objective as SLIM but 10-100× faster via ADMM
# ρ controls ADMM convergence speed
model = ADMMSLIM(λ_1=0.01, λ_2=100.0, ρ=1.0, max_iter=50, nonneg=true)
fit!(model, X)
preds = recommend(model, X; k=10)
```

### ItemKNN — Item-based K-Nearest Neighbors

```@docs
ItemKNN
```

#### Example

```julia
using Gideon, SparseArrays, Random
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
using Gideon, SparseArrays, Random

# Binary classification
X = sprand(MersenneTwister(1), 1000, 50, 0.1)
y = rand(MersenneTwister(2), [0.0, 1.0], 1000)
model = FTRL(learning_rate=0.1, λ=0.01, family=Binomial())
fit!(model, X, y)
p = predict(model, X)  # probabilities in [0, 1]

# Online/streaming: single-epoch update
update!(model, X, y)

# Gaussian regression
model_reg = FTRL(learning_rate=0.1, family=Gaussian())

# Poisson regression (count data)
model_pois = FTRL(learning_rate=0.1, family=Poisson())

# Access coefficients
w = coef(model)
```

### FM — Factorization Machines

```@docs
FM
```

#### Example

```julia
using Gideon, SparseArrays, Random

# XOR problem with second-order interactions
X = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
y = [0.0, 1.0, 1.0, 0.0]
model = FM(rank=4, family=Binomial(), max_iter=100, learning_rate_w=0.2)
fit!(model, X, y; rng=MersenneTwister(42))
predict(model, X)

# Gaussian regression with feature interactions
X_reg = sprand(MersenneTwister(1), 500, 20, 0.3)
y_reg = randn(MersenneTwister(2), 500)
model_reg = FM(rank=8, family=Gaussian(), max_iter=50)
fit!(model_reg, X_reg, y_reg; rng=MersenneTwister(42))
```
