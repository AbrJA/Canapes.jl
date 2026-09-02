<div align="center">

# Canapes.jl

**High-performance statistical learning on sparse matrices in pure Julia.**

[![Build Status](https://github.com/AbrJA/Canapes.jl/workflows/CI/badge.svg)](https://github.com/AbrJA/Canapes.jl/actions)
[![codecov](https://codecov.io/gh/AbrJA/Canapes.jl/graph/badge.svg)](https://codecov.io/gh/AbrJA/Canapes.jl)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-blue?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Canapes.jl is a production-oriented Julia toolkit for sparse statistical learning and
recommender systems: matrix factorization, item-item models, low-rank completion, and
sparse regression — all on `SparseMatrixCSC` with a single unified API.

It is built for the constraints of real recommendation pipelines: **reproducible**
training (fixed seed + environment; GloVe and BPR are the documented Hogwild exceptions),
**memory-bounded** scoring paths
that never materialize a full dense score matrix, **optional GPU** acceleration, and
numerical correctness **validated against R (rsparse) and Python (implicit, sklearn,
scipy)** references.

## Highlights

| | |
|---|---|
| **One API for everything** | `fit!` / `recommend` / `score` / `predict` / `transform` across 15 algorithms |
| **Reproducible by design** | training is reproducible for a fixed seed and environment (no `@fastmath` — NaN/Inf-correct); GloVe and BPR are the documented Hogwild exceptions |
| **Reference-validated** | weights, predictions, and losses compared numerically against R and Python reference implementations |
| **Sparse-native, memory-safe** | no dense conversions, batched top-k scoring, fit-time `max_memory` guards, sparse fitted weights |
| **Concurrency-safe** | transactional `fit!`, read-only `recommend`/`score`, safe nesting |
| **GPU when you want it** | optional CUDA.jl extension for EASE, IALS, WMF and scoring |
| **Tables.jl native** | feed it DataFrames, CSV, Arrow — `(user, item, value)` triplets in, sparse matrix out |
| **Benchmarked** | a tracked harness measures `fit!`/`recommend` time and allocations at fixed scales across commits |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/AbrJA/Canapes.jl")
```

Requires Julia ≥ 1.10. Full API reference lives in [`docs/src/`](docs/src) (build with `julia --project=docs docs/make.jl`).

---

## Quick Start

End-to-end in one flow: tabular interactions → sparse matrix → train → recommend → evaluate.

```julia
using Canapes, DataFrames, SparseArrays, Random, Statistics

# 1. Interactions as a table (any Tables.jl source: DataFrames, CSV, Arrow, …)
df = DataFrame(user=[1,1,2,3,3,4], item=[2,5,3,1,4,2], rating=[1.0,1.0,1.0,1.0,1.0,1.0])
X = triplets_to_sparse(df; user_col=:user, item_col=:item, value_col=:rating)   # 4×5

# 2. Train (seen items are masked at recommend time)
model = WMF(rank=8, λ=0.1, α=40.0, max_iter=15, verbose=false)
fit!(model, X; rng=MersenneTwister(42))

# 3. Top-k recommendations per user (k clamps to n_items)
preds = recommend(model, X; k=10)   # 4×5 Matrix{Int} of item indices

# 4. Evaluate against a held-out split (mean_ap_at_k is scalar; ndcg_at_k is per-user)
X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))
fit!(model, X_train; rng=MersenneTwister(42))
println("MAP@10      = ", round(mean_ap_at_k(recommend(model, X_train; k=10), X_test), digits=4))
println("Mean NDCG@10 = ", round(mean(ndcg_at_k(recommend(model, X_train; k=10), X_test)), digits=4))
```

---

## Model Catalog

| Model | Type | Reference |
|-------|------|-----------|
| `WMF` | Implicit / Explicit ALS (Cholesky, CG, NNLS) | Hu, Koren & Volinsky (2008) |
| `IALS` | Implicit ALS with Gramian caching | Rendle et al. (2021) |
| `EALS` | Element-wise ALS with popularity weighting | He et al. (2016) |
| `BPR` | Bayesian Personalized Ranking (pairwise SGD) | Rendle et al. (2009) |
| `GloVe` | Co-occurrence embedding (Hogwild AdaGrad) | Pennington, Socher & Manning (2014) |
| `EASE` | Embarrassingly Shallow Autoencoders (closed form) | Steck (2019) |
| `SLIM` | Sparse Linear Methods (elastic net) | Ning & Karypis (2011) |
| `ADMMSLIM` | ADMM-based SLIM (joint solve, 10–100× faster) | Steck et al. (2020) |
| `ItemKNN` | Item-based K-Nearest Neighbors (cosine / Jaccard / asymmetric / BM25) | Deshpande & Karypis (2004) |
| `RandomWalk` | 3-step graph random walk with popularity penalty ("RP3β") | Paolino et al. (2017) |
| `FTRL` | Follow The Regularized Leader (online GLM) | McMahan et al. (2013) |
| `FM` | 2nd-order Factorization Machines (AdaGrad SGD) | Rendle (2010) |
| `SoftImpute` | Low-rank matrix completion (explicit ratings only) | Hastie et al. (2014) |
| `BaselineOnly` | Rating baseline μ + b_u + b_i (ALS) | Koren (2009) |
| `SlopeOne` | Rating predictor from item-pair deviations | Lemire & Maclachlan (2005) |
| `PearsonKNN` | User mean-centered Pearson neighborhood (ratings) | Surprise KNNWithMeans |
| `PMF` *(experimental)* | Probabilistic Matrix Factorization (SGD) | Mnih & Salakhutdinov (2007) |
| `SoftSVD` | Low-rank SVD (power-iteration style) | Hastie et al. (2014) |
| `PureSVD` | Truncated SVD (SoftSVD with λ = 0) | Cremonesi et al. (2010) |

> `LogisticMF` (Johnson 2014) has been demoted to the experimental namespace
> (`Canapes.Experimental.LogisticMF`) — it is reference-validated against
> `implicit` but ranks at the bottom of implicit top-N benchmarks and needs
> fragile tuning.

**Choosing a model** — the short version:

- **Implicit feedback (clicks, views, plays)**: `WMF` (fast, any scale), `IALS` (best accuracy/cost balance), `EALS` (popularity-weighted), `BPR` (pairwise ranking, Hogwild; negative sampling via `Sampling.Uniform()` / `Sampling.Popular()` / `Sampling.Dynamic()`). (Probabilistic LogisticMF lives in `Canapes.Experimental`.)
  Note: in the literature "iALS" usually means Hu et al. (2008) — this package's `WMF`. The `IALS` type here is the improved ALS of Rendle et al. (2021, "IALS++").
- **Item-item similarity**: `EASE` (state of the art, O(n_items²) memory), `SLIM` (sparse + interpretable), `ADMMSLIM` (same solution as SLIM, dense training), `ItemKNN` (lightweight baseline), `RandomWalk` (RP3β — 3-step graph walk with long-tail bias, O(nnz) memory).
- **Embeddings / related items**: `GloVe` on co-occurrences.
- **Explicit ratings (rating prediction)**: `WMF(feedback=Explicit)` (BiasedMF — μ + b_u + b_i + UᵀV via augmented ALS; best measured accuracy), then `BaselineOnly` (bias-only baseline), `SlopeOne` (pair deviations, Surprise-exact), `PearsonKNN` (user-centered neighborhoods), `SoftImpute`/`SoftSVD`/`PureSVD` (completion). Evaluate with `rmse`/`mae` — the explicit subsystem never uses `recommend`. Note: completion models rank at the bottom of implicit top-N benchmarks; use the implicit-feedback models there.
- **Sparse regression / CTR**: `FTRL` (online, elastic-net, streaming) and `FM` (second-order feature interactions).

---

## Examples by Model

### Collaborative filtering — WMF

```julia
using Canapes, SparseArrays, Random

X = sprand(MersenneTwister(42), 1000, 500, 0.02)   # 1 K users, 500 items, 2% density

# CG-ALS (default, fastest at scale); CholeskySolver for max stability; NonNegativeSolver for NNLS
model = WMF(rank=20, λ=0.1, α=1.0, max_iter=15)
fit!(model, X; rng=MersenneTwister(1))

size(model.user_factors)   # (20, 1000)  — rank × n_users
size(model.item_factors)   # (20, 500)   — rank × n_items

# Fold-in new users from their interaction history
U_new = transform(model, sprand(MersenneTwister(7), 50, 500, 0.03))   # (20, 50)
```

### Item-item models — EASE, SLIM, ADMMSLIM, ItemKNN

```julia
using Canapes, SparseArrays, Random

X = sprand(MersenneTwister(42), 1000, 500, 0.02)

# EASE: closed-form, strongest accuracy on implicit benchmarks
ease = EASE(λ=200.0, verbose=false)
fit!(ease, X)

# SLIM: sparse, interpretable weights; ADMMSLIM: same solution, 10-100x faster
slim    = SLIM(λ_l1=0.01, λ_l2=0.1, max_iter=50, verbose=false)
admm    = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=50, verbose=false)
fit!(slim, X);  fit!(admm, X)
nnz(admm.W)                      # fitted weights are stored sparse
score(admm, X)                   # SparseMatrixCSC, not a dense matrix

# ItemKNN: fast baseline — :cosine, :jaccard, :asym_cosine, :bm25
knn = ItemKNN(k=50, similarity=:cosine, verbose=false)
fit!(knn, X)
knn_bm25 = ItemKNN(k=200, similarity=:bm25, verbose=false)   # robust to popularity skew
fit!(knn_bm25, X)
```

### Co-occurrence embeddings — GloVe

```julia
using Canapes, SparseArrays, Random

C = sprand(MersenneTwister(1), 5000, 5000, 0.005)   # square, positive co-occurrences
C = C + C'

glove = GloVe(rank=100, lr=0.05, x_max=100.0, max_iter=20)
fit!(glove, C; rng=MersenneTwister(2))

E = embeddings(glove)             # 100 × 5000 — main + context average (standard convention)
```

### Graph random walks — RandomWalk (RP3β)

```julia
using Canapes, SparseArrays, Random

X = sprand(MersenneTwister(5), 2000, 1000, 0.01)

# 3-step walk, no training loop; β penalizes popular items → long-tail recommendations
rw = RandomWalk(α=1.0, β=0.6, k=200, verbose=false)
fit!(rw, X)
size(rw.W)                                    # sparse item×item walk matrix
```

### Logistic Matrix Factorization (experimental)

```julia
using Canapes, SparseArrays, Random

using Canapes.Experimental: LogisticMF

X = sprand(MersenneTwister(3), 800, 300, 0.03)
lmf = LogisticMF(rank=15, α=1.0, λ=0.1, lr=0.01, max_iter=20, n_negative=5)
fit!(lmf, X; rng=MersenneTwister(3))

# optimizer=:adagrad (default) = implicit's lmf.pyx; :rmsprop for small-lr runs
lmf_rp = LogisticMF(rank=15, λ=0.1, lr=0.05, max_iter=50, optimizer=:rmsprop)
fit!(lmf_rp, X; rng=MersenneTwister(3))
```

### Sparse regression — FTRL & FM

```julia
using Canapes, SparseArrays, Random

# FTRL: online logistic regression, elastic-net, streaming updates
rng = MersenneTwister(7)
X_train = sprand(rng, 10_000, 50_000, 0.001)
y_train = rand(rng, Bool, 10_000) .|> Float64

ftrl = FTRL(lr=0.1, lr_decay=0.5, λ=1e-4, l1_ratio=0.9)
update!(ftrl, X_train, y_train; rng)          # one pass; call again for more epochs
ŷ = predict(ftrl, X_train)                    # probabilities ∈ (0, 1)

# FM: second-order feature interactions
fm = FM(rank=8, lr_w=0.1, lr_v=0.05,
        λ_w=1e-5, λ_v=1e-5, family=Links.Binomial())
fit!(fm, X_train, y_train; rng=MersenneTwister(9))
```

The GLM loss families live in the `Links` submodule:
`Links.Binomial()` (logistic), `Links.Gaussian()` (squared
error) and `Links.Poisson()` (counts) — `Links.<TAB>` lists
exactly these three. Bring them in bare with
`using Canapes.Links: Binomial`.

### Explicit ratings — BiasedMF, BaselineOnly, SlopeOne, PearsonKNN

```julia
using Canapes, SparseArrays, Random

rng = MersenneTwister(1)
X = sprand(rng, 500, 300, 0.1)
nonzeros(X) .= 1 .+ 4 .* rand(MersenneTwister(2), nnz(X))   # ratings 1-5
X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(3))

# BiasedMF = WMF with feedback=Explicit: μ + b_u + b_i + x_uᵀ y_i
model = WMF(rank=16, λ=0.1, max_iter=15, feedback=Explicit, verbose=false)
fit!(model, X_train; rng=MersenneTwister(4))

# Rating predictors live under AbstractExplicitModel: score/predict + rmse
preds = predict(model, X_train)                # dense n_users × n_items ratings
println("Train RMSE = ", round(rmse(preds, X_train), digits=4))

baseline = BaselineOnly(λ=0.1, max_iter=20, verbose=false)   # μ + b_u + b_i
slope    = SlopeOne(verbose=false)                           # pair deviations
knn      = PearsonKNN(k=40, verbose=false)                   # user-mean-centered
for m in (baseline, slope, knn)
    fit!(m, X_train)
    println("RMSE = ", round(rmse(predict(m, X_train), X_test), digits=4))
end
```

### Matrix completion — SoftImpute / SoftSVD / PureSVD

```julia
using Canapes, SparseArrays, LinearAlgebra, Random

X_observed = sprand(MersenneTwister(11), 200, 150, 0.3)   # 30% observed entries

model = SoftImpute(rank=10, λ=0.5, max_iter=100)
fit!(model, X_observed; rng=MersenneTwister(11))

recon = model.U * Diagonal(model.d) * model.V'            # rank-10 completion
size(recon)                                               # (200, 150)

svd_model = SoftSVD(rank=5, max_iter=50)                  # λ=1.0 Ridge damping by default
svd_pure  = PureSVD(rank=5)                               # = SoftSVD(λ=0, final_svd=false)
fit!(svd_model, X_observed; rng=MersenneTwister(11))
```

---

## Evaluation — Ranking Metrics

Metrics take a `(n_users, K)` matrix of predicted item indices and a sparse relevance
matrix: `ap_at_k`, `ndcg_at_k`, `precision_at_k` and `recall_at_k` return a per-user
vector, while `mean_ap_at_k` returns the macro-averaged scalar.

```julia
using Canapes, SparseArrays, Random, Statistics

rng = MersenneTwister(13)
n_users, n_items, K = 500, 2000, 20
actual = sprand(rng, n_users, n_items, 0.02)                          # ground truth
preds  = hcat([randperm(rng, n_items)[1:K] for _ in 1:n_users]...)'

ap   = ap_at_k(preds, actual; k=K)
ndcg = ndcg_at_k(preds, actual; k=K)
prec = precision_at_k(preds, actual; k=K)
rec  = recall_at_k(preds, actual; k=K)

println("MAP@$K     = ", round(mean_ap_at_k(preds, actual; k=K), digits=4))
println("Mean NDCG@$K = ", round(mean(ndcg), digits=4))
```

### Cross-validation & hyperparameter search

```julia
using Canapes, SparseArrays

X = sprand(1000, 500, 0.02)

X_train, X_test = random_holdout(X; test_fraction=0.2)                 # random per-user holdout (no timestamps)

best, score, results = grid_search(
    p -> WMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => [16, 32, 64], :λ => [0.01, 0.1, 1.0]);
    k=10, metric=ndcg_at_k,
)

best, score, _ = random_search(
    p -> WMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => rng -> rand(rng, [16, 32, 64, 128]),
         :λ    => rng -> 10.0^(rand(rng) * 3 - 2));
    n_trials=20, k=10, metric=ndcg_at_k,
)
```

---

## Tables.jl Integration

Feed `triplets_to_sparse` any Tables.jl-compatible source — DataFrames, CSV, Arrow,
NamedTuples, vectors of named tuples — and get a `SparseMatrixCSC` back. Repeated
`(user, item)` pairs are accumulated by summing their values.

```julia
using Canapes

# Column table (NamedTuple of vectors, DataFrame, CSV.File, …)
data = (user=[1,1,2,3,3], item=[2,5,3,1,4], value=[1.0,2.0,1.0,3.0,1.0])
X = triplets_to_sparse(data)                     # defaults: user_col=:user, …

# Row table (Vector of NamedTuples, Tables.rowtable, …)
rows = [(user=1, item=3, value=1.0), (user=2, item=1, value=2.0)]
X = triplets_to_sparse(rows)

# Binary interactions (implicit 1.0) and custom element type
X = triplets_to_sparse(clicks; value_col=nothing, T=Float32)

# Back to triplets
triplets = sparse_to_triplets(X)                 # (user=…, item=…, value=…)
```

---

## Persistence

Models are saved atomically (temp file + rename, so a crash never leaves a partial file)
with a versioned header:

```julia
using Canapes, SparseArrays

model = EASE(λ=100.0, verbose=false)
fit!(model, sprand(MersenneTwister(1), 200, 100, 0.05))

save_model(model, "model.jls")
loaded = load_model("model.jls")
recommend(loaded, sprand(MersenneTwister(2), 10, 100, 0.1); k=5)
```

---

## GPU Acceleration

With [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed, a package extension
provides GPU-accelerated training and scoring — no code changes, just load CUDA:

```julia
using Canapes, CUDA

fit_gpu!(model::EASE, X)        # fully on GPU
fit_gpu!(model::IALS, X)        # Gramian on GPU, solve on CPU
fit_gpu!(model::WMF, X)
score_gpu(model, X)
recommend_gpu(model, X; k=10)
```

---

## Shared API

Recommender models implement a shared interface via defaults on the abstract types —
no per-model boilerplate:

| Function | Description |
|----------|-------------|
| `fit!(model, X)` | Train in-place on a sparse matrix (transactional: previous state intact on failure) |
| `update!(model, X, y)` | Online / incremental update (FTRL, FM, EALS) |
| `recommend(model, X; k)` | Top-k item indices per user, seen items masked — never builds the full score matrix |
| `score(model, X)` | Full user × item score matrix |
| `score(model, users, items)` | Scores for specific (user, item) pairs |
| `transform(model, X)` | Latent embeddings for new users (fold-in) |
| `similar_items(model, id; k)` / `similar_users(model, id; k)` | Cosine nearest neighbors |
| `coef(model)` | Learned weight vector (FTRL, FM) |
| `predict(model, X)` | Regression predictions (FTRL, FM) |

**Namespaced singletons:** the small generic-value types live in exported
submodules so the root namespace stays about verbs, models and metrics —
`Links.<TAB>` lists exactly the GLM link families and `Sampling.<TAB>` the
negative-sampling strategies; their abstract types remain reachable as
`LossFamily` and `NegativeSampling` in signatures.

**Concurrency guarantees:** separate models train in parallel safely; reads on a fitted
model are thread-safe; concurrent `fit!` on the *same* instance is unsupported; training
is reproducible for a given seed and environment (GloVe and BPR excepted — Hogwild by design).

---

## Performance Design

| Technique | Where used |
|-----------|-----------|
| Pre-allocated per-thread Gram / RHS / Cholesky buffers, hoisted to fit level | WMF, IALS, EALS ALS sweeps |
| Batched BLAS gram assembly (incremental rank-1 + `BLAS.syrk!`) | WMF-Cholesky |
| `BLAS.syr!` rank-1 Gram accumulation | WMF Cholesky solver |
| Fast-path manual SIMD dot for sparse users with < 32 nnz | WMF CG `_implicit_matvec!` |
| SIMD-vectorized reductions (`@simd`), no `@fastmath` — reproducible per environment, NaN/Inf-correct | All training loops |
| Memory-bounded batched GEMM top-k scoring | EASE |
| Unified top-k paths (`_predict_sparse_score_topk`, `_predict_batched_gemm_topk`) | EASE, SLIM, ItemKNN, ADMMSLIM |
| Sparse fitted weights (`SparseMatrixCSC`, soft-thresholded exact zeros) | SLIM, ADMMSLIM |
| Fit-time peak-memory estimate + `max_memory` guard before allocating | EASE, SLIM, ADMMSLIM |
| Adaptive scoring path (sparse vs batched GEMM by expected fill of `X·W`) | ADMMSLIM |
| `@inbounds @simd` vectorized element-wise / gradient loops | WMF, LogisticMF, GloVe, BPR, EALS |
| CSR dual storage for O(nnz_u) per-user row access | All algorithms, metrics |
| `Threads.@threads` outer loops with shared chunked-buffer helpers | WMF, IALS, EALS, BPR, GloVe |
| Gramian caching (avoids per-user recomputation) | IALS, EALS |
| Lock-free single-pass SGD (Hogwild, word-block chunks) | GloVe |
| Numerical stability (epsilon floors in AdaGrad) | GloVe, FM |
| PrecompileTools workloads | All algorithms (reduces TTFX) |
| Optional GPU offloading via CUDA.jl extension | EASE, IALS, WMF, predict |

---

## Correctness & Reference Validation

Numerical correctness is validated against independent reference implementations, not
just unit tests:

- **R (rsparse)** — parity on WMF, FTRL, GloVe (½-loss convention), SoftImpute/SVD, ranking
  metrics, and FM (correlation ≥ 0.95, gated with margin because rsparse's init uses
  `std::random_device`).
- **Python (implicit, sklearn, scipy)** — parity on ALS, BPR, IALS, EALS, LMF, EASE, SLIM,
  PureSVD, ItemKNN, and ADMMSLIM.
- One command runs both suites (R via the project-managed `uvr` environment, Python via a
  venv):

```bash
julia --project=. validation/run.jl --all
PYTHON=.venv/bin/python julia --project=. validation/run.jl --all   # venv python
```

The test suite itself includes 451 pure-Julia fixtures ported from `implicit`, 227
reference-style contract tests, and 15k randomized property tests.

---

## Benchmarking

A tracked performance harness measures `fit!` / `recommend` time and allocations at
three fixed scales (hundreds / thousands / millions of interactions) with
deterministic seeds:

```bash
julia --project=benchmark --threads=8 benchmark/run.jl   # appends JSONL records
julia --project=benchmark benchmark/compare.jl --strict  # diff runs across git SHAs
```

Records (git SHA, environment, config, metrics) accumulate in
`benchmark/logs/results.jsonl`.

---

## Testing

```bash
julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'                 # full suite
TEST_SUITE=fast julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'  # fast (skip Aqua/JET, docs, GPU)
```

The full suite runs **17,450 tests** in ~5 minutes; the fast suite (~2.5 min)
skips only the slow static-analysis, documentation-validation and GPU blocks
and still covers every algorithm, metric, property, fixture and contract test.
CI runs the full suite.

Covering:

- Unit correctness for all 15 algorithms (dimensions, NaN/Inf guards, convergence)
- Pure-Julia fixtures ported from `implicit` + reference-style recommender contracts
- 15k randomized property tests (dims, densities, duplicates, value extremes)
- Static analysis via [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) and [JET.jl](https://github.com/aviatesk/JET.jl)
- Infrastructure: atomic persistence, cross-validation, callbacks, Tables.jl, concurrency
- Executable doctests for every docstring example + all `docs/src` examples
- GPU tests when CUDA is available

---

## Dependencies

| Package | Role |
|---------|------|
| `SparseArrays` (stdlib) | Core sparse matrix type |
| `LinearAlgebra` (stdlib) | BLAS / LAPACK, SVD, Cholesky |
| `SparseMatricesCSR.jl` | CSR representation for row-oriented access |
| `Tables.jl` | Interaction tables (DataFrames, CSV, Arrow, …) |
| `PrecompileTools.jl` | Precompilation workloads for faster TTFX |

### Optional (Extensions)

| Package | Role |
|---------|------|
| `CUDA.jl` | GPU acceleration via package extension |

---

## Contributing

- **Issues and PRs** are welcome. Before opening a PR, run the full suite
  (`--threads=8`) — parallelism is exercised — plus `git diff --check`.
- **Performance changes** must be validated with `benchmark/run.jl` and, where relevant,
  `validation/run.jl` for numerical parity.
- Keep training kernels SIMD-vectorized but `@fastmath`-free (NaN/Inf-correct),
  with reductions reproducible per environment.
- Keep the public API naming consistent: full words over abbreviations,
  `*_at_k` for metrics, `mean_*` for scalar aggregates, CamelCase singletons
  grouped in exported submodules (`Links`, `Sampling`) with their abstracts
  aliased at the root, and `max_iter`/`tol`/`lr`/`λ`/`rank`/`T` as the shared
  hyperparameter vocabulary.
- See [AGENTS.md](AGENTS.md) for the threading/determinism conventions and known pitfalls.

---

## License

MIT — see [LICENSE](LICENSE).
