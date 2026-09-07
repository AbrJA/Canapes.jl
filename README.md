<div align="center">

# Canapes.jl

**Sparse statistical learning and recommender systems in pure Julia.**

[![Build Status](https://github.com/AbrJA/Canapes.jl/workflows/CI/badge.svg)](https://github.com/AbrJA/Canapes.jl/actions)
[![codecov](https://codecov.io/gh/AbrJA/Canapes.jl/graph/badge.svg)](https://codecov.io/gh/AbrJA/Canapes.jl)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://AbrJA.github.io/Canapes.jl/)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-blue?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Canapes.jl is a pure-Julia library for statistical learning on sparse matrices:
matrix factorization, item-item similarity, low-rank completion, and sparse
regression — all behind one unified `fit!` / `recommend` / `score` / `predict`
API on `SparseMatrixCSC`.

- **Reproducible training** for a fixed seed and environment (GlobalVectors and PairwiseRanking are
  the documented Hogwild exceptions)
- **Memory-bounded scoring** — top-k paths never materialize a full dense score
  matrix
- **Reference-validated** — weights, predictions, and losses are compared
  numerically against R (`rsparse`) and Python (`implicit`, `scikit-learn`,
  `scikit-surprise`) implementations
- **Optional GPU** acceleration via a CUDA.jl extension, Tables.jl input,
  atomic model persistence, and a tracked benchmark harness

## Installation

```julia
using Pkg
Pkg.add("Canapes")   # once registered; before that: Pkg.add(url="https://github.com/AbrJA/Canapes.jl")
```

Requires Julia ≥ 1.10. The full API reference is at [docs](https://AbrJA.github.io/Canapes.jl/).

---

## Quick Start

```julia
using Canapes, SparseArrays, Random, Statistics

# 1. Interactions as a table (any Tables.jl source: NamedTuples, DataFrames, CSV, Arrow, …)
df = (user=[1,1,2,3,3,4], item=[2,5,3,1,4,2], rating=[1.0,1.0,1.0,1.0,1.0,1.0])
X = triplets_to_sparse(df; user_col=:user, item_col=:item, value_col=:rating)   # 4×5

# 2. Train (seen items are masked at recommend time)
model = WeightedMF(rank=8, λ=0.1, α=40.0, max_iter=15, verbose=false)
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

## Models

| Model | Type | Reference |
|-------|------|-----------|
| `WeightedMF` | Implicit/explicit ALS (Cholesky, CG, NNLS solvers) | Hu, Koren & Volinsky (2008) |
| `CachedALS` | Implicit ALS with Gramian caching | Rendle et al. (2021) |
| `ElementwiseALS` | Element-wise ALS with popularity weighting | He et al. (2016) |
| `PairwiseRanking` | Pairwise ranking via SGD | Rendle et al. (2009) |
| `GlobalVectors` | Co-occurrence embeddings | Pennington, Socher & Manning (2014) |
| `ShallowAutoencoder` | Closed-form item-item model | Steck (2019) |
| `SparseLinearModel` | Sparse linear methods (elastic net) | Ning & Karypis (2011) |
| `SparseLinearADMM` | ADMM-based SparseLinearModel | Steck et al. (2020) |
| `ItemKNN` | Item-based KNN (cosine, Jaccard, asymmetric, BM25) | Deshpande & Karypis (2004) |
| `GraphRandomWalk` | RP3β graph random walk | Paolino et al. (2017) |
| `FTRL` | Online GLM (elastic-net, streaming) | McMahan et al. (2013) |
| `FactorizationMachine` | Factorization Machines | Rendle (2010) |
| `SoftImpute` | Low-rank matrix completion | Hastie et al. (2014) |
| `SoftSVD` / `PureSVD` | Low-rank SVD (ALS style) | Hastie et al. (2014) / Cremonesi et al. (2010) |
| `BaselineOnly` | Rating baseline μ + b_u + b_i | Koren (2009) |
| `SlopeOne` | Rating predictor from pair deviations | Lemire & Maclachlan (2005) |
| `PearsonKNN` | User-centered Pearson neighborhood | Resnick et al. (1994) |

`ProbabilisticMF` and `LogisticMF` live in the `Canapes.Experimental` namespace; the latter
is reference-validated against `implicit` but ranks at the bottom of implicit
top-N benchmarks and needs fragile tuning.

Choosing a model, briefly:

- **Implicit feedback (clicks, views, plays)**: `WeightedMF` (fast, any scale), `CachedALS`
  (accuracy/cost balance), `ElementwiseALS` (popularity-weighted), `PairwiseRanking` (pairwise
  ranking). Note: "iALS" in the literature usually means Hu et al. (2008) —
  that is this package's `WeightedMF`; `CachedALS` here is Rendle et al. (2021).
- **Item-item similarity**: `ShallowAutoencoder` (accuracy), `SparseLinearModel` / `SparseLinearADMM` (sparse and
  interpretable weights), `ItemKNN` (lightweight baseline), `GraphRandomWalk`
  (long-tail bias).
- **Explicit ratings (rating prediction)**: `WeightedMF(feedback=Explicit)` (BiasedMF),
  plus `BaselineOnly`, `SlopeOne`, `PearsonKNN`, and the completion models;
  evaluated with `rmse` / `mae`.
- **Sparse regression / CTR**: `FTRL` (online) and `FactorizationMachine`.

---

## API at a glance

| Function | Description |
|----------|-------------|
| `fit!(model, X)` | Train in place on a sparse matrix (transactional: previous state intact on failure) |
| `update!(model, X, y)` | Online / incremental update (FTRL, FactorizationMachine, ElementwiseALS) |
| `recommend(model, X; k)` | Top-k item indices per user, seen items masked — never builds the full score matrix |
| `score(model, X)` | Full user × item score matrix |
| `score(model, users, items)` | Scores for specific (user, item) pairs |
| `transform(model, X)` | Latent embeddings for new users (fold-in) |
| `similar_items(model, id; k)` | Cosine nearest neighbors |
| `coef(model)` / `predict(model, X)` | Learned weights / regression predictions (FTRL, FactorizationMachine) |

Metrics (`ap_at_k`, `ndcg_at_k`, `precision_at_k`, `recall_at_k`, `mean_ap_at_k`,
`rmse`, `mae`), `grid_search` / `random_search`, and the `triplets_to_sparse` /
`sparse_to_triplets` Tables.jl round-trips are part of the same package — see
the docs for examples.

---

## Performance, Validation & Benchmarking

- **Thread-safe and reproducible**: shared chunked-buffer helpers for
  `Threads.@threads` loops, `@simd` reductions without `@fastmath`, per-chunk
  work buffers.
- **Reference parity** is enforced by `validation/run.jl` (R, Python, Surprise,
  MovieLens-1M), and a tracked harness in `benchmark/` records `fit!` /
  `recommend` time and allocations at three fixed scales across commits.

## Testing

```bash
julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'                 # full suite
TEST_SUITE=fast julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'  # fast (skips Aqua/JET, docs, GPU)
```

The suite covers unit correctness, randomized property tests, pure-Julia
fixtures, reference-style contracts, executed docstrings, README examples, and
Aqua/JET static analysis; GPU tests run when CUDA is available. CI runs the full
suite on Julia 1, LTS, and pre-release, on Linux, macOS, and Windows, and
uploads coverage to Codecov.

## GPU

With CUDA.jl installed, a package extension adds `fit_gpu!`, `score_gpu`, and
`recommend_gpu` for ShallowAutoencoder, CachedALS, and WeightedMF — load `CUDA` and the same API works.

## Contributing

Issues and PRs are welcome. Run the full suite (`--threads=8`) before opening a
PR, validate performance changes with `benchmark/run.jl`, and keep training
kernels SIMD-vectorized but `@fastmath`-free.

---

## Development with AI assistance

Substantial parts of this package — algorithm implementations, tests, and the
validation and benchmark harnesses — were written with the assistance of
LLM-based coding tools. Every generated line has been reviewed by the
maintainer before inclusion. Each algorithm is an independent Julia
implementation of the paper cited in its docstring; R (`rsparse`) and Python
(`implicit`, `scikit-learn`, `scikit-surprise`, `scipy`) are used only as
numerical references in `validation/`, and no source code is derived from
them. Tests run in CI with coverage collection, and documentation is built and
deployed from the same workflow.

## License

MIT — see [LICENSE](LICENSE).