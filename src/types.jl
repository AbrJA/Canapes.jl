# ──────────────────────────────────────────────────────────────────────────────
# Abstract type hierarchy
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractSparseModel

Root abstract type for all Canapes models that operate on sparse matrices.
"""
abstract type AbstractSparseModel end

"""
    AbstractRecommender <: AbstractSparseModel

Abstract type for recommendation models that produce top-k item lists.
Models inheriting from this type must implement `recommend` and `score`.
"""
abstract type AbstractRecommender <: AbstractSparseModel end

"""
    AbstractMatrixFactorization <: AbstractRecommender

Abstract type for matrix factorization models.
Also provides `embeddings`, `similar_items`, and `similar_users`.
"""
abstract type AbstractMatrixFactorization <: AbstractRecommender end

"""
    AbstractItemSimilarity <: AbstractRecommender

Abstract type for item-similarity (neighborhood) models.
"""
abstract type AbstractItemSimilarity <: AbstractRecommender end

"""
    AbstractSparseRegression <: AbstractSparseModel

Abstract type for sparse regression models.
These implement `predict` (regression output), not `recommend`.
"""
abstract type AbstractSparseRegression <: AbstractSparseModel end

# ──────────────────────────────────────────────────────────────────────────────
# Solver types
# ──────────────────────────────────────────────────────────────────────────────

"""
    ALSSolver

Abstract type for ALS solver strategies. Concrete subtypes:
- [`CholeskySolver`](@ref) — direct Cholesky factorization (most stable)
- [`CGSolver`](@ref) — iterative CG solver (fastest at scale)
- [`NonNegativeSolver`](@ref) — non-negative least squares
"""
abstract type ALSSolver end

"""
    CholeskySolver <: ALSSolver

Direct Cholesky factorization solver. Maximum numerical stability.
"""
struct CholeskySolver <: ALSSolver end

"""
    CGSolver <: ALSSolver

Iterative Conjugate Gradient solver. Fastest for large-scale problems.
"""
struct CGSolver <: ALSSolver end

"""
    NonNegativeSolver <: ALSSolver

Non-Negative Least Squares solver. Produces non-negative factor matrices.
"""
struct NonNegativeSolver <: ALSSolver end

# ──────────────────────────────────────────────────────────────────────────────
# Feedback enum
# ──────────────────────────────────────────────────────────────────────────────

"""
    FeedbackType

Enum for feedback type: `Implicit` or `Explicit`.
"""
@enum FeedbackType begin
    Implicit
    Explicit
end

# ──────────────────────────────────────────────────────────────────────────────
# Links — GLM link functions
# Submodule: `Links.<TAB>` lists exactly the link families, keeps the root
# namespace free of generic names that collide with Distributions.jl.
# `LossFamily` below is a root alias for the abstract type.
# ──────────────────────────────────────────────────────────────────────────────

"""
    Links

GLM link functions for the regression models (`Links.Binomial`,
`Links.Gaussian`, `Links.Poisson`). Used as
`family=Links.Binomial()` in the FTRL/FM constructors; see also the
`LossFamily` root alias for the abstract type.
"""
module Links

    """
        Links.Family

    Abstract type for GLM family (link function). Concrete subtypes:
    - `Links.Binomial` — logistic (sigmoid) link
    - `Links.Gaussian` — identity link
    - `Links.Poisson` — exponential link
    """
    abstract type Family end

    """
        Links.Binomial

    Logistic link function: sigmoid(x). For binary classification.
    """
    struct Binomial <: Family end

    """
        Links.Gaussian

    Identity link function: x. For regression.
    """
    struct Gaussian <: Family end

    """
        Links.Poisson

    Exponential link function: exp(x). For count data.
    """
    struct Poisson <: Family end

end

const LossFamily = Links.Family

# ──────────────────────────────────────────────────────────────────────────────
# Sampling — negative sampling strategies (for BPR)
# Submodule: `Sampling.<TAB>` lists exactly the available strategies.
# `NegativeSampling` below is a root alias for the abstract type.
# ──────────────────────────────────────────────────────────────────────────────

"""
    Sampling

Negative sampling strategies for BPR (`Sampling.Uniform`,
`Sampling.Popular`, `Sampling.Dynamic`). Used as
`negative_sampling=Sampling.Uniform()` in the BPR constructor.
"""
module Sampling

    """
        Sampling.Strategy

    Abstract type for negative sampling strategies. Concrete subtypes:
    - `Sampling.Uniform` — uniform random sampling
    - `Sampling.Popular` — popularity-biased sampling (proportional to √frequency)
    - `Sampling.Dynamic` — Dynamic Negative Sampling (hardest negatives)
    """
    abstract type Strategy end

    """
        Sampling.Uniform

    Uniform random negative sampling. Simple and fast.
    """
    struct Uniform <: Strategy end

    """
        Sampling.Popular

    Popularity-biased negative sampling. Samples proportional to √(item frequency).
    """
    struct Popular <: Strategy end

    """
        Sampling.Dynamic

    Dynamic Negative Sampling (DNS). Selects the hardest negative from a candidate pool.
    """
    struct Dynamic <: Strategy end

end

const NegativeSampling = Sampling.Strategy

# ──────────────────────────────────────────────────────────────────────────────
# Generic API — every model must implement these
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::AbstractSparseModel, X; kwargs...)

Fit `model` in-place on sparse matrix `X`.
"""
function fit! end

"""
    transform(model::AbstractMatrixFactorization, X)

Return user embeddings for a fitted matrix factorization model.
"""
function transform end

"""
    recommend(model::AbstractRecommender, X; k=10)

Return top-k item indices per user, excluding already-interacted items.
Returns a `Matrix{Int}` of shape (n_users, k).
"""
function recommend end

"""
    score(model::AbstractRecommender, X)
    score(model::AbstractRecommender, user_indices, item_indices)

Return raw prediction scores. The full-matrix variant returns a dense `Matrix{T}`
(n_users × n_items). The pairwise variant returns a `Vector{T}` for specific
(user, item) pairs.
"""
function score end

"""
    predict(model::AbstractSparseRegression, X)

Generate regression predictions from a fitted model. Returns `Vector{T}`.
"""
function predict end

"""
    coef(model) -> Vector

Return the fitted coefficient vector of a regression model (FTRL, FM).
"""
function coef end

"""
    update!(model, X, y; kwargs...)

Run a single epoch of online/incremental learning. For streaming models
(FTRL, FM, EALS).
"""
function update! end

"""
    embeddings(model::AbstractMatrixFactorization)

Return the embedding matrix for a fitted model.
"""
function embeddings end

"""
    similar_items(model::AbstractMatrixFactorization, item_id; k=10)

Find the k most similar items to `item_id` based on embedding cosine similarity.
Returns `(ids::Vector{Int}, scores::Vector{T})`.
"""
function similar_items end

"""
    similar_users(model::AbstractMatrixFactorization, user_id; k=10)

Find the k most similar users to `user_id` based on embedding cosine similarity.
Returns `(ids::Vector{Int}, scores::Vector{T})`.
"""
function similar_users end


