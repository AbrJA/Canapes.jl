# ──────────────────────────────────────────────────────────────────────────────
# Abstract type hierarchy
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractSparseModel

Root abstract type for all Gideon models that operate on sparse matrices.
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

Enum for feedback type: `IMPLICIT` or `EXPLICIT`.
"""
@enum FeedbackType begin
    IMPLICIT
    EXPLICIT
end

# ──────────────────────────────────────────────────────────────────────────────
# Family types (GLM link functions)
# ──────────────────────────────────────────────────────────────────────────────

"""
    Family

Abstract type for GLM family (link function). Concrete subtypes:
- [`Binomial`](@ref) — logistic (sigmoid) link
- [`Gaussian`](@ref) — identity link
- [`Poisson`](@ref) — exponential link
"""
abstract type Family end

"""
    Binomial <: Family

Logistic link function: sigmoid(x). For binary classification.
"""
struct Binomial <: Family end

"""
    Gaussian <: Family

Identity link function: x. For regression.
"""
struct Gaussian <: Family end

"""
    Poisson <: Family

Exponential link function: exp(x). For count data.
"""
struct Poisson <: Family end

# ──────────────────────────────────────────────────────────────────────────────
# Negative sampling types (for BPR)
# ──────────────────────────────────────────────────────────────────────────────

"""
    NegativeSampling

Abstract type for negative sampling strategies. Concrete subtypes:
- [`Uniform`](@ref) — uniform random sampling
- [`Popular`](@ref) — popularity-biased sampling (proportional to √frequency)
- [`Dynamic`](@ref) — Dynamic Negative Sampling (hardest negatives)
"""
abstract type NegativeSampling end

"""
    Uniform <: NegativeSampling

Uniform random negative sampling. Simple and fast.
"""
struct Uniform <: NegativeSampling end

"""
    Popular <: NegativeSampling

Popularity-biased negative sampling. Samples proportional to √(item frequency).
"""
struct Popular <: NegativeSampling end

"""
    Dynamic <: NegativeSampling

Dynamic Negative Sampling (DNS). Selects the hardest negative from a candidate pool.
"""
struct Dynamic <: NegativeSampling end

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


