module Canapes

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Random
using Printf
using Serialization
using Tables
using PrecompileTools

# ── Core types & API ──
include("types.jl")
include("utils.jl")
include("sparse_utils.jl")
include("progress.jl")
include("callbacks.jl")
include("serialization.jl")

# ── Algorithms ──
include("algorithms/weightedmf.jl")
include("algorithms/cachedals.jl")
include("algorithms/elementwiseals.jl")
include("algorithms/ftrl.jl")
include("algorithms/factorizationmachine.jl")
include("algorithms/globalvectors.jl")
include("algorithms/pairwiseranking.jl")
include("algorithms/shallowautoencoder.jl")
include("algorithms/sparselinearmodel.jl")
include("algorithms/sparselinearadmm.jl")
include("algorithms/knn.jl")
include("algorithms/graphrandomwalk.jl")
include("algorithms/soft_impute.jl")
include("algorithms/baselineonly.jl")
include("algorithms/slopeone.jl")
include("algorithms/pearsonknn.jl")

# ── Experimental algorithms (namespaced, not in root exports) ──
include("experimental.jl")

# ── Metrics & evaluation ──
include("metrics/ranking.jl")
include("metrics/error.jl")
include("crossval.jl")

# ── Tables.jl integration ──
include("tables.jl")

# ── Precompilation ──
include("precompile.jl")

# ── Public API ──
export
    # Types
    AbstractSparseModel,
    AbstractRecommender,
    AbstractMatrixFactorization,
    AbstractItemSimilarity,
    AbstractSparseRegression,
    AbstractExplicitModel,
    AbstractSoftALS,
    ALSSolver, CholeskySolver, CGSolver, NonNegativeSolver,
    FeedbackType, Implicit, Explicit,
    LossFamily, Links,
    NegativeSampling, Sampling,

    # Models
    WeightedMF,
    CachedALS,
    ElementwiseALS,
    FTRL,
    FactorizationMachine,
    GlobalVectors,
    PairwiseRanking,
    ShallowAutoencoder,
    SparseLinearModel,
    SparseLinearADMM,
    ItemKNN,
    GraphRandomWalk,
    SoftImpute,
    SoftSVD,
    PureSVD,
    BaselineOnly,
    SlopeOne,
    PearsonKNN,

    # Namespaced experimental models (Canapes.Experimental.LogisticMF)
    Experimental,

    # Generic API
    fit!,
    transform,
    recommend,
    score,
    predict,
    score_gpu,
    recommend_gpu,
    fit_gpu!,
    update!,
    coef,
    similar_items,
    similar_users,
    embeddings,

    # Metrics
    ap_at_k,
    mean_ap_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    mean_rmse,
    mean_mae,

    # Cross-validation & search
    random_holdout,
    cross_validate,
    grid_search,
    random_search,

    # Callbacks
    AbstractCallback,
    CallbackInfo,
    on_epoch_end,
    on_train_begin,
    on_train_end,
    EarlyStoppingCallback,
    LossHistoryCallback,
    CheckpointCallback,
    LearningRateCallback,

    # Serialization
    save_model,
    load_model,

    # Sparse utilities
    to_csr,

    # Tables.jl integration
    triplets_to_sparse,
    sparse_to_triplets

# ── GPU stubs (implemented by ext/CanapesCUDAExt.jl when CUDA is loaded) ──
function fit_gpu! end
function recommend_gpu end
function score_gpu end

end # module Canapes
