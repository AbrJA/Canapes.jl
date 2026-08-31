module Gideon

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
include("algorithms/wrmf.jl")
include("algorithms/ials.jl")
include("algorithms/eals.jl")
include("algorithms/ftrl.jl")
include("algorithms/fm.jl")
include("algorithms/glove.jl")
include("algorithms/lmf.jl")
include("algorithms/bpr.jl")
include("algorithms/ease.jl")
include("algorithms/slim.jl")
include("algorithms/admmslim.jl")
include("algorithms/knn.jl")
include("algorithms/soft_impute.jl")

# ── Metrics & evaluation ──
include("metrics/ranking.jl")
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
    ALSSolver, CholeskySolver, CGSolver, NonNegativeSolver,
    FeedbackType, IMPLICIT, EXPLICIT,
    Family,
    NegativeSampling,

    # Models
    WMF,
    IALS,
    EALS,
    FTRL,
    FM,
    GloVe,
    LogisticMF,
    BPR,
    EASE,
    SLIM,
    ADMMSLIM,
    ItemKNN,
    SoftImpute,
    SoftSVD,
    PureSVD,
    AbstractSoftALS,

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
    interactions_to_sparse,
    sparse_to_interactions

# ── GPU stubs (implemented by ext/GideonCUDAExt.jl when CUDA is loaded) ──
function fit_gpu! end
function recommend_gpu end
function score_gpu end

end # module Gideon
