# validation/validate_py.jl
# Optional validation against Python `implicit` / scipy / sklearn references.
# Run via: julia --project=. validation/run.jl --python
# (or directly: julia --project=. validation/validate_py.jl)

using Gideon, SparseArrays, LinearAlgebra, Random
using Test
include(joinpath(@__DIR__, "common.jl"))

# ── Centralized thresholds (env-overridable, defaults unchanged) ─────────────

soft_strict = get(ENV, "GIDEON_PY_SOFT_STRICT", "0") == "1"

const SCORE_CASES = [
    (name="WMF-Cholesky vs ALS", model=() -> WMF(rank=16, λ=0.1, α=40.0, max_iter=20,
                                                   solver=CholeskySolver(), feedback=IMPLICIT, verbose=false),
     scores_path="py_als_scores.csv", required=true, metrics_path=nothing,
     min_cor=_env_float("GIDEON_PY_ALS_MIN_COR", 0.45),
     min_overlap=_env_float("GIDEON_PY_ALS_MIN_OVERLAP", 0.20),
     max_ndcg=nothing, max_recall=nothing),
    (name="BPR", model=() -> BPR(rank=16, max_iter=40, verbose=false),
     scores_path="py_bpr_scores.csv", required=true, metrics_path="py_bpr_metrics.json",
     min_cor=_env_float("GIDEON_PY_BPR_MIN_COR", 0.20),
     min_overlap=_env_float("GIDEON_PY_BPR_MIN_OVERLAP", 0.10),
     max_ndcg=_env_float("GIDEON_PY_BPR_MAX_NDCG_DELTA", 0.05),
     max_recall=_env_float("GIDEON_PY_BPR_MAX_RECALL_DELTA", 0.07)),
    (name="IALS", model=() -> IALS(rank=16, λ=0.01, α=40.0, max_iter=15, verbose=false),
     scores_path="py_ials_scores.csv", required=true, metrics_path="py_ials_metrics.json",
     min_cor=_env_float("GIDEON_PY_IALS_MIN_COR", 0.35),
     min_overlap=_env_float("GIDEON_PY_IALS_MIN_OVERLAP", 0.15),
     max_ndcg=_env_float("GIDEON_PY_IALS_MAX_NDCG_DELTA", 0.06),
     max_recall=_env_float("GIDEON_PY_IALS_MAX_RECALL_DELTA", 0.06)),
    (name="EALS", model=() -> EALS(rank=16, λ=0.01, unobserved_weight=10.0, max_iter=20, verbose=false),
     scores_path="py_eals_scores.csv", required=true, metrics_path="py_eals_metrics.json",
     min_cor=_env_float("GIDEON_PY_EALS_MIN_COR", 0.15),
     min_overlap=_env_float("GIDEON_PY_EALS_MIN_OVERLAP", 0.10),
     max_ndcg=_env_float("GIDEON_PY_EALS_MAX_NDCG_DELTA", 0.06),
     max_recall=_env_float("GIDEON_PY_EALS_MAX_RECALL_DELTA", 0.06)),
    (name="LogisticMF", model=() -> LogisticMF(rank=16, λ=0.6, α=1.0,
                                                 lr=1.0, max_iter=30,
                                                 n_negative=30, verbose=false),
     scores_path="py_lmf_scores.csv", required=false, metrics_path="py_lmf_metrics.json",
     # Meaningful bounds (was -1.0 / 0.08, which accepted any result): the two
     # implementations share the same objective and gradient (verified against
     # implicit/cpu/lmf.pyx) but differ in negative sampling — implicit draws from
     # the flat occurrence list (can resample seen items), Gideon rejection-samples
     # unobserved items — so trajectories diverge and split-metric parity is
     # diagnostic rather than enforced.
     min_cor=_env_float("GIDEON_PY_LMF_MIN_COR", 0.40),
     min_overlap=_env_float("GIDEON_PY_LMF_MIN_OVERLAP", 0.40),
     max_ndcg=nothing, max_recall=nothing),
]

const REQUIRED_FILES = [
    "X_small.csv", "X_small_dims.csv", "X_train.csv", "X_train_dims.csv",
    "X_test.csv", "X_test_dims.csv",
    "py_als_scores.csv", "py_ials_scores.csv", "py_eals_scores.csv", "py_bpr_scores.csv",
]

const X_PATH   = joinpath(PY_FIXTURE_DIR, "X_small.csv")
const X_DIMS   = joinpath(PY_FIXTURE_DIR, "X_small_dims.csv")
const XT_PATH  = joinpath(PY_FIXTURE_DIR, "X_train.csv")
const XT_DIMS  = joinpath(PY_FIXTURE_DIR, "X_train_dims.csv")
const XE_PATH  = joinpath(PY_FIXTURE_DIR, "X_test.csv")
const XE_DIMS  = joinpath(PY_FIXTURE_DIR, "X_test_dims.csv")

require_files(joinpath.(PY_FIXTURE_DIR, REQUIRED_FILES), PY_FIXTURE_DIR,
              "Generate fixtures with: python3 validation/fixtures_py.py")

X      = _load_sparse(X_PATH,  X_DIMS)
X_train = _load_sparse(XT_PATH, XT_DIMS)
X_test  = _load_sparse(XE_PATH, XE_DIMS)

@testset "Python Reference Comparison" begin

    # ── Score parity: correlation + top-k overlap (table-driven) ────────────
    @testset "Score parity" begin
        for case in SCORE_CASES
            score_path = joinpath(PY_FIXTURE_DIR, case.scores_path)
            if !isfile(score_path)
                case.required && error("missing required fixture: $(case.scores_path)")
                @info "Skipping $(case.name) parity: $(case.scores_path) not found"
                continue
            end
            @testset "$(case.name)" begin
                assert_score_parity(
                    label=case.name,
                    model=case.model(),
                    X=X, py_scores=_read_matrix(score_path),
                    min_cor=case.min_cor, min_overlap=case.min_overlap,
                    X_train=X_train, X_test=X_test,
                    metrics_path=isnothing(case.metrics_path) ? nothing :
                                 joinpath(PY_FIXTURE_DIR, case.metrics_path),
                    max_ndcg=case.max_ndcg, max_recall=case.max_recall,
                )
            end
        end
    end

    # ── EASE: closed-form solution → exact match ────────────────────────────
    @testset "EASE vs Python" begin
        ease_path = joinpath(PY_FIXTURE_DIR, "py_ease_B.csv")
        isfile(ease_path) || error("missing required fixture: py_ease_B.csv")

        py_B = _read_matrix(ease_path)
        m = EASE(λ=100.0, verbose=false)
        fit!(m, X)
        jl_B = Matrix(m.B)

        rel_frob = norm(jl_B - py_B) / (norm(py_B) + 1e-12)
        c = _cor(vec(jl_B), vec(py_B))

        max_rel = _env_float("GIDEON_PY_EASE_MAX_REL_FROB", 1e-6)
        min_cor = _env_float("GIDEON_PY_EASE_MIN_COR", 0.9999)

        @test size(jl_B) == size(py_B)
        @test all(abs.(diag(jl_B)) .< 1e-8)
        @test rel_frob <= max_rel
        @test c >= min_cor
        println("  EASE relative Frobenius error: $rel_frob")
        println("  EASE matrix correlation: $c")
    end

    # ── SLIM vs sklearn ElasticNet (optional: sklearn may be absent) ────────
    @testset "SLIM vs Python" begin
        slim_path = joinpath(PY_FIXTURE_DIR, "py_slim_W.csv")
        if !isfile(slim_path)
            @info "Skipping SLIM parity: py_slim_W.csv not found (scikit-learn may be unavailable)"
        else
            py_w = _read_matrix(slim_path)
            m = SLIM(λ_l1=0.001, λ_l2=0.01, max_iter=50, verbose=false)
            fit!(m, X_train)
            jl_w = Matrix(m.W)

            @test size(jl_w) == size(py_w)
            c = _cor(vec(jl_w), vec(py_w))
            min_cor = _env_float("GIDEON_PY_SLIM_MIN_W_COR", 0.6)
            @test c >= min_cor
            println("  SLIM weight-matrix correlation: $c")

            assert_metric_parity(
                label="SLIM",
                model=m, X_train=X_train, X_test=X_test,
                metrics_path=joinpath(PY_FIXTURE_DIR, "py_slim_metrics.json"),
                max_ndcg=_env_float("GIDEON_PY_SLIM_MAX_NDCG_DELTA", 0.08),
                max_recall=_env_float("GIDEON_PY_SLIM_MAX_RECALL_DELTA", 0.08),
            )
        end
    end

    # ── SoftImpute: diagnostic by default (different local optima possible) ─
    @testset "SoftImpute vs Python" begin
        recon_path = joinpath(PY_FIXTURE_DIR, "py_softimpute_recon.csv")
        svals_path = joinpath(PY_FIXTURE_DIR, "py_softimpute_svals.csv")
        if !(isfile(recon_path) && isfile(svals_path))
            @info "Skipping SoftImpute parity: fixture files not found"
        else
            py_recon = _read_matrix(recon_path)
            py_svals = vec(_read_matrix(svals_path))

            m = SoftSVD(rank=10, λ=0.1, max_iter=40, tol=1e-4,
                        final_svd=false, verbose=false)
            fit!(m, X_train; rng=MersenneTwister(42))
            jl_recon = Matrix(m.U * Diagonal(m.d) * m.V')

            c = _cor(vec(jl_recon), vec(py_recon))
            nsv = min(length(m.d), length(py_svals), 10)
            sv_rel = norm(Float64.(m.d[1:nsv]) .- Float64.(py_svals[1:nsv])) /
                     (norm(Float64.(py_svals[1:nsv])) + 1e-12)

            min_cor = _env_float("GIDEON_PY_SOFT_MIN_RECON_COR", 0.75)
            max_sv_rel = _env_float("GIDEON_PY_SOFT_MAX_SVAL_REL", 0.40)

            if soft_strict
                @test c >= min_cor
                @test sv_rel <= max_sv_rel
            else
                @info "SoftImpute parity is diagnostic by default; set GIDEON_PY_SOFT_STRICT=1 to enforce thresholds"
            end
            println("  SoftImpute reconstruction correlation: $c")
            println("  SoftImpute singular-value relative error: $sv_rel")
        end
    end

    # ── PureSVD vs scipy.sparse.linalg.svds ─────────────────────────────────
    @testset "PureSVD vs scipy.svds" begin
        svals_path = joinpath(PY_FIXTURE_DIR, "py_puresvd_svals.csv")
        recon_path = joinpath(PY_FIXTURE_DIR, "py_puresvd_recon.csv")
        if !(isfile(svals_path) && isfile(recon_path))
            @info "Skipping PureSVD parity: fixture files not found"
        else
            py_svals = vec(_read_matrix(svals_path))
            py_recon = _read_matrix(recon_path)

            m = PureSVD(rank=10, max_iter=200, tol=1e-6, verbose=false)
            fit!(m, X_train; rng=MersenneTwister(42))
            jl_recon = Matrix(m.U * Diagonal(m.d) * m.V')

            nsv = min(length(m.d), length(py_svals))
            sv_rel = norm(sort(m.d[1:nsv]; rev=true) .- sort(py_svals[1:nsv]; rev=true)) /
                     (norm(py_svals[1:nsv]) + 1e-12)
            c = _cor(vec(jl_recon), vec(py_recon))

            @test sv_rel < 0.05
            @test c >= 0.99
            println("  PureSVD singular-value relative error: $(round(sv_rel; sigdigits=4))")
            println("  PureSVD reconstruction correlation: $(round(c; sigdigits=6))")
        end
    end

    # ── ItemKNN vs sklearn cosine KNN ───────────────────────────────────────
    @testset "ItemKNN vs Python cosine KNN" begin
        w_path = joinpath(PY_FIXTURE_DIR, "py_knn_W.csv")
        scores_path = joinpath(PY_FIXTURE_DIR, "py_knn_scores.csv")
        if !(isfile(w_path) && isfile(scores_path))
            @info "Skipping ItemKNN parity: fixture files not found"
        else
            py_W = _read_matrix(w_path)
            py_scores = _read_matrix(scores_path)

            m = ItemKNN(k=20, similarity=:cosine, normalize=true, shrinkage=0.0, verbose=false)
            fit!(m, X)
            jl_W = Matrix(m.W)
            jl_scores = Matrix(X * m.W)

            w_cor = _cor(vec(jl_W), vec(py_W))
            score_cor = _cor(vec(jl_scores), vec(py_scores))
            k = 10
            n_eval = min(25, size(jl_scores, 1))
            overlap = _mean_topk_overlap(_row_topk(jl_scores[1:n_eval, :], k),
                                         _row_topk(py_scores[1:n_eval, :], k))

            @test w_cor >= 0.95
            @test score_cor >= 0.95
            @test overlap >= 0.70
            println("  ItemKNN W correlation: $(round(w_cor; sigdigits=4))")
            println("  ItemKNN score correlation: $(round(score_cor; sigdigits=4))")
            println("  ItemKNN top-$k overlap: $(round(overlap; sigdigits=4))")

            assert_metric_parity(
                label="ItemKNN",
                model=m, X_train=X_train, X_test=X_test,
                metrics_path=joinpath(PY_FIXTURE_DIR, "py_knn_metrics.json"),
                max_ndcg=0.10, max_recall=nothing,
            )
        end
    end

    # ── ADMMSLIM vs Python ADMM reference ───────────────────────────────────
    @testset "ADMMSLIM vs Python ADMM" begin
        w_path = joinpath(PY_FIXTURE_DIR, "py_admmslim_W.csv")
        scores_path = joinpath(PY_FIXTURE_DIR, "py_admmslim_scores.csv")
        if !(isfile(w_path) && isfile(scores_path))
            @info "Skipping ADMMSLIM parity: fixture files not found"
        else
            py_W = _read_matrix(w_path)
            py_scores = _read_matrix(scores_path)

            m = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, ρ=1.0, max_iter=100,
                         tol=1e-5, nonneg=true, verbose=false)
            fit!(m, X)
            jl_W = Matrix(m.W)
            jl_scores = Matrix(X * m.W)

            w_frob = norm(jl_W - py_W) / (norm(py_W) + 1e-12)
            w_cor = _cor(vec(jl_W), vec(py_W))
            score_cor = _cor(vec(jl_scores), vec(py_scores))
            k = 10
            n_eval = min(25, size(jl_scores, 1))
            overlap = _mean_topk_overlap(_row_topk(jl_scores[1:n_eval, :], k),
                                         _row_topk(py_scores[1:n_eval, :], k))

            @test w_frob < 0.05
            @test w_cor >= 0.99
            @test score_cor >= 0.99
            @test overlap >= 0.85
            println("  ADMMSLIM W Frobenius relative error: $(round(w_frob; sigdigits=4))")
            println("  ADMMSLIM W correlation: $(round(w_cor; sigdigits=6))")
            println("  ADMMSLIM score correlation: $(round(score_cor; sigdigits=6))")
            println("  ADMMSLIM top-$k overlap: $(round(overlap; sigdigits=4))")

            assert_metric_parity(
                label="ADMMSLIM",
                model=m, X_train=X_train, X_test=X_test,
                metrics_path=joinpath(PY_FIXTURE_DIR, "py_admmslim_metrics.json"),
                max_ndcg=0.05, max_recall=nothing,
            )
        end
    end
end
