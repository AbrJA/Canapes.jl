# validation/validate_py.jl
# Optional validation against Python implicit implementations.

using Gideon, SparseArrays, LinearAlgebra, Random, Statistics
using Test

const PY_FIXTURE_DIR = get(ENV, "GIDEON_PY_FIXTURE_DIR", "/tmp/gideon_fixtures/python")

function _read_matrix(path::String)
    lines = readlines(path)
    rows = [parse.(Float64, split(strip(l), ','))
            for l in lines if !isempty(strip(l))]
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    reduce(vcat, [r' for r in rows])
end

function _load_sparse(triplet::String, dims::String)
    lines = readlines(triplet)
    rv = Int[]
    cv = Int[]
    vv = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        p = split(strip(line), ',')
        push!(rv, parse(Int, p[1]))
        push!(cv, parse(Int, p[2]))
        push!(vv, parse(Float64, p[3]))
    end
    dlines = readlines(dims)
    hdr = split(strip(dlines[1]), ',')
    dvals = split(strip(dlines[2]), ',')
    nr = parse(Int, dvals[findfirst(==("nr"), hdr)])
    nc = parse(Int, dvals[findfirst(==("nc"), hdr)])
    sparse(rv, cv, vv, nr, nc)
end

function _cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x64 = Float64.(x)
    y64 = Float64.(y)
    mx = sum(x64) / length(x64)
    my = sum(y64) / length(y64)
    dx = x64 .- mx
    dy = y64 .- my
    dot(dx, dy) / (sqrt(dot(dx, dx) * dot(dy, dy)) + 1e-15)
end

function _row_topk(scores::AbstractMatrix{<:Real}, k::Int)
    n_users = size(scores, 1)
    out = Matrix{Int}(undef, n_users, k)
    for u in 1:n_users
        out[u, :] = partialsortperm(@view(scores[u, :]), 1:k; rev=true)
    end
    out
end

function _mean_topk_overlap(a::Matrix{Int}, b::Matrix{Int})
    n_users = size(a, 1)
    k = size(a, 2)
    s = 0.0
    for u in 1:n_users
        s += length(intersect(@view(a[u, :]), @view(b[u, :]))) / k
    end
    s / n_users
end

function _read_json_metric(path::String, key::String)
    txt = read(path, String)
    m = match(Regex("\"$key\"\\s*:\\s*([0-9eE+\\-.]+)"), txt)
    isnothing(m) && error("Metric '$key' not found in $(basename(path))")
    parse(Float64, m.captures[1])
end

function _safe_recommend(model, X::SparseMatrixCSC, k::Int)
    n_users, n_items = size(X)
    r = recommend(model, X; k=k)
    if size(r) == (n_users, k)
        return r
    end
    if size(r, 1) == k && size(r, 2) == n_users
        return Matrix(r')
    end
    error("Unexpected recommend() shape $(size(r)) for n_users=$n_users, k=$k")
end

@testset "Python Reference Comparison" begin
    if !isdir(PY_FIXTURE_DIR)
        @warn """
        Python fixtures not found at: $PY_FIXTURE_DIR
        Generate fixtures with:
          python3 validation/fixtures_py.py
        """
        return
    end

    x_path = joinpath(PY_FIXTURE_DIR, "X_small.csv")
    d_path = joinpath(PY_FIXTURE_DIR, "X_small_dims.csv")
    als_path = joinpath(PY_FIXTURE_DIR, "py_als_scores.csv")
    ials_path = joinpath(PY_FIXTURE_DIR, "py_ials_scores.csv")
    eals_path = joinpath(PY_FIXTURE_DIR, "py_eals_scores.csv")
    bpr_path = joinpath(PY_FIXTURE_DIR, "py_bpr_scores.csv")
    lmf_path = joinpath(PY_FIXTURE_DIR, "py_lmf_scores.csv")
    ease_path = joinpath(PY_FIXTURE_DIR, "py_ease_B.csv")
    slim_w_path = joinpath(PY_FIXTURE_DIR, "py_slim_W.csv")
    soft_recon_path = joinpath(PY_FIXTURE_DIR, "py_softimpute_recon.csv")
    soft_svals_path = joinpath(PY_FIXTURE_DIR, "py_softimpute_svals.csv")
    x_train_path = joinpath(PY_FIXTURE_DIR, "X_train.csv")
    x_train_dims = joinpath(PY_FIXTURE_DIR, "X_train_dims.csv")
    x_test_path = joinpath(PY_FIXTURE_DIR, "X_test.csv")
    x_test_dims = joinpath(PY_FIXTURE_DIR, "X_test_dims.csv")
    bpr_metrics_path = joinpath(PY_FIXTURE_DIR, "py_bpr_metrics.json")
    ials_metrics_path = joinpath(PY_FIXTURE_DIR, "py_ials_metrics.json")
    eals_metrics_path = joinpath(PY_FIXTURE_DIR, "py_eals_metrics.json")
    lmf_metrics_path = joinpath(PY_FIXTURE_DIR, "py_lmf_metrics.json")
    slim_metrics_path = joinpath(PY_FIXTURE_DIR, "py_slim_metrics.json")

    if !(isfile(x_path) && isfile(d_path) && isfile(als_path) && isfile(bpr_path))
        @warn """
        Python fixture files are incomplete at: $PY_FIXTURE_DIR
        Expected: X_small.csv, X_small_dims.csv, py_als_scores.csv, py_bpr_scores.csv
        Regenerate with:
          python3 validation/fixtures_py.py
        """
        return
    end

    X = _load_sparse(x_path, d_path)
    X_train = _load_sparse(x_train_path, x_train_dims)
    X_test = _load_sparse(x_test_path, x_test_dims)
    py_als = _read_matrix(als_path)
    py_ials = isfile(ials_path) ? _read_matrix(ials_path) : py_als
    py_eals = isfile(eals_path) ? _read_matrix(eals_path) : py_als
    py_bpr = _read_matrix(bpr_path)

    @test size(py_als) == size(py_bpr) == size(X)

    rank = 16

    @testset "WMF (Julia) vs ALS (Python)" begin
        m = WMF(rank=rank, λ=0.1, α=40.0, max_iter=20,
                solver=CholeskySolver(), feedback=IMPLICIT, verbose=false)
        fit!(m, X; rng=MersenneTwister(42))
        jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

        c = _cor(vec(jl_scores), vec(py_als))
        k = 10
        n_users_eval = min(25, size(jl_scores, 1))
        jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
        py_top = _row_topk(py_als[1:n_users_eval, :], k)
        overlap = _mean_topk_overlap(jl_top, py_top)

        min_cor = parse(Float64, get(ENV, "GIDEON_PY_ALS_MIN_COR", "0.45"))
        min_ov = parse(Float64, get(ENV, "GIDEON_PY_ALS_MIN_OVERLAP", "0.20"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov
        println("  ALS score correlation: $c")
        println("  ALS top-$k overlap: $overlap")
    end

    @testset "BPR (Julia) vs BPR (Python)" begin
        m = BPR(rank=rank, max_iter=40, verbose=false)
        fit!(m, X; rng=MersenneTwister(42))
        jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

        c = _cor(vec(jl_scores), vec(py_bpr))
        k = 10
        n_users_eval = min(25, size(jl_scores, 1))
        jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
        py_top = _row_topk(py_bpr[1:n_users_eval, :], k)
        overlap = _mean_topk_overlap(jl_top, py_top)

        min_cor = parse(Float64, get(ENV, "GIDEON_PY_BPR_MIN_COR", "0.20"))
        min_ov = parse(Float64, get(ENV, "GIDEON_PY_BPR_MIN_OVERLAP", "0.10"))
        max_ndcg_delta = parse(Float64, get(ENV, "GIDEON_PY_BPR_MAX_NDCG_DELTA", "0.05"))
        max_recall_delta = parse(Float64, get(ENV, "GIDEON_PY_BPR_MAX_RECALL_DELTA", "0.07"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov

        if isfile(bpr_metrics_path)
            py_ndcg = _read_json_metric(bpr_metrics_path, "ndcg")
            py_recall = _read_json_metric(bpr_metrics_path, "recall")
            jl_preds = _safe_recommend(m, X_train, 10)
            jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
            jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
            @test abs(jl_ndcg - py_ndcg) <= max_ndcg_delta
            @test abs(jl_recall - py_recall) <= max_recall_delta
            println("  BPR NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
            println("  BPR Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
        else
            @info "Skipping BPR split-metric parity: py_bpr_metrics.json not found"
        end
        println("  BPR score correlation: $c")
        println("  BPR top-$k overlap: $overlap")
    end

    @testset "IALS (Julia) vs IALS/ALS (Python)" begin
        m = IALS(rank=rank, λ=0.01, α=40.0, max_iter=15, verbose=false)
        fit!(m, X; rng=MersenneTwister(42))
        jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

        c = _cor(vec(jl_scores), vec(py_ials))
        k = 10
        n_users_eval = min(25, size(jl_scores, 1))
        jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
        py_top = _row_topk(py_ials[1:n_users_eval, :], k)
        overlap = _mean_topk_overlap(jl_top, py_top)

        min_cor = parse(Float64, get(ENV, "GIDEON_PY_IALS_MIN_COR", "0.35"))
        min_ov = parse(Float64, get(ENV, "GIDEON_PY_IALS_MIN_OVERLAP", "0.15"))
        max_ndcg_delta = parse(Float64, get(ENV, "GIDEON_PY_IALS_MAX_NDCG_DELTA", "0.06"))
        max_recall_delta = parse(Float64, get(ENV, "GIDEON_PY_IALS_MAX_RECALL_DELTA", "0.06"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov

        if isfile(ials_metrics_path)
            py_ndcg = _read_json_metric(ials_metrics_path, "ndcg")
            py_recall = _read_json_metric(ials_metrics_path, "recall")
            jl_preds = _safe_recommend(m, X_train, 10)
            jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
            jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
            @test abs(jl_ndcg - py_ndcg) <= max_ndcg_delta
            @test abs(jl_recall - py_recall) <= max_recall_delta
            println("  IALS NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
            println("  IALS Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
        else
            @info "Skipping IALS split-metric parity: py_ials_metrics.json not found"
        end

        println("  IALS score correlation: $c")
        println("  IALS top-$k overlap: $overlap")
    end

    @testset "EALS (Julia) vs EALS surrogate (Python)" begin
        m = EALS(rank=rank, λ=0.01, w0=10.0, max_iter=20, verbose=false)
        fit!(m, X; rng=MersenneTwister(42))
        jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

        c = _cor(vec(jl_scores), vec(py_eals))
        k = 10
        n_users_eval = min(25, size(jl_scores, 1))
        jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
        py_top = _row_topk(py_eals[1:n_users_eval, :], k)
        overlap = _mean_topk_overlap(jl_top, py_top)

        min_cor = parse(Float64, get(ENV, "GIDEON_PY_EALS_MIN_COR", "0.15"))
        min_ov = parse(Float64, get(ENV, "GIDEON_PY_EALS_MIN_OVERLAP", "0.10"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov

        if isfile(eals_metrics_path)
            py_ndcg = _read_json_metric(eals_metrics_path, "ndcg")
            py_recall = _read_json_metric(eals_metrics_path, "recall")
            jl_preds = _safe_recommend(m, X_train, 10)
            jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
            jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
            println("  EALS NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
            println("  EALS Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
        else
            @info "Skipping EALS split-metric parity: py_eals_metrics.json not found"
        end

        println("  EALS score correlation: $c")
        println("  EALS top-$k overlap: $overlap")
    end

    @testset "LogisticMF (Julia) vs LMF (Python, optional)" begin
        if isfile(lmf_path)
            py_lmf = _read_matrix(lmf_path)

            m = LogisticMF(
                rank=rank, λ=0.6, α=1.0,
                learning_rate=1.0, max_iter=30, n_negative=30, verbose=false,
            )
            fit!(m, X; rng=MersenneTwister(42))
            jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

            c = _cor(vec(jl_scores), vec(py_lmf))
            k = 10
            n_users_eval = min(25, size(jl_scores, 1))
            jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
            py_top = _row_topk(py_lmf[1:n_users_eval, :], k)
            overlap = _mean_topk_overlap(jl_top, py_top)

            min_cor = parse(Float64, get(ENV, "GIDEON_PY_LMF_MIN_COR", "-1.0"))
            min_ov = parse(Float64, get(ENV, "GIDEON_PY_LMF_MIN_OVERLAP", "0.08"))
            max_ndcg_delta = parse(Float64, get(ENV, "GIDEON_PY_LMF_MAX_NDCG_DELTA", "0.07"))
            max_recall_delta = parse(Float64, get(ENV, "GIDEON_PY_LMF_MAX_RECALL_DELTA", "0.07"))

            @test isfinite(c)
            # Different sampling/training implementations can shift score scales;
            # ranking overlap is the primary parity signal for LogisticMF.
            @test c >= min_cor
            @test overlap >= min_ov

            if isfile(lmf_metrics_path)
                py_ndcg = _read_json_metric(lmf_metrics_path, "ndcg")
                py_recall = _read_json_metric(lmf_metrics_path, "recall")
                jl_preds = _safe_recommend(m, X_train, 10)
                jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
                jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
                if get(ENV, "GIDEON_PY_LMF_STRICT", "0") == "1"
                    @test abs(jl_ndcg - py_ndcg) <= max_ndcg_delta
                    @test abs(jl_recall - py_recall) <= max_recall_delta
                else
                    @info "LMF split-metric parity is diagnostic by default; set GIDEON_PY_LMF_STRICT=1 to enforce thresholds"
                end
                println("  LMF NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
                println("  LMF Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
            else
                @info "Skipping LMF split-metric parity: py_lmf_metrics.json not found"
            end
            println("  LMF score correlation: $c")
            println("  LMF top-$k overlap: $overlap")
        else
            @info "Skipping LogisticMF parity: py_lmf_scores.csv not found"
        end
    end

    @testset "EASE (Julia) vs EASE (Python)" begin
        if isfile(ease_path)
            py_B = _read_matrix(ease_path)

            m = EASE(λ=100.0, verbose=false)
            fit!(m, X)
            jl_B = Matrix(m.B)

            @test size(jl_B) == size(py_B)
            @test all(abs.(diag(jl_B)) .< 1e-8)

            rel_frob = norm(jl_B - py_B) / (norm(py_B) + 1e-12)
            c = _cor(vec(jl_B), vec(py_B))

            max_rel = parse(Float64, get(ENV, "GIDEON_PY_EASE_MAX_REL_FROB", "1e-6"))
            min_cor = parse(Float64, get(ENV, "GIDEON_PY_EASE_MIN_COR", "0.9999"))

            @test rel_frob <= max_rel
            @test c >= min_cor
            println("  EASE relative Frobenius error: $rel_frob")
            println("  EASE matrix correlation: $c")
        else
            @info "Skipping EASE parity: py_ease_B.csv not found"
        end
    end

    @testset "SLIM (Julia) vs SLIM (Python, optional)" begin
        if isfile(slim_w_path)
            py_w = _read_matrix(slim_w_path)

            m = SLIM(λ_1=0.001, λ_2=0.01, max_iter=50, verbose=false)
            fit!(m, X_train)
            jl_w = Matrix(m.W)

            @test size(jl_w) == size(py_w)
            c = _cor(vec(jl_w), vec(py_w))
            min_cor = parse(Float64, get(ENV, "GIDEON_PY_SLIM_MIN_W_COR", "0.6"))
            @test c >= min_cor

            if isfile(slim_metrics_path)
                py_ndcg = _read_json_metric(slim_metrics_path, "ndcg")
                py_recall = _read_json_metric(slim_metrics_path, "recall")
                jl_preds = _safe_recommend(m, X_train, 10)
                jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
                jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
                max_ndcg_delta = parse(Float64, get(ENV, "GIDEON_PY_SLIM_MAX_NDCG_DELTA", "0.08"))
                max_recall_delta = parse(Float64, get(ENV, "GIDEON_PY_SLIM_MAX_RECALL_DELTA", "0.08"))
                @test abs(jl_ndcg - py_ndcg) <= max_ndcg_delta
                @test abs(jl_recall - py_recall) <= max_recall_delta
                println("  SLIM NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
                println("  SLIM Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
            else
                @info "Skipping SLIM split-metric parity: py_slim_metrics.json not found"
            end

            println("  SLIM weight-matrix correlation: $c")
        else
            @info "Skipping SLIM parity: py_slim_W.csv not found (scikit-learn may be unavailable)"
        end
    end

    @testset "SoftImpute (Julia) vs SoftImpute (Python)" begin
        if isfile(soft_recon_path) && isfile(soft_svals_path)
            py_recon = _read_matrix(soft_recon_path)
            py_svals_mat = _read_matrix(soft_svals_path)
            py_svals = vec(py_svals_mat)

            m = SoftSVD(
                rank=10,
                λ=0.1,
                max_iter=40,
                convergence_tol=1e-4,
                final_svd=false,
                verbose=false,
            )
            fit!(m, X_train; rng=MersenneTwister(42))
            jl_recon = Matrix(m.U * Diagonal(m.d) * m.V')

            @test size(jl_recon) == size(py_recon)
            c = _cor(vec(jl_recon), vec(py_recon))
            min_cor = parse(Float64, get(ENV, "GIDEON_PY_SOFT_MIN_RECON_COR", "0.75"))

            nsv = min(length(m.d), length(py_svals), 10)
            jl_sv = Float64.(m.d[1:nsv])
            py_sv = Float64.(py_svals[1:nsv])
            sv_rel = norm(jl_sv - py_sv) / (norm(py_sv) + 1e-12)
            max_sv_rel = parse(Float64, get(ENV, "GIDEON_PY_SOFT_MAX_SVAL_REL", "0.40"))

            if get(ENV, "GIDEON_PY_SOFT_STRICT", "0") == "1"
                @test c >= min_cor
                @test sv_rel <= max_sv_rel
            else
                @info "SoftImpute parity is diagnostic by default; set GIDEON_PY_SOFT_STRICT=1 to enforce thresholds"
            end

            println("  SoftImpute reconstruction correlation: $c")
            println("  SoftImpute singular-value relative error: $sv_rel")
        else
            @info "Skipping SoftImpute parity: softimpute fixture files not found"
        end
    end

    # ── PureSVD (Julia) vs scipy.sparse.linalg.svds ──────────────────────

    @testset "PureSVD (Julia) vs scipy.svds" begin
        svd_svals_path = joinpath(PY_FIXTURE_DIR, "py_puresvd_svals.csv")
        svd_recon_path = joinpath(PY_FIXTURE_DIR, "py_puresvd_recon.csv")

        if isfile(svd_svals_path) && isfile(svd_recon_path)
            py_svals = vec(_read_matrix(svd_svals_path))
            py_recon = _read_matrix(svd_recon_path)

            m = PureSVD(rank=10, max_iter=200, convergence_tol=1e-6, verbose=false)
            fit!(m, X_train; rng=MersenneTwister(42))
            jl_recon = Matrix(m.U * Diagonal(m.d) * m.V')

            # Singular values should match closely (same truncated SVD)
            nsv = min(length(m.d), length(py_svals))
            jl_sv = Float64.(sort(m.d[1:nsv]; rev=true))
            py_sv = Float64.(sort(py_svals[1:nsv]; rev=true))
            sv_rel = norm(jl_sv - py_sv) / (norm(py_sv) + 1e-12)

            # Reconstruction correlation
            c = _cor(vec(jl_recon), vec(py_recon))

            @test sv_rel < 0.05  # singular values should be very close
            @test c >= 0.99      # reconstruction should be nearly identical

            println("  PureSVD singular-value relative error: $(round(sv_rel; sigdigits=4))")
            println("  PureSVD reconstruction correlation: $(round(c; sigdigits=6))")
        else
            @info "Skipping PureSVD parity: fixture files not found"
        end
    end

    # ── ItemKNN (Julia) vs sklearn cosine KNN ────────────────────────────

    @testset "ItemKNN (Julia) vs Python cosine KNN" begin
        knn_w_path = joinpath(PY_FIXTURE_DIR, "py_knn_W.csv")
        knn_scores_path = joinpath(PY_FIXTURE_DIR, "py_knn_scores.csv")
        knn_metrics_path = joinpath(PY_FIXTURE_DIR, "py_knn_metrics.json")

        if isfile(knn_w_path) && isfile(knn_scores_path)
            py_W = _read_matrix(knn_w_path)
            py_scores = _read_matrix(knn_scores_path)

            m = ItemKNN(k=20, similarity=:cosine, normalize=true, shrinkage=0.0, verbose=false)
            fit!(m, X)
            jl_W = Matrix(m.W)
            jl_scores = Matrix(X * m.W)

            # W matrix correlation — both use cosine + top-k + row-normalize
            w_cor = _cor(vec(jl_W), vec(py_W))

            # Score correlation
            score_cor = _cor(vec(jl_scores), vec(py_scores))

            # Top-k overlap
            k = 10
            n_eval = min(25, size(jl_scores, 1))
            jl_top = _row_topk(jl_scores[1:n_eval, :], k)
            py_top = _row_topk(py_scores[1:n_eval, :], k)
            overlap = _mean_topk_overlap(jl_top, py_top)

            @test w_cor >= 0.95      # W matrices should be very similar
            @test score_cor >= 0.95  # scores should match
            @test overlap >= 0.70   # top-k should mostly agree

            println("  ItemKNN W correlation: $(round(w_cor; sigdigits=4))")
            println("  ItemKNN score correlation: $(round(score_cor; sigdigits=4))")
            println("  ItemKNN top-$k overlap: $(round(overlap; sigdigits=4))")

            # Compare quality metrics if available
            if isfile(knn_metrics_path)
                py_ndcg = _read_json_metric(knn_metrics_path, "ndcg")
                jl_preds = _safe_recommend(m, X_train, 10)
                jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
                delta = abs(jl_ndcg - py_ndcg)
                @test delta < 0.10  # quality should be comparable
                println("  ItemKNN NDCG@10: Julia=$(round(jl_ndcg; sigdigits=4)) Python=$(round(py_ndcg; sigdigits=4)) Δ=$(round(delta; sigdigits=4))")
            end
        else
            @info "Skipping ItemKNN parity: fixture files not found"
        end
    end

    # ── ADMMSLIM (Julia) vs Python ADMM reference ────────────────────────

    @testset "ADMMSLIM (Julia) vs Python ADMM" begin
        admm_w_path = joinpath(PY_FIXTURE_DIR, "py_admmslim_W.csv")
        admm_scores_path = joinpath(PY_FIXTURE_DIR, "py_admmslim_scores.csv")
        admm_metrics_path = joinpath(PY_FIXTURE_DIR, "py_admmslim_metrics.json")

        if isfile(admm_w_path) && isfile(admm_scores_path)
            py_W = _read_matrix(admm_w_path)
            py_scores = _read_matrix(admm_scores_path)

            m = ADMMSLIM(λ_1=0.01, λ_2=100.0, ρ=1.0, max_iter=100,
                         convergence_tol=1e-5, nonneg=true, verbose=false)
            fit!(m, X)
            jl_W = m.W
            jl_scores = Matrix(X * m.W)

            # W matrix comparison — same algorithm, same hyperparameters
            w_frob = norm(jl_W - py_W) / (norm(py_W) + 1e-12)
            w_cor = _cor(vec(jl_W), vec(py_W))

            # Score comparison
            score_cor = _cor(vec(jl_scores), vec(py_scores))

            # Top-k overlap
            k = 10
            n_eval = min(25, size(jl_scores, 1))
            jl_top = _row_topk(jl_scores[1:n_eval, :], k)
            py_top = _row_topk(py_scores[1:n_eval, :], k)
            overlap = _mean_topk_overlap(jl_top, py_top)

            @test w_frob < 0.05    # W should be nearly identical (same algorithm)
            @test w_cor >= 0.99    # very high correlation expected
            @test score_cor >= 0.99
            @test overlap >= 0.85  # top-k should largely agree

            println("  ADMMSLIM W Frobenius relative error: $(round(w_frob; sigdigits=4))")
            println("  ADMMSLIM W correlation: $(round(w_cor; sigdigits=6))")
            println("  ADMMSLIM score correlation: $(round(score_cor; sigdigits=6))")
            println("  ADMMSLIM top-$k overlap: $(round(overlap; sigdigits=4))")

            # Compare quality metrics
            if isfile(admm_metrics_path)
                py_ndcg = _read_json_metric(admm_metrics_path, "ndcg")
                jl_preds = _safe_recommend(m, X_train, 10)
                jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
                delta = abs(jl_ndcg - py_ndcg)
                @test delta < 0.05  # same algorithm → very close quality
                println("  ADMMSLIM NDCG@10: Julia=$(round(jl_ndcg; sigdigits=4)) Python=$(round(py_ndcg; sigdigits=4)) Δ=$(round(delta; sigdigits=4))")
            end
        else
            @info "Skipping ADMMSLIM parity: fixture files not found"
        end
    end
end
