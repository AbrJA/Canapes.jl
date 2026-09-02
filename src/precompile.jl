# ──────────────────────────────────────────────────────────────────────────────
# Precompilation workloads — reduce TTFX for common workflows
# ──────────────────────────────────────────────────────────────────────────────

import PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    using SparseArrays, Random

    @compile_workload begin
        rng = MersenneTwister(1)
        n_users, n_items = 4, 5

        # Small sparse matrix for precompilation
        X_small = sprand(rng, n_users, n_items, 0.3)

        # WMF with CG solver
        m_cg = WMF(rank=4, λ=0.1, α=1.0, max_iter=2, solver=CGSolver(), verbose=false)
        fit!(m_cg, X_small; rng=MersenneTwister(2))
        recommend(m_cg, X_small; k=3)
        transform(m_cg, X_small)

        # WMF with Cholesky solver
        m_ch = WMF(rank=4, λ=0.1, α=1.0, max_iter=2, solver=CholeskySolver(), verbose=false)
        fit!(m_ch, X_small; rng=MersenneTwister(3))

        # IALS
        m_ials = IALS(rank=4, λ=0.01, α=10.0, max_iter=2, verbose=false)
        fit!(m_ials, X_small; rng=MersenneTwister(4))
        recommend(m_ials, X_small; k=3)

        # BPR
        m_bpr = BPR(rank=4, max_iter=2, n_samples=20, verbose=false)
        fit!(m_bpr, X_small; rng=MersenneTwister(5))
        recommend(m_bpr, X_small; k=3)

        # EASE
        m_ease = EASE(λ=100.0, verbose=false)
        fit!(m_ease, X_small)
        recommend(m_ease, X_small; k=3)

        # SLIM
        m_slim = SLIM(λ_l1=0.1, λ_l2=0.5, max_iter=5, verbose=false)
        fit!(m_slim, X_small)
        recommend(m_slim, X_small; k=3)

        # GloVe (square matrix)
        C = sprand(rng, 4, 4, 0.5)
        C = C + C'
        nonzeros(C) .= abs.(nonzeros(C)) .+ 0.1
        m_glove = GloVe(rank=4, max_iter=2, verbose=false)
        fit!(m_glove, C; rng=MersenneTwister(6))
        embeddings(m_glove)

        # RandomWalk
        m_rw = RandomWalk(β=0.0, k=3, verbose=false)
        fit!(m_rw, X_small)
        recommend(m_rw, X_small; k=3)

        # Explicit models
        m_baseline = BaselineOnly(max_iter=2, verbose=false)
        fit!(m_baseline, X_small)
        predict(m_baseline, X_small)
        m_slope = SlopeOne(verbose=false)
        fit!(m_slope, X_small)
        predict(m_slope, X_small)
        m_pearson = PearsonKNN(k=3, verbose=false)
        fit!(m_pearson, X_small)
        predict(m_pearson, X_small)

        # LogisticMF (experimental)
        m_lmf = Experimental.LogisticMF(rank=4, max_iter=2, verbose=false)
        fit!(m_lmf, X_small; rng=MersenneTwister(7))

        # FTRL
        y_small = rand(rng, n_users)
        m_ftrl = FTRL(lr=0.1, max_iter=1, verbose=false)
        update!(m_ftrl, X_small, y_small; rng=MersenneTwister(8))
        predict(m_ftrl, X_small)

        # FM
        m_fm = FM(rank=2, max_iter=2, verbose=false)
        fit!(m_fm, X_small, y_small; rng=MersenneTwister(9))
        predict(m_fm, X_small)

        # SoftImpute
        m_si = SoftImpute(rank=3, max_iter=3, verbose=false)
        fit!(m_si, X_small; rng=MersenneTwister(10))

        # SoftSVD
        m_svd = SoftSVD(rank=3, max_iter=3, verbose=false)
        fit!(m_svd, X_small; rng=MersenneTwister(10))

        # ItemKNN
        m_knn = ItemKNN(k=3, similarity=:cosine, verbose=false)
        fit!(m_knn, X_small; rng=MersenneTwister(11))
        recommend(m_knn, X_small; k=3)

        # ADMM-SLIM
        m_admm = ADMMSLIM(λ_l1=0.1, λ_l2=100.0, max_iter=5, verbose=false)
        fit!(m_admm, X_small; rng=MersenneTwister(12))
        recommend(m_admm, X_small; k=3)

        # Metrics
        preds_small = Matrix{Int}(hcat([randperm(rng, n_items)[1:3] for _ in 1:n_users]...)')
        actual_small = sprand(rng, n_users, n_items, 0.2)
        mean_ap_at_k(preds_small, actual_small; k=3)
        ndcg_at_k(preds_small, actual_small; k=3)
        precision_at_k(preds_small, actual_small; k=3)
        recall_at_k(preds_small, actual_small; k=3)

        # Cross-validation
        random_holdout(X_small; test_fraction=0.3, rng=MersenneTwister(10))

        # Sparse utilities
        to_csr(X_small)
        sparse_row_norms(X_small)
        sparse_col_nnz(X_small)
    end
end
