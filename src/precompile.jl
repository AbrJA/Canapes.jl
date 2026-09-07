# ──────────────────────────────────────────────────────────────────────────────
# Precompilation workloads — reduce TTFX for common workflows.
#
# One canonical fit! + output call per algorithm (default configuration), so
# the pkgimage stays complete without paying for redundant paths (extra
# solvers, transform/embeddings helpers, internal sparse utils).
# ──────────────────────────────────────────────────────────────────────────────

import PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    using SparseArrays, Random

    @compile_workload begin
        rng = MersenneTwister(1)
        X = sprand(rng, 8, 6, 0.4)
        y = rand(rng, size(X, 1))

        # ── Implicit matrix factorization ──
        fit!(WeightedMF(rank=4, max_iter=2, verbose=false), X)                        # WeightedMF (CG default)
        fit!(WeightedMF(rank=4, max_iter=2, solver=CholeskySolver(), verbose=false), X)
        m_w = WeightedMF(rank=4, max_iter=2, feedback=Explicit, verbose=false)       # BiasedMF
        fit!(m_w, X); predict(m_w, X)
        m_i = CachedALS(rank=4, max_iter=2, verbose=false); fit!(m_i, X); recommend(m_i, X; k=3)
        fit!(ElementwiseALS(rank=4, max_iter=2, verbose=false), X)
        m_b = PairwiseRanking(rank=4, max_iter=2, verbose=false); fit!(m_b, X); recommend(m_b, X; k=3)

        # ── Item similarity / neighbors ──
        m_e = ShallowAutoencoder(λ=100.0, verbose=false); fit!(m_e, X); recommend(m_e, X; k=3)
        m_s = SparseLinearModel(λ_l1=0.1, λ_l2=0.5, max_iter=3, verbose=false); fit!(m_s, X); recommend(m_s, X; k=3)
        m_a = SparseLinearADMM(λ_l1=0.1, λ_l2=100.0, max_iter=3, verbose=false); fit!(m_a, X); recommend(m_a, X; k=3)
        m_k = ItemKNN(k=3, similarity=:cosine, verbose=false); fit!(m_k, X); recommend(m_k, X; k=3)
        m_r = GraphRandomWalk(k=3, verbose=false); fit!(m_r, X); recommend(m_r, X; k=3)

        # ── Embeddings ──
        C = sprand(rng, 6, 6, 0.5); C = C + C'
        fit!(GlobalVectors(rank=4, max_iter=2, verbose=false), C)

        # ── Completion ──
        fit!(SoftImpute(rank=3, max_iter=3, verbose=false), X)
        fit!(SoftSVD(rank=3, max_iter=3, verbose=false), X)

        # ── Explicit rating predictors ──
        m_o = BaselineOnly(max_iter=2, verbose=false); fit!(m_o, X); predict(m_o, X)
        m_so = SlopeOne(verbose=false); fit!(m_so, X); predict(m_so, X)
        m_pk = PearsonKNN(k=3, verbose=false); fit!(m_pk, X); predict(m_pk, X)

        # ── Sparse regression ──
        m_f = FTRL(lr=0.1, max_iter=1, verbose=false); update!(m_f, X, y); predict(m_f, X)
        m_fm = FactorizationMachine(rank=2, max_iter=2, verbose=false); fit!(m_fm, X, y); predict(m_fm, X)

        # ── Experimental ──
        fit!(Experimental.LogisticMF(rank=4, max_iter=2, verbose=false), X)
        fit!(Experimental.ProbabilisticMF(rank=4, max_iter=2, verbose=false), X)

        # ── Metrics / cross-validation / Tables ──
        p = Matrix{Int}(hcat([randperm(rng, 6)[1:3] for _ in 1:8]...)')
        a = sprand(rng, 8, 6, 0.2)
        mean_ap_at_k(p, a; k=3); ndcg_at_k(p, a; k=3); precision_at_k(p, a; k=3); recall_at_k(p, a; k=3)
        rmse(rand(rng, 8, 6), a)
        random_holdout(X; test_fraction=0.3)
        to_csr(X)
    end
end
