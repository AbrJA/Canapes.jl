# test/test_lmf.jl — LogisticMF algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    model = LogisticMF(rank=5, λ=0.01, α=1.0, learning_rate=0.01, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)

    @test model.is_fitted
    @test size(model.user_factors) == (5, 80)
    @test size(model.item_factors) == (5, 60)
    @test all(isfinite, model.user_factors)
    @test all(isfinite, model.item_factors)
end

@testset "predict returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    model = LogisticMF(rank=5, λ=0.01, learning_rate=0.01, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)
    preds = recommend(model, X; k=5)
    @test size(preds) == (80, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 60)
end

@testset "Negative sampling" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)

    m_low = LogisticMF(rank=5, n_negative=1, max_iter=10, learning_rate=0.01, verbose=false)
    m_high = LogisticMF(rank=5, n_negative=8, max_iter=10, learning_rate=0.01, verbose=false)
    fit!(m_low, X; rng=MersenneTwister(1))
    fit!(m_high, X; rng=MersenneTwister(1))

    # Both should produce valid results
    @test all(isfinite, m_low.user_factors)
    @test all(isfinite, m_high.user_factors)
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 60, 50, 0.08)
    model = LogisticMF(rank=5, λ=0.01, learning_rate=0.01, max_iter=100,
                convergence_tol=0.001, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
end

@testset "Edge case: very sparse matrix" begin
    X = sparse([1, 2], [1, 2], [1.0, 1.0], 100, 100)
    model = LogisticMF(rank=3, max_iter=5, learning_rate=0.01, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test all(isfinite, model.user_factors)
end

@testset "Negative samples exclude observed entries" begin
    X = sparse([1, 1, 2], [1, 2, 1], [1.0, 1.0, 1.0], 2, 3)
    X_csr = Gideon.to_csr(X)
    pool = Int32[1, 2, 1, 2, 1]
    for _ in 1:100
        item = Gideon._lmf_sample_unobserved(MersenneTwister(1), 3, X_csr, 1, pool, length(pool))
        @test item == 3
    end
end
