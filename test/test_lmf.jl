# test/test_lmf.jl — LogisticMF algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    model = LogisticMF(rank=5, λ=0.01, α=1.0, lr=0.01, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)

    @test model.is_fitted
    @test size(model.user_factors) == (5 + 2, 80)
    @test size(model.item_factors) == (5 + 2, 60)
    @test all(isfinite, model.user_factors)
    @test all(isfinite, model.item_factors)
end

@testset "Bias dims (implicit factors+2 layout)" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    model = LogisticMF(rank=6, λ=0.01, lr=0.01, max_iter=3, verbose=false)
    fit!(model, X; rng=rng)

    # The user's second-to-last column and the item's last column are pinned
    # to 1 so that scores include β_u + β_i (Johnson 2014, eq. 1), matching
    # implicit's lmf.pyx pinning after every phase.
    @test model.user_factors[model.rank + 1, :] == ones(Float32, 80)
    @test model.item_factors[model.rank + 2, :] == ones(Float32, 60)

    # The bias terms actually contribute to scores: the full dot over the
    # rank+2 rows differs from the raw rank dot for some user-item pair.
    kf = model.rank + 2
    diff = maximum(abs(
        sum(model.user_factors[f, u] * model.item_factors[f, j] for f in 1:kf) -
        sum(model.user_factors[f, u] * model.item_factors[f, j] for f in 1:model.rank))
        for u in 1:80, j in 1:60)
    @test diff > 0
end

@testset "α scales positive confidence" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 60, 50, 0.1)
    m1 = LogisticMF(rank=5, λ=0.01, α=1.0, lr=0.01, max_iter=5, verbose=false)
    m2 = LogisticMF(rank=5, λ=0.01, α=2.0, lr=0.01, max_iter=5, verbose=false)
    fit!(m1, X; rng=MersenneTwister(1))
    fit!(m2, X; rng=MersenneTwister(1))
    @test all(isfinite, m1.user_factors)
    @test all(isfinite, m2.user_factors)
    @test m1.user_factors != m2.user_factors
end

@testset "predict returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    model = LogisticMF(rank=5, λ=0.01, lr=0.01, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)
    preds = recommend(model, X; k=5)
    @test size(preds) == (80, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 60)
end

@testset "Negative sampling" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)

    m_low = LogisticMF(rank=5, n_negative=1, max_iter=10, lr=0.01, verbose=false)
    m_high = LogisticMF(rank=5, n_negative=8, max_iter=10, lr=0.01, verbose=false)
    fit!(m_low, X; rng=MersenneTwister(1))
    fit!(m_high, X; rng=MersenneTwister(1))

    # Both should produce valid results
    @test all(isfinite, m_low.user_factors)
    @test all(isfinite, m_high.user_factors)
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 60, 50, 0.08)
    model = LogisticMF(rank=5, λ=0.01, lr=0.01, max_iter=100,
                tol=0.001, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
end

@testset "Edge case: very sparse matrix" begin
    X = sparse([1, 2], [1, 2], [1.0, 1.0], 100, 100)
    model = LogisticMF(rank=3, max_iter=5, lr=0.01, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test all(isfinite, model.user_factors)
end

@testset "Alias sampler reproduces the occurrence-weighted pool distribution" begin
    # P(e) ∝ counts[e] — identical to implicit's flat-observation-pool draw
    # (benfred/implicit#745: popularity-weighted measured better than uniform
    # on MovieLens-100k, so the weighting is kept).
    counts = Int[2, 5, 0, 3]
    p, alias = Canapes.Experimental._lmf_build_alias(counts, Float32)
    rng = MersenneTwister(1)
    freq = zeros(Int, 4)
    for _ in 1:500_000
        freq[Canapes.Experimental._lmf_sample_alias(rng, p, alias)] += 1
    end
    # never draws an entity with zero observations
    @test freq[3] == 0
    # observed entities match their count shares (P@1..4 = 2/10, 5/10, 0, 3/10)
    for i in (1, 2, 4)
        @test freq[i] ≈ 500_000 * counts[i] / 10 rtol=0.02
    end
    # alias table is a valid probability structure
    @test all(0 .<= p .<= 1)
    @test all(1 .<= alias .<= 4)
end

@testset "Empty input" begin
    @test_throws ArgumentError fit!(LogisticMF(rank=2, max_iter=1, verbose=false), spzeros(2, 2))
    @test_throws ArgumentError fit!(LogisticMF(rank=2, max_iter=1, verbose=false), spzeros(0, 2))
end

@testset "RMSProp optimizer" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    m = LogisticMF(rank=5, λ=0.01, lr=0.1, max_iter=5, optimizer=:rmsprop, verbose=false)
    fit!(m, X; rng=MersenneTwister(1))
    @test m.is_fitted
    @test size(m.user_factors) == (5 + 2, 80)
    @test all(isfinite, m.user_factors)
    @test all(isfinite, m.item_factors)
    # moves under a small learning rate where Adagrad stalls
    ma = LogisticMF(rank=5, λ=0.01, lr=0.1, max_iter=5, optimizer=:adagrad, verbose=false)
    fit!(ma, X; rng=MersenneTwister(1))
    @test recommend(m, X; k=5) != recommend(ma, X; k=5) || m.user_factors != ma.user_factors
end

@testset "Optimizer validation" begin
    @test LogisticMF(optimizer=:adagrad) isa LogisticMF
    @test LogisticMF(optimizer=:rmsprop) isa LogisticMF
    @test_throws ArgumentError LogisticMF(optimizer=:sgd)
end
