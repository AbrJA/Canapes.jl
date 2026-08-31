# test/test_eals.jl — Tests for element-wise ALS

@testset "EALS basics" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 30, 0.1)

    model = EALS(rank=8, λ=0.01, unobserved_weight=1.0, max_iter=10, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))

    @test model.is_fitted
    @test size(model.user_factors) == (8, 50)
    @test size(model.item_factors) == (8, 30)
    @test !any(isnan, model.user_factors)
    @test !any(isnan, model.item_factors)
end

@testset "EALS predict" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 30, 0.1)

    model = EALS(rank=8, λ=0.01, unobserved_weight=1.0, max_iter=5, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))

    preds = recommend(model, X; k=5)
    @test size(preds) == (50, 5)
    @test all(p -> 1 <= p <= 30, preds)

    # score
    scores = score(model, X)
    @test size(scores) == (50, 30)
    @test !any(isnan, scores)
end

@testset "EALS update!" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 30, 0.1)

    model = EALS(rank=4, λ=0.01, unobserved_weight=1.0, max_iter=3, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))

    # Incremental update
    update!(model, X; n_iters=2)
    @test model.is_fitted
    @test !any(isnan, model.user_factors)
end

@testset "EALS popularity weighting" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 30, 0.1)

    model = EALS(rank=4, λ=0.01, unobserved_weight=5.0, popularity_exponent=0.75, max_iter=3, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))

    @test length(model.item_weights) == 30
    @test all(w -> w > 0, model.item_weights)
end

@testset "EALS warm start" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.15)

    U_init = Float32.(randn(rng, 4, 30) .* 0.01)
    V_init = Float32.(randn(rng, 4, 20) .* 0.01)

    model = EALS(rank=4, max_iter=3, verbose=false)
    fit!(model, X; U_init=U_init, V_init=V_init, rng=MersenneTwister(2))
    @test model.is_fitted
end
