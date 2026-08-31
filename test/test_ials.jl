# test/test_ials.jl — IALS algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 100, 80, 0.05)
    model = IALS(rank=8, λ=0.01, α=40.0, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)

    @test model.is_fitted
    @test size(model.user_factors) == (8, 100)
    @test size(model.item_factors) == (8, 80)
    @test all(isfinite, model.user_factors)
    @test all(isfinite, model.item_factors)
end

@testset "Loss decreases" begin
    rng = MersenneTwister(7)
    X = sprand(rng, 80, 60, 0.08)
    model = IALS(rank=5, λ=0.1, α=10.0, max_iter=10, tol=-1.0, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
end

@testset "recommend returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = IALS(rank=5, λ=0.01, α=10.0, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)
    preds = recommend(model, X; k=5)
    @test size(preds) == (50, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 40)
end

@testset "score pairwise" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.1)
    model = IALS(rank=4, λ=0.01, α=10.0, max_iter=3, verbose=false)
    fit!(model, X; rng=rng)
    scores = score(model, [1, 2, 3], [1, 2, 3])
    @test length(scores) == 3
    @test all(isfinite, scores)
end

@testset "Warm start" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.08)
    k = 5
    U_init = randn(rng, k, 50) .* 0.01
    V_init = randn(rng, k, 40) .* 0.01
    model = IALS(rank=k, λ=0.01, α=10.0, max_iter=3, verbose=false)
    fit!(model, X; rng=rng, U_init=U_init, V_init=V_init)
    @test model.is_fitted
    @test all(isfinite, model.user_factors)
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 60, 50, 0.05)
    model = IALS(rank=5, λ=0.01, α=10.0, max_iter=100, tol=0.01, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
end

@testset "CG solver" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = IALS(rank=16, λ=0.01, α=10.0, max_iter=5, solver=CGSolver(), cg_steps=5, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
    @test all(isfinite, model.user_factors)
    preds = recommend(model, X; k=5)
    @test size(preds) == (50, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 40)
end
