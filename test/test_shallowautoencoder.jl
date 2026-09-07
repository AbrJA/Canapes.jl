# test/test_shallowautoencoder.jl — ShallowAutoencoder algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 30, 0.1)
    model = ShallowAutoencoder(λ=100.0, verbose=false)
    fit!(model, X)

    @test model.is_fitted
    @test size(model.B) == (30, 30)
    @test all(isfinite, model.B)
    # Diagonal should be zero
    @test all(abs.(diag(model.B)) .< 1e-10)
end

@testset "predict returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 30, 0.1)
    model = ShallowAutoencoder(λ=100.0, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=5)
    @test size(preds) == (50, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 30)
end

@testset "score full matrix" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.1)
    model = ShallowAutoencoder(λ=100.0, verbose=false)
    fit!(model, X)
    S = score(model, X)
    @test size(S) == (30, 20)
    @test all(isfinite, S)
end

@testset "Higher λ → smaller weights" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 25, 0.15)

    m_low = ShallowAutoencoder(λ=10.0, verbose=false)
    m_high = ShallowAutoencoder(λ=1000.0, verbose=false)
    fit!(m_low, X)
    fit!(m_high, X)

    @test sum(abs2, m_high.B) < sum(abs2, m_low.B)
end

@testset "Closed-form is deterministic" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.1)

    m1 = ShallowAutoencoder(λ=200.0, verbose=false)
    m2 = ShallowAutoencoder(λ=200.0, verbose=false)
    fit!(m1, X)
    fit!(m2, X)
    @test m1.B ≈ m2.B
end
