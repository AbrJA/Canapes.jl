# test/test_soft_impute.jl — SoftImpute / SoftSVD tests

@testset "Basic SoftImpute" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.2)
    model = SoftImpute(rank=5, λ=0.1, max_iter=20, verbose=false)
    fit!(model, X; rng=rng)
    @test model isa SoftImpute
    @test model isa AbstractSoftALS
    @test model.is_fitted
    @test size(model.U, 2) <= 5
    @test length(model.d) <= 5
    @test all(model.d .>= 0)
    @test all(isfinite, model.U)
    @test all(isfinite, model.V)

    # Reconstruction should be finite
    recon = model.U * Diagonal(model.d) * model.V'
    @test all(isfinite, recon)
end

@testset "SoftSVD mode" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.2)
    model = SoftSVD(rank=5, λ=0.0, max_iter=20, verbose=false)
    fit!(model, X; rng=rng)
    @test model isa SoftSVD
    @test model isa AbstractSoftALS
    @test all(model.d .>= 0)
end

@testset "Singular values sorted descending" begin
    rng = MersenneTwister(5)
    X = sprand(rng, 60, 50, 0.2)
    model = SoftImpute(rank=5, λ=0.1, max_iter=20, verbose=false)
    fit!(model, X; rng=rng)
    @test issorted(model.d; rev=true)
    @test all(model.d .>= 0)
end

@testset "Low-rank recovery" begin
    # Create a rank-3 matrix with noise
    rng = MersenneTwister(42)
    U_true = randn(rng, 50, 3)
    V_true = randn(rng, 40, 3)
    M_true = U_true * V_true'
    # Sample entries
    mask = sprand(rng, 50, 40, 0.5)
    X_obs = SparseMatrixCSC(mask .!= 0) .* M_true
    # Make it actually sparse
    X_obs = sparse(findnz(X_obs)..., 50, 40)

    model = SoftImpute(rank=3, λ=0.01, max_iter=50, verbose=false)
    fit!(model, X_obs; rng=rng)
    @test length(model.d) <= 3
    @test size(model.U, 1) == 50
    @test size(model.V, 1) == 40
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 25, 0.3)
    model = SoftImpute(rank=5, λ=0.1, max_iter=1000, tol=1e-6, verbose=false)
    fit!(model, X; rng=rng)
    @test all(isfinite, model.d)
end

@testset "λ shrinks singular values" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.2)
    m_low = SoftImpute(rank=5, λ=0.01, max_iter=30, verbose=false)
    m_high = SoftImpute(rank=5, λ=5.0, max_iter=30, verbose=false)
    fit!(m_low, X; rng=rng)
    fit!(m_high, X; rng=rng)
    @test sum(m_high.d) <= sum(m_low.d)
end

@testset "verbose logging with Float32 losses" begin
    # Regression: log_iteration required Float64 while Float32 models pass
    # Float32 losses — verbose fits of a Float32 model must not throw.
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 25, 0.2)
    m = SoftImpute(rank=4, λ=0.1, max_iter=3, verbose=true)
    fit!(m, X; rng=rng)
    @test m.is_fitted
end
