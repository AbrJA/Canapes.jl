# test/test_admm_slim.jl — ADMM-SLIM algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)
    model = ADMMSLIM(λ_1=0.01, λ_2=100.0, max_iter=30, verbose=false)
    fit!(model, X)

    @test model.is_fitted
    @test size(model.W) == (20, 20)
    # Diagonal should be zero
    for j in 1:20
        @test abs(model.W[j, j]) < 1e-10
    end
end

@testset "Non-negativity constraint" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 15, 0.2)
    model = ADMMSLIM(λ_1=0.01, λ_2=100.0, nonneg=true, max_iter=30, verbose=false)
    fit!(model, X)
    # All weights should be non-negative
    @test all(model.W .>= -1e-10)
end

@testset "Sparsity with L1" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)

    m_sparse = ADMMSLIM(λ_1=0.5, λ_2=100.0, max_iter=50, verbose=false)
    m_dense = ADMMSLIM(λ_1=0.001, λ_2=100.0, max_iter=50, verbose=false)
    fit!(m_sparse, X)
    fit!(m_dense, X)

    nnz_sparse = count(!iszero, m_sparse.W)
    nnz_dense = count(!iszero, m_dense.W)
    @test nnz_sparse <= nnz_dense
end

@testset "recommend returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.15)
    model = ADMMSLIM(λ_1=0.01, λ_2=100.0, max_iter=30, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=5)

    @test size(preds) == (40, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 20)
end

@testset "score returns dense matrix" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 15, 0.2)
    model = ADMMSLIM(λ_1=0.01, λ_2=100.0, max_iter=20, verbose=false)
    fit!(model, X)
    S = score(model, X)

    @test S isa Matrix{Float32}
    @test size(S) == (30, 15)
    @test all(isfinite, S)
end

@testset "Higher λ_2 → smaller weights" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)

    m_low = ADMMSLIM(λ_1=0.01, λ_2=10.0, max_iter=50, verbose=false)
    m_high = ADMMSLIM(λ_1=0.01, λ_2=1000.0, max_iter=50, verbose=false)
    fit!(m_low, X)
    fit!(m_high, X)

    @test sum(abs2, m_high.W) < sum(abs2, m_low.W)
end

@testset "Converges to SLIM-like solution" begin
    # On a small problem, ADMM-SLIM and SLIM should produce similar W
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 10, 0.25)

    m_slim = SLIM(λ_1=0.05, λ_2=1.0, max_iter=200, nonneg=true, verbose=false)
    m_admm = ADMMSLIM(λ_1=0.05, λ_2=1.0, ρ=1.0, max_iter=200, nonneg=true, verbose=false)
    fit!(m_slim, X)
    fit!(m_admm, X)

    # Solutions should be qualitatively similar (same sign pattern at least)
    W_slim = Matrix(m_slim.W)
    W_admm = m_admm.W

    # Correlation between the two weight matrices should be high
    v1 = vec(W_slim)
    v2 = vec(W_admm)
    # Remove diagonal from comparison
    mask = [i != j for i in 1:10, j in 1:10] |> vec
    v1m = v1[mask]
    v2m = v2[mask]
    corr = dot(v1m .- mean(v1m), v2m .- mean(v2m)) /
           (norm(v1m .- mean(v1m)) * norm(v2m .- mean(v2m)) + 1e-12)
    @test corr > 0.7
end

@testset "Deterministic output" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 15, 0.2)

    m1 = ADMMSLIM(λ_1=0.01, λ_2=100.0, max_iter=20, verbose=false)
    m2 = ADMMSLIM(λ_1=0.01, λ_2=100.0, max_iter=20, verbose=false)
    fit!(m1, X)
    fit!(m2, X)

    @test m1.W ≈ m2.W
end

@testset "Invalid parameters" begin
    @test_throws ArgumentError ADMMSLIM(λ_1=-0.1)
    @test_throws ArgumentError ADMMSLIM(λ_2=-1.0)
    @test_throws ArgumentError ADMMSLIM(ρ=0.0)
end
