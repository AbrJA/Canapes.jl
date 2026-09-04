# test/test_sparselinearmodel.jl — SparseLinearModel algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)
    model = SparseLinearModel(λ_l1=0.01, λ_l2=0.1, max_iter=20, verbose=false)
    fit!(model, X)

    @test model.is_fitted
    @test size(model.W) == (20, 20)
    # Diagonal should be zero
    for j in 1:20
        @test model.W[j, j] == 0.0
    end
end

@testset "Non-negativity constraint" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 15, 0.2)
    model = SparseLinearModel(λ_l1=0.01, λ_l2=0.1, nonnegative=true, max_iter=30, verbose=false)
    fit!(model, X)
    # All weights should be non-negative
    @test all(nonzeros(model.W) .>= 0.0)
end

@testset "Sparsity with L1" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)

    m_sparse = SparseLinearModel(λ_l1=0.5, λ_l2=0.1, max_iter=30, verbose=false)
    m_dense = SparseLinearModel(λ_l1=0.001, λ_l2=0.1, max_iter=30, verbose=false)
    fit!(m_sparse, X)
    fit!(m_dense, X)

    @test nnz(m_sparse.W) <= nnz(m_dense.W)
end

@testset "predict returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.15)
    model = SparseLinearModel(λ_l1=0.01, λ_l2=0.1, max_iter=20, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=5)
    @test size(preds) == (40, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 20)
end

@testset "score is sparse" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 15, 0.2)
    model = SparseLinearModel(λ_l1=0.05, λ_l2=0.1, max_iter=20, verbose=false)
    fit!(model, X)
    S = score(model, X)
    @test S isa SparseMatrixCSC
end
