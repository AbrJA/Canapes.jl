# test/test_item_knn.jl — ItemKNN algorithm tests

@testset "Basic fit (cosine)" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)
    model = ItemKNN(k=5, similarity=:cosine, verbose=false)
    fit!(model, X)

    @test model.is_fitted
    @test size(model.W) == (20, 20)
    # Diagonal should be zero (no self-similarity)
    for j in 1:20
        @test model.W[j, j] == 0.0
    end
    # Should have at most k entries per column
    for j in 1:20
        col_nnz = length(nzrange(model.W, j))
        @test col_nnz <= 5
    end
end

@testset "Basic fit (jaccard)" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)
    model = ItemKNN(k=5, similarity=:jaccard, verbose=false)
    fit!(model, X)

    @test model.is_fitted
    @test size(model.W) == (20, 20)
    for j in 1:20
        @test model.W[j, j] == 0.0
    end
end

@testset "recommend returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.15)
    model = ItemKNN(k=5, similarity=:cosine, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=5)

    @test size(preds) == (40, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 20)
end

@testset "score returns sparse matrix" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 15, 0.2)
    model = ItemKNN(k=4, similarity=:cosine, verbose=false)
    fit!(model, X)
    S = score(model, X)

    @test S isa SparseMatrixCSC
    @test size(S) == (30, 15)
end

@testset "Larger k → denser W" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.2)

    m_small = ItemKNN(k=3, similarity=:cosine, verbose=false)
    m_large = ItemKNN(k=10, similarity=:cosine, verbose=false)
    fit!(m_small, X)
    fit!(m_large, X)

    @test nnz(m_small.W) <= nnz(m_large.W)
end

@testset "Normalization effect" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 15, 0.2)

    m_norm = ItemKNN(k=5, normalize=true, verbose=false)
    m_raw = ItemKNN(k=5, normalize=false, verbose=false)
    fit!(m_norm, X)
    fit!(m_raw, X)

    # Both should have same sparsity pattern but different values
    @test nnz(m_norm.W) == nnz(m_raw.W)
    @test !(nonzeros(m_norm.W) ≈ nonzeros(m_raw.W))
end

@testset "Shrinkage regularizes rare items" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 15, 0.2)

    m_no_shrink = ItemKNN(k=5, shrinkage=0.0, verbose=false)
    m_shrink = ItemKNN(k=5, shrinkage=10.0, verbose=false)
    fit!(m_no_shrink, X)
    fit!(m_shrink, X)

    # Shrinkage should reduce similarity magnitudes
    @test sum(abs, nonzeros(m_shrink.W)) <= sum(abs, nonzeros(m_no_shrink.W))
end

@testset "Invalid parameters" begin
    @test_throws ArgumentError ItemKNN(k=0)
    @test_throws ArgumentError ItemKNN(similarity=:pearson)
    @test_throws ArgumentError ItemKNN(shrinkage=-1.0)
end

@testset "Deterministic output" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.15)

    m1 = ItemKNN(k=5, verbose=false)
    m2 = ItemKNN(k=5, verbose=false)
    fit!(m1, X)
    fit!(m2, X)

    @test m1.W ≈ m2.W
end

@testset "Asymmetric cosine & BM25 similarities" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 60, 40, 0.08)
    for sim in (:asym_cosine, :bm25)
        m = ItemKNN(k=15, similarity=sim, verbose=false)
        fit!(m, X)
        @test size(m.W) == (40, 40)
        @test all(isfinite, nonzeros(m.W))
        @test all(nonzeros(m.W) .>= 0)
        @test size(recommend(m, X; k=5)) == (60, 5)
        # score equals the dense product
        @test Matrix(score(m, X)) ≈ Matrix(X * Matrix(m.W))
    end

    # asym_cosine(α = 0.5) on binary data equals cosine exactly:
    # cosine denom is ‖i‖·‖j‖ = √deg_i·√deg_j for 0/1 vectors
    X_b = copy(X); nonzeros(X_b) .= 1.0
    m_ac = ItemKNN(k=15, similarity=:asym_cosine, asym_alpha=0.5, verbose=false)
    fit!(m_ac, X_b)
    m_co = ItemKNN(k=15, similarity=:cosine, shrinkage=0.0, verbose=false)
    fit!(m_co, X_b)
    @test m_ac.W ≈ m_co.W

    @test_throws ArgumentError ItemKNN(similarity=:asym_cosine, asym_alpha=0.0)
    @test_throws ArgumentError ItemKNN(similarity=:asym_cosine, asym_alpha=1.5)
    @test_throws ArgumentError ItemKNN(similarity=:bm25, k1=0.0)
    @test_throws ArgumentError ItemKNN(similarity=:bm25, b=1.5)
end
