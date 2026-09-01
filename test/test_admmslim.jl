# test/test_admm_slim.jl — ADMM-SLIM algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)
    model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=30, verbose=false)
    fit!(model, X)

    @test model.is_fitted
    @test model.W isa SparseMatrixCSC
    @test size(model.W) == (20, 20)
    # Soft-thresholded exact zeros are dropped (diagonal + L1-thresholded weights)
    @test nnz(model.W) < 20 * 20
    # Diagonal should be zero
    for j in 1:20
        @test abs(model.W[j, j]) < 1e-10
    end
end

@testset "Non-negativity constraint" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 15, 0.2)
    model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, nonnegative=true, max_iter=30, verbose=false)
    fit!(model, X)
    # All weights should be non-negative
    @test all(model.W .>= -1e-10)
end

@testset "Sparsity with L1" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)

    m_sparse = ADMMSLIM(λ_l1=0.5, λ_l2=100.0, max_iter=50, verbose=false)
    m_dense = ADMMSLIM(λ_l1=0.001, λ_l2=100.0, max_iter=50, verbose=false)
    fit!(m_sparse, X)
    fit!(m_dense, X)

    nnz_sparse = nnz(m_sparse.W)
    nnz_dense = nnz(m_dense.W)
    @test nnz_sparse <= nnz_dense
end

@testset "recommend returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.15)
    model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=30, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=5)

    @test size(preds) == (40, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 20)
end

@testset "score returns sparse matrix" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 15, 0.2)
    model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=20, verbose=false)
    fit!(model, X)
    S = score(model, X)

    @test S isa SparseMatrixCSC
    @test size(S) == (30, 15)
    @test all(isfinite, S)
    # Sparse score equals the dense product
    @test Matrix(S) ≈ Matrix(X * Matrix(model.W))
end

@testset "Adaptive recommend path" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.15)

    # Dense-ish W (low λ_l1) → dense batched-GEMM path
    m_dense = ADMMSLIM(λ_l1=0.001, λ_l2=100.0, max_iter=30, verbose=false)
    fit!(m_dense, X)
    @test !Canapes._use_sparse_score_path(m_dense.W, X)
    @test recommend(m_dense, X; k=5) ==
          Canapes._predict_batched_gemm_topk(X, Matrix(m_dense.W), 5)

    # Truly sparse W (high λ_l1) → sparse-score path
    m_sparse = ADMMSLIM(λ_l1=0.5, λ_l2=100.0, max_iter=30, verbose=false)
    fit!(m_sparse, X)
    @test Canapes._use_sparse_score_path(m_sparse.W, X)
    @test recommend(m_sparse, X; k=5) ==
          Canapes._predict_sparse_score_topk(X * m_sparse.W, X, 5)

    # Both paths agree with the dense score product up to FP accumulation order
    for m in (m_dense, m_sparse)
        @test Matrix(score(m, X)) ≈ Matrix(X * Matrix(m.W))
    end
end

@testset "Sparse W survives persistence round-trip" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 15, 0.2)
    model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=20, verbose=false)
    fit!(model, X)

    tmpfile = tempname() * ".jls"
    try
        save_model(model, tmpfile)
        loaded = load_model(tmpfile)
        @test loaded.W isa SparseMatrixCSC
        @test loaded.W == model.W
        @test recommend(loaded, X; k=5) == recommend(model, X; k=5)
    finally
        rm(tmpfile; force=true)
    end
end

@testset "Higher λ_l2 → smaller weights" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 20, 0.15)

    m_low = ADMMSLIM(λ_l1=0.01, λ_l2=10.0, max_iter=50, verbose=false)
    m_high = ADMMSLIM(λ_l1=0.01, λ_l2=1000.0, max_iter=50, verbose=false)
    fit!(m_low, X)
    fit!(m_high, X)

    @test sum(abs2, m_high.W) < sum(abs2, m_low.W)
end

@testset "Converges to SLIM-like solution" begin
    # On a small problem, ADMM-SLIM and SLIM should produce similar W
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 10, 0.25)

    m_slim = SLIM(λ_l1=0.05, λ_l2=1.0, max_iter=200, nonnegative=true, verbose=false)
    m_admm = ADMMSLIM(λ_l1=0.05, λ_l2=1.0, ρ=1.0, max_iter=200, nonnegative=true, verbose=false)
    fit!(m_slim, X)
    fit!(m_admm, X)

    # Solutions should be qualitatively similar (same sign pattern at least)
    W_slim = Matrix(m_slim.W)
    W_admm = Matrix(m_admm.W)

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

    m1 = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=20, verbose=false)
    m2 = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=20, verbose=false)
    fit!(m1, X)
    fit!(m2, X)

    @test m1.W ≈ m2.W
end

@testset "Invalid parameters" begin
    @test_throws ArgumentError ADMMSLIM(λ_l1=-0.1)
    @test_throws ArgumentError ADMMSLIM(λ_l2=-1.0)
    @test_throws ArgumentError ADMMSLIM(ρ=0.0)
end
