# test/test_explicit.jl — explicit subsystem: contract (AbstractExplicitModel)
# and error metrics (rmse / mae)

@testset "Explicit contract" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.25)
    nonzeros(X) .= 1.0 .+ 4.0 .* rand(MersenneTwister(3), nnz(X))   # ratings in [1, 5]
    m = SoftImpute(rank=5, λ=0.5, max_iter=20, verbose=false)
    fit!(m, X; rng=rng)
    @test m isa AbstractExplicitModel
    @test m isa AbstractSoftALS
    @test !(m isa AbstractRecommender)           # completion is not a ranking model
    @test !(m isa AbstractMatrixFactorization)
    @test !applicable(recommend, m, X)          # no top-k contract
    @test applicable(score, m, X)
    @test applicable(predict, m, X)
end

@testset "Completion score/predict" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.25)
    nonzeros(X) .= 1.0 .+ 4.0 .* rand(MersenneTwister(3), nnz(X))
    for M in (SoftImpute(rank=5, λ=0.5, max_iter=20, verbose=false),
              SoftSVD(rank=5, max_iter=20, verbose=false),
              PureSVD(rank=5, max_iter=20, verbose=false))
        fit!(M, X; rng=rng)
        S = score(M, X)
        @test size(S) == size(X)
        @test all(isfinite, S)
        @test predict(M, X) == S
        # reconstruction equals U D V'
        @test S ≈ M.U * Diagonal(M.d) * M.V'
        # pairwise scoring
        pv = score(M, [1, 5, 20], [2, 7, 19])
        @test length(pv) == 3
        @test all(isfinite, pv)
    end
    # dimension guard
    m = SoftImpute(rank=2, max_iter=3, verbose=false)
    fit!(m, X; rng=rng)
    @test_throws DimensionMismatch score(m, sprand(rng, 5, 10, 0.2))
end

@testset "RMSE / MAE" begin
    actual = sparse([1,1,2,3], [1,2,1,4], [4.0,2.0,5.0,3.0], 3, 4)
    preds = fill(3.0, 3, 4)
    @test rmse(preds, actual) ≈ sqrt(6/4)
    @test mae(preds, actual) ≈ 4/4
    @test mean_rmse(preds, actual) == rmse(preds, actual)
    @test mean_mae(preds, actual) == mae(preds, actual)

    # perfect predictions → zero error
    actual2 = sparse([1,2,2], [1,1,3], [5.0,2.0,4.0], 2, 3)
    exact = zeros(2, 3); exact[1,1] = 5.0; exact[2,1] = 2.0; exact[2,3] = 4.0
    @test rmse(exact, actual2) == 0.0
    @test mae(exact, actual2) == 0.0

    # only observed entries count (values elsewhere in predictions are ignored)
    big = zeros(3, 4); big[1,1] = 3.0; big[1,2] = 3.0; big[2,1] = 3.0; big[3,4] = 3.0
    @test rmse(big, actual) == rmse(preds, actual)

    # guards
    @test_throws DimensionMismatch rmse(zeros(2, 4), actual)
    @test_throws ArgumentError rmse(fill(NaN, 3, 4), actual)
    @test_throws ArgumentError rmse(fill(3.0, 3, 4), sparse(Int[], Int[], Float64[], 3, 4))
    @test_throws ArgumentError mae(fill(Inf, 3, 4), actual)
end

@testset "Evaluation workflow: holdout + rmse" begin
    rng = MersenneTwister(7)
    X = sprand(rng, 40, 25, 0.2)
    nonzeros(X) .= 1.0 .+ 4.0 .* rand(MersenneTwister(9), nnz(X))
    X_tr, X_te = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))
    m = SoftImpute(rank=6, λ=0.5, max_iter=30, verbose=false)
    fit!(m, X_tr; rng=rng)
    S = score(m, X_tr)
    @test rmse(S, X_te) >= 0.0
    @test mae(S, X_te) >= 0.0
    @test all(isfinite, (rmse(S, X_te), mae(S, X_te)))
end