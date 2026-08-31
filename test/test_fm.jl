# test/test_fm.jl — Factorization Machine tests

@testset "XOR: rank-2 FM learns interaction" begin
    x = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
    y_xor = [0.0, 1.0, 1.0, 0.0]

    n_correct = 0
    for seed in 1:5
        fm = FM(
            lr_w=10.0, rank=2, max_iter=200,
            λ_w=0.0, λ_v=0.0, family=Binomial(), intercept=true, verbose=false)
        fit!(fm, x, y_xor; rng=MersenneTwister(seed))
        preds = predict(fm, x)
        n_correct += (preds[1] < 0.3 && preds[2] > 0.7 && preds[3] > 0.7 && preds[4] < 0.3)
    end
    @test n_correct >= 4
end

@testset "Gaussian family" begin
    rng = MersenneTwister(42)
    Xg = sprand(rng, 100, 20, 0.3)
    yg = randn(rng, 100)
    mse(m) = sum((predict(m, Xg) .- yg).^2) / length(yg)

    fm = FM(rank=5, family=Gaussian(), lr_w=0.01, max_iter=50, verbose=false)
    fit!(fm, Xg, yg; rng=MersenneTwister(1))
    preds = predict(fm, Xg)
    @test length(preds) == 100
    @test all(isfinite, preds)
end

@testset "MSE decreases with iterations (Gaussian)" begin
    rng = MersenneTwister(42)
    Xg = sprand(rng, 100, 20, 0.3)
    yg = randn(rng, 100)
    mse(m) = sum((predict(m, Xg) .- yg).^2) / length(yg)

    m5 = FM(rank=5, family=Gaussian(), lr_w=0.01, max_iter=5, verbose=false)
    m50 = FM(rank=5, family=Gaussian(), lr_w=0.01, max_iter=50, verbose=false)
    fit!(m5, Xg, yg; rng=MersenneTwister(1))
    fit!(m50, Xg, yg; rng=MersenneTwister(1))
    @test mse(m50) < mse(m5)
end

@testset "Regularization reduces overfitting" begin
    rng = MersenneTwister(42)
    X_train = sprand(rng, 50, 20, 0.3)
    y_train = randn(rng, 50)
    X_test = sprand(rng, 30, 20, 0.3)
    y_test = randn(rng, 30)

    fm_noreg = FM(rank=5, family=Gaussian(), lr_w=0.01,
                                     λ_w=0.0, λ_v=0.0, max_iter=100, verbose=false)
    fm_reg = FM(rank=5, family=Gaussian(), lr_w=0.01,
                                   λ_w=0.1, λ_v=0.1, max_iter=100, verbose=false)
    fit!(fm_noreg, X_train, y_train; rng=MersenneTwister(1))
    fit!(fm_reg, X_train, y_train; rng=MersenneTwister(1))

    # Regularized model should have smaller factor norms
    @test sum(abs2, fm_reg.V) < sum(abs2, fm_noreg.V)
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    Xg = sprand(rng, 100, 20, 0.3)
    yg = randn(rng, 100)

    fm = FM(rank=5, family=Gaussian(), lr_w=0.01,
                               tol=0.001, max_iter=200, verbose=false)
    fit!(fm, Xg, yg; rng=MersenneTwister(1))
    @test fm.is_initialized
end

@testset "Edge cases" begin
    # Single feature
    x1 = sparse(ones(10, 1))
    y1 = Float64.([1,0,1,0,1,0,1,0,1,0])
    fm = FM(rank=2, family=Binomial(), max_iter=10, verbose=false)
    fit!(fm, x1, y1; rng=MersenneTwister(1))
    @test all(isfinite, predict(fm, x1))

    # High-rank with few features
    x2 = sprand(MersenneTwister(1), 20, 3, 0.5)
    y2 = randn(MersenneTwister(1), 20)
    fm2 = FM(rank=10, family=Gaussian(), max_iter=5, verbose=false)
    fit!(fm2, x2, y2; rng=MersenneTwister(1))
    @test all(isfinite, predict(fm2, x2))
end
