# test/test_ftrl.jl — FTRL algorithm tests

rng = MersenneTwister(42)
n, p = 500, 100
X = sprand(rng, n, p, 0.1)
w_true = zeros(p); w_true[1:5] .= 1.0
y = Float64.(vec(X * w_true) .> 0.5)

logloss(pred, label) = -sum(
    label .* log.(pred .+ 1e-10) .+
    (1 .- label) .* log.(1 .- pred .+ 1e-10)) / length(label)

@testset "Basic fit and predict (binomial)" begin
    model = FTRL(lr=0.1, λ=0.01, l1_ratio=0.5, family=Binomial(), max_iter=5, verbose=false)
    fit!(model, X, y)
    @test model.is_initialized
    @test model.n_features == p
    w = coef(model)
    @test length(w) == p
    @test !any(isnan, w)
    preds = predict(model, X)
    @test length(preds) == n
    @test all(0 .<= preds .<= 1)
end

@testset "Loss decreases with more epochs" begin
    m1 = FTRL(lr=0.1, λ=0.01, l1_ratio=0.5, verbose=false)
    update!(m1, X, y; rng=MersenneTwister(1))
    m5 = FTRL(lr=0.1, λ=0.01, l1_ratio=0.5, verbose=false)
    for _ in 1:5; update!(m5, X, y; rng=MersenneTwister(1)); end
    @test logloss(predict(m5, X), y) < logloss(predict(m1, X), y)
end

@testset "L1 regularization zeroes weights" begin
    m_l1 = FTRL(lr=0.1, λ=2.0, l1_ratio=1.0, verbose=false)
    for _ in 1:5; update!(m_l1, X, y); end
    nnz_w = sum(abs.(coef(m_l1)) .> 1e-10)
    @test nnz_w < p
end

@testset "update! updates weights" begin
    model2 = FTRL(lr=0.1, verbose=false)
    update!(model2, X, y)
    w1 = copy(coef(model2))
    update!(model2, X, y)
    w2 = coef(model2)
    @test w1 != w2
end

@testset "Gaussian family" begin
    y_cont = randn(MersenneTwister(1), n)
    model = FTRL(lr=0.01, λ=0.001, family=Gaussian(), max_iter=3, verbose=false)
    fit!(model, X, y_cont)
    preds = predict(model, X)
    @test length(preds) == n
    @test all(isfinite, preds)
    # Predictions should be real-valued (not bounded to [0,1])
    @test any(preds .< 0) || any(preds .> 1)
end

@testset "Poisson family" begin
    y_count = Float64.(rand(MersenneTwister(1), 0:5, n))
    model = FTRL(lr=0.01, λ=0.001, family=Poisson(), max_iter=3, verbose=false)
    fit!(model, X, y_count)
    preds = predict(model, X)
    @test length(preds) == n
    @test all(preds .> 0)  # Poisson link ensures positive
    @test all(isfinite, preds)
end

@testset "Gradient clipping" begin
    # With very large features, gradient clipping should prevent NaN
    X_large = sprand(MersenneTwister(1), 100, 50, 0.1) .* 1000.0
    y_bin = Float64.(rand(MersenneTwister(1), [0, 1], 100))
    model = FTRL(lr=0.1, grad_clip=10.0, max_iter=3, verbose=false)
    fit!(model, X_large, y_bin)
    @test all(isfinite, coef(model))
    @test all(isfinite, predict(model, X_large))
end

@testset "Dropout" begin
    model = FTRL(lr=0.1, dropout=0.3, max_iter=3, verbose=false)
    fit!(model, X, y; rng=MersenneTwister(42))
    preds = predict(model, X)
    @test all(0 .<= preds .<= 1)
end

@testset "Edge cases" begin
    # Single sample
    X_single = sparse([1.0 0.0 2.0])
    y_single = [1.0]
    model = FTRL(lr=0.1, verbose=false)
    update!(model, X_single, y_single)
    @test model.is_initialized
    @test length(coef(model)) == 3

    # All zeros target
    y_zero = zeros(n)
    model2 = FTRL(lr=0.1, max_iter=2, verbose=false)
    fit!(model2, X, y_zero)
    preds = predict(model2, X)
    @test all(preds .< 0.5)  # should predict low for all-zero target
end
