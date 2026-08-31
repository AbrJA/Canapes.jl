# test/test_bpr.jl — BPR algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 100, 80, 0.05)
    model = BPR(rank=8, lr=0.05, max_iter=10, verbose=false)
    fit!(model, X; rng=rng)

    @test model.is_fitted
    @test size(model.user_factors) == (8, 100)
    @test size(model.item_factors) == (8, 80)
    @test all(isfinite, model.user_factors)
    @test all(isfinite, model.item_factors)
end

@testset "Loss decreases over epochs" begin
    rng = MersenneTwister(7)
    X = sprand(rng, 80, 60, 0.1)
    model = BPR(rank=8, lr=0.05, max_iter=20, verbose=false)
    fit!(model, X; rng=rng)
    # Loss should generally decrease
    @test model.loss_history[end] < model.loss_history[1]
end

@testset "predict returns valid indices" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = BPR(rank=5, lr=0.05, max_iter=10, verbose=false)
    fit!(model, X; rng=rng)
    preds = recommend(model, X; k=5)
    @test size(preds) == (50, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 40)
end

@testset "score" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.1)
    model = BPR(rank=4, lr=0.05, max_iter=5, verbose=false)
    fit!(model, X; rng=rng)
    scores = score(model, [1, 2, 3], [1, 2, 3])
    @test length(scores) == 3
    @test all(isfinite, scores)
end

@testset "Custom n_samples" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = BPR(rank=5, lr=0.05, max_iter=5, n_samples=100, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
end

@testset "Regularization parameters" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = BPR(rank=5, λ_user=0.1, λ_pos=0.1, λ_neg=0.001,
                lr=0.05, max_iter=10, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
    @test all(isfinite, model.user_factors)
end

@testset "Popularity-biased negative sampling" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = BPR(rank=5, lr=0.05, max_iter=10,
                negative_sampling=Popular(), verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
    @test model.loss_history[end] < model.loss_history[1]
end

@testset "Dynamic Negative Sampling (DNS)" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.1)
    model = BPR(rank=5, lr=0.01, max_iter=10,
                negative_sampling=Dynamic(), dynamic_candidates=10, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
    @test all(isfinite, model.user_factors)
end

@testset "No negative items" begin
    # Users with all items observed are skipped when other users are eligible.
    X = sparse([1, 1, 2], [1, 2, 1], [1.0, 1.0, 1.0], 2, 2)
    model = BPR(rank=2, max_iter=1, n_samples=10, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    @test model.is_fitted

    # A dataset with no possible BPR triplet fails explicitly.
    X_full = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    @test_throws ArgumentError fit!(BPR(rank=2, max_iter=1, verbose=false), X_full)
    @test_throws ArgumentError fit!(BPR(rank=2, max_iter=1, verbose=false), spzeros(2, 2))
end
