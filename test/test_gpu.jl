# test/test_gpu.jl — Tests for GPU extension (skipped if CUDA unavailable)
#
# These tests verify the GPU extension interface. They are only run
# when CUDA.jl is available and a GPU device is detected.

@testset "GPU stubs exist" begin
    # Verify that GPU stub functions are defined even without CUDA
    @test isdefined(Gideon, :fit_gpu!)
    @test isdefined(Gideon, :recommend_gpu)
    @test isdefined(Gideon, :score_gpu)
end

# Only run GPU tests if CUDA is available
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

if HAS_CUDA
    @testset "GPU EASE" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.1)

        model = EASE(λ=100.0, verbose=false)
        fit_gpu!(model, X)

        @test model.is_fitted
        @test size(model.B) == (30, 30)
        @test !any(isnan, model.B)

        # Compare with CPU result
        model_cpu = EASE(λ=100.0, verbose=false)
        fit!(model_cpu, X)
        @test model.B ≈ model_cpu.B atol=1e-4
    end

    @testset "GPU IALS" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.1)

        model = IALS(rank=8, max_iter=3, verbose=false)
        fit_gpu!(model, X; rng=MersenneTwister(1))

        @test model.is_fitted
        @test size(model.user_factors) == (8, 50)
        @test size(model.item_factors) == (8, 30)
    end

    @testset "GPU WMF nonnegative solver" begin
        X = sprand(MersenneTwister(42), 20, 15, 0.1)
        model = WMF(rank=4, max_iter=1, solver=NonNegativeSolver(), verbose=false)
        fit_gpu!(model, X; rng=MersenneTwister(1))
        @test model.is_fitted
        @test all(>=(0), model.user_factors)
        @test all(>=(0), model.item_factors)
    end

    @testset "GPU score" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 30, 20, 0.1)

        model = IALS(rank=4, max_iter=3, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))

        scores_gpu = score_gpu(model, X)
        scores_cpu = model.user_factors' * model.item_factors

        @test size(scores_gpu) == size(scores_cpu)
        @test scores_gpu ≈ scores_cpu atol=1e-5
    end

    @testset "GPU predict top-k" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 30, 20, 0.1)

        model = IALS(rank=4, max_iter=3, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))

        preds = recommend_gpu(model, X; k=5)
        @test size(preds) == (30, 5)
        @test all(p -> 1 <= p <= 20, preds)
    end
else
    @info "CUDA not available — skipping GPU tests"
end
