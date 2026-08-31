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
    @testset "GPU fit parity vs CPU" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 100, 80, 0.05)

        # EASE (fully on GPU)
        me = EASE(λ=100.0, verbose=false)
        fit_gpu!(me, X)
        mec = EASE(λ=100.0, verbose=false)
        fit!(mec, X)
        @test me.B ≈ mec.B atol=1e-6

        # IALS (Gramian on GPU, solves on CPU)
        mi = IALS(rank=8, max_iter=5, α=10.0, verbose=false)
        fit_gpu!(mi, X; rng=MersenneTwister(1))
        mic = IALS(rank=8, max_iter=5, α=10.0, verbose=false)
        fit!(mic, X; rng=MersenneTwister(1))
        @test mi.user_factors ≈ mic.user_factors atol=1e-4
        @test mi.item_factors ≈ mic.item_factors atol=1e-4
        @test mi.user_factors' * mi.item_factors ≈
              mic.user_factors' * mic.item_factors atol=1e-4

        # WMF Cholesky
        mw = WMF(rank=8, max_iter=5, solver=CholeskySolver(), verbose=false)
        fit_gpu!(mw, X; rng=MersenneTwister(1))
        mwc = WMF(rank=8, max_iter=5, solver=CholeskySolver(), verbose=false)
        fit!(mwc, X; rng=MersenneTwister(1))
        @test mw.user_factors ≈ mwc.user_factors atol=1e-4
        @test mw.item_factors ≈ mwc.item_factors atol=1e-4
        @test mw.user_factors' * mw.item_factors ≈
              mwc.user_factors' * mwc.item_factors atol=1e-4

        # WMF NonNegativeSolver
        mn = WMF(rank=4, max_iter=5, solver=NonNegativeSolver(), verbose=false)
        fit_gpu!(mn, X; rng=MersenneTwister(2))
        mnc = WMF(rank=4, max_iter=5, solver=NonNegativeSolver(), verbose=false)
        fit!(mnc, X; rng=MersenneTwister(2))
        @test mn.user_factors ≈ mnc.user_factors atol=1e-4
        @test mn.item_factors ≈ mnc.item_factors atol=1e-4
    end

    @testset "GPU input contracts" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.1)
        X_bad = copy(X)
        nonzeros(X_bad)[1] = NaN

        # Unfitted models are rejected by scoring paths
        m_uf = IALS(rank=4, max_iter=3, verbose=false)
        @test_throws ArgumentError score_gpu(m_uf, X)
        @test_throws ArgumentError recommend_gpu(m_uf, X)

        # NaN and empty inputs are rejected by fit_gpu! and leave the model
        # unfitted (transactional)
        for m in (EASE(λ=100.0, verbose=false),
                  IALS(rank=4, max_iter=1, verbose=false),
                  WMF(rank=4, max_iter=1, verbose=false))
            @test_throws ArgumentError fit_gpu!(m, X_bad)
            @test !m.is_fitted
            @test_throws ArgumentError fit_gpu!(m, spzeros(0, 0))
            @test !m.is_fitted
        end

        # A failed GPU refit keeps the previously fitted state intact
        m = WMF(rank=4, max_iter=2, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        U_before = copy(m.user_factors)
        @test_throws ArgumentError fit_gpu!(m, X_bad)
        @test m.is_fitted
        @test m.user_factors == U_before

        # Unsupported solvers fail loudly instead of silently using Cholesky
        @test_throws ArgumentError fit_gpu!(
            IALS(rank=4, solver=CGSolver(), max_iter=1, verbose=false), X)
        @test_throws ArgumentError fit_gpu!(
            WMF(rank=4, solver=CGSolver(), max_iter=1, verbose=false), X)

        # Dimension mismatch on a fitted model
        m = IALS(rank=4, max_iter=2, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        X_wide = sprand(rng, 50, 40, 0.1)
        @test_throws DimensionMismatch score_gpu(m, X_wide)
        @test_throws DimensionMismatch recommend_gpu(m, X_wide)

        # k clamps to n_items
        @test size(recommend_gpu(m, X; k=1000)) == (50, 30)
    end

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
