# test/test_memory_limits.jl — Fit-time memory estimation and limits (Item 6)

# Dense n_items² matrices counted in each model's peak estimate (see fit!)
const _EASE_MATRICES = 4
const _SLIM_MATRICES = 2

@testset "Fit-time memory guard" begin
    @testset "estimate helper" begin
        est32 = Canapes._fit_memory_estimate(20, 4, Float32)
        est64 = Canapes._fit_memory_estimate(20, 4, Float64)
        @test est32 == 20 * 20 * 4 * sizeof(Float32)
        @test est64 == 2 * est32
        @test Canapes._fit_memory_estimate(0, 4, Float32) == 0
    end

    rng = MersenneTwister(42)
    X_small = sprand(rng, 50, 10, 0.2)
    X_large = sprand(rng, 50, 20, 0.2)

    @testset "ShallowAutoencoder" begin
        # Default: unlimited.
        m = ShallowAutoencoder(λ=100.0, verbose=false)
        @test m.max_memory === nothing
        fit!(m, X_small)
        @test m.is_fitted

        # Limit above the estimate → fits; equality with the estimate fits too.
        est = Canapes._fit_memory_estimate(10, _EASE_MATRICES, Float32)
        m = ShallowAutoencoder(λ=100.0, verbose=false, max_memory=est + 1000)
        fit!(m, X_small)
        @test m.is_fitted
        m = ShallowAutoencoder(λ=100.0, verbose=false, max_memory=est)
        fit!(m, X_small)
        @test m.is_fitted

        # Tiny limit → early ArgumentError, model stays unfitted.
        m = ShallowAutoencoder(λ=100.0, verbose=false, max_memory=1024)
        @test_throws ArgumentError fit!(m, X_small)
        @test !m.is_fitted

        # Transactional: a refit that exceeds the limit keeps prior state.
        m = ShallowAutoencoder(λ=100.0, verbose=false,
                 max_memory=Canapes._fit_memory_estimate(10, _EASE_MATRICES, Float32))
        fit!(m, X_small)
        B_before = copy(m.B)
        @test_throws ArgumentError fit!(m, X_large)  # 20 items exceeds the 10-item limit
        @test m.is_fitted
        @test m.B == B_before

        # Constructor validation.
        @test_throws ArgumentError ShallowAutoencoder(max_memory=0)
        @test_throws ArgumentError ShallowAutoencoder(max_memory=-5)
    end

    @testset "SparseLinearModel" begin
        m = SparseLinearModel(max_iter=5, verbose=false)
        @test m.max_memory === nothing

        est = Canapes._fit_memory_estimate(10, _SLIM_MATRICES, Float32)
        m = SparseLinearModel(max_iter=5, verbose=false, max_memory=est)
        fit!(m, X_small)
        @test m.is_fitted

        m = SparseLinearModel(max_iter=5, verbose=false, max_memory=100)
        @test_throws ArgumentError fit!(m, X_small)
        @test !m.is_fitted

        @test_throws ArgumentError SparseLinearModel(max_memory=0)
    end

    @testset "SparseLinearADMM" begin
        m = SparseLinearADMM(max_iter=5, verbose=false)
        @test m.max_memory === nothing

        est = Canapes._fit_memory_estimate(10, Canapes._ADMMSLIM_DENSE_MATRICES, Float32)
        m = SparseLinearADMM(max_iter=5, verbose=false, max_memory=est)
        fit!(m, X_small)
        @test m.is_fitted

        m = SparseLinearADMM(max_iter=5, verbose=false, max_memory=1024)
        @test_throws ArgumentError fit!(m, X_small)
        @test !m.is_fitted

        @test_throws ArgumentError SparseLinearADMM(max_memory=-1)
    end

    @testset "guard message mentions the limit" begin
        m = ShallowAutoencoder(λ=100.0, verbose=false, max_memory=1)
        err = try
            fit!(m, X_small)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("max_memory=1", sprint(showerror, err))
        @test occursin("ShallowAutoencoder", sprint(showerror, err))
    end
end
