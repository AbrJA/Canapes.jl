# test/test_finite_input.jl — uniform NaN/Inf rejection across all fit! paths (Item 11)

const _FINITE_MODEL_FACTORIES = [
    (name="WMF",     model=() -> WMF(rank=2, max_iter=1, verbose=false), y=false),
    (name="IALS",    model=() -> IALS(rank=2, max_iter=1, verbose=false), y=false),
    (name="EALS",    model=() -> EALS(rank=2, max_iter=1, verbose=false), y=false),
    (name="BPR",     model=() -> BPR(rank=2, max_iter=1, verbose=false), y=false),
    (name="LogisticMF", model=() -> LogisticMF(rank=2, max_iter=1, verbose=false), y=false),
    (name="EASE",    model=() -> EASE(λ=100.0, verbose=false), y=false),
    (name="SLIM",    model=() -> SLIM(max_iter=2, verbose=false), y=false),
    (name="ADMMSLIM", model=() -> ADMMSLIM(max_iter=2, verbose=false), y=false),
    (name="ItemKNN", model=() -> ItemKNN(k=2, verbose=false), y=false),
    (name="SoftImpute", model=() -> SoftImpute(rank=2, max_iter=1, verbose=false), y=false),
    (name="GloVe",   model=() -> GloVe(rank=2, max_iter=1, verbose=false), y=false),
    (name="FTRL",    model=() -> FTRL(max_iter=1, verbose=false), y=true),
    (name="FM",      model=() -> FM(rank=2, max_iter=1, verbose=false), y=true),
]

@testset "NaN/Inf input rejection" begin
    rng = MersenneTwister(42)

    for bad in (NaN, Inf, -Inf), T in (Float32, Float64)
        @testset "X with $bad ($T)" begin
            X0 = T == Float32 ? convert(SparseMatrixCSC{Float32,Int}, sprand(rng, 30, 20, 0.15)) :
                                 sprand(rng, 30, 20, 0.15)
            nnz(X0) > 0 || continue
            X_bad = copy(X0)
            nonzeros(X_bad)[1] = bad

            for spec in _FINITE_MODEL_FACTORIES
                m = spec.model()
                y_ok = spec.y === true ? ones(30) : nothing
                err = try
                    y_ok === nothing ? fit!(m, X_bad; rng=MersenneTwister(1)) :
                                       fit!(m, X_bad, y_ok; rng=MersenneTwister(1))
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin(spec.name, sprint(showerror, err))
                # the rejected fit must not leave a fitted model behind
                if hasproperty(m, :is_fitted)
                    @test !m.is_fitted
                end
            end
        end
    end

    @testset "y targets with NaN/Inf" begin
        X = sprand(rng, 30, 20, 0.15)
        for bad in (NaN, Inf, -Inf), T in (Float32, Float64)
            y_bad = T == Float32 ? fill(Float32(1.0), 30) : fill(1.0, 30)
            y_bad[1] = bad

            m_ftrl = FTRL(max_iter=1, verbose=false)
            @test_throws ArgumentError fit!(m_ftrl, X, y_bad)
            @test !m_ftrl.is_initialized

            m_fm = FM(rank=2, max_iter=1, verbose=false)
            @test_throws ArgumentError fit!(m_fm, X, y_bad)
            @test !m_fm.is_initialized

            # Integer targets are always finite and must be accepted
            @test update!(FTRL(max_iter=1, verbose=false), X, fill(1, 30)) isa FTRL
        end
    end

    @testset "transactional refit keeps prior state" begin
        X0 = sprand(rng, 30, 20, 0.15)
        X_bad = copy(X0)
        nonzeros(X_bad)[1] = NaN

        m = EASE(λ=100.0, verbose=false)
        fit!(m, X0)
        B_before = copy(m.B)
        @test_throws ArgumentError fit!(m, X_bad)
        @test m.is_fitted
        @test m.B == B_before

        m = WMF(rank=2, max_iter=1, verbose=false)
        fit!(m, X0; rng=MersenneTwister(1))
        U_before = copy(m.user_factors)
        @test_throws ArgumentError fit!(m, X_bad; rng=MersenneTwister(1))
        @test m.is_fitted
        @test m.user_factors == U_before
    end

    @testset "GloVe keeps its positivity requirement" begin
        X = sprand(rng, 20, 20, 0.2)
        nonzeros(X)[1] = -1.0
        @test_throws ArgumentError fit!(GloVe(rank=2, max_iter=1, verbose=false), X)
    end

    @testset "EALS update! rejects non-finite data" begin
        X0 = sprand(rng, 30, 20, 0.15)
        m = EALS(rank=2, max_iter=1, verbose=false)
        fit!(m, X0; rng=MersenneTwister(1))
        X_bad = copy(X0)
        nonzeros(X_bad)[1] = Inf
        @test_throws ArgumentError update!(m, X_bad)
        @test m.is_fitted
    end
end
