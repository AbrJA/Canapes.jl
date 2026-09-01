# test/test_concurrency.jl — Concurrency guarantees
#
# Contract:
# - Fitting the same model from multiple threads is unsupported unless the
#   caller synchronizes externally.
# - Read-only operations (recommend/score) on a fitted model are safe to run
#   concurrently.
# - Independent models may be fitted concurrently (shared-nothing).

@testset "Concurrency" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 300, 150, 0.05)
    C = sprand(rng, 150, 150, 0.05)
    C = C + C'

    # Deterministic solvers: serial and parallel fits must be bit-identical.
    # GloVe (and BPR) is Hogwild lock-free single-pass SGD, so it is excluded
    # from the equality check and asserted for validity only.
    det_model = () -> [
        WMF(rank=8, max_iter=5, verbose=false),
        IALS(rank=8, max_iter=5, verbose=false),
        EALS(rank=8, max_iter=5, verbose=false),
        LogisticMF(rank=8, max_iter=3, verbose=false),
        EASE(λ=200.0, verbose=false),
        SLIM(max_iter=10, verbose=false),
        ADMMSLIM(max_iter=10, verbose=false),
        ItemKNN(k=20, verbose=false),
    ]

    @testset "shared-nothing concurrent training" begin
        serial = det_model()
        for m in serial
            fit!(m, X; rng=MersenneTwister(1))
        end
        serial_recs = [recommend(m, X; k=5) for m in serial]

        parallel = det_model()
        parallel_recs = Vector{Matrix{Int}}(undef, length(parallel))
        Threads.@threads for i in eachindex(parallel)
            fit!(parallel[i], X; rng=MersenneTwister(1))
            parallel_recs[i] = recommend(parallel[i], X; k=5)
        end

        @test parallel_recs == serial_recs
    end

    # BPR and GloVe use Hogwild! lock-free SGD: concurrent writes to shared
    # factors are intentionally racing, so fits are not bit-reproducible across
    # thread counts or schedules. Concurrent training must still succeed and
    # produce valid output.
    @testset "hogwild concurrent training" begin
        models = [BPR(rank=8, max_iter=3, verbose=false) for _ in 1:2]
        recs = Vector{Matrix{Int}}(undef, length(models))
        Threads.@threads for i in eachindex(models)
            fit!(models[i], X; rng=MersenneTwister(1))
            recs[i] = recommend(models[i], X; k=5)
        end
        @test all(r -> size(r) == (size(X, 1), 5), recs)
        @test all(r -> all(in(1:size(X, 2)), r), recs)

        glove = [GloVe(rank=8, max_iter=3, verbose=false) for _ in 1:2]
        glove_recs = Vector{Matrix{Int}}(undef, length(glove))
        Threads.@threads for i in eachindex(glove)
            fit!(glove[i], C; rng=MersenneTwister(1))
            glove_recs[i] = recommend(glove[i], C; k=5)
        end
        @test all(r -> size(r) == (size(C, 1), 5), glove_recs)
        @test all(r -> all(in(1:size(C, 2)), r), glove_recs)
    end

    @testset "concurrent reads on a shared fitted model" begin
        model = WMF(rank=8, max_iter=5, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        reference = recommend(model, X; k=5)

        results = Vector{Matrix{Int}}(undef, max(2 * Threads.nthreads(), 4))
        Threads.@threads for i in eachindex(results)
            results[i] = recommend(model, X; k=5)
        end
        @test all(==(reference), results)

        bpr = BPR(rank=8, max_iter=3, verbose=false)
        fit!(bpr, X; rng=MersenneTwister(1))
        bpr_ref = recommend(bpr, X; k=5)
        bpr_results = Vector{Matrix{Int}}(undef, max(2 * Threads.nthreads(), 4))
        Threads.@threads for i in eachindex(bpr_results)
            bpr_results[i] = recommend(bpr, X; k=5)
        end
        @test all(==(bpr_ref), bpr_results)
    end

    @testset "multithreaded subprocess smoke test" begin
        script = """
        using Canapes, SparseArrays, Random
        X = sprand(MersenneTwister(1), 100, 50, 0.05)
        for m in (WMF(rank=4, max_iter=3, verbose=false), EASE(λ=100.0, verbose=false))
            fit!(m, X; rng=MersenneTwister(2))
            size(recommend(m, X; k=5)) == (100, 5) || exit(1)
        end
        exit(0)
        """
        for nthreads in (2, 4)
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$nthreads -e $script`
            @test success(pipeline(cmd; stdout=devnull, stderr=devnull))
        end
    end
end
