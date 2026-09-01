# test/test_infrastructure.jl — Callbacks, Serialization, Cross-validation

mutable struct LifecycleCallback <: AbstractCallback
    events::Vector{Symbol}
end

struct FailingCallback <: AbstractCallback end

Canapes.on_train_begin(cb::LifecycleCallback, _) = push!(cb.events, :begin)
Canapes.on_epoch_end(cb::LifecycleCallback, ::Canapes.CallbackInfo) = (push!(cb.events, :epoch); :continue)
Canapes.on_train_end(cb::LifecycleCallback, _) = push!(cb.events, :end)
Canapes.on_epoch_end(::FailingCallback, ::Canapes.CallbackInfo) =
    throw(ArgumentError("intentional callback failure"))

@testset "Callbacks" begin
    @testset "EarlyStoppingCallback" begin
        cb = EarlyStoppingCallback(patience=3, min_delta=0.01)
        model = IALS(rank=3, verbose=false)

        # Improving losses
        for loss in [1.0, 0.9, 0.8, 0.7]
            info = Canapes.CallbackInfo(1, loss, 0.0, model)
            @test on_epoch_end(cb, info) == :continue
        end

        # Stagnating losses → should stop after patience
        for i in 1:3
            info = Canapes.CallbackInfo(i, 0.7, 0.0, model)
            result = on_epoch_end(cb, info)
            if i < 3
                @test result == :continue
            else
                @test result == :stop
            end
        end
    end

    @testset "LossHistoryCallback" begin
        cb = LossHistoryCallback()
        model = IALS(rank=3, verbose=false)
        for i in 1:5
            info = Canapes.CallbackInfo(i, Float64(i) * 0.1, 0.0, model)
            @test on_epoch_end(cb, info) == :continue
        end
        @test length(cb.losses) == 5
        @test cb.losses[1] ≈ 0.1
    end

    @testset "LearningRateCallback" begin
        cb = LearningRateCallback(decay=0.5, min_lr=0.001)
        model = BPR(rank=3, lr=1.0, verbose=false)
        info = Canapes.CallbackInfo(1, 0.5, 0.0, model)
        on_epoch_end(cb, info)
        @test model.lr ≈ 0.5
        on_epoch_end(cb, info)
        @test model.lr ≈ 0.25
    end

    @testset "run_callbacks" begin
        cb1 = LossHistoryCallback()
        cb2 = EarlyStoppingCallback(patience=1, min_delta=0.0)
        model = IALS(rank=3, verbose=false)

        # First call - improving
        info1 = Canapes.CallbackInfo(1, 1.0, 0.0, model)
        @test run_callbacks([cb1, cb2], info1) == false

        # Second call - stagnant
        info2 = Canapes.CallbackInfo(2, 1.0, 1.0, model)
        @test run_callbacks([cb1, cb2], info2) == true
    end
end

@testset "Training lifecycle" begin
    cb = LifecycleCallback(Symbol[])
    model = WMF(rank=2, max_iter=1, verbose=false)
    X = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 3)
    fit!(model, X; rng=MersenneTwister(1), callbacks=[cb])
    @test cb.events == [:begin, :epoch, :end]

    for fit_model in [
        FTRL(max_iter=1, verbose=false),
        FM(rank=2, max_iter=1, verbose=false),
    ]
        current = LifecycleCallback(Symbol[])
        fit!(fit_model, X, [1.0, 0.0]; rng=MersenneTwister(1), callbacks=[current])
        @test current.events == [:begin, :epoch, :end]
    end

    model = WMF(rank=2, max_iter=1, verbose=false)
    fit!(model, X; rng=MersenneTwister(2))
    old_user_factors = copy(model.user_factors)
    old_item_factors = copy(model.item_factors)
    @test_throws ArgumentError fit!(model, X; rng=MersenneTwister(3), callbacks=[FailingCallback()])
    @test model.is_fitted
    @test model.user_factors == old_user_factors
    @test model.item_factors == old_item_factors

    for fit_model in [
        IALS(rank=2, max_iter=1, verbose=false),
        EALS(rank=2, max_iter=1, verbose=false),
        BPR(rank=2, max_iter=1, verbose=false),
        LogisticMF(rank=2, max_iter=1, verbose=false),
    ]
        fit!(fit_model, X; rng=MersenneTwister(2))
        old_user_factors = copy(fit_model.user_factors)
        old_item_factors = copy(fit_model.item_factors)
        @test_throws ArgumentError fit!(fit_model, X;
            rng=MersenneTwister(3), callbacks=[FailingCallback()])
        @test fit_model.is_fitted
        @test fit_model.user_factors == old_user_factors
        @test fit_model.item_factors == old_item_factors
    end

    for (fit_model, state_field) in [
        (EASE(λ=100.0, verbose=false), :B),
        (SLIM(max_iter=1, verbose=false), :W),
        (ADMMSLIM(max_iter=1, verbose=false), :W),
        (ItemKNN(k=2, verbose=false), :W),
    ]
        fit!(fit_model, X; rng=MersenneTwister(2))
        old_state = copy(getproperty(fit_model, state_field))
        @test_throws ArgumentError fit!(fit_model, spzeros(0, 0))
        @test fit_model.is_fitted
        @test getproperty(fit_model, state_field) == old_state
    end
end

@testset "Serialization" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.1)
    model = EASE(λ=100.0, verbose=false)
    fit!(model, X)

    # Save and load
    tmpfile = tempname() * ".jls"
    try
        save_model(model, tmpfile)
        @test isfile(tmpfile)

        loaded = load_model(tmpfile)
        @test loaded isa EASE
        @test loaded.is_fitted
        @test loaded.B ≈ model.B
        @test loaded.λ ≈ model.λ

        # Incompatible format versions must be rejected.
        bytes = read(tmpfile)
        header = bytes[1:findfirst(==(UInt8('\n')), bytes) - 1]
        version_pos = findlast(==(UInt8('v')), header) + 1
        bytes[version_pos] = UInt8('9')
        write(tmpfile, bytes)
        @test_throws ArgumentError load_model(tmpfile)
    finally
        rm(tmpfile; force=true)
    end
end

@testset "Atomic save" begin
    rng = MersenneTwister(7)
    X = sprand(rng, 30, 20, 0.1)
    model = EASE(λ=100.0, verbose=false)
    fit!(model, X)

    dir = mktempdir()
    try
        # Nested (non-existent) directories are created; no temp litter.
        target = joinpath(dir, "nested", "model.jls")
        save_model(model, target)
        @test isfile(target)
        @test readdir(joinpath(dir, "nested")) == ["model.jls"]
        @test load_model(target).B ≈ model.B

        # Overwriting an existing file replaces it atomically: last write wins,
        # and no temp files are left behind.
        model2 = EASE(λ=500.0, verbose=false)
        fit!(model2, X)
        save_model(model2, target)
        @test readdir(joinpath(dir, "nested")) == ["model.jls"]
        @test load_model(target).λ ≈ 500.0

        # A directory target is rejected before any write; nothing is created.
        dir_target = joinpath(dir, "dir_target")
        mkpath(dir_target)
        @test_throws ArgumentError save_model(model, dir_target)
        @test isempty(readdir(dir_target))

        # A failed write cleans up its temp file.
        mktempdir() do empty_dir
            cd(empty_dir) do
                @test_throws Exception save_model(model, "")
            end
            @test isempty(readdir(empty_dir))
        end
    finally
        rm(dir; recursive=true, force=true)
    end
end

@testset "Cross-validation" begin
    @testset "random_holdout" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.2)
        X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))

        # Sizes match
        @test size(X_train) == size(X)
        @test size(X_test) == size(X)

        # No overlap: train and test shouldn't share entries
        overlap = X_train .* X_test
        @test nnz(overlap) == 0

        # Union roughly equals original
        @test nnz(X_train) + nnz(X_test) == nnz(X)
    end

    @testset "cross_validate" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.15)

        mean_score, std_score, fold_scores = cross_validate(
            () -> EASE(λ=200.0, verbose=false),
            X; n_folds=3, k=5, metric=mean_ap_at_k, rng=MersenneTwister(1)
        )

        @test length(fold_scores) == 3
        @test all(s -> 0.0 <= s <= 1.0, fold_scores)
        @test mean_score ≈ sum(fold_scores) / 3
        @test std_score >= 0.0
    end

    @testset "grid_search" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.15)

        best_params, best_score, results = grid_search(
            p -> EASE(λ=p.λ, verbose=false),
            X,
            Dict(:λ => [100.0, 500.0]);
            k=5, test_fraction=0.3, verbose=false, rng=MersenneTwister(1)
        )

        @test length(results) == 2
        @test haskey(best_params, :λ)
        @test best_score >= 0.0
    end

    @testset "random_search" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.15)

        best_params, best_score, results = random_search(
            p -> EASE(λ=p.λ, verbose=false),
            X,
            Dict(:λ => r -> 10.0^(rand(r) * 3));
            n_trials=3, k=5, test_fraction=0.3, verbose=false, rng=MersenneTwister(1)
        )

        @test length(results) == 3
        @test best_score >= 0.0
    end

    @testset "vector metrics in search" begin
        # Regression: ndcg_at_k returns per-user vectors; cross_validate, grid_search
        # and random_search must reduce them to a scalar instead of failing.
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 30, 0.15)

        mean_s, std_s, fold_scores = cross_validate(
            () -> EASE(λ=200.0, verbose=false),
            X; n_folds=3, k=5, metric=ndcg_at_k, rng=MersenneTwister(1)
        )
        @test all(isfinite, fold_scores)
        @test mean_s ≈ sum(fold_scores) / 3

        best_params, best_score, results = grid_search(
            p -> EASE(λ=p.λ, verbose=false),
            X,
            Dict(:λ => [100.0, 500.0]);
            k=5, test_fraction=0.3, verbose=false, rng=MersenneTwister(1),
            metric=ndcg_at_k
        )
        @test all(r -> isfinite(r.score), results)
        @test best_score == maximum(r.score for r in results)

        best_params, best_score, results = random_search(
            p -> EASE(λ=p.λ, verbose=false),
            X,
            Dict(:λ => r -> 10.0^(rand(r) * 3));
            n_trials=3, k=5, test_fraction=0.3, verbose=false, rng=MersenneTwister(1),
            metric=ndcg_at_k
        )
        @test all(r -> isfinite(r.score), results)
    end
end
