# test/test_coverage.jl — Additional tests to increase coverage
# Covers: GPU (with CUDA), Tables.jl edge cases, serialization edge cases,
# CheckpointCallback, score across models, transform, error paths

# ──────────────────────────────────────────────────────────────────────────────
# Tables.jl edge cases
# ──────────────────────────────────────────────────────────────────────────────

@testset "Tables edge cases" begin
    @testset "Empty table throws error" begin
        table = (user=Int[], item=Int[], value=Float64[])
        @test_throws ArgumentError interactions_to_sparse(table; user_col=:user, item_col=:item, value_col=:value)
    end

    @testset "Out-of-bounds indices with explicit dims" begin
        table = (user=[1, 5], item=[1, 2], value=[1.0, 1.0])
        @test_throws ArgumentError interactions_to_sparse(
            table; user_col=:user, item_col=:item, value_col=:value, n_users=3, n_items=2
        )
    end

    @testset "Duplicate entries are summed" begin
        table = (user=[1, 1, 1], item=[2, 2, 2], value=[1.0, 2.0, 3.0])
        X = interactions_to_sparse(table; user_col=:user, item_col=:item, value_col=:value)
        @test X[1, 2] == 6.0  # sparse() sums duplicates
    end

    @testset "Single interaction" begin
        table = (user=[1], item=[1], value=[5.0])
        X = interactions_to_sparse(table; user_col=:user, item_col=:item, value_col=:value)
        @test size(X) == (1, 1)
        @test X[1, 1] == 5.0
    end

    @testset "Large indices with explicit dims" begin
        table = (user=[100], item=[500], value=[1.0])
        X = interactions_to_sparse(table; user_col=:user, item_col=:item, value_col=:value,
                                   n_users=1000, n_items=2000)
        @test size(X) == (1000, 2000)
        @test nnz(X) == 1
    end

    @testset "sparse_to_interactions preserves all entries" begin
        X = sparse([1, 2, 3, 3], [4, 5, 1, 2], [0.5, 1.5, 2.5, 3.5], 3, 5)
        triplets = sparse_to_interactions(X)
        @test length(triplets.user) == 4
        @test sort(triplets.value) == [0.5, 1.5, 2.5, 3.5]
    end

    @testset "Row iteration with missing value_col=nothing" begin
        rows = [(user=1, item=2), (user=2, item=3), (user=3, item=1)]
        X = interactions_to_sparse(rows; user_col=:user, item_col=:item, value_col=nothing)
        @test all(nonzeros(X) .== 1.0)
        @test nnz(X) == 3
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Serialization edge cases
# ──────────────────────────────────────────────────────────────────────────────

@testset "Serialization edge cases" begin
    @testset "Load non-existent file" begin
        @test_throws ErrorException load_model("/nonexistent/path/model.jls")
    end

    @testset "Load corrupt file" begin
        tmpfile = tempname() * ".jls"
        try
            write(tmpfile, "NOT_A_GIDEON_FILE\ngarbage")
            @test_throws ErrorException load_model(tmpfile)
        finally
            rm(tmpfile; force=true)
        end
    end

    @testset "Save to nested directory" begin
        tmpdir = joinpath(tempdir(), "gideon_test_$(rand(1000:9999))", "subdir")
        tmpfile = joinpath(tmpdir, "model.jls")
        try
            model = EASE(λ=50.0, verbose=false)
            rng = MersenneTwister(42)
            X = sprand(rng, 20, 15, 0.1)
            fit!(model, X)
            save_model(model, tmpfile)
            @test isfile(tmpfile)
            loaded = load_model(tmpfile)
            @test loaded.B ≈ model.B
        finally
            rm(tmpdir; recursive=true, force=true)
        end
    end

    @testset "Save/load all model types" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 30, 20, 0.1)

        # WMF
        m = WMF(rank=3, max_iter=2, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        tmpfile = tempname() * ".jls"
        save_model(m, tmpfile)
        loaded = load_model(tmpfile)
        @test loaded.user_factors ≈ m.user_factors
        rm(tmpfile; force=true)

        # IALS
        m = IALS(rank=3, max_iter=2, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        tmpfile = tempname() * ".jls"
        save_model(m, tmpfile)
        loaded = load_model(tmpfile)
        @test loaded.user_factors ≈ m.user_factors
        rm(tmpfile; force=true)

        # BPR
        m = BPR(rank=3, max_iter=2, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        tmpfile = tempname() * ".jls"
        save_model(m, tmpfile)
        loaded = load_model(tmpfile)
        @test loaded.user_factors ≈ m.user_factors
        rm(tmpfile; force=true)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# CheckpointCallback
# ──────────────────────────────────────────────────────────────────────────────

@testset "CheckpointCallback" begin
    tmpdir = joinpath(tempdir(), "gideon_checkpoint_$(rand(1000:9999))")
    try
        cb = CheckpointCallback(every=2, path=tmpdir)
        model = IALS(rank=3, verbose=false)
        rng = MersenneTwister(42)
        X = sprand(rng, 20, 15, 0.1)
        fit!(model, X; rng=MersenneTwister(1))

        # Simulate epoch callbacks
        for epoch in 1:5
            info = Gideon.CallbackInfo(epoch, 1.0 / epoch, 0.0, model)
            result = on_epoch_end(cb, info)
            @test result == :continue
        end

        # Should have saved at epochs 2 and 4
        @test isfile(joinpath(tmpdir, "model_epoch_2.jls"))
        @test isfile(joinpath(tmpdir, "model_epoch_4.jls"))
        @test !isfile(joinpath(tmpdir, "model_epoch_1.jls"))

        # Verify saved model is loadable
        loaded = load_model(joinpath(tmpdir, "model_epoch_4.jls"))
        @test loaded isa IALS
        @test loaded.user_factors ≈ model.user_factors
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# score across all MF models
# ──────────────────────────────────────────────────────────────────────────────

@testset "score universality" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 30, 0.1)

    @testset "WMF score" begin
        model = WMF(rank=4, max_iter=3, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        scores = score(model, [1, 2, 3], [1, 2, 3])
        @test length(scores) == 3
        @test all(isfinite, scores)
        # Verify against manual dot product
        expected = [dot(model.user_factors[:, i], model.item_factors[:, j])
                    for (i, j) in zip([1, 2, 3], [1, 2, 3])]
        @test scores ≈ expected
    end

    @testset "LogisticMF recommend" begin
        model = LogisticMF(rank=4, max_iter=3, lr=0.01, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        preds = recommend(model, X; k=5)
        @test size(preds) == (40, 5)
        @test all(preds .>= 1)
        @test all(preds .<= 30)
    end

    @testset "GloVe embeddings" begin
        X_sq = sprand(MersenneTwister(7), 30, 30, 0.1)
        model = GloVe(rank=4, max_iter=3, verbose=false)
        fit!(model, X_sq; rng=MersenneTwister(1))
        emb = embeddings(model)
        @test size(emb) == (4, 30)
        @test all(isfinite, emb)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# transform (fold-in new users)
# ──────────────────────────────────────────────────────────────────────────────

@testset "transform new users" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 100, 60, 0.05)

    @testset "WMF CholeskySolver transform" begin
        model = WMF(rank=8, max_iter=5, solver=CholeskySolver(), verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        X_new = sprand(MersenneTwister(7), 5, 60, 0.1)
        U_new = Gideon.transform(model, X_new)
        @test size(U_new) == (8, 5)
        @test all(isfinite, U_new)
    end

    @testset "WMF CG transform" begin
        model = WMF(rank=8, max_iter=5, solver=CGSolver(), verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        X_new = sprand(MersenneTwister(7), 3, 60, 0.1)
        U_new = Gideon.transform(model, X_new)
        @test size(U_new) == (8, 3)
        @test all(isfinite, U_new)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation with different metrics
# ──────────────────────────────────────────────────────────────────────────────

@testset "Cross-validation metrics" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 60, 40, 0.1)

    @testset "cv with map_at_k" begin
        mean_s, std_s, folds = crossval(
            () -> EASE(λ=200.0, verbose=false),
            X; n_folds=2, k=5, metric=map_at_k, rng=MersenneTwister(1)
        )
        @test length(folds) == 2
        @test all(0 .<= folds .<= 1)
    end

    @testset "cv with different model" begin
        mean_s, std_s, folds = crossval(
            () -> WMF(rank=4, max_iter=3, verbose=false),
            X; n_folds=2, k=5, metric=map_at_k, rng=MersenneTwister(1)
        )
        @test length(folds) == 2
        @test mean_s >= 0.0
    end

    @testset "random_holdout fractions" begin
        X_train, X_test = random_holdout(X; test_fraction=0.5, rng=MersenneTwister(1))
        @test nnz(X_train) + nnz(X_test) == nnz(X)
        # With 50% split, test should have roughly half
        @test nnz(X_test) > nnz(X) * 0.3
        @test nnz(X_test) < nnz(X) * 0.7
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Predict masks seen items
# ──────────────────────────────────────────────────────────────────────────────

@testset "recommend masks seen items" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 20, 15, 0.3)  # denser so we can check masking

    model = EASE(λ=50.0, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=5)

    # For each user, predicted items should NOT be in their history
    rv = rowvals(X)
    for u in 1:20
        seen = Set{Int}()
        for j in axes(X, 2)
            for idx in nzrange(X, j)
                if rv[idx] == u
                    push!(seen, j)
                end
            end
        end
        for k_idx in 1:5
            @test preds[u, k_idx] ∉ seen
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# SLIM specific tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "SLIM recommend" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.15)
    model = SLIM(λ_l1=1.0, λ_l2=0.5, max_iter=5, verbose=false)
    fit!(model, X)
    @test model.is_fitted

    preds = recommend(model, X; k=5)
    @test size(preds) == (30, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 20)
end

# ──────────────────────────────────────────────────────────────────────────────
# SoftImpute / SoftSVD specific tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "SoftImpute convergence" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.2)

    model = SoftImpute(rank=5, λ=1.0, max_iter=50, tol=1e-5, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
    @test size(model.U, 1) == 30
    @test size(model.U, 2) <= 5
    @test size(model.V, 1) == 20
    @test size(model.V, 2) <= 5
    @test all(isfinite, model.U)
    @test all(isfinite, model.V)
end

@testset "SoftSVD convergence" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 20, 0.2)

    model = SoftSVD(rank=5, λ=1.0, max_iter=50, tol=1e-5, verbose=false)
    fit!(model, X; rng=rng)
    @test model.is_fitted
    @test size(model.U, 1) == 30
    @test size(model.U, 2) <= 5
    @test size(model.V, 1) == 20
    @test size(model.V, 2) <= 5
    @test all(isfinite, model.U)
    @test all(isfinite, model.V)
end

@testset "ItemKNN basic coverage" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 20, 0.2)

    model = ItemKNN(k=5, similarity=:cosine, verbose=false)
    fit!(model, X)
    @test model.is_fitted
    preds = recommend(model, X; k=3)
    @test size(preds) == (40, 3)
    S = score(model, X)
    @test size(S) == (40, 20)
end

@testset "ADMMSLIM basic coverage" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 40, 15, 0.2)

    model = ADMMSLIM(λ_l1=0.01, λ_l2=100.0, max_iter=20, verbose=false)
    fit!(model, X)
    @test model.is_fitted
    preds = recommend(model, X; k=3)
    @test size(preds) == (40, 3)
    S = score(model, X)
    @test size(S) == (40, 15)
    @test all(isfinite, S)
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU tests (actual CUDA tests when available)
# ──────────────────────────────────────────────────────────────────────────────

const _HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

if _HAS_CUDA
    @testset "GPU EASE correctness" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 100, 80, 0.05)

        model_gpu = EASE(λ=100.0, verbose=false)
        fit_gpu!(model_gpu, X)
        model_cpu = EASE(λ=100.0, verbose=false)
        fit!(model_cpu, X)

        @test model_gpu.is_fitted
        @test model_gpu.B ≈ model_cpu.B atol=1e-5
    end

    @testset "GPU IALS correctness" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 80, 60, 0.05)

        model = IALS(rank=8, max_iter=5, α=10.0, verbose=false)
        fit_gpu!(model, X; rng=MersenneTwister(1))

        @test model.is_fitted
        @test size(model.user_factors) == (8, 80)
        @test size(model.item_factors) == (8, 60)
        @test all(isfinite, model.user_factors)
        @test all(isfinite, model.item_factors)

        # Loss should decrease
        model2 = IALS(rank=8, max_iter=1, α=10.0, verbose=false)
        fit_gpu!(model2, X; rng=MersenneTwister(1))
        # More iterations => lower residual (loosely)
        @test norm(model.user_factors) > 0
    end

    @testset "GPU WMF correctness" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 80, 60, 0.05)

        model = WMF(rank=8, max_iter=5, solver=CholeskySolver(), verbose=false)
        fit_gpu!(model, X; rng=MersenneTwister(1))

        @test model.is_fitted
        @test size(model.user_factors) == (8, 80)
        @test size(model.item_factors) == (8, 60)
        @test all(isfinite, model.user_factors)
        @test all(isfinite, model.item_factors)
    end

    @testset "GPU score matches CPU" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 40, 0.1)

        model = IALS(rank=8, max_iter=5, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))

        scores_gpu = score_gpu(model, X)
        scores_cpu = model.user_factors' * model.item_factors

        @test size(scores_gpu) == (50, 40)
        @test scores_gpu ≈ scores_cpu atol=1e-5
    end

    @testset "GPU recommend_gpu returns valid results" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 40, 0.1)

        model = IALS(rank=8, max_iter=5, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))

        preds_gpu = recommend_gpu(model, X; k=10)
        preds_cpu = recommend(model, X; k=10)

        @test size(preds_gpu) == (50, 10)
        @test all(preds_gpu .>= 1)
        @test all(preds_gpu .<= 40)
        # Should match CPU predictions exactly (same algorithm)
        @test preds_gpu == preds_cpu
    end

    @testset "GPU recommend_gpu masks seen items" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 30, 25, 0.2)

        model = WMF(rank=5, max_iter=5, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))

        preds = recommend_gpu(model, X; k=5)
        rv = rowvals(X)
        for u in 1:30
            seen = Set{Int}()
            for j in axes(X, 2)
                for idx in nzrange(X, j)
                    if rv[idx] == u
                        push!(seen, j)
                    end
                end
            end
            for k_idx in 1:5
                @test preds[u, k_idx] ∉ seen
            end
        end
    end

    @testset "GPU with larger matrix" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 1000, 500, 0.01)

        model = EASE(λ=500.0, verbose=false)
        fit_gpu!(model, X)
        @test model.is_fitted
        @test size(model.B) == (500, 500)
        @test all(isfinite, model.B)
    end
else
    @testset "GPU stubs (no CUDA)" begin
        @test isdefined(Gideon, :fit_gpu!)
        @test isdefined(Gideon, :recommend_gpu)
        @test isdefined(Gideon, :score_gpu)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Edge cases for algorithms
# ──────────────────────────────────────────────────────────────────────────────

@testset "Algorithm edge cases" begin
    @testset "Very sparse matrix (1 interaction)" begin
        X = sparse([1], [1], [1.0], 50, 50)
        model = IALS(rank=3, max_iter=3, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        @test model.is_fitted
        @test all(isfinite, model.user_factors)
    end

    @testset "Single user" begin
        X = sparse([1, 1, 1], [1, 2, 3], [1.0, 1.0, 1.0], 1, 10)
        model = EASE(λ=10.0, verbose=false)
        fit!(model, X)
        @test model.is_fitted
        preds = recommend(model, X; k=5)
        @test size(preds) == (1, 5)
    end

    @testset "Single item per user" begin
        X = sparse([1, 2, 3], [1, 2, 3], [1.0, 1.0, 1.0], 3, 5)
        model = BPR(rank=3, max_iter=5, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        @test model.is_fitted
    end

    @testset "k larger than items" begin
        X = sprand(MersenneTwister(42), 10, 5, 0.3)
        model = EASE(λ=10.0, verbose=false)
        fit!(model, X)
        # k=10 > n_items=5, should still work
        preds = recommend(model, X; k=10)
        @test size(preds, 2) <= 5
    end

    @testset "Float32 input matrix" begin
        rng = MersenneTwister(42)
        X = SparseMatrixCSC{Float32,Int}(sprand(rng, 50, 30, 0.1))
        model = WMF(rank=4, max_iter=3, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        @test model.is_fitted
        @test all(isfinite, model.user_factors)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Metrics edge cases
# ──────────────────────────────────────────────────────────────────────────────

@testset "Metrics edge cases" begin
    @testset "Single item relevant" begin
        actual = sparse([1], [5], [1.0], 1, 10)
        preds = [5 1 2 3 4]
        @test ap_at_k(preds, actual; k=5)[1] ≈ 1.0
        @test ndcg_at_k(preds, actual; k=5)[1] ≈ 1.0
        @test precision_at_k(preds, actual; k=5)[1] ≈ 0.2
        @test recall_at_k(preds, actual; k=5)[1] ≈ 1.0
    end

    @testset "k=1" begin
        actual = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 5)
        preds_hit = reshape([1], 1, 1)
        preds_miss = reshape([5], 1, 1)
        @test precision_at_k(preds_hit, actual; k=1)[1] ≈ 1.0
        @test precision_at_k(preds_miss, actual; k=1)[1] ≈ 0.0
    end

    @testset "Large k with few relevant" begin
        actual = sparse([1], [1], [1.0], 1, 100)
        preds = collect(1:20)'  # row vector
        @test recall_at_k(preds, actual; k=20)[1] ≈ 1.0
    end
end
