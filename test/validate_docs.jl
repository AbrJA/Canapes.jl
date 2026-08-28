# Validate all documentation code examples
# Run with: julia --project=test -t4 validate_docs.jl

using Pkg
Pkg.develop(path=".")
using Gideon, SparseArrays, Random, LinearAlgebra, Test

println("=" ^ 60)
println("Validating documentation code examples")
println("=" ^ 60)

macro validate(name, ex)
    quote
        print("  ", $name, "... ")
        try
            $(esc(ex))
            println("OK")
        catch e
            println("FAILED: ", e)
            global failed = true
        end
    end
end

failed = false

# ──────────────────────────────────────────────────────────────
println("\n[index.md] Quick Start")
# ──────────────────────────────────────────────────────────────

@validate "Quick Start" begin
    X = sprand(MersenneTwister(42), 1000, 500, 0.02)
    X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))
    model = WMF(rank=10, λ=0.1, α=40.0, max_iter=15, verbose=false)
    fit!(model, X_train)
    recommendations = recommend(model, X_train; k=10)
    map_score  = map_at_k(recommendations, X_test; k=10)
    ndcg_score = ndcg_at_k(recommendations, X_test; k=10)
    scores = score(model, X_train)
    pair_scores = score(model, [1, 2, 3], [10, 20, 30])
    ids, sims = similar_items(model, 42; k=5)
end

# ──────────────────────────────────────────────────────────────
println("\n[index.md] Cross-Validation & Hyperparameter Search")
# ──────────────────────────────────────────────────────────────

@validate "crossval" begin
    X = sprand(MersenneTwister(42), 1000, 500, 0.02)
    mean_map, std_map, scores = crossval(
        () -> WMF(rank=10, λ=0.1, α=40.0, max_iter=10, verbose=false),
        X; n_folds=5, k=10, metric=map_at_k
    )
    @assert mean_map >= 0
end

@validate "grid_search" begin
    X = sprand(MersenneTwister(42), 1000, 500, 0.02)
    best_params, best_score, results = grid_search(
        p -> EASE(λ=p.λ, verbose=false),
        X,
        Dict(:λ => [10.0, 100.0, 500.0, 1000.0]);
        k=10
    )
    @assert best_score >= 0
end

@validate "random_search" begin
    X = sprand(MersenneTwister(42), 1000, 500, 0.02)
    best_params, best_score, _ = random_search(
        p -> WMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
        X,
        Dict(:rank => rng -> rand(rng, [10, 20, 50, 100]),
             :λ    => rng -> 10.0^(rand(rng) * 3 - 2));
        n_trials=5  # reduced for speed
    )
end

# ──────────────────────────────────────────────────────────────
println("\n[index.md] Callbacks")
# ──────────────────────────────────────────────────────────────

@validate "Callbacks" begin
    X = sprand(MersenneTwister(42), 1000, 500, 0.02)
    early_stop = EarlyStoppingCallback(patience=3)
    loss_history = LossHistoryCallback()
    checkpoint = CheckpointCallback(path=tempname(), every=5)
    model = WMF(rank=10, λ=0.1, α=40.0, max_iter=50, verbose=false)
    fit!(model, X; callbacks=[early_stop, loss_history, checkpoint])
    @assert length(loss_history.losses) > 0
end

# ──────────────────────────────────────────────────────────────
println("\n[index.md] Tables.jl Integration")
# ──────────────────────────────────────────────────────────────

@validate "Tables - NamedTuple" begin
    data = (user=[1, 1, 2, 2, 3], item=[1, 3, 2, 4, 1], value=[1.0, 1.0, 1.0, 1.0, 1.0])
    X = interactions_to_sparse(data; user_col=:user, item_col=:item, value_col=:value)
    @assert size(X) == (3, 4)
end

@validate "Tables - binary" begin
    clicks = (user_id=[1, 1, 2, 3], item_id=[10, 20, 10, 30])
    X = interactions_to_sparse(clicks; user_col=:user_id, item_col=:item_id, value_col=nothing)
    @assert nnz(X) == 4
end

@validate "Tables - roundtrip" begin
    data = (user=[1, 1, 2, 2, 3], item=[1, 3, 2, 4, 1], value=[1.0, 1.0, 1.0, 1.0, 1.0])
    X = interactions_to_sparse(data; user_col=:user, item_col=:item, value_col=:value)
    triplets = sparse_to_interactions(X)
    @assert length(triplets.user) == 5
end

# ──────────────────────────────────────────────────────────────
println("\n[index.md] Serialization")
# ──────────────────────────────────────────────────────────────

@validate "Serialization" begin
    X = sprand(MersenneTwister(42), 100, 50, 0.1)
    model = EASE(λ=100.0, verbose=false)
    fit!(model, X)
    tmpf = tempname() * ".jls"
    save_model(model, tmpf)
    loaded = load_model(tmpf)
    rm(tmpf; force=true)
    @assert loaded.is_fitted
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] WMF")
# ──────────────────────────────────────────────────────────────

@validate "WMF - CholeskySolver" begin
    X = sprand(MersenneTwister(1), 500, 300, 0.03)
    model = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CholeskySolver(), verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    preds = recommend(model, X; k=5)
    @assert size(preds) == (500, 5)
end

@validate "WMF - ConjugateGradient" begin
    X = sprand(MersenneTwister(1), 500, 300, 0.03)
    model_cg = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=ConjugateGradient(), cg_steps=3, verbose=false)
    fit!(model_cg, X; rng=MersenneTwister(42))
end

@validate "WMF - NonNegative" begin
    X = sprand(MersenneTwister(1), 500, 300, 0.03)
    model_nn = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=NonNegative(), verbose=false)
    fit!(model_nn, X; rng=MersenneTwister(42))
end

@validate "WMF - score & similar" begin
    X = sprand(MersenneTwister(1), 500, 300, 0.03)
    model = WMF(rank=10, λ=0.1, α=40.0, max_iter=5, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    scores = score(model, X)
    ids, sims = similar_items(model, 42; k=5)
    @assert length(ids) == 5
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] IALS")
# ──────────────────────────────────────────────────────────────

@validate "IALS" begin
    X = sprand(MersenneTwister(1), 1000, 500, 0.02)
    model = IALS(rank=32, λ=0.01, α=1.0, max_iter=15, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    preds = recommend(model, X; k=10)
    scores = score(model, X)
    @assert size(preds) == (1000, 10)
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] EALS")
# ──────────────────────────────────────────────────────────────

@validate "EALS" begin
    X = sprand(MersenneTwister(1), 1000, 500, 0.02)
    model = EALS(rank=64, λ=0.01, w0=10.0, max_iter=20, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    preds = recommend(model, X; k=10)
    @assert size(preds) == (1000, 10)
end

@validate "EALS - update!" begin
    X = sprand(MersenneTwister(1), 1000, 500, 0.02)
    model = EALS(rank=64, λ=0.01, w0=10.0, max_iter=5, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    X_new = sprand(MersenneTwister(2), 1000, 500, 0.01)
    update!(model, X_new; n_iter=3)
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] BPR")
# ──────────────────────────────────────────────────────────────

@validate "BPR" begin
    X = sprand(MersenneTwister(1), 500, 300, 0.03)
    model = BPR(rank=32, λ_user=0.01, λ_pos=0.01, λ_neg=0.01, learning_rate=0.05, max_iter=50, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    preds = recommend(model, X; k=10)
    @assert size(preds) == (500, 10)
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] LogisticMF")
# ──────────────────────────────────────────────────────────────

@validate "LogisticMF" begin
    X = sprand(MersenneTwister(1), 800, 300, 0.03)
    model = LogisticMF(rank=15, α=1.0, λ=0.1, learning_rate=0.01, max_iter=20, n_negative=5, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    preds = recommend(model, X; k=10)
    scores = score(model, X)
    @assert size(preds) == (800, 10)
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] GloVe")
# ──────────────────────────────────────────────────────────────

@validate "GloVe" begin
    X = sprand(MersenneTwister(1), 100, 100, 0.1)
    X = X + X'
    nonzeros(X) .= abs.(nonzeros(X))
    model = GloVe(rank=50, x_max=100.0, learning_rate=0.05, max_iter=25, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    E = embeddings(model)
    @assert size(E, 1) == 50
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] SoftImpute / SoftSVD / PureSVD")
# ──────────────────────────────────────────────────────────────

@validate "SoftImpute" begin
    X = sprand(MersenneTwister(1), 200, 150, 0.3)
    model = SoftImpute(rank=10, λ=0.5, max_iter=100, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    preds = recommend(model, X; k=10)
    @assert size(preds) == (200, 10)
end

@validate "SoftSVD" begin
    X = sprand(MersenneTwister(1), 200, 150, 0.3)
    model_svd = SoftSVD(rank=10, λ=0.5, max_iter=100, verbose=false)
    fit!(model_svd, X; rng=MersenneTwister(1))
end

@validate "PureSVD" begin
    X = sprand(MersenneTwister(1), 200, 150, 0.3)
    model_pure = PureSVD(rank=10, max_iter=100, verbose=false)
    fit!(model_pure, X; rng=MersenneTwister(1))
    @assert model_pure isa SoftSVD
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] EASE")
# ──────────────────────────────────────────────────────────────

@validate "EASE" begin
    X = sprand(MersenneTwister(1), 500, 200, 0.05)
    model = EASE(λ=500.0, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=10)
    scores = score(model, X)
    @assert scores isa Matrix
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] SLIM")
# ──────────────────────────────────────────────────────────────

@validate "SLIM" begin
    X = sprand(MersenneTwister(1), 500, 100, 0.05)
    model = SLIM(λ_1=0.01, λ_2=0.1, max_iter=100, nonneg=true, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=10)
    scores = score(model, X)
    @assert scores isa SparseMatrixCSC
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] ADMMSLIM")
# ──────────────────────────────────────────────────────────────

@validate "ADMMSLIM" begin
    X = sprand(MersenneTwister(1), 500, 100, 0.05)
    model = ADMMSLIM(λ_1=0.01, λ_2=100.0, ρ=1.0, max_iter=50, nonneg=true, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=10)
    @assert size(preds) == (500, 10)
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] ItemKNN")
# ──────────────────────────────────────────────────────────────

@validate "ItemKNN - cosine" begin
    X = sprand(MersenneTwister(1), 500, 200, 0.05)
    model = ItemKNN(k=20, similarity=:cosine, normalize=true, verbose=false)
    fit!(model, X)
    preds = recommend(model, X; k=10)
    @assert size(preds) == (500, 10)
end

@validate "ItemKNN - jaccard" begin
    X = sprand(MersenneTwister(1), 500, 200, 0.05)
    model_jac = ItemKNN(k=20, similarity=:jaccard, shrinkage=10.0, verbose=false)
    fit!(model_jac, X)
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] FTRL")
# ──────────────────────────────────────────────────────────────

@validate "FTRL - Binomial" begin
    X = sprand(MersenneTwister(1), 1000, 50, 0.1)
    y = rand(MersenneTwister(2), [0.0, 1.0], 1000)
    model = FTRL(learning_rate=0.1, λ=0.01, family=Binomial(), verbose=false)
    fit!(model, X, y)
    p = predict(model, X)
    @assert all(0 .<= p .<= 1)
end

@validate "FTRL - update!" begin
    X = sprand(MersenneTwister(1), 1000, 50, 0.1)
    y = rand(MersenneTwister(2), [0.0, 1.0], 1000)
    model = FTRL(learning_rate=0.1, λ=0.01, family=Binomial(), verbose=false)
    update!(model, X, y)
end

@validate "FTRL - Gaussian" begin
    model_reg = FTRL(learning_rate=0.1, family=Gaussian(), verbose=false)
    @assert model_reg isa FTRL
end

@validate "FTRL - Poisson" begin
    model_pois = FTRL(learning_rate=0.1, family=Poisson(), verbose=false)
    @assert model_pois isa FTRL
end

@validate "FTRL - coef" begin
    X = sprand(MersenneTwister(1), 100, 10, 0.3)
    y = rand(MersenneTwister(2), [0.0, 1.0], 100)
    model = FTRL(learning_rate=0.1, family=Binomial(), verbose=false)
    fit!(model, X, y)
    w = coef(model)
    @assert length(w) > 0
end

# ──────────────────────────────────────────────────────────────
println("\n[algorithms.md] FM")
# ──────────────────────────────────────────────────────────────

@validate "FM - XOR" begin
    X = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
    y = [0.0, 1.0, 1.0, 0.0]
    model = FM(rank=4, family=Binomial(), max_iter=100, learning_rate_w=0.2, verbose=false)
    fit!(model, X, y; rng=MersenneTwister(42))
    p = predict(model, X)
    @assert length(p) == 4
end

@validate "FM - Gaussian" begin
    X_reg = sprand(MersenneTwister(1), 500, 20, 0.3)
    y_reg = randn(MersenneTwister(2), 500)
    model_reg = FM(rank=8, family=Gaussian(), max_iter=50, verbose=false)
    fit!(model_reg, X_reg, y_reg; rng=MersenneTwister(42))
end

# ──────────────────────────────────────────────────────────────
println("\n[metrics.md] Ranking Metrics")
# ──────────────────────────────────────────────────────────────

@validate "Metrics - basic" begin
    actual = sparse([1,1,1], [3,7,9], ones(3), 1, 10)
    predictions = [3 7 1 9]
    m = map_at_k(predictions, actual; k=4)
    n = ndcg_at_k(predictions, actual; k=4)
    p = precision_at_k(predictions, actual; k=4)
    r = recall_at_k(predictions, actual; k=4)
    @assert p[1] ≈ 0.75
    @assert r[1] ≈ 1.0
end

@validate "Metrics - workflow" begin
    X = sprand(MersenneTwister(42), 1000, 500, 0.02)
    X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))
    model = EASE(λ=500.0, verbose=false)
    fit!(model, X_train)
    preds = recommend(model, X_train; k=10)
    m = map_at_k(preds, X_test; k=10)
    @assert m >= 0
end

# ──────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
if failed
    println("SOME EXAMPLES FAILED — see above")
    exit(1)
else
    println("ALL DOCUMENTATION EXAMPLES VALIDATED SUCCESSFULLY")
end
