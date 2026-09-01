# test/test_fixtures.jl
# ──────────────────────────────────────────────────────────────────────────────
# Pure-Julia ports of the `implicit` library's correctness fixtures
# (test_factorize, test_cg_nan, test_small_nan, test_calculate_loss, and the
# recommender_base behaviors). No external (Python/R) dependencies.
# ──────────────────────────────────────────────────────────────────────────────

using Test, SparseArrays, LinearAlgebra, Random

# ── Fixture: test_factorize — explicit ALS reconstructs X ≈ U·Vᵀ ──

@testset "Fixture: test_factorize (explicit reconstruction)" begin
    X = sparse([1.0 2.0 3.0; 4.0 5.0 6.0])

    model = WMF(rank=2, λ=0.0, max_iter=100, solver=CholeskySolver(),
                feedback=Explicit, tol=-1.0, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))

    @test size(model.user_factors) == (2, 2)
    @test size(model.item_factors) == (2, 3)

    recon = score(model, X)
    @test size(recon) == (2, 3)
    @test all(isfinite, recon)
    # Explicit ALS should factorize X ≈ U·Vᵀ (unregularized, up to rotation).
    @test maximum(abs.(recon .- Matrix(X))) < 1e-3
end

# ── Fixture: test_cg_nan — CG stays finite on adversarial tiny inputs ──
# https://github.com/benfred/implicit/issues/157

@testset "Fixture: test_cg_nan" begin
    X = sparse([0.0 1.0; 1.0 0.0])
    model = WMF(rank=10, λ=0.01, max_iter=10, solver=CGSolver(),
                cg_steps=3, feedback=Implicit, verbose=false)
    fit!(model, X; rng=MersenneTwister(42))
    @test all(isfinite, model.user_factors)
    @test all(isfinite, model.item_factors)

    # Single-rating rows/columns must not poison the CG solve either.
    X2 = sparse([1.0 0 0 0 0; 0 1.0 0 0 0])
    model2 = WMF(rank=3, λ=0.01, max_iter=5, solver=CGSolver(),
                 cg_steps=3, feedback=Implicit, verbose=false)
    fit!(model2, X2; rng=MersenneTwister(7))
    @test all(isfinite, model2.user_factors)
    @test all(isfinite, model2.item_factors)
end

# ── Fixture: test_small_nan — tiny matrices produce finite factors ──

@testset "Fixture: test_small_nan" begin
    X = sparse([1.0 0 0 0 0; 0 1.0 0 0 0])

    for (name, factory) in [
        ("WMF", () -> WMF(rank=3, λ=0.0, max_iter=2, solver=CholeskySolver(), verbose=false)),
        ("WMF-CG", () -> WMF(rank=3, λ=0.0, max_iter=2, solver=CGSolver(), cg_steps=3, verbose=false)),
        ("IALS", () -> IALS(rank=3, λ=0.0, max_iter=2, verbose=false)),
        ("IALS-CG", () -> IALS(rank=3, λ=0.0, max_iter=2, solver=CGSolver(), verbose=false)),
        ("EALS", () -> EALS(rank=3, λ=0.0, max_iter=2, verbose=false)),
        ("LogisticMF", () -> LogisticMF(rank=3, λ=0.0, max_iter=2, n_negative=5, verbose=false)),
    ]
        model = factory()
        fit!(model, X; rng=MersenneTwister(42))
        @test all(isfinite, model.user_factors)
        @test all(isfinite, model.item_factors)
    end
end

# ── Fixture: test_calculate_loss — reported loss matches a manual recompute ──

@testset "Fixture: test_calculate_loss" begin
    X = sprand(MersenneTwister(7), 50, 40, 0.1)

    function manual_wmf_loss(model, X)
        U = model.user_factors; V = model.item_factors
        λ = model.λ; α = model.α; k = model.rank
        rv = rowvals(X); nz = nonzeros(X)
        loss = 0.0
        for j in axes(X, 2), idx in nzrange(X, j)
            i = rv[idx]; r = Float64(nz[idx])
            pred = dot(view(U, :, i), view(V, :, j))
            if model.feedback == Implicit
                c = max(1.0, 1.0 + Float64(α) * r)
                loss += c * (1.0 - pred)^2
            else
                loss += (r - pred)^2
            end
        end
        loss + Float64(λ) * (sum(abs2, U) + sum(abs2, V))
    end

    for feedback in [Implicit, Explicit]
        model = WMF(rank=5, λ=0.1, α=1.0, max_iter=8, solver=CholeskySolver(),
                    feedback=feedback, verbose=false)
        fit!(model, X; rng=MersenneTwister(1))
        reported = Canapes._compute_loss(model, X)
        manual = manual_wmf_loss(model, X)
        @test isapprox(reported, manual; rtol=1e-5)

        # More iterations strictly lower the loss (calculate_loss semantics).
        early = WMF(rank=5, λ=0.1, α=1.0, max_iter=2, solver=CholeskySolver(),
                    feedback=feedback, tol=-1.0, verbose=false)
        fit!(early, X; rng=MersenneTwister(1))
        @test manual_wmf_loss(model, X) < manual_wmf_loss(early, X)
    end
end

# ── Fixture: recommender_base — similar_items / similar_users batch + filter ──

@testset "Fixture: similar_items/users batch + self-filter" begin
    X = sprand(MersenneTwister(11), 60, 50, 0.1)
    specs = [
        (name="WMF", factory=() -> WMF(rank=4, max_iter=5, verbose=false)),
        (name="IALS", factory=() -> IALS(rank=4, max_iter=5, verbose=false)),
        (name="EALS", factory=() -> EALS(rank=4, max_iter=5, verbose=false)),
        (name="BPR", factory=() -> BPR(rank=4, max_iter=5, verbose=false)),
        (name="LogisticMF", factory=() -> LogisticMF(rank=4, max_iter=5, n_negative=10, verbose=false)),
        (name="GloVe", factory=() -> GloVe(rank=4, max_iter=5, verbose=false)),
    ]
    for spec in specs
        model = spec.factory()
        if spec.name == "GloVe"
            C = sprand(MersenneTwister(5), 50, 50, 0.1)
            C = C + C'
            nonzeros(C) .= abs.(nonzeros(C)) .+ 0.1
            fit!(model, C; rng=MersenneTwister(1))
            n = size(C, 1)
        else
            fit!(model, X; rng=MersenneTwister(1))
            n = size(X, 2)
        end
        for q in [1, 2, 3]
            ids, sims = similar_items(model, q; k=5)
            @test length(ids) == min(5, n - 1)
            @test q ∉ ids
            @test all(ids .>= 1) && all(ids .<= n)
            @test issorted(sims; rev=true)
            @test all(isfinite, sims)
        end
        uid, usims = similar_users(model, 1; k=3)
        @test 1 ∉ uid
        @test issorted(usims; rev=true)
    end
end

# ── Fixture: recommender_base — rank_items / filter_items behaviors ──

@testset "Fixture: rank/filter item behaviors" begin
    X = sprand(MersenneTwister(21), 40, 30, 0.15)
    specs = [
        (name="WMF", factory=() -> WMF(rank=4, max_iter=5, verbose=false)),
        (name="IALS", factory=() -> IALS(rank=4, max_iter=5, verbose=false)),
        (name="BPR", factory=() -> BPR(rank=4, max_iter=5, verbose=false)),
        (name="SLIM", factory=() -> SLIM(max_iter=2, verbose=false)),
        (name="ItemKNN", factory=() -> ItemKNN(k=5, verbose=false)),
        (name="EASE", factory=() -> EASE(λ=10.0, verbose=false)),
    ]
    for spec in specs
        model = spec.factory()
        fit!(model, X; rng=MersenneTwister(1))
        full = score(model, X)

        # rank_items: a candidate set is ranked by descending score.
        candidates = [1, 3, 5, 7, 9, 11]
        for u in [1, 2, 3]
            s = if model isa AbstractMatrixFactorization
                score(model, fill(u, length(candidates)), candidates)
            else
                [full[u, c] for c in candidates]
            end
            @test length(s) == length(candidates)
            @test all(isfinite, s)
            ranked = candidates[sortperm(collect(s); rev=true)]
            @test length(unique(ranked)) == length(candidates)
        end

        # filter_items: seen items never appear in recommendations.
        for u in axes(X, 1)
            seen = findall(!iszero, X[u, :])
            preds = recommend(model, X; k=min(5, size(X, 2)))[u, :]
            @test isempty(intersect(seen, preds))
        end
    end
end

# ── Fixture: recommender_base — T preservation ──

@testset "Fixture: T preservation" begin
    X = sprand(MersenneTwister(31), 40, 30, 0.1)
    specs = [
        ("WMF", () -> WMF(rank=4, max_iter=3, verbose=false),
                () -> WMF(rank=4, max_iter=3, T=Float64, verbose=false)),
        ("IALS", () -> IALS(rank=4, max_iter=3, verbose=false),
                 () -> IALS(rank=4, max_iter=3, T=Float64, verbose=false)),
        ("EALS", () -> EALS(rank=4, max_iter=3, verbose=false),
                 () -> EALS(rank=4, max_iter=3, T=Float64, verbose=false)),
        ("BPR", () -> BPR(rank=4, max_iter=3, verbose=false),
                () -> BPR(rank=4, max_iter=3, T=Float64, verbose=false)),
        ("LogisticMF", () -> LogisticMF(rank=4, max_iter=3, verbose=false),
                       () -> LogisticMF(rank=4, max_iter=3, T=Float64, verbose=false)),
        ("EASE", () -> EASE(λ=10.0, verbose=false),
                 () -> EASE(λ=10.0, T=Float64, verbose=false)),
        ("SLIM", () -> SLIM(max_iter=2, verbose=false),
                 () -> SLIM(max_iter=2, T=Float64, verbose=false)),
        ("ItemKNN", () -> ItemKNN(k=5, verbose=false),
                    () -> ItemKNN(k=5, T=Float64, verbose=false)),
    ]
    for (name, f32, f64) in specs
        m32 = f32()
        m64 = f64()
        fit!(m32, X; rng=MersenneTwister(1))
        fit!(m64, X; rng=MersenneTwister(1))
        for f in (:user_factors, :item_factors, :W, :B)
            if hasproperty(m32, f)
                @test eltype(getproperty(m32, f)) == Float32
                @test eltype(getproperty(m64, f)) == Float64
            end
        end
    end
end

# ── Fixture: recommender_base — transform parity (fold-in) ──

@testset "Fixture: transform parity (fold-in)" begin
    X = sprand(MersenneTwister(41), 60, 40, 0.1)
    model = WMF(rank=4, λ=0.1, α=1.0, max_iter=150, solver=CholeskySolver(),
                feedback=Implicit, tol=-1.0, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))

    U_foldin = transform(model, X)
    @test size(U_foldin) == size(model.user_factors)
    @test all(isfinite, U_foldin)
    # A converged model's fold-in on the training matrix reproduces the fitted
    # user factors (the fit stops after an item sweep, so allow a small delta).
    rel = norm(U_foldin - model.user_factors) / (norm(model.user_factors) + 1e-12)
    @test rel < 1e-3

    # Score parity: fold-in embeddings reproduce fitted user-item scores.
    scores_foldin = U_foldin' * model.item_factors
    scores_fit = model.user_factors' * model.item_factors
    @test maximum(abs.(scores_foldin .- scores_fit)) < 1e-3
end
