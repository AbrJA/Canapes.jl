# test/test_properties.jl — randomized sparse property-based testing (Item 17)
#
# No external property framework: draws are driven by a seeded MersenneTwister
# over dimensions, densities, dtypes, duplicate triplets and value scales.
# Invariants (shape, seen-exclusion, finiteness, determinism, persistence)
# are asserted on every draw.

# ── Random sparse generators ──────────────────────────────────────────────────

function _loguniform(rng::AbstractRNG, lo::Real, hi::Real, ::Type{T}) where {T<:AbstractFloat}
    lo == hi && return T(lo)
    T(10.0^(rand(rng) * (log10(hi) - log10(lo)) + log10(lo)))
end

function _random_sparse(rng::AbstractRNG, users::Int, items::Int, density::Float64,
                        ::Type{T}; lo::Real=1e-2, hi::Real=1e2,
                        positive::Bool=false, signed::Bool=false,
                        duplicates::Bool=true) where {T<:AbstractFloat}
    n_entries = max(1, round(Int, users * items * density))
    rows = rand(rng, 1:users, n_entries)
    cols = rand(rng, 1:items, n_entries)
    vals = T[_loguniform(rng, lo, hi, T) * (positive || !signed || rand(rng, Bool) ? one(T) : -one(T))
             for _ in 1:n_entries]
    if duplicates
        n_dup = max(1, div(n_entries, 2))
        append!(rows, rows[1:n_dup])
        append!(cols, cols[1:n_dup])
        append!(vals, vals[1:n_dup] .* T(0.5))
    end
    sparse(rows, cols, vals, users, items)
end

# ── Property battery ──────────────────────────────────────────────────────────

function _assert_valid_predictions(preds::Matrix{Int}, X::SparseMatrixCSC,
                                   k::Int, n_items::Int)
    n_users = size(X, 1)
    @test size(preds) == (n_users, k)
    for u in 1:n_users
        @test all(p -> 1 <= p <= n_items, preds[u, :])
        @test length(unique(preds[u, :])) == k
        # Seen-item exclusion holds whenever enough unseen items remain
        seen = findnz(X[u, :])[1]
        if n_items - length(seen) >= k
            @test isempty(intersect(preds[u, :], seen))
        end
    end
end

const _PROPERTY_MODELS = [
    (name="WMF",         model=() -> WMF(rank=3, max_iter=3, verbose=false), explicit=false, square=false),
    (name="IALS",        model=() -> IALS(rank=3, max_iter=2, verbose=false), explicit=false, square=false),
    (name="EALS",        model=() -> EALS(rank=3, max_iter=2, verbose=false), explicit=false, square=false),
    (name="LogisticMF",  model=() -> LogisticMF(rank=3, max_iter=2, verbose=false), explicit=false, square=false),
    (name="BPR",         model=() -> BPR(rank=3, max_iter=2, verbose=false), explicit=false, square=false),
    (name="EASE",        model=() -> EASE(λ=100.0, verbose=false), explicit=false, square=false),
    (name="SLIM",        model=() -> SLIM(λ_l1=0.01, max_iter=3, verbose=false), explicit=false, square=false),
    (name="ADMMSLIM",    model=() -> ADMMSLIM(λ_l1=0.01, max_iter=3, verbose=false), explicit=false, square=false),
    (name="ItemKNN",     model=() -> ItemKNN(k=3, verbose=false), explicit=false, square=false),
    (name="SoftImpute",  model=() -> SoftImpute(rank=3, max_iter=2, verbose=false), square=false, explicit=true),
    (name="GloVe",       model=() -> GloVe(rank=3, max_iter=2, verbose=false), explicit=false, square=true, hogwild=true),
]

@testset "Randomized round-trip properties" begin
    rng = MersenneTwister(2026)

    for iter in 1:12
        users = rand(rng, 5:40)
        items = rand(rng, 5:30)
        density = 0.05 + 0.35 * rand(rng)
        T = rand(rng, Bool) ? Float32 : Float64
        X = _random_sparse(rng, users, items, density, T)
        n_users, n_items = size(X)
        k = rand(rng, 1:min(5, n_items))
        seed = rand(rng, 1:10^6)

        @testset "draw $iter ($(n_users)×$(n_items), $T, ρ=$(round(density; digits=2)))" begin
            # The generator duplicates triplets; accumulation must have happened
            @test nnz(X) > 0

            for spec in _PROPERTY_MODELS
                name = spec.name
                m = spec.model()
                Xm = spec.square ? _random_sparse(rng, n_items, n_items, density, T;
                                                  positive=true, duplicates=true) : X

                # LogisticMF needs at least one unobserved item per user for
                # negative sampling; skip it on draws with fully-observed rows.
                if name == "LogisticMF"
                    X_csr = to_csr(Xm)
                    max_row_nnz = maximum(length(nzrange(X_csr, u)) for u in 1:n_users)
                    max_row_nnz >= size(Xm, 2) && continue
                end

                @testset "$name" begin
                    fit!(m, Xm; rng=MersenneTwister(seed))
                    @test m.is_fitted

                    preds = nothing
                    if spec.explicit
                        # Explicit contract: dense rating predictions (score),
                        # finite and shape-consistent; no recommend.
                        S = score(m, Xm)
                        @test size(S) == size(Xm)
                        @test all(isfinite, Matrix(S))
                        @test !applicable(recommend, m, Xm)
                    else
                        preds = recommend(m, Xm; k=k)
                        _assert_valid_predictions(preds, Xm, k, size(Xm, 2))
                    end

                    # Score matrix is finite and consistent in shape
                    S = score(m, Xm)
                    @test size(S) == size(Xm)
                    @test all(isfinite, Matrix(S))

                    # Deterministic re-fit reproduces recommendations
                    # (BPR and GloVe are Hogwild by design and are excluded;
                    # explicit models compare score instead)
                    if !(name in ("BPR", "GloVe"))
                        m2 = spec.model()
                        fit!(m2, Xm; rng=MersenneTwister(seed))
                        if spec.explicit
                            @test score(m2, Xm) == S
                        else
                            @test recommend(m2, Xm; k=k) == preds
                        end
                    end

                    if !spec.explicit
                        # k saturation: k > n_items clamps to n_items
                        @test size(recommend(m, Xm; k=n_items + 10), 2) == n_items
                    end

                    # Persistence round-trip reproduces predictions
                    tmpfile = tempname() * ".jls"
                    try
                        save_model(m, tmpfile)
                        loaded = load_model(tmpfile)
                        @test loaded.is_fitted
                        if spec.explicit
                            @test score(loaded, Xm) == S
                        else
                            @test recommend(loaded, Xm; k=k) == preds
                        end
                    finally
                        rm(tmpfile; force=true)
                    end
                end
            end
        end
    end
end

@testset "Signed feedback stays valid" begin
    # Implicit-feedback models (IALS/LMF/BPR) assume non-negative ratings by
    # design; the signed-feedback models are exercised separately.
    rng = MersenneTwister(99)
    signed_models = [
        () -> WMF(rank=3, max_iter=3, feedback=Explicit, verbose=false),
        () -> EASE(λ=100.0, verbose=false),
        () -> SLIM(λ_l1=0.01, max_iter=3, verbose=false),
        () -> ADMMSLIM(λ_l1=0.01, max_iter=3, verbose=false),
        () -> ItemKNN(k=3, verbose=false),
        () -> SoftImpute(rank=3, max_iter=2, verbose=false),
    ]

    for iter in 1:5
        users = rand(rng, 10:40)
        items = rand(rng, 8:25)
        T = rand(rng, Bool) ? Float32 : Float64
        X = _random_sparse(rng, users, items, 0.2, T; signed=true)

        @testset "signed draw $iter ($T)" begin
            for (i, factory) in enumerate(signed_models)
                @testset "model $i" begin
                    m = factory()
                    fit!(m, X; rng=MersenneTwister(3))
                    S = score(m, X)
                    @test all(isfinite, S isa SparseMatrixCSC ? Matrix(S) : S)
                    if applicable(recommend, m, X)
                        _assert_valid_predictions(recommend(m, X; k=3), X, 3, items)
                    end
                end
            end
        end
    end
end

@testset "Numerical extremes stay finite" begin
    # Feedback magnitudes far outside the typical 0-1 range. The bands are
    # kept scale-uniform: gramian-based solvers (Cholesky) are well-posed
    # only when the gramian's conditioning stays within the float eps, so
    # values spanning many orders of magnitude in one matrix are out of
    # scope (they legitimately fail with PosDefException or NaN).
    rng = MersenneTwister(77)
    robust_models = [
        () -> WMF(rank=3, max_iter=3, verbose=false),
        () -> IALS(rank=3, max_iter=2, verbose=false),
        () -> EALS(rank=3, max_iter=2, verbose=false),
        () -> EASE(λ=100.0, verbose=false),
        () -> SLIM(λ_l1=0.01, max_iter=3, verbose=false),
        () -> ADMMSLIM(λ_l1=0.01, max_iter=3, verbose=false),
        () -> ItemKNN(k=3, verbose=false),
        () -> SoftImpute(rank=3, max_iter=2, verbose=false),
    ]
    bands = [
        (Float32, 1e-4, 1e-3),  # tiny but uniform
        (Float32, 1e3, 1e4),    # large but uniform
        (Float64, 1e-8, 1e-7),
        (Float64, 1e5, 1e6),
    ]

    for (i, (T, lo, hi)) in enumerate(bands)
        users = rand(rng, 10:40)
        items = rand(rng, 8:25)
        X = _random_sparse(rng, users, items, 0.15, T; lo=lo, hi=hi)

        @testset "extremes draw $i ($T, [$lo, $hi])" begin
            for (j, factory) in enumerate(robust_models)
                m = factory()
                @testset "model $j" begin
                    fit!(m, X; rng=MersenneTwister(3))
                    S = score(m, X)
                    @test all(isfinite, S isa SparseMatrixCSC ? Matrix(S) : S)
                    if applicable(recommend, m, X)
                        preds = recommend(m, X; k=3)
                        _assert_valid_predictions(preds, X, 3, items)
                    end
                end
            end
        end
    end
end
