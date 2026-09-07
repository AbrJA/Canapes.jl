# Contract tests adapted from implicit's recommender_base_test.py.
# These test API invariants shared by recommenders, not implementation details.

function _reference_interactions()
    rows = repeat(1:8, inner=3)
    cols = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6,
            5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]
    sparse(rows, cols, ones(length(rows)), 8, 10)
end

const RECOMMENDER_FACTORIES = [
    (name="WeightedMF", factory=() -> WeightedMF(rank=3, max_iter=2, verbose=false)),
    (name="CachedALS", factory=() -> CachedALS(rank=3, max_iter=2, verbose=false)),
    (name="ElementwiseALS", factory=() -> ElementwiseALS(rank=3, max_iter=2, verbose=false)),
    (name="PairwiseRanking", factory=() -> PairwiseRanking(rank=3, max_iter=2, verbose=false)),
    (name="LogisticMF", factory=() -> LogisticMF(rank=3, max_iter=2, verbose=false)),
    (name="ShallowAutoencoder", factory=() -> ShallowAutoencoder(λ=10.0, verbose=false)),
    (name="SparseLinearModel", factory=() -> SparseLinearModel(max_iter=2, verbose=false)),
    (name="ItemKNN", factory=() -> ItemKNN(k=3, verbose=false)),
]

@testset "Reference-style recommender contracts" begin
    X = _reference_interactions()

    for spec in RECOMMENDER_FACTORIES
        @testset "$(spec.name)" begin
            model = spec.factory()
            fit!(model, X; rng=MersenneTwister(42))
            predictions = recommend(model, X; k=3)

            @test size(predictions) == (size(X, 1), 3)
            @test all(1 .<= predictions .<= size(X, 2))
            @test all(isfinite, score(model, X))

            for u in axes(X, 1)
                seen = Set(findall(!iszero, X[u, :]))
                @test isempty(intersect(seen, Set(predictions[u, :])))
                @test length(Set(predictions[u, :])) == 3
            end
        end
    end
end

@testset "Reference-style score ordering" begin
    X = _reference_interactions()
    for spec in RECOMMENDER_FACTORIES
        @testset "$(spec.name)" begin
            model = spec.factory()
            fit!(model, X; rng=MersenneTwister(42))
            predictions = recommend(model, X; k=3)
            scores = score(model, X)
            for u in axes(predictions, 1)
                @test issorted(scores[u, predictions[u, :]]; rev=true)
            end
        end
    end
end

@testset "Reference-style deterministic repeatability" begin
    X = _reference_interactions()
    for spec in [RECOMMENDER_FACTORIES[6], RECOMMENDER_FACTORIES[7], RECOMMENDER_FACTORIES[8]]
        @testset "$(spec.name)" begin
            first = spec.factory()
            second = spec.factory()
            fit!(first, X; rng=MersenneTwister(42))
            fit!(second, X; rng=MersenneTwister(42))
            @test recommend(first, X; k=3) == recommend(second, X; k=3)
            @test score(first, X) == score(second, X)
        end
    end
end

@testset "Reference-style edge cases" begin
    X = _reference_interactions()
    X[8, :] .= 0
    X[:, 10] .= 0

    model = WeightedMF(rank=3, max_iter=2, verbose=false, T=Float64)
    fit!(model, X; rng=MersenneTwister(42))
    predictions = recommend(model, X; k=3)

    @test eltype(model.user_factors) == Float64
    @test eltype(model.item_factors) == Float64
    @test size(predictions) == (8, 3)
    @test all(isfinite, model.user_factors)
    @test all(isfinite, model.item_factors)
end
