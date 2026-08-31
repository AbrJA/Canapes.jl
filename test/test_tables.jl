# test/test_tables.jl — Tables.jl integration

# `import` (not `using`) so DataFrames' exports (e.g. `transform`) do not
# leak into the shared test module and shadow Gideon's.
import DataFrames
import Tables

@testset "triplets_to_sparse" begin
    # NamedTuple of vectors (simplest Tables.jl compatible)
    table = (user=[1,1,2,3,3], item=[2,5,3,1,4], value=[1.0, 2.0, 1.0, 3.0, 1.0])
    X = triplets_to_sparse(table; user_col=:user, item_col=:item, value_col=:value)

    @test size(X) == (3, 5)
    @test X[1, 2] == 1.0
    @test X[1, 5] == 2.0
    @test X[2, 3] == 1.0
    @test X[3, 1] == 3.0
    @test X[3, 4] == 1.0
    @test nnz(X) == 5
end

@testset "triplets_to_sparse with DataFrame" begin
    # DataFrame is the primary Tables.jl source; it must use the columnar path
    df = DataFrames.DataFrame(user=[1,1,2,3,3], item=[2,5,3,1,4], value=[1.0, 2.0, 1.0, 3.0, 1.0])
    X = triplets_to_sparse(df; user_col=:user, item_col=:item, value_col=:value)
    @test size(X) == (3, 5)
    @test X[1, 2] == 1.0
    @test X[3, 1] == 3.0
    @test nnz(X) == 5

    # Same result as the NamedTuple equivalent
    nt = (user=df.user, item=df.item, value=df.value)
    @test triplets_to_sparse(nt; user_col=:user, item_col=:item, value_col=:value) == X
end

@testset "triplets_to_sparse with explicit dimensions" begin
    table = (user=[1,2], item=[1,2], value=[1.0, 1.0])
    X = triplets_to_sparse(table; user_col=:user, item_col=:item,
                               value_col=:value, n_users=10, n_items=20)
    @test size(X) == (10, 20)
    @test nnz(X) == 2
end

@testset "triplets_to_sparse with Vector of NamedTuples" begin
    rows = [(user=1, item=3, value=1.0),
            (user=2, item=1, value=2.0),
            (user=3, item=2, value=1.5)]
    X = triplets_to_sparse(rows; user_col=:user, item_col=:item, value_col=:value)
    @test size(X) == (3, 3)
    @test X[1, 3] == 1.0
    @test X[2, 1] == 2.0
    @test X[3, 2] == 1.5
end

@testset "triplets_to_sparse implicit (no value column)" begin
    table = (user=[1,1,2,2], item=[1,2,3,4])
    X = triplets_to_sparse(table; user_col=:user, item_col=:item, value_col=nothing)
    @test size(X) == (2, 4)
    @test all(nonzeros(X) .== 1.0)
end

@testset "T kwarg" begin
    table = (user=[1,2], item=[1,2], value=[1.0, 2.0])
    X32 = triplets_to_sparse(table; user_col=:user, item_col=:item,
                                 value_col=:value, T=Float32)
    @test eltype(X32) == Float32
    X64 = triplets_to_sparse(table; user_col=:user, item_col=:item,
                                 value_col=:value, T=Float64)
    @test eltype(X64) == Float64
    @test X32 ≈ X64
end

@testset "duplicate pairs accumulate" begin
    table = (user=[1,1,1], item=[2,2,2], value=[1.0, 2.0, 3.0])
    X = triplets_to_sparse(table; user_col=:user, item_col=:item, value_col=:value)
    @test nnz(X) == 1
    @test X[1, 2] == 6.0
end

@testset "invalid input errors" begin
    # Not a table at all
    @test_throws ArgumentError triplets_to_sparse(rand(3, 3))

    # Missing columns
    @test_throws ArgumentError triplets_to_sparse((a=[1], item=[2]))
    @test_throws ArgumentError triplets_to_sparse((user=[1], a=[2]))
    @test_throws ArgumentError triplets_to_sparse(
        (user=[1], item=[2]); value_col=:missing_col)

    # Empty table
    @test_throws ArgumentError triplets_to_sparse((user=Int[], item=Int[]))

    # Non-integer indices → clean ArgumentError, not InexactError/MethodError
    @test_throws ArgumentError triplets_to_sparse(
        (user=[1.5, 2.0], item=[1, 2]); user_col=:user, item_col=:item)
    @test_throws ArgumentError triplets_to_sparse(
        (user=["a", "b"], item=[1, 2]); user_col=:user, item_col=:item)

    # Indices out of the explicit bounds
    @test_throws ArgumentError triplets_to_sparse(
        (user=[1, 3], item=[1, 2]); n_users=2)

    # Mismatched column lengths
    @test_throws DimensionMismatch triplets_to_sparse(
        (user=[1, 2, 3], item=[1, 2]); user_col=:user, item_col=:item)
end

@testset "sparse_to_triplets roundtrip" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 10, 8, 0.2)

    triplets = sparse_to_triplets(X)
    @test length(triplets.user) == nnz(X)
    @test length(triplets.item) == nnz(X)
    @test length(triplets.value) == nnz(X)

    # Roundtrip
    X2 = triplets_to_sparse(
        (user=triplets.user, item=triplets.item, value=triplets.value);
        user_col=:user, item_col=:item, value_col=:value,
        n_users=size(X,1), n_items=size(X,2)
    )
    @test X2 ≈ X

    # Roundtrip via Tables.rowtable
    X3 = triplets_to_sparse(
        Tables.rowtable((user=triplets.user, item=triplets.item, value=triplets.value));
        user_col=:user, item_col=:item, value_col=:value,
        n_users=size(X,1), n_items=size(X,2)
    )
    @test X3 ≈ X
end
