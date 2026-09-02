# test/test_randomwalk.jl — RandomWalk (RP3β) algorithm tests

# Dense transcription of the reference R3β formula (similaripy `py_rp3beta`):
# with m = items×users, P1 = rows l1-normalized, P2 = rows of mᵀ normalized,
# entries ^ α, W = P1@P2, columns scaled by pop^{-β}.
function reference_rp3beta(X::SparseMatrixCSC{Tv,Ti}, α::Float64, β::Float64;
                           T::Type=Float64) where {Tv,Ti}
    n_users, n_items = size(X)
    m = Matrix{T}(X')                                   # items × users
    pop = vec(sum(m, dims=2))
    pop_inv = [pop[j] == 0 ? 0.0 : 1.0 / pop[j]^β for j in 1:n_items]

    P1 = m ./ max.(sum(m, dims=2), 1.0)
    P2 = permutedims(m ./ max.(sum(m, dims=1), 1.0))
    P1 = P1 .^ α
    P2 = P2 .^ α
    W = P1 * P2
    W = W * Diagonal(pop_inv)
    for j in 1:n_items
        W[j, j] = 0.0
    end
    W
end

@testset "RandomWalk basic fit" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 80, 60, 0.05)
    m = RandomWalk(β=0.0, k=60, verbose=false)
    fit!(m, X)
    @test m.is_fitted
    @test size(m.W) == (60, 60)
    @test nnz(m.W) <= 60 * 60
    @test all(isfinite, nonzeros(m.W))
    @test all(nonzeros(m.W) .>= 0)
    # no self-recommendation
    @test all(getindex(m.W, i, i) == 0 for i in 1:60)
    @test size(recommend(m, X; k=10)) == (80, 10)
    @test score(m, X) isa SparseMatrixCSC
end

@testset "RandomWalk matches the RP3β reference (α=1, β=0/0.6)" begin
    rng = MersenneTwister(7)
    X = sprand(rng, 30, 20, 0.15)
    nonzeros(X) .= 1.0  # binary interactions
    for β in (0.0, 0.6)
        m = RandomWalk(α=1.0, β=β, k=nothing, verbose=false, T=Float64)
        fit!(m, X)
        ref = reference_rp3beta(X, 1.0, β; T=Float64)
        @test Matrix(m.W) ≈ ref atol=1e-8
    end
end

@testset "RandomWalk α-power matches reference" begin
    rng = MersenneTwister(7)
    # weighted interactions so the α power is not a no-op
    X = sprand(rng, 30, 20, 0.15)
    nonzeros(X) .= 0.5 .+ 0.5 .* rand(MersenneTwister(3), nnz(X))
    for α in (0.5, 2.0)
        m = RandomWalk(α=α, β=0.0, k=nothing, verbose=false, T=Float64)
        fit!(m, X)
        ref = reference_rp3beta(X, α, 0.0; T=Float64)
        @test Matrix(m.W) ≈ ref atol=1e-8
    end
end

@testset "RandomWalk popularity penalization" begin
    rng = MersenneTwister(11)
    X = sprand(rng, 50, 40, 0.1)
    nonzeros(X) .= 1.0
    m0 = RandomWalk(β=0.0, k=nothing, verbose=false, T=Float64)
    m1 = RandomWalk(β=0.7, k=nothing, verbose=false, T=Float64)
    fit!(m0, X); fit!(m1, X)
    # heads (popular items) lose column mass with β>0; tails gain relative share
    pop = vec(sum(X .> 0; dims=1))
    head = argmax(pop)
    @test sum(m1.W[:, head]) < sum(m0.W[:, head])
    # penalized matrix keeps non-negativity and zero diagonal
    @test all(nonzeros(m1.W) .>= 0)
    @test all(getindex(m1.W, i, i) == 0 for i in 1:40)
end

@testset "RandomWalk invalid parameters" begin
    @test_throws ArgumentError RandomWalk(α=0.0)
    @test_throws ArgumentError RandomWalk(β=-0.1)
    @test_throws ArgumentError RandomWalk(k=0)
    @test_throws ArgumentError RandomWalk(k=-1)
    # negative interaction values are rejected before fitting
    X = sparse([1, 2], [1, 2], [-1.0, 1.0], 3, 3)
    m = RandomWalk(verbose=false)
    @test_throws ArgumentError fit!(m, X)
end