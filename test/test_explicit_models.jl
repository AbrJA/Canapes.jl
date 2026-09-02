# test/test_explicit_models.jl — BaselineOnly, SlopeOne, PearsonKNN

function _rating_matrix(rng, n_u, n_i, ρ)
    rows, cols = Int[], Int[]
    for u in 1:n_u, i in 1:n_i
        rand(rng) < ρ && (push!(rows, u); push!(cols, i))
    end
    vals = [3.5 + 0.5 * sin(r) + 0.4 * cos(c) + randn(rng) * 0.5
            for (r, c) in zip(rows, cols)]
    sparse(rows, cols, vals, n_u, n_i)
end

@testset "Explicit models: contract" begin
    rng = MersenneTwister(11)
    X = _rating_matrix(rng, 80, 50, 0.25)
    for (name, mk) in (("BaselineOnly", () -> BaselineOnly(verbose=false)),
                       ("SlopeOne", () -> SlopeOne(verbose=false)),
                       ("PearsonKNN", () -> PearsonKNN(k=20, verbose=false)))
        m = mk()
        fit!(m, X)
        @test m.is_fitted
        @test m isa AbstractExplicitModel
        @test !(m isa AbstractRecommender)
        @test !applicable(recommend, m, X)
        P = predict(m, X)
        @test size(P) == size(X)
        @test all(isfinite, P)
        @test score(m, X) == P
        errs = [abs(P[r, c] - v) for (r, c, v) in zip(findnz(X)[1], findnz(X)[2], findnz(X)[3])]
        @test rmse(P, X) ≈ sqrt(sum(abs2, errs) / length(errs)) atol=1e-6
        # dimension guards
        @test_throws DimensionMismatch predict(m, sparse([1,2,3], [1,2,4], [3.0,4.0,5.0], 5, 5))
    end
end

@testset "BaselineOnly" begin
    rng = MersenneTwister(3)
    n_u, n_i = 60, 40
    rows, cols = Int[], Int[]
    for u in 1:n_u, i in 1:n_i
        rand(rng) < 0.3 && (push!(rows, u); push!(cols, i))
    end
    μ0, ub, ib = 3.0, [0.8 * sin(u) for u in 1:n_u], [0.6 * cos(i) for i in 1:n_i]
    vals = [clamp(μ0 + ub[r] + ib[c] + randn(rng) * 0.4, 1, 5) for (r, c) in zip(rows, cols)]
    X = sparse(rows, cols, vals, n_u, n_i)

    m = BaselineOnly(λ=0.02, max_iter=20, verbose=false)
    fit!(m, X)
    P = predict(m, X)
    # global mean captured
    @test m.global_mean ≈ sum(vals) / length(vals) atol=1e-6
    # prediction formula exact
    @test P[5, 7] ≈ m.global_mean + m.user_bias[5] + m.item_bias[7] atol=1e-6
    # captures the user/item bias structure — beats the plain global mean
    base_err = sqrt(sum(((v - μ0)^2 for (_, _, v) in zip(rows, cols, vals))) / length(vals))
    errs = [abs(P[r, c] - v) for (r, c, v) in zip(rows, cols, vals)]
    @test sqrt(sum(abs2, errs) / length(errs)) < base_err
    # pairwise score matches matrix positions
    pv = score(m, [5, 6], [7, 8])
    @test pv[1] ≈ P[5, 7] atol=1e-6
    @test pv[2] ≈ P[6, 8] atol=1e-6
end

@testset "SlopeOne" begin
    # Hand-checked tiny example
    X = sparse([1, 1, 2, 2, 3, 3], [1, 2, 1, 3, 2, 3], [1.0, 2.0, 1.0, 4.0, 2.0, 4.0], 3, 3)
    m = SlopeOne(verbose=false)
    fit!(m, X)
    # dev(1,2) = mean(r_1 - r_2) over users rating both: u1: 1-2=-1; u3: (0-2)=-2 → wait:
    # users with both: u1 (1,2): dev(1,2) += 1-2 = -1, freq(1,2)=1; u3 has no item 1.
    # dev(2,1) += 2-1 = 1
    @test m.dev[1, 2] == -1.0
    @test m.freq[1, 2] == 1
    @test m.dev[2, 1] == 1.0
    # dev(1,3): u2: 1-4 = -3, freq=1
    @test m.dev[1, 3] == -3.0
    # predict for user1 at item 3: pairs with j ∈ {1,2}:
    #   j=1: dev(3,1) + r_1 = 3 + 1 = 4, freq=1 → contributes (dev(3,1)+1)·1
    #   actually i=3: j in {1,2}: (dev(3,1)+1)*1 + (dev(3,2)+2)*1 ... dev(3,2): u3: 4-2=2
    #   num = (3+1)*1 + (2+2)*1 = 8, den = 2 → 4.0
    P = predict(m, X)
    @test P[1, 3] ≈ 4.0 atol=1e-5
    # rated items are predicted too, via their pair deviations:
    # ŷ_1,1 = dev(1,2) + r_1,2 = -1 + 2 = 1.0
    @test P[1, 1] ≈ 1.0 atol=1e-5
    @test P[2, 2] ≈ m.dev[2, 1] + 1.0 atol=1e-5   # dev(2,1) + r_2,1 = 1 + 1
    # pairwise score is intentionally unsupported
    @test_throws ArgumentError score(m, [1], [3])

    # quality on the biased synthetic set
    rng = MersenneTwister(11)
    Xb = _rating_matrix(rng, 80, 50, 0.25)
    m2 = SlopeOne(verbose=false)
    fit!(m2, Xb)
    @test rmse(predict(m2, Xb), Xb) < 1.0
end

@testset "PearsonKNN" begin
    rng = MersenneTwister(11)
    X = _rating_matrix(rng, 80, 50, 0.25)
    m = PearsonKNN(k=20, verbose=false)
    fit!(m, X)
    P = predict(m, X)
    @test rmse(P, X) < 1.0
    # prediction = user mean + normalized neighbor contribution; a user with
    # no positive-neighbor signal (or whose neighbors never rated the item)
    # falls back to the user mean
    μu = m.user_mean
    for u in 1:80
        if isempty(m.neighbors[u])
            @test all(P[u, :] .≈ μu[u])
        end
    end
    # pairwise matches the matrix at the same positions (Surprise semantics)
    pv = score(m, [1, 2, 3], [4, 5, 6])
    @test all(isapprox.(pv, [P[1, 4], P[2, 5], P[3, 6]]; atol=1e-5))
    # k bounds the number of neighbors
    @test all(length(m.neighbors[u]) <= 20 for u in 1:80)
    # more neighbors do not hurt that less-populated users still get means
    @test all(all(isfinite, s for (_, s) in m.neighbors[u]) for u in 1:80)
    # invalid args
    @test_throws ArgumentError PearsonKNN(k=0)
    @test_throws ArgumentError PearsonKNN(min_k=0)
end

@testset "Explicit holdout workflow" begin
    rng = MersenneTwister(7)
    X = _rating_matrix(rng, 60, 40, 0.2)
    X_tr, X_te = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))
    for (name, mk) in (("BaselineOnly", () -> BaselineOnly(verbose=false)),
                       ("SlopeOne", () -> SlopeOne(verbose=false)),
                       ("PearsonKNN", () -> PearsonKNN(k=15, verbose=false)))
        m = mk()
        fit!(m, X_tr)
        r_tr = rmse(predict(m, X_tr), X_te)
        r_mae = mae(predict(m, X_tr), X_te)
        @test r_tr >= 0.0 && r_mae >= 0.0
        @test all(isfinite, (r_tr, r_mae))
        # baselines must beat a constant-mean predictor on these biased data
        μ = sum(nonzeros(X_tr)) / nnz(X_tr)
        const_rmse = sqrt(sum(((v - μ)^2 for v in nonzeros(X_te))) / nnz(X_te))
        @test r_tr < const_rmse
    end
end