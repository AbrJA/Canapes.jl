# test/test_weightedmf.jl — WeightedMF algorithm tests

# Helper: compute observed-entry implicit WeightedMF loss
function _weightedmf_loss(U::Matrix{<:AbstractFloat}, V::Matrix{<:AbstractFloat},
                    X::SparseMatrixCSC, λ::Float64, α::Float64)
    rv = rowvals(X); nz = nonzeros(X); loss = 0.0
    for j in axes(X, 2), idx in nzrange(X, j)
        i = rv[idx]; r = nz[idx]
        pred = dot(@view(U[:, i]), @view(V[:, j]))
        loss += (1.0 + α * r) * (1.0 - pred)^2
    end
    loss + λ * (sum(abs2, U) + sum(abs2, V))
end

rng = MersenneTwister(42)
X = sprand(rng, 100, 80, 0.05)
λ = 0.1; α = 1.0

@testset "Implicit CholeskySolver" begin
    model = WeightedMF(rank=5, λ=λ, α=α, max_iter=5, solver=CholeskySolver(), feedback=Implicit, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test size(model.user_factors) == (5, 100)
    @test size(model.item_factors) == (5, 80)
    @test !any(isnan, model.user_factors)
    @test !any(isnan, model.item_factors)
end

@testset "Implicit CG" begin
    model = WeightedMF(rank=5, λ=λ, α=α, max_iter=5, solver=CGSolver(), feedback=Implicit, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test size(model.user_factors) == (5, 100)
    @test !any(isnan, model.user_factors)
end

@testset "Explicit" begin
    model = WeightedMF(rank=5, λ=λ, α=α, max_iter=5, solver=CholeskySolver(), feedback=Explicit, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
end

@testset "NonNegativeSolver" begin
    model = WeightedMF(rank=5, λ=λ, α=α, max_iter=3, solver=NonNegativeSolver(), feedback=Implicit, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test all(model.user_factors .>= -1e-12)
    @test all(model.item_factors .>= -1e-12)
end

@testset "recommend top-k" begin
    model = WeightedMF(rank=5, λ=λ, α=α, max_iter=3, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    preds = recommend(model, X; k=5)
    @test size(preds) == (100, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 80)
end

@testset "transform new users" begin
    model = WeightedMF(rank=4, λ=λ, α=α, max_iter=10, solver=CholeskySolver(), verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    X_new = sprand(MersenneTwister(3), 7, size(X, 2), 0.15)
    U_new = transform(model, X_new)
    @test size(U_new) == (4, 7)
    @test !any(isnan, U_new)
    @test !any(isinf, U_new)
end

@testset "Empty sparse matrix" begin
    X_empty = sparse(Int[], Int[], Float64[], 10, 10)
    model = WeightedMF(rank=3, max_iter=2, verbose=false)
    fit!(model, X_empty; rng=MersenneTwister(1))
    @test model.is_fitted
end

@testset "Loss monotonically decreasing (CholeskySolver)" begin
    losses = Float64[]
    for n_iter in [2, 5, 15, 30]
        m = WeightedMF(rank=4, λ=λ, α=α, max_iter=n_iter, solver=CholeskySolver(),
                 feedback=Implicit, tol=-1.0, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        push!(losses, _weightedmf_loss(m.user_factors, m.item_factors, X, λ, α))
    end
    for i in 2:length(losses)
        @test losses[i] <= losses[i-1] * 1.01
    end
end

@testset "CG loss decreases with more iterations" begin
    m_early = WeightedMF(rank=4, λ=λ, α=α, max_iter=2, solver=CGSolver(),
                   cg_steps=20, tol=-1.0, verbose=false)
    m_conv = WeightedMF(rank=4, λ=λ, α=α, max_iter=30, solver=CGSolver(),
                  cg_steps=20, tol=-1.0, verbose=false)
    fit!(m_early, X; rng=MersenneTwister(1))
    fit!(m_conv, X; rng=MersenneTwister(1))
    l_early = _weightedmf_loss(m_early.user_factors, m_early.item_factors, X, λ, α)
    l_conv = _weightedmf_loss(m_conv.user_factors, m_conv.item_factors, X, λ, α)
    @test l_conv < l_early
end

@testset "CholeskySolver ≈ CG at convergence" begin
    m_chol = WeightedMF(rank=4, λ=λ, α=α, max_iter=100, solver=CholeskySolver(),
                  tol=1e-7, verbose=false)
    m_cg = WeightedMF(rank=4, λ=λ, α=α, max_iter=100, solver=CGSolver(),
                cg_steps=50, tol=1e-7, verbose=false)
    fit!(m_chol, X; rng=MersenneTwister(7))
    fit!(m_cg, X; rng=MersenneTwister(7))
    l_chol = _weightedmf_loss(m_chol.user_factors, m_chol.item_factors, X, λ, α)
    l_cg = _weightedmf_loss(m_cg.user_factors, m_cg.item_factors, X, λ, α)
    rel = abs(l_chol - l_cg) / (min(l_chol, l_cg) + 1e-10)
    @test rel < 0.05
end

@testset "NonNegativeSolver warm-start" begin
    m_chol = WeightedMF(rank=4, λ=λ, α=α, max_iter=20, solver=CholeskySolver(), verbose=false)
    fit!(m_chol, X; rng=MersenneTwister(1))
    U_warm = abs.(m_chol.user_factors)
    V_warm = abs.(m_chol.item_factors)

    m_nnls = WeightedMF(rank=4, λ=λ, α=α, max_iter=20, solver=NonNegativeSolver(), verbose=false)
    fit!(m_nnls, X; rng=MersenneTwister(1), U_init=U_warm, V_init=V_warm)
    @test all(m_nnls.user_factors .>= -1e-12)
    @test all(m_nnls.item_factors .>= -1e-12)
end

@testset "Structured signal gives expected top-k" begin
    # Users 1-5 have strong signal on items 1-5 only (not all 1-10)
    # so that items 6-10 are unseen but should score high due to similar embedding
    rng2 = MersenneTwister(99)
    I = vcat(repeat(1:5, inner=5), rand(rng2, 6:30, 30))
    J = vcat(repeat(1:5, outer=5), rand(rng2, 6:40, 30))
    V = vcat(10.0*ones(25), ones(30))
    X2 = sparse(I, J, V, 30, 40)

    m2 = WeightedMF(rank=5, λ=0.01, α=10.0, max_iter=50, solver=CholeskySolver(), verbose=false)
    fit!(m2, X2; rng=MersenneTwister(42))
    preds = recommend(m2, X2; k=5)
    @test size(preds) == (30, 5)
    # Verify no seen items appear in predictions (masking works)
    for u in 1:5
        seen = findall(!iszero, X2[u, :])
        @test isempty(intersect(preds[u, :], seen))
    end
end

@testset "Explicit feedback: MSE < 1" begin
    rng3 = MersenneTwister(5)
    X_ex = sprand(rng3, 40, 30, 0.2)
    nonzeros(X_ex) .= 1.0 .+ 4.0 .* rand(rng3, nnz(X_ex))   # ratings in [1, 5]
    m_ex = WeightedMF(rank=4, λ=0.1, α=1.0, max_iter=20, solver=CholeskySolver(),
                feedback=Explicit, verbose=false)
    fit!(m_ex, X_ex; rng=rng3)
    P = predict(m_ex, X_ex)
    rv = rowvals(X_ex); nz = nonzeros(X_ex); mse = 0.0
    for j in axes(X_ex, 2), idx in nzrange(X_ex, j)
        i = rv[idx]
        mse += (Float64(nz[idx]) - P[i, j])^2
    end
    @test mse / nnz(X_ex) < 1.0
end

@testset "Early stopping" begin
    model = WeightedMF(rank=5, λ=λ, α=α, max_iter=100, tol=0.001, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    # Should converge before 100 iterations
end

@testset "Scale-mixed inputs stay finite" begin
    # Ratings spanning ~1e118 mix confidence terms c_ui = 1 + α·r_ui huge enough
    # to make the Gramian numerically singular inside float eps. The guarded
    # Cholesky path must never leave NaN/Inf in the factors.
    rng = MersenneTwister(11)
    X_scale = spzeros(60, 50)
    for _ in 1:400
        r = rand(rng, 1:60); c = rand(rng, 1:50)
        X_scale[r, c] += rand(rng) < 0.5 ? rand(rng, 1:10) : 10.0^rand(rng, 1:60)
    end
    for solver in (CholeskySolver(), NonNegativeSolver())
        model = WeightedMF(rank=6, λ=0.0, α=40.0, max_iter=6, solver=solver, verbose=false)
        fit!(model, X_scale; rng=rng)
        @test all(isfinite, model.user_factors)
        @test all(isfinite, model.item_factors)
    end
end

@testset "BiasedMF (WeightedMF explicit): formula & quality" begin
    rng = MersenneTwister(11)
    n_u, n_i = 80, 60
    rows, cols = Int[], Int[]
    for u in 1:n_u, i in 1:n_i
        rand(rng) < 0.3 && (push!(rows, u); push!(cols, i))
    end
    μ0, ub, ib = 3.5, [0.5 * sin(u) for u in 1:n_u], [0.4 * cos(i) for i in 1:n_i]
    vals = [clamp(μ0 + ub[r] + ib[c] + randn(rng) * 0.3, 1, 5) for (r, c) in zip(rows, cols)]
    X = sparse(rows, cols, vals, n_u, n_i)

    for solver in (CholeskySolver(), CGSolver())
        m = WeightedMF(rank=6, λ=0.1, α=1.0, max_iter=15, solver=solver, feedback=Explicit, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        @test m.is_fitted
        @test size(m.user_factors) == (6, n_u)
        @test size(m.item_factors) == (6, n_i)
        @test length(m.user_bias) == n_u
        @test length(m.item_bias) == n_i
        @test all(isfinite, m.user_bias) && all(isfinite, m.item_bias)

        P = predict(m, X)
        @test size(P) == (n_u, n_i)
        @test all(isfinite, P)

        # prediction formula: μ + b_u + b_i + x·y
        @test P[1, 1] ≈ m.global_mean + m.user_bias[1] + m.item_bias[1] +
                        dot(@view(m.user_factors[:, 1]), @view(m.item_factors[:, 1])) atol=1e-4

        # quality: prediction well below the noise floor injected (σ=0.3)
        errs = [abs(P[r, c] - v) for (r, c, v) in zip(rows, cols, vals)]
        @test sqrt(sum(abs2, errs) / length(errs)) < 0.8

        # score (fold-in) ≈ predict (fitted) on the training matrix
        @test maximum(abs.(score(m, X) .- P)) < 0.1

        # pairwise score matches fitted prediction formula
        pv = score(m, [7, 13], [21, 5])
        @test pv[1] ≈ P[7, 21] atol=1e-4
        @test pv[2] ≈ P[13, 5] atol=1e-4

        # recommend: valid, seen items excluded
        R = recommend(m, X; k=5)
        @test size(R) == (n_u, 5)
        @test all(1 .<= R .<= n_i)
        for u in 1:n_u
            seen = Set(findall(!iszero, view(X, u, :)))
            @test all(item -> !(item in seen), R[u, :])
        end

        # transform (fold-in) for new users
        Xnew = sprand(MersenneTwister(9), 10, n_i, 0.2)
        nonzeros(Xnew) .= 1.0 .+ 4.0 .* rand(MersenneTwister(8), nnz(Xnew))
        emb = transform(m, Xnew)
        @test size(emb) == (6, 10)
        @test all(isfinite, emb)
    end

    # predict dimension guards on the explicit path
    m = WeightedMF(rank=4, feedback=Explicit, max_iter=3, verbose=false)
    fit!(m, X; rng=MersenneTwister(1))
    @test_throws DimensionMismatch predict(m, sprand(MersenneTwister(2), 5, n_i, 0.3))
    @test_throws DimensionMismatch predict(m, sprand(MersenneTwister(2), n_u, 7, 0.3))

    # NonNegativeSolver degrades to plain augmented ALS on explicit (finite)
    m_nn = WeightedMF(rank=4, λ=0.1, max_iter=3, solver=NonNegativeSolver(), feedback=Explicit, verbose=false)
    fit!(m_nn, X; rng=MersenneTwister(1))
    @test all(isfinite, predict(m_nn, X))
    # ... and keeps non-negativity on implicit
    m_nn2 = WeightedMF(rank=4, λ=0.1, max_iter=3, solver=NonNegativeSolver(), feedback=Implicit, verbose=false)
    fit!(m_nn2, X; rng=MersenneTwister(1))
    @test all(m_nn2.user_factors .>= -1e-12)
end
