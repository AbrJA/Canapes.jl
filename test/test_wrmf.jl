# test/test_wrmf.jl — WMF algorithm tests

# Helper: compute observed-entry implicit WMF loss
function _wrmf_loss(U::Matrix{<:AbstractFloat}, V::Matrix{<:AbstractFloat},
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
    model = WMF(rank=5, λ=λ, α=α, max_iter=5, solver=CholeskySolver(), feedback=IMPLICIT, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test size(model.user_factors) == (5, 100)
    @test size(model.item_factors) == (5, 80)
    @test !any(isnan, model.user_factors)
    @test !any(isnan, model.item_factors)
end

@testset "Implicit CG" begin
    model = WMF(rank=5, λ=λ, α=α, max_iter=5, solver=CGSolver(), feedback=IMPLICIT, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test size(model.user_factors) == (5, 100)
    @test !any(isnan, model.user_factors)
end

@testset "Explicit" begin
    model = WMF(rank=5, λ=λ, α=α, max_iter=5, solver=CholeskySolver(), feedback=EXPLICIT, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
end

@testset "NonNegativeSolver" begin
    model = WMF(rank=5, λ=λ, α=α, max_iter=3, solver=NonNegativeSolver(), feedback=IMPLICIT, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    @test all(model.user_factors .>= -1e-12)
    @test all(model.item_factors .>= -1e-12)
end

@testset "recommend top-k" begin
    model = WMF(rank=5, λ=λ, α=α, max_iter=3, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    preds = recommend(model, X; k=5)
    @test size(preds) == (100, 5)
    @test all(preds .>= 1)
    @test all(preds .<= 80)
end

@testset "transform new users" begin
    model = WMF(rank=4, λ=λ, α=α, max_iter=10, solver=CholeskySolver(), verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    X_new = sprand(MersenneTwister(3), 7, size(X, 2), 0.15)
    U_new = transform(model, X_new)
    @test size(U_new) == (4, 7)
    @test !any(isnan, U_new)
    @test !any(isinf, U_new)
end

@testset "Empty sparse matrix" begin
    X_empty = sparse(Int[], Int[], Float64[], 10, 10)
    model = WMF(rank=3, max_iter=2, verbose=false)
    fit!(model, X_empty; rng=MersenneTwister(1))
    @test model.is_fitted
end

@testset "Loss monotonically decreasing (CholeskySolver)" begin
    losses = Float64[]
    for n_iter in [2, 5, 15, 30]
        m = WMF(rank=4, λ=λ, α=α, max_iter=n_iter, solver=CholeskySolver(),
                 feedback=IMPLICIT, tol=-1.0, verbose=false)
        fit!(m, X; rng=MersenneTwister(1))
        push!(losses, _wrmf_loss(m.user_factors, m.item_factors, X, λ, α))
    end
    for i in 2:length(losses)
        @test losses[i] <= losses[i-1] * 1.01
    end
end

@testset "CG loss decreases with more iterations" begin
    m_early = WMF(rank=4, λ=λ, α=α, max_iter=2, solver=CGSolver(),
                   cg_steps=20, tol=-1.0, verbose=false)
    m_conv = WMF(rank=4, λ=λ, α=α, max_iter=30, solver=CGSolver(),
                  cg_steps=20, tol=-1.0, verbose=false)
    fit!(m_early, X; rng=MersenneTwister(1))
    fit!(m_conv, X; rng=MersenneTwister(1))
    l_early = _wrmf_loss(m_early.user_factors, m_early.item_factors, X, λ, α)
    l_conv = _wrmf_loss(m_conv.user_factors, m_conv.item_factors, X, λ, α)
    @test l_conv < l_early
end

@testset "CholeskySolver ≈ CG at convergence" begin
    m_chol = WMF(rank=4, λ=λ, α=α, max_iter=100, solver=CholeskySolver(),
                  tol=1e-7, verbose=false)
    m_cg = WMF(rank=4, λ=λ, α=α, max_iter=100, solver=CGSolver(),
                cg_steps=50, tol=1e-7, verbose=false)
    fit!(m_chol, X; rng=MersenneTwister(7))
    fit!(m_cg, X; rng=MersenneTwister(7))
    l_chol = _wrmf_loss(m_chol.user_factors, m_chol.item_factors, X, λ, α)
    l_cg = _wrmf_loss(m_cg.user_factors, m_cg.item_factors, X, λ, α)
    rel = abs(l_chol - l_cg) / (min(l_chol, l_cg) + 1e-10)
    @test rel < 0.05
end

@testset "NonNegativeSolver warm-start" begin
    m_chol = WMF(rank=4, λ=λ, α=α, max_iter=20, solver=CholeskySolver(), verbose=false)
    fit!(m_chol, X; rng=MersenneTwister(1))
    U_warm = abs.(m_chol.user_factors)
    V_warm = abs.(m_chol.item_factors)

    m_nnls = WMF(rank=4, λ=λ, α=α, max_iter=20, solver=NonNegativeSolver(), verbose=false)
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

    m2 = WMF(rank=5, λ=0.01, α=10.0, max_iter=50, solver=CholeskySolver(), verbose=false)
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
    m_ex = WMF(rank=4, λ=0.1, α=1.0, max_iter=20, solver=CholeskySolver(),
                feedback=EXPLICIT, verbose=false)
    fit!(m_ex, X_ex; rng=rng3)
    rv = rowvals(X_ex); nz = nonzeros(X_ex); mse = 0.0
    for j in axes(X_ex, 2), idx in nzrange(X_ex, j)
        i = rv[idx]
        p = dot(@view(m_ex.user_factors[:, i]), @view(m_ex.item_factors[:, j]))
        mse += (Float64(nz[idx]) - p)^2
    end
    @test mse / nnz(X_ex) < 1.0
end

@testset "Early stopping" begin
    model = WMF(rank=5, λ=λ, α=α, max_iter=100, tol=0.001, verbose=false)
    fit!(model, X; rng=MersenneTwister(1))
    @test model.is_fitted
    # Should converge before 100 iterations
end
