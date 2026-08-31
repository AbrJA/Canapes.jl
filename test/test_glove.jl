# test/test_glove.jl — GloVe algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    n = 50
    A = sprand(rng, n, n, 0.1)
    A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1

    model = GloVe(rank=10, x_max=10.0, lr=0.15, max_iter=5, verbose=false)
    fit!(model, A; rng=rng)

    @test model.is_fitted
    @test size(model.W_main) == (10, n)
    @test size(model.W_ctx) == (10, n)
    @test length(model.loss_history) >= 1
    @test all(isfinite, model.loss_history)

    emb = embeddings(model)
    @test size(emb) == (10, n)
    @test all(isfinite, emb)
end

@testset "Cost generally decreasing" begin
    rng = MersenneTwister(42)
    n = 80
    A = sprand(rng, n, n, 0.1); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=5, x_max=10.0, lr=0.15, max_iter=20, verbose=false)
    fit!(model, A; rng=rng)
    @test length(model.loss_history) == 20
    @test sum(diff(model.loss_history) .< 0) >= 15
end

@testset "Embeddings finite with non-trivial variance" begin
    rng = MersenneTwister(7)
    n = 60
    A = sprand(rng, n, n, 0.15); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=8, x_max=10.0, lr=0.15, max_iter=30, verbose=false)
    fit!(model, A; rng=rng)
    emb = embeddings(model)
    @test all(isfinite, emb)
    mx = sum(emb) / length(emb)
    @test sqrt(sum((emb .- mx).^2) / length(emb)) > 0.01
end

@testset "Block structure: within > cross community similarity" begin
    I = vcat([i for i in 1:10 for j in i+1:10],
             [i for i in 11:20 for j in i+1:20])
    J = vcat([j for i in 1:10 for j in i+1:10],
             [j for i in 11:20 for j in i+1:20])
    V = fill(5.0, length(I))
    A = sparse(vcat(I,J), vcat(J,I), vcat(V,V), 20, 20)
    model = GloVe(rank=4, x_max=10.0, lr=0.15, max_iter=50, verbose=false)
    fit!(model, A; rng=MersenneTwister(1))
    emb = embeddings(model)
    vcos(a, b) = dot(a, b) / (norm(a)*norm(b) + 1e-8)
    vbar(v) = sum(v) / length(v)
    s_within = vbar([vcos(emb[:,i], emb[:,j]) for i in 1:10 for j in i+1:10])
    s_cross = vbar([vcos(emb[:,i], emb[:,j]) for i in 1:10 for j in 11:20])
    @test s_within > s_cross
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    n = 50
    A = sprand(rng, n, n, 0.1); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=5, x_max=10.0, lr=0.15, tol=0.001, max_iter=100, verbose=false)
    fit!(model, A; rng=rng)
    @test model.is_fitted
    # Should converge before 100 iterations
    @test length(model.loss_history) <= 100
end

@testset "Regularization" begin
    rng = MersenneTwister(42)
    n = 40
    A = sprand(rng, n, n, 0.2); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1

    m_noreg = GloVe(rank=5, x_max=10.0, lr=0.15, λ=0.0, max_iter=20, verbose=false)
    m_reg = GloVe(rank=5, x_max=10.0, lr=0.15, λ=1.0, max_iter=20, verbose=false)
    fit!(m_noreg, A; rng=MersenneTwister(1))
    fit!(m_reg, A; rng=MersenneTwister(1))

    # Regularization should produce smaller embeddings
    @test sum(abs2, embeddings(m_reg)) < sum(abs2, embeddings(m_noreg))
end

@testset "Empty input" begin
    @test_throws ArgumentError fit!(GloVe(rank=2, max_iter=1, verbose=false), spzeros(3, 3))
    @test_throws ArgumentError fit!(GloVe(rank=2, max_iter=1, verbose=false), spzeros(0, 0))
end

@testset "Deterministic across thread partitions" begin
    # The reordered word-ownership epoch is embarrassingly parallel over words:
    # results must be bit-identical regardless of how the work is chunked.
    rng = MersenneTwister(42)
    n = 60
    A = sprand(rng, n, n, 0.1); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1

    X_csr = Gideon.to_csr(A)
    perm = Gideon._glove_csc_to_csr_pos(X_csr, A)

    function train_epochs(nt, nepochs)
        m = GloVe(rank=8, x_max=10.0, lr=0.15, max_iter=1, verbose=false)
        fit!(m, A; rng=MersenneTwister(7))
        grad_buf = zeros(Float32, nnz(A))
        cost_buf = zeros(Float32, nnz(A))
        losses = Float32[]
        for _ in 1:nepochs
            push!(losses, Gideon._glove_epoch!(m, X_csr, A, grad_buf, cost_buf,
                                               perm, nothing; nt=nt))
        end
        (m, losses)
    end

    m1, l1 = train_epochs(1, 5)
    m4, l4 = train_epochs(4, 5)
    m8, l8 = train_epochs(8, 5)

    # Loss curves and all learned state are bit-identical across chunk counts.
    @test isequal(l1, l4)
    @test isequal(l4, l8)
    @test isequal(m1.W_main, m4.W_main)
    @test isequal(m1.W_main, m8.W_main)
    @test isequal(m1.W_ctx, m4.W_ctx)
    @test isequal(m1.b_main, m4.b_main)
    @test isequal(m1.b_ctx, m4.b_ctx)
    @test isequal(m1.grad_W_main, m4.grad_W_main)
    @test isequal(m1.grad_W_ctx, m4.grad_W_ctx)
    @test isequal(m1.grad_b_main, m4.grad_b_main)
    @test isequal(m1.grad_b_ctx, m4.grad_b_ctx)
end

@testset "Shuffled pair order stays finite" begin
    rng = MersenneTwister(42)
    n = 40
    A = sprand(rng, n, n, 0.2); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=5, x_max=10.0, lr=0.15, max_iter=10,
                  shuffle=true, verbose=false)
    fit!(model, A; rng=rng)
    @test model.is_fitted
    @test all(isfinite, embeddings(model))
end
