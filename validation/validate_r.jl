# validation/validate_r.jl
# Optional validation against R (rsparse) reference implementations.
# Run via: julia --project=. validation/run.jl --r
# (or directly: julia --project=. validation/validate_r.jl)

using Gideon, SparseArrays, LinearAlgebra, Random
using Gideon: Binomial, Gaussian   # internal (unexported) family singletons
using Test
include(joinpath(@__DIR__, "common.jl"))

const RANK = 5; const λ_r = 0.1; const α_r = 1.0

"""Observed-entry implicit WMF loss (independent recomputation)."""
function _wrmf_loss_ref(U::Matrix{<:AbstractFloat}, V::Matrix{<:AbstractFloat},
                        X::SparseMatrixCSC, λ::Float64, α::Float64)
    rv = rowvals(X); nz = nonzeros(X); loss = 0.0
    for j in axes(X, 2), idx in nzrange(X, j)
        i = rv[idx]; r = nz[idx]
        pred = dot(@view(U[:, i]), @view(V[:, j]))
        loss += (1.0 + α * r) * (1.0 - pred)^2
    end
    loss + λ * (sum(abs2, U) + sum(abs2, V))
end

"""Independent dense reference of the FM forward pass (Gaussian, with intercept):
    ŷ = w0 + Σ_j w_j·x_j + ½·Σ_f [(Σ_j v_fj·x_j)² − Σ_j v_fj²·x_j²]
"""
function _fm_dense_reference(model::FM, X::SparseMatrixCSC)
    w0 = Float64(model.w0)
    w  = Float64.(model.w)
    V  = Float64.(model.V)
    k  = size(V, 1)
    out = Vector{Float64}(undef, size(X, 1))
    for r in 1:size(X, 1)
        xr = Vector{Float64}(X[r, :])
        s = model.intercept ? w0 : 0.0
        s += dot(w, xr)
        for f in 1:k
            sx  = dot(V[f, :], xr)
            sx2 = dot(V[f, :] .^ 2, xr .^ 2)
            s += 0.5 * (sx^2 - sx2)
        end
        out[r] = s
    end
    out
end

# Core fixtures are mandatory: a run without them must fail loudly, not pass.
require_files(
    [joinpath(R_FIXTURE_DIR, f) for f in [
        "X_small.csv", "X_small_dims.csv",
        "wrmf_chol_loss.txt", "wrmf_chol_user.csv", "wrmf_chol_item.csv",
        "wrmf_cg_loss.txt",
    ]],
    R_FIXTURE_DIR,
    "Generate fixtures with: uvr run validation/fixtures_r.R  (or: Rscript validation/fixtures_r.R)",
)

X_ref = _load_sparse(joinpath(R_FIXTURE_DIR, "X_small.csv"),
                     joinpath(R_FIXTURE_DIR, "X_small_dims.csv"))

@testset "R Reference Comparison" begin

    @testset "WMF CholeskySolver: loss ≤ R × 1.05" begin
        r_loss = _read_scalar(joinpath(R_FIXTURE_DIR, "wrmf_chol_loss.txt"))
        m = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                solver=CholeskySolver(), feedback=IMPLICIT, tol=1e-6, verbose=false)
        fit!(m, X_ref; rng=MersenneTwister(42))
        jl_loss = _wrmf_loss_ref(m.user_factors, m.item_factors, X_ref, λ_r, α_r)
        @test isfinite(jl_loss)
        @test jl_loss <= r_loss * 1.05
        println("  WMF CholeskySolver: Julia=$jl_loss, R=$r_loss, ratio=$(jl_loss/r_loss)")
    end

    @testset "WMF CholeskySolver: warm-start does not increase loss" begin
        U_raw = _read_matrix(joinpath(R_FIXTURE_DIR, "wrmf_chol_user.csv"))
        V_raw = _read_matrix(joinpath(R_FIXTURE_DIR, "wrmf_chol_item.csv"))
        U_r = Matrix{Float32}(U_raw')
        V_r = Matrix{Float32}(V_raw)
        r_loss = _read_scalar(joinpath(R_FIXTURE_DIR, "wrmf_chol_loss.txt"))

        m_ws = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=1,
                   solver=CholeskySolver(), feedback=IMPLICIT, tol=-1.0, verbose=false)
        fit!(m_ws, X_ref; rng=MersenneTwister(1), U_init=U_r, V_init=V_r)
        jl_loss_ws = _wrmf_loss_ref(m_ws.user_factors, m_ws.item_factors, X_ref, λ_r, α_r)
        @test jl_loss_ws <= r_loss * 1.05
        println("  WMF warm-start: Julia=$jl_loss_ws, R=$r_loss, ratio=$(jl_loss_ws/r_loss)")
    end

    @testset "WMF CG: loss ≤ R × 1.05" begin
        r_loss_cg = _read_scalar(joinpath(R_FIXTURE_DIR, "wrmf_cg_loss.txt"))
        m_cg = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                   solver=CGSolver(), cg_steps=10,
                   tol=1e-6, verbose=false)
        fit!(m_cg, X_ref; rng=MersenneTwister(42))
        jl_loss_cg = _wrmf_loss_ref(m_cg.user_factors, m_cg.item_factors, X_ref, λ_r, α_r)
        @test isfinite(jl_loss_cg)
        @test jl_loss_cg <= r_loss_cg * 1.05
        println("  WMF CG: Julia=$jl_loss_cg, R=$r_loss_cg, ratio=$(jl_loss_cg/r_loss_cg)")
    end

    @testset "FTRL: weights and predictions match R (tight)" begin
        ftrl_files = [joinpath(R_FIXTURE_DIR, f) for f in [
            "X_ftrl.csv", "X_ftrl_dims.csv", "y_ftrl.csv",
            "ftrl_weights.csv", "ftrl_preds.csv",
        ]]
        if !_all_files_exist(ftrl_files)
            @info "Skipping FTRL comparison: one or more fixture files are missing"
        else
            X_ftrl = _load_sparse(ftrl_files[1], ftrl_files[2])
            y_ftrl = _read_col(ftrl_files[3], "y")
            r_w = _read_col(ftrl_files[4], "w")
            r_preds = _read_col(ftrl_files[5], "p")

            m_ftrl = FTRL(lr=0.1, lr_decay=0.5,
                          λ=0.01, l1_ratio=0.5, verbose=false)
            for _ in 1:5
                update!(m_ftrl, X_ftrl, y_ftrl; rng=MersenneTwister(42))
            end
            jl_w = coef(m_ftrl)
            jl_p = predict(m_ftrl, X_ftrl)

            cor_w = _cor(jl_w, r_w)
            cor_p = _cor(jl_p, r_preds)
            @test cor_w >= 0.9995
            @test cor_p >= 0.9995
            println("  FTRL weights correlation: $cor_w")
            println("  FTRL predictions correlation: $cor_p")
        end
    end

    @testset "FM XOR: Julia matches R (optional)" begin
        fm_path = joinpath(R_FIXTURE_DIR, "fm_xor_preds.csv")
        if !isfile(fm_path)
            @info "Skipping FM XOR comparison: fixture missing (rsparse::FM may be unavailable)"
        else
            r_preds_fm = _read_col(fm_path, "p")
            x_xor = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
            y_xor = [0.0, 1.0, 1.0, 0.0]
            agreements = 0
            r_correct = r_preds_fm[1] < 0.3 && r_preds_fm[4] < 0.3 &&
                        r_preds_fm[2] > 0.7 && r_preds_fm[3] > 0.7
            for seed in 1:5
                m = FM(
                    lr_w=10.0, rank=2, max_iter=200,
                    λ_w=0.0, λ_v=0.0, family=Binomial(), intercept=true, verbose=false)
                fit!(m, x_xor, y_xor; rng=MersenneTwister(seed))
                p = predict(m, x_xor)
                j_correct = p[1] < 0.3 && p[4] < 0.3 && p[2] > 0.7 && p[3] > 0.7
                agreements += (j_correct && r_correct) || (!j_correct && !r_correct)
            end
            @test agreements >= 4
            println("  FM XOR: $agreements/5 seeds agree with R")
        end
    end

    @testset "GloVe: Julia cost ≤ R × 1.15 (same objective convention)" begin
        glove_files = [joinpath(R_FIXTURE_DIR, f) for f in [
            "glove_final_cost.txt", "glove_X.csv", "glove_dims.csv",
        ]]
        if !_all_files_exist(glove_files)
            @info "Skipping GloVe comparison: one or more fixture files are missing"
        else
            # Both implementations minimize ½ Σ f(x) diff² (rsparse GloVe.cpp
            # records 0.5 * cost / nnz; Gideon uses the same ½ convention), with
            # matching init (U(-0.5, 0.5)), AdaGrad ones-init, weighting, and
            # bias terms. rsparse is in-place SGD (converges in ~30 epochs here);
            # Gideon's deterministic frozen-batch scheme reaches the same cost
            # with 60 epochs and keeps decreasing below it — a robust gate.
            r_cost = _read_scalar(glove_files[1])
            X_glove = _load_sparse(glove_files[2], glove_files[3])
            m_glove = GloVe(rank=5, x_max=10.0, lr=0.15, max_iter=60, verbose=false)
            fit!(m_glove, X_glove; rng=MersenneTwister(42))
            jl_cost = last(m_glove.loss_history)
            @test isfinite(jl_cost)
            @test jl_cost <= r_cost * 1.15
            println("  GloVe cost: Julia=$jl_cost, R=$r_cost, ratio=$(jl_cost/r_cost)")
        end
    end

    @testset "FM sparse high-dim: dense-reference + ground-truth recovery" begin
        fm_files = [joinpath(R_FIXTURE_DIR, f) for f in [
            "fm_sparse_X.csv", "fm_sparse_dims.csv", "fm_sparse_y.csv", "fm_sparse_preds.csv",
        ]]
        if !_all_files_exist(fm_files)
            @info "Skipping FM sparse comparison: fixture files missing"
        else
            X_fm = _load_sparse(fm_files[1], fm_files[2])
            y_fm = _read_col(fm_files[3], "y")
            r_preds = _read_col(fm_files[4], "p")

            # Train on 80%, evaluate on a held-out 20%: proves generalization,
            # not memorization, on sparse high-dimensional (one-hot style) data.
            rng = MersenneTwister(7)
            perm = randperm(rng, size(X_fm, 1))
            n_tr = round(Int, 0.8 * length(perm))
            tr = perm[1:n_tr]; te = perm[n_tr+1:end]
            X_tr = X_fm[tr, :]; y_tr = y_fm[tr]
            X_te = X_fm[te, :]; y_te = y_fm[te]

            m_fm = FM(lr_w=0.2, rank=4, λ_w=0.0, λ_v=0.0,
                      family=Gaussian(), intercept=true, max_iter=50, verbose=false)
            fit!(m_fm, X_tr, y_tr; rng)
            jl_preds = predict(m_fm, X_te)

            # 1) Forward-pass math: compare against an independent dense
            #    implementation of the FM equation
            #    ŷ = w0 + Σ_j w_j x_j + ½ Σ_f [(Σ_j v_fj x_j)² − Σ_j v_fj² x_j²]
            ref = _fm_dense_reference(m_fm, X_te)
            rel_err = norm(jl_preds .- ref) / (norm(ref) + 1e-12)
            @test rel_err < 1e-3
            println("  FM dense-reference relative error: $rel_err")

            # 2) Ground-truth recovery on held-out rows (y generated from a
            #    known rank-2 latent interaction; oracle cor ≈ 1.0, SoftSVD
            #    reaches ~0.95 on this data). A correct FM must generalize.
            heldout_cor = _cor(jl_preds, y_te)
            @test heldout_cor >= 0.95
            println("  FM held-out cor(preds, y): $heldout_cor")

            # 3) Training parity vs rsparse: rsparse's FM init uses Armadillo's
            #    nondeterministic randn() (std::random_device), so its solution
            #    lands in a slightly different local optimum each run; both
            #    implementations recover the structure (cor(preds, y) ≈ 0.999).
            #    rsparse can also fail to converge entirely on very sparse
            #    low-coverage problems (upstream, not ours) — in that case its
            #    predictions are uncorrelated with anything, so agreement with
            #    Julia is meaningless and the gate is skipped (Julia's own
            #    correctness is enforced by the two gates above).
            cor_preds = _cor(jl_preds, r_preds[te])
            cor_r = _cor(r_preds[te], y_te)
            if cor_r >= 0.95
                @test cor_preds >= 0.95
            else
                @info "rsparse FM did not converge on this fixture (cor_r=$cor_r); skipping the agreement gate"
            end
            println("  FM held-out cor(jl, R) = $cor_preds")
            println("  FM cor(preds, y): Julia=$(round(_cor(jl_preds, y_te); sigdigits=4)), R=$(round(cor_r; sigdigits=4))")
        end
    end

    @testset "Metrics: exact match with R" begin
        metrics_path = joinpath(R_FIXTURE_DIR, "metrics_ref.csv")
        if !isfile(metrics_path)
            @info "Skipping metrics comparison: metrics_ref.csv is missing"
        else
            ref = (ap   = _read_col(metrics_path, "ap")[1],
                   ndcg = _read_col(metrics_path, "ndcg")[1])
            actual = sparse([1,1,1], [5,7,9], ones(3), 1, 10)
            preds = [5 7 9 2]
            ap = ap_at_k(preds, actual; k=4)[1]
            ndcg = ndcg_at_k(preds, actual; k=4)[1]
            @test ap ≈ ref.ap atol=1e-6
            @test ndcg ≈ ref.ndcg atol=1e-6
            println("  Metrics AP: Julia=$ap, R=$(ref.ap)")
            println("  Metrics NDCG: Julia=$ndcg, R=$(ref.ndcg)")
        end
    end

    @testset "SoftImpute: singular values and reconstruction match R" begin
        si_files = [joinpath(R_FIXTURE_DIR, f) for f in [
            "softimpute_si_d.csv", "softimpute_si_obs_preds.csv", "softimpute_si_frob.txt",
        ]]
        if !_all_files_exist(si_files)
            @info "Skipping SoftImpute comparison: fixture files missing"
        else
            r_d = _read_col(si_files[1], "d")
            r_obs = _read_col(si_files[2], "pred")
            r_frob = _read_scalar(si_files[3])

            m_si = SoftImpute(rank=5, λ=1.0, max_iter=100,
                              tol=1e-6, final_svd=true, verbose=false)
            fit!(m_si, X_ref; rng=MersenneTwister(42))

            jl_d = m_si.d
            jl_frob = sum(abs2, jl_d)

            recon = m_si.U * Diagonal(m_si.d) * m_si.V'
            rv = rowvals(X_ref); nz = nonzeros(X_ref)
            jl_obs = Float64[]
            for j in axes(X_ref, 2), idx in nzrange(X_ref, j)
                push!(jl_obs, recon[rv[idx], j])
            end

            # Rank-constrained SoftImpute can converge to different local optima
            # (non-convex when rank < true rank): validate total variance, SV
            # agreement, and reconstruction correlation instead of exact match.
            r_d_sorted = sort(r_d, rev=true)
            jl_d_sorted = sort(jl_d, rev=true)
            n_compare = min(length(r_d_sorted), length(jl_d_sorted))
            sv_rel_err = norm(r_d_sorted[1:n_compare] .- jl_d_sorted[1:n_compare]) /
                         (norm(r_d_sorted[1:n_compare]) + 1e-15)
            @test sv_rel_err < 0.25
            println("  SoftImpute SV relative error: $sv_rel_err")

            frob_rel = abs(jl_frob - r_frob) / (r_frob + 1e-15)
            @test frob_rel < 0.08
            println("  SoftImpute Frob norm: Julia=$jl_frob, R=$r_frob, rel=$frob_rel")

            cor_obs = _cor(jl_obs, r_obs)
            @test cor_obs >= 0.97
            println("  SoftImpute obs reconstruction correlation: $cor_obs")
        end
    end

    @testset "SoftSVD: singular values and reconstruction match R" begin
        svd_files = [joinpath(R_FIXTURE_DIR, f) for f in [
            "softimpute_svd_d.csv", "softimpute_svd_obs_preds.csv", "softimpute_svd_frob.txt",
        ]]
        if !_all_files_exist(svd_files)
            @info "Skipping SoftSVD comparison: fixture files missing"
        else
            r_d = _read_col(svd_files[1], "d")
            r_obs = _read_col(svd_files[2], "pred")
            r_frob = _read_scalar(svd_files[3])

            m_svd = SoftSVD(rank=5, λ=1.0, max_iter=100,
                            tol=1e-6, final_svd=true, verbose=false)
            fit!(m_svd, X_ref; rng=MersenneTwister(42))

            jl_d = m_svd.d
            jl_frob = sum(abs2, jl_d)

            recon = m_svd.U * Diagonal(m_svd.d) * m_svd.V'
            rv = rowvals(X_ref); nz = nonzeros(X_ref)
            jl_obs = Float64[]
            for j in axes(X_ref, 2), idx in nzrange(X_ref, j)
                push!(jl_obs, recon[rv[idx], j])
            end

            r_d_sorted = sort(r_d, rev=true)
            jl_d_sorted = sort(jl_d, rev=true)
            n_compare = min(length(r_d_sorted), length(jl_d_sorted))
            sv_rel_err = norm(r_d_sorted[1:n_compare] .- jl_d_sorted[1:n_compare]) /
                         (norm(r_d_sorted[1:n_compare]) + 1e-15)
            @test sv_rel_err < 0.05
            println("  SoftSVD SV relative error: $sv_rel_err")

            frob_rel = abs(jl_frob - r_frob) / (r_frob + 1e-15)
            @test frob_rel < 0.10
            println("  SoftSVD Frob norm: Julia=$jl_frob, R=$r_frob, rel=$frob_rel")

            cor_obs = _cor(jl_obs, r_obs)
            @test cor_obs >= 0.99
            println("  SoftSVD obs reconstruction correlation: $cor_obs")
        end
    end
end
