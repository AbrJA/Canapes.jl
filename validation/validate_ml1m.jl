# validation/validate_ml1m.jl — MovieLens-1M scale comparison vs Python (implicit).
#
# Mirrors usage/ml1m_compare.jl's random 80/20 protocol exactly (binarized
# ratings, MersenneTwister(42)), fits the same model families on both sides,
# and evaluates BOTH sets of recommendations with Canapes' own metrics so the
# comparison is apples-to-apples (no metric-definition mismatch). Timings are
# reported for both sides.
#
# Run: julia --project=. validation/validate_ml1m.jl
# Env: CANAPES_ML1M_PYTHON (default python3), CANAPES_ML1M_MAX_MAP_DELTA,
#      CANAPES_ML1M_MAX_NDCG_DELTA, CANAPES_ML1M_FIXTURE_DIR.
using Canapes
using SparseArrays
using Random
using Statistics
using Printf

const DATA = joinpath(dirname(@__DIR__), "usage", "ml-1m", "ratings.dat")
const FDIR = get(ENV, "CANAPES_ML1M_FIXTURE_DIR", joinpath(tempdir(), "canapes_ml1m"))
const PY = get(ENV, "CANAPES_ML1M_PYTHON", "python3")
const MAX_MAP = parse(Float64, get(ENV, "CANAPES_ML1M_MAX_MAP_DELTA", "0.03"))
const MAX_NDCG = parse(Float64, get(ENV, "CANAPES_ML1M_MAX_NDCG_DELTA", "0.04"))

isfile(DATA) || error("MovieLens-1M ratings not found at $DATA")

# ── split (identical to usage/ml1m_compare.jl) ───────────────────────────────
function load_split()
    users, items, t_last = Int[], Int[], Float64[]
    open(DATA) do io
        for line in eachline(io)
            f = split(line, "::")
            push!(users, parse(Int, f[1])); push!(items, parse(Int, f[2]))
            push!(t_last, parse(Float64, f[4]))
        end
    end
    X = triplets_to_sparse((user=users, item=items, value=ones(length(users)));
                           n_users=6040, n_items=3952)
    random_holdout(X; test_fraction=0.2, rng=MersenneTwister(42))
end

function write_csvs(X_tr, X_te)
    mkpath(FDIR)
    open(joinpath(FDIR, "train.txt"), "w") do io
        for j in axes(X_tr, 2), idx in nzrange(X_tr, j)
            println(io, rowvals(X_tr)[idx], " ", j)
        end
    end
    open(joinpath(FDIR, "test.txt"), "w") do io
        for j in axes(X_te, 2), idx in nzrange(X_te, j)
            println(io, rowvals(X_te)[idx], " ", j)
        end
    end
end

function load_recs(path, n_users, K=10)
    M = Matrix{Int}(undef, n_users, K)
    open(path) do io
        u = 1
        for line in eachline(io)
            M[u, :] = parse.(Int, split(line))
            u += 1
        end
    end
    M
end

function main()
    println("loading + splitting ml-1m ...")
    X_tr, X_te = load_split()
    write_csvs(X_tr, X_te)
    println("  train $(nnz(X_tr)) / test $(nnz(X_te)) → $FDIR")

    # ── Canapes models (matching hyperparameters from usage/ml1m_compare.jl) ──
    function run_jl(name, m)
        fit_alloc = @allocated fit!(m, X_tr; rng=MersenneTwister(1))
        t0 = time(); fit!(m, X_tr; rng=MersenneTwister(1)); fit_s = time() - t0
        preds = recommend(m, X_tr; k=10)
        rec_alloc = @allocated recommend(m, X_tr; k=10)
        map_ = mean_ap_at_k(preds, X_te; k=10)
        ndcg = mean(ndcg_at_k(preds, X_te; k=10))
        (name, fit_s, fit_alloc, rec_alloc, map_, ndcg)
    end
    jl = []
    push!(jl, run_jl("als", WMF(rank=32, α=40.0, λ=0.1, max_iter=10,
                                 solver=CholeskySolver(), verbose=false)))
    push!(jl, run_jl("bpr", BPR(rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01,
                                 lr=0.1, max_iter=100, verbose=false)))
    push!(jl, run_jl("itemknn", ItemKNN(k=400, similarity=:cosine, verbose=false)))

    # ── Python reference ──
    println("running python (implicit) reference ...")
    run(`$PY $(joinpath(@__DIR__, "ml1m_ref.py")) $FDIR 6040 3952`)
    timings = Dict{String,Float64}()
    if isfile(joinpath(FDIR, "py_timings.txt"))
        for line in eachline(joinpath(FDIR, "py_timings.txt"))
            f = split(line)
            timings[f[1]] = parse(Float64, f[2])
        end
    end

    n_users = size(X_tr, 1)
    println("\n" * "─"^88)
    println("Model      │ Canapes MAP  NDCG  fit(s)  fit(MB) rec(MB) │ Python  MAP  NDCG  fit(s) │  ΔMAP   ΔNDCG")
    println("─"^100)
    all_ok = true
    for (name, fit_s, fit_alloc, rec_alloc, jmap, jndcg) in jl
        precs = load_recs(joinpath(FDIR, "py_$(name).txt"), n_users)
        pmap = mean_ap_at_k(precs, X_te; k=10)
        pndcg = mean(ndcg_at_k(precs, X_te; k=10))
        pfit = get(timings, name, NaN)
        dmap = abs(jmap - pmap); dndcg = abs(jndcg - pndcg)
        # BPR is SGD-based: both implementations are valid but trajectories
        # diverge, so it is reported as diagnostic rather than gated.
        diagnostic = name == "bpr"
        ok = diagnostic || (dmap <= MAX_MAP && dndcg <= MAX_NDCG)
        all_ok &= ok
        @printf "%-10s |  %.4f  %.4f  %5.1f  %7.1f  %7.1f |  %.4f  %.4f  %5.1f |  %.4f  %.4f  %s\n" name jmap jndcg fit_s (fit_alloc / 2^20) (rec_alloc / 2^20) pmap pndcg pfit dmap dndcg (diagnostic ? "·" : (ok ? "✓" : "✗"))
    end
    println("─"^100)

    println(all_ok ? "\nML1M: PASSED" : "\nML1M: FAILED")
    exit(all_ok ? 0 : 1)
end

main()