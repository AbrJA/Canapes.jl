# ─────────────────────────────────────────────────────────────────────────────
# MovieLens 1M — explicit (rating prediction) benchmark for the explicit
# subsystem: BiasedMF (WMF-Explicit), BaselineOnly, SlopeOne, PearsonKNN,
# SoftImpute/SoftSVD/PureSVD, plus the plain global-mean baseline.
#
# Protocol: per-user temporal split (first 80% of each user's ratings since
# chronological is the honest deployment setting), RMSE/MAE over the held-out
# ratings. Ratings are the raw 1-5 scores (not binarized).
#
# Run:  julia --project=. --threads=8 usage/ml1m_explicit.jl
# Output: printed table + usage/ml1m_explicit_results.csv
# ─────────────────────────────────────────────────────────────────────────────
using Canapes
using SparseArrays
using Random
using Statistics
using Dates
using Printf

const DATA = joinpath(@__DIR__, "ml-1m", "ratings.dat")
const OUT  = joinpath(@__DIR__, "ml1m_explicit_results.csv")

# ── load ─────────────────────────────────────────────────────────────────────
println("Started: ", now(), " (t=0s)")
const _T0 = time()
println("Loading $(DATA) ...")
users, items, t_last, ratings = Int[], Int[], Float64[], Float64[]
open(DATA) do io
    for line in eachline(io)
        f = split(line, "::")
        push!(users, parse(Int, f[1]))
        push!(items, parse(Int, f[2]))
        push!(ratings, parse(Float64, f[3]))
        push!(t_last, parse(Float64, f[4]))
    end
end
println("  $(length(users)) ratings, max user $(maximum(users)), max item $(maximum(items))")

# ── per-user temporal split (80/20) ──────────────────────────────────────────
order = sortperm(collect(zip(users, t_last)))
n_user = zeros(Int, 6040)
for u in users; n_user[u] += 1; end
tr_u, tr_i, tr_v, te_u, te_i, te_v = Int[], Int[], Float64[], Int[], Int[], Float64[]
seen = zeros(Int, 6040)
for i in order
    u, j, r = users[i], items[i], ratings[i]
    seen[u] += 1
    if seen[u] <= round(Int, 0.8 * n_user[u])
        push!(tr_u, u); push!(tr_i, j); push!(tr_v, r)
    else
        push!(te_u, u); push!(te_i, j); push!(te_v, r)
    end
end
X_train = triplets_to_sparse((user=tr_u, item=tr_i, value=tr_v); n_users=6040, n_items=3952)
X_test  = sparse(te_u, te_i, te_v, 6040, 3952)
println("  train: $(nnz(X_train)) ratings, test: $(nnz(X_test)) ratings")

# ── evaluate ──────────────────────────────────────────────────────────────────
# mk returns an UNFITTED model; eval_model fits it against X_train and
# scores with predict (explicit contract) + rmse/mae over the held-out set.
function eval_model(name, mk)
    m = mk()
    t0 = time()
    fit!(m, X_train)
    fit_s = time() - t0
    t0 = time()
    P = predict(m, X_train)
    pred_s = time() - t0
    r  = rmse(P, X_test)
    ma = mae(P, X_test)
    (name=name, fit_s=fit_s, pred_s=pred_s, rmse=r, mae=ma)
end

results = []
push!(results, eval_model("GlobalMean", () -> BaselineOnly(max_iter=1, verbose=false)))
push!(results, eval_model("BaselineOnly", () -> BaselineOnly(λ=0.1, max_iter=20, verbose=false)))
push!(results, eval_model("BiasedMF (WMF-Explicit)", () -> WMF(rank=16, λ=0.1, max_iter=15, solver=CholeskySolver(),
                                                               feedback=Explicit, verbose=false)))
push!(results, eval_model("BiasedMF-CG", () -> WMF(rank=16, λ=0.1, max_iter=15, solver=CGSolver(),
                                                   feedback=Explicit, verbose=false)))
push!(results, eval_model("SlopeOne", () -> SlopeOne(verbose=false)))
push!(results, eval_model("PearsonKNN k=40", () -> PearsonKNN(k=40, verbose=false)))
push!(results, eval_model("PearsonKNN k=500", () -> PearsonKNN(k=500, verbose=false)))
push!(results, eval_model("SoftImpute", () -> SoftImpute(rank=20, λ=0.5, max_iter=50, verbose=false)))
push!(results, eval_model("SoftSVD", () -> SoftSVD(rank=20, max_iter=50, verbose=false)))
push!(results, eval_model("PureSVD", () -> PureSVD(rank=20, max_iter=50, verbose=false)))

# ── report ───────────────────────────────────────────────────────────────────
println("\n" * "─"^92)
hdr = "Model               │  Fit (s) │ Pred (s) │    RMSE │     MAE"
println(hdr); println("─"^92)
for r in sort(results, by=r -> r.rmse)
    @printf "%-20s │ %8.1f │ %8.1f │ %7.4f │ %7.4f\n" r.name r.fit_s r.pred_s r.rmse r.mae
end
println("─"^92)

open(OUT, "w") do io
    println(io, "model,fit_s,pred_s,rmse,mae")
    for r in sort(results, by=r -> r.rmse)
        println(io, join((r.name, round(r.fit_s, digits=1), round(r.pred_s, digits=1),
                          round(r.rmse, digits=4), round(r.mae, digits=4)), ","))
    end
end
println("Finished: ", now(), " (elapsed ", round(time() - _T0, digits=1), "s)")
println("Saved results to $(OUT)")