# ─────────────────────────────────────────────────────────────────────────────
# MovieLens benchmark — validate all Canapes.jl recommender algorithms and rank
# them by ranking metrics on a held-out split.
#
# Run:  julia --project=. --threads=8 usage/ml1m_compare.jl [ml-1m|ml-100k] [temporal|random]
# Output:  printed summary table + usage/ml1m_results_<dataset>_<split>.csv
# ─────────────────────────────────────────────────────────────────────────────
using Canapes
using SparseArrays
using Random
using Statistics
using Dates
using Printf

DATASET = get(ARGS, 1, "ml-1m")
SPLIT   = get(ARGS, 2, "temporal")
DATASET in ("ml-1m", "ml-100k") || error("dataset must be ml-1m or ml-100k")
SPLIT   in ("temporal", "random") || error("split must be temporal or random")

const N_USERS = DATASET == "ml-1m" ? 6040 : 943
const N_ITEMS = DATASET == "ml-1m" ? 3952 : 1682
const DATA = joinpath(@__DIR__, DATASET, DATASET == "ml-1m" ? "ratings.dat" : "u.data")
const OUT  = joinpath(@__DIR__, "ml1m_results_$(DATASET)_$(SPLIT).csv")
const K    = 10             # evaluation cutoff
const NT   = 20             # popularity baseline cutoff

# ── 1. Load ratings ──────────────────────────────────────────────────────────
println("Started: ", now(), " (t=0s)")
const _T0 = time()
println("Loading $(DATA) ...")
users  = Int[]
items  = Int[]
t_last = Float64[]
open(DATA) do io
    for line in eachline(io)
        f = split(line, r"::|\t")
        push!(users,  parse(Int, f[1]))
        push!(items,  parse(Int, f[2]))
        push!(t_last, parse(Float64, f[4]))
    end
end
println("  $(length(users)) ratings, $(maximum(users)) users, $(maximum(items)) items")

X = triplets_to_sparse((user=users, item=items, value=ones(length(users)));
                       n_users=N_USERS, n_items=N_ITEMS)
println("  matrix: $(size(X, 1)) users × $(size(X, 2)) items, $(nnz(X)) interactions",
        " ($(round(nnz(X) / (size(X,1) * size(X,2)) * 100, sigdigits=3))% density)")

# ── 2. Train/test split ──────────────────────────────────────────────────────
if SPLIT == "random"
    println("Splitting train/test (80/20, random per-user holdout) ...")
    X_train, X_test = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(42))
else
    println("Splitting train/test (80/20, per-user chronological) ...")
    order = sortperm(collect(zip(users, t_last)))
    n_user = zeros(Int, N_USERS)
    for u in users
        n_user[u] += 1
    end
    train_rows = Int[]; train_cols = Int[]; train_vals = Float64[]
    test_rows  = Int[]; test_cols  = Int[]; test_vals  = Float64[]
    n_seen_user = zeros(Int, N_USERS)
    for i in order
        u, j = users[i], items[i]
        n_seen_user[u] += 1
        # each user's ratings are visited in chronological order; keep the first 80%
        if n_seen_user[u] <= round(Int, 0.8 * n_user[u])
            push!(train_rows, u); push!(train_cols, j); push!(train_vals, 1.0)
        else
            push!(test_rows,  u); push!(test_cols,  j); push!(test_vals,  1.0)
        end
    end
    X_train = sparse(train_rows, train_cols, train_vals, size(X, 1), size(X, 2))
    X_test  = sparse(test_rows,  test_cols,  test_vals,  size(X, 1), size(X, 2))
end
println("  train: $(nnz(X_train)) interactions, test: $(nnz(X_test)) interactions",
        " ($(round(nnz(X_test) / nnz(X) * 100, sigdigits=3))% held out)")
test_nnz = zeros(Int, size(X_test, 1))
for j in axes(X_test, 2)
    rv = rowvals(X_test); nz = nonzeros(X_test)
    for idx in nzrange(X_test, j)
        test_nnz[rv[idx]] += 1
    end
end
users_with_test = count(!iszero, test_nnz)
println("  $users_with_test of $(size(X_test, 1)) users have ≥1 test item")

# ── 3. Evaluate a fitted model ───────────────────────────────────────────────
function evaluate(model, name; rng=MersenneTwister(1))
    t0 = time()
    try
        fit!(model, X_train; rng=rng)
    catch e
        @warn "fit! failed for $name: $e"
        return nothing
    end
    fit_s = time() - t0

    preds = try
        recommend(model, X_train; k=K)
    catch e
        @warn "recommend failed for $name: $e"
        return nothing
    end
    map_  = mean_ap_at_k(preds, X_test; k=K)
    ndcg  = mean(ndcg_at_k(preds, X_test; k=K))
    prec  = mean(precision_at_k(preds, X_test; k=K))
    rec   = mean(recall_at_k(preds, X_test; k=K))
    (name=name, fit_s=fit_s, map=map_, ndcg=ndcg, prec=prec, rec=rec)
end

# ── 4. Model catalog ─────────────────────────────────────────────────────────
# Hyperparameters use the tuned settings found in usage/ml1m_tune.jl (ml-100k
# random split); stock defaults flag as: ItemKNN k, EALS unobserved_weight,
# LogisticMF* lr (experimental), ADMMSLIM λ_l2/max_iter, BPR max_iter.
results = []

push!(results, evaluate(WMF(rank=32, α=40.0, λ=0.1, max_iter=10, verbose=false), "WMF"))
push!(results, evaluate(IALS(rank=64, α=40.0, max_iter=10, verbose=false), "IALS"))
push!(results, evaluate(EALS(rank=64, λ=0.01, unobserved_weight=64.0, max_iter=30,
                             verbose=false), "EALS"))
push!(results, evaluate(BPR(rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01,
                            max_iter=30, verbose=false), "BPR"))
push!(results, evaluate(Canapes.Experimental.LogisticMF(rank=32, λ=0.6, lr=1.0, α=1.0, max_iter=50,
                                   n_negative=5, verbose=false), "LogisticMF*"))

push!(results, evaluate(EASE(λ=200.0, verbose=false), "EASE"))
push!(results, evaluate(SLIM(λ_l1=0.01, λ_l2=0.1, max_iter=10, verbose=false), "SLIM"))
push!(results, evaluate(ADMMSLIM(λ_l1=0.01, λ_l2=500.0, max_iter=50, verbose=false), "ADMMSLIM"))
push!(results, evaluate(ItemKNN(k=400, similarity=:cosine, verbose=false), "ItemKNN"))
push!(results, evaluate(RandomWalk(β=0.0, k=nothing, verbose=false), "RandomWalk β=0"))
push!(results, evaluate(RandomWalk(β=0.6, k=nothing, verbose=false), "RandomWalk β=0.6"))
push!(results, evaluate(SoftImpute(rank=20, λ=0.5, max_iter=50, verbose=false), "SoftImpute"))
push!(results, evaluate(SoftSVD(rank=20, max_iter=50, verbose=false), "SoftSVD"))
push!(results, evaluate(PureSVD(rank=20, max_iter=50, verbose=false), "PureSVD"))

# ── 5. Popularity baseline ───────────────────────────────────────────────────
println("\nPopularity baseline ...")
t0 = time()
X_csr = Canapes.to_csr(X_train)                # row-wise access
order = sortperm(vec(sum(X_train, dims=1)), rev=true)   # global popularity once
preds_pop = Matrix{Int}(undef, size(X_train, 1), NT)
seen = BitVector(undef, size(X_train, 2))
for u in 1:size(X_train, 1)
    seen .= false
    for idx in nzrange(X_csr, u)
        seen[X_csr.colval[idx]] = true
    end
    kk = 0
    for j in order
        if !seen[j]
            preds_pop[u, kk += 1] = j
            kk >= NT && break
        end
    end
end
push!(results, (name="Popularity", fit_s=time()-t0,
                map=mean_ap_at_k(preds_pop, X_test; k=K), ndcg=mean(ndcg_at_k(preds_pop, X_test; k=K)),
                prec=mean(precision_at_k(preds_pop, X_test; k=K)),
                rec=mean(recall_at_k(preds_pop, X_test; k=K))))

# ── 6. Report ────────────────────────────────────────────────────────────────
println("\n" * "─"^98)
hdr = "Model          │   Fit (s) │    MAP@10 │   NDCG@10 │    P@10 │    R@10"
println(hdr); println("─"^98)
for r in sort(results, by=r -> r.map, rev=true)
    @printf "%-14s │ %8.1f │ %9.4f │ %9.4f │ %7.4f │ %7.4f\n" r.name r.fit_s r.map r.ndcg r.prec r.rec
end
println("─"^98)

open(OUT, "w") do io
    println(io, "model,fit_s,map_at_10,ndcg_at_10,precision_at_10,recall_at_10")
    for r in sort(results, by=r -> r.map, rev=true)
        println(io, join((r.name, round(r.fit_s, digits=1), round(r.map, digits=4),
                          round(r.ndcg, digits=4), round(r.prec, digits=4),
                          round(r.rec, digits=4)), ","))
    end
end
println("Finished: ", now(), " (elapsed ", round(time() - _T0, digits=1), "s)")
println("Saved results to $(OUT)")