# validation/validate_surprise.jl — Surprise parity for the explicit subsystem.
#
# Generates a deterministic ratings fixture, fits the Surprise reference models
# (via surprise_ref.py) and the Canapes explicit models, and compares held-out
# predictions. Gating is correlation + RMSE-distance based, overridable with
# CANAPES_SURPRISE_MIN_COR / CANAPES_SURPRISE_MAX_RMSE_DELTA.
# Run: julia --project=. validation/validate_surprise.jl
using Canapes
using SparseArrays
using Random
using Statistics
using Printf

const FDIR = get(ENV, "CANAPES_SURPRISE_FIXTURE_DIR", joinpath(tempdir(), "canapes_surprise"))
const PY = get(ENV, "CANAPES_SURPRISE_PYTHON", "python3")

# ── fixture ───────────────────────────────────────────────────────────────────
function generate_fixture(fdir::String)
    mkpath(fdir)
    rng = MersenneTwister(2024)
    n_u, n_i = 90, 70
    rows, cols = Int[], Int[]
    for u in 1:n_u, i in 1:n_i
        rand(rng) < 0.25 && (push!(rows, u); push!(cols, i))
    end
    # biased generative model: μ + b_u + b_i + latent + noise, ratings 1..5
    μ0 = 3.5
    ub = [0.6 * sin(u) for u in 1:n_u]
    ib = [0.5 * cos(i) for i in 1:n_i]
    U0 = randn(rng, 4, n_u) .* 0.3
    V0 = randn(rng, 4, n_i) .* 0.3
    vals = Float64[]
    for (r, c) in zip(rows, cols)
        v = clamp(μ0 + ub[r] + ib[c] + sum(U0[:, r] .* V0[:, c]) + randn(rng) * 0.25, 1, 5)
        push!(vals, Float64(round(Int, v)))   # integer ratings: Surprise's
        # SlopeOne only supports them (its fit truncates floats to ints)
    end
    X = sparse(rows, cols, vals, n_u, n_i)
    X_tr, X_te = random_holdout(X; test_fraction=0.2, rng=MersenneTwister(1))

    # Surprise file format: first line = number of entries
    open(joinpath(fdir, "train.csv"), "w") do io
        println(io, nnz(X_tr))
        for j in axes(X_tr, 2), idx in nzrange(X_tr, j)
            println(io, rowvals(X_tr)[idx], "\t", j, "\t", nonzeros(X_tr)[idx])
        end
    end
    open(joinpath(fdir, "test.csv"), "w") do io
        println(io, "user item rating")
        for j in axes(X_te, 2), idx in nzrange(X_te, j)
            println(io, rowvals(X_te)[idx], "\t", j, "\t", nonzeros(X_te)[idx])
        end
    end
    (X_tr, X_te)
end

function reference_predictions(fdir::String, py::String)
    # run the reference script; throws on failure
    run(`$py $(joinpath(@__DIR__, "surprise_ref.py")) $fdir`)
end

function load_test_values(fdir::String)
    rows = Int[]; cols = Int[]; vals = Float64[]
    open(joinpath(fdir, "test.csv")) do io
        readline(io)
        for line in eachline(io)
            f = split(line)
            push!(rows, parse(Int, f[1])); push!(cols, parse(Int, f[2]))
            push!(vals, parse(Float64, f[3]))
        end
    end
    (rows, cols, vals)
end

function load_preds(fdir::String, name::String)
    rows = Int[]; cols = Int[]; vals = Float64[]
    open(joinpath(fdir, "surprise_$(name).csv")) do io
        readline(io)  # header
        for line in eachline(io)
            f = split(line, ",")
            push!(rows, parse(Int, f[1])); push!(cols, parse(Int, f[2]))
            push!(vals, parse(Float64, f[3]))
        end
    end
    (rows, cols, vals)
end

# predictions of a Canapes model at the (u, i) test positions
function predict_at(model, P::Matrix{T}, rows, cols) where {T}
    [P[r, c] for (r, c) in zip(rows, cols)]
end

function main()
    X_tr, X_te = generate_fixture(FDIR)
    println("fixture: $(nnz(X_tr)) train / $(nnz(X_te)) test ratings → $FDIR")

    min_cor = parse(Float64, get(ENV, "CANAPES_SURPRISE_MIN_COR", "0.95"))
    max_delta = parse(Float64, get(ENV, "CANAPES_SURPRISE_MAX_RMSE_DELTA", "0.30"))

    # reference
    reference_predictions(FDIR, PY)
    te_rows, te_cols, te_vals = load_test_values(FDIR)  # actual test ratings
    ref = Dict(name => Float64[] for name in ("slopeone", "baseline", "pearson", "svd"))
    for name in keys(ref)
        ref[name] = last(load_preds(FDIR, name))
    end

    rmse_of(a, b) = sqrt(sum(abs2, a .- b) / length(a))

    # Canapes models
    models = [
        ("slopeone", SlopeOne(verbose=false, T=Float64)),
        ("baseline", BaselineOnly(λ=0.02, max_iter=15, verbose=false)),
        ("pearson", PearsonKNN(k=20, min_k=1, verbose=false)),
    ]
    models_jl = Dict{String,Vector{Float64}}()
    for (name, m) in models
        fit!(m, X_tr)
        P = predict(m, X_tr)
        models_jl[name] = [Float64(P[r, c]) for (r, c) in zip(te_rows, te_cols)]
    end
    # BiasedMF (ALS) vs Surprise SVD (SGD) — diagnostic correlation only
    m_biased = WeightedMF(rank=12, λ=0.05, max_iter=15, solver=CholeskySolver(),
                   feedback=Explicit, verbose=false)
    fit!(m_biased, X_tr)
    P_b = predict(m_biased, X_tr)
    biased_jl = [Float64(P_b[r, c]) for (r, c) in zip(te_rows, te_cols)]

    checks = Bool[]
    function check(name, a, b, expect_exact::Bool)
        cor = cor_pearson(a, b)
        delta = rmse_of(a, b)
        truermse = rmse_of(b, te_vals)
        ok = expect_exact ?
             maximum(abs.(a .- b)) < 1e-4 :
             cor >= min_cor && delta <= max_delta
        push!(checks, ok)
        println((ok ? "✓ " : "✗ ") * rpad(name, 12) * " cor=$(round(cor; digits=4)) " *
                "Δrmse=$(round(delta; digits=4)) ref-rmse=$(round(truermse; digits=4))")
    end

    println("\nSurprise reference comparison")
    check("slopeone", models_jl["slopeone"], ref["slopeone"], true)
    check("baseline", models_jl["baseline"], ref["baseline"], false)
    check("pearson",  models_jl["pearson"],  ref["pearson"],  false)
    println("(svd↔BiasedMF is diagnostic: ALS vs SGD trajectories)")
    check("svd", biased_jl, ref["svd"], false)

    ok_all = all(checks)
    println(ok_all ? "\nSURPRISE: PASSED" : "\nSURPRISE: FAILED")
    exit(ok_all ? 0 : 1)
end

function cor_pearson(a::Vector{Float64}, b::Vector{Float64})
    ma = mean(a); mb = mean(b)
    num = sum((a .- ma) .* (b .- mb))
    den = sqrt(sum((a .- ma) .^ 2)) * sqrt(sum((b .- mb) .^ 2))
    den > 0 ? num / den : NaN
end

main()