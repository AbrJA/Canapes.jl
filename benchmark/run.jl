# benchmark/run.jl — Tracked performance harness.
#
# Runs every algorithm at three fixed scales, records fit!/recommend time,
# allocations and heap growth, and appends one JSONL record per result to
# benchmark/logs/results.jsonl (git sha, environment, config and metrics).
#
# Usage:
#   julia --project=. --threads=8 benchmark/run.jl
#
# Compare runs:
#   julia --project=. benchmark/compare.jl --strict
#
# Benchmarks are deterministic: fixed seeds, fixed matrices. Run on the same
# machine and thread count to compare commits.

using Canapes
using SparseArrays
using Random
using LinearAlgebra
using Printf
using Dates
using JSON

const LOG_DIR = joinpath(@__DIR__, "logs")
const LOG_FILE = joinpath(LOG_DIR, "results.jsonl")
const K = 10

const SCALES = [
    (name="hundreds",  n_users=500,     n_items=300,      density=0.10),  # ~15K nnz
    (name="thousands", n_users=5_000,   n_items=3_000,    density=0.02),  # ~300K nnz
    (name="millions",  n_users=100_000, n_items=50_000,   density=0.001), # ~5M nnz
]

# ──────────────────────────────────────────────────────────────────────────────
# Deterministic data generation
# ──────────────────────────────────────────────────────────────────────────────

function generate_matrix(n_users::Int, n_items::Int, density::Float64; seed::Int=42)
    rng = MersenneTwister(seed)
    nnz_target = round(Int, n_users * n_items * density)
    rows = rand(rng, 1:n_users, nnz_target)
    cols = rand(rng, 1:n_items, nnz_target)
    sparse(rows, cols, ones(Float64, nnz_target), n_users, n_items)
end

function generate_cooccurrence(n::Int, density::Float64; seed::Int=7)
    rng = MersenneTwister(seed)
    A = sprand(rng, n, n, density)
    A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    A
end

# ──────────────────────────────────────────────────────────────────────────────
# Measurement
# ──────────────────────────────────────────────────────────────────────────────

function measure(f)
    live0 = Base.gc_live_bytes()
    t0 = time_ns()
    bytes = @allocated f()
    t1 = time_ns()
    (
        seconds = (t1 - t0) / 1e9,
        bytes = bytes,
        live_bytes = Base.gc_live_bytes(),
    )
end

function best_of(f, reps::Int)
    best = measure(f)
    for _ in 2:reps
        m = measure(f)
        m.seconds < best.seconds && (best = m)
    end
    best
end

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark definitions
# ──────────────────────────────────────────────────────────────────────────────

function benchmark_models(X; include_dense=true)
    models = [
        (name="WMF-Cholesky", model=WMF(rank=64, λ=0.1, α=40.0, max_iter=10,
            solver=CholeskySolver(), verbose=false)),
        (name="WMF-CG", model=WMF(rank=64, λ=0.1, α=40.0, max_iter=10,
            solver=CGSolver(), cg_steps=3, verbose=false)),
        (name="IALS", model=IALS(rank=64, max_iter=10, verbose=false)),
        (name="EALS", model=EALS(rank=64, max_iter=10, verbose=false)),
        (name="BPR", model=BPR(rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01,
            lr=0.05, max_iter=10, verbose=false)),
        (name="LogisticMF", model=LogisticMF(rank=64, λ=0.6, lr=1.0,
            max_iter=10, n_negative=30, tol=-1.0, verbose=false)),
    ]
    include_dense || return models
    vcat(models, [
        (name="EASE", model=EASE(λ=200.0, verbose=false)),
        (name="SLIM", model=SLIM(max_iter=10, verbose=false)),
        (name="ADMMSLIM", model=ADMMSLIM(max_iter=10, verbose=false)),
        (name="ItemKNN", model=ItemKNN(k=50, verbose=false)),
    ])
end

function compile_warmup()
    X = sparse([1, 2], [1, 2], [1.0, 1.0], 50, 40)
    for spec in benchmark_models(X)
        fit!(spec.model, X; rng=MersenneTwister(1))
        recommend(spec.model, X; k=K)
    end
    C = sprand(40, 40, 0.2)
    C = C + C'
    nonzeros(C) .= abs.(nonzeros(C)) .+ 0.1
    g = GloVe(rank=8, max_iter=2, verbose=false)
    fit!(g, C; rng=MersenneTwister(1))
    recommend(g, C; k=K)
end

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

function git_sha()
    try
        strip(String(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`)))
    catch
        "unknown"
    end
end

function config_string(model)
    parts = String[]
    for f in fieldnames(typeof(model))
        v = getfield(model, f)
        if !(v isa AbstractArray) && f != :is_fitted
            push!(parts, string(f, "=", v))
        end
    end
    join(parts, ", ")
end

function log_record(record)
    mkpath(LOG_DIR)
    open(LOG_FILE; append=true) do io
        println(io, JSON.json(record))
    end
    record
end

function print_record(record)
    @printf("  %-13s %-10s fit=%8.3fs  alloc=%9.1fMB  rec=%6.3fs  alloc=%6.1fMB\n",
        record["algorithm"], record["scale"],
        record["fit_seconds"], record["fit_bytes"] / 2^20,
        record["recommend_seconds"], record["recommend_bytes"] / 2^20)
end

function env_record()
    Dict(
        "git_sha" => git_sha(),
        "julia_version" => string(VERSION),
        "date" => string(now()),
        "nthreads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 78)
    println("Canapes.jl performance harness")
    println("=" ^ 78)
    println("  Julia:     $(VERSION)")
    println("  Threads:   $(Threads.nthreads())")
    println("  BLAS:      $(BLAS.get_num_threads()) threads")
    println("  Git:       $(git_sha())")
    println("  Log:       $LOG_FILE")
    println()

    println("Compile warmup...")
    compile_warmup()

    reps = Dict("hundreds" => 3, "thousands" => 2, "millions" => 1)

    for scale in SCALES
        println("-" ^ 78)
        println("Scale: $(scale.name)  ($(scale.n_users) users × $(scale.n_items) items, " *
                "density $(scale.density))")
        X = generate_matrix(scale.n_users, scale.n_items, scale.density)
        @printf("  X nnz: %d\n\n", nnz(X))

        # Dense item-similarity models are O(n_items²) in memory and time;
        # they are only benchmarked at the two smaller scales.
        include_dense = scale.name != "millions"
        for spec in benchmark_models(X; include_dense=include_dense)
            model = spec.model
            cfg = config_string(model)

            fit_m = best_of(() -> fit!(model, X; rng=MersenneTwister(1)), reps[scale.name])
            rec_m = best_of(() -> recommend(model, X; k=K), reps[scale.name])

            record = log_record(merge(env_record(), Dict(
                "algorithm" => spec.name,
                "config" => cfg,
                "scale" => scale.name,
                "n_users" => scale.n_users,
                "n_items" => scale.n_items,
                "nnz" => nnz(X),
                "fit_seconds" => fit_m.seconds,
                "fit_bytes" => fit_m.bytes,
                "fit_live_bytes" => fit_m.live_bytes,
                "recommend_seconds" => rec_m.seconds,
                "recommend_bytes" => rec_m.bytes,
                "recommend_live_bytes" => rec_m.live_bytes,
            )))
            print_record(record)
        end

        # GloVe runs on a square co-occurrence matrix sized to match the scale.
        n = scale.n_users
        d = scale.density * scale.n_items / n
        C = generate_cooccurrence(n, d)
        glove = GloVe(rank=64, lr=0.05, max_iter=10, verbose=false)
        fit_m = best_of(() -> fit!(glove, C; rng=MersenneTwister(1)), reps[scale.name])
        rec_m = best_of(() -> recommend(glove, C; k=K), reps[scale.name])
        record = log_record(merge(env_record(), Dict(
            "algorithm" => "GloVe",
            "config" => "rank=64,lr=0.05,max_iter=10",
            "scale" => scale.name,
            "n_users" => n,
            "n_items" => n,
            "nnz" => nnz(C),
            "fit_seconds" => fit_m.seconds,
            "fit_bytes" => fit_m.bytes,
            "fit_live_bytes" => fit_m.live_bytes,
            "recommend_seconds" => rec_m.seconds,
            "recommend_bytes" => rec_m.bytes,
            "recommend_live_bytes" => rec_m.live_bytes,
        )))
        print_record(record)
        println()
    end

    println("Done. Results appended to $LOG_FILE")
end

main()
